#!/usr/bin/env python3
"""Self-supervised meta-learning for subject-adaptive mouse brain extraction.

Stage 1 pretrains a 2.5D CNN-Transformer encoder with masked reconstruction on
training mice, stage 2 meta-trains it across training mice with Reptile, and
stage 3 adapts it to each unseen mouse (self-supervised test-time adaptation,
then fine-tuning on k annotated slices).

Evaluation uses animal-level 5-fold cross-validation stratified by acquisition
cohort (1c-10c are 256 px scans, 21c-30c are 200 px scans). Validation and test
animals are never used for pretraining or meta-training. For every k, Dice is
measured on the same query slices: all slices except the k=3 support slices.
"""

import argparse
import copy
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "dataset_split_70_15_15"
SLICE_PATTERN = re.compile(r"(?P<animal>.+)_slice_(?P<slice>\d+)$")
COHORTS = {"A": [f"{i}c" for i in range(1, 11)], "B": [f"{i}c" for i in range(21, 31)]}
METHODS = ["scratch_joint", "ssl_joint", "scratch_reptile", "ssl_reptile"]
SHOTS = (0, 1, 3)


@dataclass
class RunConfig:
    data_root: str
    output_dir: str
    fold: int = 0
    seed: int = 1
    regime: str = "full"  # "full": every slice of a training mouse is labelled; "sparse": only its 3 support slices
    image_size: int = 256
    batch_size: int = 8
    patch_size: int = 32
    mask_ratio: float = 0.5
    ssl_steps: int = 4000
    ssl_lr: float = 1e-3
    joint_steps: int = 10000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    meta_steps: int = 1000
    meta_batch: int = 4
    meta_lr: float = 1.0
    inner_steps: int = 15
    inner_lr: float = 1e-4
    tta_steps: int = 20
    tta_lr: float = 1e-3
    eval_every: int = 500


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cohort(animal):
    return "A" if animal in COHORTS["A"] else "B"


def make_split(fold, n_folds=5):
    rng = random.Random(2026)
    a, b = COHORTS["A"][:], COHORTS["B"][:]
    rng.shuffle(a); rng.shuffle(b)
    folds = [a[2 * i:2 * i + 2] + b[2 * i:2 * i + 2] for i in range(n_folds)]
    test = folds[fold]
    following = folds[(fold + 1) % n_folds]
    val = [following[0], following[2]]
    train = sorted((x for x in COHORTS["A"] + COHORTS["B"] if x not in test + val), key=lambda x: int(x[:-1]))
    return train, val, test


def support_slices(n_slices, k):
    if k == 0:
        return []
    if k == 1:
        return [n_slices // 2]
    return [n_slices // 4, n_slices // 2, (3 * n_slices) // 4]


def load_volumes(root, size, device):
    raw = {}
    for split in ("train", "val", "test"):
        for path in sorted((root / split / "images").glob("*.tif")):
            match = SLICE_PATTERN.match(path.stem)
            image = np.asarray(Image.open(path), dtype=np.float32)
            # Read raw values: masks 1c-10c are palette images whose palette maps index 0
            # to white, so converting them to greyscale inverts brain and background.
            mask = np.asarray(Image.open(root / split / "masks" / path.name)) > 127
            raw.setdefault(match["animal"], []).append((int(match["slice"]), image, mask))
    volumes = {}
    for animal, slices in raw.items():
        slices.sort(key=lambda s: s[0])
        images = torch.from_numpy(np.stack([s[1] for s in slices]))
        masks = torch.from_numpy(np.stack([s[2] for s in slices]).astype(np.float32))
        # Raw intensities are float and differ per scan, so normalise each volume
        # instead of casting to 8-bit (which kept only ~16-45 grey levels).
        low, high = np.percentile(images.numpy(), [0.5, 99.5])
        images = ((images - low) / (high - low)).clamp(0, 1)
        images = F.interpolate(images[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
        masks = F.interpolate(masks[:, None], size=(size, size), mode="nearest")[:, 0]
        volumes[animal] = (images.to(device), masks.to(device))
    return volumes


def context_stack(images, index):
    """Previous, center and next slice for each requested center slice."""
    last = images.shape[0] - 1
    index = torch.as_tensor(list(index), device=images.device)
    return torch.stack([images[(index - 1).clamp(min=0)], images[index], images[(index + 1).clamp(max=last)]], 1)


def batch_pairs(volumes, items):
    x = torch.cat([context_stack(volumes[a][0], [i]) for a, i in items])
    y = torch.stack([volumes[a][1][i] for a, i in items])[:, None]
    return x, y


def augment(x, y=None):
    n, device = x.shape[0], x.device
    angle = (torch.rand(n, device=device) - 0.5) * math.radians(30)
    scale = 1 + (torch.rand(n, device=device) - 0.5) * 0.2
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    cos, sin, zero = torch.cos(angle) / scale, torch.sin(angle) / scale, torch.zeros(n, device=device)
    theta = torch.stack([torch.stack([cos * flip, -sin, zero], 1), torch.stack([sin * flip, cos, zero], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", align_corners=False).clamp(0, 1)
    x = x ** torch.exp((torch.rand(n, 1, 1, 1, device=device) - 0.5) * 0.6)
    if y is None:
        return x
    return x, F.grid_sample(y, grid, mode="nearest", align_corners=False)


def gn(channels):
    # GroupNorm instead of BatchNorm: adaptation batches hold only 1-3 distinct slices.
    return nn.GroupNorm(8, channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), gn(channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1), gn(channels))

    def forward(self, x):
        return F.gelu(x + self.body(x))


class Encoder(nn.Module):
    """mixer: "transformer" (global self-attention over the 16x16 grid) or "cnn" (residual
    convolutions, local context only). context: "2.5d" uses the neighbouring slices,
    "2d" replaces them with the centre slice."""

    def __init__(self, width=64, depth=4, heads=8, mixer="transformer", context="2.5d"):
        super().__init__()
        self.mixer, self.context = mixer, context
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1), gn(width), nn.GELU(), nn.Conv2d(width, width, 3, stride=2, padding=1), gn(width), nn.GELU())
        self.down = nn.Sequential(nn.Conv2d(width, width * 2, 3, stride=2, padding=1), gn(width * 2), nn.GELU(), nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), gn(width * 4), nn.GELU())
        if mixer == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=width * 4, nhead=heads, dim_feedforward=width * 8, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        else:
            # Two residual blocks: about the same parameter count as four Transformer layers.
            self.convs = nn.Sequential(*[ResidualBlock(width * 4) for _ in range(depth // 2)])
        self.norm = nn.LayerNorm(width * 4)

    def forward(self, x):
        if self.context == "2d":
            x = x[:, 1:2].expand(-1, 3, -1, -1)
        shallow = self.stem(x)
        encoded = self.down(shallow)
        batch, channels, height, width = encoded.shape
        if self.mixer == "transformer":
            tokens = self.transformer(encoded.flatten(2).transpose(1, 2))
        else:
            tokens = self.convs(encoded).flatten(2).transpose(1, 2)
        return shallow, self.norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)


class SegmentationModel(nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = Encoder(**encoder_kwargs)
        self.decoder = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GELU(), nn.ConvTranspose2d(128, 64, 2, stride=2), nn.GELU(), nn.ConvTranspose2d(64, 32, 2, stride=2), nn.GELU())
        self.skip = nn.Conv2d(64, 32, 1)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        shallow, encoded = self.encoder(x)
        logits = self.head(self.decoder(encoded) + self.skip(shallow))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ReconstructionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.GELU(), nn.ConvTranspose2d(128, 64, 2, stride=2), nn.GELU(), nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, encoded, size):
        return F.interpolate(self.layers(encoded), size=size, mode="bilinear", align_corners=False)


def patch_mask(x, patch, ratio):
    n, _, h, w = x.shape
    gh, gw = h // patch, w // patch
    rank = torch.rand(n, gh * gw, device=x.device).argsort(1).argsort(1)
    keep = (rank >= int(gh * gw * ratio)).view(n, 1, gh, gw).float()
    return x * keep.repeat_interleave(patch, 2).repeat_interleave(patch, 3)


def reconstruction_loss(encoder, decoder, x, cfg):
    _, encoded = encoder(patch_mask(x, cfg.patch_size, cfg.mask_ratio))
    return F.l1_loss(decoder(encoded, x.shape[-2:]), x[:, 1:2])


def dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum((1, 2, 3))
    return 1 - ((2 * intersection + 1e-6) / (probs.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)).mean()


def segmentation_loss(logits, target):
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * dice_loss(logits, target)


def cosine(optimizer, steps):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))


def pretrain(volumes, animals, cfg, device, log):
    encoder, decoder = Encoder().to(device), ReconstructionDecoder().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.ssl_lr, weight_decay=0.05)
    scheduler = cosine(optimizer, cfg.ssl_steps)
    pool = [(a, i) for a in animals for i in range(volumes[a][0].shape[0])]
    encoder.train(); decoder.train()
    for step in range(1, cfg.ssl_steps + 1):
        x = augment(torch.cat([context_stack(volumes[a][0], [i]) for a, i in random.choices(pool, k=cfg.batch_size * 2)]))
        loss = reconstruction_loss(encoder, decoder, x, cfg)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        if step % 500 == 0:
            log(f"ssl step {step} | l1={loss.item():.5f}")
    return encoder, decoder


def fine_tune(model, images, masks, support, steps, lr, batch_size):
    """Few-shot adaptation on one mouse's annotated support slices (in place)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        index = random.choices(support, k=batch_size)
        x = context_stack(images, index)
        y = masks[torch.as_tensor(index, device=masks.device)][:, None]
        x, y = augment(x, y)
        loss = segmentation_loss(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()


def self_supervised_tta(model, decoder, images, cfg):
    """Masked reconstruction on the unseen mouse's own unlabelled slices.

    Only the encoder's normalisation parameters adapt; updating every encoder
    weight drifts the features away from what the segmentation decoder expects.
    """
    decoder = copy.deepcopy(decoder)
    norms = [p for m in model.encoder.modules() if isinstance(m, (nn.GroupNorm, nn.LayerNorm)) for p in m.parameters()]
    optimizer = torch.optim.Adam(norms + list(decoder.parameters()), lr=cfg.tta_lr)
    model.train(); decoder.train()
    for _ in range(cfg.tta_steps):
        x = augment(context_stack(images, random.choices(range(images.shape[0]), k=cfg.batch_size)))
        loss = reconstruction_loss(model.encoder, decoder, x, cfg)
        optimizer.zero_grad(); loss.backward(); optimizer.step()


def largest_component(mask):
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return ndimage.binary_fill_holes(labels == (np.argmax(sizes) + 1))


@torch.no_grad()
def predict(model, images):
    model.eval()
    x = context_stack(images, range(images.shape[0]))
    probs = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
    return np.stack([largest_component(p) for p in (probs[:, 0] > 0.5).cpu().numpy()])


def hd95(pred, truth):
    if not truth.any():
        return np.nan
    if not pred.any():
        return float(np.hypot(*truth.shape))
    edge_p = pred ^ ndimage.binary_erosion(pred)
    edge_t = truth ^ ndimage.binary_erosion(truth)
    distances = np.concatenate([ndimage.distance_transform_edt(~edge_t)[edge_p], ndimage.distance_transform_edt(~edge_p)[edge_t]])
    return float(np.percentile(distances, 95))


def score(pred, truth, query):
    pred, truth = pred[query], truth[query]
    intersection = (pred & truth).sum()
    dice = 2 * intersection / max(pred.sum() + truth.sum(), 1)
    iou = intersection / max((pred | truth).sum(), 1)
    return float(dice), float(iou), float(np.nanmean([hd95(p, t) for p, t in zip(pred, truth)]))


def adapt_and_score(model, decoder, images, masks, k, cfg, tta):
    n = images.shape[0]
    query = [i for i in range(n) if i not in support_slices(n, 3)]
    adapted = copy.deepcopy(model)
    if tta:
        self_supervised_tta(adapted, decoder, images, cfg)
    if k:
        fine_tune(adapted, images, masks, support_slices(n, k), cfg.inner_steps, cfg.inner_lr, cfg.batch_size)
    return score(predict(adapted, images), masks.cpu().numpy() > 0.5, query)


def validate(model, volumes, animals, cfg):
    """Mean query Dice over validation mice and k in {0, 1, 3}; same criterion for every method."""
    state = random.getstate()
    random.seed(0)
    dices = [adapt_and_score(model, None, *volumes[a], k, cfg, tta=False)[0] for a in animals for k in SHOTS]
    random.setstate(state)
    return float(np.mean(dices))


def labelled_slices(volumes, animals, regime):
    return {a: list(range(volumes[a][0].shape[0])) if regime == "full" else support_slices(volumes[a][0].shape[0], 3) for a in animals}


def train_joint(model, volumes, labelled, val_animals, cfg, log):
    pool = [(a, i) for a, slices in labelled.items() for i in slices]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine(optimizer, cfg.joint_steps)
    best, best_state = -1.0, None
    for step in range(1, cfg.joint_steps + 1):
        model.train()
        x, y = augment(*batch_pairs(volumes, random.choices(pool, k=cfg.batch_size)))
        loss = segmentation_loss(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        if step % cfg.eval_every == 0:
            val = validate(model, volumes, val_animals, cfg)
            log(f"joint step {step} | loss={loss.item():.4f} | val_dice={val:.4f}")
            if val > best:
                best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best


def train_reptile(model, volumes, labelled, val_animals, cfg, log):
    animals = sorted(labelled)
    best, best_state = -1.0, None
    for step in range(1, cfg.meta_steps + 1):
        epsilon = cfg.meta_lr * (1 - (step - 1) / cfg.meta_steps)
        start = [p.detach().clone() for p in model.parameters()]
        delta = [torch.zeros_like(p) for p in start]
        for _ in range(cfg.meta_batch):
            animal = random.choice(animals)
            slices = labelled[animal]
            support = random.sample(slices, random.randint(1, min(3, len(slices))))
            fine_tune(model, *volumes[animal], support, cfg.inner_steps, cfg.inner_lr, cfg.batch_size)
            with torch.no_grad():
                for d, p, s in zip(delta, model.parameters(), start):
                    d += p - s
                    p.copy_(s)
        with torch.no_grad():
            for p, s, d in zip(model.parameters(), start, delta):
                p.copy_(s + epsilon * d / cfg.meta_batch)
        if step % (cfg.eval_every // 2) == 0:
            val = validate(model, volumes, val_animals, cfg)
            log(f"reptile step {step} | epsilon={epsilon:.3f} | val_dice={val:.4f}")
            if val > best:
                best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--save-checkpoints", action="store_true")
    for field, value in RunConfig.__dataclass_fields__.items():
        if field not in {"data_root", "output_dir"}:
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(value.default), default=value.default)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "experiments" / "meta_ssl" / args.regime / f"fold{args.fold}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(**{k: (str(output) if k == "output_dir" else str(v) if k == "data_root" else v) for k, v in vars(args).items() if k in RunConfig.__dataclass_fields__} | {"output_dir": str(output)})
    log_file = (output / "train.log").open("a")

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True); log_file.write(line + "\n"); log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = make_split(cfg.fold)
    log(f"fold={cfg.fold} seed={cfg.seed} regime={cfg.regime} train={train} val={val} test={test}")
    (output / "config.json").write_text(json.dumps(asdict(cfg) | {"train": train, "val": val, "test": test}, indent=2) + "\n")
    volumes = load_volumes(Path(cfg.data_root), cfg.image_size, device)
    labelled = labelled_slices(volumes, train, cfg.regime)

    set_seed(cfg.seed)
    ssl_encoder, ssl_decoder = pretrain(volumes, train, cfg, device, log)
    rows, val_scores = [], {}
    for offset, method in enumerate(args.methods):
        set_seed(cfg.seed * 100 + offset)
        model = SegmentationModel().to(device)
        if method.startswith("ssl"):
            model.encoder.load_state_dict(ssl_encoder.state_dict())
        started = time.time()
        trainer = train_joint if method.endswith("joint") else train_reptile
        val_scores[method] = trainer(model, volumes, labelled, val, cfg, lambda m: log(f"{method} {m}"))
        log(f"{method} trained in {time.time() - started:.0f}s, best val {val_scores[method]:.4f}")
        if args.save_checkpoints:
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, output / f"{method}.pt")
        for animal in test:
            for tta in ([False, True] if method.startswith("ssl") else [False]):
                for k in SHOTS:
                    set_seed(cfg.seed * 1000 + k)
                    dice, iou, hd = adapt_and_score(model, ssl_decoder, *volumes[animal], k, cfg, tta)
                    rows.append({"regime": cfg.regime, "fold": cfg.fold, "seed": cfg.seed, "method": method, "tta": int(tta), "k": k, "animal": animal, "cohort": cohort(animal), "dice": dice, "iou": iou, "hd95": hd})
                    log(f"{method} tta={int(tta)} k={k} {animal}: dice={dice:.4f} hd95={hd:.2f}")
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "val_scores.json").write_text(json.dumps(val_scores, indent=2) + "\n")
    log("done")


if __name__ == "__main__":
    main()
