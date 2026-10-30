#!/usr/bin/env python3
"""Cross-slice self-supervised pretraining for mouse brain extraction with
one or three annotated slices per training mouse.

Pretraining objectives (any combination, on the unlabelled slices of the
training mice only):
  A  latent prediction: from one isolated slice, predict the EMA encoder's
     tokens of the slice Delta in {-2, -1, +1, +2} away (Delta is embedded).
  B  deformation: a pair head on both slices' features predicts the flow that
     warps slice i onto slice i+1 (local NCC + smoothness).
  C  appearance invariance: BYOL on pooled tokens of two augmented views.
  D  topology events: intensity-tertile structures of slice i are labelled
     persisting or disappearing by whether they continue into slice i+1, and
     those of slice i+1 persisting or appearing; the pair head predicts both maps.
  M  masked image modelling (the previous paper's objective), as a baseline.

Each pretrained encoder initialises the same 2.5D CNN-Transformer, which is
trained on the annotated slices only ("one_mid" or "quartile" layout) and
tested on the fold's unseen mice. Split, training and scoring are those of
run_label_propagation_study.py.
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from run_label_propagation_study import anchor_slices, build_targets, ncc_loss, smoothness, train_segmentation, warp
from run_meta_ssl_study import (DEFAULT_DATA, ROOT, Encoder, ReconstructionDecoder, augment, cohort, context_stack, cosine, gn,
                                load_volumes, make_split, predict, reconstruction_loss, score, set_seed)

VARIANTS = ["scratch", "M", "A", "B", "C", "D", "ABCD"]
REGIMES = ["one_mid", "quartile"]


@dataclass
class RunConfig:
    data_root: str
    output_dir: str
    fold: int = 0
    seed: int = 1
    image_size: int = 256
    batch_size: int = 8
    ssl_steps: int = 4000
    ssl_lr: float = 5e-4
    ema: float = 0.996
    smooth_weight: float = 1.0
    steps: int = 8000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    eval_every: int = 500
    mixer: str = "transformer"
    context: str = "2.5d"


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, o))


class CrossSliceSSL(nn.Module):
    def __init__(self, objectives, **encoder_kwargs):
        super().__init__()
        self.objectives = objectives
        self.encoder = Encoder(**encoder_kwargs)
        if "A" in objectives or "C" in objectives:
            self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        if "A" in objectives:
            self.delta = nn.Embedding(4, 256)
            layer = nn.TransformerEncoderLayer(256, 8, 512, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.predictor = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        if "C" in objectives:
            self.projector, self.predictor_c = mlp(256, 512, 128), mlp(128, 512, 128)
            self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
        if "B" in objectives or "D" in objectives:
            # Pair head on both slices' stem (64@64^2) and upsampled token (256@16^2) features.
            self.pair = nn.Sequential(nn.Conv2d(640, 128, 3, padding=1), gn(128), nn.GELU(), nn.Conv2d(128, 128, 3, padding=1), gn(128), nn.GELU())
        if "B" in objectives:
            self.flow = nn.Conv2d(128, 2, 3, padding=1)
            nn.init.normal_(self.flow.weight, std=1e-5); nn.init.zeros_(self.flow.bias)
        if "D" in objectives:
            self.events = nn.Conv2d(128, 6, 1)
        if "M" in objectives:
            self.reconstruction = ReconstructionDecoder()

    def pair_features(self, x_i, x_j):
        feats = []
        for x in (x_i, x_j):
            shallow, encoded = self.encoder(x)
            feats += [shallow, F.interpolate(encoded, size=shallow.shape[-2:], mode="bilinear", align_corners=False)]
        return self.pair(torch.cat(feats, 1))

    @torch.no_grad()
    def update_targets(self, momentum):
        pairs = []
        if hasattr(self, "target"):
            pairs.append((self.encoder, self.target))
        if hasattr(self, "target_projector"):
            pairs.append((self.projector, self.target_projector))
        for online, target in pairs:
            for p, t in zip(online.parameters(), target.parameters()):
                t.mul_(momentum).add_(p.detach(), alpha=1 - momentum)


def intensity_augment(x):
    """Gamma and noise only, so slices of a pair and their labels stay aligned."""
    x = x.clamp(0, 1) ** torch.exp((torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.6)
    return (x + 0.02 * torch.randn_like(x)).clamp(0, 1)


def event_labels(images, min_size=8, overlap=0.5):
    """Per adjacent pair (i, i+1): 64x64 maps for slice i (0 none, 1 persists, 2 disappears)
    and slice i+1 (0 none, 1 persists, 2 appears). Structures are connected components
    of the volume's dark/mid/bright intensity tertiles inside the head."""
    small = F.avg_pool2d(images[:, None], 4)[:, 0].cpu().numpy()
    low, high = np.percentile(small[small > 0.05], [100 / 3, 200 / 3])
    classes = np.digitize(small, [0.05, low, high])

    def label(cur, other):
        out = np.zeros(cur.shape, np.int64)
        for c in (1, 2, 3):
            components, count = ndimage.label(cur == c)
            for k in range(1, count + 1):
                comp = components == k
                if comp.sum() >= min_size:
                    out[comp] = 1 if (comp & (other == c)).sum() >= overlap * comp.sum() else 2
        return out

    return {i: (label(classes[i], classes[i + 1]), label(classes[i + 1], classes[i])) for i in range(len(classes) - 1)}


def pretrain(objectives, volumes, animals, cfg, device, log):
    model = CrossSliceSSL(objectives, mixer=cfg.mixer, context=cfg.context).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.ssl_lr, weight_decay=0.05)
    scheduler = cosine(optimizer, cfg.ssl_steps)
    events = {a: event_labels(volumes[a][0]) for a in animals} if "D" in objectives else None
    if events:
        counts = np.bincount(np.concatenate([np.concatenate([e[0].ravel(), e[1].ravel()]) for v in events.values() for e in v.values()]), minlength=3)
        log(f"event label fractions (none/persist/change): {np.round(counts / counts.sum(), 3).tolist()}")
        # Changes are rare (about 7% of pixels), so weight classes by inverse frequency.
        event_weight = torch.as_tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    singles = [(a, i) for a in animals for i in range(volumes[a][0].shape[0])]
    adjacent = [(a, i) for a in animals for i in range(volumes[a][0].shape[0] - 1)]
    mim_cfg = SimpleNamespace(patch_size=32, mask_ratio=0.5)
    model.train()
    for step in range(1, cfg.ssl_steps + 1):
        losses, pooled = {}, []
        if "M" in objectives:
            x = augment(torch.cat([context_stack(volumes[a][0], [i]) for a, i in random.choices(singles, k=cfg.batch_size)]))
            losses["M"] = reconstruction_loss(model.encoder, model.reconstruction, x, mim_cfg)
        if "A" in objectives:
            items = []
            while len(items) < cfg.batch_size:
                a, i = random.choice(singles)
                d = random.choice([-2, -1, 1, 2])
                if 0 <= i + d < volumes[a][0].shape[0]:
                    items.append((a, i, d))
            # The online encoder sees the slice alone, so the neighbour cannot be read off the input.
            alone = intensity_augment(torch.stack([volumes[a][0][i] for a, i, _ in items])[:, None].repeat(1, 3, 1, 1))
            tokens = model.encoder(alone)[1].flatten(2).transpose(1, 2)
            delta = torch.as_tensor([[-2, -1, 1, 2].index(d) for _, _, d in items], device=device)
            predicted = model.predictor(tokens + model.delta(delta)[:, None])
            with torch.no_grad():
                target = model.target(torch.cat([context_stack(volumes[a][0], [i + d]) for a, i, d in items]))[1].flatten(2).transpose(1, 2)
                target = F.layer_norm(target, target.shape[-1:])
            losses["A"] = (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()
            pooled.append(tokens.mean(1))
        if "C" in objectives:
            x = torch.cat([context_stack(volumes[a][0], [i]) for a, i in random.choices(singles, k=cfg.batch_size)])
            features = model.encoder(augment(x))[1].mean((2, 3))
            pooled.append(features)
            online = model.predictor_c(model.projector(features))
            with torch.no_grad():
                target = model.target_projector(model.target(augment(x))[1].mean((2, 3)))
            losses["C"] = (2 - 2 * F.cosine_similarity(online, target, dim=-1)).mean()
        if "B" in objectives or "D" in objectives:
            items = random.choices(adjacent, k=cfg.batch_size)
            x_i = intensity_augment(torch.cat([context_stack(volumes[a][0], [i]) for a, i in items]))
            x_j = intensity_augment(torch.cat([context_stack(volumes[a][0], [i + 1]) for a, i in items]))
            features = model.pair_features(x_i, x_j)
            if "B" in objectives:
                flow = 4 * F.interpolate(model.flow(features), scale_factor=4, mode="bilinear", align_corners=False)
                moving = torch.stack([volumes[a][0][i] for a, i in items])[:, None]
                fixed = torch.stack([volumes[a][0][i + 1] for a, i in items])[:, None]
                losses["B"] = ncc_loss(warp(moving, flow), fixed) + cfg.smooth_weight * smoothness(flow)
            if "D" in objectives:
                logits = model.events(features)
                labels_i = torch.as_tensor(np.stack([events[a][i][0] for a, i in items]), device=device)
                labels_j = torch.as_tensor(np.stack([events[a][i][1] for a, i in items]), device=device)
                losses["D"] = (F.cross_entropy(logits[:, :3], labels_i, weight=event_weight) + F.cross_entropy(logits[:, 3:], labels_j, weight=event_weight)) / 2
        if pooled:
            # Variance floor on the online features (as in VICReg) guards the EMA objectives against collapse.
            spread = torch.cat(pooled).std(0)
            losses["var"] = F.relu(1 - spread).mean()
        loss = sum(losses.values())
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        # EMA momentum ramps from cfg.ema to 1 (as in I-JEPA), so late targets stop drifting.
        model.update_targets(1 - (1 - cfg.ema) * (math.cos(math.pi * step / cfg.ssl_steps) + 1) / 2)
        if step % 500 == 0:
            log(f"ssl step {step} | " + " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                + (f" | feature std={spread.mean().item():.3f}" if pooled else ""))
    return copy.deepcopy(model.encoder.state_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--regimes", nargs="+", default=REGIMES, choices=REGIMES)
    for field, value in RunConfig.__dataclass_fields__.items():
        if field not in {"data_root", "output_dir"}:
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(value.default), default=value.default)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "experiments" / "cross_slice_ssl" / f"fold{args.fold}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(**{k: v for k, v in vars(args).items() if k in RunConfig.__dataclass_fields__} | {"data_root": str(args.data_root), "output_dir": str(output)})
    log_file = (output / "train.log").open("a")

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True); log_file.write(line + "\n"); log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = make_split(cfg.fold)
    log(f"fold={cfg.fold} seed={cfg.seed} train={train} val={val} test={test}")
    (output / "config.json").write_text(json.dumps(asdict(cfg) | {"train": train, "val": val, "test": test, "variants": args.variants}, indent=2) + "\n")
    volumes = load_volumes(Path(cfg.data_root), cfg.image_size, device)

    rows, val_scores = [], {}
    for v_index, variant in enumerate(args.variants):
        set_seed(cfg.seed * 1000 + v_index)
        state = None
        if variant != "scratch":
            started = time.time()
            state = pretrain(set(variant), volumes, train, cfg, device, lambda m: log(f"{variant} {m}"))
            log(f"{variant} pretrained in {time.time() - started:.0f}s")
        for regime in args.regimes:
            set_seed(cfg.seed * 100 + REGIMES.index(regime))
            anchors = {a: anchor_slices(volumes[a][0].shape[0], regime) for a in train}
            targets, _ = build_targets("sparse", volumes, train, anchors, {}, None)
            name = f"{variant}/{regime}"
            model, val_scores[name] = train_segmentation(volumes, targets, val, cfg, lambda m: log(f"{name} {m}"), encoder_state=state,
                                                        encoder_kwargs={"mixer": cfg.mixer, "context": cfg.context})
            for animal in test:
                images, masks = volumes[animal]
                dice, iou, hd = score(predict(model, images), masks.cpu().numpy() > 0.5, list(range(images.shape[0])))
                rows.append({"fold": cfg.fold, "seed": cfg.seed, "variant": variant, "regime": regime, "mixer": cfg.mixer, "context": cfg.context, "animal": animal,
                             "cohort": cohort(animal), "dice": dice, "iou": iou, "hd95": hd})
                log(f"{name} {animal}: dice={dice:.4f} hd95={hd:.2f}")
            with (output / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "val_scores.json").write_text(json.dumps(val_scores, indent=2) + "\n")
    log("done")


if __name__ == "__main__":
    main()
