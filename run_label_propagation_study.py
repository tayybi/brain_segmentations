#!/usr/bin/env python3
"""Sparse-to-dense label propagation for mouse brain extraction.

Each training mouse has only 3 annotated slices (anchors at 1/4, 1/2, 3/4 of the
stack). A small registration network learns slice-to-slice deformations from
image similarity, plus an anchor-consistency loss: chaining the deformations
from one anchor to the next must carry the first anchor's mask onto the second.
The proposed variant (reg_teacher) fuses the propagated masks with the
predictions of a model trained on the anchors alone, ignores pixels where the
two disagree, and retrains the 2.5D segmentation model on anchors plus fused
pseudo-labels.

Baselines: anchors only, copying the nearest anchor mask, signed-distance shape
interpolation, registration without and with the anchor loss, plain
self-training on the anchor model's predictions, and all slices labelled.
Splits, preprocessing and the segmentation model come from run_meta_ssl_study.py.
"""

import argparse
import copy
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from run_meta_ssl_study import (COHORTS, DEFAULT_DATA, ROOT, SegmentationModel, augment, cohort, context_stack, cosine,
                                load_volumes, make_split, predict, score, set_seed, support_slices)


# "sparse" must run first: its model is the teacher for self_train, reg_teacher and sam2_teacher.
# sam2 methods read propagations precomputed by scripts/sam2_propagate.py.
METHODS = ["sparse", "copy", "sdf", "reg_unsup", "reg_anchor", "self_train", "reg_teacher", "full"]
SAM2_METHODS = ["sam2", "sam2_teacher"]


@dataclass
class RunConfig:
    data_root: str
    output_dir: str
    fold: int = 0
    seed: int = 1
    image_size: int = 256
    batch_size: int = 8
    steps: int = 8000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    eval_every: int = 500
    reg_steps: int = 4000
    reg_lr: float = 3e-4
    smooth_weight: float = 1.0
    anchor_weight: float = 1.0
    split: str = "cv"  # "cv": five-fold split of both cohorts; "A_to_B" / "B_to_A": train on one cohort, test on the other
    anchor_layout: str = "quartile"  # "quartile": slices n/4, n/2, 3n/4; "spread": slices 1, n/2, n-2


class RegistrationNet(nn.Module):
    """Small U-Net predicting a displacement field (pixels) that warps moving onto fixed."""

    def __init__(self, width=32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(o, o, 3, padding=1), nn.LeakyReLU(0.2))

        w = width
        self.e1, self.e2, self.e3, self.e4 = block(2, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w)
        self.d3, self.d2, self.d1 = block(8 * w, 4 * w), block(6 * w, 2 * w), block(3 * w, w)
        self.flow = nn.Conv2d(w, 2, 3, padding=1)
        nn.init.normal_(self.flow.weight, std=1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, moving, fixed):
        # Adjacent-slice deformations are smooth, so estimate them at half resolution.
        up = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return 2 * up(self.half_resolution_flow(F.avg_pool2d(moving, 2), F.avg_pool2d(fixed, 2)))

    def half_resolution_flow(self, moving, fixed):
        up = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        e1 = self.e1(torch.cat([moving, fixed], 1))
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        e4 = self.e4(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([up(e4), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        return self.flow(self.d1(torch.cat([up(d2), e1], 1)))


def warp(image, flow):
    n, _, h, w = image.shape
    identity = F.affine_grid(torch.eye(2, 3, device=image.device).expand(n, 2, 3), [n, 1, h, w], align_corners=False)
    offset = torch.stack([flow[:, 0] * 2 / w, flow[:, 1] * 2 / h], -1)
    return F.grid_sample(image, identity + offset, mode="bilinear", padding_mode="border", align_corners=False)


def ncc_loss(a, b, window=9):
    pool = lambda x: F.avg_pool2d(x, window, stride=1, padding=window // 2)
    mean_a, mean_b = pool(a), pool(b)
    var_a, var_b = pool(a * a) - mean_a ** 2, pool(b * b) - mean_b ** 2
    covariance = pool(a * b) - mean_a * mean_b
    return -(covariance ** 2 / (var_a * var_b + 1e-5)).mean()


def smoothness(flow):
    return ((flow[:, :, 1:] - flow[:, :, :-1]) ** 2).mean() + ((flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2).mean()


def soft_dice_loss(pred, target):
    return 1 - (2 * (pred * target).sum() + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def chain(net, images, mask, start, end):
    """Carry a mask from slice start to slice end through adjacent-slice deformations."""
    step = 1 if end > start else -1
    for j in range(start, end, step):
        mask = warp(mask, net(images[j:j + 1, None], images[j + step:j + step + 1, None]))
    return mask


def train_registration(volumes, animals, anchors, cfg, device, use_anchors, log):
    net = RegistrationNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.reg_lr)
    scheduler = cosine(optimizer, cfg.reg_steps)
    pairs = [(a, i) for a in animals for i in range(volumes[a][0].shape[0] - 1)]
    for step in range(1, cfg.reg_steps + 1):
        net.train()
        batch = random.choices(pairs, k=16)
        reverse = [random.random() < 0.5 for _ in batch]
        moving = torch.stack([volumes[a][0][i + 1 if r else i] for (a, i), r in zip(batch, reverse)])[:, None]
        fixed = torch.stack([volumes[a][0][i if r else i + 1] for (a, i), r in zip(batch, reverse)])[:, None]
        if random.random() < 0.5:
            moving, fixed = moving.flip(-1), fixed.flip(-1)
        flow = net(moving, fixed)
        loss = ncc_loss(warp(moving, flow), fixed) + cfg.smooth_weight * smoothness(flow)
        if use_anchors:
            animal = random.choice(animals)
            images, masks = volumes[animal]
            pair = random.randrange(len(anchors[animal]) - 1)
            start, end = anchors[animal][pair], anchors[animal][pair + 1]
            if random.random() < 0.5:
                start, end = end, start
            carried = chain(net, images, masks[start][None, None], start, end)
            loss = loss + cfg.anchor_weight * soft_dice_loss(carried, masks[end][None, None])
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        if step % 1000 == 0:
            log(f"registration step {step} | loss={loss.item():.4f}")
    return net.eval()


def cross_cohort_split(split):
    """Train on eight mice of one acquisition cohort, validate on two, test on all ten of the other."""
    source, target = (COHORTS["A"], COHORTS["B"]) if split == "A_to_B" else (COHORTS["B"], COHORTS["A"])
    return list(source[:8]), list(source[8:]), list(target)


def anchor_slices(n_slices, layout):
    if layout == "spread":
        # Second and second-to-last slice: no unannotated slice is more than one step past an anchor.
        return [1, n_slices // 2, n_slices - 2]
    if layout == "one_mid":
        # One annotated slice per mouse, the middle one.
        return [n_slices // 2]
    return support_slices(n_slices, 3)


def signed_distance(mask):
    mask = mask.astype(bool)
    return ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)


@torch.no_grad()
def propagate(method, net, images, masks, anchors):
    """Soft pseudo-label in [0, 1] for every non-anchor slice, keyed by slice index."""
    labels = {}
    for i in range(images.shape[0]):
        if i in anchors:
            continue
        lower = max((a for a in anchors if a < i), default=None)
        upper = min((a for a in anchors if a > i), default=None)
        sides = [(a, abs(i - a)) for a in (lower, upper) if a is not None]
        total = sum(1 / d for _, d in sides)
        if method == "sdf":
            distance = sum(signed_distance(masks[a].cpu().numpy() > 0.5) / d for a, d in sides) / total
            labels[i] = torch.from_numpy((distance > 0).astype(np.float32)).to(images.device)
            continue
        fused = 0
        for a, d in sides:
            carried = masks[a][None, None] if method == "copy" else chain(net, images, masks[a][None, None], a, i)
            fused = fused + carried[0, 0] / d
        labels[i] = fused / total
    return labels


@torch.no_grad()
def teacher_probabilities(model, images):
    model.eval()
    x = context_stack(images, range(images.shape[0]))
    return ((torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2)[:, 0]


def build_targets(method, volumes, animals, anchors, nets, teacher, sam2=None):
    """Per animal: list of (slice, target, pixel weight); also pseudo-label quality rows."""
    targets, quality = {}, []
    for animal in animals:
        images, masks = volumes[animal]
        ones = torch.ones_like(masks[0])
        if method == "full":
            targets[animal] = [(i, masks[i], ones) for i in range(images.shape[0])]
            continue
        targets[animal] = [(i, masks[i], ones) for i in anchors[animal]]
        if method == "sparse":
            continue
        if method == "self_train":
            probs = teacher_probabilities(teacher, images)
            pseudo = {i: probs[i] for i in range(images.shape[0]) if i not in anchors[animal]}
        elif method.startswith("sam2"):
            pseudo = {i: sam2[animal][i] for i in range(images.shape[0]) if i not in anchors[animal]}
        else:
            net = nets.get("reg_anchor" if method == "reg_teacher" else method)
            pseudo = propagate(method if method in ("copy", "sdf") else "reg", net, images, masks, anchors[animal])
        if method in ("reg_teacher", "sam2_teacher"):
            probs = teacher_probabilities(teacher, images)
        for i, soft in pseudo.items():
            hard = (soft > 0.5).float()
            weight = ones
            if method in ("reg_teacher", "sam2_teacher"):
                # Geometry (propagation) and appearance (anchor model) must agree.
                weight = ((soft > 0.5) == (probs[i] > 0.5)).float()
                if method == "reg_teacher":
                    hard = ((soft + probs[i]) / 2 > 0.5).float()
                # sam2_teacher keeps SAM2's label: it is more accurate than the anchor-only teacher.
            targets[animal].append((i, hard, weight))
            truth = masks[i] > 0.5
            kept = weight > 0
            dice = 2 * ((hard > 0) & truth).sum() / ((hard > 0).sum() + truth.sum()).clamp(min=1)
            accuracy_kept = ((hard > 0) == truth)[kept].float().mean() if kept.any() else torch.tensor(float("nan"))
            quality.append({"method": method, "animal": animal, "slice": i, "distance_to_anchor": min(abs(i - a) for a in anchors[animal]),
                            "pseudo_dice": dice.item(), "ignored_fraction": 1 - kept.float().mean().item(), "accuracy_on_kept": accuracy_kept.item()})
    return targets, quality


def weighted_segmentation_loss(logits, target, weight):
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum().clamp(min=1)
    probs, target = torch.sigmoid(logits) * weight, target * weight
    dice = 1 - ((2 * (probs * target).sum((1, 2, 3)) + 1e-6) / (probs.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)).mean()
    return 0.5 * bce + 0.5 * dice


def volume_dice(model, volumes, animals):
    dices = []
    for animal in animals:
        images, masks = volumes[animal]
        dices.append(score(predict(model, images), masks.cpu().numpy() > 0.5, list(range(images.shape[0])))[0])
    return float(np.mean(dices))


def train_segmentation(volumes, targets, val_animals, cfg, log, encoder_state=None, encoder_kwargs=None):
    model = SegmentationModel(**(encoder_kwargs or {})).to(next(iter(volumes.values()))[0].device)
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state)
    pool = [(a, i, t, w) for a, rows in targets.items() for i, t, w in rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine(optimizer, cfg.steps)
    best, best_state = -1.0, None
    for step in range(1, cfg.steps + 1):
        model.train()
        batch = random.choices(pool, k=cfg.batch_size)
        x = torch.cat([context_stack(volumes[a][0], [i]) for a, i, _, _ in batch])
        y = torch.stack([torch.stack([t, w]) for _, _, t, w in batch])
        x, y = augment(x, y)
        loss = weighted_segmentation_loss(model(x), y[:, :1], y[:, 1:])
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        if step % cfg.eval_every == 0:
            val = volume_dice(model, volumes, val_animals)
            log(f"step {step} | loss={loss.item():.4f} | val_dice={val:.4f}")
            if val > best:
                best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS + SAM2_METHODS)
    for field, value in RunConfig.__dataclass_fields__.items():
        if field not in {"data_root", "output_dir"}:
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(value.default), default=value.default)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "experiments" / "label_prop" / f"fold{args.fold}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(**{k: v for k, v in vars(args).items() if k in RunConfig.__dataclass_fields__} | {"data_root": str(args.data_root), "output_dir": str(output)})
    log_file = (output / "train.log").open("a")

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True); log_file.write(line + "\n"); log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = make_split(cfg.fold) if cfg.split == "cv" else cross_cohort_split(cfg.split)
    log(f"split={cfg.split} fold={cfg.fold} seed={cfg.seed} train={train} val={val} test={test}")
    (output / "config.json").write_text(json.dumps(asdict(cfg) | {"train": train, "val": val, "test": test}, indent=2) + "\n")
    volumes = load_volumes(Path(cfg.data_root), cfg.image_size, device)
    anchors = {a: anchor_slices(volumes[a][0].shape[0], cfg.anchor_layout) for a in train}

    nets = {}
    needed = {"reg_unsup": {"reg_unsup"}, "reg_anchor": {"reg_anchor", "reg_teacher"}}
    for name, use_anchors in (("reg_unsup", False), ("reg_anchor", True)):
        if needed[name] & set(args.methods):
            set_seed(cfg.seed * 10 + use_anchors)
            started = time.time()
            nets[name] = train_registration(volumes, train, anchors, cfg, device, use_anchors, lambda m: log(f"{name} {m}"))
            log(f"{name} trained in {time.time() - started:.0f}s")

    sam2 = None
    if any(m.startswith("sam2") for m in args.methods):
        stored = np.load(ROOT / "results" / f"sam2_pseudo_{cfg.anchor_layout}.npz")
        sam2 = {a: torch.from_numpy(stored[a]).to(device) for a in train}
    rows, quality, val_scores, teacher = [], [], {}, None
    for offset, method in enumerate(args.methods):
        set_seed(cfg.seed * 100 + offset)
        targets, method_quality = build_targets(method, volumes, train, anchors, nets, teacher, sam2)
        quality += method_quality
        if method_quality:
            log(f"{method} pseudo-label dice={np.mean([q['pseudo_dice'] for q in method_quality]):.4f} ignored={np.mean([q['ignored_fraction'] for q in method_quality]):.3f}")
        started = time.time()
        model, val_scores[method] = train_segmentation(volumes, targets, val, cfg, lambda m: log(f"{method} {m}"))
        log(f"{method} trained in {time.time() - started:.0f}s, best val {val_scores[method]:.4f}")
        if method == "sparse":
            teacher = model
        for animal in test:
            images, masks = volumes[animal]
            dice, iou, hd = score(predict(model, images), masks.cpu().numpy() > 0.5, list(range(images.shape[0])))
            rows.append({"fold": cfg.fold, "seed": cfg.seed, "method": method, "animal": animal, "cohort": cohort(animal), "dice": dice, "iou": iou, "hd95": hd})
            log(f"{method} {animal}: dice={dice:.4f} hd95={hd:.2f}")
        for name, data in (("metrics.csv", rows), ("pseudo_quality.csv", quality)):
            if data:
                with (output / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (output / "val_scores.json").write_text(json.dumps(val_scores, indent=2) + "\n")
    log("done")


if __name__ == "__main__":
    main()
