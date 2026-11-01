#!/usr/bin/env python3
"""Adapting a trained brain-extraction network to a new acquisition protocol
with one annotated slice per mouse.

A network is first trained with full annotation on the source cohort (eight
mice; two more for checkpoint selection), exactly as in the cross-cohort runs of
run_label_propagation_study.py. It is then fine-tuned on k mice of the target
cohort, each with only its middle slice annotated, and tested on held-out mice
of the target cohort. Target mice are split into five folds of two test mice;
the other eight form an ordered adaptation pool, and the first k of them are
used, so k = 2, 4, 8 are nested and every target mouse is tested once per k.

Methods (target-side supervision during fine-tuning):
    source     no fine-tuning (the cross-cohort baseline)
    ft_anchor  the annotated middle slice of each adaptation mouse only
    ft_sam2    that slice plus SAM2 labels for the other slices
    ft_full    every slice of each adaptation mouse annotated (reference)

Fine-tuning mixes source and target slices half and half in every batch and
keeps the final checkpoint, because no annotated target mouse is available for
checkpoint selection in this setting.

    python run_cohort_adaptation.py --split B_to_A --seed 1
SAM2 labels are read from results/sam2_pseudo_one_mid.npz
(scripts/sam2_propagate.py --anchor-layout one_mid), which covers all 20 mice.
"""

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from run_label_propagation_study import cross_cohort_split, train_segmentation, weighted_segmentation_loss
from run_meta_ssl_study import (DEFAULT_DATA, ROOT, SegmentationModel, augment, cohort, context_stack, cosine,
                                load_volumes, predict, score, set_seed)

METHODS = ["ft_anchor", "ft_sam2", "ft_full"]


@dataclass
class AdaptConfig:
    data_root: str
    output_dir: str
    split: str = "B_to_A"
    seed: int = 1
    image_size: int = 256
    batch_size: int = 8
    steps: int = 8000          # source training, as in the main study
    lr: float = 3e-4
    weight_decay: float = 1e-4
    eval_every: int = 500
    ft_steps: int = 2000       # fine-tuning on the target cohort
    ft_lr: float = 1e-4
    sam2_file: str = "results/sam2_pseudo_one_mid.npz"


def target_folds(target, n_folds=5):
    """Five folds of two test mice; the remaining eight, in a fixed order, form the adaptation pool."""
    order = sorted(target, key=lambda a: int(a[:-1]))
    random.Random(2027).shuffle(order)
    folds = []
    for f in range(n_folds):
        test = order[2 * f:2 * f + 2]
        pool = [a for a in order[2 * f + 2:] + order[:2 * f]]
        folds.append((test, pool))
    return folds


def target_rows(method, volumes, animals, sam2):
    """Per animal: list of (slice, target, pixel weight), as in build_targets."""
    rows = {}
    for animal in animals:
        images, masks = volumes[animal]
        ones = torch.ones_like(masks[0])
        if method == "ft_full":
            rows[animal] = [(i, masks[i], ones) for i in range(images.shape[0])]
            continue
        anchors = [int(i) for i in sam2[f"{animal}_anchors"]]
        rows[animal] = [(i, masks[i], ones) for i in anchors]
        if method == "ft_sam2":
            probs = torch.from_numpy(sam2[animal]).to(masks.device)
            rows[animal] += [(i, (probs[i] > 0.5).float(), ones) for i in range(images.shape[0]) if i not in anchors]
    return rows


def fine_tune(source_state, volumes, source_rows, target, cfg):
    device = next(iter(volumes.values()))[0].device
    model = SegmentationModel().to(device)
    model.load_state_dict(source_state)
    src_pool = [(a, i, t, w) for a, rows in source_rows.items() for i, t, w in rows]
    tgt_pool = [(a, i, t, w) for a, rows in target.items() for i, t, w in rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.weight_decay)
    scheduler = cosine(optimizer, cfg.ft_steps)
    half = cfg.batch_size // 2
    for _ in range(cfg.ft_steps):
        model.train()
        batch = random.choices(src_pool, k=cfg.batch_size - half) + random.choices(tgt_pool, k=half)
        x = torch.cat([context_stack(volumes[a][0], [i]) for a, i, _, _ in batch])
        y = torch.stack([torch.stack([t, w]) for _, _, t, w in batch])
        x, y = augment(x, y)
        loss = weighted_segmentation_loss(model(x), y[:, :1], y[:, 1:])
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--k", nargs="+", type=int, default=[2, 4, 8], help="adaptation mice from the target cohort")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    for field, value in AdaptConfig.__dataclass_fields__.items():
        if field not in {"data_root", "output_dir"}:
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(value.default), default=value.default)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "experiments" / "cohort_adaptation" / f"{args.split}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    cfg = AdaptConfig(**{k: v for k, v in vars(args).items() if k in AdaptConfig.__dataclass_fields__}
                      | {"data_root": str(args.data_root), "output_dir": str(output)})
    log_file = (output / "train.log").open("a")

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True); log_file.write(line + "\n"); log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_train, source_val, target = cross_cohort_split(cfg.split)
    folds = target_folds(target)
    log(f"split={cfg.split} seed={cfg.seed} source_train={source_train} source_val={source_val} k={args.k} folds={args.folds}")
    (output / "config.json").write_text(json.dumps(asdict(cfg) | {"source_train": source_train, "source_val": source_val,
                                                                  "k": args.k, "methods": args.methods,
                                                                  "target_folds": [{"test": t, "pool": p} for t, p in folds]}, indent=2) + "\n")
    volumes = load_volumes(Path(cfg.data_root), cfg.image_size, device)
    sam2 = np.load(ROOT / cfg.sam2_file) if {"ft_anchor", "ft_sam2"} & set(args.methods) else None

    source_rows = {a: [(i, volumes[a][1][i], torch.ones_like(volumes[a][1][i])) for i in range(volumes[a][0].shape[0])]
                   for a in source_train}
    checkpoint = output / "source.pt"
    if checkpoint.exists():
        source_state = torch.load(checkpoint, map_location=device)
        log("loaded source model")
    else:
        set_seed(cfg.seed)
        started = time.time()
        model, best = train_segmentation(volumes, source_rows, source_val, cfg, lambda m: log(f"source {m}"))
        source_state = model.state_dict()
        torch.save(source_state, checkpoint)
        log(f"source trained in {time.time() - started:.0f}s, best val {best:.4f}")

    rows = []

    def evaluate(model, method, k, fold, test):
        for animal in test:
            images, masks = volumes[animal]
            dice, iou, hd = score(predict(model, images), masks.cpu().numpy() > 0.5, list(range(images.shape[0])))
            rows.append({"split": cfg.split, "seed": cfg.seed, "target_fold": fold, "k": k, "method": method,
                         "animal": animal, "cohort": cohort(animal), "dice": dice, "iou": iou, "hd95": hd})
            log(f"fold{fold} k={k} {method} {animal}: dice={dice:.4f} hd95={hd:.2f}")
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    source_model = SegmentationModel().to(device)
    source_model.load_state_dict(source_state)
    for fold in args.folds:
        test, pool = folds[fold]
        evaluate(source_model.eval(), "source", 0, fold, test)
        for k in args.k:
            adapt = pool[:k]
            for offset, method in enumerate(args.methods):
                set_seed(cfg.seed * 1000 + fold * 100 + k * 10 + offset)
                started = time.time()
                model = fine_tune(source_state, volumes, source_rows, target_rows(method, volumes, adapt, sam2), cfg)
                log(f"fold{fold} k={k} {method} adapt={adapt} fine-tuned in {time.time() - started:.0f}s")
                evaluate(model, method, k, fold, test)
    log("done")


if __name__ == "__main__":
    main()
