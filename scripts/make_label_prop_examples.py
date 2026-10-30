#!/usr/bin/env python3
"""Retrain one fold of the label-propagation study and save qualitative examples:
pseudo-labels on a training mouse and test-mouse predictions, for figures."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_label_propagation_study import (DEFAULT_DATA, ROOT, RunConfig, build_targets, load_volumes, make_split, predict,
                                         set_seed, support_slices, train_registration, train_segmentation)

FOLD, SEED = 0, 1


def main():
    output = ROOT / "results" / "label_prop_examples"
    output.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(data_root=str(DEFAULT_DATA), output_dir=str(output), fold=FOLD, seed=SEED)
    device = torch.device("cuda")
    train, val, test = make_split(FOLD)
    volumes = load_volumes(DEFAULT_DATA, cfg.image_size, device)
    anchors = {a: support_slices(volumes[a][0].shape[0], 3) for a in train}
    set_seed(SEED * 10 + 1)
    nets = {"reg_anchor": train_registration(volumes, train, anchors, cfg, device, True, print)}

    saved = {f"{a}_image": volumes[a][0].cpu().numpy() for a in train + test}
    saved |= {f"{a}_truth": volumes[a][1].cpu().numpy() for a in train + test}
    teacher = None
    for offset, method in enumerate(["sparse", "copy", "sdf", "self_train", "reg_teacher", "full"]):
        set_seed(SEED * 100 + offset)
        targets, _ = build_targets(method, volumes, train, anchors, nets, teacher)
        for animal, rows in targets.items():
            pseudo = np.full(volumes[animal][1].shape, np.nan, dtype=np.float32)
            ignored = np.zeros_like(pseudo)
            for i, target, weight in rows:
                pseudo[i] = target.cpu().numpy()
                ignored[i] = (weight == 0).cpu().numpy()
            saved[f"{animal}_pseudo_{method}"], saved[f"{animal}_ignored_{method}"] = pseudo, ignored
        if method in ("sparse", "reg_teacher", "full"):
            model, _ = train_segmentation(volumes, targets, val, cfg, lambda m: print(method, m))
            if method == "sparse":
                teacher = model
            for animal in test:
                saved[f"{animal}_prediction_{method}"] = predict(model, volumes[animal][0])
    np.savez_compressed(output / f"fold{FOLD}_seed{SEED}.npz", anchors=np.array([anchors[a] for a in train]), train=np.array(train), test=np.array(test), **saved)
    print("saved", output / f"fold{FOLD}_seed{SEED}.npz")


if __name__ == "__main__":
    main()
