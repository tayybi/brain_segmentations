#!/usr/bin/env python3
"""Retrain the registration network with anchor loss for fold 0, seed 1 (three anchors)
and save its unfiltered propagated labels on the training mice, for Fig. 2
(results/label_prop_examples/fold0_seed1_reg_anchor.npz)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_label_propagation_study import (DEFAULT_DATA, ROOT, RunConfig, build_targets, load_volumes, make_split,
                                         set_seed, support_slices, train_registration)

FOLD, SEED = 0, 1


def main():
    output = ROOT / "results" / "label_prop_examples"
    cfg = RunConfig(data_root=str(DEFAULT_DATA), output_dir=str(output), fold=FOLD, seed=SEED)
    device = torch.device("cuda")
    train, _, _ = make_split(FOLD)
    volumes = load_volumes(DEFAULT_DATA, cfg.image_size, device)
    anchors = {a: support_slices(volumes[a][0].shape[0], 3) for a in train}  # same anchors as fold0_seed1.npz
    set_seed(SEED * 10 + 1)
    nets = {"reg_anchor": train_registration(volumes, train, anchors, cfg, device, True, print)}
    targets, _ = build_targets("reg_anchor", volumes, train, anchors, nets, None)
    saved = {}
    for animal, rows in targets.items():
        pseudo = np.full(volumes[animal][1].shape, np.nan, dtype=np.float32)
        for i, target, _ in rows:
            pseudo[i] = target.cpu().numpy()
        saved[f"{animal}_pseudo_reg_anchor"] = pseudo
    np.savez_compressed(output / f"fold{FOLD}_seed{SEED}_reg_anchor.npz", **saved)
    print("saved", output / f"fold{FOLD}_seed{SEED}_reg_anchor.npz")


if __name__ == "__main__":
    main()
