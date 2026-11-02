#!/usr/bin/env python3
"""Retrain the one-slice + SAM2 network of fold 0, seed 1 and save its test-mouse
predictions for the workflow figure (results/label_prop_examples/fold0_seed1_sam2_one_mid.npz)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_label_propagation_study import (DEFAULT_DATA, ROOT, RunConfig, anchor_slices, build_targets, load_volumes,
                                         make_split, predict, score, set_seed, train_segmentation)

FOLD, SEED = 0, 1


def main():
    output = ROOT / "results" / "label_prop_examples"
    cfg = RunConfig(data_root=str(DEFAULT_DATA), output_dir=str(output), fold=FOLD, seed=SEED, anchor_layout="one_mid")
    device = torch.device("cuda")
    train, val, test = make_split(FOLD)
    volumes = load_volumes(DEFAULT_DATA, cfg.image_size, device)
    anchors = {a: anchor_slices(volumes[a][0].shape[0], "one_mid") for a in train}
    stored = np.load(ROOT / "results" / "sam2_pseudo_one_mid.npz")
    sam2 = {a: torch.from_numpy(stored[a]).to(device) for a in train}

    set_seed(SEED * 100 + 1)  # same offset as "sam2" after "sparse" in the one_mid sweep
    targets, _ = build_targets("sam2", volumes, train, anchors, {}, None, sam2)
    model, best = train_segmentation(volumes, targets, val, cfg, lambda m: print("sam2", m))
    saved = {}
    for animal in test:
        images, masks = volumes[animal]
        saved[f"{animal}_prediction_sam2"] = prediction = predict(model, images)
        dice, _, _ = score(prediction, masks.cpu().numpy() > 0.5, list(range(images.shape[0])))
        print(f"{animal}: dice={dice:.4f}")
    np.savez_compressed(output / f"fold{FOLD}_seed{SEED}_sam2_one_mid.npz", **saved)
    print("saved", output / f"fold{FOLD}_seed{SEED}_sam2_one_mid.npz")


if __name__ == "__main__":
    main()
