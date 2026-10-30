#!/usr/bin/env python3
"""nnU-Net (2D) baseline on the same folds and label budgets as the paper.

Conditions, all trained on the 14 training mice of a fold:
  full       every slice annotated
  sam2_1     middle slice annotated + SAM2 labels on the other slices
  sparse_1   middle slice annotated, other slices set to nnU-Net's ignore label

Run with the nnU-Net environment:
  .venv_nnunet/bin/python scripts/nnunet_baseline.py prepare
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train <id> 2d all -tr nnUNetTrainer_250epochs   (see launch_nnunet.sh)
  .venv_nnunet/bin/python scripts/nnunet_baseline.py score
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "nnunet"
os.environ.setdefault("nnUNet_raw", str(BASE / "raw"))
os.environ.setdefault("nnUNet_preprocessed", str(BASE / "preprocessed"))
os.environ.setdefault("nnUNet_results", str(BASE / "results"))
sys.path.insert(0, str(ROOT))

import nibabel as nib
from run_label_propagation_study import anchor_slices
from run_meta_ssl_study import DEFAULT_DATA, cohort, largest_component, load_volumes, make_split, score

CONDITIONS = {"full": 0, "sam2_1": 1, "sparse_1": 2}
SPACING = (1.0, 14 / 256, 14 / 256)  # slice thickness, in-plane after resampling to 256


def dataset_id(fold, condition):
    return 600 + 10 * CONDITIONS[condition] + fold


def dataset_name(fold, condition):
    return f"Dataset{dataset_id(fold, condition)}_mouse_{condition}_f{fold}"


def save(array, path):
    affine = np.diag([SPACING[2], SPACING[1], SPACING[0], 1.0])
    # nnU-Net reads NIfTI as (x, y, z); store slices along z.
    nib.save(nib.Nifti1Image(np.ascontiguousarray(array.transpose(2, 1, 0)), affine), path)


def prepare():
    volumes = load_volumes(DEFAULT_DATA, 256, "cpu")
    sam2 = np.load(ROOT / "results" / "sam2_pseudo_one_mid.npz")
    for fold in range(5):
        train, _, test = make_split(fold)
        for condition in CONDITIONS:
            folder = BASE / "raw" / dataset_name(fold, condition)
            (folder / "imagesTr").mkdir(parents=True, exist_ok=True)
            (folder / "labelsTr").mkdir(parents=True, exist_ok=True)
            (folder / "imagesTs").mkdir(parents=True, exist_ok=True)
            for animal in train:
                images, masks = (t.numpy() for t in volumes[animal])
                label = (masks > 0.5).astype(np.uint8)
                anchors = anchor_slices(len(images), "one_mid")
                others = [i for i in range(len(images)) if i not in anchors]
                if condition == "sam2_1":
                    label[others] = (sam2[animal][others] > 0.5).astype(np.uint8)
                elif condition == "sparse_1":
                    label[others] = 2
                save(images.astype(np.float32), folder / "imagesTr" / f"{animal}_0000.nii.gz")
                save(label, folder / "labelsTr" / f"{animal}.nii.gz")
            for animal in test:
                save(volumes[animal][0].numpy().astype(np.float32), folder / "imagesTs" / f"{animal}_0000.nii.gz")
            labels = {"background": 0, "brain": 1} | ({"ignore": 2} if condition == "sparse_1" else {})
            (folder / "dataset.json").write_text(json.dumps({
                "channel_names": {"0": "MRI"}, "labels": labels, "numTraining": len(train), "file_ending": ".nii.gz"}, indent=2))
    print("prepared", sorted(p.name for p in (BASE / "raw").iterdir()))


def score_all():
    volumes = load_volumes(DEFAULT_DATA, 256, "cpu")
    rows = []
    for fold in range(5):
        _, _, test = make_split(fold)
        for condition in CONDITIONS:
            predictions = BASE / "predictions" / dataset_name(fold, condition)
            for animal in test:
                path = predictions / f"{animal}.nii.gz"
                if not path.exists():
                    continue
                pred = np.asarray(nib.load(path).dataobj).transpose(2, 1, 0) == 1
                # Same post-processing as the paper's network: largest component, holes filled.
                pred = np.stack([largest_component(p) for p in pred])
                truth = volumes[animal][1].numpy() > 0.5
                dice, iou, hd = score(pred, truth, list(range(len(truth))))
                rows.append({"fold": fold, "condition": condition, "animal": animal, "cohort": cohort(animal), "dice": dice, "iou": iou, "hd95": hd})
    out = ROOT / "experiments" / "nnunet_baseline"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"{len(rows)} scores written to {out / 'metrics.csv'}")


if __name__ == "__main__":
    {"prepare": prepare, "score": score_all}[sys.argv[1]]()
