#!/usr/bin/env python3
"""SAM2 baseline: treat each mouse's slice stack as a video, prompt SAM2 with
the annotated slices' masks and propagate them to every slice.

Run with the SAM2 environment:
    .venv_sam2/bin/python scripts/sam2_propagate.py --anchor-layout quartile
Writes results/sam2_pseudo_<layout>.npz with one [slices, H, W] probability
array per mouse. Propagation only uses that mouse's own anchors, so the same
file serves every fold."""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_label_propagation_study import anchor_slices
from run_meta_ssl_study import DEFAULT_DATA, ROOT, cohort, load_volumes

from sam2.build_sam import build_sam2_video_predictor


# Layouts with fewer than three anchors, for the annotation-budget headroom test.
# "stagger2" puts the pair at a different depth in each mouse of a cohort, so that
# every slice depth is annotated in some animal.
EXTRA_LAYOUTS = {"one_mid": lambda n, k: [n // 2], "two_third": lambda n, k: [n // 3, 2 * n // 3],
                 "two_spread": lambda n, k: [1, n - 2], "stagger2": lambda n, k: [k % (n - n // 2), k % (n - n // 2) + n // 2]}


def layout_slices(n, layout, k):
    return EXTRA_LAYOUTS[layout](n, k) if layout in EXTRA_LAYOUTS else anchor_slices(n, layout)


@torch.inference_mode()
def propagate(predictor, images, masks, anchors):
    with tempfile.TemporaryDirectory() as frames:
        for i, image in enumerate(images):
            Image.fromarray((image * 255).astype(np.uint8)).convert("RGB").save(Path(frames) / f"{i:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=frames)
        for a in anchors:
            predictor.add_new_mask(state, frame_idx=a, obj_id=1, mask=masks[a] > 0.5)
        sums, counts = np.zeros(images.shape, np.float32), np.zeros(len(images), np.float32)
        for reverse in (False, True):
            for frame, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                sums[frame] += torch.sigmoid(logits[0, 0].float()).cpu().numpy()
                counts[frame] += 1
        predictor.reset_state(state)
    return sums / counts.clip(min=1)[:, None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-layout", default="quartile", choices=["quartile", "spread", *EXTRA_LAYOUTS])
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = parser.parse_args()
    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
    volumes = load_volumes(DEFAULT_DATA, 256, "cpu")
    results = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        animals = sorted(volumes, key=lambda a: int(a[:-1]))
        for animal in animals:
            images, masks = (t.numpy() for t in volumes[animal])
            k = [a for a in animals if cohort(a) == cohort(animal)].index(animal)
            anchors = layout_slices(len(images), args.anchor_layout, k)
            probs = propagate(predictor, images, masks, anchors)
            results[animal] = probs
            results[f"{animal}_anchors"] = np.array(anchors)
            others = [i for i in range(len(images)) if i not in anchors]
            pred, truth = probs[others] > 0.5, masks[others] > 0.5
            print(f"{animal}: pseudo-label dice on unannotated slices = {2 * (pred & truth).sum() / (pred.sum() + truth.sum()):.4f}", flush=True)
    output = ROOT / "results" / f"sam2_pseudo_{args.anchor_layout}.npz"
    np.savez_compressed(output, **results)
    print("saved", output)


if __name__ == "__main__":
    main()
