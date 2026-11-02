#!/usr/bin/env python3
"""Fig. 1 of the ISBI paper: simplified workflow in four parts, drawn at print size
(full text width, 7.0 in) so that every label stays legible.

(a) one annotated slice per mouse, SAM2 propagates it to the other slices;
(b) a segmentation network is trained on those labels;
(c) it segments new mice without annotation or SAM2;
(d) it adapts to a new protocol from one annotated slice per new mouse.

Colours: green = manual annotation, blue = propagated labels, purple = network.
Inputs: results/label_prop_examples/fold0_seed1.npz (images, masks),
results/sam2_pseudo_one_mid.npz (SAM2 labels) and
results/label_prop_examples/fold0_seed1_sam2_one_mid.npz (test predictions of the
one-slice + SAM2 network, scripts/make_sam2_one_slice_example.py).
Panel (d) numbers: experiments/cohort_adaptation/summary.txt (B_to_A, k = 2).
Writes paper_isbi2027/figures/fig_workflow.{png,pdf}."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_isbi2027" / "figures"
EXAMPLES = ROOT / "results" / "label_prop_examples" / "fold0_seed1.npz"
PREDICTIONS = ROOT / "results" / "label_prop_examples" / "fold0_seed1_sam2_one_mid.npz"
SAM2 = ROOT / "results" / "sam2_pseudo_one_mid.npz"

INK, MUTED, LINE = "#0b0b0b", "#52514e", "#b9b7b1"
ANNOTATION, PROPAGATED, NETWORK = "#1baf7a", "#2a78d6", "#6250d6"
TRAIN_MOUSE, TEST_MOUSE, NEW_MOUSE = "2c", "1c", "7c"  # 7c: 256x256 cohort, the target of the B_to_A example
SHOWN = [2, 4, 8]

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 600, "savefig.bbox": "tight"})


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="none", ec=LINE, lw=0.7))
    ax.text(x + 1.2, y + h - 1.9, title, fontsize=7.2, weight="bold", color=INK, ha="left", va="center")


def box(ax, x, y, w, h, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=color, lw=1.1, zorder=3))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=1.0, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=6.3, color=INK, weight="normal", ha="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va="center", zorder=5, **kw)


def thumb(ax, image, x, y, size, mask=None, color=PROPAGATED, frame=None):
    crop = image[24:232, 24:232]
    ax.imshow(crop, cmap="gray", vmin=0, vmax=1, extent=(x, x + size, y, y + size), zorder=2)
    if mask is not None:
        m = mask[24:232, 24:232]
        ax.contour(np.linspace(x, x + size, m.shape[1]), np.linspace(y + size, y, m.shape[0]), m, levels=[0.5],
                   colors=[color], linewidths=1.0, zorder=3)
    ax.add_patch(Rectangle((x, y), size, size, fc="none", ec=frame or "white", lw=1.4 if frame else 0.5, zorder=4))


def main():
    data, sam2 = np.load(EXAMPLES), np.load(SAM2)
    images, truth = data[f"{TRAIN_MOUSE}_image"], data[f"{TRAIN_MOUSE}_truth"]
    labels = sam2[TRAIN_MOUSE]
    middle = len(images) // 2

    fig, ax = plt.subplots(figsize=(7.0, 3.71))
    ax.set_xlim(0, 100); ax.set_ylim(0, 53); ax.set_aspect("equal"); ax.axis("off")

    # (a) The main pathway: one annotated slice, SAM2 labels the rest ------------------
    panel(ax, 0, 27.5, 100, 25.5, "(a) Label the training set from one annotated slice per mouse")
    for k in range(4):
        thumb(ax, images[middle - 3 + 2 * k], 2.0 + k * 1.4, 37.0 - k * 1.4, 12.0)
    text(ax, 10.1, 30.6, "mouse MRI volume\n10–11 slices", size=6.0, color=MUTED, linespacing=1.25)
    arrow(ax, (20.0, 38.5), (23.0, 38.5))

    thumb(ax, images[middle], 23.4, 31.2, 14.6, truth[middle], ANNOTATION, frame=ANNOTATION)
    text(ax, 30.7, 47.8, "1 manually annotated slice", size=6.5, color=ANNOTATION, weight="bold")
    arrow(ax, (38.6, 38.5), (41.6, 38.5))

    box(ax, 42.0, 33.0, 17.0, 11.0, PROPAGATED)
    text(ax, 50.5, 41.2, "SAM2.1", size=8.0, color=PROPAGATED, weight="bold")
    text(ax, 50.5, 36.2, "propagates the\nmask to the\nother slices", size=6.0, linespacing=1.2)
    arrow(ax, (59.6, 38.5), (62.6, 38.5))

    for k, i in enumerate(SHOWN):
        x = 63.0 + k * 12.2
        thumb(ax, images[i], x, 33.0, 11.6, labels[i], PROPAGATED)
        text(ax, x + 5.8, 31.3, f"slice {i}", size=5.8, color=MUTED)
    text(ax, 80.9, 47.8, "SAM2-propagated masks", size=6.5, color=PROPAGATED, weight="bold")
    arrow(ax, (14.0, 27.4), (14.0, 25.6))

    # (b) Train the segmentation network on annotated + propagated labels ----------------
    panel(ax, 0, 0, 28.0, 25.5, "(b) Train a network")
    for k, i in enumerate([middle - 1, middle, middle + 1]):  # only the front slice shows its label
        thumb(ax, images[i], 1.5 + k * 1.3, 8.6 - k * 1.3, 10.0, labels[i] if k == 2 else None, PROPAGATED)
    text(ax, 7.8, 3.3, "slices + labels", size=6.0, color=MUTED)
    arrow(ax, (14.2, 12.0), (15.6, 12.0))
    box(ax, 15.8, 7.5, 11.4, 9.0, NETWORK)
    text(ax, 21.5, 12.0, "train\nnetwork", size=6.3, color=NETWORK, weight="bold", linespacing=1.2)
    text(ax, 21.5, 4.0, "2.5D CNN–\nTransformer\nor nnU-Net", size=5.4, color=MUTED, linespacing=1.15)

    # (c) Segment new mice without annotation or SAM2 ------------------------------------
    panel(ax, 29.5, 0, 33.5, 25.5, "(c) Segment new mice")
    test_image = data[f"{TEST_MOUSE}_image"]
    test_middle = len(test_image) // 2
    prediction = np.load(PREDICTIONS)[f"{TEST_MOUSE}_prediction_sam2"]
    thumb(ax, test_image[test_middle], 31.0, 7.0, 9.6)
    text(ax, 35.8, 4.0, "unseen\nmouse MRI", size=6.0, color=MUTED, linespacing=1.15)
    arrow(ax, (41.0, 11.8), (42.6, 11.8))
    box(ax, 42.8, 8.3, 7.8, 7.0, NETWORK)
    text(ax, 46.7, 11.8, "trained\nnetwork", size=5.6, color=NETWORK, weight="bold", linespacing=1.15)
    text(ax, 46.7, 4.4, "no labels,\nno SAM2", size=5.6, color=MUTED, linespacing=1.15)
    arrow(ax, (51.0, 11.8), (52.6, 11.8))
    thumb(ax, test_image[test_middle], 52.8, 7.0, 9.6, prediction[test_middle], NETWORK)
    text(ax, 57.6, 4.4, "brain mask", size=6.0, color=MUTED)

    # (d) Adapt to a new acquisition protocol -------------------------------------------
    panel(ax, 64.5, 0, 35.5, 25.5, "(d) Adapt to a new protocol")
    new_image, new_truth = data[f"{NEW_MOUSE}_image"], data[f"{NEW_MOUSE}_truth"]
    new_middle = len(new_image) // 2
    thumb(ax, new_image[new_middle], 67.3, 9.0, 9.6, new_truth[new_middle], ANNOTATION, frame=ANNOTATION)
    text(ax, 72.1, 6.0, "1 slice per\nnew mouse", size=5.8, color=ANNOTATION, weight="bold", linespacing=1.15)
    text(ax, 72.1, 2.6, "+ SAM2 labels", size=5.8, color=PROPAGATED, weight="bold")
    arrow(ax, (77.3, 13.8), (79.3, 13.8))
    box(ax, 79.5, 10.3, 8.0, 7.0, NETWORK)
    text(ax, 83.5, 13.8, "fine-\ntune", size=6.0, color=NETWORK, weight="bold", linespacing=1.15)
    arrow(ax, (87.9, 13.8), (89.7, 13.8))
    text(ax, 94.4, 19.6, "harder shift", size=5.0, color=MUTED)
    text(ax, 94.4, 17.6, "(200 → 256 px)", size=5.0, color=MUTED)
    text(ax, 94.4, 14.6, "0.806", size=7.0, weight="bold")
    text(ax, 94.4, 11.7, "→ 0.928", size=7.0, weight="bold")
    text(ax, 94.4, 8.2, "Dice with 2\nadaptation mice", size=5.0, color=MUTED, linespacing=1.15)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig_workflow.{suffix}")
    print("saved", OUT / "fig_workflow.pdf")


if __name__ == "__main__":
    main()
