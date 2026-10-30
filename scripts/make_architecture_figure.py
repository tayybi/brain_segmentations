#!/usr/bin/env python3
"""Architecture/pipeline figure for the ISBI paper (PNG at 300 dpi + PDF) in
results/figures/fig_architecture. Thumbnails come from training mouse 2c
(results/label_prop_examples/fold0_seed1.npz and results/sam2_pseudo_quartile.npz)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
EXAMPLES = ROOT / "results" / "label_prop_examples" / "fold0_seed1.npz"
SAM2 = ROOT / "results" / "sam2_pseudo_one_mid.npz"

INK, MUTED, LINE, FILL = "#0b0b0b", "#52514e", "#b9b7b1", "#f4f3f0"
ANCHOR, PROP, NET, WARN = "#1baf7a", "#2a78d6", "#6250d6", "#c2410c"
ANIMAL, TEST, ANCHORS, SHOWN = "2c", "1c", [5], [0, 3, 10]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "savefig.dpi": 300, "savefig.bbox": "tight"})


def box(ax, x, y, w, h, fc=FILL, ec=LINE, lw=0.7, ls="-", r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(ax, start, end, color=MUTED, ls="-", lw=0.9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def thumb(ax, image, x, y, size, contour=None, color=PROP, frame=None):
    crop = image[24:232, 24:232]
    ax.imshow(crop, cmap="gray", vmin=0, vmax=1, extent=(x, x + size, y, y + size), zorder=2)
    if contour is not None:
        c = contour[24:232, 24:232]
        ax.contour(np.linspace(x, x + size, c.shape[1]), np.linspace(y + size, y, c.shape[0]), c, levels=[0.5],
                   colors=[color], linewidths=0.8, zorder=3)
    ax.add_patch(Rectangle((x, y), size, size, fc="none", ec=frame or "white", lw=1.0 if frame else 0.4, zorder=4))


def slice_strip(ax, x, y, n, colors, cell=1.55, gap=0.3):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (cell + gap), y), cell, cell, fc=colors[i], ec="none", zorder=3))
    return x + n * (cell + gap) - gap


def label(ax, x, y, text, size=6.5, color=INK, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, text, fontsize=size, color=color, weight=weight, ha=ha, va=va, zorder=5, **kw)


def main():
    data, sam2 = np.load(EXAMPLES), np.load(SAM2)
    images, truth, probs = data[f"{ANIMAL}_image"], data[f"{ANIMAL}_truth"], sam2[ANIMAL]
    n = len(images)

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 55.7); ax.set_aspect("equal"); ax.axis("off")
    title = dict(size=6.8, weight="bold", ha="left")

    # (a) Training labels -------------------------------------------------------------
    box(ax, 0, 29.0, 100, 26.7, fc="none")
    label(ax, 1.2, 53.6, "(a) Labelling the training set from 1 annotated slice per mouse (up to 3 compared)", **title)

    size = 7.4
    for i in ANCHORS:
        thumb(ax, images[i], 12.2 - size / 2, 38.6, size, truth[i], ANCHOR, frame=ANCHOR)
        label(ax, 12.2, 37.3, f"slice {i} (middle)", size=5.3, color=MUTED)
    label(ax, 12.2, 48.4, "annotated slice", size=5.8, color=ANCHOR)
    end = slice_strip(ax, 3.0, 32.6, n, [ANCHOR if i in ANCHORS else "#d9d8d4" for i in range(n)], cell=1.4)
    label(ax, 12.2, 30.6, "1 of 11 slices outlined", size=5.3, color=MUTED)

    arrow(ax, (25.2, 42.3), (28.6, 42.3))
    box(ax, 28.8, 35.6, 22.4, 13.4, fc="white", ec=PROP, lw=1.0)
    label(ax, 40.0, 45.6, "SAM2.1 (frozen)", size=6.4, weight="bold", color=PROP)
    label(ax, 40.0, 40.6, "slice stack as a video,\nprompted with the mask,\nforward + backward", size=5.5, linespacing=1.35)
    label(ax, 40.0, 32.0, "also compared: copy, shape interpolation,\nself-training, registration", size=5.0, color=MUTED, linespacing=1.3)

    arrow(ax, (51.4, 42.3), (54.8, 42.3))
    for k, i in enumerate(SHOWN):
        x = 55.0 + k * (size + 0.7)
        thumb(ax, images[i], x, 38.6, size, probs[i], PROP, frame=ANCHOR if i in ANCHORS else None)
        label(ax, x + size / 2, 37.3, f"slice {i}", size=5.3, color=MUTED)
    label(ax, 65.7, 48.4, "propagated labels", size=5.8, color=PROP)
    slice_strip(ax, 56.5, 32.6, n, [ANCHOR if i in ANCHORS else PROP for i in range(n)], cell=1.4)
    label(ax, 65.7, 30.6, "all 11 slices labelled", size=5.3, color=MUTED)

    # Optional agreement filter (dashed): analysed, not part of the final SAM2 model.
    arrow(ax, (79.4, 42.3), (82.3, 42.3), color=WARN, ls=(0, (2, 1.5)))
    box(ax, 82.5, 33.0, 16.3, 18.6, fc="white", ec=WARN, ls=(0, (3, 2)), lw=0.8)
    label(ax, 90.65, 48.6, "optional:\nagreement filter", size=5.8, weight="bold", color=WARN, linespacing=1.2)
    label(ax, 90.65, 41.6, "ignore pixels where\nthe label and an\nanchor-only network\ndisagree", size=5.0, linespacing=1.25)
    label(ax, 90.65, 35.4, "helps registration,\nhurts SAM2", size=5.0, color=WARN, linespacing=1.2)

    # (b) Network ---------------------------------------------------------------------
    box(ax, 0, 0, 64.0, 26.4, fc="none")
    label(ax, 1.2, 24.3, "(b) 2.5D CNN–Transformer, trained with BCE + Dice", **title)
    arrow(ax, (40.0, 29.0), (40.0, 26.5))
    label(ax, 40.8, 27.7, "anchors + propagated labels", size=5.0, color=MUTED, ha="left")

    yc = 11.6
    for k, i in enumerate([4, 5, 6]):
        thumb(ax, images[i], 1.5 + k * 1.6, yc - 1.2 - k * 1.6, 6.0)
    label(ax, 6.2, 3.0, "slices $i{-}1$, $i$, $i{+}1$", size=5.0, color=MUTED)

    def block(x, h, w, name, sub, color, h2=None):
        h2 = h if h2 is None else h2
        pts = [(x, yc - h / 2), (x + w, yc - h2 / 2), (x + w, yc + h2 / 2), (x, yc + h / 2)]
        ax.add_patch(Polygon(pts, fc=color, ec="none", alpha=0.16, zorder=2))
        ax.add_patch(Polygon(pts, fc="none", ec=color, lw=0.8, zorder=3))
        label(ax, x + w / 2, yc + (1.0 if sub else 0), name, size=5.8, weight="bold", linespacing=1.1)
        label(ax, x + w / 2, yc - 1.6, sub, size=4.9, color=MUTED, linespacing=1.2)

    arrow(ax, (11.4, yc), (13.0, yc))
    block(13.1, 14.0, 7.4, "stem", "64@64²", NET, 10.0)
    arrow(ax, (20.6, yc), (22.2, yc))
    block(22.3, 10.0, 7.4, "down", "256@16²", NET, 7.0)
    arrow(ax, (29.8, yc), (31.4, yc))
    block(31.5, 7.6, 10.6, "Trans-\nformer", "", PROP)
    label(ax, 36.8, yc - 5.6, "4 layers, 16×16 tokens", size=4.9, color=MUTED)
    arrow(ax, (42.2, yc), (43.8, yc))
    block(43.9, 7.0, 7.4, "decoder", "32@64²", NET, 10.0)
    ax.add_patch(FancyArrowPatch((16.8, yc + 7.1), (47.6, yc + 5.1), connectionstyle="arc3,rad=-0.18", arrowstyle="-|>",
                                 mutation_scale=6, color=MUTED, lw=0.7, ls=(0, (2, 1.5))))
    label(ax, 32.2, 21.3, "skip connection", size=5.0, color=MUTED)
    arrow(ax, (51.4, yc), (53.0, yc))
    thumb(ax, images[5], 53.2, yc - 4.0, 8.0, probs[5], PROP)
    label(ax, 57.2, 3.0, "mask of slice $i$", size=5.0, color=MUTED)

    # (c) Inference -------------------------------------------------------------------
    box(ax, 66.0, 0, 34.0, 26.4, fc="none")
    label(ax, 67.2, 24.3, "(c) Inference on a new mouse", **title)
    label(ax, 67.2, 20.9, "no annotation, no SAM2", size=5.3, color=MUTED, ha="left")
    test_images, test_pred = data[f"{TEST}_image"], data[f"{TEST}_prediction_reg_teacher"]
    for k in range(3):
        thumb(ax, test_images[4 + k], 67.6 + k * 1.4, yc - 1.4 - k * 1.4, 6.6)
    label(ax, 72.0, 3.0, "all slices", size=5.0, color=MUTED)
    arrow(ax, (77.2, yc), (78.8, yc))
    box(ax, 79.0, yc - 5.0, 8.6, 10.0, fc="white", ec=NET, lw=0.9)
    label(ax, 83.3, yc + 2.3, "network\n(b)", size=5.6, weight="bold", linespacing=1.2)
    label(ax, 83.3, yc - 2.6, "+ flip TTA\n+ largest\ncomponent", size=4.7, color=MUTED, linespacing=1.15)
    arrow(ax, (87.8, yc), (89.4, yc))
    thumb(ax, test_images[5], 89.6, yc - 4.0, 8.0, test_pred[5], NET)
    label(ax, 93.6, 3.0, "brain mask", size=5.0, color=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig_architecture.{suffix}")
    print("saved", OUT / "fig_architecture.pdf")


if __name__ == "__main__":
    main()
