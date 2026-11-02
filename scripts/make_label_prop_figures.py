#!/usr/bin/env python3
"""Paper figures for the label-propagation study (PNG at 300 dpi + PDF) in
results/figures/. Quantitative figures read experiments/label_prop; the
qualitative figure needs results/label_prop_examples/fold0_seed1.npz."""

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "label_prop"
OUT = ROOT / "results" / "figures"
EXAMPLES = ROOT / "results" / "label_prop_examples" / "fold0_seed1.npz"
SAM2 = ROOT / "results" / "sam2_pseudo_quartile.npz"

INK, MUTED, GRID, OURS, BASE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#9a9893"
SERIES = {"copy": ("Copy nearest", "#eb6834"), "sdf": ("Shape interpolation", "#eda100"), "self_train": ("Self-training", "#1baf7a"),
          "reg_anchor": ("Registration + anchor loss", "#6250d6"), "reg_teacher": ("Ours", OURS)}
NAMES = {"sparse": "Anchors only", "copy": "+ copy nearest", "sdf": "+ shape interpolation", "self_train": "+ self-training",
         "reg_unsup": "+ registration", "reg_anchor": "+ registration, anchor loss", "reg_teacher": "+ reg.-guided self-training (ours)",
         "full": "All slices labelled"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 600, "savefig.bbox": "tight"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}")
    plt.close(fig)


def bootstrap_ci(values, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n, len(values))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def method_comparison(per_animal):
    order = ["sparse", "copy", "sdf", "self_train", "reg_unsup", "reg_anchor", "reg_teacher", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, metric, label in ((axes[0], "dice", "Dice (higher is better)"), (axes[1], "hd95", "HD95, px (lower is better)")):
        for y, method in enumerate(order[::-1]):
            values = per_animal[per_animal.method == method][metric].to_numpy()
            low, high = bootstrap_ci(values)
            color = OURS if method == "reg_teacher" else INK if method in ("sparse", "full") else BASE
            ax.plot([low, high], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(values.mean(), y, "o", ms=6, color=color, mec="white", mew=1.2, zorder=3)
            ax.annotate(f"{values.mean():.3f}" if metric == "dice" else f"{values.mean():.1f}", (high, y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color=MUTED,
                        bbox={"boxstyle": "square,pad=0.1", "fc": "white", "lw": 0})
        for method, style in (("sparse", ":"), ("full", "--")):
            ax.axvline(per_animal[per_animal.method == method][metric].mean(), color=MUTED, lw=0.7, ls=style, zorder=0)
        ax.set_xlabel(label)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)), [NAMES[m] for m in order[::-1]])
    axes[0].get_yticklabels()[order[::-1].index("reg_teacher")].set_color(OURS)
    axes[0].set_xlim(0.895, 0.972)
    axes[1].set_xlim(7, 17.5)
    fig.suptitle("Test performance with 3 annotated slices per training mouse (20 mice, mean and 95% CI)", fontsize=8.5, color=INK, x=0.02, ha="left")
    save(fig, "fig_method_comparison")


def pseudo_label_quality(quality):
    table = quality.groupby(["method", "distance_to_anchor"]).pseudo_dice.mean().unstack()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    label_y = {}
    for method in sorted(SERIES, key=lambda m: table.loc[m, 2]):
        label_y[method] = max(table.loc[method, 2], max(label_y.values(), default=-1) + 0.014)
    for method, (label, color) in SERIES.items():
        near, edge = table.loc[method, 1], table.loc[method, 2]
        ax.plot([0, 1], [near, edge], "-o", color=color, lw=2 if method == "reg_teacher" else 1.4, ms=5, mec="white", mew=1, zorder=3 if method == "reg_teacher" else 2)
        ax.annotate(f"{label}  {edge:.3f}", (1, edge), xytext=(1.06, label_y[method]), textcoords="data", va="center", fontsize=7,
                    color=OURS if method == "reg_teacher" else INK)
    ax.set_xticks([0, 1], ["1 slice\n(between anchors)", "2 slices\n(ends of the brain)"])
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylabel("Pseudo-label Dice vs. hidden truth")
    ax.set_xlabel("Distance to nearest annotated slice")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig_pseudo_label_quality")


def per_animal_gain(per_animal):
    dice = per_animal.pivot(index="animal", columns="method", values="dice")
    dice = dice.sort_values("sparse")
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for y, (animal, row) in enumerate(dice.iterrows()):
        gained = row["reg_teacher"] >= row["sparse"]
        ax.plot([row["sparse"], row["reg_teacher"]], [y, y], color=OURS if gained else "#e34948", lw=1.6, solid_capstyle="round")
        ax.plot(row["sparse"], y, "o", color=BASE, ms=4.5, mec="white", mew=0.8)
        ax.plot(row["reg_teacher"], y, "o", color=OURS if gained else "#e34948", ms=4.5, mec="white", mew=0.8)
        ax.plot(row["full"], y, "|", color=INK, ms=7, mew=1.2)
    ax.set_yticks(range(len(dice)), dice.index)
    ax.set_xlabel("Dice")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    improved = int((dice["reg_teacher"] > dice["sparse"]).sum())
    ax.set_title(f"Per mouse: anchors only (grey) → ours (blue); | = all labelled\nOurs improves {improved} of {len(dice)} mice", fontsize=7.5, color=INK, loc="left")
    save(fig, "fig_per_animal_gain")


def qualitative():
    """End slices of three training mice in fold 0, beyond the outermost anchor, with the
    same crop in every column: two where SAM2 succeeds and the others fail (one per
    cohort) and one where SAM2 fails. Interpolation, registration (anchor loss, no
    agreement filter; scripts/make_registration_example.py) and SAM2 use three anchors;
    the last column is SAM2 from the middle slice only. Also copied to
    paper_isbi2027/figures."""
    if not EXAMPLES.exists():
        print(f"skip qualitative figure: {EXAMPLES} not found")
        return
    data, sam2_3, sam2_1 = np.load(EXAMPLES), np.load(SAM2), np.load(SAM2.with_name("sam2_pseudo_one_mid.npz"))
    registration = np.load(EXAMPLES.with_name("fold0_seed1_reg_anchor.npz"))
    # (mouse, slice, row label): Mouse 1 = 10c (256x256 cohort), Mouse 2 = 30c and Mouse 3 = 21c (200x200 cohort).
    rows = [("10c", 0, "Mouse 1\n(256²)"), ("30c", -1, "Mouse 2\n(200²)"), ("21c", -1, "Mouse 3\n(200²)")]
    columns = [("sdf", "Shape\ninterp."), ("reg_anchor", "Regis-\ntration"), ("sam2_3", "SAM2\n3 slices"), ("sam2_1", "SAM2\n1 slice")]
    truth_colour, fp, fn = "#1baf7a", [0.89, 0.15, 0.15, 0.6], [0.98, 0.70, 0.0, 0.7]
    fig, axes = plt.subplots(len(rows), len(columns) + 1, figsize=(3.1, 0.565 * len(rows) + 0.45),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.05})
    for r, (animal, index, note) in enumerate(rows):
        images, truth = data[f"{animal}_image"], data[f"{animal}_truth"] > 0.5
        i = index % len(images)
        maps = {"sdf": data[f"{animal}_pseudo_sdf"][i] > 0.5, "reg_anchor": registration[f"{animal}_pseudo_reg_anchor"][i] > 0.5,
                "sam2_3": sam2_3[animal][i] > 0.5, "sam2_1": sam2_1[animal][i] > 0.5}
        # One square crop per row around the true brain and every label, so leaks stay visible.
        union = truth[i] | np.any(list(maps.values()), 0)
        ys, xs = np.nonzero(union)
        cy, cx, half = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2, max(np.ptp(ys), np.ptp(xs)) // 2 + 12
        y0, x0 = np.clip([cy - half, cx - half], 0, 256 - 2 * half)
        crop = lambda a: a[y0:y0 + 2 * half, x0:x0 + 2 * half]
        axes[r, 0].imshow(crop(images[i]), cmap="gray", vmin=0, vmax=1)
        axes[r, 0].contour(crop(truth[i]), levels=[0.5], colors=[truth_colour], linewidths=1.1)
        axes[r, 0].set_ylabel(note, fontsize=5.6, color=INK, labelpad=2, linespacing=1.15)
        for c, (method, title) in enumerate(columns, start=1):
            label = maps[method]
            dice = 2 * (label & truth[i]).sum() / max(label.sum() + truth[i].sum(), 1)
            overlay = np.zeros((*label.shape, 4))
            overlay[label & ~truth[i]] = fp
            overlay[~label & truth[i]] = fn
            axes[r, c].imshow(crop(images[i]), cmap="gray", vmin=0, vmax=1)
            axes[r, c].imshow(crop(overlay))
            axes[r, c].contour(crop(truth[i]), levels=[0.5], colors=[truth_colour], linewidths=0.8)
            axes[r, c].text(0.05, 0.94, f"{dice:.2f}", transform=axes[r, c].transAxes, fontsize=6.5, color="white",
                            weight="bold", va="top", bbox={"boxstyle": "round,pad=0.18", "fc": "black", "alpha": 0.65, "lw": 0})
            if r == 0:
                axes[r, c].set_title(title, fontsize=6.0, linespacing=1.1, color=OURS if method.startswith("sam2") else INK, pad=2.5)
        if r == 0:
            axes[r, 0].set_title("Ground\ntruth", fontsize=6.0, linespacing=1.1, color=INK, pad=2.5)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [Line2D([], [], color=truth_colour, lw=1.4, label="ground truth"),
               Patch(fc=fp, label="false positive"), Patch(fc=fn, label="false negative")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.6, frameon=False, handlelength=1.1,
               handletextpad=0.4, columnspacing=0.9, bbox_to_anchor=(0.53, -0.01))
    save(fig, "fig_qualitative_pseudo_labels")
    paper = ROOT / "paper_isbi2027" / "figures"
    for suffix in ("png", "pdf"):
        shutil.copy(OUT / f"fig_qualitative_pseudo_labels.{suffix}", paper / f"fig_qualitative_pseudo_labels.{suffix}")


def main():
    per_animal = pd.concat(pd.read_csv(p) for p in RUNS.glob("fold*_seed*/metrics.csv")).groupby(["method", "animal"], as_index=False)[["dice", "hd95"]].mean()
    quality = pd.concat(pd.read_csv(p) for p in RUNS.glob("fold*_seed*/pseudo_quality.csv"))
    method_comparison(per_animal)
    pseudo_label_quality(quality)
    per_animal_gain(per_animal)
    qualitative()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
