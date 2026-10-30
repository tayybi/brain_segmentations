#!/usr/bin/env python3
"""Fig. 3: (a) test Dice against annotation budget, (b) non-inferiority forest plot.

Reads per-mouse metrics from experiments/ and writes paper_isbi2027/figures/fig_results.{pdf,png}.
"""

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "paper_isbi2027" / "figures"
MARGIN = 0.01
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8a85", "#30302e"


def per_mouse(folder, method):
    frames = [pd.read_csv(f) for f in glob.glob(str(EXP / folder / "*" / "metrics.csv"))]
    df = pd.concat(frames)
    return df[df.method == method].groupby("animal").dice.mean()


def nnunet(condition):
    df = pd.read_csv(EXP / "nnunet_baseline" / "metrics.csv")
    return df[df.condition == condition].groupby("animal").dice.mean()


def bootstrap_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    values = diff.to_numpy()
    means = rng.choice(values, (n, len(values))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    full = per_mouse("label_prop", "full")
    budget = {  # folder per number of annotated slices
        1: "label_prop_sam2_one_mid", 2: "label_prop_sam2_two_third", 3: "label_prop_sam2_quartile"}
    percent = {1: 9, 2: 18, 3: 28}
    alone = {k: per_mouse(f, "sparse").mean() for k, f in budget.items()}
    alone[3] = per_mouse("label_prop", "sparse").mean()  # three-slice anchors-only from the main study
    sam2 = {k: per_mouse(f, "sam2") for k, f in budget.items()}

    rows = [(f"{k} slice{'s' if k > 1 else ''} + SAM2", sam2[k] - full) for k in (1, 2, 3)]
    rows.append(("nnU-Net, 1 slice + SAM2", nnunet("sam2_1") - nnunet("full")))
    for split, label in [("A_to_B", "256→200, 1 slice + SAM2"), ("B_to_A", "200→256, 1 slice + SAM2")]:
        frames = pd.concat(pd.read_csv(f) for f in glob.glob(str(EXP / "cross_cohort" / f"{split}_seed*" / "metrics.csv")))
        means = frames.groupby(["method", "animal"]).dice.mean()
        rows.append((label, means["sam2"] - means["full"]))

    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": GREY,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.9), gridspec_kw={"height_ratios": [1, 1.05]})

    # (a) budget curve
    xs = [percent[k] for k in (1, 2, 3)] + [100]
    full_mean = full.mean()
    ax1.axhspan(full_mean - MARGIN, full_mean, color=GREY, alpha=0.15, lw=0)
    ax1.axhline(full_mean, color=GREY, lw=0.8, ls="--")
    ax1.plot(xs, [alone[k] for k in (1, 2, 3)] + [full_mean], color=ORANGE, lw=1.5, marker="o", ms=4,
             label="Annotated slices only")
    ax1.plot(xs, [sam2[k].mean() for k in (1, 2, 3)] + [full_mean], color=BLUE, lw=1.5, marker="o", ms=4,
             label="+ SAM2 labels")
    nn = {"sparse_1": nnunet("sparse_1").mean(), "sam2_1": nnunet("sam2_1").mean(), "full": nnunet("full").mean()}
    ax1.plot([9], [nn["sparse_1"]], ls="none", marker="s", ms=4, mfc="white", mec=ORANGE, mew=1)
    ax1.plot([9], [nn["sam2_1"]], ls="none", marker="s", ms=4, mfc="white", mec=BLUE, mew=1)
    ax1.plot([100], [nn["full"]], ls="none", marker="s", ms=4, mfc="white", mec=INK, mew=1,
             label="nnU-Net (open squares)")
    ax1.set_xscale("log")
    ax1.set_xticks(xs, ["1 slice\n9%", "2\n18%", "3\n28%", "All\n100%"])
    ax1.minorticks_off()
    ax1.set_ylim(0.86, 0.97)
    ax1.set_ylabel("Test Dice")
    ax1.text(11, full_mean - MARGIN + 0.002, "margin: full − 0.01", color=GREY, fontsize=6)
    ax1.legend(frameon=False, loc="lower right", fontsize=6)
    ax1.set_title("(a) Annotation budget", loc="left", fontsize=7.5)

    # (b) forest plot
    ax2.axvspan(-MARGIN, 0.03, color=GREY, alpha=0.15, lw=0)
    ax2.axvline(0, color=GREY, lw=0.8)
    ax2.axvline(-MARGIN, color=GREY, lw=0.8, ls="--")
    for i, (label, diff) in enumerate(rows):
        y = len(rows) - 1 - i
        low, high = bootstrap_ci(diff)
        shown = low > -MARGIN  # non-inferiority shown: filled marker; not shown: open marker
        ax2.plot([low, high], [y, y], color=BLUE, lw=1.5, solid_capstyle="round")
        ax2.plot(diff.mean(), y, "o", ms=4, mec=BLUE, mew=1.2, mfc=BLUE if shown else "white")
        ax2.text(0.031, y, f"{diff.mean():+.3f}", va="center", fontsize=6, color=INK)
    ax2.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax2.set_xlim(-0.04, 0.03)
    ax2.set_xlabel("Dice difference to full annotation (95% CI)")
    ax2.text(0.0015, len(rows) - 0.5, "non-inferior region", color=GREY, fontsize=6)
    ax2.set_ylim(-0.6, len(rows) - 0.2)
    ax2.set_title("(b) Non-inferiority, margin 0.01", loc="left", fontsize=7.5)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2)
    fig.tight_layout(h_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_results.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    for label, diff in rows:
        print(f"{label:32s} {diff.mean():+.4f} CI {np.round(bootstrap_ci(diff), 4)} n={len(diff)}")


if __name__ == "__main__":
    main()
