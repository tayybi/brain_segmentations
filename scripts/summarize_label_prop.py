#!/usr/bin/env python3
"""Summarise experiments/label_prop: per-animal scores averaged over seeds, then
mean +/- SD across the 20 cross-validated test animals, the share of the
sparse-to-full gap each method closes, pseudo-label quality on the training
mice, and paired Wilcoxon tests against reg_teacher."""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "label_prop"
LABELS = {
    "sparse": "Anchors only (3 slices/mouse)",
    "copy": "+ copy nearest anchor",
    "sdf": "+ shape-based interpolation",
    "reg_unsup": "+ registration propagation",
    "reg_anchor": "+ registration w/ anchor loss",
    "self_train": "+ self-training",
    "reg_teacher": "+ registration-guided self-training (ours)",
    "full": "All slices labelled (upper bound)",
}


def main():
    runs = sorted(ROOT.glob("fold*_seed*/metrics.csv"))
    if not runs:
        sys.exit(f"no metrics.csv under {ROOT}")
    data = pd.concat(pd.read_csv(p) for p in runs)
    quality = pd.concat(pd.read_csv(p) for p in ROOT.glob("fold*_seed*/pseudo_quality.csv"))
    per_animal = data.groupby(["method", "animal", "cohort"], as_index=False)[["dice", "iou", "hd95"]].mean()
    dice = per_animal.pivot(index="animal", columns="method", values="dice")
    sparse, full = dice["sparse"].mean(), dice["full"].mean() if "full" in dice else float("nan")
    pseudo = quality.groupby("method").pseudo_dice.mean()
    pseudo_edge = quality[quality.distance_to_anchor >= 2].groupby("method").pseudo_dice.mean()
    complete = data.groupby(["fold", "seed"]).method.nunique().eq(len(LABELS)).sum()

    lines = [f"Runs: {data[['fold', 'seed']].drop_duplicates().shape[0]} (complete: {complete}); test animals: {dice.shape[0]}\n",
             "| Method | Dice (mean ± SD) | Cohort A | Cohort B | HD95 px | Gap closed | Pseudo-label Dice (all / edge) | p vs ours |",
             "|---|---|---:|---:|---:|---:|---|---:|"]
    for method, label in LABELS.items():
        rows = per_animal[per_animal.method == method]
        if rows.empty:
            continue
        cohorts = rows.groupby("cohort").dice.mean()
        gap = (rows.dice.mean() - sparse) / (full - sparse) * 100 if method not in ("sparse", "full") else float("nan")
        common = dice[[method, "reg_teacher"]].dropna() if "reg_teacher" in dice else pd.DataFrame()
        p = "–" if method == "reg_teacher" or len(common) < 6 else f"{wilcoxon(common[method], common['reg_teacher']).pvalue:.3g}"
        pq = f"{pseudo[method]:.3f} / {pseudo_edge.get(method, float('nan')):.3f}" if method in pseudo else "–"
        lines.append(f"| {label} | {rows.dice.mean():.3f} ± {rows.dice.std():.3f} | {cohorts.get('A', float('nan')):.3f} | {cohorts.get('B', float('nan')):.3f} | "
                     f"{rows.hd95.mean():.2f} | {'' if gap != gap else f'{gap:.0f}%'} | {pq} | {p} |")
    text = "\n".join(lines)
    print(text)
    (ROOT / "summary.md").write_text("# Label propagation study results\n\n" + text + "\n")
    per_animal.to_csv(ROOT / "per_animal_mean_over_seeds.csv", index=False)


if __name__ == "__main__":
    main()
