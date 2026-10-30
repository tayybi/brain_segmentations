#!/usr/bin/env python3
"""Summarise experiments/meta_ssl: per-animal scores averaged over seeds, then
mean +/- SD across the 20 cross-validated test animals, with paired Wilcoxon
tests against SSL + Reptile + TTA at the same regime and k."""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "meta_ssl"
LABELS = {
    ("scratch_joint", 0): "Supervised, scratch + FT",
    ("ssl_joint", 0): "SSL + supervised + FT",
    ("ssl_joint", 1): "SSL + supervised + TTA + FT",
    ("scratch_reptile", 0): "Reptile (no SSL) + FT",
    ("ssl_reptile", 0): "SSL + Reptile + FT",
    ("ssl_reptile", 1): "SSL + Reptile + TTA + FT (full)",
}


def main():
    frames = [pd.read_csv(p) for p in sorted(ROOT.glob("*/fold*_seed*/metrics.csv"))]
    if not frames:
        sys.exit(f"no metrics.csv under {ROOT}")
    data = pd.concat(frames)
    runs = data.groupby("regime")[["fold", "seed"]].apply(lambda d: len(d.drop_duplicates())).to_dict()
    per_animal = data.groupby(["regime", "method", "tta", "k", "animal", "cohort"], as_index=False)[["dice", "iou", "hd95"]].mean()
    per_animal["label"] = [LABELS[(m, t)] for m, t in zip(per_animal.method, per_animal.tta)]
    lines = []
    for regime, block in per_animal.groupby("regime"):
        lines.append(f"\n## Regime: {regime} ({runs[regime]} fold/seed runs)\n")
        lines.append("| Method | k | Dice (mean ± SD) | Cohort A | Cohort B | HD95 px | worst animal | p vs full |")
        lines.append("|---|---:|---|---:|---:|---:|---|---:|")
        for k, kblock in block.groupby("k"):
            reference = kblock[kblock.label == LABELS[("ssl_reptile", 1)]].set_index("animal").dice
            for label in LABELS.values():
                rows = kblock[kblock.label == label].set_index("animal")
                if rows.empty:
                    continue
                worst = rows.dice.idxmin()
                common = rows.index.intersection(reference.index)
                p = "–" if label == LABELS[("ssl_reptile", 1)] or len(common) < 6 else f"{wilcoxon(rows.dice[common], reference[common]).pvalue:.3g}"
                cohort = rows.groupby("cohort").dice.mean()
                lines.append(f"| {label} | {k} | {rows.dice.mean():.3f} ± {rows.dice.std():.3f} | {cohort.get('A', float('nan')):.3f} | {cohort.get('B', float('nan')):.3f} | {rows.hd95.mean():.2f} | {worst} ({rows.dice[worst]:.3f}) | {p} |")
    text = "\n".join(lines)
    print(text)
    (ROOT / "summary.md").write_text("# Meta-SSL study results\n" + text + "\n")
    per_animal.to_csv(ROOT / "per_animal_mean_over_seeds.csv", index=False)


if __name__ == "__main__":
    main()
