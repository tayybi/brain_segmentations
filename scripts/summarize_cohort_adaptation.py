#!/usr/bin/env python3
"""Summarise run_cohort_adaptation.py: per-mouse means over seeds, then paired tests.

    python scripts/summarize_cohort_adaptation.py experiments/cohort_adaptation
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/cohort_adaptation")
scores = defaultdict(list)  # (split, k, method, animal) -> [(dice, hd95)]
for path in root.glob("*/metrics.csv"):
    for r in csv.DictReader(path.open()):
        scores[(r["split"], int(r["k"]), r["method"], r["animal"])].append((float(r["dice"]), float(r["hd95"])))
mean = {key: np.mean(v, 0) for key, v in scores.items()}

for split in sorted({k[0] for k in mean}):
    animals = sorted({k[3] for k in mean if k[0] == split}, key=lambda a: int(a[:-1]))
    source = np.array([mean[(split, 0, "source", a)] for a in animals])
    print(f"\n{split}  ({len(animals)} target mice, seeds: {len(scores[(split, 0, 'source', animals[0])])})")
    print(f"  {'k':>2} {'method':10} {'Dice':>6} {'HD95':>6}   vs ft_anchor   vs ft_full (non-inf, 0.01)")
    print(f"  {0:>2} {'source':10} {source[:, 0].mean():6.3f} {source[:, 1].mean():6.1f}")
    for k in sorted({key[1] for key in mean if key[0] == split and key[1] > 0}):
        get = lambda m: np.array([mean[(split, k, m, a)] for a in animals]) if (split, k, m, animals[0]) in mean else None
        anchor, full = get("ft_anchor"), get("ft_full")
        for method in ("ft_anchor", "ft_sam2", "ft_full"):
            x = get(method)
            if x is None:
                continue
            text = f"  {k:>2} {method:10} {x[:, 0].mean():6.3f} {x[:, 1].mean():6.1f}"
            if method == "ft_sam2" and anchor is not None:
                text += f"   p={wilcoxon(x[:, 0], anchor[:, 0]).pvalue:.3f} ({(x[:, 0] > anchor[:, 0]).sum()}/{len(animals)} better)"
                if full is not None:
                    text += f"   p={wilcoxon(x[:, 0] - full[:, 0] + 0.01, alternative='greater').pvalue:.3f}"
            print(text)
