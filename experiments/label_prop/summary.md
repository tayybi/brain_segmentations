# Label propagation study results

Runs: 15 (complete: 15); test animals: 20

| Method | Dice (mean ± SD) | Cohort A | Cohort B | HD95 px | Gap closed | Pseudo-label Dice (all / edge) | p vs ours |
|---|---|---:|---:|---:|---:|---|---:|
| Anchors only (3 slices/mouse) | 0.924 ± 0.036 | 0.919 | 0.930 | 13.23 |  | – | 0.000168 |
| + copy nearest anchor | 0.908 ± 0.022 | 0.909 | 0.908 | 13.26 | -74% | 0.864 / 0.705 | 9.54e-06 |
| + shape-based interpolation | 0.909 ± 0.023 | 0.911 | 0.906 | 12.82 | -72% | 0.871 / 0.705 | 1.34e-05 |
| + registration propagation | 0.930 ± 0.030 | 0.936 | 0.923 | 11.68 | 25% | 0.902 / 0.780 | 0.261 |
| + registration w/ anchor loss | 0.931 ± 0.029 | 0.936 | 0.926 | 11.22 | 32% | 0.906 / 0.782 | 0.596 |
| + self-training | 0.927 ± 0.035 | 0.922 | 0.932 | 12.60 | 11% | 0.893 / 0.726 | 0.000483 |
| + registration-guided self-training (ours) | 0.933 ± 0.034 | 0.933 | 0.934 | 10.97 | 41% | 0.911 / 0.786 | – |
| All slices labelled (upper bound) | 0.946 ± 0.033 | 0.953 | 0.940 | 9.05 |  | – | 0.000851 |
