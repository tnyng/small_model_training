# Mean vs Mode for `heuristic(new) + gpt-4.1-mini`

Compared:

- **new score**: `weighted_scale_sums` from `experiments/mean_annotation_gpt4.1mini_heuristic/score_outputs/`
- **old score**: integer parsed from `output` in `experiments/annotation_gpt4.1mini_heuristic/accuracy_outputs/`

This is only a **partial overlap** comparison, because the new heuristic-parsed files are not identical to the old ones.

## Main result

For 5 of the 6 regenerated datasets, the shift from old mode to new mean is **moderate, not dramatic** on overlapping rows:

| Dataset | Overlap rows | Mean abs diff | Rounded label changed |
|---|---:|---:|---:|
| `article_bias` | 1180 | 0.230 | 12.8% |
| `bureaucracies` | 676 | 0.300 | 18.5% |
| `gwsd` | 508 | 0.219 | 14.0% |
| `hatecot` | 1352 | 0.269 | 15.4% |
| `implicit_hate` | 980 | 0.331 | 21.2% |

These five datasets still show high correlation between old mode and new mean, so the mean target looks like a smoothing / softening of the same signal rather than a wholesale change.

## Outlier

| Dataset | Overlap rows | Mean abs diff | Rounded label changed |
|---|---:|---:|---:|
| `misinfo_cancer` | 751 | 1.440 | 46.5% |

- Because the heuristic parser output also changed, this note should **not** be read as a perfectly controlled same-data ablation.

## Files

- [mean_vs_mode_summary.csv](/Users/yangtian/Documents/law/calib/RuleFollower/modernbert_uq/results/audit/temp_mean_vs_mode_compare/mean_vs_mode_summary.csv)
- [mean_vs_mode_largest_diffs.csv](/Users/yangtian/Documents/law/calib/RuleFollower/modernbert_uq/results/audit/temp_mean_vs_mode_compare/mean_vs_mode_largest_diffs.csv)
