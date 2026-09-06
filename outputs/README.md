# Output layout

`generalization/` contains the active `tau_g=0.8`, `tau_l=0.5`, `k=20`
experiments. The Git repository tracks only `generalization/summaries/`.
Detailed per-sample runs and `history/` remain local and are intentionally
ignored because they are large or superseded.

```text
generalization/
  cifar10/<backbone>/<parameter_feature_variant>/<noise>/seed_<seed>/
  cifar100/<backbone>/<parameter_feature_variant>/<noise>/seed_<seed>/
  simifeat/<dataset>/<backbone>/<feature_hash>/<noise>/seed_<seed>/
  summaries/<backbone>/
    runs.csv                 # Global-Local GMM runs
    summary.csv              # Global-Local GMM mean/std
    f1.csv                   # Global-Local GMM F1 table
    comparison_runs.csv      # GMM + SimiFeat per-seed runs
    comparison_summary.csv   # GMM + SimiFeat mean/std
    comparison_f1.csv        # GMM + SimiFeat eight-setting F1 table
  summaries/backbone_comparison_f1.csv  # all completed backbones
```

Active backbone directory names are canonical: `vit_b16_imagenet`,
`vit_l16_imagenet`, `clip_vit_b16`, `clip_vit_l14`, and `dinov2_vit_b14`.
The dated directory `history/generalization_pre_consolidation_2026-09-07/`
preserves the old `tau_l=0.8` runs and ambiguous duplicate summaries.
