# Output layout

The repository uses an explicit result whitelist. New compact results from a
local machine or server are expected to appear in Git and should be committed.
Large intermediate artifacts remain local even when they are written below
`outputs/`.

Tracked formal results:

- `generalization/summaries/**`: aggregate first-stage GMM and SimiFeat tables.
- `generalization/baselines/summaries/**`: compact CLIPCleaner/DeFT run and mean/std tables.
- `stage2/**/runs.csv`: final robust linear-probe metrics for each seed.
- `stage2/**/summary.csv`: final robust linear-probe mean/std tables.
- `clean_lp/*.csv`: DINOv2 clean-label linear-probe reference tables.

Ignored artifacts include per-sample `partition.csv`, detailed first-stage run
directories, diagnostic experiments, logs, feature caches, and historical
outputs. Do not force-add these ignored files.

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
  baselines/
    clipcleaner/<dataset>/<backbone>/<feature_hash>/<noise>/seed_<seed>/
    deft/<dataset>/<backbone>/<feature_hash>/<noise>/seed_<seed>/
    summaries/<backbone>/<method>/
      runs.csv               # all completed per-seed detection metrics
      summary.csv            # Precision/Recall/F1/AUROC/AUPRC mean/std
      f1.csv                 # noise-setting F1 table

stage2/<backbone>/<dataset>/
  runs.csv                 # fixed CE-GCE-SoftCE metrics for each seed
  summary.csv              # fixed CE-GCE-SoftCE mean/std
stage2/backbone_comparison/
  summary.csv              # all-backbone Accuracy/Macro-F1 mean/std table

clean_lp/
  dinov2_clean_lp_raw.csv       # six per-seed Clean-LP runs
  clean_lp_summary.csv          # eight-column Clean-LP mean/std table
  dinov2_ours_vs_clean_lp.csv   # eight-setting Accuracy comparison
```

Active backbone directory names are canonical: `vit_b16_imagenet`,
`vit_l16_imagenet`, `clip_vit_b16`, `clip_vit_l14`, and `dinov2_vit_b14`.
The dated directory `history/generalization_pre_consolidation_2026-09-07/`
preserves the old `tau_l=0.8` runs and ambiguous duplicate summaries.

When a new backbone finishes, files such as
`generalization/summaries/dinov2_vit_b14/...` show as untracked (`U`) because
they are important results selected for synchronization. Files under
`generalization/cifar10/` or `generalization/cifar100/` stay ignored because
they contain the much larger per-sample partitions.
