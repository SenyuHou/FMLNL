# Frozen Features + Global-Local GMM

The pipeline has three steps: extract frozen pretrained ViT features, partition
CIFAR training samples into Clean (0), Hard (1), and Noisy (2), then train the
fixed robust linear probe.
The partitioner only consumes features and noisy labels. True labels are used
to generate benchmark noise and evaluate noisy-label detection, not to fit GMMs.
There is no OOF reference model or hardness oracle.

The repository intentionally excludes CIFAR data, extracted feature caches,
per-sample partitions, HOC traces, historical outputs, and validation-only
files. It tracks the core implementation and compact final tables under
`outputs/generalization/summaries/`. On a new server, install the environment
and run feature extraction before starting experiments.

## Run

From the FMLNL directory in PowerShell:

```powershell
conda activate fmlnl

# CIFAR-10: extract train/test once, then run all four settings with seeds 1, 2, 3.
python scripts/extract_features.py --dataset cifar10 --splits train test
python scripts/run_partition_generalization.py --dataset cifar10

# CIFAR-100: run separately; Human defaults to human_noisy_label.
python scripts/extract_features.py --dataset cifar100 --splits train test
python scripts/run_partition_generalization.py --dataset cifar100

# Rebuild summaries for both datasets without running experiments.
python scripts/run_partition_generalization.py --summarize_only

# After train and test features and first-stage partitions exist, train the
# fixed Clean-CE + Hard-GCE + Noisy-Prototype-SoftCE linear probe.
python scripts/train_stage2.py --dataset cifar10
python scripts/train_stage2.py --dataset cifar100
```

Existing `features/<dataset>/vit_b16/train_l2.pt` caches are validated and reused
without loading/downloading ViT. Only training features are required. Existing
test caches are retained; `--splits train test` extracts both splits if needed.
Use `--force_extract` only to deliberately replace a cache.

Supported backbones and aliases:

| Name | Checkpoint | Output dimension | Evaluation preprocessing |
| --- | --- | ---: | --- |
| `vit_b16_imagenet` (`vit_b16`) | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 768 | timm pretrained config |
| `vit_l16_imagenet` (`vit_l16`) | `vit_large_patch16_224.augreg_in21k_ft_in1k` | 1024 | timm pretrained config |
| `clip_vit_b16` | OpenAI `ViT-B-16` | 512 | open_clip OpenAI transform |
| `clip_vit_l14` | OpenAI `ViT-L-14` | 768 | open_clip OpenAI transform |
| `dinov2_vit_b14` | `vit_base_patch14_dinov2.lvd142m` | 768 | timm pretrained config |

Select one with `--set backbone=NAME`, for example:

```powershell
python scripts/extract_features.py --dataset cifar10 --splits train test --set backbone=clip_vit_b16
```

Each backbone uses the transform supplied by its pretrained checkpoint. Newly
extracted global features are L2-normalized once before saving and record their
actual `feature_dim`. Legacy ImageNet ViT caches without this metadata remain
valid: canonical names check their new directory first and then directly reuse
the old `vit_b16` or `vit_l16` path without copying or loading a model.

For an unavailable Hugging Face connection:

```powershell
python scripts/extract_features.py --dataset cifar10 --set hf_endpoint=https://hf-mirror.com
```

For a single setting/seed or another CIFAR-10 Human annotation:

```powershell
python scripts/run_partition_generalization.py --dataset cifar10 --settings asymmetric --seeds 1
python scripts/run_partition_generalization.py --dataset cifar10 --settings human --human_noise_type human_aggre_label
```

`--settings` accepts `human`, `symmetric`, `pairflip` (alias `asymmetric`), and
`instance` (alias `idn`). Paths in `configs/default.yaml` resolve relative to
FMLNL. `--set key=value` overrides its existing keys; `--output_dir PATH` on
the runner selects a separate output root relative to the working directory.

## Fixed Noise Protocol

| Setting | Generation |
| --- | --- |
| Human | Read original CIFAR-N annotations, using JYP's key selection. |
| Symm. 0.6 | Keep the label with probability 0.4; each other class gets 0.6/(C-1). |
| Asym. 0.3 / pairflip | Flip class c to (c+1) mod C with probability 0.3, for every class. |
| Inst. 0.4 / IDN | JYP `generate_instance_noise_labels`, tau=0.4, std=0.1. |

Both CIFAR-10 and CIFAR-100 use one full-class cycle for pairflip. There are no
hand-selected semantic flip pairs or five-class block cycles. With 100 classes,
class 4 flips to 5, and class 99 flips to 0. Actual noise rates are measured and
saved, rather than assumed equal to the nominal rate.

All synthetic labels are generated in memory for each run. IDN uses normalized
32x32 pixels, a seeded random class-dependent weight matrix, truncated-normal
per-sample flip probabilities, and softmax sampling. It reads neither saved
noisy labels nor prediction/softmax files. CPU and CUDA floating-point behavior
may differ; each run records its device and actual generated labels.

Human files are the only label files read:

- `data/noise_label_human/CIFAR-10_human.pt`
- `data/noise_label_human/CIFAR-100_human.pt`

CIFAR-10 defaults to `human_worse_label`; selectable options follow JYP:
`human_worse_label`, `human_aggre_label`, `human_random_label1`,
`human_random_label2`, `human_random_label3`, `human_noisy_label`.
The selected key must exist in the annotation file. CIFAR-100 only accepts
`human_noisy_label`, which is selected automatically. Label length, range, and
the annotation's clean-label ordering are checked.

## Partition Rule

1. Normalize features, form prototypes using noisy labels, and compute the
   assigned-class similarity minus the strongest competing-class similarity.
2. Fit a two-component GMM to this global margin. Posterior >= 0.8 assigns a
   reliable Clean or Noisy state; remaining samples are ambiguous.
3. Compute cosine kNN label agreement with k=20. Fit a second two-component GMM
   only on ambiguous samples. Local noisy posterior >= 0.5 gives Noisy;
   otherwise retain the sample as Hard.

The existing median fallback for small/degenerate ambiguous subsets is retained
and recorded in the metrics. Hard is a predicted reliability group, not a
ground-truth hard-clean label. GMM uses 10 initializations and the run seed.

## Fixed Stage-2 Training

The final second-stage method freezes the backbone and trains only
`nn.Linear(feature_dim, num_classes)`. Feature dimension is read from the cache.
It uses one fixed loss design for every dataset and noise setting:

- Clean: cross entropy with the current noisy label.
- Hard: generalized cross entropy with fixed `q=0.7` and the current noisy label.
- Noisy: entropy-confidence-weighted soft cross entropy against class-prototype
  targets. Prototypes use only predicted Clean samples grouped by noisy label;
  prototype temperature is fixed at `0.1`.

The optimizer is AdamW with learning rate `1e-3`, weight decay `1e-4`, batch
size 256, and 50 epochs. Seeds default to `1, 2, 3`. Human labels and their
first-stage partition are fixed across classifier seeds; synthetic settings use
the matching seed's saved labels and partition. Ground-truth training labels are
never read by the stage-2 trainer. Official clean test labels are used only for
final Accuracy, Macro-F1, and top-label ECE with 15 equal-width confidence bins.

After Ours finishes training, it performs Reliability-Anchored Temperature
Calibration without changing the classifier. For each class, candidates are
predicted-Clean samples whose noisy label and linear-head prediction agree.
The highest `global_margin` 10% per class are used as anchors; classes with at
most 10 candidates retain all candidates. A single positive scalar temperature
is fitted so the anchors' mean maximum confidence approaches `0.995`. This fit
does not accept clean training labels, a ground-truth noise mask, validation
labels, or test labels. Temperature scaling changes probabilities only, so
Accuracy and Macro-F1 are checked to remain exactly unchanged.

Stage-2 results are written locally under
`outputs/stage2/<backbone>/<dataset>/`. Existing runs are skipped unless
`--force` is provided. Runs from the previous CSV schema are automatically
recomputed when their ECE/calibration fields are missing. Each run records
Accuracy, Macro-F1, raw ECE, temperature, and calibrated ECE; three-seed
mean/std tables are refreshed at
`outputs/stage2/backbone_comparison/summary.csv`. Detailed logits,
probabilities, and anchor indices are saved locally under `artifacts/` and are
excluded from version control.

## Results

Each run writes only `metrics.json` and `partition.csv` under:

```text
outputs/generalization/<dataset>/<backbone>/<parameter_feature_variant>/<noise_name>/seed_<seed>/
```

`metrics.json` records precision, recall, F1 (all on the 0-1 scale), actual noise
rate, sample counts, seed, parameters, device, feature hash and noise protocol.
`partition.csv` records each sample's noisy label, partition and GMM scores.
Its labels are an output record only and are never reloaded to generate noise.

After each completed run the following tables are rebuilt under
`outputs/generalization/summaries/<backbone>/`, with dataset-specific copies in
its `cifar10/` and `cifar100/` subdirectories:

- `runs.csv`: per-seed metrics.
- `summary.csv`: means, sample standard deviations and number of runs.
- `f1.csv`: wide F1 table in percent. `pairflip_0.3` is the Asym. 0.3 column.

Missing settings remain absent/blank. A single run has no sample standard
deviation. Feature/parameter variants are grouped separately. Rerunning a seed
replaces that run; it does not duplicate it. Historical results are excluded.

## Alternative Detection Metrics

The formal local threshold is now 0.5 (global threshold remains 0.8).
To regenerate formal partitions and metrics from existing GMM posteriors:

```powershell
python scripts/run_partition_generalization.py --apply_local_threshold
```

This creates new parameter-specific run directories. Superseded 0.8 results
are preserved under `outputs/history/`. Standard `runs.csv`, `summary.csv` and `f1.csv` summarize only the
backbone and GMM parameters selected by the current configuration. For formal
0.5 runs, the alternative 0.5 detection rule below is identical to the original
Noisy-only metric; the old 0.8 runs remain available for historical comparison.

The original rule detects only Noisy samples. An additional rule detects Noisy
plus Hard samples whose saved local noisy posterior is >= 0.5. Clean samples
stay unchanged. Ground-truth labels are used only for evaluation, never to
choose which Hard samples to add. The original partition and metric files are
preserved. This is an alternative binary decision rule, not a relabeling oracle.

```powershell
python scripts/run_partition_generalization.py --recalculate_metrics
# Optional: one dataset or another explicitly reported threshold.
python scripts/run_partition_generalization.py --recalculate_metrics --dataset cifar100 --hard_detection_threshold 0.5
```

This reads completed partitions without extracting features, regenerating
noise, or refitting GMMs. A normal experiment run also generates the comparison
for its dataset after finishing. Outputs are `detection_comparison_tau_0.5_runs.csv`
and `detection_comparison_tau_0.5_summary.csv`, under
`outputs/generalization/summaries/<backbone>/<dataset-or-all>/`. They contain paired Precision/Recall/F1, confusion
counts per seed, added true-noisy/true-clean counts, and summary means/stds.
Metrics are on the 0-1 scale; `delta_f1_pp` is in percentage points. Threshold
variants have separate filenames. Comparisons on already examined benchmark
labels are diagnostic; do not present post-hoc threshold selection as an
independently validated improvement.

## Historical Results

The previous 24 selected runs are retained in `outputs/history/runs.csv`, with
explicit noise protocols. `outputs/history/comparison_clip_reference.csv`
preserves the previously requested comparison table. Its CIFAR-10 Asym value
is pairflip, but its CIFAR-100 Asym value uses the OLD five-class block cycle
and is not a result of the current protocol. CIFAR-100 pairflip must be run.

Our cached features are from timm ViT-B/16, whereas the SimiFeat reference values
in that historical table use CLIP. That table is not a controlled comparison
under identical features or verified identical noise protocols.

Old baseline checkpoints, diagnostic plots, OOF reference caches, redundant
tables and synthetic-noise caches are isolated under `outputs/history/legacy_*`.
Automatic approval blocked their deletion; these directories are not consumed
by the current pipeline. The original datasets, Human annotations and expensive
feature caches are retained in their original locations.

## Files

The formal pipeline uses `scripts/extract_features.py`,
`scripts/run_partition_generalization.py`, and `scripts/train_stage2.py`.
`scripts/run_simifeat.py` remains the fixed-feature baseline comparison; see
`docs/simifeat/REPRODUCTION.md`. The package contains the frozen backbone,
CIFAR loader, noise generation, feature cache/extraction, GMM, fixed robust
linear probe, and configuration utilities. Diagnostic scripts and outputs are
kept only in the ignored local `history/` directory.

## Clean-LP Reference

`scripts/run_clean_lp.py` trains the formal clean-label linear-probe reference
only for `dinov2_vit_b14`. It validates and reuses the same train/test feature
caches as Ours, reads original CIFAR targets, and trains `nn.Linear` with the
same AdamW configuration, initialization, shuffling, epochs, batch size, and
final-epoch evaluation. It does not load noisy annotations or partitions.
Clean-LP reports Accuracy, Macro-F1, and raw 15-bin ECE. As a baseline it never
invokes reliability-anchored calibration.

Run CIFAR-10 and CIFAR-100 separately; the second completed command creates the
eight-column summary and direct Accuracy comparison with Ours:

```bash
python scripts/run_clean_lp.py --dataset cifar10 --set backbone=dinov2_vit_b14
python scripts/run_clean_lp.py --dataset cifar100 --set backbone=dinov2_vit_b14
```

Formal outputs are `outputs/clean_lp/dinov2_clean_lp_raw.csv`,
`outputs/clean_lp/clean_lp_summary.csv`, and
`outputs/clean_lp/dinov2_ours_vs_clean_lp.csv`.

## CLIP Stage-1 Baselines

`scripts/run_stage1_baseline.py` adapts the original CLIPCleaner and DeFT
noisy-label detectors to the exact labels already stored by the formal FMLNL
Global-Local-GMM runs. Each command runs one method, one dataset, and one CLIP
backbone over Human, symmetric 0.6, all-class pairflip 0.3, and matrix IDN 0.4
for seeds 1, 2, and 3.

Noisy-label source partitions are feature-independent. The runner first
prefers an exact feature match, then the same backbone, then any other formal
backbone, and verifies that all available candidates have identical noisy-label
hashes. Baseline outputs remain separated by the evaluated backbone and its
own feature hash, so local and server feature variants cannot silently replace
one another.

```bash
python scripts/run_stage1_baseline.py --method clipcleaner --dataset cifar10 --backbone clip_vit_b16
python scripts/run_stage1_baseline.py --method deft --dataset cifar100 --backbone clip_vit_l14
```

CLIPCleaner reuses the cached normalized CLIP image features and loads the
matching OpenAI CLIP text encoder for the authors' descriptor prompts. DeFT
loads the full matching OpenAI CLIP model because its original detector trains
deep visual prompts and positive/negative text prompts for ten epochs. DeFT's
upstream batch size is 64; use `--batch_size` only if a larger CLIP model does
not fit available GPU memory. Ground-truth clean labels are only read after
detection to calculate Precision, Recall, F1, AUROC, and AUPRC.

Compact outputs are written below
`outputs/generalization/baselines/summaries/<backbone>/<method>/`. Detailed
per-sample predictions and per-run metadata stay below the method directory
and are intentionally ignored by Git. See `docs/baselines/README.md` for the
adaptation boundary and source licenses.
