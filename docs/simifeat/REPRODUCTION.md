# SimiFeat Fixed-Feature Comparison

Reference: Zhaowei Zhu, Zihao Dong, Yang Liu. Detecting Corrupted Labels Without
Training a Model to Predict. ICML 2022. https://proceedings.mlr.press/v162/zhu22a.html
Source: https://github.com/UCSC-REAL/SimiFeat (local SimiFeat-main copy).
Upstream's CC BY-NC 4.0 license is reproduced in LICENSE.md.

## Controlled Inputs

All three methods use exactly the existing L2-normalized timm ViT-B/16 feature
cache and the same 50,000 training indices. SimiFeat reads noisy labels from
each completed formal GMM partition, rather than generating a fresh noise
realization. The adapter verifies the saved GMM F1 and records SHA256 hashes
of both the feature matrix and noisy label array.

The eight settings are CIFAR-10/100 with Human, Symmetric 0.6, full-class
Pairflip 0.3 and JYP matrix IDN 0.4. CIFAR-10 Human uses worse_label;
CIFAR-100 uses noisy_label. GMM uses tau_global=0.8, tau_local=0.5, k=20.

## SimiFeat Rules Preserved

- k=10, min_similarity=0, similarity-weighted class counts, L2 normalization.
- The upstream k neighbors INCLUDE the first/self match. Its distance is
  replaced with 2*d(second)-d(third) before weighting; it is not dropped.
- V: argmax of the weighted neighborhood distribution, compared to the noisy
  label. Ties follow the implementation's first-index argmax.
- R: negative log of the observed noisy-label coordinate of the L2-normalized
  neighborhood vector. This is monotonic in the paper's cosine score.
- HOC: first/second/third label moments from the center and its two nearest
  other matches, on 10 random subsets of 15,000 examples per detection round.
- HOC objective: sum of the three Frobenius/L2 moment residual norms; Adam;
  diagonal-biased transition initialization and random prior initialization.
  First round: 400 objective evaluations, lr=0.1. Subsequent rounds: 20,
  lr=0.01, using warm-start logits. Like upstream, each has steps-1 updates.
- R estimates class-specific clean proportions with Bayes' rule, perturbs each
  diagonal with Uniform(-0.05, 0.05), then applies the upstream percentile rule
  with Tii_offset=1.0 and its 0.05/0.95 boundary handling.
- R uses a majority of 21 detection rounds. The V decision is unchanged across
  rounds on fixed features, so its single decision is equivalent to 21 votes.

## Explicit Adaptations

This is a FIXED-FEATURE reproduction, not reproduction of the paper's CLIP
table. By user choice, no random image crops/flips or 21-view re-extraction
are performed. R still repeats random HOC estimation and threshold jitter.

Cosine search is exact but blockwise on the available device, instead of
materializing N-by-N distance matrices. HOC expected moments and gradients are
vectorized contractions mathematically equivalent to upstream's loops. TF32
is disabled. Floating-point/tie effects can differ from the original CPU code.

Sample order is the canonical CIFAR index order. Sampling and jitter use
separate seeded RNG streams to permit sharing label-independent geometry
across noise settings without changing their marginal distributions. This
does not promise bit-for-bit RNG equivalence to upstream's shuffled loader.

Upstream's detached optimizer checkpoint aliases its live parameters and
therefore returns the final iterate. The adapter preserves that actual
behavior rather than substituting a cloned best-loss checkpoint.

True clean labels are only used AFTER prediction to score precision/recall/F1.
Neither HOC nor voting/ranking receives clean labels, true transitions, nominal
noise ratios, or oracle thresholds. HOC optimizes a small transition model;
it does not train a classifier on images or features.

## Running and Outputs

```powershell
python scripts/run_simifeat.py --dataset cifar10 --set backbone=vit_b16_imagenet
python scripts/run_simifeat.py --dataset cifar100 --set backbone=vit_b16_imagenet
python scripts/run_simifeat.py --summarize_only --set backbone=vit_b16_imagenet
```

Completed dataset/setting/seed results resume automatically. The fixed-feature
protocol identifier is simifeat_fixed_features_k10_m21_v1.

Under `outputs/generalization/simifeat/<dataset>/<backbone>/<feature_hash>/`,
each seed saves a `predictions.csv`, `hoc_trace.pt` (estimated
matrices/priors/objectives for each round), and separate V/R `metrics.json`
files. Under `outputs/generalization/summaries/<backbone>/`:

- comparison_runs.csv: paired per-seed results of all three methods.
- comparison_summary.csv: means, sample standard deviations and run counts.
- comparison_f1.csv: eight-setting F1 (%) table.

These tables contain locally computed results only, not the paper's numbers.
The GMM's original runs.csv, summary.csv, and f1.csv are not overwritten.
The GMM local threshold 0.5 was chosen after earlier benchmark inspection;
this descriptive comparison is not an independently held-out tuning study.

## Checks

tests/test_simifeat.py extracts original functions through Python AST and
checks neighborhood distributions, empirical HOC moments, theoretical moments
and their gradients, solver outputs, and both final decisions on small inputs.

```powershell
python -m unittest discover -s tests -v
```
