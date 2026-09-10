# CLIP noisy-label detection baselines

The formal entry point is `scripts/run_stage1_baseline.py`. It consumes the
same CIFAR sample order and saved noisy labels as the current FMLNL
Global-Local-GMM/SimiFeat comparison. Clean labels are not passed to either
detector and are used only by the shared binary evaluation step.

Source partitions do not need to share the evaluated feature hash or backbone.
The runner validates that all available formal candidates for each
dataset/noise/seed contain the same noisy-label vector, then prefers an exact
feature match, the same backbone, or another backbone in that order. The
evaluated feature hash remains part of every output path and metrics row.

## CLIPCleaner

The implementation ports `combined_selection` from the original CLIPCleaner
repository. It retains:

- the original class-specific CIFAR descriptor prompts;
- zero-shot probabilities with descriptor aggregation and temperature 0.07;
- balanced logistic regression on normalized visual CLIP features;
- per-class normalization followed by two-component GMM selection;
- zero-shot and visual label-consistency thresholds;
- intersection of all four selected sets and the empty-class safeguard.

The binary prediction is the original final selected set. For threshold-free
metrics, the noise score is the negative minimum margin among the four rules,
which is the continuous counterpart of their intersection. The original
source is MIT licensed; see `CLIPCLEANER_LICENSE.md`.

## DeFT

The implementation ports phase one of DeFT. It retains the authors' OpenAI
CLIP model, deep VPT length 20, learnable positive/negative prompts with 16
context tokens, SGD and cosine schedule, ten epochs, clean threshold 0.5, and
the original positive/negative losses. Synthetic settings follow
`main_phase1.py` with one warm-up epoch. Human noise follows
`main_real_phase1.py`, including SCE warm-up for five epochs and the additional
positive-class argmax selection condition. The final-epoch `1 - p_clean` is
used as the continuous noise score. The original source is MIT licensed; see
`DEFT_LICENSE.md`.

DeFT cannot use only a frozen feature cache: VPT changes intermediate visual
tokens and its learned dual text prompts are part of the detector. The cache
is still loaded and hashed to bind every run to the same formal FMLNL source
partition and CLIP backbone.
