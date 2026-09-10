import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.data.datasets import DATASET_META, load_cifar_base
from lnl_foundation.data.noise import HUMAN_CHOICES, canonical_noise_type, human_noise_key, make_noisy_labels
from lnl_foundation.backbones.frozen import canonical_backbone_name
from lnl_foundation.features import feature_cache_path, load_features
from lnl_foundation.partition.global_local_gmm import CLEAN, HARD, NOISY, GlobalLocalGMMPartitioner
from lnl_foundation.partition.saved import FORMAL_PARTITION_PROTOCOL
from lnl_foundation.utils import load_config, save_json


PROTOCOL = FORMAL_PARTITION_PROTOCOL
SETTINGS = {"human": None, "symmetric": 0.6, "pairflip": 0.3, "instance": 0.4}


def compare_detection_rules(partition, local_noisy_prob, truth, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("Hard detection threshold must be in [0, 1].")
    partition = np.asarray(partition)
    local_noisy_prob = np.asarray(local_noisy_prob)
    truth = np.asarray(truth, dtype=bool)
    if partition.shape != truth.shape or local_noisy_prob.shape != truth.shape:
        raise ValueError("Partition, probability and truth shapes must match.")
    hard = partition == HARD
    if not np.isin(partition, [CLEAN, HARD, NOISY]).all() or not np.isfinite(local_noisy_prob[hard]).all():
        raise ValueError("Invalid partition or missing local posterior for Hard samples.")
    if ((local_noisy_prob[hard] < 0) | (local_noisy_prob[hard] > 1)).any():
        raise ValueError("Local posteriors must be in [0, 1].")
    original = partition == NOISY
    # Selection uses model outputs only; truth is used exclusively for scoring.
    added = hard & (local_noisy_prob >= threshold)
    result = {"hard_detection_threshold": threshold, "added_count": int(added.sum()),
              "added_true_noisy": int((added & truth).sum()),
              "added_true_clean": int((added & ~truth).sum())}
    for name, detected in [("original", original), ("expanded", original | added)]:
        result.update({
            f"{name}_precision": float(precision_score(truth, detected, zero_division=0)),
            f"{name}_recall": float(recall_score(truth, detected, zero_division=0)),
            f"{name}_f1": float(f1_score(truth, detected, zero_division=0)),
            f"{name}_tp": int((detected & truth).sum()),
            f"{name}_fp": int((detected & ~truth).sum()),
            f"{name}_fn": int((~detected & truth).sum()),
            f"{name}_tn": int((~detected & ~truth).sum()),
        })
    result["delta_f1_pp"] = 100 * (result["expanded_f1"] - result["original_f1"])
    return result


def recalculate_metrics(cfg, threshold=0.5, dataset=None):
    if not 0 <= threshold <= 1:
        raise ValueError("Hard detection threshold must be in [0, 1].")
    root = Path(cfg["output_dir"])
    clean_labels, rows = {}, []
    groups = ["dataset", "backbone", "feature_sha256", "device", "tau_global", "tau_local", "knn_k", "noise_name"]
    for path in sorted(root.glob("cifar*/**/metrics.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (metadata.get("protocol") != PROTOCOL
            or canonical_backbone_name(metadata["backbone"]) != cfg["backbone"]
            or (dataset and metadata["dataset"] != dataset)):
            continue
        name = metadata["dataset"]
        if name not in clean_labels:
            clean_labels[name] = np.asarray(load_cifar_base(name, cfg["data_root"], download=False).targets)
        df = pd.read_csv(path.with_name("partition.csv"), usecols=["index", "noisy_label", "partition", "local_noisy_prob"])
        indices = df["index"].to_numpy()
        if len(df) != metadata["num_samples"] or not np.array_equal(np.sort(indices), np.arange(len(clean_labels[name]))):
            raise ValueError(f"Invalid saved sample indices: {path.parent}")
        truth = df["noisy_label"].to_numpy() != clean_labels[name][indices.astype(np.int64)]
        metrics = compare_detection_rules(df["partition"].to_numpy(), df["local_noisy_prob"].to_numpy(), truth, threshold)
        for metric in ("precision", "recall", "f1"):
            if not np.isclose(metrics[f"original_{metric}"], metadata[metric], rtol=0, atol=1e-10):
                raise ValueError(f"Recomputed {metric} differs from the saved result: {path}")
        rows.append({**{key: metadata[key] for key in groups}, "seed": metadata["seed"], **metrics})
    if not rows:
        print("No completed runs available for metric comparison.")
        return
    runs = pd.DataFrame(rows)
    aggregations = {"n_runs": ("seed", "size")}
    for field in ("original_precision", "original_recall", "original_f1", "expanded_precision", "expanded_recall", "expanded_f1", "delta_f1_pp", "added_count", "added_true_noisy", "added_true_clean"):
        aggregations[f"{field}_mean"] = (field, "mean")
        aggregations[f"{field}_std"] = (field, "std")
    summary = runs.groupby(groups + ["hard_detection_threshold"], dropna=False).agg(**aggregations).reset_index()
    destination = root / "summaries" / cfg["backbone"] / (dataset or "all")
    destination.mkdir(parents=True, exist_ok=True)
    prefix = f"detection_comparison_tau_{threshold:g}"
    runs.to_csv(destination / f"{prefix}_runs.csv", index=False)
    summary.to_csv(destination / f"{prefix}_summary.csv", index=False)
    display = summary[groups + ["original_f1_mean", "expanded_f1_mean", "delta_f1_pp_mean"]].copy()
    display[["original_f1_mean", "expanded_f1_mean"]] *= 100
    print(display.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"Metric comparison: {destination / (prefix + '_summary.csv')}", flush=True)


def run_one(cfg, dataset, features, base, noise_type, seed, human_noise_type=None):
    ratio = SETTINGS[noise_type]
    key = human_noise_key(dataset, human_noise_type) if noise_type == "human" else ""
    noise_name = f"human_{key}" if key else f"{noise_type}_{ratio:g}"
    print(f"{dataset} | {noise_name} | seed={seed}: generating labels", flush=True)
    clean = np.asarray(base.targets, dtype=np.int64)
    noisy = make_noisy_labels(
        dataset, cfg["data_root"], base.data, clean, DATASET_META[dataset]["num_classes"],
        noise_type, ratio, seed, device=cfg["device"], human_noise_type=human_noise_type,
    )
    print(f"Actual noise: {np.mean(noisy != clean):.4f}; fitting Global-Local GMM", flush=True)
    partitioner = GlobalLocalGMMPartitioner(
        DATASET_META[dataset]["num_classes"], tau_global=cfg["global_posterior_threshold"],
        local_posterior_threshold=cfg["local_posterior_threshold"], k=cfg["knn_k"], seed=seed,
    )
    pred = partitioner.fit_predict(features, torch.from_numpy(noisy))
    partition = pred["partition"].numpy()
    truth, detected = noisy != clean, partition == NOISY
    row = {
        "protocol": PROTOCOL, "dataset": dataset, "backbone": cfg["backbone"],
        "noise_type": noise_type, "noise_name": noise_name, "noise_ratio": ratio,
        "noise_generator": "jyp_matrix_idn" if noise_type == "instance" else noise_name,
        "seed": seed, "device": cfg["device"], "num_samples": len(clean),
        "feature_sha256": cfg["feature_sha256"],
        "tau_global": cfg["global_posterior_threshold"],
        "tau_local": cfg["local_posterior_threshold"], "knn_k": cfg["knn_k"],
        "actual_noise_rate": float(truth.mean()),
        "precision": float(precision_score(truth, detected, zero_division=0)),
        "recall": float(recall_score(truth, detected, zero_division=0)),
        "f1": float(f1_score(truth, detected, zero_division=0)),
        "clean_count": int((partition == CLEAN).sum()),
        "hard_count": int((partition == HARD).sum()),
        "noisy_count": int(detected.sum()),
        "local_gmm_fallback": bool(pred["local_gmm_fallback"]),
    }
    # Separate parameter/feature variants so partial reruns cannot mix experiments.
    variant = f"g{row['tau_global']:g}_l{row['tau_local']:g}_k{row['knn_k']}_{row['device'].replace(':', '_')}_{row['feature_sha256'][:12]}"
    out = Path(cfg["output_dir"]) / dataset / cfg["backbone"] / variant / noise_name / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "index": np.arange(len(clean)), "noisy_label": noisy,
        **{name: pred[name].numpy() for name in (
            "partition", "global_margin", "global_clean_prob", "global_noisy_prob",
            "global_state", "local_consistency", "local_hard_prob", "local_noisy_prob",
        )},
    }).to_csv(out / "partition.csv", index=False)
    save_json(out / "metrics.json", row)
    print(f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} -> {out}", flush=True)
    return row


def apply_saved_local_threshold(cfg):
    """Materialize a new formal partition from unchanged fitted posteriors."""
    root = Path(cfg["output_dir"])
    target = float(cfg["local_posterior_threshold"])
    if not 0.5 <= target < 1:
        raise ValueError("Local posterior threshold must be in [0.5, 1).")
    labels, count = {}, 0
    for path in sorted(root.glob("cifar*/**/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("protocol") != PROTOCOL or row["tau_local"] == target
            or row["backbone"] != cfg["backbone"]
            or row["tau_global"] != cfg["global_posterior_threshold"]
            or row["knn_k"] != cfg["knn_k"]):
            continue
        dataset = row["dataset"]
        if dataset not in labels:
            labels[dataset] = np.asarray(load_cifar_base(dataset, cfg["data_root"], download=False).targets)
        df = pd.read_csv(path.with_name("partition.csv"))
        indices = df["index"].to_numpy()
        if not np.array_equal(np.sort(indices), np.arange(len(labels[dataset]))):
            raise ValueError(f"Invalid sample indices: {path}")
        truth = df["noisy_label"].to_numpy() != labels[dataset][indices.astype(np.int64)]
        previous = compare_detection_rules(df.partition.to_numpy(), df.local_noisy_prob.to_numpy(), truth, target)
        for metric in ("precision", "recall", "f1"):
            if not np.isclose(previous[f"original_{metric}"], row[metric], rtol=0, atol=1e-10):
                raise ValueError(f"Saved metrics do not match partition: {path}")
        ambiguous = df.global_state.eq(HARD).to_numpy()
        if not np.isfinite(df.loc[ambiguous, "local_noisy_prob"]).all():
            raise ValueError(f"Missing ambiguous-sample posteriors: {path}")
        partition = np.full(len(df), NOISY, dtype=np.int64)
        partition[df.global_state.eq(CLEAN)] = CLEAN
        partition[ambiguous & df.local_noisy_prob.lt(target).to_numpy()] = HARD
        if row["local_gmm_fallback"]:
            # The median fallback is independent of the posterior threshold.
            partition = df.partition.to_numpy().copy()
        df["partition"] = partition
        detected = partition == NOISY
        row.update(tau_local=target, precision=float(precision_score(truth, detected, zero_division=0)),
                   recall=float(recall_score(truth, detected, zero_division=0)),
                   f1=float(f1_score(truth, detected, zero_division=0)),
                   clean_count=int((partition == CLEAN).sum()), hard_count=int((partition == HARD).sum()),
                   noisy_count=int(detected.sum()), derived_from=str(path.relative_to(root)))
        variant = f"g{row['tau_global']:g}_l{target:g}_k{row['knn_k']}_{row['device'].replace(':', '_')}_{row['feature_sha256'][:12]}"
        out = root / dataset / row["backbone"] / variant / row["noise_name"] / f"seed_{row['seed']}"
        if (out / "metrics.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "partition.csv", index=False)
        save_json(out / "metrics.json", row)
        count += 1
    print(f"Created {count} formal partitions with tau_local={target:g}; original runs retained.")
    summarize(root, cfg)


def summarize(output_root, cfg=None):
    output_root = Path(output_root)
    rows = []
    for path in sorted(output_root.glob("cifar*/**/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("protocol") == PROTOCOL and (cfg is None or (
            row["backbone"] == cfg["backbone"] and row["tau_global"] == cfg["global_posterior_threshold"]
            and row["tau_local"] == cfg["local_posterior_threshold"] and row["knn_k"] == cfg["knn_k"]
        )):
            rows.append(row)
    if not rows:
        print("No results for the current protocol. Run a dataset first.")
        return
    runs = pd.DataFrame(rows)
    groups = ["dataset", "backbone", "feature_sha256", "device", "tau_global", "tau_local", "knn_k", "noise_name"]
    summary_root = output_root / "summaries" / cfg["backbone"]
    for destination, frame in [(summary_root, runs)] + [
        (summary_root / dataset, group) for dataset, group in runs.groupby("dataset")
    ]:
        destination.mkdir(parents=True, exist_ok=True)
        summary = frame.groupby(groups, dropna=False).agg(
            n_runs=("f1", "size"), actual_noise_rate=("actual_noise_rate", "mean"),
            precision_mean=("precision", "mean"), precision_std=("precision", "std"),
            recall_mean=("recall", "mean"), recall_std=("recall", "std"),
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        ).reset_index()
        frame.to_csv(destination / "runs.csv", index=False)
        summary.to_csv(destination / "summary.csv", index=False)
        f1 = summary.pivot(index=groups[:-1], columns="noise_name", values="f1_mean") * 100
        f1.to_csv(destination / "f1.csv")
    print("\nNoisy-label detection F1 (%); Asym = all-class pairflip")
    print((runs.groupby(groups)["f1"].agg(["mean", "std"]) * 100).to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"Full summaries: {summary_root / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Frozen features -> Global-Local GMM partition.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"])
    parser.add_argument("--settings", nargs="+", type=canonical_noise_type, choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--human_noise_type", choices=HUMAN_CHOICES)
    parser.add_argument("--output_dir")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--recalculate_metrics", action="store_true", help="Compare saved runs without refitting; optionally select --dataset.")
    parser.add_argument("--apply_local_threshold", action="store_true", help="Create formal partitions for both datasets using the configured local threshold and saved posteriors.")
    parser.add_argument("--hard_detection_threshold", type=float, default=0.5)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg["backbone"] = canonical_backbone_name(cfg["backbone"])
    if args.output_dir:
        cfg["output_dir"] = str(Path(args.output_dir).resolve())
    if not 0 <= args.hard_detection_threshold <= 1:
        parser.error("--hard_detection_threshold must be in [0, 1].")
    if args.apply_local_threshold:
        apply_saved_local_threshold(cfg)
        return
    if args.recalculate_metrics:
        recalculate_metrics(cfg, args.hard_detection_threshold, args.dataset)
        return
    if args.summarize_only:
        summarize(cfg["output_dir"], cfg)
        return
    if not args.dataset:
        parser.error("--dataset is required unless --summarize_only is used.")
    if "human" in args.settings:
        human_noise_key(args.dataset, args.human_noise_type)
    base = load_cifar_base(args.dataset, cfg["data_root"])
    path = feature_cache_path(cfg["features_root"], args.dataset, cfg["backbone"])
    if not path.exists():
        raise FileNotFoundError(f"Run extract_features.py --dataset {args.dataset} first: {path}")
    features = load_features(path, base.targets, expected_backbone=cfg["backbone"])
    cfg["feature_sha256"] = hashlib.sha256(features.numpy().tobytes()).hexdigest()
    print(f"Loaded {args.dataset}: {tuple(features.shape)} frozen features", flush=True)
    for noise_type in dict.fromkeys(args.settings):
        for seed in dict.fromkeys(args.seeds or cfg["seeds"]):
            run_one(cfg, args.dataset, features, base, noise_type, seed, args.human_noise_type)
            summarize(cfg["output_dir"], cfg)
    recalculate_metrics(cfg, args.hard_detection_threshold, args.dataset)


if __name__ == "__main__":
    main()
