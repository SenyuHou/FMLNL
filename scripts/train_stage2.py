import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.backbones.frozen import canonical_backbone_name
from lnl_foundation.data.noise import canonical_noise_type
from lnl_foundation.features import feature_cache_path
from lnl_foundation.partition.global_local_gmm import CLEAN, HARD, NOISY
from lnl_foundation.training import GCE_Q, PROTOTYPE_TEMPERATURE, train_robust_linear_probe
from lnl_foundation.utils import get_device, load_config


METHOD = "CE-GCE-SoftCE"
SETTINGS = {"human": None, "symmetric": 0.6, "pairflip": 0.3, "instance": 0.4}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
FIXED_PARTITION = {"global_posterior_threshold": 0.8, "local_posterior_threshold": 0.5, "knn_k": 20}


def load_cache(path, with_labels=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload["features"].float()
    indices = payload["indices"].long()
    if (
        features.ndim != 2
        or not torch.isfinite(features).all()
        or not torch.equal(indices, torch.arange(len(features)))
        or not payload.get("normalized", False)
    ):
        raise ValueError(f"Invalid or unordered L2 feature cache: {path}")
    labels = payload["clean_labels"].long() if with_labels else None
    return features, labels


def setting_name(dataset, noise_type, ratio):
    if noise_type == "human":
        return "human_worse_label" if dataset == "cifar10" else "human_noisy_label"
    return f"{noise_type}_{ratio:g}"


def find_partition(root, dataset, backbone, name, feature_sha256, seed, cfg):
    partition_seed = 1 if name.startswith("human_") else seed
    matches = []
    for metrics_path in Path(root).glob(f"{dataset}/{backbone}/**/metrics.json"):
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            metadata.get("noise_name") == name
            and metadata.get("seed") == partition_seed
            and metadata.get("tau_global") == cfg["global_posterior_threshold"]
            and metadata.get("tau_local") == cfg["local_posterior_threshold"]
            and metadata.get("knn_k") == cfg["knn_k"]
            and metadata.get("feature_sha256") == feature_sha256
        ):
            matches.append(metrics_path.with_name("partition.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one partition for {dataset}/{name}/seed_{partition_seed}, found {len(matches)}.")
    return matches[0], partition_seed


def load_partition(path, num_samples):
    frame = pd.read_csv(path, usecols=["index", "noisy_label", "partition"])
    if not np.array_equal(frame["index"].to_numpy(), np.arange(num_samples)):
        raise ValueError(f"Partition rows are not in CIFAR order: {path}")
    labels = torch.from_numpy(frame["noisy_label"].to_numpy(dtype=np.int64, copy=True))
    partition = torch.from_numpy(frame["partition"].to_numpy(dtype=np.int64, copy=True))
    counts = {
        "n_clean": int((partition == CLEAN).sum()),
        "n_hard": int((partition == HARD).sum()),
        "n_noisy": int((partition == NOISY).sum()),
    }
    if sum(counts.values()) != num_samples:
        raise ValueError(f"Partition counts do not sum to {num_samples}: {path}")
    return labels, partition, counts


def save_results(rows, path):
    frame = pd.DataFrame(rows).sort_values(["noise_type", "seed"])
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def merge_backbone_summaries(output_root):
    """Build one compact table across all completed stage-2 backbones."""
    frames = [pd.read_csv(path) for path in sorted(Path(output_root).glob("*/*/summary.csv"))]
    if not frames:
        return None
    summary = pd.concat(frames, ignore_index=True)
    setting_names = {
        "human_worse_label": "Human",
        "human_noisy_label": "Human",
        "symmetric_0.6": "Symm0.6",
        "pairflip_0.3": "Pairflip0.3",
        "instance_0.4": "Inst0.4",
    }
    summary["setting"] = summary["dataset"].str.upper() + "-" + summary["noise_name"].map(setting_names)
    if summary["setting"].isna().any():
        unknown = sorted(summary.loc[summary["setting"].isna(), "noise_name"].unique())
        raise ValueError(f"Unknown stage-2 noise names: {unknown}")

    rows = []
    for metric, mean_column, std_column in (
        ("Accuracy", "accuracy_mean", "accuracy_std"),
        ("Macro-F1", "macro_f1_mean", "macro_f1_std"),
    ):
        for statistic, column in (("Mean (%)", mean_column), ("Std (pp)", std_column)):
            part = summary[["backbone", "setting", column]].rename(columns={column: "value"})
            part["Metric"] = metric
            part["Statistic"] = statistic
            rows.append(part)
    tidy = pd.concat(rows, ignore_index=True)
    if tidy.duplicated(["backbone", "Metric", "Statistic", "setting"]).any():
        raise ValueError("Duplicate stage-2 backbone summary rows.")

    columns = [
        f"{dataset}-{setting}"
        for dataset in ("CIFAR10", "CIFAR100")
        for setting in ("Human", "Symm0.6", "Pairflip0.3", "Inst0.4")
    ]
    table = tidy.pivot(index=["backbone", "Metric", "Statistic"], columns="setting", values="value")
    table = (table.reindex(columns=columns) * 100).round(2).reset_index()
    table = table.rename(columns={"backbone": "Backbone"})
    destination = Path(output_root) / "backbone_comparison" / "summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination


def main():
    parser = argparse.ArgumentParser(description="Train the fixed robust linear probe.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dataset", required=True, choices=NUM_CLASSES)
    parser.add_argument("--settings", nargs="+", type=canonical_noise_type, choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    for key, expected in FIXED_PARTITION.items():
        if cfg[key] != expected:
            raise ValueError(f"The fixed stage-2 method requires {key}={expected}, got {cfg[key]}.")
    backbone = canonical_backbone_name(cfg["backbone"])
    seeds = tuple(dict.fromkeys(args.seeds or cfg["seeds"]))
    output_root = Path(args.output_dir).resolve() if args.output_dir else Path(cfg["output_dir"]).parent / "stage2"
    output_path = output_root / backbone / args.dataset / "runs.csv"

    train_path = feature_cache_path(cfg["features_root"], args.dataset, backbone, split="train")
    test_path = feature_cache_path(cfg["features_root"], args.dataset, backbone, split="test")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Extract train and test features first: {train_path}, {test_path}")
    train_features, _ = load_cache(train_path)
    test_features, test_labels = load_cache(test_path, with_labels=True)
    feature_sha256 = hashlib.sha256(train_features.numpy().tobytes()).hexdigest()
    rows = [] if args.force or not output_path.exists() else pd.read_csv(output_path).to_dict("records")
    completed = {(row["noise_type"], int(row["seed"])) for row in rows}
    device = get_device(cfg["device"])
    print(f"Loaded {args.dataset} {backbone}: train={tuple(train_features.shape)}, test={tuple(test_features.shape)}")

    for noise_type in dict.fromkeys(args.settings):
        ratio = SETTINGS[noise_type]
        name = setting_name(args.dataset, noise_type, ratio)
        for seed in seeds:
            if (noise_type, seed) in completed and not args.force:
                print(f"Skipping completed run {args.dataset}/{name}/seed_{seed}")
                continue
            partition_path, partition_seed = find_partition(
                cfg["output_dir"], args.dataset, backbone, name, feature_sha256, seed, cfg
            )
            noisy_labels, partition, counts = load_partition(partition_path, len(train_features))
            print(
                f"{args.dataset}/{name}/seed_{seed} (partition seed {partition_seed}): "
                f"Clean={counts['n_clean']} Hard={counts['n_hard']} Noisy={counts['n_noisy']}"
            )
            metrics = train_robust_linear_probe(
                train_features,
                noisy_labels,
                partition,
                test_features,
                test_labels,
                NUM_CLASSES[args.dataset],
                seed,
                device,
            )
            row = {
                "method": METHOD,
                "dataset": args.dataset,
                "noise_type": noise_type,
                "noise_rate": ratio,
                "noise_name": name,
                "backbone": backbone,
                "feature_dim": train_features.shape[-1],
                "feature_sha256": feature_sha256,
                "seed": seed,
                "partition_seed": partition_seed,
                "tau_global": cfg["global_posterior_threshold"],
                "tau_local": cfg["local_posterior_threshold"],
                "knn_k": cfg["knn_k"],
                "gce_q": GCE_Q,
                "prototype_temperature": PROTOTYPE_TEMPERATURE,
                **counts,
                **metrics,
            }
            rows.append(row)
            completed.add((noise_type, seed))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_results(rows, output_path)
            print(f"Accuracy={metrics['accuracy']:.6f}, Macro-F1={metrics['macro_f1']:.6f}")

    runs = pd.DataFrame(rows)
    summary = runs.groupby(["dataset", "noise_name", "backbone"], dropna=False).agg(
        n_runs=("seed", "size"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary.to_csv(output_path.with_name("summary.csv"), index=False)
    backbone_summary_path = merge_backbone_summaries(output_root)
    print(summary.to_string(index=False))
    print(f"Runs: {output_path}")
    print(f"Summary: {output_path.with_name('summary.csv')}")
    if backbone_summary_path:
        print(f"Backbone summary: {backbone_summary_path}")


if __name__ == "__main__":
    main()
