import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.backbones.frozen import canonical_backbone_name
from lnl_foundation.data.datasets import DATASET_META, load_cifar_base
from lnl_foundation.features import feature_cache_path, load_features
from lnl_foundation.training import LinearProbeConfig, train_clean_linear_probe
from lnl_foundation.utils import get_device, load_config


METHOD = "Clean-LP"
BACKBONE = "dinov2_vit_b14"
SEEDS = (1, 2, 3)
DATASETS = ("cifar10", "cifar100")
SETTINGS = ("Human", "Symm0.6", "Pairflip0.3", "Inst0.4")


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_summary(raw, output_path):
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    actual = set(zip(raw["dataset"], raw["seed"].astype(int)))
    if actual != expected or raw.duplicated(["dataset", "seed"]).any():
        print("Clean-LP summary is waiting for all CIFAR-10/100 seeds 1, 2, 3.", flush=True)
        return None

    grouped = raw.groupby("dataset").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    rows = []
    for metric, mean_column, std_column in (
        ("Accuracy", "accuracy_mean", "accuracy_std"),
        ("Macro-F1", "macro_f1_mean", "macro_f1_std"),
    ):
        for statistic, column in (("Mean (%)", mean_column), ("Std (pp)", std_column)):
            row = {"Backbone": BACKBONE, "Method": METHOD, "Metric": metric, "Statistic": statistic}
            for dataset in DATASETS:
                value = float(grouped.loc[dataset, column]) * 100
                for setting in SETTINGS:
                    row[f"{dataset.upper()}-{setting}"] = round(value, 2)
            rows.append(row)
    summary = pd.DataFrame(rows)
    numeric = summary.iloc[:, 4:].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Clean-LP summary contains NaN or Inf.")
    save_csv(summary, output_path)
    return summary


def build_comparison(clean_summary, ours_path, output_path):
    if clean_summary is None:
        return None
    ours = pd.read_csv(ours_path)
    ours = ours[(ours["Backbone"] == BACKBONE) & (ours["Metric"] == "Accuracy")]
    if set(ours["Statistic"]) != {"Mean (%)", "Std (pp)"}:
        raise ValueError(f"Missing DINOv2 Accuracy rows in Ours summary: {ours_path}")
    ours = ours.set_index("Statistic")
    clean = clean_summary[clean_summary["Metric"] == "Accuracy"].set_index("Statistic")
    rows = []
    for dataset in ("CIFAR10", "CIFAR100"):
        for noise in SETTINGS:
            column = f"{dataset}-{noise}"
            ours_mean = float(ours.loc["Mean (%)", column])
            ours_std = float(ours.loc["Std (pp)", column])
            clean_mean = float(clean.loc["Mean (%)", column])
            clean_std = float(clean.loc["Std (pp)", column])
            rows.append({
                "Dataset": dataset,
                "Noise": noise,
                "Ours_Acc_Mean": ours_mean,
                "Ours_Acc_Std": ours_std,
                "CleanLP_Acc_Mean": clean_mean,
                "CleanLP_Acc_Std": clean_std,
                "Ours_minus_CleanLP": round(ours_mean - clean_mean, 2),
            })
    comparison = pd.DataFrame(rows)
    if not np.isfinite(comparison.iloc[:, 2:].to_numpy(dtype=float)).all():
        raise ValueError("Ours versus Clean-LP comparison contains NaN or Inf.")
    save_csv(comparison, output_path)
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Clean-label linear-probe reference on frozen DINOv2 features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    backbone = canonical_backbone_name(cfg["backbone"])
    if backbone != BACKBONE:
        raise ValueError(f"The formal Clean-LP reference only supports backbone={BACKBONE}, got {backbone}.")
    device = get_device(cfg["device"])
    output_root = Path(args.output_dir).resolve() if args.output_dir else Path(cfg["output_dir"]).parent / "clean_lp"
    raw_path = output_root / "dinov2_clean_lp_raw.csv"
    summary_path = output_root / "clean_lp_summary.csv"
    comparison_path = output_root / "dinov2_ours_vs_clean_lp.csv"
    ours_path = Path(cfg["output_dir"]).parent / "stage2" / "backbone_comparison" / "summary.csv"

    train_base = load_cifar_base(args.dataset, cfg["data_root"], train=True, download=False)
    test_base = load_cifar_base(args.dataset, cfg["data_root"], train=False, download=False)
    train_path = feature_cache_path(cfg["features_root"], args.dataset, backbone, split="train")
    test_path = feature_cache_path(cfg["features_root"], args.dataset, backbone, split="test")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"DINOv2 feature cache is missing: {train_path}, {test_path}")

    clean_train_labels = torch.as_tensor(train_base.targets, dtype=torch.long)
    clean_test_labels = torch.as_tensor(test_base.targets, dtype=torch.long)
    train_features = load_features(train_path, clean_train_labels, expected_backbone=backbone)
    test_features = load_features(test_path, clean_test_labels, expected_backbone=backbone)
    expected_train = 50000
    expected_test = 10000
    if len(train_features) != expected_train or len(test_features) != expected_test:
        raise ValueError(
            f"Unexpected {args.dataset} sizes: train={len(train_features)}, test={len(test_features)}"
        )
    feature_sha256 = hashlib.sha256(train_features.numpy().tobytes()).hexdigest()
    config = LinearProbeConfig()
    ours_runs_path = Path(cfg["output_dir"]).parent / "stage2" / backbone / args.dataset / "runs.csv"
    if not ours_runs_path.exists():
        raise FileNotFoundError(f"Formal Ours runs are required to verify the feature cache: {ours_runs_path}")
    ours_runs = pd.read_csv(ours_runs_path)
    if set(ours_runs["feature_sha256"]) != {feature_sha256}:
        raise ValueError(
            f"Clean-LP feature cache does not match formal Ours for {args.dataset}: {train_path}"
        )
    if set(ours_runs["feature_dim"].astype(int)) != {int(train_features.shape[-1])}:
        raise ValueError(f"Clean-LP feature dimension does not match formal Ours: {ours_runs_path}")

    print(f"dataset={args.dataset}", flush=True)
    print(f"backbone={backbone}", flush=True)
    print(f"train_feature_cache={train_path}", flush=True)
    print(f"test_feature_cache={test_path}", flush=True)
    print(f"feature_sha256={feature_sha256}", flush=True)
    print(f"feature_dim={train_features.shape[-1]}", flush=True)
    print(f"n_train={len(train_features)} n_test={len(test_features)}", flush=True)
    print("training_label_source=clean_ground_truth", flush=True)
    print(
        f"optimizer=AdamW lr={config.learning_rate} weight_decay={config.weight_decay} "
        f"epochs={config.epochs} batch_size={config.batch_size} scheduler=None device={device}",
        flush=True,
    )

    columns = [
        "dataset", "backbone", "method", "seed", "feature_dim", "feature_sha256",
        "n_train", "n_test", "training_label_source", "accuracy", "macro_f1",
    ]
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=columns)
    if len(raw) and not set(columns).issubset(raw.columns):
        raise ValueError(f"Invalid existing Clean-LP raw file: {raw_path}")
    if args.force:
        raw = raw[raw["dataset"] != args.dataset].copy()
    existing = raw[raw["dataset"] == args.dataset]
    if len(existing) and (
        set(existing["backbone"]) != {backbone}
        or set(existing["method"]) != {METHOD}
        or set(existing["training_label_source"]) != {"clean_ground_truth"}
        or set(existing["feature_sha256"]) != {feature_sha256}
    ):
        raise ValueError(
            f"Existing Clean-LP rows use different labels, backbone, or features: {raw_path}. "
            "Use --force only when intentionally replacing this dataset."
        )
    completed = set(raw.loc[raw["dataset"] == args.dataset, "seed"].astype(int))
    for seed in SEEDS:
        if seed in completed:
            print(f"Skipping completed Clean-LP {args.dataset}/seed_{seed}", flush=True)
            continue
        metrics = train_clean_linear_probe(
            train_features,
            clean_train_labels,
            test_features,
            clean_test_labels,
            DATASET_META[args.dataset]["num_classes"],
            seed,
            device,
            config,
        )
        row = {
            "dataset": args.dataset,
            "backbone": backbone,
            "method": METHOD,
            "seed": seed,
            "feature_dim": int(train_features.shape[-1]),
            "feature_sha256": feature_sha256,
            "n_train": len(train_features),
            "n_test": len(test_features),
            "training_label_source": "clean_ground_truth",
            **metrics,
        }
        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
        raw = raw.sort_values(["dataset", "seed"]).reset_index(drop=True)
        save_csv(raw[columns], raw_path)
        print(
            f"{args.dataset}/seed_{seed}: Accuracy={metrics['accuracy']:.6f}, "
            f"Macro-F1={metrics['macro_f1']:.6f}",
            flush=True,
        )

    clean_summary = build_summary(raw[columns], summary_path)
    comparison = build_comparison(clean_summary, ours_path, comparison_path)
    print(f"Raw: {raw_path}", flush=True)
    if clean_summary is not None:
        print(clean_summary.to_string(index=False), flush=True)
        print(f"Summary: {summary_path}", flush=True)
        print(comparison.to_string(index=False), flush=True)
        print(f"Comparison: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
