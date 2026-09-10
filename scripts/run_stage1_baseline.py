import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.backbones.frozen import canonical_backbone_name
from lnl_foundation.baselines.clipcleaner import CLIPCleaner, zero_shot_prediction
from lnl_foundation.baselines.deft import DeFTConfig, run_deft
from lnl_foundation.data.datasets import load_cifar_base
from lnl_foundation.data.noise import HUMAN_CHOICES, human_noise_key
from lnl_foundation.features import feature_cache_path, load_features
from lnl_foundation.utils import get_device, load_config, save_json, set_seed


SOURCE_PROTOCOL = "global_local_gmm_pairflip_v1"
BASELINE_PROTOCOL = "stage1_clip_baselines_v1"
METHOD_NAMES = {"clipcleaner": "CLIPCleaner", "deft": "DeFT"}
CLIP_BACKBONES = ("clip_vit_b16", "clip_vit_l14")
SYNTHETIC_SETTINGS = ("symmetric_0.6", "pairflip_0.3", "instance_0.4")
SUMMARY_METRICS = ("precision", "recall", "f1", "auroc", "auprc")


def expected_settings(dataset, human_noise_type):
    human = human_noise_key(dataset, human_noise_type)
    return (f"human_{human}", *SYNTHETIC_SETTINGS)


def load_source_runs(root, dataset, backbone, feature_sha256, cfg, human_noise_type):
    expected = expected_settings(dataset, human_noise_type)
    candidates = {}
    for path in sorted((root / dataset).glob("*/**/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("protocol") != SOURCE_PROTOCOL
            or row.get("tau_global") != cfg["global_posterior_threshold"]
            or row.get("tau_local") != cfg["local_posterior_threshold"]
            or row.get("knn_k") != cfg["knn_k"]
            or row.get("noise_name") not in expected
            or int(row.get("seed", -1)) not in (1, 2, 3)
        ):
            continue
        key = (row["noise_name"], int(row["seed"]))
        partition_path = path.with_name("partition.csv")
        if partition_path.exists():
            candidates.setdefault(key, []).append((path, row))
    required = {(name, seed) for name in expected for seed in (1, 2, 3)}
    missing = sorted(required - set(candidates))
    if missing:
        raise ValueError(
            "Missing formal Global-Local-GMM source partitions for this dataset: "
            f"{missing}. Partitions may come from any backbone; feature hashes need not match."
        )

    matches = {}
    for key in sorted(required):
        resolved = []
        for path, row in candidates[key]:
            frame = pd.read_csv(path.with_name("partition.csv"), usecols=["index", "noisy_label"])
            frame = frame.sort_values("index")
            if not np.array_equal(frame["index"].to_numpy(), np.arange(len(frame))):
                raise ValueError(f"Misaligned source labels: {path.with_name('partition.csv')}")
            labels = frame["noisy_label"].to_numpy(dtype=np.int64)
            label_sha256 = hashlib.sha256(labels.tobytes()).hexdigest()
            resolved.append((path, row, labels, label_sha256))
        label_hashes = {item[3] for item in resolved}
        if len(label_hashes) != 1:
            details = [(str(item[0]), item[3]) for item in resolved]
            raise ValueError(f"Conflicting noisy labels for {dataset}/{key}: {details}")
        resolved.sort(key=lambda item: (
            item[1].get("feature_sha256") != feature_sha256,
            item[1].get("backbone") != backbone,
            str(item[0]),
        ))
        matches[key] = resolved[0]
    return matches


def detection_metrics(truth, predicted_noisy, noise_score):
    truth = np.asarray(truth, dtype=bool)
    predicted_noisy = np.asarray(predicted_noisy, dtype=bool)
    noise_score = np.asarray(noise_score, dtype=np.float64)
    if not np.isfinite(noise_score).all():
        raise ValueError("Detection score contains NaN or Inf.")
    return {
        "precision": float(precision_score(truth, predicted_noisy, zero_division=0)),
        "recall": float(recall_score(truth, predicted_noisy, zero_division=0)),
        "f1": float(f1_score(truth, predicted_noisy, zero_division=0)),
        "auroc": float(roc_auc_score(truth, noise_score)),
        "auprc": float(average_precision_score(truth, noise_score)),
    }


def summarize(root, backbone, method):
    rows = []
    result_root = root / method.lower()
    for path in sorted(result_root.glob(f"cifar*/{backbone}/**/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("comparison_protocol") == BASELINE_PROTOCOL and row.get("method") == method:
            rows.append(row)
    if not rows:
        return
    runs = pd.DataFrame(rows).sort_values(["dataset", "noise_name", "seed"])
    groups = ["dataset", "backbone", "method", "feature_sha256", "noise_name"]
    aggregations = {
        "n_runs": ("seed", "size"),
        "actual_noise_rate": ("actual_noise_rate", "mean"),
    }
    for metric in SUMMARY_METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    summary = runs.groupby(groups, dropna=False).agg(**aggregations).reset_index()
    summary_root = root / "summaries" / backbone / method.lower()
    destinations = [(summary_root, runs)]
    destinations.extend((summary_root / dataset, frame) for dataset, frame in runs.groupby("dataset"))
    for destination, frame in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        local_summary = summary[summary.dataset.isin(frame.dataset.unique())]
        frame.to_csv(destination / "runs.csv", index=False)
        local_summary.to_csv(destination / "summary.csv", index=False)
        f1 = local_summary.pivot(
            index=["dataset", "backbone", "method", "feature_sha256"],
            columns="noise_name",
            values="f1_mean",
        ) * 100
        f1.to_csv(destination / "f1.csv")
    print((summary[[*groups, "n_runs", "f1_mean", "f1_std"]]).to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ), flush=True)
    print(f"Summary: {summary_root / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run one CLIP noisy-label detection baseline on one CIFAR dataset."
    )
    parser.add_argument("--method", required=True, choices=tuple(METHOD_NAMES))
    parser.add_argument("--dataset", required=True, choices=("cifar10", "cifar100"))
    parser.add_argument("--backbone", required=True, choices=CLIP_BACKBONES)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--human_noise_type", choices=HUMAN_CHOICES)
    parser.add_argument("--device")
    parser.add_argument("--batch_size", type=int, default=64, help="DeFT only; upstream default is 64.")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    backbone = canonical_backbone_name(args.backbone)
    if backbone not in CLIP_BACKBONES:
        parser.error("Stage-one CLIP baselines only support CLIP ViT-B/16 and ViT-L/14.")
    device = get_device(args.device or cfg["device"])
    method = METHOD_NAMES[args.method]
    base = load_cifar_base(args.dataset, cfg["data_root"], download=False)
    clean = np.asarray(base.targets, dtype=np.int64)
    feature_path = feature_cache_path(cfg["features_root"], args.dataset, backbone)
    features = load_features(feature_path, clean, expected_backbone=backbone)
    feature_sha256 = hashlib.sha256(features.numpy().tobytes()).hexdigest()
    generalization_root = Path(cfg["output_dir"])
    sources = load_source_runs(
        generalization_root, args.dataset, backbone, feature_sha256, cfg, args.human_noise_type
    )
    output_root = generalization_root / "baselines"
    variant_root = output_root / args.method / args.dataset / backbone / feature_sha256[:12]
    prediction_zero = checkpoint = None

    for noise_name in expected_settings(args.dataset, args.human_noise_type):
        for seed in (1, 2, 3):
            source_path, source, labels, label_sha256 = sources[(noise_name, seed)]
            if len(labels) != len(clean):
                raise ValueError(f"Unexpected source-label count: {source_path.with_name('partition.csv')}")
            destination = variant_root / noise_name / f"seed_{seed}"
            metrics_path = destination / "metrics.json"
            if metrics_path.exists() and not args.force:
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                if (
                    saved.get("comparison_protocol") != BASELINE_PROTOCOL
                    or saved.get("feature_sha256") != feature_sha256
                    or saved.get("noisy_labels_sha256") != label_sha256
                ):
                    raise ValueError(f"Existing baseline result has different inputs: {metrics_path}")
                print(f"Skipping completed {method} {args.dataset}/{noise_name}/seed_{seed}", flush=True)
                continue

            set_seed(seed)
            if args.method == "clipcleaner":
                if prediction_zero is None:
                    print(
                        f"Computing CLIPCleaner zero-shot descriptor predictions for {args.dataset}",
                        flush=True,
                    )
                    prediction_zero, checkpoint = zero_shot_prediction(
                        features, args.dataset, backbone, device
                    )
                result = CLIPCleaner(theta_gmm=0.5, theta_cons=0.8).detect(
                    features, labels, prediction_zero
                )
                method_parameters = {
                    "theta_gmm": 0.5,
                    "theta_consistency": 0.8,
                    "selection": "intersection_of_four_upstream_rules",
                    "clip_checkpoint": checkpoint,
                }
            else:
                deft_config = DeFTConfig(batch_size=args.batch_size)
                result = run_deft(
                    base.data,
                    labels,
                    args.dataset,
                    backbone,
                    seed,
                    device,
                    num_workers=args.num_workers if args.num_workers is not None else cfg["num_workers"],
                    config=deft_config,
                    human=noise_name.startswith("human_"),
                )
                method_parameters = {
                    "epochs": deft_config.epochs,
                    "warmup": result["warmup"],
                    "batch_size": deft_config.batch_size,
                    "learning_rate": deft_config.learning_rate,
                    "weight_decay": deft_config.weight_decay,
                    "momentum": deft_config.momentum,
                    "vpt_len": deft_config.vpt_len,
                    "n_ctx": deft_config.n_ctx,
                    "clip_checkpoint": result["checkpoint"],
                }

            predicted_noisy = ~result["predicted_clean"]
            truth = labels != clean
            metrics = detection_metrics(truth, predicted_noisy, result["noise_score"])
            row = {
                "comparison_protocol": BASELINE_PROTOCOL,
                "method_protocol": (
                    "clipcleaner_combined_selection_v1"
                    if args.method == "clipcleaner" else "deft_phase1_v1"
                ),
                "dataset": args.dataset,
                "backbone": backbone,
                "method": method,
                "noise_name": noise_name,
                "seed": seed,
                "actual_noise_rate": float(truth.mean()),
                "feature_sha256": feature_sha256,
                "noisy_labels_sha256": label_sha256,
                "source_backbone": source["backbone"],
                "source_feature_sha256": source["feature_sha256"],
                "source_partition": str(source_path.with_name("partition.csv").relative_to(generalization_root)),
                "predicted_clean": int(result["predicted_clean"].sum()),
                "predicted_noisy": int(predicted_noisy.sum()),
                "score_direction": "higher_is_noisier",
                **method_parameters,
                **metrics,
            }
            destination.mkdir(parents=True, exist_ok=True)
            predictions = {
                "index": np.arange(len(clean)),
                "noisy_label": labels,
                "predicted_noisy": predicted_noisy,
                "noise_score": result["noise_score"],
            }
            if "clean_score" in result:
                predictions["clean_score"] = result["clean_score"]
            if "component_clean_confidence" in result:
                for index, values in enumerate(result["component_clean_confidence"].T, start=1):
                    predictions[f"component_{index}_clean_confidence"] = values
            pd.DataFrame(predictions).to_csv(destination / "predictions.csv", index=False)
            save_json(metrics_path, row)
            print(
                f"{method} {args.dataset}/{noise_name}/seed_{seed}: "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f} "
                f"AUPRC={metrics['auprc']:.4f}",
                flush=True,
            )
            summarize(output_root, backbone, method)
    summarize(output_root, backbone, method)


if __name__ == "__main__":
    main()
