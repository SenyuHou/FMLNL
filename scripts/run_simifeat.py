import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lnl_foundation.data.datasets import load_cifar_base, DATASET_META
from lnl_foundation.backbones.frozen import canonical_backbone_name
from lnl_foundation.features import feature_cache_path, load_features
from lnl_foundation.partition.saved import resolve_saved_partition
from lnl_foundation.partition.simifeat import nearest, detect_settings
from lnl_foundation.utils import load_config, save_json


PROTOCOL = "simifeat_fixed_features_k10_m21_v1"


def merge_backbone_tables(root):
    frames = []
    for path in sorted((Path(root) / "summaries").glob("*/comparison_f1.csv")):
        frame = pd.read_csv(path, index_col="Method")
        frame.insert(0, "Backbone", path.parent.name)
        frames.append(frame.reset_index())
    if frames:
        pd.concat(frames, ignore_index=True).set_index(["Backbone", "Method"]).to_csv(
            Path(root) / "summaries" / "backbone_comparison_f1.csv"
        )


def merge_results(root, cfg):
    root = Path(root)
    summary_root = root / "summaries" / cfg["backbone"]
    ours = pd.read_csv(summary_root / "runs.csv")
    ours["method"] = "Global-Local-GMM"
    baseline_rows = []
    for path in sorted((root / "simifeat").glob("cifar*/**/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["comparison_protocol"] == PROTOCOL:
            baseline_rows.append(row)
    if not baseline_rows:
        return
    baselines = pd.DataFrame(baseline_rows)
    if "source_feature_sha256" not in baselines:
        # Older runs required exact hashes, so their source and evaluated features were identical.
        baselines["source_feature_sha256"] = baselines["feature_sha256"]
    baselines = baselines[baselines["backbone"] == cfg["backbone"]]
    if cfg.get("feature_sha256"):
        baselines = baselines[baselines["feature_sha256"] == cfg["feature_sha256"]]
    keys = ["dataset", "noise_name", "seed"]
    if baselines.duplicated(keys + ["method"]).any():
        raise ValueError("Multiple SimiFeat feature variants match; run normally instead of --summarize_only.")
    # Pair each baseline to the exact source partition it reused, not to its current feature bytes.
    ours_source = ours.rename(columns={"feature_sha256": "source_feature_sha256"})
    source_keys = keys + ["source_feature_sha256"]
    baselines = baselines.merge(ours_source[source_keys], on=source_keys, validate="many_to_one")
    common = keys + ["backbone", "feature_sha256", "method", "precision", "recall", "f1", "actual_noise_rate"]
    runs = pd.concat([ours[common], baselines[common]], ignore_index=True)
    summary = runs.groupby(["dataset", "noise_name", "method"], dropna=False).agg(
        n_runs=("f1", "size"), precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"), f1_mean=("f1", "mean"), f1_std=("f1", "std"),
    ).reset_index()
    settings = {"human_worse_label": "Human", "human_noisy_label": "Human", "symmetric_0.6": "Symm0.6", "pairflip_0.3": "Pairflip0.3", "instance_0.4": "Inst0.4"}
    summary["setting"] = summary.dataset.str.upper() + "-" + summary.noise_name.map(settings)
    f1 = summary.pivot(index="method", columns="setting", values="f1_mean") * 100
    f1 = f1.reindex(index=["SimiFeat-V", "SimiFeat-R", "Global-Local-GMM"], columns=[
        f"{d}-{n}" for d in ("CIFAR10", "CIFAR100") for n in ("Human", "Symm0.6", "Pairflip0.3", "Inst0.4")])
    summary_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(summary_root / "comparison_runs.csv", index=False)
    summary.to_csv(summary_root / "comparison_summary.csv", index=False)
    f1.to_csv(summary_root / "comparison_f1.csv", index_label="Method")
    merge_backbone_tables(root)
    print(f1.to_string(float_format=lambda x: f"{x:.2f}"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="SimiFeat on current features with compatible saved noisy labels.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg["backbone"] = canonical_backbone_name(cfg["backbone"])
    root = Path(cfg["output_dir"])
    if args.summarize_only:
        merge_results(root, cfg)
        return
    if not args.dataset:
        parser.error("--dataset is required.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load_cifar_base(args.dataset, cfg["data_root"], download=False)
    features = load_features(
        feature_cache_path(cfg["features_root"], args.dataset, cfg["backbone"]),
        base.targets,
        expected_backbone=cfg["backbone"],
    )
    digest = hashlib.sha256(features.numpy().tobytes()).hexdigest()
    cfg["feature_sha256"] = digest
    source_rows = []
    noise_names = (
        "human_worse_label" if args.dataset == "cifar10" else "human_noisy_label",
        "symmetric_0.6",
        "pairflip_0.3",
        "instance_0.4",
    )
    for name in noise_names:
        for seed in (1, 2, 3):
            path, row, match_type = resolve_saved_partition(
                root,
                args.dataset,
                cfg["backbone"],
                name,
                seed,
                cfg["global_posterior_threshold"],
                cfg["local_posterior_threshold"],
                cfg["knn_k"],
                current_feature_sha256=digest,
                equivalence_columns=("index", "noisy_label", "partition"),
            )
            source_rows.append((path.with_name("metrics.json"), row, match_type))
    clean = np.asarray(base.targets)
    print(f"{args.dataset}: computing exact k=10 neighbors on {tuple(features.shape)} fixed features", flush=True)
    neighbors = nearest(features.to(cfg["device"]), 10)
    output = root / "simifeat" / args.dataset / cfg["backbone"] / digest[:12]
    for seed in [1, 2, 3]:
        label_sets, metadata = {}, {}
        for path, row, match_type in source_rows:
            if row["seed"] != seed:
                continue
            name = row["noise_name"]
            out = output / name / f"seed_{seed}"
            frame = pd.read_csv(path.with_name("partition.csv")).sort_values("index")
            if not np.array_equal(frame["index"], np.arange(len(clean))):
                raise ValueError("Misaligned source labels.")
            labels = frame.noisy_label.to_numpy(dtype=np.int64, copy=True)
            label_digest = hashlib.sha256(labels.tobytes()).hexdigest()
            completed = [out / method / "metrics.json" for method in ("SimiFeat-V", "SimiFeat-R")]
            if all(p.exists() for p in completed):
                for p in completed:
                    saved = json.loads(p.read_text(encoding="utf-8"))
                    if saved["comparison_protocol"] != PROTOCOL or saved["feature_sha256"] != digest or saved["noisy_labels_sha256"] != label_digest:
                        raise ValueError(f"Existing SimiFeat run has different inputs/protocol: {p}")
                continue
            original_f1 = f1_score(labels != clean, frame.partition.to_numpy() == 2)
            if not np.isclose(original_f1, row["f1"], atol=1e-12, rtol=0):
                raise ValueError("Source GMM result does not match saved noisy labels.")
            label_sets[name] = labels
            metadata[name] = {**row, "noisy_labels_sha256": label_digest,
                "source_feature_sha256": row.get("feature_sha256"),
                "partition_match": match_type,
                "source_partition": path.with_name("partition.csv").relative_to(root).as_posix()}
        if not label_sets:
            continue
        results = detect_settings(features, label_sets, DATASET_META[args.dataset]["num_classes"], seed, neighbors, cfg["device"])
        for name, result in results.items():
            labels = label_sets[name]
            out = output / name / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(result["trace"], out / "hoc_trace.pt")
            pd.DataFrame({"index": np.arange(len(labels)), "noisy_label": labels,
                "simifeat_v_noisy": result["vote"], "simifeat_r_noisy": result["rank"],
                "rank_vote_count": result["rank_vote_count"], "rank_score": result["score"]}).to_csv(out / "predictions.csv", index=False)
            for method, key in [("SimiFeat-V", "vote"), ("SimiFeat-R", "rank")]:
                detected, truth = result[key], labels != clean
                row = {k: metadata[name][k] for k in ["dataset", "backbone", "noise_name", "seed", "actual_noise_rate", "noisy_labels_sha256", "source_feature_sha256", "partition_match", "source_partition"]}
                row["feature_sha256"] = digest
                row.update(method=method, comparison_protocol=PROTOCOL, k=10, detection_rounds=21,
                    feature_views=1, augmentation=False, hoc_trials=10, hoc_sample_size=15000,
                    hoc_first_steps=400, hoc_warm_steps=20, device=cfg["device"],
                    precision=float(precision_score(truth, detected, zero_division=0)),
                    recall=float(recall_score(truth, detected, zero_division=0)),
                    f1=float(f1_score(truth, detected, zero_division=0)), predicted_noisy=int(detected.sum()))
                save_json(out / method / "metrics.json", row)
                print(f"{args.dataset} {name} seed={seed} {method}: F1={row['f1']*100:.2f}", flush=True)
        merge_results(root, cfg)
    merge_results(root, cfg)


if __name__ == "__main__":
    main()
