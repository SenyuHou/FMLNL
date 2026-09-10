import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.backbones.frozen import canonical_backbone_name
from lnl_foundation.data.datasets import DATASET_META, load_cifar_base
from lnl_foundation.data.noise import canonical_noise_type, make_noisy_labels
from lnl_foundation.features import feature_cache_path, load_features
from lnl_foundation.results.ece import build_ece_reports
from lnl_foundation.training import ECE_BINS, STAGE2_BASELINE_METHODS, train_stage2_baseline
from lnl_foundation.utils import ROOT, get_device, load_config


SETTINGS = {"human": None, "symmetric": 0.6, "pairflip": 0.3, "instance": 0.4}
METHOD_NAMES = {
    "ce": "CE",
    "gce": "GCE",
    "coteaching": "Co-teaching",
    "dividemix": "DivideMix",
    "disc": "DISC",
    "clipcleaner": "CLIPCleaner",
}
REQUIRED_METRICS = ("accuracy", "macro_f1", "ece_raw")
PROTOCOL = "frozen_feature_stage2_baselines_v1"


def _noise_name(dataset, noise_type, ratio):
    if noise_type == "human":
        return "human_worse_label" if dataset == "cifar10" else "human_noisy_label"
    return f"{noise_type}_{ratio:g}"


def _load_baseline_config(path):
    path = Path(path)
    resolved = path if path.is_absolute() else ROOT / path
    with resolved.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    missing = set(STAGE2_BASELINE_METHODS) - set(config)
    if missing:
        raise ValueError(f"Baseline configuration is missing methods: {sorted(missing)}")
    return config


def _feature_sha256(features):
    return hashlib.sha256(features.numpy().tobytes()).hexdigest()


def _result_complete(row):
    try:
        return all(np.isfinite(float(row[name])) for name in REQUIRED_METRICS)
    except (KeyError, TypeError, ValueError):
        return False


def _save_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _resolve_clipcleaner_selection(output_dir, dataset, source_backbone, noise_name, seed,
                                   noisy_labels, source_feature_hash=None):
    root = Path(output_dir) / "baselines" / "clipcleaner" / dataset / source_backbone
    pattern = f"*/{noise_name}/seed_{seed}/predictions.csv"
    candidates = sorted(root.glob(pattern)) if root.exists() else []
    if source_feature_hash:
        candidates = [path for path in candidates if path.parents[2].name.startswith(source_feature_hash)]
    if not candidates:
        expected = root / (source_feature_hash or "<feature_sha256>") / noise_name / f"seed_{seed}" / "predictions.csv"
        raise FileNotFoundError(
            "Missing CLIPCleaner Stage-1 selection. Run or transfer the corresponding Stage-1 "
            f"CLIPCleaner result first. Expected: {expected}"
        )
    if len(candidates) > 1:
        choices = "\n".join(str(path) for path in candidates)
        raise RuntimeError(
            "Multiple CLIPCleaner selections match this run. Pass --clipcleaner_feature_hash "
            f"to choose one:\n{choices}"
        )
    path = candidates[0]
    frame = pd.read_csv(path, usecols=["index", "noisy_label", "predicted_noisy"])
    frame = frame.sort_values("index")
    expected_indices = np.arange(len(noisy_labels))
    if not np.array_equal(frame["index"].to_numpy(), expected_indices):
        raise ValueError(f"CLIPCleaner selection is not in CIFAR sample order: {path}")
    if not np.array_equal(frame["noisy_label"].to_numpy(dtype=np.int64), noisy_labels):
        raise ValueError(
            "CLIPCleaner selection uses different noisy labels from the current experiment: "
            f"{path}"
        )
    values = frame["predicted_noisy"]
    if values.dtype == object:
        values = values.astype(str).str.lower().map({"true": True, "false": False})
    if values.isna().any():
        raise ValueError(f"Invalid predicted_noisy values: {path}")
    selected = np.flatnonzero(~values.to_numpy(dtype=bool))
    if len(selected) == 0:
        raise ValueError(f"CLIPCleaner selected no clean samples: {path}")
    return torch.from_numpy(selected), path


def _summarize(runs):
    groups = ["method", "backbone", "dataset", "noise_type", "noise_rate", "noise_name"]
    return runs.groupby(groups, dropna=False).agg(
        n_runs=("seed", "size"),
        n_train_mean=("n_train", "mean"),
        n_train_std=("n_train", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        ece_raw_mean=("ece_raw", "mean"),
        ece_raw_std=("ece_raw", "std"),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Run one frozen-feature Stage-2 LNL baseline on one CIFAR dataset."
    )
    parser.add_argument("--method", required=True, choices=STAGE2_BASELINE_METHODS)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_META))
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--device")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baseline_config", default="configs/stage2_baselines.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--settings", nargs="+", type=canonical_noise_type,
                        choices=tuple(SETTINGS), default=tuple(SETTINGS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--clipcleaner_source_backbone", default="clip_vit_b16",
                        choices=("clip_vit_b16", "clip_vit_l14"))
    parser.add_argument("--clipcleaner_feature_hash")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    baseline_cfg = _load_baseline_config(args.baseline_config)
    method, dataset = args.method, args.dataset
    backbone = canonical_backbone_name(args.backbone)
    effective_method_config = dict(baseline_cfg[method])
    if method == "clipcleaner":
        effective_method_config.update({
            "source_backbone": canonical_backbone_name(args.clipcleaner_source_backbone),
            "source_feature_hash": args.clipcleaner_feature_hash,
        })
    method_config_json = json.dumps(effective_method_config, sort_keys=True, separators=(",", ":"))
    method_config_sha256 = hashlib.sha256(method_config_json.encode("utf-8")).hexdigest()
    device = get_device(args.device or cfg["device"])
    seeds = tuple(dict.fromkeys(args.seeds or cfg["seeds"]))
    base = load_cifar_base(dataset, cfg["data_root"], download=False)
    clean_labels = np.asarray(base.targets, dtype=np.int64)
    test_base = load_cifar_base(dataset, cfg["data_root"], train=False, download=False)
    official_test_labels = torch.as_tensor(test_base.targets, dtype=torch.long)
    train_path = feature_cache_path(cfg["features_root"], dataset, backbone, split="train")
    test_path = feature_cache_path(cfg["features_root"], dataset, backbone, split="test")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Extract train and test features first: {train_path}, {test_path}")
    train_features = load_features(train_path, clean_labels, expected_backbone=backbone)
    test_features = load_features(test_path, official_test_labels, expected_backbone=backbone)
    feature_hash = _feature_sha256(train_features)
    num_classes = DATASET_META[dataset]["num_classes"]
    output_root = (Path(args.output_dir).resolve() if args.output_dir else
                   Path(cfg["output_dir"]).parent / "stage2_baselines")
    output_path = output_root / method / backbone / dataset / "runs.csv"
    rows = [] if not output_path.exists() else pd.read_csv(output_path).to_dict("records")
    completed = {
        (row["noise_type"], int(row["seed"])) for row in rows
        if _result_complete(row)
        and row.get("protocol") == PROTOCOL
        and row.get("training_config_sha256") == method_config_sha256
    }
    existing = {(row["noise_type"], int(row["seed"])): row for row in rows}
    print(
        f"Loaded {dataset} {backbone}: train={tuple(train_features.shape)}, "
        f"test={tuple(test_features.shape)}, device={device}", flush=True,
    )

    for noise_type in dict.fromkeys(args.settings):
        ratio = SETTINGS[noise_type]
        noise_name = _noise_name(dataset, noise_type, ratio)
        for seed in seeds:
            identity = (noise_type, seed)
            if identity in completed and not args.force:
                print(f"Skipping completed {METHOD_NAMES[method]} {dataset}/{noise_name}/seed_{seed}", flush=True)
                continue
            if identity in existing and not args.force:
                raise RuntimeError(
                    f"Existing result uses an older protocol or different configuration: {output_path} "
                    "Use --force to replace only the requested setting/seed."
                )
            noisy = make_noisy_labels(
                dataset, cfg["data_root"], base.data, clean_labels, num_classes,
                noise_type, ratio, seed, device=device,
            )
            # Clean training labels stop at cache validation/noise construction and are
            # deliberately absent from the baseline trainer interface below.
            selected_indices = None
            selection_path = None
            if method == "clipcleaner":
                source_backbone = canonical_backbone_name(args.clipcleaner_source_backbone)
                selected_indices, selection_path = _resolve_clipcleaner_selection(
                    cfg["output_dir"], dataset, source_backbone, noise_name, seed, noisy,
                    source_feature_hash=args.clipcleaner_feature_hash,
                )
                print(f"Using CLIPCleaner selection: {selection_path}", flush=True)
            result = train_stage2_baseline(
                method, train_features, torch.from_numpy(noisy), test_features,
                official_test_labels, num_classes, dataset, noise_type, ratio, seed,
                device, baseline_cfg, selected_indices=selected_indices,
            )
            row = {
                "protocol": PROTOCOL,
                "method": METHOD_NAMES[method],
                "method_key": method,
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": ratio,
                "noise_name": noise_name,
                "backbone": backbone,
                "feature_dim": train_features.shape[1],
                "feature_sha256": feature_hash,
                "seed": seed,
                "ece_num_bins": ECE_BINS,
                "training_config": method_config_json,
                "training_config_sha256": method_config_sha256,
                "source_selection": (
                    selection_path.relative_to(Path(cfg["output_dir"])).as_posix()
                    if selection_path is not None else None
                ),
                **result.metrics,
            }
            rows = [old for old in rows if (old["noise_type"], int(old["seed"])) != identity]
            rows.append(row)
            completed.add(identity)
            runs = pd.DataFrame(rows).sort_values(["noise_type", "seed"]).reset_index(drop=True)
            _save_frame(runs, output_path)
            artifact_path = output_path.parent / "artifacts" / noise_name / f"seed_{seed}.pt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({**row, **result.artifacts}, artifact_path)
            _save_frame(_summarize(runs), output_path.with_name("summary.csv"))
            print(
                f"{METHOD_NAMES[method]} {dataset}/{noise_name}/seed_{seed}: "
                f"n_train={row['n_train']} Accuracy={row['accuracy']:.6f} "
                f"Macro-F1={row['macro_f1']:.6f} ECE-Raw={row['ece_raw']:.6f}", flush=True,
            )

    runs = pd.DataFrame(rows).sort_values(["noise_type", "seed"]).reset_index(drop=True)
    summary = _summarize(runs)
    _save_frame(summary, output_path.with_name("summary.csv"))
    print(summary.to_string(index=False), flush=True)
    print(f"Runs: {output_path}", flush=True)
    print(f"Summary: {output_path.with_name('summary.csv')}", flush=True)
    try:
        reports = build_ece_reports(output_root.parent)
        print(f"ECE summary: {reports['summary']}", flush=True)
    except ValueError as error:
        print(f"ECE report not refreshed yet: {error}", flush=True)


if __name__ == "__main__":
    main()
