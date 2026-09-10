from pathlib import Path

import numpy as np
import pandas as pd


NOISE_NAMES = {
    "cifar10": ("human_worse_label", "symmetric_0.6", "pairflip_0.3", "instance_0.4"),
    "cifar100": ("human_noisy_label", "symmetric_0.6", "pairflip_0.3", "instance_0.4"),
}
SETTING_LABELS = {
    "human_worse_label": "Human",
    "human_noisy_label": "Human",
    "symmetric_0.6": "Symm0.6",
    "pairflip_0.3": "Pairflip0.3",
    "instance_0.4": "Inst0.4",
}
RUN_COLUMNS = (
    "method", "backbone", "dataset", "noise_name", "seed", "ece_type", "ece", "source_file"
)


def _read_runs(path, ours, source_file):
    frame = pd.read_csv(path)
    required = {"backbone", "dataset", "seed", "ece_raw"}
    missing = required - set(frame.columns)
    if ours and "ece_calibrated" not in frame:
        missing.add("ece_calibrated")
    if missing:
        raise ValueError(f"ECE source is missing columns {sorted(missing)}: {path}")
    if not ours and "method" not in frame:
        raise ValueError(f"Baseline ECE source must include method: {path}")
    if "ece_num_bins" in frame:
        bins = frame["ece_num_bins"].dropna().astype(int).unique()
        if len(bins) and set(bins) != {15}:
            raise ValueError(f"ECE source does not use 15 equal-width bins: {path}")

    records = []
    for row in frame.to_dict("records"):
        method = "Ours" if ours else str(row["method"])
        noise_names = [row.get("noise_name")]
        if pd.isna(noise_names[0]) or not noise_names[0]:
            noise_names = NOISE_NAMES.get(row["dataset"], ())
        for noise_name in noise_names:
            records.append({
                "method": method,
                "backbone": row["backbone"],
                "dataset": row["dataset"],
                "noise_name": noise_name,
                "seed": int(row["seed"]),
                "ece_type": "Raw",
                "ece": float(row["ece_raw"]),
                "source_file": source_file,
            })
            if ours:
                records.append({
                    "method": method,
                    "backbone": row["backbone"],
                    "dataset": row["dataset"],
                    "noise_name": noise_name,
                    "seed": int(row["seed"]),
                    "ece_type": "Calibrated",
                    "ece": float(row["ece_calibrated"]),
                    "source_file": source_file,
                })
    return records


def _validate_runs(runs):
    if runs.empty:
        raise ValueError("No ECE results were found.")
    if not np.isfinite(runs["ece"]).all() or ((runs["ece"] < 0) | (runs["ece"] > 1)).any():
        raise ValueError("ECE values must be finite and in [0, 1].")
    keys = ["method", "backbone", "dataset", "noise_name", "seed", "ece_type"]
    duplicates = runs.duplicated(keys, keep=False)
    if duplicates.any():
        duplicate_rows = runs.loc[duplicates, keys + ["source_file"]]
        raise ValueError(f"Duplicate ECE experiment identities:\n{duplicate_rows.to_string(index=False)}")

    ours = runs[runs["method"] == "Ours"]
    expected = {
        (backbone, dataset, noise_name, seed, ece_type)
        for backbone in ours["backbone"].unique()
        for dataset, names in NOISE_NAMES.items()
        for noise_name in names
        for seed in (1, 2, 3)
        for ece_type in ("Raw", "Calibrated")
    }
    actual = set(ours[["backbone", "dataset", "noise_name", "seed", "ece_type"]].itertuples(
        index=False, name=None
    ))
    if actual != expected:
        raise ValueError(
            f"Incomplete Ours ECE results: missing={len(expected - actual)}, extra={len(actual - expected)}"
        )


def _save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_ece_reports(outputs_root="outputs"):
    """Build raw, tidy mean/std, and publication-style ECE tables."""
    outputs_root = Path(outputs_root)
    records = []
    ours_paths = sorted((outputs_root / "stage2").glob("*/cifar*/runs.csv"))
    for path in ours_paths:
        records.extend(_read_runs(path, ours=True, source_file=path.relative_to(outputs_root).as_posix()))
    for path in sorted((outputs_root / "stage2_baselines").glob("**/runs.csv")):
        records.extend(_read_runs(path, ours=False, source_file=path.relative_to(outputs_root).as_posix()))

    clean_lp_path = outputs_root / "clean_lp" / "dinov2_clean_lp_raw.csv"
    if clean_lp_path.exists() and "ece_raw" in pd.read_csv(clean_lp_path, nrows=0).columns:
        records.extend(_read_runs(
            clean_lp_path,
            ours=False,
            source_file=clean_lp_path.relative_to(outputs_root).as_posix(),
        ))

    runs = pd.DataFrame(records, columns=RUN_COLUMNS)
    _validate_runs(runs)
    runs["_ece_order"] = runs["ece_type"].map({"Raw": 0, "Calibrated": 1})
    runs = runs.sort_values(
        ["method", "backbone", "dataset", "noise_name", "_ece_order", "seed"]
    ).drop(columns="_ece_order").reset_index(drop=True)
    summary = runs.groupby(
        ["method", "backbone", "dataset", "noise_name", "ece_type"], dropna=False
    ).agg(
        n_runs=("seed", "size"),
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
    ).reset_index()
    summary["ece_mean_percent"] = summary["ece_mean"] * 100
    summary["ece_std_pp"] = summary["ece_std"] * 100
    summary["_ece_order"] = summary["ece_type"].map({"Raw": 0, "Calibrated": 1})
    summary = summary.sort_values(
        ["method", "backbone", "dataset", "noise_name", "_ece_order"]
    ).drop(columns="_ece_order").reset_index(drop=True)

    columns = [
        f"{dataset.upper()}-{SETTING_LABELS[name]}"
        for dataset, names in NOISE_NAMES.items()
        for name in names
    ]
    table_rows = []
    for (method, backbone, ece_type), frame in summary.groupby(
        ["method", "backbone", "ece_type"], sort=True
    ):
        row = {"Method": method, "Backbone": backbone, "ECE": ece_type}
        for item in frame.itertuples(index=False):
            column = f"{item.dataset.upper()}-{SETTING_LABELS[item.noise_name]}"
            std = "NA" if pd.isna(item.ece_std_pp) else f"{item.ece_std_pp:.2f}"
            row[column] = f"{item.ece_mean_percent:.2f} +/- {std}"
        table_rows.append(row)
    table = pd.DataFrame(table_rows).reindex(columns=["Method", "Backbone", "ECE", *columns])
    table["_ece_order"] = table["ECE"].map({"Raw": 0, "Calibrated": 1})
    table = table.sort_values(["Method", "Backbone", "_ece_order"]).drop(columns="_ece_order")

    destination = outputs_root / "ece"
    _save_csv(runs, destination / "ece_runs.csv")
    _save_csv(summary, destination / "ece_summary.csv")
    _save_csv(table, destination / "ece_table.csv")
    return {
        "runs": destination / "ece_runs.csv",
        "summary": destination / "ece_summary.csv",
        "table": destination / "ece_table.csv",
        "n_runs": len(runs),
        "n_groups": len(summary),
    }
