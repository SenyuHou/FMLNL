import hashlib
import json
from pathlib import Path

import pandas as pd

from lnl_foundation.backbones.frozen import canonical_backbone_name


FORMAL_PARTITION_PROTOCOL = "global_local_gmm_pairflip_v1"


def _partition_fingerprint(path, columns):
    frame = pd.read_csv(path, usecols=columns).sort_values("index")
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def resolve_saved_partition(
    root,
    dataset,
    backbone,
    noise_name,
    seed,
    tau_global,
    tau_local,
    knn_k,
    current_feature_sha256=None,
    equivalence_columns=("index", "noisy_label", "partition", "global_margin"),
):
    """Resolve a formal partition without requiring platform-identical feature bytes."""
    root = Path(root)
    backbone = canonical_backbone_name(backbone)
    candidates = []
    for metrics_path in sorted((root / dataset).glob("*/**/metrics.json")):
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        partition_path = metrics_path.with_name("partition.csv")
        if (
            metadata.get("protocol") == FORMAL_PARTITION_PROTOCOL
            and canonical_backbone_name(metadata.get("backbone", "")) == backbone
            and metadata.get("dataset") == dataset
            and metadata.get("noise_name") == noise_name
            and int(metadata.get("seed", -1)) == int(seed)
            and metadata.get("tau_global") == tau_global
            and metadata.get("tau_local") == tau_local
            and metadata.get("knn_k") == knn_k
            and partition_path.exists()
        ):
            candidates.append((partition_path, metadata))

    if not candidates:
        raise RuntimeError(
            "No formal partition matches "
            f"{dataset}/{backbone}/{noise_name}/seed_{seed} with "
            f"g={tau_global}, l={tau_local}, k={knn_k}."
        )

    exact = [
        item for item in candidates
        if current_feature_sha256 and item[1].get("feature_sha256") == current_feature_sha256
    ]
    pool = exact or candidates
    match_type = "exact_feature_hash" if exact else "compatible_backbone"
    if len(pool) > 1:
        fingerprints = {
            _partition_fingerprint(path, equivalence_columns) for path, _ in pool
        }
        if len(fingerprints) != 1:
            variants = [
                f"{metadata.get('feature_sha256', 'unknown')[:12]}:{path}"
                for path, metadata in pool
            ]
            raise RuntimeError(
                "Multiple non-equivalent partitions match the same experiment identity. "
                "Keep only the intended variant or restore its matching feature cache: "
                + "; ".join(variants)
            )
    partition_path, metadata = pool[0]
    return partition_path, metadata, match_type
