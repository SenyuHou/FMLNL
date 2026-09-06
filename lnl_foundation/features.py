from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lnl_foundation.backbones.frozen import BACKBONE_ALIASES, canonical_backbone_name


def feature_cache_candidates(root, dataset, backbone, split="train") -> tuple[Path, ...]:
    canonical = canonical_backbone_name(backbone)
    candidates = [Path(root) / dataset / canonical / f"{split}_l2.pt"]
    for alias, target in BACKBONE_ALIASES.items():
        if target == canonical:
            candidates.append(Path(root) / dataset / alias / f"{split}_l2.pt")
    return tuple(candidates)


def feature_cache_path(root, dataset, backbone, split="train", prefer_existing=True) -> Path:
    candidates = feature_cache_candidates(root, dataset, backbone, split)
    if prefer_existing:
        for path in candidates:
            if path.exists():
                return path
    return candidates[0]


def load_features(path, clean_labels, expected_backbone=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload["features"].float()
    labels = torch.as_tensor(clean_labels, dtype=torch.long)
    cached_dim = payload.get("feature_dim")
    cached_backbone = payload.get("backbone")
    valid = (
        features.ndim == 2
        and len(features) == len(labels)
        and torch.isfinite(features).all()
        and torch.equal(payload["indices"], torch.arange(len(labels)))
        and torch.equal(payload["clean_labels"], labels)
        and payload.get("normalized", False)
        and (cached_dim is None or int(cached_dim) == features.shape[-1])
    )
    if expected_backbone is not None:
        expected = canonical_backbone_name(expected_backbone)
        if cached_backbone is not None:
            valid = valid and canonical_backbone_name(cached_backbone) == expected
        else:
            legacy_name = Path(path).parent.name
            valid = valid and BACKBONE_ALIASES.get(legacy_name) == expected
    if not valid:
        raise ValueError(f"Invalid feature cache or CIFAR sample order: {path}")
    return features


@torch.no_grad()
def extract_features(backbone, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    backbone.to(device).eval()
    all_features, labels = [], []
    for images, clean in tqdm(loader, desc="Extracting features"):
        raw = backbone(images.to(device, non_blocking=True))
        if not torch.isfinite(raw).all():
            raise ValueError("Backbone output contains NaN or Inf values.")
        all_features.append(F.normalize(raw, dim=1).cpu())
        labels.append(clean)
    features = torch.cat(all_features)
    return {
        "features": features,
        "clean_labels": torch.cat(labels).long(),
        "indices": torch.arange(len(dataset)),
        "normalized": True,
        "backbone": backbone.model_name,
        "feature_dim": int(features.shape[-1]),
        "checkpoint": backbone.spec.checkpoint,
        "preprocess_source": backbone.preprocessing_source,
        "preprocess_config": backbone.preprocess_config,
    }
