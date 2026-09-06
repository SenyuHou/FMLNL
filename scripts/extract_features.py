import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.backbones.frozen import FrozenBackbone
from lnl_foundation.data.datasets import load_cifar_base
from lnl_foundation.features import extract_features, feature_cache_path, load_features
from lnl_foundation.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Extract frozen pretrained vision features once per dataset.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train"])
    parser.add_argument("--force_extract", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if cfg["hf_endpoint"]:
        os.environ["HF_ENDPOINT"] = cfg["hf_endpoint"].rstrip("/")
    set_seed(1)
    backbone = None
    for split in dict.fromkeys(args.splits):
        base = load_cifar_base(args.dataset, cfg["data_root"], train=split == "train", download=True)
        path = feature_cache_path(
            cfg["features_root"], args.dataset, cfg["backbone"], split,
            prefer_existing=not args.force_extract,
        )
        if path.exists() and not args.force_extract:
            features = load_features(path, base.targets, expected_backbone=cfg["backbone"])
            print(f"Reusing {split}: {tuple(features.shape)} -> {path}", flush=True)
            continue
        if backbone is None:
            backbone = FrozenBackbone(cfg["backbone"], pretrained=True)
        dataset = load_cifar_base(
            args.dataset,
            cfg["data_root"],
            train=split == "train",
            transform=backbone.preprocess,
            download=False,
        )
        payload = extract_features(backbone, dataset, cfg["batch_size"], cfg["num_workers"], torch.device(cfg["device"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"Saved {split}: {tuple(payload['features'].shape)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
