import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device="auto") -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def load_config(path, overrides=()) -> dict:
    path = Path(path)
    with (path if path.is_absolute() else ROOT / path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or ():
        key, value = item.split("=", 1)
        if key not in cfg:
            raise ValueError(f"Unknown configuration key: {key}")
        cfg[key] = yaml.safe_load(value)
    for key in ("data_root", "features_root", "output_dir"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    cfg["device"] = str(get_device(cfg["device"]))
    return cfg


def save_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
