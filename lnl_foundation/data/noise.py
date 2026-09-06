from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torchvision import transforms
from tqdm import tqdm

from lnl_foundation.utils import set_seed


HUMAN_CHOICES = (
    "human_worse_label", "human_aggre_label", "human_random_label1",
    "human_random_label2", "human_random_label3", "human_noisy_label",
)


def canonical_noise_type(name: str) -> str:
    name = name.lower()
    return {
        "sym": "symmetric", "asym": "pairflip", "asymmetric": "pairflip",
        "pair": "pairflip", "idn": "instance", "inst": "instance",
        "instance-dependent": "instance",
    }.get(name, name)


def build_transition(num_classes: int, ratio: float, noise_type: str) -> np.ndarray:
    if num_classes < 2 or not 0 <= ratio <= 1:
        raise ValueError("Require num_classes >= 2 and noise ratio in [0, 1].")
    noise_type = canonical_noise_type(noise_type)
    if noise_type == "symmetric":
        transition = np.full((num_classes, num_classes), ratio / (num_classes - 1))
    elif noise_type == "pairflip":
        transition = np.zeros((num_classes, num_classes))
        transition[np.arange(num_classes), (np.arange(num_classes) + 1) % num_classes] = ratio
    else:
        raise ValueError(f"No transition matrix for {noise_type}.")
    np.fill_diagonal(transition, 1 - ratio)
    return transition


def generate_instance_noise_labels(
    data, targets, transform, num_classes, tau=0.2, std=0.1,
    feature_size=3 * 32 * 32, seed=42, device="cuda",
):
    """JYP matrix IDN: preserve its RNG order, pixel transform and sampling."""
    if not 0 <= tau <= 1 or std <= 0:
        raise ValueError("Require tau in [0, 1] and std > 0.")
    targets = torch.as_tensor(targets, dtype=torch.long).cpu()
    if seed is not None:
        set_seed(seed)
    device = torch.device(device)
    flip_distribution = stats.truncnorm((0 - tau) / std, (1 - tau) / std, loc=tau, scale=std)
    q = flip_distribution.rvs(len(targets))
    weights = torch.tensor(np.random.randn(num_classes, feature_size, num_classes)).float().to(device)
    prob_rows = []
    for i in tqdm(range(len(targets)), desc="Generating instance noise"):
        x = transform(Image.fromarray(data[i])).to(device)
        y = int(targets[i])
        p = x.reshape(1, -1).mm(weights[y]).squeeze(0)
        p[y] = -float("inf")
        p = q[i] * F.softmax(p, dim=0)
        p[y] += 1 - q[i]
        prob_rows.append(p)
    probs = torch.stack(prob_rows).cpu().numpy()
    return np.asarray([np.random.choice(num_classes, p=p) for p in probs], dtype=np.int64)


def human_noise_key(dataset: str, noise_type: str | None = None) -> str:
    name = noise_type or ("human_worse_label" if dataset == "cifar10" else "human_noisy_label")
    if name not in HUMAN_CHOICES:
        raise ValueError(f"Unknown human noise option: {name}")
    if dataset == "cifar100" and name != "human_noisy_label":
        raise ValueError("CIFAR-100 only supports human_noisy_label, matching JYP.")
    return name.removeprefix("human_")


def make_noisy_labels(
    dataset, data_root, images, clean_labels, num_classes,
    noise_type, noise_ratio, seed, device="cpu", human_noise_type=None,
) -> np.ndarray:
    """Generate synthetic noise on every call; only Human reads a label file."""
    clean = np.asarray(clean_labels, dtype=np.int64)
    noise_type = canonical_noise_type(noise_type)
    if noise_type == "human":
        key = human_noise_key(dataset, human_noise_type)
        filename = "CIFAR-10_human.pt" if dataset == "cifar10" else "CIFAR-100_human.pt"
        path = Path(data_root) / "noise_label_human" / filename
        # Original CIFAR-N annotations may contain NumPy arrays; keep JYP compatibility.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if key not in payload:
            raise ValueError(f"Missing Human key {key!r} in {path}; available: {list(payload)}")
        if not np.array_equal(np.asarray(payload["clean_label"]), clean):
            raise ValueError("Human clean labels do not match CIFAR sample order.")
        noisy = np.asarray(payload[key], dtype=np.int64)
    elif noise_type in {"symmetric", "pairflip"}:
        transition = build_transition(num_classes, noise_ratio, noise_type)
        set_seed(seed)
        noisy = np.asarray([np.random.choice(num_classes, p=transition[y]) for y in clean])
    elif noise_type == "instance":
        if not 0 <= noise_ratio <= 1:
            raise ValueError("Noise ratio must be in [0, 1].")
        if noise_ratio == 0:
            return clean.copy()
        noisy = generate_instance_noise_labels(
            images, clean,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            num_classes, tau=noise_ratio, std=0.1, seed=seed, device=device,
        )
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    if noisy.shape != clean.shape or ((noisy < 0) | (noisy >= num_classes)).any():
        raise ValueError("Noisy labels have an invalid shape or class index.")
    return noisy.astype(np.int64)
