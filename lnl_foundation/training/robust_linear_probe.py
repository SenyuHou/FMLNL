from dataclasses import dataclass
import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from lnl_foundation.partition.global_local_gmm import CLEAN, HARD, NOISY
from lnl_foundation.training.calibration import evaluate_logits, reliability_anchored_temperature
from lnl_foundation.utils import set_seed


GCE_Q = 0.7
PROTOTYPE_TEMPERATURE = 0.1


@dataclass(frozen=True)
class LinearProbeConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 256


@torch.no_grad()
def _collect_logits(classifier, features, batch_size):
    classifier.eval()
    return torch.cat([
        classifier(features[start:start + batch_size]).float().cpu()
        for start in range(0, len(features), batch_size)
    ])


def build_prototype_targets(features, noisy_labels, partition, num_classes):
    """Build targets using predicted Clean samples and noisy labels only."""
    prototypes = []
    for class_id in range(num_classes):
        selected = (partition == CLEAN) & (noisy_labels == class_id)
        if not selected.any():
            warnings.warn(
                f"No Clean samples for noisy-label class {class_id}; using all samples in that class.",
                stacklevel=2,
            )
            selected = noisy_labels == class_id
        if not selected.any():
            raise ValueError(f"No samples available for noisy-label class {class_id}.")
        prototypes.append(F.normalize(features[selected].mean(dim=0), dim=0))
    prototypes = torch.stack(prototypes)
    q_proto = F.softmax(features @ prototypes.T / PROTOTYPE_TEMPERATURE, dim=1)
    entropy = -(q_proto * q_proto.clamp_min(torch.finfo(q_proto.dtype).tiny).log()).sum(dim=1)
    confidence = (1.0 - entropy / math.log(num_classes)).clamp(0.0, 1.0)

    if not all(torch.isfinite(value).all() for value in (prototypes, q_proto, confidence)):
        raise ValueError("Prototype, target, or confidence contains NaN or Inf.")
    if not torch.allclose(q_proto.sum(dim=1), torch.ones(len(q_proto)), atol=1e-6, rtol=0):
        raise ValueError("Prototype targets are not normalized.")
    return prototypes, q_proto, confidence


def _fixed_stage2_loss(logits, noisy_labels, partition, q_proto, confidence):
    clean = partition == CLEAN
    hard = partition == HARD
    noisy = partition == NOISY
    numerator = logits.new_zeros(())
    denominator = logits.new_zeros(())
    if clean.any():
        numerator = numerator + F.cross_entropy(logits[clean], noisy_labels[clean], reduction="sum")
        denominator = denominator + clean.sum()
    if hard.any():
        probability = F.softmax(logits[hard], dim=1)
        p_y = probability.gather(1, noisy_labels[hard, None]).squeeze(1).clamp_min(1e-12)
        numerator = numerator + ((1.0 - p_y.pow(GCE_Q)) / GCE_Q).sum()
        denominator = denominator + hard.sum()
    if noisy.any():
        soft_ce = -(q_proto[noisy] * F.log_softmax(logits[noisy], dim=1)).sum(dim=1)
        numerator = numerator + (confidence[noisy] * soft_ce).sum()
        denominator = denominator + confidence[noisy].sum()
    loss = numerator / (denominator + 1e-12)
    if not torch.isfinite(loss):
        raise ValueError("Linear-probe loss is NaN or Inf.")
    return loss


def train_robust_linear_probe(
    train_features,
    noisy_labels,
    partition,
    test_features,
    test_labels,
    num_classes,
    seed,
    device,
    config=LinearProbeConfig(),
    global_margin=None,
):
    """Train Clean-CE + Hard-GCE + Noisy-Prototype-SoftCE."""
    if train_features.ndim != 2 or test_features.shape[-1] != train_features.shape[-1]:
        raise ValueError("Train and test feature dimensions must match.")
    if len(train_features) != len(noisy_labels) or len(partition) != len(noisy_labels):
        raise ValueError("Training features, labels, and partition lengths must match.")
    if not torch.isin(partition, torch.tensor([CLEAN, HARD, NOISY])).all():
        raise ValueError("Partition contains an unknown state.")

    _, q_proto, confidence = build_prototype_targets(
        train_features, noisy_labels, partition, num_classes
    )
    generator = torch.Generator().manual_seed(seed)
    epoch_orders = [torch.randperm(len(train_features), generator=generator) for _ in range(config.epochs)]
    set_seed(seed)
    classifier = nn.Linear(train_features.shape[-1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_features = train_features.to(device)
    noisy_labels = noisy_labels.to(device)
    partition = partition.to(device)
    q_proto = q_proto.to(device)
    confidence = confidence.to(device)
    test_features = test_features.to(device)

    progress = tqdm(epoch_orders, desc=f"Linear probe seed={seed}", unit="epoch", dynamic_ncols=True)
    for cpu_order in progress:
        classifier.train()
        order = cpu_order.to(device)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            logits = classifier(train_features[index])
            loss = _fixed_stage2_loss(
                logits,
                noisy_labels[index],
                partition[index],
                q_proto[index],
                confidence[index],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        progress.set_postfix(loss=f"{epoch_loss / num_batches:.4f}")

    if global_margin is None:
        raise ValueError("Ours calibration requires saved first-stage global margins.")
    train_logits = _collect_logits(classifier, train_features, config.batch_size)
    test_logits = _collect_logits(classifier, test_features, config.batch_size)
    calibration = reliability_anchored_temperature(
        train_logits,
        noisy_labels.cpu(),
        partition.cpu(),
        global_margin,
        num_classes,
    )
    raw = evaluate_logits(test_logits, test_labels, temperature=1.0)
    calibrated = evaluate_logits(
        test_logits, test_labels, temperature=calibration["temperature"]
    )
    if raw["accuracy"] != calibrated["accuracy"] or raw["macro_f1"] != calibrated["macro_f1"]:
        raise ValueError("Temperature scaling changed Accuracy or Macro-F1.")
    return {
        "accuracy": raw["accuracy"],
        "macro_f1": raw["macro_f1"],
        "ece_raw": raw["ece"],
        "temperature": calibration["temperature"],
        "ece_calibrated": calibrated["ece"],
        "n_calibration_anchors": calibration["n_calibration_anchors"],
        "anchor_mean_confidence": calibration["anchor_mean_confidence"],
        "anchor_top_fraction": calibration["anchor_top_fraction"],
        "anchor_target_confidence": calibration["anchor_target_confidence"],
        "_artifacts": {
            "test_logits": test_logits,
            "test_probabilities_raw": raw["probabilities"],
            "test_probabilities_calibrated": calibrated["probabilities"],
            "anchor_indices": calibration["anchor_indices"],
            "anchor_candidate_counts": calibration["anchor_candidate_counts"],
            "anchor_selected_counts": calibration["anchor_selected_counts"],
        },
    }


def train_clean_linear_probe(
    train_features,
    clean_labels,
    test_features,
    test_labels,
    num_classes,
    seed,
    device,
    config=LinearProbeConfig(),
):
    """Train the Clean-LP reference with ordinary CE on ground-truth labels."""
    if train_features.ndim != 2 or test_features.shape[-1] != train_features.shape[-1]:
        raise ValueError("Train and test feature dimensions must match.")
    if len(train_features) != len(clean_labels) or len(test_features) != len(test_labels):
        raise ValueError("Features and clean labels must have matching lengths.")
    if not all(torch.isfinite(value).all() for value in (train_features, test_features)):
        raise ValueError("Clean-LP features contain NaN or Inf.")
    if clean_labels.min() < 0 or clean_labels.max() >= num_classes:
        raise ValueError("Clean-LP training labels are outside the class range.")
    if test_labels.min() < 0 or test_labels.max() >= num_classes:
        raise ValueError("Clean-LP test labels are outside the class range.")

    generator = torch.Generator().manual_seed(seed)
    epoch_orders = [torch.randperm(len(train_features), generator=generator) for _ in range(config.epochs)]
    set_seed(seed)
    classifier = nn.Linear(train_features.shape[-1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_features = train_features.to(device)
    clean_labels = clean_labels.to(device)
    test_features = test_features.to(device)

    progress = tqdm(epoch_orders, desc=f"Clean-LP seed={seed}", unit="epoch", dynamic_ncols=True)
    for cpu_order in progress:
        classifier.train()
        order = cpu_order.to(device)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            logits = classifier(train_features[index])
            loss = F.cross_entropy(logits, clean_labels[index])
            if not torch.isfinite(loss):
                raise ValueError("Clean-LP loss is NaN or Inf.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        progress.set_postfix(loss=f"{epoch_loss / num_batches:.4f}")

    test_logits = _collect_logits(classifier, test_features, config.batch_size)
    raw = evaluate_logits(test_logits, test_labels, temperature=1.0)
    return {
        "accuracy": raw["accuracy"],
        "macro_f1": raw["macro_f1"],
        "ece_raw": raw["ece"],
        "_artifacts": {
            "test_logits": test_logits,
            "test_probabilities_raw": raw["probabilities"],
        },
    }
