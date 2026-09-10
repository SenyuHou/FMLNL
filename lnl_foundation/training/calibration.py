import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from lnl_foundation.partition.global_local_gmm import CLEAN


ECE_BINS = 15
ANCHOR_TOP_FRACTION = 0.1
ANCHOR_TARGET_CONFIDENCE = 0.995


def expected_calibration_error(probabilities, labels, num_bins=ECE_BINS):
    """Top-label ECE with equal-width confidence bins on [0, 1]."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must have compatible shapes.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or Inf.")
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).to(torch.float64)
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64)
    bin_index = torch.bucketize(confidence, boundaries[1:-1])
    ece = torch.zeros((), dtype=torch.float64)
    for index in range(num_bins):
        selected = bin_index == index
        if selected.any():
            weight = selected.to(torch.float64).mean()
            ece += weight * (correct[selected].mean() - confidence[selected].mean()).abs()
    return float(ece)


def evaluate_logits(logits, labels, temperature=1.0):
    logits = torch.as_tensor(logits, dtype=torch.float32).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    temperature = float(temperature)
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive.")
    probabilities = F.softmax(logits / temperature, dim=1)
    predictions = probabilities.argmax(dim=1)
    accuracy = float(predictions.eq(labels).float().mean())
    macro_f1 = float(f1_score(labels.numpy(), predictions.numpy(), average="macro"))
    ece = expected_calibration_error(probabilities, labels)
    if not np.isfinite([accuracy, macro_f1, ece]).all():
        raise ValueError("Evaluation metrics contain NaN or Inf.")
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "probabilities": probabilities,
    }


def select_reliability_anchors(
    train_logits,
    noisy_labels,
    partition,
    global_margin,
    num_classes,
    top_fraction=ANCHOR_TOP_FRACTION,
):
    """Select per-class anchors without accepting ground-truth training labels."""
    logits = torch.as_tensor(train_logits, dtype=torch.float32).cpu()
    noisy_labels = torch.as_tensor(noisy_labels, dtype=torch.long).cpu()
    partition = torch.as_tensor(partition, dtype=torch.long).cpu()
    global_margin = torch.as_tensor(global_margin, dtype=torch.float32).cpu()
    if not 0 < top_fraction <= 1:
        raise ValueError("Anchor top fraction must be in (0, 1].")
    if not (len(logits) == len(noisy_labels) == len(partition) == len(global_margin)):
        raise ValueError("Anchor inputs must have matching lengths.")
    if not torch.isfinite(logits).all() or not torch.isfinite(global_margin).all():
        raise ValueError("Anchor logits or global margins contain NaN or Inf.")

    prediction = logits.argmax(dim=1)
    selected_parts = []
    candidate_counts = []
    selected_counts = []
    minimum_for_ranking = math.ceil(1.0 / top_fraction)
    for class_id in range(num_classes):
        candidates = torch.where(
            (partition == CLEAN)
            & (noisy_labels == class_id)
            & (prediction == class_id)
        )[0]
        candidate_counts.append(int(len(candidates)))
        if len(candidates) == 0:
            selected = candidates
        elif len(candidates) <= minimum_for_ranking:
            selected = candidates
        else:
            count = max(1, math.ceil(top_fraction * len(candidates)))
            order = torch.topk(global_margin[candidates], count, largest=True).indices
            selected = candidates[order]
        selected_parts.append(selected)
        selected_counts.append(int(len(selected)))
    anchors = torch.cat(selected_parts) if selected_parts else torch.empty(0, dtype=torch.long)
    if len(anchors) == 0:
        raise ValueError("No reliability anchors were selected.")
    return anchors, candidate_counts, selected_counts


def fit_anchor_temperature(
    anchor_logits,
    target_confidence=ANCHOR_TARGET_CONFIDENCE,
    max_iter=100,
):
    """Fit one positive scalar T to the anchor confidence target."""
    logits = torch.as_tensor(anchor_logits, dtype=torch.float64).cpu()
    if logits.ndim != 2 or len(logits) == 0 or not torch.isfinite(logits).all():
        raise ValueError("Anchor logits must be a finite non-empty matrix.")
    if not 0 < target_confidence < 1:
        raise ValueError("Target confidence must be in (0, 1).")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        mean_confidence = F.softmax(logits / temperature, dim=1).max(dim=1).values.mean()
        loss = (mean_confidence - target_confidence).square()
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    mean_confidence = float(F.softmax(logits / temperature, dim=1).max(dim=1).values.mean())
    if not np.isfinite([temperature, mean_confidence]).all() or temperature <= 0:
        raise ValueError("Temperature optimization produced an invalid result.")
    return temperature, mean_confidence


def reliability_anchored_temperature(
    train_logits,
    noisy_labels,
    partition,
    global_margin,
    num_classes,
):
    anchors, candidate_counts, selected_counts = select_reliability_anchors(
        train_logits,
        noisy_labels,
        partition,
        global_margin,
        num_classes,
    )
    temperature, anchor_mean_confidence = fit_anchor_temperature(train_logits[anchors])
    return {
        "temperature": temperature,
        "anchor_indices": anchors,
        "anchor_candidate_counts": candidate_counts,
        "anchor_selected_counts": selected_counts,
        "n_calibration_anchors": int(len(anchors)),
        "anchor_mean_confidence": anchor_mean_confidence,
        "anchor_top_fraction": ANCHOR_TOP_FRACTION,
        "anchor_target_confidence": ANCHOR_TARGET_CONFIDENCE,
    }
