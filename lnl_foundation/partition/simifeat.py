"""Fixed-feature adaptation of UCSC-REAL/SimiFeat (Zhu, Dong, Liu, ICML 2022).

Based on main_fast.py, hoc.py and utils.py, licensed CC BY-NC 4.0:
https://github.com/UCSC-REAL/SimiFeat
Changes: bounded-memory kNN, vectorized HOC moments, shared geometry across
noise settings, separate seeded sampling/jitter streams, no image augmentation.
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def nearest(features, k, block_size=512):
    """Exact cosine neighbors, including the first/self match as in upstream."""
    z = F.normalize(features, dim=1)
    all_indices, all_distances = [], []
    for start in range(0, len(z), block_size):
        distance = 1 - z[start:start + block_size] @ z.T
        values, indices = distance.topk(k, dim=1, largest=False, sorted=True)
        all_indices.append(indices.cpu())
        all_distances.append(values.cpu())
    return torch.cat(all_indices), torch.cat(all_distances)


def neighbor_distribution(indices, distances, labels, classes):
    # Preserve upstream's unusual first-neighbor distance correction.
    distance = distances.clone()
    distance[:, 0] = 2 * distance[:, 1] - distance[:, 2]
    neighbor_labels = labels[indices]
    counts = torch.stack([((1 - distance) * (neighbor_labels == c)).sum(1) for c in range(classes)], dim=1)
    return F.normalize(counts, p=2, dim=1)


def empirical_moments(labels, neighbor_indices, classes):
    y = np.asarray(labels, dtype=np.int64)
    b, c = y[neighbor_indices[:, 1]], y[neighbor_indices[:, 2]]
    n = len(y)
    return [
        torch.from_numpy(np.bincount(y, minlength=classes).astype(np.float32) / n),
        torch.from_numpy(np.bincount(y * classes + b, minlength=classes ** 2).reshape(classes, classes).astype(np.float32) / n),
        torch.from_numpy(np.bincount((y * classes + b) * classes + c, minlength=classes ** 3).reshape(classes, classes, classes).astype(np.float32) / n),
    ]


def model_moments(transition, prior):
    classes = len(transition)
    weighted = transition * prior.reshape(-1, 1)
    pair = weighted[:, :, None] * transition[:, None, :]
    return [weighted.sum(0), transition.T @ weighted,
            (pair.reshape(classes, classes * classes).T @ transition).reshape(classes, classes, classes)]


def solve_hoc(moments, classes, device, state=None, steps=400):
    if state is None:
        transition = 5 * torch.eye(classes) - torch.ones(classes, classes)
        prior = torch.ones(classes, 1) / classes + torch.rand(classes, 1) * 0.1
    else:
        transition, prior = state
    transition = transition.detach().to(device).clone().requires_grad_()
    prior = prior.detach().to(device).clone().requires_grad_()
    observed = [value.to(device) for value in moments]
    optimizer = torch.optim.Adam([transition, prior], lr=0.1 if state is None else 0.01)
    # Upstream's detached checkpoint aliases live parameters, returning the final
    # iterate. Keep that behavior, including steps-1 optimizer updates.
    for step in range(steps):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        predicted = model_moments(transition.softmax(1), prior.softmax(0))
        loss = sum(torch.norm(a - b) for a, b in zip(observed, predicted))
    if not torch.isfinite(loss):
        raise ValueError("Non-finite HOC objective.")
    state = (transition.detach(), prior.detach())
    return transition.softmax(1).detach().cpu().numpy(), prior.softmax(0).detach().cpu().numpy(), state, float(loss.detach())


def rank_detection(score, labels, transition, prior, jitter):
    classes = len(transition)
    noisy_prior = np.bincount(labels, minlength=classes) / len(labels)
    # Only the diagonal of upstream's Bayes matrix is used by rank1.
    clean_probability = np.diag(transition) * prior.reshape(-1) / noisy_prior
    clean_probability += jitter
    noise_rate = 1 - np.minimum(clean_probability, 1.0)
    noise_rate = np.where(noise_rate >= 1, 0.95, np.where(noise_rate <= 0, 0.05, noise_rate))
    detected = np.zeros(len(labels), dtype=bool)
    for c in range(classes):
        subset = labels == c
        if subset.any():
            threshold = np.percentile(score[subset], 100 * (1 - noise_rate[c]))
            detected[subset] = score[subset] >= threshold
    return detected, noise_rate


def detect_settings(features, label_sets, classes, seed, neighbors, device,
                    epochs=21, trials=10, sample_size=15000, first_steps=400, warm_steps=20):
    """No clean labels or true noise rates are accepted by this detector."""
    torch.manual_seed(seed)
    sampling = np.random.RandomState(seed)
    jitters = {name: np.random.RandomState(seed + 10000) for name in label_sets}
    states = {name: None for name in label_sets}
    votes = {name: np.zeros(len(features), dtype=np.int32) for name in label_sets}
    scores, result = {}, {}
    for name, labels in label_sets.items():
        distribution = neighbor_distribution(*neighbors, torch.from_numpy(labels), classes)
        scores[name] = -torch.log(distribution[torch.arange(len(labels)), labels] + 1e-8).numpy()
        if not np.isfinite(scores[name]).all():
            raise ValueError("Invalid neighborhood score.")
        result[name] = {"vote": distribution.argmax(1).numpy() != labels, "trace": []}
    geometry = features.to(device)
    for epoch in range(epochs):
        moments = {name: [torch.zeros(classes), torch.zeros(classes, classes), torch.zeros(classes, classes, classes)] for name in label_sets}
        for _ in range(trials):
            subset = sampling.choice(len(features), min(sample_size, len(features)), replace=False)
            indices, _ = nearest(geometry[torch.from_numpy(subset).to(device)], 3)
            for name, labels in label_sets.items():
                measured = empirical_moments(labels[subset], indices.numpy(), classes)
                for order in range(3):
                    moments[name][order] += measured[order]
        for name, labels in label_sets.items():
            measured = [value / trials for value in moments[name]]
            # Reinitialize the same seed for each setting's first HOC fit, so
            # batching settings does not change its initialization.
            if states[name] is None:
                torch.manual_seed(seed)
            transition, prior, states[name], loss = solve_hoc(
                measured, classes, device, states[name], first_steps if epoch == 0 else warm_steps)
            predicted, rate = rank_detection(scores[name], labels, transition, prior, jitters[name].uniform(-0.05, 0.05, classes))
            votes[name] += predicted
            result[name]["trace"].append({"epoch": epoch + 1, "loss": loss,
                "transition": transition, "prior": prior, "estimated_noise_rate_by_noisy_class": rate})
        print(f"SimiFeat seed={seed}: HOC/vote round {epoch + 1}/{epochs} complete", flush=True)
    for name in label_sets:
        result[name]["rank"] = votes[name] > epochs // 2
        result[name]["rank_vote_count"] = votes[name]
        result[name]["score"] = scores[name]
    return result
