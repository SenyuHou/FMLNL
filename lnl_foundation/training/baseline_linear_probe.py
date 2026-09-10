from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from lnl_foundation.training.calibration import evaluate_logits
from lnl_foundation.utils import set_seed


METHODS = ("ce", "gce", "coteaching", "dividemix", "disc", "clipcleaner")


@dataclass
class BaselineResult:
    metrics: dict
    artifacts: dict


def _head(feature_dim, num_classes, device):
    return nn.Linear(feature_dim, num_classes).to(device)


def _optimizer(model, cfg):
    name = cfg["optimizer"].lower()
    arguments = {
        "lr": float(cfg["learning_rate"]),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
    }
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=float(cfg.get("momentum", 0.0)), **arguments)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **arguments)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **arguments)
    raise ValueError(f"Unsupported optimizer: {name}")


def _epoch_batches(indices, batch_size, device):
    order = indices[torch.randperm(len(indices))]
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size].to(device)


def _view(features, dropout):
    if dropout <= 0:
        return features
    return F.normalize(F.dropout(features, p=float(dropout), training=True), dim=1)


def _gce(logits, labels, q=0.7, reduction="mean"):
    probabilities = F.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1).clamp_min(1e-7)
    losses = (1.0 - probabilities.pow(float(q))) / float(q)
    return losses if reduction == "none" else losses.mean()


@torch.no_grad()
def _logits(model, features, device, batch_size=4096):
    model.eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        outputs.append(model(features[start:start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


def _finish(models, test_features, test_labels, device, diagnostics, reduction="mean"):
    stacked = torch.stack([_logits(model, test_features, device) for model in models])
    logits = stacked.sum(0) if reduction == "sum" else stacked.mean(0)
    evaluated = evaluate_logits(logits, test_labels)
    metrics = {
        "accuracy": evaluated["accuracy"],
        "macro_f1": evaluated["macro_f1"],
        "ece_raw": evaluated["ece"],
        **diagnostics,
    }
    return BaselineResult(metrics, {
        "test_logits": logits,
        "test_probabilities_raw": evaluated["probabilities"],
        "test_predictions": logits.argmax(1),
    })


def _train_single(train_features, noisy_labels, test_features, test_labels, num_classes,
                  seed, device, cfg, selected_indices=None, gce_q=None):
    set_seed(seed)
    model = _head(train_features.shape[1], num_classes, device)
    optimizer = _optimizer(model, cfg)
    selected = (torch.arange(len(train_features)) if selected_indices is None
                else torch.as_tensor(selected_indices, dtype=torch.long).cpu())
    if len(selected) == 0:
        raise ValueError("The selected training subset is empty.")
    features = train_features.to(device)
    labels = noisy_labels.to(device)
    progress = tqdm(range(int(cfg["epochs"])), desc="Linear baseline", unit="epoch", dynamic_ncols=True)
    for _ in progress:
        model.train()
        total = 0.0
        for batch in _epoch_batches(selected, int(cfg["batch_size"]), device):
            logits = model(features[batch])
            loss = (F.cross_entropy(logits, labels[batch]) if gce_q is None
                    else _gce(logits, labels[batch], gce_q))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        progress.set_postfix(loss=f"{total / len(selected):.4f}")
    return _finish([model], test_features, test_labels, device, {"n_train": int(len(selected))})


def _train_coteaching(train_features, noisy_labels, test_features, test_labels, num_classes,
                      seed, device, cfg, forget_rate):
    set_seed(seed)
    models = [_head(train_features.shape[1], num_classes, device) for _ in range(2)]
    optimizers = [_optimizer(model, cfg) for model in models]
    features, labels = train_features.to(device), noisy_labels.to(device)
    epochs = int(cfg["epochs"])
    gradual = int(cfg["num_gradual"])
    schedule = np.full(epochs, float(forget_rate))
    schedule[:gradual] = np.linspace(0, float(forget_rate) ** float(cfg["exponent"]), gradual)
    initial_lr = float(cfg["learning_rate"])
    decay_start = int(cfg["epoch_decay_start"])
    progress = tqdm(range(epochs), desc="Co-teaching", unit="epoch", dynamic_ncols=True)
    all_indices = torch.arange(len(features))
    for epoch in progress:
        if epoch >= decay_start:
            lr = initial_lr * max(0.0, (epochs - epoch) / max(1, epochs - decay_start))
            beta1 = float(cfg["beta1_final"])
        else:
            lr = initial_lr
            beta1 = float(cfg["beta1_initial"])
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["betas"] = (beta1, group["betas"][1])
        mean_loss = 0.0
        for batch in _epoch_batches(all_indices, int(cfg["batch_size"]), device):
            logits = [model(features[batch]) for model in models]
            losses = [F.cross_entropy(value, labels[batch], reduction="none") for value in logits]
            remember = max(1, int((1.0 - schedule[epoch]) * len(batch)))
            chosen = [torch.argsort(value)[:remember] for value in losses]
            exchanged = [losses[0][chosen[1]].mean(), losses[1][chosen[0]].mean()]
            for optimizer in optimizers:
                optimizer.zero_grad()
            (exchanged[0] + exchanged[1]).backward()
            for optimizer in optimizers:
                optimizer.step()
            mean_loss += float(sum(exchanged).detach()) * len(batch) / 2
        progress.set_postfix(loss=f"{mean_loss / len(features):.4f}", forget=f"{schedule[epoch]:.3f}")
    return _finish(models, test_features, test_labels, device, {
        "n_train": len(features), "forget_rate": float(forget_rate), "n_heads": 2,
    })


@torch.no_grad()
def _divide_clean_probability(model, features, labels, device, seed):
    logits = _logits(model, features, device)
    losses = F.cross_entropy(logits, labels, reduction="none").numpy()
    span = losses.max() - losses.min()
    normalized = ((losses - losses.min()) / span if span > 1e-12 else np.zeros_like(losses))
    gmm = GaussianMixture(
        n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=seed,
    ).fit(normalized.reshape(-1, 1))
    probability = gmm.predict_proba(normalized.reshape(-1, 1))
    return torch.from_numpy(probability[:, gmm.means_.argmin()]).float()


def _sharpen(probability, temperature):
    value = probability.pow(1.0 / float(temperature))
    return value / value.sum(dim=1, keepdim=True)


def _random_sample(indices, count, device):
    if len(indices) == 0:
        raise ValueError("DivideMix produced an empty labeled or unlabeled split.")
    return indices[torch.randint(len(indices), (count,))].to(device)


def _divide_train_epoch(model, peer, optimizer, features, labels, clean_probability,
                        epoch, warmup, cfg, device):
    threshold = float(cfg["clean_probability_threshold"])
    labeled = torch.where(clean_probability >= threshold)[0]
    unlabeled = torch.where(clean_probability < threshold)[0]
    if len(labeled) == 0 or len(unlabeled) == 0:
        order = torch.argsort(clean_probability)
        boundary = min(max(1, len(order) // 2), len(order) - 1)
        unlabeled, labeled = order[:boundary], order[boundary:]
    batch_size = int(cfg["batch_size"])
    iterations = max(1, len(labeled) // batch_size + 1)
    model.train()
    peer.eval()
    for _ in range(iterations):
        lx = _random_sample(labeled, batch_size, device)
        ux = _random_sample(unlabeled, batch_size, device)
        x = features[lx]
        u = features[ux]
        x1, x2 = _view(x, cfg["weak_feature_dropout"]), _view(x, cfg["weak_feature_dropout"])
        u1, u2 = _view(u, cfg["weak_feature_dropout"]), _view(u, cfg["weak_feature_dropout"])
        one_hot = F.one_hot(labels[lx], num_classes=model.out_features).float()
        with torch.no_grad():
            pu = sum(F.softmax(net(view), dim=1) for net in (model, peer) for view in (u1, u2)) / 4
            targets_u = _sharpen(pu, cfg["sharpen_temperature"])
            px = (F.softmax(model(x1), dim=1) + F.softmax(model(x2), dim=1)) / 2
            weight = clean_probability[lx.cpu()].to(device)[:, None]
            targets_x = _sharpen(weight * one_hot + (1 - weight) * px, cfg["sharpen_temperature"])
        inputs = torch.cat((x1, x2, u1, u2))
        targets = torch.cat((targets_x, targets_x, targets_u, targets_u))
        beta = np.random.beta(float(cfg["mixup_alpha"]), float(cfg["mixup_alpha"]))
        mix = max(beta, 1 - beta)
        permutation = torch.randperm(len(inputs), device=device)
        mixed_inputs = mix * inputs + (1 - mix) * inputs[permutation]
        mixed_targets = mix * targets + (1 - mix) * targets[permutation]
        logits = model(mixed_inputs)
        supervised_count = 2 * batch_size
        loss_x = -(mixed_targets[:supervised_count] * F.log_softmax(logits[:supervised_count], 1)).sum(1).mean()
        probs_u = F.softmax(logits[supervised_count:], dim=1)
        loss_u = (probs_u - mixed_targets[supervised_count:]).square().mean()
        ramp = np.clip((epoch - warmup) / max(1, int(cfg["rampup_length"])), 0, 1)
        prior = torch.full((model.out_features,), 1.0 / model.out_features, device=device)
        prediction_mean = F.softmax(logits, dim=1).mean(0)
        penalty = (prior * torch.log(prior / prediction_mean.clamp_min(1e-8))).sum()
        loss = loss_x + float(cfg["unsupervised_weight"]) * ramp * loss_u + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return len(labeled), len(unlabeled)


def _negative_entropy(logits):
    probabilities = F.softmax(logits, dim=1)
    return (probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(1).mean()


def _train_dividemix(train_features, noisy_labels, test_features, test_labels, num_classes,
                     dataset, noise_type, seed, device, cfg):
    set_seed(seed)
    models = [_head(train_features.shape[1], num_classes, device) for _ in range(2)]
    optimizers = [_optimizer(model, cfg) for model in models]
    features, labels = train_features.to(device), noisy_labels.to(device)
    warmup = int(cfg[f"warmup_{dataset}"])
    progress = tqdm(range(int(cfg["epochs"])), desc="DivideMix", unit="epoch", dynamic_ncols=True)
    final_counts = [(len(features), 0), (len(features), 0)]
    all_indices = torch.arange(len(features))
    for epoch in progress:
        if epoch >= int(cfg["learning_rate_decay_epoch"]):
            lr = float(cfg["learning_rate"]) * float(cfg["learning_rate_decay"])
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr
        if epoch < warmup:
            for model, optimizer in zip(models, optimizers):
                model.train()
                for batch in _epoch_batches(all_indices, int(cfg["batch_size"]), device):
                    logits = model(_view(features[batch], cfg["weak_feature_dropout"]))
                    loss = F.cross_entropy(logits, labels[batch])
                    if noise_type == "pairflip":
                        loss = loss + _negative_entropy(logits)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        else:
            clean = [
                _divide_clean_probability(model, train_features, noisy_labels, device, seed + epoch * 2 + i)
                for i, model in enumerate(models)
            ]
            final_counts[0] = _divide_train_epoch(
                models[0], models[1], optimizers[0], features, labels, clean[1], epoch,
                warmup, cfg, device,
            )
            final_counts[1] = _divide_train_epoch(
                models[1], models[0], optimizers[1], features, labels, clean[0], epoch,
                warmup, cfg, device,
            )
        progress.set_postfix(labeled=f"{final_counts[0][0]}/{final_counts[1][0]}")
    return _finish(models, test_features, test_labels, device, {
        "n_train": len(features), "n_heads": 2, "warmup_epochs": warmup,
        "final_labeled_head1": final_counts[0][0], "final_labeled_head2": final_counts[1][0],
    }, reduction="sum")


def _mixup_ce(model, inputs, targets, alpha):
    if len(inputs) == 0:
        return inputs.sum() * 0
    beta = np.random.beta(float(alpha), float(alpha))
    mix = max(beta, 1 - beta)
    permutation = torch.randperm(len(inputs), device=inputs.device)
    logits = model(mix * inputs + (1 - mix) * inputs[permutation])
    loss_a = F.cross_entropy(logits, targets, reduction="none")
    loss_b = F.cross_entropy(logits, targets[permutation], reduction="none")
    return (mix * loss_a + (1 - mix) * loss_b).mean()


def _train_disc(train_features, noisy_labels, test_features, test_labels, num_classes,
                noise_type, noise_rate, seed, device, cfg):
    set_seed(seed)
    model = _head(train_features.shape[1], num_classes, device)
    optimizer = _optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(x) for x in cfg["milestones"]], gamma=float(cfg["learning_rate_decay"])
    )
    features, labels = train_features.to(device), noisy_labels.to(device)
    n = len(features)
    weak_conf = torch.full((n,), 1.0 / num_classes, device=device)
    strong_conf = weak_conf.clone()
    clean_mask = torch.zeros(n, dtype=torch.bool, device=device)
    hard_mask = clean_mask.clone()
    purified_mask = clean_mask.clone()
    corrected_labels = labels.clone()
    start_epoch = int(cfg["instance_start_epoch"] if noise_type == "instance" else cfg["start_epoch"])
    high_noise = noise_rate is not None and float(noise_rate) >= 0.6
    lower_momentum = high_noise or noise_type == "pairflip"
    confidence_momentum = float(
        cfg["high_noise_confidence_momentum"] if lower_momentum else cfg["confidence_momentum"]
    )
    hard_weight = float(cfg["high_noise_hard_weight"] if high_noise else 1.0)
    all_indices = torch.arange(n)
    progress = tqdm(range(int(cfg["epochs"])), desc="DISC", unit="epoch", dynamic_ncols=True)
    for epoch in progress:
        probabilities_w = torch.empty((n, num_classes), device=device)
        probabilities_s = torch.empty((n, num_classes), device=device)
        model.train()
        for batch in _epoch_batches(all_indices, int(cfg["batch_size"]), device):
            weak = _view(features[batch], cfg["weak_feature_dropout"])
            strong = _view(features[batch], cfg["strong_feature_dropout"])
            logits_w, logits_s = model(weak), model(strong)
            probabilities_w[batch] = F.softmax(logits_w.detach(), dim=1)
            probabilities_s[batch] = F.softmax(logits_s.detach(), dim=1)
            if epoch < start_epoch:
                loss = F.cross_entropy(logits_w, labels[batch]) + F.cross_entropy(logits_s, labels[batch])
            else:
                local_clean, local_hard, local_purified = clean_mask[batch], hard_mask[batch], purified_mask[batch]
                loss = (logits_w.sum() + logits_s.sum()) * 0
                if local_clean.any():
                    scale = local_clean.float().mean()
                    loss = loss + scale * (
                        F.cross_entropy(logits_w[local_clean], labels[batch][local_clean])
                        + F.cross_entropy(logits_s[local_clean], labels[batch][local_clean])
                    )
                if local_hard.any():
                    scale = local_hard.float().mean()
                    loss = loss + hard_weight * scale * (
                        _gce(logits_w[local_hard], labels[batch][local_hard], cfg["gce_q"])
                        + _gce(logits_s[local_hard], labels[batch][local_hard], cfg["gce_q"])
                    )
                if local_purified.any():
                    target = corrected_labels[batch][local_purified]
                    loss = loss + _mixup_ce(model, weak[local_purified], target, cfg["mixup_alpha"])
                    loss = loss + _mixup_ce(model, strong[local_purified], target, cfg["mixup_alpha"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            confidence_w = probabilities_w.gather(1, labels[:, None]).squeeze(1)
            confidence_s = probabilities_s.gather(1, labels[:, None]).squeeze(1)
            selected_w = confidence_w > weak_conf
            selected_s = confidence_s > strong_conf
            clean_mask = selected_w & selected_s
            hard_mask = selected_w ^ selected_s
            selected = selected_w | selected_s
            ensemble = (probabilities_w + probabilities_s) / 2
            max_confidence, predicted = ensemble.max(1)
            correction_threshold = torch.clamp((weak_conf + strong_conf) / 2 + float(cfg["correction_sigma"]), max=0.99)
            correction = (~selected) & (max_confidence > correction_threshold)
            purified_mask = selected | correction
            corrected_labels = torch.where(correction, predicted, labels)
            weak_conf = confidence_momentum * weak_conf + (1 - confidence_momentum) * probabilities_w.max(1).values
            strong_conf = confidence_momentum * strong_conf + (1 - confidence_momentum) * probabilities_s.max(1).values
        progress.set_postfix(clean=int(clean_mask.sum()), hard=int(hard_mask.sum()), corrected=int(correction.sum()))
    return _finish([model], test_features, test_labels, device, {
        "n_train": n, "start_epoch": start_epoch, "final_clean": int(clean_mask.sum()),
        "final_hard": int(hard_mask.sum()), "final_purified": int(purified_mask.sum()),
        "final_corrected": int(correction.sum()),
    })


def train_stage2_baseline(method, train_features, noisy_labels, test_features, test_labels,
                          num_classes, dataset, noise_type, noise_rate, seed, device, cfg,
                          selected_indices=None):
    """Train a baseline without accepting any Stage-1 Ours reliability signal."""
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unknown baseline method: {method}")
    train_features = torch.as_tensor(train_features, dtype=torch.float32).cpu()
    noisy_labels = torch.as_tensor(noisy_labels, dtype=torch.long).cpu()
    test_features = torch.as_tensor(test_features, dtype=torch.float32).cpu()
    test_labels = torch.as_tensor(test_labels, dtype=torch.long).cpu()
    method_cfg = cfg[method]
    if method in {"ce", "clipcleaner"}:
        return _train_single(
            train_features, noisy_labels, test_features, test_labels, num_classes, seed,
            device, method_cfg, selected_indices=selected_indices,
        )
    if method == "gce":
        return _train_single(
            train_features, noisy_labels, test_features, test_labels, num_classes, seed,
            device, method_cfg, gce_q=float(method_cfg["q"]),
        )
    if method == "coteaching":
        forget = float(noise_rate if noise_rate is not None else method_cfg["human_forget_rate"])
        return _train_coteaching(
            train_features, noisy_labels, test_features, test_labels, num_classes, seed,
            device, method_cfg, forget,
        )
    if method == "dividemix":
        return _train_dividemix(
            train_features, noisy_labels, test_features, test_labels, num_classes, dataset,
            noise_type, seed, device, method_cfg,
        )
    return _train_disc(
        train_features, noisy_labels, test_features, test_labels, num_classes, noise_type,
        noise_rate, seed, device, method_cfg,
    )
