from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

CLEAN = 0
HARD = 1
NOISY = 2
RELIABLE_CLEAN = 0
AMBIGUOUS = 1
RELIABLE_NOISY = 2


class GlobalLocalGMMPartitioner:
    """Unsupervised Global-Local hierarchical reliability partitioner.

    NO CLEAN LABELS ARE USED HERE. The method only consumes frozen features and
    noisy labels. Ground-truth clean labels/noise masks are reserved for external
    evaluation code.
    """

    def __init__(
        self,
        num_classes: int,
        tau_global: float = 0.8,
        local_posterior_threshold: float = 0.5,
        k: int = 20,
        seed: int = 1,
    ):
        if not 0.0 < tau_global < 1.0:
            raise ValueError("tau_global must be in (0, 1).")
        if not 0.5 <= local_posterior_threshold < 1.0:
            raise ValueError("local_posterior_threshold must be in [0.5, 1.0).")
        self.num_classes = num_classes
        self.tau_global = tau_global
        self.local_posterior_threshold = local_posterior_threshold
        self.k = k
        self.seed = seed
        self.local_gmm_fallback = False

    def fit_predict(self, features: torch.Tensor, noisy_labels: torch.Tensor) -> dict:
        features = features.float().cpu()
        noisy_labels = noisy_labels.long().cpu()
        if (
            features.ndim != 2 or noisy_labels.ndim != 1
            or len(features) != len(noisy_labels) or len(features) < 2
            or not torch.isfinite(features).all()
            or (noisy_labels < 0).any() or (noisy_labels >= self.num_classes).any()
            or not 1 <= self.k < len(features)
        ):
            raise ValueError("Invalid features, labels, or k (require 1 <= k < N).")
        self._fit_global(features, noisy_labels)
        local_consistency = self._compute_local_consistency(features, noisy_labels)
        global_state = self._global_state(self.global_clean_prob_, self.global_noisy_prob_)
        ambiguous = global_state == AMBIGUOUS
        self._fit_local(local_consistency, ambiguous)
        global_clean_prob = self.global_clean_prob_
        global_noisy_prob = self.global_noisy_prob_
        local_hard_prob = np.full(len(noisy_labels), np.nan, dtype=np.float32)
        local_noisy_prob = np.full(len(noisy_labels), np.nan, dtype=np.float32)

        partition = np.full(len(noisy_labels), NOISY, dtype=np.int64)
        partition[global_state == RELIABLE_CLEAN] = CLEAN
        partition[global_state == RELIABLE_NOISY] = NOISY
        ambiguous = global_state == AMBIGUOUS
        if ambiguous.any():
            amb_local = local_consistency[ambiguous]
            if self.local_gmm_fallback:
                hard = amb_local >= self.local_fallback_threshold_
                local_hard_prob[ambiguous] = hard.astype(np.float32)
                local_noisy_prob[ambiguous] = (~hard).astype(np.float32)
            else:
                local_prob = self.local_gmm_.predict_proba(amb_local.reshape(-1, 1))
                local_hard_prob[ambiguous] = local_prob[:, self.local_hard_component_]
                local_noisy_prob[ambiguous] = local_prob[:, self.local_noisy_component_]
                hard = local_noisy_prob[ambiguous] < self.local_posterior_threshold
            amb_idx = np.where(ambiguous)[0]
            partition[amb_idx[hard]] = HARD
            partition[amb_idx[~hard]] = NOISY

        return {
            "partition": torch.as_tensor(partition, dtype=torch.long),
            "global_margin": torch.as_tensor(self.global_margin_, dtype=torch.float32),
            "positive_similarity": torch.as_tensor(self.positive_similarity_, dtype=torch.float32),
            "negative_similarity": torch.as_tensor(self.negative_similarity_, dtype=torch.float32),
            "global_clean_prob": torch.as_tensor(global_clean_prob, dtype=torch.float32),
            "global_noisy_prob": torch.as_tensor(global_noisy_prob, dtype=torch.float32),
            "global_state": torch.as_tensor(global_state, dtype=torch.long),
            "local_consistency": torch.as_tensor(local_consistency, dtype=torch.float32),
            "local_hard_prob": torch.as_tensor(local_hard_prob, dtype=torch.float32),
            "local_noisy_prob": torch.as_tensor(local_noisy_prob, dtype=torch.float32),
            "reliability_score": torch.as_tensor(global_clean_prob - global_noisy_prob, dtype=torch.float32),
            "local_gmm_fallback": self.local_gmm_fallback,
        }

    def _fit_global(self, features: torch.Tensor, noisy_labels: torch.Tensor) -> None:
        global_values = self._compute_global_margin(features, noisy_labels)
        margin = global_values["global_margin"]
        self.global_gmm_ = self._make_gmm().fit(margin.reshape(-1, 1))
        means = self.global_gmm_.means_.reshape(-1)
        self.global_noisy_component_ = int(np.argmin(means))
        self.global_clean_component_ = int(np.argmax(means))
        prob = self.global_gmm_.predict_proba(margin.reshape(-1, 1))
        self.global_margin_ = margin
        self.positive_similarity_ = global_values["positive_similarity"]
        self.negative_similarity_ = global_values["negative_similarity"]
        self.global_clean_prob_ = prob[:, self.global_clean_component_]
        self.global_noisy_prob_ = prob[:, self.global_noisy_component_]

    def _compute_global_margin(self, features: torch.Tensor, noisy_labels: torch.Tensor) -> dict[str, np.ndarray]:
        z = F.normalize(features, dim=1)
        global_mean = z.mean(dim=0)
        prototypes = []
        for cls in range(self.num_classes):
            # NO CLEAN LABELS ARE USED HERE: prototypes are grouped by noisy labels.
            mask = noisy_labels == cls
            proto = z[mask].mean(dim=0) if mask.any() else global_mean
            prototypes.append(F.normalize(proto, dim=0))
        prototypes = torch.stack(prototypes, dim=0)
        sims = z @ prototypes.T
        positive = sims.gather(1, noisy_labels.view(-1, 1)).squeeze(1)
        masked = sims.clone()
        masked[torch.arange(len(noisy_labels)), noisy_labels] = -float("inf")
        negative = masked.max(dim=1).values
        margin = positive - negative
        return {
            "global_margin": margin.numpy(),
            "positive_similarity": positive.numpy(),
            "negative_similarity": negative.numpy(),
        }

    def _compute_local_consistency(self, features: torch.Tensor, noisy_labels: torch.Tensor) -> np.ndarray:
        # NO CLEAN LABELS ARE USED HERE: neighbor agreement compares noisy labels only.
        n_neighbors = min(self.k + 1, len(noisy_labels))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(features.numpy())
        _, indices = knn.kneighbors(features.numpy())
        neighbor_idx = indices[:, 1 : self.k + 1]
        labels = noisy_labels.numpy()
        return np.asarray([(labels[row] == label).mean() for row, label in zip(neighbor_idx, labels)], dtype=np.float32)

    def _fit_local(self, local_consistency: np.ndarray, ambiguous: np.ndarray) -> None:
        n_ambiguous = int(ambiguous.sum())
        min_required = max(50, 2 * self.k)
        amb_values = local_consistency[ambiguous]
        if n_ambiguous < min_required or len(np.unique(amb_values)) < 2:
            self.local_gmm_fallback = True
            self.local_fallback_threshold_ = float(np.median(amb_values)) if n_ambiguous else float(np.median(local_consistency))
            warnings.warn("local_gmm_fallback=true: ambiguous subset too small or degenerate; using median local consistency.", RuntimeWarning)
            return
        self.local_gmm_fallback = False
        self.local_gmm_ = self._make_gmm().fit(amb_values.reshape(-1, 1))
        means = self.local_gmm_.means_.reshape(-1)
        self.local_noisy_component_ = int(np.argmin(means))
        self.local_hard_component_ = int(np.argmax(means))

    def _global_state(self, clean_prob: np.ndarray, noisy_prob: np.ndarray) -> np.ndarray:
        state = np.full(len(clean_prob), AMBIGUOUS, dtype=np.int64)
        state[clean_prob >= self.tau_global] = RELIABLE_CLEAN
        state[noisy_prob >= self.tau_global] = RELIABLE_NOISY
        return state

    def _make_gmm(self) -> GaussianMixture:
        return GaussianMixture(n_components=2, random_state=self.seed, n_init=10, reg_covar=1e-6)
