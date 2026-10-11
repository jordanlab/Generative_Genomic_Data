"""Privacy diagnostics for synthetic genotype datasets.

These diagnostics do not constitute a differential-privacy guarantee.  They are
intended to detect obvious memorization/proximity leakage and a simple distance-
based membership-inference signal.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _standardized_pc_embedding(train: np.ndarray, *others: np.ndarray, n_components: int = 20):
    pooled = np.vstack((train,) + others).astype(np.float64)
    scaler = StandardScaler(with_mean=True, with_std=True)
    pooled = scaler.fit_transform(pooled)
    k = min(n_components, pooled.shape[0] - 1, pooled.shape[1])
    pca = PCA(n_components=max(1, k), random_state=42)
    embedded = pca.fit_transform(pooled)
    sizes = [len(train)] + [len(x) for x in others]
    cuts = np.cumsum([0] + sizes)
    return [embedded[cuts[i]:cuts[i + 1]] for i in range(len(sizes))]


def _min_sqdist(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float64)
    b2 = (b * b).sum(axis=1)[None, :]
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk]
        d2 = (x * x).sum(axis=1)[:, None] + b2 - 2.0 * x @ b.T
        out[i:i + len(x)] = np.maximum(d2, 0.0).min(axis=1)
    return out


def privacy_diagnostics(
    train_real: np.ndarray,
    holdout_real: np.ndarray,
    synthetic: np.ndarray,
    n_components: int = 20,
) -> Dict[str, float]:
    """Compute proximity and a simple membership-inference AUC.

    Membership attack score for a real genome x is s(x)=-min_y ||phi(x)-phi(y)||^2,
    where y ranges over synthetic genomes and phi is a PCA embedding fit jointly on
    the supplied datasets.  AUC > 0.5 means training members tend to lie closer to
    synthetic genomes than held-out non-members.  AUC near 0.5 means this particular
    attack cannot distinguish membership; it is not a proof of privacy.
    """
    train_real = np.asarray(train_real)
    holdout_real = np.asarray(holdout_real)
    synthetic = np.asarray(synthetic)
    tr, ho, sy = _standardized_pc_embedding(train_real, holdout_real, synthetic, n_components=n_components)
    d_train = _min_sqdist(tr, sy)
    d_hold = _min_sqdist(ho, sy)
    y = np.r_[np.ones(len(d_train)), np.zeros(len(d_hold))]
    score = -np.r_[d_train, d_hold]
    auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan

    # Exact row replay on the integer genotype matrix.
    synth_rows = {row.tobytes() for row in np.ascontiguousarray(synthetic)}
    exact_train = np.mean([row.tobytes() in synth_rows for row in np.ascontiguousarray(train_real)])
    exact_hold = np.mean([row.tobytes() in synth_rows for row in np.ascontiguousarray(holdout_real)])
    return {
        "membership_distance_auc": float(auc),
        "member_min_pc_distance_mean": float(np.sqrt(d_train).mean()),
        "holdout_min_pc_distance_mean": float(np.sqrt(d_hold).mean()),
        "exact_training_row_replay_fraction": float(exact_train),
        "exact_holdout_row_replay_fraction": float(exact_hold),
    }
