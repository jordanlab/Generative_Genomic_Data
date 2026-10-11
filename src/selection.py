"""Multi-metric model/run selection helpers.

Scores are intended for comparing models evaluated on the same real holdout dataset.
They are not universal goodness scores.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


DEFAULT_WEIGHTS = {
    "maf_mae": 1.0,
    "heterozygosity_mean_abs_error": 0.5,
    "ld_r2_mae": 1.0,
    "pc_rbf_mmd2": 1.0,
    "aa_distance_from_half": 1.0,
}


def metric_loss_vector(metrics: Mapping[str, float]) -> Dict[str, float]:
    """Convert mixed-direction quality metrics to quantities where lower is better."""
    het_real = float(metrics.get("mean_heterozygosity_real", np.nan))
    het_syn = float(metrics.get("mean_heterozygosity_synthetic", np.nan))
    aa = float(metrics.get("pc_adversarial_accuracy_ts", np.nan))
    return {
        "maf_mae": float(metrics.get("maf_mae", np.nan)),
        "heterozygosity_mean_abs_error": abs(het_real - het_syn),
        "ld_r2_mae": float(metrics.get("ld_adjacent_r2_mae", np.nan)),
        "pc_rbf_mmd2": float(metrics.get("pc_rbf_mmd2", np.nan)),
        "aa_distance_from_half": abs(aa - 0.5),
    }


def composite_quality_score(
    metrics: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
) -> float:
    """Weighted dimensionless loss; smaller is better.

    ``scales`` should be estimated from the candidate set (e.g. median absolute
    deviation or interquartile range per metric).  Without scales the raw metric
    units are mixed, so the result is useful only as a transparent heuristic.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    losses = metric_loss_vector(metrics)
    num = den = 0.0
    for key, weight in weights.items():
        value = losses.get(key, np.nan)
        if not np.isfinite(value) or weight <= 0:
            continue
        scale = 1.0 if scales is None else float(scales.get(key, 1.0))
        scale = max(abs(scale), 1e-12)
        num += weight * value / scale
        den += weight
    return float(num / den) if den else float("nan")
