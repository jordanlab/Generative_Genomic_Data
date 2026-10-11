"""Conditioning utilities for ancestry-aware synthetic genotype models."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SUPERPOPS = ("AFR", "EUR", "AMR")


def _normalize_simplex(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"Expected an (n,3) ancestry matrix, received {q.shape}.")
    if not np.all(np.isfinite(q)):
        raise ValueError("External ancestry coefficients contain NaN or infinite values.")
    if np.any(q < -1e-7):
        raise ValueError("External ancestry coefficients must be non-negative.")
    q = np.clip(q, 0.0, None)
    row_sums = q.sum(axis=1, keepdims=True)
    if np.any(row_sums <= eps):
        raise ValueError("Every ancestry row must have positive total mass.")
    return (q / row_sums).astype(np.float32)


def load_external_conditions(
    path: str,
    expected_rows: int,
    order: Iterable[str] = SUPERPOPS,
) -> np.ndarray:
    """Load externally estimated global ancestry fractions.

    Accepted formats
    ----------------
    1. CSV/TSV with named columns AFR, EUR, AMR (extra columns are ignored).
    2. ADMIXTURE-like whitespace ``.Q`` file with exactly three numeric columns.
       For a .Q file the columns are assumed to follow ``order``.

    Row order must be identical to the genotype CSV row order.  This is deliberate:
    genotype CSVs used by the parent pipeline do not currently carry a mandatory
    sample-ID column, so silently joining on guessed identifiers would be unsafe.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)
    order = tuple(x.upper() for x in order)
    if set(order) != set(SUPERPOPS) or len(order) != 3:
        raise ValueError("order must contain AFR, EUR, AMR exactly once each.")

    # First try a delimiter-aware table with headers.
    table = None
    try:
        table = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        table = None

    if table is not None and set(SUPERPOPS).issubset({str(c).upper() for c in table.columns}):
        rename = {c: str(c).upper() for c in table.columns}
        table = table.rename(columns=rename)
        q = table.loc[:, list(SUPERPOPS)].to_numpy(dtype=np.float32)
    else:
        # ADMIXTURE .Q is whitespace-separated and normally has no header.
        raw = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
        if raw.shape[1] != 3:
            raise ValueError(
                f"External condition file {path} has {raw.shape[1]} columns; "
                "expected named AFR/EUR/AMR columns or exactly three .Q columns."
            )
        q_in = raw.to_numpy(dtype=np.float32)
        q = np.zeros_like(q_in)
        for src_idx, pop in enumerate(order):
            q[:, SUPERPOPS.index(pop)] = q_in[:, src_idx]

    if q.shape[0] != expected_rows:
        raise ValueError(
            f"External condition file {path} has {q.shape[0]} rows but its genotype "
            f"dataset has {expected_rows} rows. Row-order alignment is required."
        )
    return _normalize_simplex(q)
