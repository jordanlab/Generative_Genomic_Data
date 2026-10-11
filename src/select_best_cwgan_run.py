#!/usr/bin/env python3
"""Select among completed cWGAN model folders using multiple fidelity metrics.

Example:
  python select_best_cwgan_run.py runs/20261003_* --output model_comparison.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cwgan_extensions.selection import DEFAULT_WEIGHTS, metric_loss_vector


def discover(paths):
    files = []
    for token in paths:
        matches = glob.glob(token)
        if not matches:
            matches = [token]
        for match in matches:
            p = Path(match)
            if p.is_file() and p.name == "quality_metrics.json":
                files.append(p)
            elif p.is_dir():
                files.extend(p.rglob("quality_metrics.json"))
    return sorted(set(files))


def robust_scale(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 1.0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr > 1e-12:
        return float(iqr)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(mad if mad > 1e-12 else max(abs(med), 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Run/model directories, globs, or quality_metrics.json files.")
    ap.add_argument("--output", default="model_comparison.csv")
    ap.add_argument("--weight", action="append", default=[], metavar="METRIC=VALUE",
                    help="Override a composite-score weight. Repeatable.")
    args = ap.parse_args()

    weights = dict(DEFAULT_WEIGHTS)
    for item in args.weight:
        key, value = item.split("=", 1)
        weights[key] = float(value)

    files = discover(args.paths)
    if not files:
        raise SystemExit("No quality_metrics.json files found.")
    rows = []
    for path in files:
        with open(path) as fh:
            metrics = json.load(fh)
        row = {"metrics_file": str(path), "model_dir": str(path.parent.parent)}
        row.update(metrics)
        row.update({f"loss_{k}": v for k, v in metric_loss_vector(metrics).items()})
        rows.append(row)
    df = pd.DataFrame(rows)

    scales = {}
    for key in weights:
        col = f"loss_{key}"
        if col in df:
            scales[key] = robust_scale(df[col].to_numpy())
    scores = []
    for _, row in df.iterrows():
        num = den = 0.0
        for key, w in weights.items():
            value = row.get(f"loss_{key}", np.nan)
            if np.isfinite(value) and w > 0:
                num += w * value / scales[key]
                den += w
        scores.append(num / den if den else np.nan)
    df["composite_score_lower_is_better"] = scores
    df = df.sort_values("composite_score_lower_is_better", na_position="last")
    df.to_csv(args.output, index=False)
    print(df[["model_dir", "composite_score_lower_is_better"]].to_string(index=False))
    print(f"\nSaved: {args.output}")
    print("Note: this is a within-candidate, same-holdout comparison; it is not a universal quality score.")


if __name__ == "__main__":
    main()
