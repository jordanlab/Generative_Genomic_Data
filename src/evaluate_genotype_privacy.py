#!/usr/bin/env python3
"""Run simple memorization/membership diagnostics on synthetic genotypes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cwgan_extensions.privacy import privacy_diagnostics


def load(path):
    return pd.read_csv(path).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_real", required=True, help="Real genomes actually used to train the generator.")
    ap.add_argument("--holdout_real", required=True, help="Independent real genomes not used for training.")
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--n_components", type=int, default=20)
    ap.add_argument("--output", default="privacy_metrics.json")
    args = ap.parse_args()
    metrics = privacy_diagnostics(load(args.train_real), load(args.holdout_real), load(args.synthetic), args.n_components)
    Path(args.output).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
