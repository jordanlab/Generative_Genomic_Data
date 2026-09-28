#!/usr/bin/env python3
"""
realign_joint_pca.py  —  Re-plot joint PLINK2 PCA with consistent axis orientation
====================================================================================
PLINK2 PCA eigenvectors are sign-ambiguous: each run may flip one or more PC axes
independently.  This script:

  1. Loads the merged_pca.eigenvec + merged_pca.eigenval for every seed in a
     validation output directory.
  2. Determines the canonical orientation from a reference seed (seed42 by default)
     — specifically, the sign of the real BAQ centroid on each PC axis.
  3. Flips any axis in any seed that disagrees with the reference sign.
  4. Computes shared axis limits (global min/max across all seeds, with padding)
     so every plot occupies the same coordinate range.
  5. Writes aligned PNG files to a new subfolder  `<input_dir>/aligned_joint_pca/`
     without touching the originals.

Usage
-----
    python src/realign_joint_pca.py \\
        --input_dir results/plink_pca_validation/cWGAN_binom_gen-data

    # Override reference seed or output subfolder name:
    python src/realign_joint_pca.py \\
        --input_dir results/plink_pca_validation/cWGAN_binom_gen-data \\
        --ref_seed  seed42 \\
        --out_subdir aligned_joint_pca
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders (duplicated from plink_pca_validation.py so this script is standalone)
# ---------------------------------------------------------------------------

def load_eigenvec(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as fh:
        first = fh.readline().strip()
    if first.startswith('#FID') or first.startswith('FID'):
        df = pd.read_csv(path, sep=r'\s+')
        df.columns = [c.lstrip('#') for c in df.columns]
        ids = df['IID'].values.astype(str)
        pcs = df[[c for c in df.columns if c.startswith('PC')]].values.astype(float)
    else:
        df = pd.read_csv(path, sep=r'\s+', header=None)
        ids = df.iloc[:, 1].values.astype(str)
        pcs = df.iloc[:, 2:].values.astype(float)
    return ids, pcs


def load_eigenval(path: Path) -> np.ndarray:
    vals = np.loadtxt(path)
    total = vals.sum()
    return (vals / total) if total > 0 else vals


# ---------------------------------------------------------------------------
# Core: discover files and load all seeds
# ---------------------------------------------------------------------------

def discover_seeds(input_dir: Path) -> dict[str, dict]:
    """Find all *_merged_pca.eigenvec files and return a dict keyed by seed label."""
    seeds = {}
    for evec in sorted(input_dir.glob('*_merged_pca.eigenvec')):
        m = re.search(r'(seed\d+)', evec.name)
        if not m:
            continue
        seed_key = m.group(1)
        eval_p   = evec.with_suffix('.eigenval')
        stem     = evec.name.replace('_merged_pca.eigenvec', '')
        if not eval_p.exists():
            print(f'  WARNING: missing eigenval for {evec.name}, skipping.')
            continue
        seeds[seed_key] = {
            'evec_path': evec,
            'eval_path': eval_p,
            'stem':      stem,
        }
    return seeds


def load_all(seeds: dict) -> dict:
    """Load eigenvec/eigenval data for every seed."""
    for key, info in seeds.items():
        ids, pcs = load_eigenvec(info['evec_path'])
        var_exp  = load_eigenval(info['eval_path'])
        baq_mask  = np.array([i.startswith('BAQ') for i in ids])
        syn_mask  = np.array([i.startswith('SYN') for i in ids])
        info['ids']      = ids
        info['pcs']      = pcs
        info['var_exp']  = var_exp
        info['baq_mask'] = baq_mask
        info['syn_mask'] = syn_mask
        info['n_real']   = int(baq_mask.sum())
        info['n_synth']  = int(syn_mask.sum())
    return seeds


# ---------------------------------------------------------------------------
# Orientation correction
# ---------------------------------------------------------------------------

def compute_sign_flips(seeds: dict, ref_seed: str) -> dict[str, np.ndarray]:
    """Return per-seed sign-flip vectors (+1 / -1 per PC axis).

    The canonical orientation is defined by the reference seed:
      canonical sign of PC_k = sign of BAQ centroid on PC_k in ref_seed.
    Any other seed that has the opposite sign for a given axis gets a -1 flip.
    """
    ref   = seeds[ref_seed]
    ref_centroid = ref['pcs'][ref['baq_mask']].mean(axis=0)  # (n_pcs,)
    # Replace exact zeros (unlikely but safe) with +1
    canonical = np.where(ref_centroid == 0, 1.0, np.sign(ref_centroid))

    flips = {}
    for key, info in seeds.items():
        centroid = info['pcs'][info['baq_mask']].mean(axis=0)
        seed_sign = np.where(centroid == 0, 1.0, np.sign(centroid))
        # Flip is -1 wherever this seed disagrees with the canonical sign
        flip = np.where(seed_sign == canonical, 1.0, -1.0)
        flips[key] = flip
        n_flipped = int((flip < 0).sum())
        if n_flipped:
            axes = np.where(flip < 0)[0] + 1  # 1-indexed PC numbers
            print(f'  {key}: flipping PC{list(axes)} to match {ref_seed}')
        else:
            print(f'  {key}: no flips needed (matches {ref_seed})')
    return flips


def apply_flips(seeds: dict, flips: dict) -> None:
    """Apply sign flips in-place."""
    for key, info in seeds.items():
        info['pcs_aligned'] = info['pcs'] * flips[key]


# ---------------------------------------------------------------------------
# Compute shared axis limits
# ---------------------------------------------------------------------------

def shared_limits(seeds: dict, pad: float = 0.05) -> dict[int, tuple[float, float]]:
    """Global (xmin, xmax) for each PC axis (1-indexed) across all seeds."""
    # Collect all aligned PC values per axis index
    n_pcs = min(info['pcs_aligned'].shape[1] for info in seeds.values())
    limits = {}
    for k in range(n_pcs):
        all_vals = np.concatenate([info['pcs_aligned'][:, k] for info in seeds.values()])
        lo, hi   = all_vals.min(), all_vals.max()
        margin   = (hi - lo) * pad
        limits[k] = (lo - margin, hi + margin)
    return limits


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_aligned_joint(baq_pcs: np.ndarray, synth_pcs: np.ndarray,
                        var_exp: np.ndarray, out_path: Path,
                        label: str, n_real: int, n_synth: int,
                        axis_limits: dict, seed_key: str) -> None:
    """Two-panel joint PCA with enforced shared axis limits."""
    n_pcs = min(baq_pcs.shape[1], synth_pcs.shape[1])
    ve    = [f'{var_exp[i]*100:.1f}%' if i < len(var_exp) else '?' for i in range(3)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [(0, 1), (0, 2)] if n_pcs > 2 else [(0, 1), (0, 1)]

    for ax, (xi, yi) in zip(axes, pairs):
        ax.scatter(baq_pcs[:, xi], baq_pcs[:, yi],
                   c='#E63946', alpha=0.85, s=50,
                   label=f'Real BAQ (n={n_real})',
                   edgecolors='white', linewidths=0.4, zorder=2)
        ax.scatter(synth_pcs[:, xi], synth_pcs[:, yi],
                   c='#7B2FBE', alpha=0.75, s=40,
                   label=f'Synthetic (n={n_synth})',
                   edgecolors='white', linewidths=0.4, zorder=3)

        pc_labels = [f'PC1 ({ve[0]})', f'PC2 ({ve[1]})', f'PC3 ({ve[2]})']
        ax.set_xlabel(pc_labels[xi], fontsize=11)
        ax.set_ylabel(pc_labels[yi], fontsize=11)
        ax.set_title(f'PC{xi+1} vs PC{yi+1}')

        # Enforce shared limits
        if xi in axis_limits:
            ax.set_xlim(axis_limits[xi])
        if yi in axis_limits:
            ax.set_ylim(axis_limits[yi])

    axes[0].legend(fontsize=9, markerscale=0.9)

    plt.suptitle(
        f'Joint PLINK2 PCA — Real BAQ + {seed_key}  (axis-aligned across seeds)\n'
        f'PC axes from merged real+synthetic data — sign-corrected to canonical orientation',
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing *_merged_pca.eigenvec files')
    parser.add_argument('--ref_seed', default='seed42',
                        help='Seed whose BAQ centroid defines the canonical orientation '
                             '(default: seed42)')
    parser.add_argument('--out_subdir', default='aligned_joint_pca',
                        help='Subfolder name inside input_dir for output PNGs '
                             '(default: aligned_joint_pca)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir   = input_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nDiscovering seeds in: {input_dir}')
    seeds = discover_seeds(input_dir)
    if not seeds:
        print('ERROR: no *_merged_pca.eigenvec files found.')
        return
    print(f'Found {len(seeds)} seeds: {sorted(seeds)}')

    if args.ref_seed not in seeds:
        fallback = sorted(seeds)[0]
        print(f'WARNING: --ref_seed={args.ref_seed} not found; using {fallback}')
        args.ref_seed = fallback

    print('\nLoading eigenvec data...')
    seeds = load_all(seeds)

    print(f'\nComputing sign flips (reference: {args.ref_seed})...')
    flips = compute_sign_flips(seeds, args.ref_seed)

    print('\nApplying flips...')
    apply_flips(seeds, flips)

    print('\nComputing shared axis limits...')
    limits = shared_limits(seeds)
    for k, (lo, hi) in limits.items():
        print(f'  PC{k+1}: [{lo:.4f}, {hi:.4f}]')

    print(f'\nWriting aligned plots to: {out_dir}')
    for key, info in sorted(seeds.items()):
        pcs     = info['pcs_aligned']
        baq_pcs = pcs[info['baq_mask']]
        syn_pcs = pcs[info['syn_mask']]
        out_png = out_dir / f'{info["stem"]}_joint_pca_aligned.png'
        plot_aligned_joint(
            baq_pcs, syn_pcs, info['var_exp'],
            out_png, info['stem'],
            info['n_real'], info['n_synth'],
            limits, key,
        )

    print(f'\nDone. {len(seeds)} aligned plots in {out_dir}')


if __name__ == '__main__':
    main()
