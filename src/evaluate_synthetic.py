"""
Synthetic Genotype Evaluation
==============================
Quantitative metrics comparing real BAQ genotypes to synthetic genotypes
in both PC space and genotype space.

Metrics
-------
PC space (fit on real BAQ, project synthetic into same axes):
  frechet_dist   — Fréchet distance between real and synthetic PC distributions.
                   Lower = more similar. Analogue of FID for images.
  mmd            — Maximum Mean Discrepancy (RBF kernel, median bandwidth).
                   Lower = more similar. Robust to non-Gaussian distributions.
  nn_accuracy    — 1-NN leave-one-out accuracy on pooled real+synthetic.
                   50% = indistinguishable. 100% = perfectly separable.

Genotype space:
  af_r2          — Pearson r² of per-SNP allele frequencies.
                   ≥ 0.99 = genotypes faithfully represent real population.
  het_rate_real  — Mean per-sample heterozygosity rate in real data.
  het_rate_synth — Mean per-sample heterozygosity rate in synthetic data.
  het_delta      — Absolute difference. Small = biologically plausible.

Outputs (per evaluation run)
----------------------------
  metrics.csv          — one-row CSV, designed to be concatenated across runs
  eval_panels.png      — 4-panel figure: PCA overlay, AF scatter,
                         heterozygosity histogram, metric summary

Usage
-----
Single file:
  python evaluate_synthetic.py \\
      --real       data/BAQ/BAQ_qc2_grch38_final_qc_ldpruned_common_snps_unphased_tensor.csv \\
      --synthetic  results/cWGAN_vs1p5_gen-data/synthetic_genotypes_seed42.csv \\
      --output_dir results/eval/cWGAN_vs1p5_seed42 \\
      --label      cWGAN_vs1p5_seed42

Multiple files (batch mode — one output dir per file):
  python evaluate_synthetic.py \\
      --real        data/BAQ/... \\
      --synthetic   results/cWGAN_vs1p5_gen-data/*.csv \\
      --output_dir  results/eval/cWGAN_vs1p5 \\
      --label       cWGAN_vs1p5
  Produces results/eval/cWGAN_vs1p5/metrics_all.csv concatenating all runs.
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# Metric functions
# =============================================================================

def frechet_distance(real_pcs: np.ndarray, synth_pcs: np.ndarray) -> float:
    """Fréchet distance between real and synthetic PC distributions."""
    mu_r, mu_s = real_pcs.mean(0), synth_pcs.mean(0)
    sr = np.cov(real_pcs.T) if real_pcs.shape[0] > 1 else np.eye(real_pcs.shape[1])
    ss = np.cov(synth_pcs.T) if synth_pcs.shape[0] > 1 else np.eye(synth_pcs.shape[1])
    diff = mu_r - mu_s
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        covmean = sqrtm(sr @ ss)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(sr + ss - 2.0 * covmean))
    return fd


def mmd(real_pcs: np.ndarray, synth_pcs: np.ndarray) -> float:
    """Unbiased MMD² with RBF kernel (median bandwidth heuristic)."""
    all_pts = np.vstack([real_pcs, synth_pcs])
    dists   = np.sum((all_pts[:, None] - all_pts[None, :]) ** 2, axis=-1)
    gamma   = 1.0 / (np.median(dists[dists > 0]) + 1e-10)

    def rbf(A, B):
        d = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-gamma * d)

    Kxx = rbf(real_pcs, real_pcs)
    Kyy = rbf(synth_pcs, synth_pcs)
    Kxy = rbf(real_pcs, synth_pcs)

    n, m = len(real_pcs), len(synth_pcs)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    val = (Kxx.sum() / (n * (n - 1))
           + Kyy.sum() / (m * (m - 1))
           - 2.0 * Kxy.mean())
    return float(val)


def nn_accuracy(real_pcs: np.ndarray, synth_pcs: np.ndarray, seed: int = 42) -> float:
    """Leave-one-out 1-NN accuracy on pooled real (0) + synthetic (1).
    50% = indistinguishable. 100% = perfectly separable.
    """
    rng = np.random.default_rng(seed)
    n_r, n_s = len(real_pcs), len(synth_pcs)
    X = np.vstack([real_pcs, synth_pcs])
    y = np.array([0] * n_r + [1] * n_s)

    # LOO 1-NN: for each point, find nearest neighbor excluding itself
    correct = 0
    for i in range(len(X)):
        dists = np.sum((X - X[i]) ** 2, axis=1)
        dists[i] = np.inf
        nn_label = y[np.argmin(dists)]
        correct += int(nn_label == y[i])
    return correct / len(X)


def allele_freq_r2(real_geno: np.ndarray,
                   synth_geno: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Pearson r² of per-SNP allele frequencies (dosage / 2)."""
    af_r = real_geno.mean(0) / 2.0
    af_s = synth_geno.mean(0) / 2.0
    # exclude monomorphic in both (uninformative for r²)
    variable = (af_r > 0) | (af_s > 0)
    r, _ = pearsonr(af_r[variable], af_s[variable])
    return float(r ** 2), af_r, af_s


def heterozygosity(geno: np.ndarray) -> np.ndarray:
    """Per-sample heterozygosity rate (fraction of het sites, dosage=1)."""
    return (geno == 1).mean(axis=1)


# =============================================================================
# Plotting
# =============================================================================

def make_figure(real_pcs, synth_pcs, af_real, af_synth,
                het_real, het_synth, metrics: dict,
                label: str, save_path: str):
    """4-panel evaluation figure."""
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(f"Synthetic Genotype Evaluation — {label}", fontsize=13,
                 fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 1: PC1/PC2 overlay ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(real_pcs[:, 0],  real_pcs[:, 1],
                c='#E63946', s=35, alpha=0.75, label='Real (BAQ)', zorder=3)
    ax1.scatter(synth_pcs[:, 0], synth_pcs[:, 1],
                c='#457B9D', s=35, alpha=0.65, marker='^',
                label='Synthetic', zorder=2)
    ax1.set_xlabel('PC1', fontsize=9)
    ax1.set_ylabel('PC2', fontsize=9)
    ax1.set_title('PC Space Overlay', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8, framealpha=0.8)

    # ── Panel 2: Allele frequency scatter ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    # subsample for readability if many SNPs
    idx = np.arange(len(af_real))
    if len(idx) > 20_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, 20_000, replace=False)
    ax2.scatter(af_real[idx], af_synth[idx],
                s=2, alpha=0.25, c='#1D3557', rasterized=True)
    lim = max(af_real.max(), af_synth.max()) * 1.05
    ax2.plot([0, lim], [0, lim], 'r--', lw=1.2, label='y = x')
    ax2.set_xlabel('Real allele frequency', fontsize=9)
    ax2.set_ylabel('Synthetic allele frequency', fontsize=9)
    ax2.set_title(f"Allele Frequency Correlation  (r²={metrics['af_r2']:.4f})",
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)

    # ── Panel 3: Heterozygosity distribution ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(
        min(het_real.min(), het_synth.min()) - 0.01,
        max(het_real.max(), het_synth.max()) + 0.01,
        40
    )
    ax3.hist(het_real,  bins=bins, alpha=0.65, color='#E63946', label='Real (BAQ)',
             density=True, edgecolor='none')
    ax3.hist(het_synth, bins=bins, alpha=0.65, color='#457B9D', label='Synthetic',
             density=True, edgecolor='none')
    ax3.set_xlabel('Per-sample heterozygosity rate', fontsize=9)
    ax3.set_ylabel('Density', fontsize=9)
    ax3.set_title('Heterozygosity Distribution', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)

    # ── Panel 4: Metric summary ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    def _grade(key, val):
        thresholds = {
            'frechet_dist':  [(5,  'excellent'), (15, 'good'), (40, 'fair')],
            'mmd':           [(0.02,'excellent'),(0.1,'good'), (0.3,'fair')],
            'nn_accuracy':   [(0.57,'excellent'),(0.65,'good'),(0.75,'fair')],
            'af_r2':         [(0.999,'excellent'),(0.99,'good'),(0.97,'fair')],
            'het_delta':     [(0.005,'excellent'),(0.01,'good'),(0.02,'fair')],
        }
        colors = {'excellent': '#2DC653', 'good': '#F6AE2D',
                  'fair': '#F26419', 'poor': '#D62839'}
        for thresh, grade in thresholds.get(key, []):
            if key == 'nn_accuracy':
                if val <= thresh: return colors[grade]
            elif key in ('frechet_dist', 'mmd', 'het_delta'):
                if val <= thresh: return colors[grade]
            else:  # af_r2 — higher is better
                if val >= thresh: return colors[grade]
        return colors['poor']

    rows = [
        ('Metric', 'Value', 'Space'),
        ('Fréchet Distance',  f"{metrics['frechet_dist']:.3f}",  'PC'),
        ('MMD',               f"{metrics['mmd']:.4f}",           'PC'),
        ('1-NN Accuracy',     f"{metrics['nn_accuracy']:.3f}",   'PC'),
        ('AF r²',             f"{metrics['af_r2']:.5f}",         'Genotype'),
        ('Het rate (real)',   f"{metrics['het_rate_real']:.4f}", 'Genotype'),
        ('Het rate (synth)',  f"{metrics['het_rate_synth']:.4f}",'Genotype'),
        ('Het delta',         f"{metrics['het_delta']:.4f}",     'Genotype'),
    ]
    col_x = [0.02, 0.52, 0.85]
    y_start, row_h = 0.92, 0.11
    metric_keys = [None, 'frechet_dist', 'mmd', 'nn_accuracy',
                   'af_r2', None, None, 'het_delta']
    for i, (row, key) in enumerate(zip(rows, metric_keys)):
        y = y_start - i * row_h
        is_header = (i == 0)
        for j, (txt, x) in enumerate(zip(row, col_x)):
            weight = 'bold' if is_header else 'normal'
            color  = 'black'
            if not is_header and key and j == 1:
                color = _grade(key, float(txt))
            ax4.text(x, y, txt, transform=ax4.transAxes,
                     fontsize=9.5, fontweight=weight, color=color,
                     va='top')
        if not is_header:
            ax4.axhline(y - row_h * 0.15, xmin=0.01, xmax=0.99,
                        color='#CCCCCC', lw=0.5,
                        transform=ax4.transAxes)

    ax4.set_title('Metric Summary', fontsize=10, fontweight='bold')
    color_legend = [
        ('#2DC653', 'Excellent'), ('#F6AE2D', 'Good'),
        ('#F26419', 'Fair'),      ('#D62839', 'Poor'),
    ]
    for cx, (c, label_) in enumerate(color_legend):
        ax4.text(col_x[0] + cx * 0.24, 0.01, f'■ {label_}',
                 transform=ax4.transAxes, fontsize=7.5, color=c, va='bottom')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {save_path}", flush=True)


# =============================================================================
# Core evaluation
# =============================================================================

def evaluate(real_path: str,
             synth_path: str,
             output_dir: str,
             label: str,
             n_components: int = 50,
             seed: int = 42) -> dict:
    """Evaluate one synthetic file against real BAQ data. Returns metrics dict."""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    stem = Path(synth_path).stem
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: {stem}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading real genotypes...", flush=True)
    real_df  = pd.read_csv(real_path,  header=0)
    print(f"  Real : {real_df.shape[0]} samples × {real_df.shape[1]} SNPs", flush=True)

    print("Loading synthetic genotypes...", flush=True)
    synth_df = pd.read_csv(synth_path, header=0)
    print(f"  Synth: {synth_df.shape[0]} samples × {synth_df.shape[1]} SNPs", flush=True)

    # Align SNP columns
    shared = list(real_df.columns.intersection(synth_df.columns))
    if len(shared) < real_df.shape[1]:
        print(f"  NOTE: aligning to {len(shared)} shared SNPs", flush=True)
    real_df  = real_df[shared]
    synth_df = synth_df[shared]

    real_geno  = real_df.values.astype(np.float32)
    synth_geno = synth_df.values.astype(np.float32)
    del real_df, synth_df

    # ── PCA — fit on real, project both ──────────────────────────────────
    print(f"Fitting PCA (n={n_components}) on real data...", flush=True)
    n_comp = min(n_components, real_geno.shape[0] - 1, synth_geno.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    real_pcs  = pca.fit_transform(real_geno)
    synth_pcs = pca.transform(synth_geno)
    var_exp   = pca.explained_variance_ratio_
    print(f"  PC1+PC2 explain {100*var_exp[:2].sum():.1f}% variance", flush=True)

    # ── PC-space metrics ──────────────────────────────────────────────────
    print("Computing PC-space metrics...", flush=True)
    fd  = frechet_distance(real_pcs, synth_pcs)
    m   = mmd(real_pcs, synth_pcs)
    nna = nn_accuracy(real_pcs, synth_pcs, seed=seed)
    print(f"  Fréchet Distance : {fd:.3f}", flush=True)
    print(f"  MMD              : {m:.5f}", flush=True)
    print(f"  1-NN Accuracy    : {nna:.3f}  (50% = perfect)", flush=True)

    # ── Genotype-space metrics ────────────────────────────────────────────
    print("Computing genotype-space metrics...", flush=True)
    r2, af_r, af_s = allele_freq_r2(real_geno, synth_geno)
    het_r = heterozygosity(real_geno)
    het_s = heterozygosity(synth_geno)
    het_delta = float(abs(het_r.mean() - het_s.mean()))
    print(f"  AF r²            : {r2:.5f}", flush=True)
    print(f"  Het rate (real)  : {het_r.mean():.4f} ± {het_r.std():.4f}", flush=True)
    print(f"  Het rate (synth) : {het_s.mean():.4f} ± {het_s.std():.4f}", flush=True)
    print(f"  Het delta        : {het_delta:.4f}", flush=True)

    metrics = {
        'label':          label,
        'synthetic_file': stem,
        'n_real':         real_geno.shape[0],
        'n_synthetic':    synth_geno.shape[0],
        'n_snps':         len(shared),
        'frechet_dist':   round(fd,  4),
        'mmd':            round(m,   6),
        'nn_accuracy':    round(nna, 4),
        'af_r2':          round(r2,  6),
        'het_rate_real':  round(float(het_r.mean()), 5),
        'het_rate_synth': round(float(het_s.mean()), 5),
        'het_delta':      round(het_delta, 5),
    }

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, 'metrics.csv')
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"  Metrics saved: {csv_path}", flush=True)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_path = os.path.join(output_dir, 'eval_panels.png')
    make_figure(
        real_pcs=real_pcs[:, :2],
        synth_pcs=synth_pcs[:, :2],
        af_real=af_r,
        af_synth=af_s,
        het_real=het_r,
        het_synth=het_s,
        metrics=metrics,
        label=label,
        save_path=plot_path,
    )

    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    return metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate synthetic genotypes against real BAQ data.'
    )
    parser.add_argument('--real',         required=True,
                        help='Real BAQ genotype CSV (samples × SNPs)')
    parser.add_argument('--synthetic',    required=True, nargs='+',
                        help='Synthetic genotype CSV(s). Multiple files = batch mode.')
    parser.add_argument('--output_dir',   required=True,
                        help='Directory for metrics CSV(s) and plots.')
    parser.add_argument('--label',        default='',
                        help='Run label prefix (prepended to each stem in batch mode).')
    parser.add_argument('--n_components', type=int, default=50)
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    all_metrics = []
    for synth_path in args.synthetic:
        stem       = Path(synth_path).stem
        run_label  = f"{args.label}/{stem}" if args.label else stem
        run_outdir = os.path.join(args.output_dir, stem)
        m = evaluate(
            real_path    = args.real,
            synth_path   = synth_path,
            output_dir   = run_outdir,
            label        = run_label,
            n_components = args.n_components,
            seed         = args.seed,
        )
        all_metrics.append(m)

    if len(all_metrics) > 1:
        summary_path = os.path.join(args.output_dir, 'metrics_all.csv')
        pd.DataFrame(all_metrics).to_csv(summary_path, index=False)
        print(f"\nBatch summary saved: {summary_path}", flush=True)
        print(pd.DataFrame(all_metrics).to_string(index=False), flush=True)
