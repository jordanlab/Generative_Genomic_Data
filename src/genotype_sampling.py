"""Post-hoc genotype samplers and matched continuous/discrete diagnostics.

Sampling changes no WGAN training gradients.  All methods share an underlying
continuous dosage matrix and PCA basis for scientifically controlled contrasts.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('binomial', 'nearest', 'categorical', 'ternary', 'ternary_normal')


def sample_dosages(raw, method='binomial', seed=42, temperature=0.35):
    raw = np.asarray(raw)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)):
        raise ValueError('Generator output must be a finite two-dimensional dosage array.')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    if method not in METHODS:
        raise ValueError(f'Unknown sampling method: {method}; choose from {METHODS}')
    rng = np.random.default_rng(seed)
    if method == 'binomial':
        return rng.binomial(2, np.clip(raw / 2., 0., 1.)).astype(np.int8)
    if method == 'nearest':
        return np.clip(np.floor(raw + 0.5), 0, 2).astype(np.int8)
    if method == 'categorical':
        # Avoid an enormous (n_samples, n_snps, 3) array for genome-wide
        # sampling; process disjoint chunks without changing RNG draw order.
        flat = raw.ravel()
        out = np.empty(flat.size, dtype=np.int8)
        for begin in range(0, flat.size, 500_000):
            stop = min(begin + 500_000, flat.size)
            chunk = flat[begin:stop].astype(np.float64)
            scores = -((chunk[:, None] - np.arange(3)) / temperature) ** 2 / 2.
            scores -= np.max(scores, axis=1, keepdims=True)
            prob = np.exp(scores)
            prob /= prob.sum(axis=1, keepdims=True)
            u = rng.random(stop - begin)
            out[begin:stop] = ((u > prob[:, 0]).astype(np.int8) +
                                (u > prob[:, 0] + prob[:, 1]).astype(np.int8))
        return out.reshape(raw.shape)
    centered = np.tanh((raw - 1.) / temperature)
    if method == 'ternary_normal':
        centered = np.clip(centered + rng.normal(0, temperature, raw.shape), -1, 1)
    return (1 + (centered > 0.5).astype(np.int8) -
            (centered < -0.5).astype(np.int8))


def inspect_raw(raw, output_dir, seed=42, snp_names=None):
    """Audit *unclipped* synthetic dosages and exactly what binomial clipping does.

    All summary statistics are computed on the full matrix; histograms subsample
    at most 500k entries.  Per-SNP counts are preserved, not resampled.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError('Expected a nonempty finite (samples, SNPs) dosage matrix')
    below, above = raw < 0, raw > 2
    clipped = below | above
    v = raw.ravel()
    # Global counts and extreme values are exact. Quantiles/plots sample large
    # matrices to avoid expensive multi-pass sorting of tens of millions of SNPs.
    quantile_sample = (np.random.default_rng(seed).choice(v, size=500000, replace=False)
                       if v.size > 500000 else v)
    qs = (0, .01, .05, .25, .5, .75, .95, .99, 1)
    q_values = np.quantile(quantile_sample, qs)
    stats = {'min': float(v.min()), 'max': float(v.max()),
             'mean': float(v.mean()), 'std': float(v.std()),
             'fraction_below_zero': float(below.mean()),
             'fraction_above_two': float(above.mean()),
             'fraction_clipped_by_binomial': float(clipped.mean()),
             'n_entries': int(v.size), 'n_samples': int(raw.shape[0]),
             'n_snps': int(raw.shape[1]),
             'n_below_zero': int(below.sum()), 'n_above_two': int(above.sum()),
             'n_clipped_by_binomial': int(clipped.sum()),
             'mean_absolute_clipping_change': float(
                 (np.maximum(-raw, 0) + np.maximum(raw - 2, 0)).mean()),
             'quantiles': {str(q): float(y) for q, y in zip(qs, q_values)},
             'quantile_sample_size': int(quantile_sample.size),
             'quantiles_approximate': bool(v.size > 500000),
             'binomial_probability_definition': 'p=clip(continuous_dosage/2,0,1)',
             'note': 'Other samplers do not necessarily divide by 2; these bounds quantify diploid-dosage violations.'}
    (output_dir / 'preclip_dosage_summary.json').write_text(json.dumps(stats, indent=2))
    labels = snp_names if snp_names is not None else [f'SNP_{i}' for i in range(raw.shape[1])]
    if len(labels) != raw.shape[1]:
        raise ValueError('snp_names length does not match dosage columns')
    with (output_dir / 'preclip_per_snp.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['snp', 'n_below_zero', 'n_above_two', 'fraction_clipped',
                         'mean_raw_dosage', 'min_raw_dosage', 'max_raw_dosage'])
        for j, label in enumerate(labels):
            col = raw[:, j]
            writer.writerow([label, int(below[:, j].sum()), int(above[:, j].sum()),
                             float(clipped[:, j].mean()), float(col.mean()),
                             float(col.min()), float(col.max())])
    v = quantile_sample
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(v, bins=90, density=True, alpha=.75)
    ax.axvline(0, color='red', linestyle='--', label='diploid range [0, 2]')
    ax.axvline(2, color='red', linestyle='--')
    ax.axvline(1, color='black', alpha=.4, linestyle=':')
    ax.set(xlabel='Continuous expected genotype dosage (before clipping)',
           ylabel='Density', title='Generator output before genotype sampling')
    ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / 'preclip_dosage_distribution.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(v, bins=90, density=True, alpha=.45, label='Raw dosage')
    ax.hist(np.clip(v, 0, 2), bins=90, density=True, histtype='step',
            linewidth=1.7, label='After clip(d, 0, 2)')
    ax.set(xlabel='Expected genotype dosage', ylabel='Density',
           title=f'Clipping comparison: {100 * stats["fraction_clipped_by_binomial"]:.2f}% affected')
    ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / 'dosage_before_vs_after_clipping.png', dpi=160)
    plt.close(fig)
    return stats


def plot_pca_sampling_diagnostics(real_pcs, continuous_pcs, sampled_pcs, output_dir,
                                  explained_variance=None, label='Synthetic'):
    """Same PC basis for real, pre-sampling and each sampled synthetic matrix.

    Output layout: plots/pca_overlay_continuous.png,
    plots/pca_overlay_METHOD.png, plots/pca_score_distributions_METHOD.png,
    and plots/pca_sampling_shift_summary.csv/.json.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    real_pcs = np.asarray(real_pcs, dtype=float)
    continuous_pcs = np.asarray(continuous_pcs, dtype=float)
    if real_pcs.ndim != 2 or continuous_pcs.ndim != 2 or real_pcs.shape[1] != continuous_pcs.shape[1]:
        raise ValueError('Real and continuous PCA arrays must be 2D and use the same PCA basis.')
    n_pc = real_pcs.shape[1]
    if n_pc < 1:
        raise ValueError('No PCA components for diagnostic')
    ev = np.asarray(explained_variance) if explained_variance is not None else np.array([])

    def pc_label(j):
        return f'PC{j+1}' + (f' ({100*ev[j]:.1f}%)' if len(ev) > j else '')

    def overlay(synth, path, title):
        fig, ax = plt.subplots(figsize=(7, 6))
        if n_pc >= 2:
            ax.scatter(real_pcs[:, 0], real_pcs[:, 1], s=19, alpha=.6, label='Real')
            ax.scatter(synth[:, 0], synth[:, 1], s=19, alpha=.6, label=label)
            ax.set(xlabel=pc_label(0), ylabel=pc_label(1))
        else:
            ax.hist(real_pcs[:, 0], bins=25, alpha=.5, label='Real')
            ax.hist(synth[:, 0], bins=25, alpha=.5, label=label)
            ax.set(xlabel=pc_label(0), ylabel='Count')
        ax.set_title(title); ax.legend(); fig.tight_layout()
        fig.savefig(path, dpi=150); plt.close(fig)

    overlay(continuous_pcs, dest / 'pca_overlay_continuous.png', 'Real vs continuous generator output')
    records = []
    for method, pcs in sampled_pcs.items():
        if method not in METHODS:
            raise ValueError(f'Unexpected sampling method {method}')
        pcs = np.asarray(pcs, dtype=float)
        if pcs.ndim != 2 or pcs.shape[1] != n_pc:
            raise ValueError(f'{method}: PCA component mismatch')
        overlay(pcs, dest / f'pca_overlay_{method}.png', f'Real vs sampled genotypes ({method})')
        n_show = min(6, n_pc)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for j, ax in enumerate(axes.flat):
            if j >= n_show:
                ax.axis('off'); continue
            arrays = (real_pcs[:, j], continuous_pcs[:, j], pcs[:, j])
            low = min(float(x.min()) for x in arrays)
            high = max(float(x.max()) for x in arrays)
            bins = np.linspace(low - .5 if low == high else low,
                               high + .5 if low == high else high, 31)
            for x, title in zip(arrays, ('Real', 'Continuous', f'Sampled ({method})')):
                ax.hist(x, bins=bins, density=True, alpha=.38, label=title)
            ax.set(xlabel=pc_label(j), ylabel='Density')
            ax.legend(fontsize=7)
        fig.suptitle(f'PCA-score distributions: {method}')
        fig.tight_layout(); fig.savefig(dest / f'pca_score_distributions_{method}.png', dpi=150)
        plt.close(fig)
        for j in range(n_pc):
            records.append({
                'method': method, 'pc': j + 1,
                'real_mean': float(real_pcs[:, j].mean()),
                'real_sd': float(real_pcs[:, j].std()),
                'continuous_mean': float(continuous_pcs[:, j].mean()),
                'continuous_sd': float(continuous_pcs[:, j].std()),
                'sampled_mean': float(pcs[:, j].mean()),
                'sampled_sd': float(pcs[:, j].std()),
                'sampling_mean_shift': float(pcs[:, j].mean() - continuous_pcs[:, j].mean()),
                'sampling_sd_shift': float(pcs[:, j].std() - continuous_pcs[:, j].std()),
            })
    import pandas as pd
    pd.DataFrame(records).to_csv(dest / 'pca_sampling_shift_summary.csv', index=False)
    (dest / 'pca_sampling_shift_summary.json').write_text(json.dumps(records, indent=2))
    return records
