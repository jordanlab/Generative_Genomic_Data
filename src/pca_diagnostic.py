"""
Standalone PCA diagnostic: fit on reference, project a query population.
Outputs a plot colored by superpopulation (AFR/EUR/AMR) + query samples.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from plot_colors import get_real_color

# ── Population map (matches conditional_wgan_pipeline.py) ───────────────────
_POP_TO_SUPERPOP = {
    'ESN': 'AFR', 'GWD': 'AFR', 'YRI': 'AFR',
    'GBR': 'EUR', 'IBS': 'EUR', 'TSI': 'EUR',
    'Kar': 'AMR', 'Karitiana': 'AMR',
    'May': 'AMR', 'Maya':      'AMR',
    'PEL': 'AMR',
    'Pim': 'AMR', 'Pima':      'AMR',
    'Sur': 'AMR', 'Surui':     'AMR',
}
_SUPERPOPS  = ['AFR', 'EUR', 'AMR']
_POP_COLORS = {'AFR': '#2166AC', 'EUR': '#E87D00', 'AMR': '#89CFF0', 'OTHER': '#AAAAAA'}


def load_population_labels(fam_path):
    labels = []
    with open(fam_path) as fh:
        for line in fh:
            cols = line.strip().split()
            if not cols:
                continue
            iid    = cols[1]
            prefix = 'forReference'
            pop    = (iid[len(prefix):].split('_')[0]
                      if iid.startswith(prefix) else iid.split('_')[0])
            labels.append(_POP_TO_SUPERPOP.get(pop, 'OTHER'))
    return labels


def main():
    p = argparse.ArgumentParser(description="Diagnostic PCA: reference + query population")
    p.add_argument('--query',        required=True,  help='Query population tensor CSV')
    p.add_argument('--ref',          required=True,  help='Reference tensor CSV')
    p.add_argument('--fam',          required=True,  help='Reference .fam file for population labels')
    p.add_argument('--plot',         required=True,  help='Output PNG path')
    p.add_argument('--n_components', type=int, default=10)
    p.add_argument('--query_name',   default='Query', help='Label for query population in plot')
    p.add_argument('--query_color',  default=None,
                    help='Hex color for query samples. Defaults to the '
                         'registered color for --query_name (see plot_colors.py).')
    args = p.parse_args()
    query_color = args.query_color or get_real_color(args.query_name)

    print(f"Loading reference: {args.ref}", flush=True)
    ref_df = pd.read_csv(args.ref)
    print(f"  {ref_df.shape[0]} samples × {ref_df.shape[1]} SNPs", flush=True)

    print(f"Loading query: {args.query}", flush=True)
    query_df = pd.read_csv(args.query)
    print(f"  {query_df.shape[0]} samples × {query_df.shape[1]} SNPs", flush=True)

    # Common SNPs
    common = [c for c in ref_df.columns if c in query_df.columns]
    print(f"Common SNPs: {len(common)}", flush=True)
    if len(common) == 0:
        sys.exit("ERROR: no common SNP columns between query and reference")

    X_ref   = ref_df[common].values.astype(np.float32)
    X_query = query_df[common].values.astype(np.float32)
    np.nan_to_num(X_ref,   copy=False)
    np.nan_to_num(X_query, copy=False)

    # Population labels from FAM
    pop_labels = load_population_labels(args.fam)
    # FAM may have more entries than tensor rows after QC filtering
    if len(pop_labels) > X_ref.shape[0]:
        pop_labels = pop_labels[:X_ref.shape[0]]
    elif len(pop_labels) < X_ref.shape[0]:
        pop_labels += ['OTHER'] * (X_ref.shape[0] - len(pop_labels))
    pop_labels = np.array(pop_labels)

    # Fit PCA on reference only, project query
    print(f"Fitting PCA ({args.n_components} components) on reference...", flush=True)
    pca = PCA(n_components=args.n_components, random_state=42)
    ref_pcs   = pca.fit_transform(X_ref)
    query_pcs = pca.transform(X_query)
    evr = pca.explained_variance_ratio_
    print(f"  PC1={evr[0]*100:.1f}%  PC2={evr[1]*100:.1f}%", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    for sp in _SUPERPOPS:
        mask = pop_labels == sp
        if mask.any():
            ax.scatter(ref_pcs[mask, 0], ref_pcs[mask, 1],
                       c=_POP_COLORS[sp], alpha=0.40, s=22,
                       label=sp, edgecolors='none', zorder=1)

    other_mask = pop_labels == 'OTHER'
    if other_mask.any():
        ax.scatter(ref_pcs[other_mask, 0], ref_pcs[other_mask, 1],
                   c=_POP_COLORS['OTHER'], alpha=0.30, s=15,
                   label='OTHER (ref)', edgecolors='none', zorder=1)

    ax.scatter(query_pcs[:, 0], query_pcs[:, 1],
               c=query_color, alpha=0.9, s=50,
               label=f'{args.query_name} (n={query_pcs.shape[0]})',
               edgecolors='white', linewidths=0.4, zorder=2)

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=12)
    ax.set_title(f"PCA Diagnostic: {args.query_name} vs Reference  |  "
                 f"{len(common):,} common SNPs", fontsize=9)
    ax.legend(fontsize=9, markerscale=1.1, loc='best')
    ax.grid(alpha=0.12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    plt.savefig(args.plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {args.plot}", flush=True)


if __name__ == "__main__":
    main()
