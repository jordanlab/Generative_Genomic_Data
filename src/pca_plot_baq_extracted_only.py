import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Plot PCA scores from multiple datasets with configurable filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────
    parser.add_argument('--input', type=str,
                        default=str(PROJECT_ROOT / "results/PCA/pca_scores_3_datasets.csv"),
                        help='Path to input PCA scores CSV.')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Path to save PCA plot PNG. Defaults to results/PCA/pca_plot_<datasets>.png.')
    parser.add_argument('--output-filtered', type=str, default=None,
                        help='Path to save filtered PCA scores CSV. Defaults to results/PCA/pca_scores_filtered_<datasets>.csv.')

    # ── Dataset selection ────────────────────────────────────────
    parser.add_argument('--datasets', type=str, nargs='+', default=['BAQ', 'Extracted_ChrAll'],
                        help='Dataset names to include in plot (space-separated). '
                             'E.g., --datasets BAQ Extracted_ChrAll Generated')

    # ── Plot styling ─────────────────────────────────────────────
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title. Defaults to "PCA Scores: <datasets>".')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7.5, 6],
                        help='Figure size (width height).')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved plot.')
    parser.add_argument('--pc-x', type=int, default=1,
                        help='First principal component to plot (1-indexed).')
    parser.add_argument('--pc-y', type=int, default=2,
                        help='Second principal component to plot (1-indexed).')

    return parser


def main():
    args = build_argparser().parse_args()

    # ── Load data ────────────────────────────────────────────────
    print(f"[INFO] Loading PCA scores from {args.input}", flush=True)
    df = pd.read_csv(args.input)

    # ── Filter datasets ─────────────────────────────────────────
    keep = set(args.datasets)
    df2 = df[df["source"].isin(keep)].copy()

    if df2.empty:
        raise ValueError(f"No rows found for datasets: {keep}")

    print(f"[INFO] Keeping datasets: {', '.join(keep)}", flush=True)
    print(f"[INFO] Filtered to {len(df2)} rows", flush=True)

    # ── Set default output paths ────────────────────────────────
    datasets_str = '_'.join(sorted([d.replace(' ', '_') for d in keep]))
    if args.output_plot is None:
        args.output_plot = str(PROJECT_ROOT / f"results/PCA/pca_plot_{datasets_str}.png")
    if args.output_filtered is None:
        args.output_filtered = str(PROJECT_ROOT / f"results/PCA/pca_scores_filtered_{datasets_str}.csv")

    # ── Save filtered scores ────────────────────────────────────
    out_filtered = Path(args.output_filtered)
    out_filtered.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_filtered, index=False)
    print(f"[INFO] Saved filtered scores to {out_filtered}", flush=True)

    # ── Create plot ─────────────────────────────────────────────
    pc_x_col = f"PC{args.pc_x}"
    pc_y_col = f"PC{args.pc_y}"

    if pc_x_col not in df2.columns or pc_y_col not in df2.columns:
        raise ValueError(f"Columns {pc_x_col} or {pc_y_col} not found in PCA scores.")

    plt.figure(figsize=args.figsize)

    # ── Color palette ────────────────────────────────────────────
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H']

    for idx, dataset_name in enumerate(sorted(keep)):
        data = df2[df2["source"] == dataset_name]
        if data.empty:
            continue
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.scatter(
            data[pc_x_col], data[pc_y_col],
            c=color, marker=marker, alpha=0.75,
            edgecolor="white", linewidth=0.6, label=dataset_name
        )

    plt.xlabel(pc_x_col)
    plt.ylabel(pc_y_col)

    if args.title is None:
        args.title = f"PCA Scores: {' vs '.join(sorted(keep))}"
    plt.title(args.title)

    plt.legend(frameon=False)
    plt.grid(alpha=0.15)
    plt.tight_layout()

    # ── Save plot ───────────────────────────────────────────────
    out_plot = Path(args.output_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Saved plot to {out_plot}", flush=True)


if __name__ == "__main__":
    main()
