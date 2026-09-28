import pandas as pd
import numpy as np
import csv
import time
import argparse
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def inject_slurm_id(filepath, slurm_id):
    """Insert SLURM ID before file extension if not already present."""
    if slurm_id is None:
        return filepath
    p = Path(filepath)
    if f"_{slurm_id}" not in p.stem:  # Avoid double-adding
        return str(p.parent / f"{p.stem}_{slurm_id}{p.suffix}")
    return filepath


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Compute PCA across multiple CSV datasets with low memory footprint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input files ──────────────────────────────────────────────
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help='Paths to input CSV files (space-separated).')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                        help='Dataset names for each file (space-separated). '
                             'Defaults to file stems if not provided.')

    # ── Output paths ─────────────────────────────────────────────
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Path to save PCA plot PNG. Defaults to results/PCA/pca_plot_<N>_datasets.png.')
    parser.add_argument('--output-scores', type=str, default=None,
                        help='Path to save PCA scores CSV. Defaults to results/PCA/pca_scores_<N>_datasets.csv.')

    # ── PCA parameters ───────────────────────────────────────────
    parser.add_argument('--n-components', type=int, default=2,
                        help='Number of principal components to compute.')
    parser.add_argument('--pc-x', type=int, default=1,
                        help='First principal component to plot (1-indexed).')
    parser.add_argument('--pc-y', type=int, default=2,
                        help='Second principal component to plot (1-indexed).')
    parser.add_argument('--batch-rows', type=int, default=4,
                        help='Number of rows per batch for low-memory processing.')

    # ── Plot styling ─────────────────────────────────────────────
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title. Defaults to "PCA of <N> Datasets".')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7.5, 6],
                        help='Figure size (width height).')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved plot.')

    # ── Output filename customization ───────────────────────────
    parser.add_argument('--plot-name', type=str, default=None,
                        help='Base name for PCA plot and scores files (no extension). If provided, overrides default naming.')

    # ── SLURM Integration ────────────────────────────────────────
    parser.add_argument('--slurm-id', type=str, default=None,
                        help='SLURM job ID to include in output filenames for tracking.')

    return parser


def read_header_fast(csv_path: Path):
    """Read only the first CSV row (header) quickly without pandas parsing overhead."""
    with csv_path.open("r", newline="") as f:
        return next(csv.reader(f))


def iter_numeric_batches(csv_path: Path, selected_idx: np.ndarray, batch_rows: int):
    """Yield tiny float32 batches parsed line-by-line (very low memory footprint)."""
    rows = []
    max_idx = int(selected_idx.max()) if selected_idx.size else -1
    with csv_path.open("r") as f:
        _ = f.readline()  # skip header
        for line in f:
            arr = np.fromstring(line, sep=",", dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.size <= max_idx:
                raise ValueError(f"Row has {arr.size} columns, expected at least {max_idx + 1} in {csv_path}")
            row = arr[selected_idx]
            np.nan_to_num(row, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(row)
            if len(rows) >= batch_rows:
                yield np.vstack(rows).astype(np.float32, copy=False)
                rows.clear()
    if rows:
        yield np.vstack(rows).astype(np.float32, copy=False)


def main():
    args = build_argparser().parse_args()

    # ── Parse file paths and names ───────────────────────────────
    file_paths = [Path(f) for f in args.files]
    if not all(f.exists() for f in file_paths):
        raise FileNotFoundError(f"One or more input files not found: {args.files}")

    if args.names is None:
        dataset_names = [f.stem for f in file_paths]
    else:
        if len(args.names) != len(file_paths):
            raise ValueError(f"Number of --names ({len(args.names)}) must match number of --files ({len(file_paths)})")
        dataset_names = args.names

    print(f"[INFO] Processing {len(file_paths)} files:", flush=True)
    for fp, name in zip(file_paths, dataset_names):
        print(f"  - {fp} → {name}", flush=True)

    # ── Validate PCA components ──────────────────────────────────
    if args.pc_x < 1 or args.pc_y < 1 or args.pc_x > args.n_components or args.pc_y > args.n_components:
        raise ValueError(f"PC indices must be between 1 and {args.n_components}")

    # ── 1) Find common columns first (header-only read; memory safe) ──
    print("[INFO] Reading headers to determine common features...", flush=True)
    t0 = time.time()
    headers = [read_header_fast(fp) for fp in file_paths]
    common_set = set(headers[0])
    for h in headers[1:]:
        common_set &= set(h)
    common = [c for c in headers[0] if c in common_set]

    if len(common) == 0:
        raise ValueError("No common columns found across all input files.")

    print(f"[INFO] Found {len(common)} common columns across all {len(file_paths)} files in {time.time() - t0:.2f}s", flush=True)

    # ── Build index mappings for each file ────────────────────────
    index_maps = []
    for h in headers:
        h_pos = {c: i for i, c in enumerate(h)}
        idx = np.array([h_pos[c] for c in common], dtype=np.int64)
        index_maps.append(idx)

    # ── 2) Low-memory scaler fit (pass 1) ────────────────────────
    print(f"[INFO] Fitting scaler in tiny batches (rows/batch={args.batch_rows})...", flush=True)
    t1 = time.time()
    scaler = StandardScaler(copy=False)
    for fp, idx, name in zip(file_paths, index_maps, dataset_names):
        n_seen = 0
        for X_chunk in iter_numeric_batches(fp, idx, args.batch_rows):
            scaler.partial_fit(X_chunk)
            n_seen += X_chunk.shape[0]
            if n_seen % 200 == 0:
                print(f"[INFO] Scaler pass: {name} processed {n_seen} rows", flush=True)
    print(f"[INFO] Scaler fit complete in {time.time() - t1:.2f}s", flush=True)

    # ── 3) Low-memory PCA fit (pass 2) ───────────────────────────
    print(f"[INFO] Fitting IncrementalPCA (n_components={args.n_components}) in chunks...", flush=True)
    t2 = time.time()
    pca = IncrementalPCA(n_components=args.n_components, batch_size=args.batch_rows)
    for fp, idx, name in zip(file_paths, index_maps, dataset_names):
        n_seen = 0
        for X_chunk in iter_numeric_batches(fp, idx, args.batch_rows):
            X_scaled_chunk = scaler.transform(X_chunk).astype(np.float32, copy=False)
            pca.partial_fit(X_scaled_chunk)
            n_seen += X_chunk.shape[0]
            if n_seen % 200 == 0:
                print(f"[INFO] PCA pass: {name} processed {n_seen} rows", flush=True)
    print(f"[INFO] IncrementalPCA fit complete in {time.time() - t2:.2f}s", flush=True)

    # ── 4) Transform to PC scores (pass 3) ────────────────────────
    print("[INFO] Transforming chunks to PC scores...", flush=True)
    t3 = time.time()
    pcs = []
    sources = []
    for fp, idx, name in zip(file_paths, index_maps, dataset_names):
        n_seen = 0
        for X_chunk in iter_numeric_batches(fp, idx, args.batch_rows):
            X_scaled_chunk = scaler.transform(X_chunk).astype(np.float32, copy=False)
            pc_chunk = pca.transform(X_scaled_chunk).astype(np.float32, copy=False)
            pcs.append(pc_chunk)
            sources.extend([name] * pc_chunk.shape[0])
            n_seen += X_chunk.shape[0]
            if n_seen % 200 == 0:
                print(f"[INFO] Transform pass: {name} processed {n_seen} rows", flush=True)

    PC = np.vstack(pcs)
    sources = np.array(sources)
    evr = pca.explained_variance_ratio_
    print(f"[INFO] Transform complete in {time.time() - t3:.2f}s; total samples={PC.shape[0]}", flush=True)
    for i, var in enumerate(evr):
        print(f"[INFO] PC{i+1} explained variance: {var:.4f}", flush=True)

    # ── 5) Plot with different colors/markers for each dataset ──
    pc_x_idx = args.pc_x - 1  # Convert to 0-indexed
    pc_y_idx = args.pc_y - 1

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'H']

    plt.figure(figsize=args.figsize)
    for idx, name in enumerate(dataset_names):
        mask = sources == name
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.scatter(PC[mask, pc_x_idx], PC[mask, pc_y_idx],
                    c=color, marker=marker, alpha=0.75,
                    edgecolor="white", linewidth=0.6, label=name)

    plt.xlabel(f"PC{args.pc_x} ({evr[pc_x_idx]*100:.1f}% var)")
    plt.ylabel(f"PC{args.pc_y} ({evr[pc_y_idx]*100:.1f}% var)")

    if args.title is None:
        args.title = f"PCA of {len(dataset_names)} Dataset{'s' if len(dataset_names) > 1 else ''}"
    plt.title(args.title)

    plt.legend(frameon=False)
    plt.grid(alpha=0.15)
    plt.tight_layout()

    # ── 6) Save plot and scores ──────────────────────────────────
    if args.plot_name:
        base_name = args.plot_name
        if args.slurm_id:
            base_name += f"_{args.slurm_id}"
        plot_filename = base_name + ".png"
        scores_filename = base_name + ".csv"
        if args.output_plot is None:
            args.output_plot = str(PROJECT_ROOT / f"results/PCA/plots/{plot_filename}")
        if args.output_scores is None:
            args.output_scores = str(PROJECT_ROOT / f"results/PCA/scores/{scores_filename}")
    else:
        if args.output_plot is None:
            plot_filename = f"pca_plot_{len(dataset_names)}_datasets"
            if args.slurm_id:
                plot_filename += f"_{args.slurm_id}"
            plot_filename += ".png"
            args.output_plot = str(PROJECT_ROOT / f"results/PCA/plots/{plot_filename}")
        if args.output_scores is None:
            scores_filename = f"pca_scores_{len(dataset_names)}_datasets"
            if args.slurm_id:
                scores_filename += f"_{args.slurm_id}"
            scores_filename += ".csv"
            args.output_scores = str(PROJECT_ROOT / f"results/PCA/scores/{scores_filename}")

    out_plot = Path(args.output_plot)
    out_scores = Path(args.output_scores)
    
    # Inject SLURM ID into filenames if provided
    args.output_plot = inject_slurm_id(args.output_plot, args.slurm_id)
    args.output_scores = inject_slurm_id(args.output_scores, args.slurm_id)
    
    out_plot = Path(args.output_plot)
    out_scores = Path(args.output_scores)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    out_scores.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_plot, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Saved plot to {out_plot}", flush=True)

    pc_cols = [f"PC{i+1}" for i in range(args.n_components)]
    pd.DataFrame(PC, columns=pc_cols).assign(source=sources).to_csv(out_scores, index=False)
    print(f"[INFO] Saved PCA scores to {out_scores}", flush=True)


if __name__ == "__main__":
    main()