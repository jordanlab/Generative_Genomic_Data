import pandas as pd
import numpy as np
import csv
import time
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

from plot_colors import SAMPLE_COLORS, SYNTHETIC_COLOR


def resolve_color(name, i, fallback_palette):
    """Registry color for a known population/synthetic name, else cycle the fallback palette."""
    if name in SAMPLE_COLORS:
        return SAMPLE_COLORS[name]
    if 'synth' in name.lower():
        return SYNTHETIC_COLOR
    return fallback_palette[i % len(fallback_palette)]


def inject_slurm_id(filepath, slurm_id):
    """Insert SLURM ID before file extension if not already present."""
    if slurm_id is None:
        return filepath
    p = Path(filepath)
    if f"_{slurm_id}" not in p.stem:
        return str(p.parent / f"{p.stem}_{slurm_id}{p.suffix}")
    return filepath


# ═══════════════════════════════════════════════════════════════════
#  FAST BATCH MODE  (--reference + --generated)
#  Reads reference once, loops over generated files.
#  Bulk pandas read — fast for <1000 samples × 120K SNPs.
# ═══════════════════════════════════════════════════════════════════

def run_single_pca(ref_df, gen_path, ref_name, gen_name, n_components):
    """Run PCA for one reference+generated pair.  Returns (PC, sources, evr, tag)."""
    t0 = time.time()
    gen_df = pd.read_csv(gen_path)
    print(f"  [{gen_name}] loaded {gen_df.shape[0]} rows × {gen_df.shape[1]} cols", flush=True)

    # Common columns
    common = [c for c in ref_df.columns if c in gen_df.columns]
    if len(common) == 0:
        print(f"  [{gen_name}] ERROR: no common columns with reference!", flush=True)
        return None
    print(f"  [{gen_name}] {len(common)} common SNPs", flush=True)

    X_ref = ref_df[common].values.astype(np.float32)
    X_gen = gen_df[common].values.astype(np.float32)
    np.nan_to_num(X_ref, copy=False)
    np.nan_to_num(X_gen, copy=False)
    X = np.vstack([X_ref, X_gen])
    sources = np.array([ref_name] * X_ref.shape[0] + [gen_name] * X_gen.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    pca = PCA(n_components=n_components)
    PC = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    tag = Path(gen_path).stem
    print(f"  [{gen_name}] PCA done in {time.time()-t0:.1f}s  "
          f"(PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%)", flush=True)
    return PC, sources, evr, tag


def save_pca_plot(PC, sources, evr, dataset_names, tag, args):
    """Create and save one PCA scatter plot + scores CSV."""
    pc_x_idx = args.pc_x - 1
    pc_y_idx = args.pc_y - 1

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'H']

    plt.figure(figsize=args.figsize)
    for i, name in enumerate(dataset_names):
        mask = sources == name
        plt.scatter(PC[mask, pc_x_idx], PC[mask, pc_y_idx],
                    c=resolve_color(name, i, colors),
                    marker=markers[i % len(markers)],
                    alpha=0.75, edgecolor="white", linewidth=0.6, label=name)

    plt.xlabel(f"PC{args.pc_x} ({evr[pc_x_idx]*100:.1f}% var)")
    plt.ylabel(f"PC{args.pc_y} ({evr[pc_y_idx]*100:.1f}% var)")
    plt.title(args.title or f"PCA: {' vs '.join(dataset_names)}")
    plt.legend(frameon=False)
    plt.grid(alpha=0.15)
    plt.tight_layout()

    plot_path = PROJECT_ROOT / f"results/PCA/plots/pca_{tag}.png"
    scores_path = PROJECT_ROOT / f"results/PCA/scores/pca_{tag}.csv"
    plot_path = Path(inject_slurm_id(str(plot_path), args.slurm_id))
    scores_path = Path(inject_slurm_id(str(scores_path), args.slurm_id))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot   → {plot_path}", flush=True)

    pc_cols = [f"PC{i+1}" for i in range(args.n_components)]
    pd.DataFrame(PC, columns=pc_cols).assign(source=sources).to_csv(scores_path, index=False)
    print(f"  Saved scores → {scores_path}", flush=True)


def batch_main(args):
    """Fast batch mode: one reference vs multiple generated files."""
    print("[INFO] Batch PCA — backend: CPU (scikit-learn)", flush=True)
    print(f"[INFO] Reference: {args.reference}", flush=True)
    print(f"[INFO] Generated files: {len(args.generated)}", flush=True)

    t0 = time.time()
    ref_df = pd.read_csv(args.reference)
    ref_name = args.ref_name or Path(args.reference).stem
    print(f"[INFO] Reference loaded: {ref_df.shape[0]} × {ref_df.shape[1]} "
          f"in {time.time()-t0:.1f}s", flush=True)

    for gen_path in args.generated:
        gen_p = Path(gen_path)
        if not gen_p.exists():
            print(f"[WARN] Skipping (not found): {gen_path}", flush=True)
            continue

        gen_name = args.gen_name or "Generated"
        print(f"\n{'='*60}\nProcessing: {gen_p.name}\n{'='*60}", flush=True)

        result = run_single_pca(ref_df, gen_path, ref_name, gen_name,
                    args.n_components)
        if result is None:
            continue
        PC, sources, evr, tag = result
        save_pca_plot(PC, sources, evr, [ref_name, gen_name], tag, args)

    print(f"\n[INFO] All done! Processed {len(args.generated)} file(s).", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  LEGACY MODE  (--files)
#  Low-memory incremental PCA with line-by-line CSV reading.
# ═══════════════════════════════════════════════════════════════════

def read_header_fast(csv_path: Path):
    """Read only the first CSV row (header) quickly."""
    with csv_path.open("r", newline="") as f:
        return next(csv.reader(f))


def iter_numeric_batches(csv_path: Path, selected_idx: np.ndarray, batch_rows: int):
    """Yield tiny float32 batches parsed line-by-line (very low memory)."""
    rows = []
    max_idx = int(selected_idx.max()) if selected_idx.size else -1
    with csv_path.open("r") as f:
        _ = f.readline()
        for line in f:
            arr = np.fromstring(line, sep=",", dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.size <= max_idx:
                raise ValueError(f"Row has {arr.size} cols, expected ≥ {max_idx+1} in {csv_path}")
            row = arr[selected_idx]
            np.nan_to_num(row, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(row)
            if len(rows) >= batch_rows:
                yield np.vstack(rows).astype(np.float32, copy=False)
                rows.clear()
    if rows:
        yield np.vstack(rows).astype(np.float32, copy=False)


def legacy_main(args):
    """Original low-memory multi-file PCA (--files mode)."""
    file_paths = [Path(f) for f in args.files]
    if not all(f.exists() for f in file_paths):
        raise FileNotFoundError(f"Not found: {args.files}")

    dataset_names = args.names if args.names else [f.stem for f in file_paths]
    if len(dataset_names) != len(file_paths):
        raise ValueError("--names count must match --files count")

    print(f"[INFO] Processing {len(file_paths)} files:", flush=True)
    for fp, name in zip(file_paths, dataset_names):
        print(f"  - {fp} → {name}", flush=True)

    if args.pc_x < 1 or args.pc_y < 1 or args.pc_x > args.n_components or args.pc_y > args.n_components:
        raise ValueError(f"PC indices must be between 1 and {args.n_components}")

    # 1) Common columns
    headers = [read_header_fast(fp) for fp in file_paths]
    common_set = set(headers[0])
    for h in headers[1:]:
        common_set &= set(h)
    common = [c for c in headers[0] if c in common_set]
    if not common:
        raise ValueError("No common columns found.")
    print(f"[INFO] {len(common)} common columns", flush=True)

    index_maps = []
    for h in headers:
        h_pos = {c: i for i, c in enumerate(h)}
        index_maps.append(np.array([h_pos[c] for c in common], dtype=np.int64))

    # 2) Scaler
    scaler = StandardScaler(copy=False)
    for fp, idx in zip(file_paths, index_maps):
        for X_chunk in iter_numeric_batches(fp, idx, args.batch_rows):
            scaler.partial_fit(X_chunk)

    # 3) PCA fit
    pca = IncrementalPCA(n_components=args.n_components, batch_size=args.batch_rows)
    for fp, idx in zip(file_paths, index_maps):
        for X_chunk in iter_numeric_batches(fp, idx, args.batch_rows):
            pca.partial_fit(scaler.transform(X_chunk).astype(np.float32, copy=False))

    # 4) Transform
    pcs, sources = [], []
    for fp, idx, name in zip(file_paths, index_maps, dataset_names):
        for X_chunk in iter_numeric_batches(fp, idx, args.batch_rows):
            pc_chunk = pca.transform(scaler.transform(X_chunk).astype(np.float32, copy=False))
            pcs.append(pc_chunk.astype(np.float32))
            sources.extend([name] * pc_chunk.shape[0])

    PC = np.vstack(pcs)
    sources = np.array(sources)
    evr = pca.explained_variance_ratio_

    # 5) Plot
    pc_x_idx, pc_y_idx = args.pc_x - 1, args.pc_y - 1
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', '^', 's', 'D', 'v', '<']

    plt.figure(figsize=args.figsize)
    for i, name in enumerate(dataset_names):
        mask = sources == name
        plt.scatter(PC[mask, pc_x_idx], PC[mask, pc_y_idx],
                    c=resolve_color(name, i, colors), marker=markers[i % len(markers)],
                    alpha=0.75, edgecolor="white", linewidth=0.6, label=name)
    plt.xlabel(f"PC{args.pc_x} ({evr[pc_x_idx]*100:.1f}% var)")
    plt.ylabel(f"PC{args.pc_y} ({evr[pc_y_idx]*100:.1f}% var)")
    plt.title(args.title or f"PCA of {len(dataset_names)} Datasets")
    plt.legend(frameon=False); plt.grid(alpha=0.15); plt.tight_layout()

    # 6) Save
    tag = args.plot_name or f"pca_{len(dataset_names)}_datasets"
    if args.slurm_id:
        tag += f"_{args.slurm_id}"
    if args.output_plot is None:
        args.output_plot = str(PROJECT_ROOT / f"results/PCA/plots/{tag}.png")
    if args.output_scores is None:
        args.output_scores = str(PROJECT_ROOT / f"results/PCA/scores/{tag}.csv")

    args.output_plot = inject_slurm_id(args.output_plot, args.slurm_id)
    args.output_scores = inject_slurm_id(args.output_scores, args.slurm_id)
    out_plot, out_scores = Path(args.output_plot), Path(args.output_scores)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    out_scores.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_plot, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Saved plot to {out_plot}", flush=True)
    pc_cols = [f"PC{i+1}" for i in range(args.n_components)]
    pd.DataFrame(PC, columns=pc_cols).assign(source=sources).to_csv(out_scores, index=False)
    print(f"[INFO] Saved scores to {out_scores}", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  CLI & dispatch
# ═══════════════════════════════════════════════════════════════════

def build_argparser():
    p = argparse.ArgumentParser(
        description="PCA for genomic data — batch mode + legacy mode (CPU/scikit-learn).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Batch mode (fast) ────────────────────────────────────────
    batch = p.add_argument_group("Batch mode (recommended)",
        "Compare one reference file against multiple generated files. "
        "Reads reference once, produces a separate plot per generated file.")
    batch.add_argument('--reference', type=str, default=None,
                       help='Reference (real) data CSV.')
    batch.add_argument('--generated', type=str, nargs='+', default=None,
                       help='One or more generated data CSVs.')
    batch.add_argument('--ref-name', type=str, default=None,
                       help='Display name for reference dataset.')
    batch.add_argument('--gen-name', type=str, default=None,
                       help='Display name for generated datasets.')

    # ── Legacy mode ──────────────────────────────────────────────
    legacy = p.add_argument_group("Legacy mode (--files)",
        "Multi-file incremental PCA with low-memory processing.")
    legacy.add_argument('--files', type=str, nargs='+', default=None,
                        help='Input CSV files (space-separated).')
    legacy.add_argument('--names', type=str, nargs='+', default=None,
                        help='Dataset names for each --files entry.')

    # ── Output (legacy mode) ─────────────────────────────────────
    p.add_argument('--output-plot', type=str, default=None)
    p.add_argument('--output-scores', type=str, default=None)

    # ── PCA parameters ───────────────────────────────────────────
    p.add_argument('--n-components', type=int, default=2)
    p.add_argument('--pc-x', type=int, default=1)
    p.add_argument('--pc-y', type=int, default=2)
    p.add_argument('--batch-rows', type=int, default=4,
                   help='Rows per batch (legacy incremental mode only).')

    # ── Plot styling ─────────────────────────────────────────────
    p.add_argument('--title', type=str, default=None)
    p.add_argument('--figsize', type=float, nargs=2, default=[7.5, 6])
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--plot-name', type=str, default=None)

    # ── SLURM ────────────────────────────────────────────────────
    p.add_argument('--slurm-id', type=str, default=None)

    return p


def main():
    args = build_argparser().parse_args()

    if args.reference and args.generated:
        batch_main(args)
    elif args.files:
        legacy_main(args)
    else:
        print("ERROR: Provide either --reference + --generated (batch mode) "
              "or --files (legacy mode).")
        build_argparser().print_help()
        exit(1)


if __name__ == "__main__":
    main()