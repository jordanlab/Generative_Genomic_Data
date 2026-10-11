# Conditional WGAN improvement bundle

This bundle keeps `conditional_wgan_pipeline_extended.py` backward compatible by default and adds opt-in experimental backends/features.

## Main switches

- `--model_backend baseline_pca` — current PCA-space concatenation model (default).
- `--model_backend pca_film` — PCA-space FiLM conditioning at every hidden block.
- `--model_backend genotype_conv` — direct SNP-space residual Conv1D WGAN-GP.
- `--balanced_sampling` — inverse-frequency query/reference group sampler (FiLM/Conv backends).
- `--pack_size 2` — PacGAN-style critic packing (FiLM/Conv backends).
- `--condition_source external --condition_input NAME=PATH` — external AFR/EUR/AMR coefficients.

## External ancestry file

Either:

1. CSV/TSV containing named `AFR`, `EUR`, `AMR` columns, or
2. a three-column headerless ADMIXTURE-like `.Q` file. Use `--external_condition_order` to specify its column order.

Rows must be in the same order as the genotype CSV.

## Multi-run selection

```bash
python select_best_cwgan_run.py 'runs/20261003_*' --output model_comparison.csv
```

This is a within-candidate comparison on the same holdout data, not a universal quality score.

## Privacy audit

```bash
python evaluate_genotype_privacy.py \
  --train_real real_train.csv \
  --holdout_real real_holdout.csv \
  --synthetic runs/.../outputs/synthetic_genotypes.csv \
  --output privacy_metrics.json
```

The privacy diagnostics do **not** provide differential-privacy guarantees.

## Validation-selected checkpoints and dataset subsetting (latest update)

The pipeline now saves `checkpoint_last.pt` on every run. When `--val_fraction > 0`,
it scores synthetic data generated from held-out validation conditions and saves
`checkpoint_best.pt` for the lowest validation selection score among evaluated epochs.
When `--val_fraction 0`, all rows are used for fitting and the same checkpoint metric is
computed in-sample against the query-population training rows; `checkpoint_best.pt` is
then the lowest in-sample selection score. This is useful for checkpoint selection while
training on 100% of the available data, but it is not an out-of-sample generalization
estimate.
`conditional_wgan_checkpoint.pt` is the compatibility/default checkpoint and points to
the restored best model when one exists, otherwise the final epoch.

Default checkpoint selection uses a composite of normalized MAF MAE, heterozygosity
error, adjacent-LD r^2 MAE, PC-space RBF MMD^2, and nearest-neighbor adversarial-
accuracy deviation from 0.5. The raw and normalized components are written to
`outputs/validation_selection_history.csv` and plotted in
`plots/validation_selection_metric.png`. Use `--checkpoint_eval_every 1` to score every
epoch; the default is 50 to limit evaluation cost.

Random query-dataset subsetting now supports both a single size and a nested sample-size sweep.
For example:

```bash
--datasets BOG COR \
--subset_samples "{'BOG': [10,20,30,'all'], 'COR': [10,20]}"
```

This expands to six independently trained models in the same timestamped run: `BOG_n10`,
`BOG_n20`, `BOG_n30`, `BOG_all`, `COR_n10`, and `COR_n20`. A scalar such as
`--subset_samples "{'BOG': 20}"` remains supported and produces one subset-size job.

Subsetting is without replacement and reproducible from `--seed`. For each dataset the code
constructs one deterministic random permutation and defines an `n`-sample subset as the first
`n` entries of that permutation (sorted back to original file order). Therefore, for requested
sizes `10 < 20 < 30`, the selected samples satisfy `S_10 subset S_20 subset S_30`. `all` uses
the complete query dataset. Exact zero-based row indices are written to each model folder's
`outputs/dataset_subset_indices.json`. The reference panel is not subset by this option. If
external ancestry conditions are used, the same selected row indices are applied to the
condition file.

For a combined dataset specification, sweeps are expanded as a Cartesian product. For example:

```bash
--datasets '{BOG,BAQ}' \
--subset_samples "{'BOG': [10,20], 'BAQ': [30,40]}"
```

creates four independently trained combined models corresponding to `(BOG=10, BAQ=30)`,
`(BOG=10, BAQ=40)`, `(BOG=20, BAQ=30)`, and `(BOG=20, BAQ=40)`. The existing
`--bog_baq_harmonized` requirement still applies.

`--n_synthetic` is independent of the training-set size. The default is 200 synthetic
samples per independently trained model.

## First / best / last saved-model outputs

Use one or more checkpoint labels:

```bash
--save_models first best last
```

- `first` = weights immediately after epoch 1.
- `best` = lowest checkpoint-selection score among evaluated epochs: held-out validation when `--val_fraction > 0`, or in-sample query-training score when `--val_fraction 0`.
- `last` = weights after the final training epoch.

Only requested semantic checkpoint files are saved (`checkpoint_first.pt`,
`checkpoint_best.pt`, `checkpoint_last.pt`). The compatibility file
`conditional_wgan_checkpoint.pt` duplicates a requested state (priority: best, last,
first) and therefore never introduces an unrequested model state.

The loss plot and validation-selection plot contain vertical lines at all requested
saved epochs. If labels overlap, for example `best == last`, the log prints a
`[CHECKPOINT OVERLAP]` message. Identical weights share one generated-data folder under
`saved_model_outputs/epoch_<epoch>_<labels>/` to avoid duplicating large genotype CSVs.
Each unique saved weight state receives its own synthetic genotype CSV, conditioning
CSV, PCA overlay, quality plots, quality metrics, and copies of the shared loss and
validation-selection diagnostic plots.

With `--val_fraction 0`, checkpoint selection is not disabled: the real comparison
cohort is the query-population training data itself, and the log/history explicitly labels
the score as `training_in_sample`. Only if no usable comparison cohort exists does
`best` fall back to `last` with a `[CHECKPOINT NOTICE]`.
