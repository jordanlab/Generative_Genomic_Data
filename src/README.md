# Conditional WGAN improvement bundle

This bundle keeps `conditional_wgan_pipeline_extended.py` backward compatible by default and adds opt-in experimental backends/features.

## Yelmen et al. (2023) unconditional model benchmarks — NEW

The **separate** entry point `src/yelmen2023_pipeline.py` trains the original
multi-resolution convolutional WGAN-GP network, an unphased **three-category
visible** RBM/conditional RBM, and optionally the 2023 convolutional VAE.
These models **do not take `--ref_input`, `--fam`, or ancestry conditions**;
the original `src/conditional_wgan_pipeline.py` and its training modes remain
unchanged. The source for the WGAN/VAE architectures is the
`artificial_genomes-master.zip` supplied with this project (Yelmen et al., 2023,
[DOI](https://doi.org/10.1371/journal.pcbi.1011584)). The RBM adaptation
uses a three-way softmax at each SNP, not fabricated phase.

### Training (paper-style presets, selectable sample-size sweeps)

```bash
python src/yelmen2023_pipeline.py \
  --datasets BOG COR \
  --dataset_input BOG=Ninh_Datasets/BOG/BOG_refintersect_qc_ldpruned_unphased_tensor.csv \
  --dataset_input COR=Ninh_Datasets/COR/COR_qc2_grch38_final_qc_ldpruned_common_snps_unphased_tensor.csv \
  --subset_samples "{'BOG':[10,25,100,'all'],'COR':[10,25,'all']}" \
  --models wgan crbm \
  --preset paper2023 \
  --genome_scope per_chromosome \
  --n_synthetic 200 \
  --save_models baseline best last \
  --seed 42
```

**Caution: paper presets are computationally expensive.** They use 10,001 WGAN
epochs (10 critic updates per generator update, Adam 0.0005, betas 0.5/0.9,
gradient penalty 10, pack size 3, original-source real-score jitter enabled) and 100,000 RBM epochs with Gibbs-100,
1,000 hidden nodes and update rate 0.005. Consult the code for exact defaults.
To run a quick functional check first, add `--wgan_epochs 2 --rbm_epochs 2
--wgan_channels 2 --rbm_hidden 16 --rbm_gibbs_steps 2 --batch_size 2
--pack_size 1 --wgan_n_critic 1 --no-evaluate`. **This small run is a software
check, not a scientific benchmark.** The paper's published training stopping
used PCA inspection, whereas the new `best` checkpoint automatically selects
an explicitly labeled *in-sample* allele-frequency + heterozygosity score.
Use `--holdout_fraction 0.2` to reserve genuinely untrained individuals for
privacy comparisons; current checkpoint selection still uses the training rows.

`--wgan_finetune_epochs 200` enables the optional low-generator-learning-rate
second stage from the paper (new Adam optimizer state). The author source
perturbs the real critic score by a uniform noise coefficient; disable it with
`--no-wgan_real_label_noise` to obtain the unperturbed WGAN-GP objective.

`--models vae` activates the optional 10K VAE; it uses BCE on normalized
0/0.5/1 inputs and is **not** a trained three-class likelihood. This is an
explicit unphased adaptation, and the original paper found its large-SNP VAE
unsatisfactory. `--preset repository_example` uses the supplied RBM example's
2,000 hidden units, 50 Gibbs steps, and learning rate 0.001, which differ from
the large CRBM settings in the published paper.

### Data and genome handling

All input CSVs contain individuals in rows and SNP site names in columns,
for example `chr10:706253:T:C`. The seven-cohort audit showed the hard calls
`0,1,2` and missing sentinel `-1`; no legitimate fractional dosages were found.
**Both training pipelines now use per-SNP mean imputation by default** so the
unconditional benchmarks are comparable to the existing cWGAN preprocessing.

**What is kept comparable:** For each selected query cohort, `-1` calls are
replaced by the mean of nonmissing dosages at that SNP among *selected training
rows*. If every training call is missing at a SNP, the mean-fill fallback is
`0`, matching the preexisting cWGAN code. The original cWGAN imputes query and
reference matrices separately, after shared-SNP intersection; Yelmen has no
reference panel and sorts SNPs into physical genomic order. The Yelmen WGAN
and VAE retain mean-imputed continuous dosages; the **categorical CRBM alone**
must convert values such as `0.37` to the nearest genotype (half-up rounding).
Its additional hard-call count is explicitly audited, so this cannot be
mistaken for identical input representations.

Yelmen `--imputation cwgan_mean` is the **new default**; `--imputation modal`
retains the previous categorical imputer as an optional experiment (and requires
integer calls). `--fractional_genotypes preserve` is the default: finite
fractional dosages in `[0,2]` remain meaningful numeric dosages rather than
being silently labeled missing. Other policies are `missing`, `round`, and
`error`. Both pipelines report `-1`, `-9`, missing/NaN, fractional values,
observed genotype frequencies and per-SNP missingness. cWGAN's legacy imputer
uses `x < 0` and **does not fill NaNs**; it reports this limitation instead
of changing the training results. For your audited `-1/0/1/2` CSVs, the
mean-imputation conventions agree at cohort level (though the chromosome
ordering and CRBM discretization differ).

**Automatic QC report files:** Each cWGAN model writes
`<model_dir>/input_qc/input_qc_report.json`, `input_qc_summary.csv`, and
`aligned_*_before_imputation_per_snp.csv` plus
`query_imputed_snps.csv` / `reference_imputed_snps.csv` with per-SNP fill
counts and actual fill values. Each Yelmen cohort/subset writes
`<cohort_subset>/input_qc/` with the same summary, per-SNP table, and `selected_training_imputed_snps.csv`,
while CRBM writes an additional `yelmen_crbm/input_qc/` report for hard-calling.
For the Yelmen loader, source `-1` and `-9` totals remain in the
`loader_input_report` metadata, while the stage matrix audit reports them as
NaNs after sentinel decoding. The Yelmen subsets now use **the same
SHA-256 seed-per-cohort nested permutation as cWGAN**, so equal cohort and
subset sizes select identical original row indices.
The console prints `[GENOTYPE QC]` summaries at raw, selected, aligned and
post-imputation stages. No QC flag silently deletes variants or changes
existing training architecture, and the raw input CSV is never overwritten. In the existing cWGAN
retrospective evaluator, real dosages are converted to discrete hard calls by
per-SNP modal filling specifically for integer-only paper-style statistics.
The Yelmen evaluator now uses that same evaluation conversion and records
it separately. This **does not** change the mean-imputed training input.

**Audit-based input quality and safety:** The seven-dataset audit reported
`-1` as the *only* nonstandard value: COR 0.048%, BOG 0.164%, BAQ 0.065%,
SAI **72.739%**, VDC2 0.050%, VDC3 **47.624%**, UKB 0.307% missing calls.
No dataset contained finite fractional dosages. SAI contained **zero
observed genotype-0 calls**; its encoding and original VCF/sample merge
must be verified before using it for modeling. VDC3 also needs review.
Mean-imputing 47–73% missing calls would not provide a scientifically
trustworthy evaluation, even though it is technically possible. Both pipelines print warnings above 10% missingness and when genotype 0 is
absent. A stricter *optional* guard fails the input check if a specified
fraction is exceeded:

```bash
python src/yelmen2023_pipeline.py --datasets SAI \
  --dataset_input SAI=Ninh_Datasets/SAI/SAI_refintersect_qc_ldpruned_unphased_tensor.csv \
  --check_inputs --max_missing_fraction 0.10
```

`--max_missing_fraction` is a proportion in `[0,1]` (e.g. `0.10` = 10%).
Without this option, the code only warns and retains previous behavior;
**warnings are not an endorsement of training on high-missingness data**.
The QC summaries record missingness, raw value categories, observed genotype counts,
and SNPs at least 50% or 90% missing. `--fail_on_absent_zero` is also opt-in. The cWGAN mean-imputation rule uses 0 as a fallback if an entire SNP column
is uncalled in the training subset; these cases are counted in the QC report.
Otherwise, `-1` is replaced by the corresponding observed SNP mean, not 0. Revisit the original VCF and
cohort/site harmonization before recoding any genotype.

**Same optional safeguards in either pipeline:** Use `--max_missing_fraction 0.10`
to reject inputs above 10% missingness or `--fail_on_absent_zero` to reject
an entire cohort with zero observed genotype-0 calls. Without those flags,
QC **warns but trains**, preserving existing sample-size experiments.
The cWGAN also supports `--check_inputs`, using `--dataset_input` and
`--ref_input` without requiring `--fam` or launching training:

```bash
python src/conditional_wgan_pipeline_extended.py \
  --check_inputs --datasets COR \
  --dataset_input COR=Ninh_Datasets/COR/COR_qc2_grch38_final_qc_ldpruned_common_snps_unphased_tensor.csv \
  --ref_input COR=Ninh_Datasets/reference/extractedChrAll_COR_intersect_qc_ldpruned_unphased_tensor.csv \
  --qc_output_dir runs/qc_preflight_cwgan
```

**Input validation without starting training** (recommended before launching
10,001-epoch WGAN / 100,000-epoch RBM jobs):

```bash
python src/yelmen2023_pipeline.py \
  --datasets COR \
  --dataset_input COR=Ninh_Datasets/COR/COR_qc2_grch38_final_qc_ldpruned_common_snps_unphased_tensor.csv \
  --subset_samples "{'COR':['all']}" \
  --check_inputs
```

Add the remaining `--datasets`/`--dataset_input` pairs from your normal command
to validate all cohorts. This performs genotype parsing and training-subset
imputation checks and writes reports under `--qc_output_dir` but **does not fit a model**.
The old error `found 2696 invalid non-integer genotype dosages` for
COR was specifically caused by **2696 `-1` missing calls**, not fractional
values. The patched loader separates these categories.
Use `--fractional_genotypes error` only to reject *genuinely fractional* dosages. `--subset_samples` saves original zero-based row indices; smaller
subsets are nested in larger subsets for each cohort.

**Retrospective evaluation:** the script automatically reuses the genotype
preprocessing policy recorded in the saved model's `args.json` so it compares with the same training representation. Conflicting
`--fractional_genotypes` or `--imputation` values raise errors. Models
created before this revision are interpreted with their recorded policies,
falling back to the previous `modal` imputer when that field is absent;
retrain to change the model input representation.

By default `--genome_scope per_chromosome`, each chromosome is trained
independently. WGAN SNP segments are padded to the **unchanged** original
16,383 or 65,535 input length; chromosomes longer than 65,535 SNPs are
split into non-overlapping windows. VAE windows are limited to 16,383. CRBM
uses an initial 5,000-SNP model followed by overlapping conditional windows
with up to 5,000 pinned and 5,000 inferred genotypes (`--rbm_window_snps`,
`--rbm_overlap_snps`). **Independently stitched segments do not preserve
cross-segment LD or genome-wide admixture coherence.** They should not be
mistaken for true whole-genome joint synthetic individuals. For an explicit
whole-sequence experiment, use `--genome_scope joint_genome`; this is *less*
faithful to the single-chromosome publication and similarly has WGAN window
seams. Large LD-pruned datasets can require extensive GPU memory.

### Outputs, retrospective evaluation and resampling

```text
runs/20261029_..._yelmen2023/
  args.json, run_manifest.json
  BOG_n25/
    outputs/dataset_subset_indices.json
    yelmen_wgan/                   # or yelmen_crbm, yelmen_vae
      args.json
      checkpoints/checkpoint_baseline.pt
      checkpoints/checkpoint_best.pt
      checkpoints/checkpoint_last.pt
      outputs/training_history.csv
      saved_model_outputs/epoch_00000_baseline/
      saved_model_outputs/epoch_XXXXX_best/
        outputs/synthetic_genotypes.csv
        sampling_diagnostics/
        paper_evaluation/initial/{metrics,plots}/
```

**Yelmen checkpoint convention (now matching the existing cWGAN):**
`--save_models baseline best last` is the default. `baseline` saves each
WGAN generator **and critic**, each CRBM/RBM block, or VAE at **epoch 0,
before any training update**. `first` (epoch 1) remains available only
when explicitly requested; it is not a synonym for `baseline`. `best`
continues to mean the lowest **in-sample** allele-frequency MAE plus
mean heterozygosity error at checked epochs (not the cWGAN composite
metric), and `last` is the final epoch. Baseline synthetic outputs are
under `saved_model_outputs/epoch_00000_baseline/`. With multiple
chromosomes/windows, each bundle saves the untrained epoch-0 state of
**every** segment, and `best` may combine different trained epochs.
In retrospective mode, `--weights` also defaults to `baseline best last`;
missing labels from older runs are skipped. To evaluate only particular
checkpoints, supply `--weights best last` explicitly.

The canonical `synthetic_genotypes.csv` contains the original genomic SNP
headers and integer dosages 0/1/2. There is **no** synthetic ancestry-conditions
CSV because these models are unconditional. Training evaluates chosen weights
by default; `--no-evaluate` disables costly plots. Each checkpoint file is a
consolidated bundle of all its chromosome/window states.

Re-evaluate the **existing CSVs without generating anything new**:

```bash
python src/yelmen2023_pipeline.py \
  --evaluate_run runs/20261029_..._yelmen2023 \
  --real_input BOG=Ninh_Datasets/BOG/BOG_refintersect_qc_ldpruned_unphased_tensor.csv \
  --weights best last
```

This `BOG=...` selects **only** `BOG_*` model folders, and the recorded sample
indices select their actual training individuals. Without any `--real_input`,
each cohort uses the real CSV path stored in its own `args.json` (paths must
remain accessible). **The default never loads a generator checkpoint** to
sample: it reads `outputs/synthetic_genotypes.csv`.

Regenerate from stored checkpoints **without overwriting** the canonical CSV:

```bash
python src/yelmen2023_pipeline.py \
  --evaluate_run runs/20261029_..._yelmen2023 \
  --real_input BOG=Ninh_Datasets/BOG/BOG_refintersect_qc_ldpruned_unphased_tensor.csv \
  --resample --weights best \
  --sampling_methods nearest binomial categorical ternary ternary_normal \
  --n_synthetic 200 --seed 43
```

WGAN/VAE resampling supports all five post-hoc genotype samplers. CRBM draws
three-category Gibbs states directly and therefore **ignores WGAN sampler
choices** (avoid requesting multiple synonymous CRBM methods). If you omit `--n_synthetic` when resampling, the script uses the count
recorded in that model's original `args.json`. All resamples
are saved under `synthetic_genotypes_METHOD_resample_seed_SEED.csv`.
The evaluation reuses `src/genotype_paper_evaluation.py`: MAF, heterozygosity,
HWE residuals, population structure, pooled and per-chromosome dosage LD,
three-point genotype moments, genotype motifs, nearest-neighbor diagnostics,
and radar summaries. Phase-specific haplotype results are not identifiable.

### Software validation

```bash
python examples/test_yelmen2023_pipeline.py
python src/yelmen2023_pipeline.py --help
```

See `MODEL_IMPLEMENTATION_MAP.md` and the revised
`model_improvements_overleaf.tex` for the file placement, mathematics,
departures from the paper, and experimental limitations.
`VALIDATION_YELMEN2023.md` records passed small-data tests and unverified scenarios.

## cWGAN sampling-method selection and dosage/PCA diagnostics

These options affect **`src/conditional_wgan_pipeline_extended.py`** (not
`src/yelmen2023_pipeline.py`). By default cWGAN still uses binomial
post-processing, continuous WGAN gradient updates, and the existing
checkpoint-fidelity composite. **`best` is now scored with the first method
listed in `--sampling_methods`**; override this with
`--checkpoint_sampling_method METHOD`. Thus, specifying
`--sampling_methods nearest categorical` selects `best` with `nearest`
unless the explicit checkpoint override is set. Every scored epoch uses
fixed latent/random seeds to reduce Monte Carlo selection noise. The
checkpoint-specific selection method and temperature are recorded in
`outputs/validation_selection_history.csv` and
`metrics/validation_selection_summary.json`. Choosing the selection sampler
can alter which epoch is saved as `best` but **never changes generator/critic
training updates**.

Example with a method-specific checkpoint decision:

```bash
python src/conditional_wgan_pipeline_extended.py \
  --datasets COR --dataset_input COR=data/COR.csv \
  --ref_input COR=data/ref_COR.csv --fam data/reference.fam \
  --sampling_methods nearest binomial categorical ternary ternary_normal \
  --checkpoint_sampling_method categorical \
  --sampling_temperature 0.35 --save_models baseline best last
```

For every checkpoint under
`MODEL/saved_model_outputs/epoch_XXXXX_LABELS/`, new diagnostics include:

```
sampling_diagnostics/
  preclip_dosage_summary.json
  preclip_per_snp.csv
  preclip_dosage_distribution.png
  dosage_before_vs_after_clipping.png
plots/
  pca_overlay.png                       # canonical first sampled genotype CSV
  pca_overlay_continuous.png            # BEFORE discretization
  pca_overlay_METHOD.png                # sampled 0/1/2 genotypes
  pca_overlay_METHOD_reference.png      # sample + ancestry reference triangle
  pca_score_distributions_METHOD.png    # real/continuous/sampled PCs 1-6
  pca_sampling_shift_summary.csv
outputs/
  synthetic_genotypes.csv               # first requested sampler
  synthetic_genotypes_METHOD.csv        # each sampler (including checkpoint sampler)
```

The preclip summary counts all values below 0 and above 2 and provides
the exact fraction changed by `clip(d,0,2)` (equivalent to clipping `d/2`
to [0,1] then multiplying by 2). The per-SNP CSV identifies problematic
loci and includes the unaltered raw range. Real, continuous synthetic, and
sampled synthetic PC plots use one reference-fitted PCA basis. **For the
`genotype_conv` backend, `pca_overlay.png` now shows the sampled genotypes
instead of silently plotting only continuous outputs**; the latter are
preserved in `pca_overlay_continuous.png`.

### Replot / resample a saved cWGAN checkpoint without retraining

```bash
python src/conditional_wgan_pipeline_extended.py \
  --evaluate_run runs/RUN_NAME --resample \
  --weights best last \
  --sampling_methods binomial nearest categorical ternary ternary_normal \
  --sampling_temperature 0.35 --seed 42 \
  --experiment_name sampler_comparison_v1 --plots_only
```

Resampling requires a trusted `.pt` checkpoint, its saved `pca_model.npz`,
the original query genotype CSV (from `args.json`, or `--real_input`),
and a saved synthetic conditioning CSV. The results live entirely under
`MODEL/saved_model_outputs/epoch_XXXXX_LABELS/sampling_experiments/sampler_comparison_v1/`
with `outputs/`, `plots/`, `sampling_diagnostics/`, and
`experiment_manifest.json`. Original CSVs, figures, and model weights
are not overwritten. An existing experiment folder is rejected; choose a
new name. `--plots_only` skips costly paper-style metric evaluation; omit
it to also write `paper_evaluation/METHOD/`. If `--n_synthetic` is omitted,
the saved number of synthetic ancestry conditions is reused. A new sampler
**cannot retroactively change which model was selected as historical
`best`**: it tests the saved weights. To *select* a new best checkpoint
with a different sampler, retraining (or archived per-epoch weights) is
necessary. Default `--evaluate_run` without `--resample` still reuses
existing saved genotype/conditioning CSVs and does not regenerate anything.

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
