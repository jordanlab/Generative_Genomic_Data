# Installation guide — conditional WGAN and paper-style genotype evaluation

This updates the supplied installation guide without changing the underlying
PyTorch/scikit-learn/Pandas model dependencies. The paper evaluator additionally
requires SciPy. Commands assume the project root contains `src/` and
`requirements.txt` from the updated release.

## 1. Supported versions and dependencies

- Python 3.10 or newer (verified here with Python 3.13).
- NumPy, pandas, SciPy, scikit-learn, matplotlib, PyTorch.
- Optional NVIDIA GPU: choose a PyTorch wheel matching your driver/CUDA stack.
- A GPU is not necessary for evaluation; training large datasets on CPU is slow.

New local modules (`genotype_sampling`, `genotype_paper_evaluation`,
`evaluate_saved_genotypes`, `genotype_membership_attacks`,
`run_membership_attacks`) must stay alongside the other Python files in `src/`.

## 2. Install with Conda or pip

Create the CPU environment supplied in this update:

```bash
conda env create -f environment.yml
conda activate cwgan-genotype
```

Or use an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

For CUDA acceleration, install a PyTorch GPU wheel using the current official
PyTorch selector: https://pytorch.org/get-started/locally/ . Do not mix
incompatible CPU and GPU torch wheels in the same environment.

## 3. Validate the environment

```bash
python - <<'PY'
import numpy, pandas, scipy, sklearn, matplotlib, torch
print('numpy',numpy.__version__)
print('pandas',pandas.__version__)
print('scipy',scipy.__version__)
print('sklearn',sklearn.__version__)
print('matplotlib',matplotlib.__version__)
print('torch',torch.__version__)
print('CUDA usable',torch.cuda.is_available())
PY
python -m compileall -q src
python src/conditional_wgan_pipeline.py --help
python src/evaluate_saved_genotypes.py --help
python examples/test_retrospective_routing.py
python src/run_membership_attacks.py --help
```

For SLURM/HPC headless execution, set `export MPLBACKEND=Agg`. Request GPU
resources through the scheduler; the Python script itself cannot reserve them.

## 4. Inputs required for training

- CSV of samples (rows) by SNP sites (columns), dosage values `0,1,2`.
- Reference CSV with exactly the same allele orientation and overlapping SNPs.
- A row-aligned 6-column `.fam` with recognized population in *individual ID*,
  e.g. `FAM forReferenceYRI_0001 0 0 1 -9` for the African anchor. FAM family
  ID alone is not used to assign ancestry. AFR/EUR/AMR reference groups must
  each have at least one recognized sample for PCA-barycentric conditioning.
- Enough reference samples for your requested PCA dimensions.

Missing `-1` genotypes are mean-imputed at each SNP using selected training
individuals, matching the cWGAN baseline. Only the categorical RBM needs an
additional logged conversion of fractional dosages to hard 0/1/2 categories.
Check all input quality, genomic build and strand/alternate-allele coding
upstream.

## 5. Minimal training example

```bash
python src/conditional_wgan_pipeline.py  --datasets BAQ  --dataset_input BAQ=data/BAQ/genotypes.csv  --ref_input BAQ=data/reference/BAQ_reference.csv  --fam data/reference/labels.fam  --wgan_epochs 1000 --n_synthetic 500  --save_models baseline best last  --sampling_methods binomial categorical ternary --seed 42
```

Unless `--no_evaluate` is given, all paper-inspired analyses are run automatically
for each saved checkpoint and sampling method. The existing `baseline_pca`,
`pca_film`, and `genotype_conv` backends remain available.

## 6. Evaluate existing run outputs (no training or sampling by default)

```bash
# Only BOG_* model folders are selected; reuse saved genotype and ancestry CSVs:
python src/conditional_wgan_pipeline.py \
  --evaluate_run runs/20261004_105931_sample_size_sweep \
  --real_input BOG=Ninh_Datasets/BOG/BOG_refintersect_qc_ldpruned_unphased_tensor.csv
```

Repeat `--real_input NAME=PATH` for each desired cohort. Supplying at least
one mapping restricts evaluation to those cohorts' model folders; supplying
none evaluates all model folders using their own `args.json` input paths.
The input CSV must be the same-row-order **full** cohort used by the original
training run so saved subset indices select the right individuals. The
retrospective evaluator requires an existing
`synthetic_genotypes.csv` **and** `synthetic_conditioning_admixture.csv` for
the default workflow; it does not load checkpoints or regenerate samples.
Missing saved pairs are skipped and reported in the run manifest.

### Yelmen unconditional benchmarks: checkpoint convention

`src/yelmen2023_pipeline.py` now defaults to `--save_models baseline best last`
and writes `checkpoint_baseline.pt` and
`saved_model_outputs/epoch_00000_baseline/outputs/synthetic_genotypes.csv`
for each dataset/model. Unlike `first` (optional epoch 1), `baseline` is the
**epoch-zero, untrained** generator/critic, categorical RBM/CRBM, or VAE.
For retrospective evaluation or resampling, `--weights` defaults to
`baseline best last`; specify a subset if only those weights are needed.
The Yelmen `best` selection still uses the in-sample MAF+heterozygosity
criterion, not the cWGAN composite score. No extra packages are needed.

## 7. Resample saved model weights (explicit opt-in)

```bash
python src/conditional_wgan_pipeline.py \
  --evaluate_run runs/20261004_105931_sample_size_sweep \
  --real_input BOG=Ninh_Datasets/BOG/BOG_refintersect_qc_ldpruned_unphased_tensor.csv \
  --resample \
  --weights best last \
  --sampling_methods binomial nearest categorical ternary ternary_normal \
  --sampling_temperature 0.35 --seed 42 \
  --experiment_name test_sampling_v1 --plots_only
```

`--resample` loads saved `.pt` model states and requires
`checkpoints/pca_model.npz` and historical ancestry-condition CSVs.
For the cWGAN, `--resample` writes `outputs/synthetic_genotypes_METHOD.csv`
and `outputs/synthetic_conditioning_admixture.csv` into a **new**
`MODEL/saved_model_outputs/epoch_XXXXX_LABELS/sampling_experiments/NAME/`
directory, together with preclipping summaries and method-specific sampled
PCA overlays/distribution plots. Original checkpoint outputs remain unchanged.
Use `--experiment_name NAME` to choose a clear name, or let the script
construct one from the sampler/seed settings; existing experiment names are
rejected to prevent overwrites. Add `--plots_only` for fast plotting without
paper-style evaluation. `--sampling_methods` and `--weights` are meaningful for
retrospective generation only when `--resample` is passed. Set
`--n_synthetic` to override the saved number of conditions, or omit it to
reuse that count. Load only checkpoints from trusted training runs; old PCA
checkpoints may need PC1 sign compatibility inspection (see README).

For a resampling experiment, inspect `sampling_experiment_run_manifests/NAME.json`
and each experiment's `experiment_manifest.json`; for plain reuse of saved
outputs, inspect `paper_evaluation_run_manifest.json`. Both record selected
cohorts, run mode and excluded folders. This output is particularly useful
when several population and subset models share one timestamped run.

### cWGAN sampling-aware training and checkpoint diagnostics

When training with `src/conditional_wgan_pipeline_extended.py`, the first
`--sampling_methods` entry now controls checkpoint scoring by default.
`--checkpoint_sampling_method METHOD` overrides it. This affects *only*
checkpoint evaluation/selection, not gradient training. With the default
`--sampling_methods binomial`, historical checkpoint selection is unchanged.

```bash
python src/conditional_wgan_pipeline_extended.py \
  --datasets COR --dataset_input COR=data/COR.csv \
  --ref_input COR=data/ref_COR.csv --fam data/reference.fam \
  --sampling_methods nearest binomial categorical \
  --checkpoint_sampling_method nearest \
  --save_models baseline best last --seed 42
```

The `saved_model_outputs/epoch_XXXXX_LABELS/sampling_diagnostics/` directory
contains unclipped dosage histograms, counts of values outside [0,2], the
before/after clipping overlay, and `preclip_per_snp.csv`.
`plots/` contains PCA overlays for the continuous output and each discrete
method, PC1--PC6 distributions, and per-PC sampling-induced shifts. For
`genotype_conv`, the canonical `pca_overlay.png` now corresponds to sampled
genotypes (the previous continuous version is separately retained). Run
`python examples/test_cwgan_sampling_experiments.py` to test the diagnostics.

## 8. Run opt-in privacy ranking

```bash
python src/run_membership_attacks.py --help
```

Valid membership inference needs confirmed member and never-used nonmember
CSV files as well as per-subject ancestry-condition CSV files. The white-box
ranking uses the saved conditional critic. The optional black-box surrogate
re-trains a separate PCA-space critic; it is not the exact architecture of
all original checkpoint backends.

## 9. Common issues

- `Reference .fam has no AFR/EUR/AMR subjects`: fix IID coding and ordering.
- `No shared SNPs`: make the columns and allele orientation consistent.
- CUDA unavailable: check `nvidia-smi`, installed wheel, and scheduler allocation.
- GPU memory error: lower `--batch_size` or use `baseline_pca` instead of direct
  `genotype_conv` on whole chromosomes.
- Some comparisons give NaN correlations: constant/monomorphic SNPs and
  insufficient samples cannot have a Pearson coefficient; this is not a zero.
- Earlier output roots may use relocated paths: supply `--real_input NAME=PATH`.
- Full-scale 8-site motif tables and cross-cohort distances are expensive;
  use default bounded `--paper_max_motif_windows`,
  `--paper_max_distance_samples`, and `--metric_max_ld_pairs`.

## 10. Reproducibility

```bash
python -m pip freeze > pip_freeze.txt
conda env export > environment_exact.yml
```

Log your source commit/checksum, seeds, dataset provenance, genomic build,
LD-pruning status, actual training/test membership, and whether the PCA model
was saved before or after its PC1 orientation correction.

## Yelmen et al. (2023) benchmarking extension

No new third-party dependencies are required beyond those in the existing
`requirements.txt` (PyTorch, NumPy, pandas, SciPy, matplotlib, scikit-learn).
The new files must be kept under `src/yelmen2023/` and the entry point at
`src/yelmen2023_pipeline.py`. No modification to the existing `conditional_wgan_pipeline.py`
is needed. The original WGAN/VAE modules are vendored in `src/yelmen2023/vendor_*.py`.

```bash
python -m py_compile src/yelmen2023_pipeline.py src/yelmen2023/*.py
python src/yelmen2023_pipeline.py --help
OMP_NUM_THREADS=2 python examples/test_yelmen2023_pipeline.py
```

WGAN uses padded 16,383 or 65,535 SNP positions per independent segment and
can consume substantial RAM/VRAM even if its original channel multiplier is
small. Use a GPU for scientific-scale training, and first test with reduced
`--wgan_channels`, `--batch_size`, `--pack_size`, `--wgan_epochs`, and
`--wgan_n_critic`. The RBM uses O(number-of-SNPs × 3 × hidden-units) floating
point trainable weights per segment; the original large-study 5,000/10,000 SNP
windows and 1,000 hidden units are resource intensive. The short smoke test is
not a calibration of scientific convergence or genotype fidelity.

Pre-existing cWGAN training and retrospective scripts remain the same; the
Yelmen benchmark has a separate checkpoint format and must be loaded with its
own `--evaluate_run`, not `evaluate_saved_genotypes.py`.


## Shared input QC, compatibility, and optional safeguards

No additional dependencies are necessary. Both training entry points import
`src/genotype_input_qc.py`, which must remain next to their CLI scripts.
The default Yelmen imputation is now `--imputation cwgan_mean`, and the
fractional policy is `--fractional_genotypes preserve`. For `-1/0/1/2`
cohort matrices, the WGAN/VAE input processing follows the existing cWGAN:
negative missing calls are filled with selected-training per-SNP means,
including a zero fallback for completely uncalled loci. The RBM/CRBM needs
additional nearest-category hard calls and reports the number changed.
The old Yelmen method is available as `--imputation modal` with a suitable
hard-call fractional policy.

Run preflight audits without fitting models:

```bash
python src/yelmen2023_pipeline.py --check_inputs \
  --datasets COR \
  --dataset_input COR=Ninh_Datasets/COR/COR_qc2_grch38_final_qc_ldpruned_common_snps_unphased_tensor.csv \
  --qc_output_dir runs/qc_preflight_yelmen

python src/conditional_wgan_pipeline_extended.py --check_inputs \
  --datasets COR \
  --dataset_input COR=Ninh_Datasets/COR/COR_qc2_grch38_final_qc_ldpruned_common_snps_unphased_tensor.csv \
  --ref_input COR=Ninh_Datasets/reference/extractedChrAll_COR_intersect_qc_ldpruned_unphased_tensor.csv \
  --qc_output_dir runs/qc_preflight_cwgan
```

During training, the logs print `[GENOTYPE QC]` statistics, and files are
written to `input_qc/` beside model output folders. Each report includes
`input_qc_report.json`, `input_qc_summary.csv`, per-SNP missingness and
0/1/2 counts, and per-SNP imputed counts and fill values. The Yelmen CRBM also saves its model-specific hard-call audit.
These files document stages and do not edit input CSVs. The two
paper-style evaluators use a common hard-call modal conversion for
integer-only metrics; the training matrices remain mean-imputed.

Both entry points accept **opt-in** safeguards:

```text
--max_missing_fraction 0.10
--fail_on_absent_zero
```

The first rejects source datasets above the chosen missing-call fraction;
the second rejects datasets without a single genotype-0 observation. By
default the programs issue warnings but do not reject SAI or VDC3. Very high
missingness still makes synthetic-data fidelity comparisons unreliable.
The existing cWGAN `x<0` imputer does not process NaN automatically, so any
NaNs revealed by QC should be corrected upstream; adding NaN handling to that
legacy pipeline would be a separate experimental change.

For regression checks:

```bash
python examples/test_shared_genotype_qc.py
python examples/test_yelmen2023_pipeline.py
```
