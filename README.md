# Generative_Genomic_Data

Synthetic genotype generation for admixed Latin American cohorts using a
**conditional Wasserstein GAN with gradient penalty (cWGAN-GP)** trained in
principal-component space and conditioned on continental ancestry proportions
(AFR / EUR / AMR).

The goal is to generate genotypes for a target population given its known
admixture profile, so that synthetic individuals reproduce both the genome-wide
correlation structure and the ancestry composition of the real cohort.

## Method

The generator never sees raw genotypes. Everything happens in a compressed
PC space fit on a reference panel:

1. **Load** target-cohort and reference genotype tensors (additive dosage, 0/1/2).
2. **Intersect** to the SNPs shared by both; impute residual missingness to per-SNP means.
3. **Fit PCA on the reference panel** (default 50 PCs) and project both datasets in.
4. **Assign admixture conditions.** Reference samples get their known one-hot label
   (`[1,0,0]` / `[0,1,0]` / `[0,0,1]`); target samples get a barycentric projection
   onto the triangle formed by the AFR/EUR/AMR reference centroids.
5. **Train the cWGAN-GP** jointly on reference + target samples, with both generator
   and critic conditioned on the 3-dim admixture vector.
6. **Sample** ancestry proportions from a Dirichlet fit to the observed target
   admixture (or pass `--target_admixture` explicitly).
7. **Invert the PCA**, discretize back to 0/1/2, and write genotypes plus a PCA overlay plot.

The model drives generation end to end — there is no post-hoc Dirichlet
resampling of the GAN output.

## Repository layout

```
src/
  conditional_wgan_pipeline.py   # the cWGAN-GP model — the only model script here
  plot_colors.py                 # per-population colors; imported by the pipeline
  vcf_to_unphased_tensor.py      # VCF -> additive dosage tensor (builds pipeline input)
docs/
  pipeline_walkthrough.md        # line-by-line walkthrough of the cWGAN pipeline
results/<COHORT>/
  cWGAN_PCA/                     # PCA overlay plots (committed)
  cWGAN_gen-data/                # synthetic genotype CSVs (gitignored — large)
```

`src/` is deliberately scoped to the cWGAN pipeline. Earlier work — the
unconditional PCA+WGAN pipeline, the VAE experiments, and the standalone PCA and
evaluation scripts — is kept out of the repo via `.gitignore` rather than deleted,
so it still exists in the original working tree.

`data/`, `snps/`, and `slurm/` are gitignored: they hold raw VCFs, PLINK
binaries, genotype tensors, SNP lists, and job logs. Synthetic genotype CSVs are
also gitignored — they are large and fully reproducible from the pipeline.

## Running

```bash
python src/conditional_wgan_pipeline.py \
    --baq_input      data/<COHORT>/<COHORT>_refintersect_qc_ldpruned_unphased_tensor.csv \
    --ref_input      data/reference/extractedChrAll_<COHORT>_intersect_qc_ldpruned_unphased_tensor.csv \
    --fam            data/reference/extractedChrAll_common_snps_forReference.fam \
    --output         results/<COHORT>/cWGAN_gen-data/synthetic_genotypes_seed42.csv \
    --plot           results/<COHORT>/cWGAN_PCA/pca_overlay_seed42.png \
    --sample_name    <COHORT> \
    --n_components   50 \
    --n_synthetic    200 \
    --wgan_epochs    1000 \
    --wgan_latent_dim 32 \
    --batch_size     64 \
    --seed           42
```

To generate for an explicit ancestry profile instead of the cohort-fitted
Dirichlet, add `--target_admixture 0.6,0.2,0.2` (AFR, EUR, AMR).

Other useful flags: `--alpha_concentration` (Dirichlet tightness),
`--variance_scale`, `--real_color`.

### Environment

`environment.yml` defines the full conda environment (`GGD_env`). Runs on PACE
require a GPU node; the pipeline falls back to CPU but is substantially slower.

```bash
conda env create -f environment.yml && conda activate GGD_env
```

## Current results

Latest sweep: **4 cohorts x 5 seeds (42-46), 20 runs, all completed.**
Reference panel: 644 samples (AFR = 312, EUR = 273, AMR = 59). 50 PCs, 1000 epochs,
200 synthetic samples per run. PCA overlays are in `results/<COHORT>/cWGAN_PCA/`.

| Cohort | Real N | SNPs (post-intersect) | Missingness imputed | Var. explained (50 PC) | Fixed SNPs | Assessment |
|--------|--------|----------------------|---------------------|------------------------|------------|------------|
| BOG    | 337    | 28,514               | 0.16%               | 20.1%                  | 0–1        | Usable |
| UKB    | 245    | 77,387               | 0.31%               | 19.1%                  | 46–58      | Usable |
| VDC3   | 24     | 6,688                | 47.6%               | 23.9%                  | 1–2        | Over-dispersed |
| SAI    | 10     | 50,795               | 72.7%               | 24.7%                  | 107–150    | Not usable |

### Known limitations

- **Small-N cohorts fail.** SAI (10 samples, 72.7% imputed) produces a degenerate
  fit: the Dirichlet collapses to `alpha = [261.25, 0.00, 47.95]`, giving an exactly
  zero EUR component and effectively no variance in the sampled conditions. The
  synthetic cloud spans far beyond the real data and partly outside the ancestry
  triangle. VDC3 (24 samples, 47.6% imputed) is over-dispersed for the same reason.
  Treat both as diagnostic, not as usable synthetic data.
- **Mode-dropping on the admixed tail.** For BOG and UKB the synthetic distribution
  covers the core of the real cloud but under-represents individuals extending
  toward the AFR vertex.
- **Low retained variance.** 50 PCs capture only 19–25% of total variance, so
  reconstruction discards most of the fine-scale signal by construction.
- **Stale log labels.** The pipeline prints `BAQ` for every cohort (e.g.
  "Loading BAQ genotypes...") regardless of `--sample_name`. Cosmetic only — the
  data loaded is correct — but it makes logs confusing to read.
