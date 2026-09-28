# cWGAN run logs and submission scripts

This directory holds the Slurm pieces needed to reproduce the conditional
WGAN-GP results in `results/<COHORT>/cWGAN_PCA/`.

Most of `slurm/` is gitignored (job logs are noisy and large). The files here
are deliberate exceptions: the cWGAN submission scripts, and the `.out` log for
each of the eight reference runs.

## The eight reference runs

Each log is the full stdout of one run: input shapes, SNP intersection, PCA
variance retained, per-epoch losses, admixture fit, and output paths.

| Cohort | Job ID | Log | Seed |
|--------|--------|-----|------|
| ATQ3 | 10076013 | `out/cwgan/cwgan_ATQ3_10076013.out` | 42 |
| COR  | 9870696  | `out/cwgan/cwgan_cwgan_9870696.out` | 46 |
| VDC2 | 9870682  | `out/cwgan/cwgan_cwgan_9870682.out` | 46 |
| BOG  | 13421019 | `out/cwgan/cwgan_BOG_13421019.out`  | 42 |
| SAI  | 13421402 | `out/cwgan/cwgan_SAI_13421402.out`  | 45 |
| UKB  | 13421020 | `out/cwgan/cwgan_UKB_13421020.out`  | 42 |
| VDC3 | 13421018 | `out/cwgan/cwgan_VDC3_13421018.out` | 42 |
| BAQ  | 9398110  | **not retained — see below**        | 2024 |

The COR and VDC2 logs are named `cwgan_cwgan_*` because those jobs were
submitted with a generic `--job-name`. The cohort is confirmed inside each file
on the `Conditional WGAN-GP — <COHORT> Dataset` line.

### The missing BAQ log

`BAQ_9398110` ran, but its log is gone from disk and the script that produced it
was never committed. Slurm accounting still has the record:

```
$ sacct -j 9398099 --format=JobID,JobIDRaw,JobName,State,Start,Elapsed,Partition

JobID      JobIDRaw  JobName          State      Start                Elapsed   Partition
9398099_0  9398101   cwgan_vs0_sweep  COMPLETED  2026-06-02T16:38:56  00:20:20  gpu-v100
9398099_1  9398102   cwgan_vs0_sweep  COMPLETED  2026-06-02T16:39:29  00:21:16  gpu-v100
9398099_2  9398108   cwgan_vs0_sweep  COMPLETED  2026-06-02T16:40:05  00:20:10  gpu-v100
9398099_3  9398110   cwgan_vs0_sweep  COMPLETED  2026-06-02T16:40:24  00:19:48  gpu-v100
9398099_4  9398099   cwgan_vs0_sweep  COMPLETED  2026-06-02T16:41:07  00:19:59  gpu-v100
```

It was task 3 of a five-task array (one task per seed), submitted on
2026-06-02 as:

```
sbatch slurm/scripts/cwgan/submit_cwgan_vs0_seed_sweep.sbatch
```

Resources: gpu-v100, 8 CPU, 64 GB, 1 GPU. The five plots in
`results/BAQ/cWGAN_vs0_PCA_Final/` are that array's output — job IDs 9398101,
9398102, 9398108, 9398110 and 9398099 map to seeds 42, 7, 123, 2024 and 999.

`submit_cwgan_vs0_seed_sweep.sbatch` no longer exists anywhere on disk. To
regenerate a comparable BAQ run, use `scripts/cwgan/submit_cwgan.sbatch` with
`--sample_name BAQ`; note the original used a `variance_scale` of 0 (the "vs0"
in the name), which is not the current default.

## Scripts

```
scripts/
  run_dataset_sweep.sh                  # main entry point — full sweep per cohort
  cwgan/
    submit_cwgan.sbatch                 # single cWGAN run; called by the sweep
    submit_conditional_wgan.sbatch      # earlier standalone variant
    submit_cwgan_ATQ3.sbatch            # per-cohort variants (ATQ3/COR/VDC2 runs)
    submit_cwgan_COR.sbatch
    submit_cwgan_VDC2.sbatch
    run_cwgan.sbatch                    # self-contained home-filesystem variant
  ref_preprocess/
    rebuild_reference.sbatch            # dependency of run_dataset_sweep.sh
```

Two families exist because of a mid-project filesystem outage:

- **`scripts/cwgan/submit_cwgan*.sbatch`** read from and write to `/storage/scratch1`.
  These produced the ATQ3, COR and VDC2 runs.
- **`scripts/cwgan/run_cwgan.sbatch`** keeps interpreter, source, inputs and outputs
  entirely on the home filesystem. It was written when scratch was down and produced
  the BOG, SAI, UKB and VDC3 runs. It expects `~/ggd_env`, `~/ggd_src`, `~/ggd_data`
  and `~/ggd_results`, and is driven by environment variables:
  `SAMPLE_NAME=BOG SEED=42 sbatch scripts/cwgan/run_cwgan.sbatch`

`rebuild_reference.sbatch` is not itself a cWGAN script, but `run_dataset_sweep.sh`
calls it and the pipeline cannot run without its output, so it is included here.

## Running a sweep

```bash
./slurm/scripts/run_dataset_sweep.sh BOG SAI UKB VDC3
```

Per cohort this submits one `rebuild_reference.sbatch` (cpu-small, ~6 min) plus
five `submit_cwgan.sbatch` GPU jobs chained with `--dependency=afterok`, so
nothing trains on half-written input. Output is five PCA overlays in
`results/<COHORT>/cWGAN_PCA/`.

Useful variants:

```bash
DRY_RUN=1 ./slurm/scripts/run_dataset_sweep.sh BOG        # print, don't submit
SEEDS="42 7 123 999 2024" ./slurm/scripts/run_dataset_sweep.sh BOG
./slurm/scripts/run_dataset_sweep.sh VDC3=data/VDC3/VDC3_qc4.vcf.gz
```

### Before you run

- **`BASE_DIR` in `run_dataset_sweep.sh` is hardcoded** to the original author's
  scratch path. Change it to your own checkout before running anything.
- The `--account` in each `.sbatch` is `gts-jhamilton80`. Change it to your allocation.
- Input tensors under `data/` are **not** in this repo. You need
  `<COHORT>_refintersect_qc_ldpruned_unphased_tensor.csv`, the matching reference
  tensor, and `extractedChrAll_common_snps_forReference.fam`.
- The sweep auto-detects the raw VCF inside `data/<COHORT>/`, preferring
  `<COHORT>_qc2_grch38_final.vcf.gz`.
