#!/bin/bash
# ===========================================================================
# run_dataset_sweep.sh — preprocess a population and run the 5-seed cWGAN sweep
#
# For each SAMPLE_NAME given, this submits:
#   1 x rebuild_reference.sbatch   (cpu-small, ~6 min)
#   5 x submit_cwgan.sbatch        (gpu-v100, chained with --dependency=afterok)
#
# The five GPU jobs wait for the rebuild to succeed, so nothing trains on
# missing or half-written input. Each emits one PC1-vs-PC2 overlay, giving
# 5 plots per dataset in results/<NAME>/cWGAN_PCA/.
#
# The raw VCF is auto-detected inside data/<NAME>/ — the conventional
# <NAME>_qc2_grch38_final.vcf.gz is preferred, otherwise the single .vcf.gz
# present is used. Override per dataset with NAME=/explicit/path.vcf.gz.
#
# Usage:
#   ./slurm/scripts/run_dataset_sweep.sh BOG SAI UKB VDC3
#   ./slurm/scripts/run_dataset_sweep.sh VDC3=data/VDC3/VDC3_qc4.vcf.gz
#   SEEDS="42 7 123 999 2024" ./slurm/scripts/run_dataset_sweep.sh BOG
#   DRY_RUN=1 ./slurm/scripts/run_dataset_sweep.sh BOG SAI UKB VDC3
# ===========================================================================

set -uo pipefail

BASE_DIR="/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data"
REBUILD_SBATCH="${BASE_DIR}/slurm/scripts/ref_preprocess/rebuild_reference.sbatch"
CWGAN_SBATCH="${BASE_DIR}/slurm/scripts/cwgan/submit_cwgan.sbatch"

# Seeds 42-46 match the most recent COR and VDC2 sweeps, so new results line
# up with the existing ones. Override with SEEDS="..." for a different set.
SEEDS="${SEEDS:-42 43 44 45 46}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <SAMPLE_NAME[=/path/to.vcf.gz]> ..."
    exit 1
fi

cd "${BASE_DIR}" || { echo "ERROR: cannot cd to ${BASE_DIR}"; exit 1; }

SUMMARY=""
FAILED=""

for ARG in "$@"; do
    NAME="${ARG%%=*}"
    EXPLICIT_VCF=""
    [ "${ARG}" != "${NAME}" ] && EXPLICIT_VCF="${ARG#*=}"

    echo ""
    echo "==========================================================="
    echo "  ${NAME}"
    echo "==========================================================="

    # -------------------------------------------------------------------
    # Resolve the raw VCF. None of the 2026-09 datasets followed the
    # <NAME>_qc2_grch38_final.vcf.gz convention, so fall back to whatever
    # single .vcf.gz sits in the dataset directory.
    # -------------------------------------------------------------------
    if [ -n "${EXPLICIT_VCF}" ]; then
        VCF="${EXPLICIT_VCF}"
    elif [ -f "data/${NAME}/${NAME}_qc2_grch38_final.vcf.gz" ]; then
        VCF="data/${NAME}/${NAME}_qc2_grch38_final.vcf.gz"
    else
        mapfile -t FOUND < <(find "data/${NAME}" -maxdepth 1 -name '*.vcf.gz' 2>/dev/null)
        if [ "${#FOUND[@]}" -eq 1 ]; then
            VCF="${FOUND[0]}"
            echo "  note      : using non-conventional filename"
        elif [ "${#FOUND[@]}" -eq 0 ]; then
            echo "  ERROR: no .vcf.gz found in data/${NAME}/"
            FAILED="${FAILED} ${NAME}"; continue
        else
            echo "  ERROR: ${#FOUND[@]} .vcf.gz files in data/${NAME}/ — disambiguate with"
            echo "         ${NAME}=/path/to/file.vcf.gz"
            printf '           %s\n' "${FOUND[@]}"
            FAILED="${FAILED} ${NAME}"; continue
        fi
    fi

    if [ ! -f "${VCF}" ]; then
        echo "  ERROR: input VCF not found: ${VCF}"
        FAILED="${FAILED} ${NAME}"; continue
    fi
    echo "  input     : ${VCF} ($(du -h "${VCF}" 2>/dev/null | cut -f1))"

    # -------------------------------------------------------------------
    # Intake check — advisory only, and deliberately written against the
    # system python3 rather than bcftools, so it still runs when the conda
    # env is unavailable. A wrong variant-ID convention is the one failure
    # that survives the whole pipeline silently: the run exits 0 having
    # trained on the handful of SNPs that happened to match.
    # -------------------------------------------------------------------
    python3 - "${VCF}" <<'PYEOF'
import gzip, re, sys
path = sys.argv[1]
ID_RE = re.compile(r'^chr[0-9XYM]+:\d+:[ACGTN]+:[ACGTN]+$')
nsamp, first = 0, ''
try:
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                nsamp = len(line.rstrip('\n').split('\t')) - 9
            else:
                first = line.split('\t', 3)[2]
                break
except Exception as e:
    print(f"  (intake check could not read the VCF: {e})")
    sys.exit(0)
print(f"  samples   : {nsamp}")
print(f"  first ID  : {first or '<none>'}")
if nsamp and nsamp < 30:
    print(f"  *** NOTE: only {nsamp} samples — the admixture estimate and the")
    print(f"  ***       real-sample layer of each plot rest on very few points.")
if first and not ID_RE.match(first):
    print("  *** WARNING: variant IDs are not chr:pos:REF:ALT (e.g. chr1:874202:T:C).")
    print("  ***          The reference intersect is keyed on this, so the sweep")
    print("  ***          will likely train on very few SNPs. Fix with:")
    print("  ***            bcftools annotate --set-id '%CHROM:%POS:%REF:%ALT' ...")
    print("  ***          Submitting anyway, as configured.")
PYEOF

    # -------------------------------------------------------------------
    # Step 1 — preprocessing. --job-name makes the log land at
    # slurm/out/rebuild_ref_<NAME>_<jobid>.out rather than the default
    # rebuild_ref_rebuild_ref_<jobid>.out, since the sbatch uses %x.
    # -------------------------------------------------------------------
    REBUILD_CMD=(sbatch --parsable --job-name="${NAME}"
                 --export=ALL,SAMPLE_NAME="${NAME}",INPUT_VCF="${VCF}"
                 "${REBUILD_SBATCH}")

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "  DRY RUN   : ${REBUILD_CMD[*]}"
        RID="<rebuild-id>"
    else
        RID=$("${REBUILD_CMD[@]}")
        if [ -z "${RID}" ]; then
            echo "  ERROR: rebuild submission failed for ${NAME}"
            FAILED="${FAILED} ${NAME}"; continue
        fi
        echo "  rebuild   : job ${RID}"
    fi

    # -------------------------------------------------------------------
    # Step 2 — the 5-seed sweep, each chained to that rebuild.
    # -------------------------------------------------------------------
    SEED_IDS=""
    for S in ${SEEDS}; do
        CWGAN_CMD=(sbatch --parsable --job-name="${NAME}"
                   --dependency=afterok:"${RID}"
                   --export=ALL,SAMPLE_NAME="${NAME}",SEED="${S}"
                   "${CWGAN_SBATCH}")
        if [ "${DRY_RUN}" -eq 1 ]; then
            echo "  DRY RUN   : ${CWGAN_CMD[*]}"
            SEED_IDS="${SEED_IDS} seed${S}=<id>"
        else
            JID=$("${CWGAN_CMD[@]}")
            if [ -z "${JID}" ]; then
                echo "  WARNING: seed ${S} submission failed"
            else
                echo "  seed ${S}   : job ${JID}"
                SEED_IDS="${SEED_IDS} ${JID}"
            fi
        fi
    done

    SUMMARY="${SUMMARY}\n  ${NAME}: rebuild ${RID}; seeds${SEED_IDS}"
done

echo ""
echo "==========================================================="
echo "  Submitted"
echo "==========================================================="
printf '%b\n' "${SUMMARY}"
[ -n "${FAILED}" ] && { echo ""; echo "  NOT submitted:${FAILED}"; }

echo ""
echo "  Watch:    squeue -u \$USER"
echo "  Overlap:  grep -E 'final SNPs|Labels:' slurm/out/rebuild_ref_<NAME>_*.out"
echo "  Training: grep -E 'aligning to|samples' slurm/out/cwgan_<NAME>_*.out"
echo "  Plots:    results/<NAME>/cWGAN_PCA/pca_overlay_seed*.png"
echo ""

[ -n "${FAILED}" ] && exit 1
exit 0
