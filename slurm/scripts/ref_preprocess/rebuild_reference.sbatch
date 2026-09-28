#!/bin/bash
#SBATCH --job-name=rebuild_ref
#SBATCH --account=gts-jhamilton80
#SBATCH --partition=cpu-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data/slurm/out/rebuild_ref_%x_%j.out
#SBATCH --error=/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data/slurm/err/rebuild_ref_%x_%j.err

# ===========================================================================
# Standardized reference-rebuild pipeline (the "ATQ3 methodology"), applied
# to any population: intersect the RAW reference panel and the RAW query
# VCF on variant ID first, THEN QC-filter each side independently
# (Data_Preprocessing.ipynb flags), THEN LD-prune each side independently.
#
# This replaces the three different ad hoc approaches previously used per
# population (BAQ: subset directly onto reference's finalized SNP list;
# COR: pruned independently then intersected with reference as a final
# step; VDC2/ATQ3: pruned independently with no reference alignment at
# all) with one identical, repeatable procedure for every population.
#
# Usage:
#   sbatch --export=ALL,SAMPLE_NAME=COR  rebuild_reference.sbatch
#   sbatch --export=ALL,SAMPLE_NAME=VDC2 rebuild_reference.sbatch
#
# For a non-standard raw VCF name, pass INPUT_VCF explicitly:
#   sbatch --export=ALL,SAMPLE_NAME=BAQ,INPUT_VCF=/path/to/BAQ_raw.vcf.gz rebuild_reference.sbatch
# ===========================================================================

if [ -z "${SAMPLE_NAME}" ]; then
    echo "ERROR: SAMPLE_NAME is required. Pass via --export=ALL,SAMPLE_NAME=<name>"
    exit 1
fi

BASE_DIR="/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data"
cd "${BASE_DIR}"
mkdir -p data/reference "data/${SAMPLE_NAME}" snps slurm/out slurm/err

source /storage/home/hcoda1/8/sverma87/miniconda3/etc/profile.d/conda.sh
GPU_ENV_PREFIX="/storage/scratch1/8/sverma87/conda-envs/GGD_gpu_env"
if [ -d "${GPU_ENV_PREFIX}" ]; then
    conda activate "${GPU_ENV_PREFIX}"
else
    conda activate GGD_env
fi

RAW_REF_VCF="data/reference/extractedChrAll.vcf.gz"
INPUT_VCF="${INPUT_VCF:-data/${SAMPLE_NAME}/${SAMPLE_NAME}_qc2_grch38_final.vcf.gz}"
REF_SAMPLES="data/reference/extractedChrAll_common_snps_forReference.fam"   # fixed 644-sample curated panel
SAMPLE_IDS="snps/${SAMPLE_NAME}_snps.txt"

REF_INTERSECT="data/reference/extractedChrAll_${SAMPLE_NAME}_intersect"
REF_INTERSECT_QC="data/reference/extractedChrAll_${SAMPLE_NAME}_intersect_qc"
REF_FINAL="data/reference/extractedChrAll_${SAMPLE_NAME}_intersect_qc_ldpruned"

SAMPLE_INTERSECT_QC="data/${SAMPLE_NAME}/${SAMPLE_NAME}_refintersect_qc"
SAMPLE_FINAL="data/${SAMPLE_NAME}/${SAMPLE_NAME}_refintersect_qc_ldpruned"

echo "=============================================="
echo "Rebuild Reference Panel — intersected against ${SAMPLE_NAME} raw catalog"
echo "SLURM Job  : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "=============================================="
echo "Input VCF  : ${INPUT_VCF}"
echo "Python : $(which python)"
echo "plink2 : $(which plink2)"
echo ""

if [ ! -f "${INPUT_VCF}" ]; then
    echo "ERROR: Input VCF not found: ${INPUT_VCF}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 0: Extract the sample population's raw variant ID list.
# ---------------------------------------------------------------------------
echo "[0/6] Extracting ${SAMPLE_NAME} raw variant IDs..."
bcftools query -f '%ID\n' "${INPUT_VCF}" > "${SAMPLE_IDS}"
wc -l "${SAMPLE_IDS}"

# ---------------------------------------------------------------------------
# Step 1: Intersect the RAW reference panel against the sample's raw IDs,
#         restricted to the 644-sample curated reference panel.
# ---------------------------------------------------------------------------
echo "[1/6] Intersecting raw reference against ${SAMPLE_NAME}'s raw variant IDs..."
plink2 \
    --vcf "${RAW_REF_VCF}" \
    --allow-extra-chr \
    --extract "${SAMPLE_IDS}" \
    --keep "${REF_SAMPLES}" \
    --make-bed \
    --out "${REF_INTERSECT}"

# ---------------------------------------------------------------------------
# Step 2: QC-filter the intersected reference (Data_Preprocessing.ipynb flags).
# ---------------------------------------------------------------------------
echo "[2/6] QC-filtering intersected reference..."
plink2 \
    --bfile "${REF_INTERSECT}" \
    --allow-extra-chr \
    --chr 1-22 \
    --mind 0.02 \
    --geno 0.02 \
    --hwe 1e-6 \
    --maf 0.01 \
    --max-alleles 2 \
    --split-par b38 \
    --make-bed \
    --out "${REF_INTERSECT_QC}"

# ---------------------------------------------------------------------------
# Step 3: LD-prune the QC'd, intersected reference.
# ---------------------------------------------------------------------------
echo "[3/6] LD-pruning intersected reference..."
plink2 \
    --bfile "${REF_INTERSECT_QC}" \
    --allow-extra-chr \
    --indep-pairwise 50 5 0.1 \
    --out "${REF_FINAL}_prune"

plink2 \
    --bfile "${REF_INTERSECT_QC}" \
    --allow-extra-chr \
    --extract "${REF_FINAL}_prune.prune.in" \
    --recode vcf bgz \
    --out "${REF_FINAL}"

plink2 \
    --bfile "${REF_INTERSECT_QC}" \
    --allow-extra-chr \
    --extract "${REF_FINAL}_prune.prune.in" \
    --make-bed \
    --out "${REF_FINAL}"

# ---------------------------------------------------------------------------
# Step 4: QC-filter the sample population, restricted to the same raw
#         intersected ID set (symmetric with Step 2).
# ---------------------------------------------------------------------------
echo "[4/6] QC-filtering ${SAMPLE_NAME}, restricted to the raw intersected ID set..."
awk '{print $2}' "${REF_INTERSECT}.bim" > "snps/reference_${SAMPLE_NAME}_raw_intersect_ids.txt"
wc -l "snps/reference_${SAMPLE_NAME}_raw_intersect_ids.txt"

plink2 \
    --vcf "${INPUT_VCF}" \
    --allow-extra-chr \
    --extract "snps/reference_${SAMPLE_NAME}_raw_intersect_ids.txt" \
    --chr 1-22 \
    --mind 0.02 \
    --geno 0.02 \
    --hwe 1e-6 \
    --maf 0.01 \
    --max-alleles 2 \
    --split-par b38 \
    --make-bed \
    --out "${SAMPLE_INTERSECT_QC}"

# ---------------------------------------------------------------------------
# Step 5: LD-prune QC'd sample population.
# ---------------------------------------------------------------------------
echo "[5/6] LD-pruning ${SAMPLE_NAME}..."
plink2 \
    --bfile "${SAMPLE_INTERSECT_QC}" \
    --allow-extra-chr \
    --indep-pairwise 50 5 0.1 \
    --out "${SAMPLE_FINAL}_prune"

plink2 \
    --bfile "${SAMPLE_INTERSECT_QC}" \
    --allow-extra-chr \
    --extract "${SAMPLE_FINAL}_prune.prune.in" \
    --recode vcf bgz \
    --out "${SAMPLE_FINAL}"

plink2 \
    --bfile "${SAMPLE_INTERSECT_QC}" \
    --allow-extra-chr \
    --extract "${SAMPLE_FINAL}_prune.prune.in" \
    --make-bed \
    --out "${SAMPLE_FINAL}"

# ---------------------------------------------------------------------------
# Step 6: Convert both final VCFs to unphased tensor CSVs.
# ---------------------------------------------------------------------------
echo "[6/6] Converting to unphased tensor CSVs..."
python -u src/vcf_to_unphased_tensor.py \
    --input  "${REF_FINAL}.vcf.gz" \
    --input-type vcf \
    --output "${REF_FINAL}_unphased_tensor.csv"

python -u src/vcf_to_unphased_tensor.py \
    --input  "${SAMPLE_FINAL}.vcf.gz" \
    --input-type vcf \
    --output "${SAMPLE_FINAL}_unphased_tensor.csv"

echo ""
echo "=============================================="
echo "End time  : $(date)"
echo "=============================================="
echo "Reference final SNPs (${SAMPLE_NAME}-intersected):"
wc -l "${REF_FINAL}.bim" 2>/dev/null
echo "${SAMPLE_NAME} final SNPs:"
wc -l "${SAMPLE_FINAL}.bim" 2>/dev/null
