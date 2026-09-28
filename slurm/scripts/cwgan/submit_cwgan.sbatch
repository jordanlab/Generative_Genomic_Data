#!/bin/bash
#SBATCH --job-name=cwgan
#SBATCH --account=gts-jhamilton80
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data/slurm/out/cwgan_%x_%j.out
#SBATCH --error=/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data/slurm/err/cwgan_%x_%j.err

# ===========================================================================
# Usage:
#   sbatch --export=ALL,SAMPLE_NAME=VDC2 submit_cwgan.sbatch
#   sbatch --export=ALL,SAMPLE_NAME=VDC2,SEED=43 submit_cwgan.sbatch
#   sbatch --export=ALL,SAMPLE_NAME=VDC2,SEED=43,TARGET_ADMIXTURE="0.6,0.2,0.2" submit_cwgan.sbatch
#
# For samples with non-standard CSV names, pass INPUT_CSV explicitly:
#   sbatch --export=ALL,SAMPLE_NAME=COR,INPUT_CSV=/path/to/file.csv submit_cwgan.sbatch
# ===========================================================================

if [ -z "${SAMPLE_NAME}" ]; then
    echo "ERROR: SAMPLE_NAME is required. Pass via --export=ALL,SAMPLE_NAME=<name>"
    exit 1
fi

BASE_DIR="/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data"

# Default paths follow rebuild_reference.sbatch's output naming: the query and
# the reference are the matched pair produced by the SAME intersect-first run.
# Do not mix a rebuilt query with the old panel-wide reference — the two share
# almost no SNPs and the pipeline silently trains on the tiny intersection.
# Override with INPUT_CSV / REF_CSV if either file has a non-standard name.
INPUT_CSV="${INPUT_CSV:-${BASE_DIR}/data/${SAMPLE_NAME}/${SAMPLE_NAME}_refintersect_qc_ldpruned_unphased_tensor.csv}"
REF_CSV="${REF_CSV:-${BASE_DIR}/data/reference/extractedChrAll_${SAMPLE_NAME}_intersect_qc_ldpruned_unphased_tensor.csv}"
FAM_FILE="${BASE_DIR}/data/reference/extractedChrAll_common_snps_forReference.fam"

for f in "${INPUT_CSV}" "${REF_CSV}" "${FAM_FILE}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required input not found: ${f}"
        echo "       Run rebuild_reference.sbatch for SAMPLE_NAME=${SAMPLE_NAME} first."
        exit 1
    fi
done

GEN_DATA_DIR="${BASE_DIR}/results/${SAMPLE_NAME}/cWGAN_gen-data"
PCA_DIR="${BASE_DIR}/results/${SAMPLE_NAME}/cWGAN_PCA"

# ===========================================================================
# Parameters  (override via --export=ALL,VAR=val)
# ===========================================================================
N_SYNTHETIC=${N_SYNTHETIC:-200}
N_COMPONENTS=${N_COMPONENTS:-50}
SEED=${SEED:-42}
WGAN_EPOCHS=${WGAN_EPOCHS:-3000}
WGAN_LATENT_DIM=${WGAN_LATENT_DIM:-32}
BATCH_SIZE=${BATCH_SIZE:-64}
ALPHA_CONCENTRATION=${ALPHA_CONCENTRATION:-20.0}
TARGET_ADMIXTURE=${TARGET_ADMIXTURE:-""}
REAL_COLOR=${REAL_COLOR:-"#E63946"}

# ===========================================================================
# Setup
# ===========================================================================
echo "=============================================="
echo "Conditional WGAN-GP  —  ${SAMPLE_NAME} Dataset"
echo "SLURM Job  : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "=============================================="
echo "Input CSV   : ${INPUT_CSV}"
echo "Ref input   : ${REF_CSV}"
echo "FAM file    : ${FAM_FILE}"
echo "Target admix: ${TARGET_ADMIXTURE:-BAQ-fitted Dirichlet (conservative)}"
echo "Seed        : ${SEED}"
echo ""

if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: Input CSV not found: ${INPUT_CSV}"
    exit 1
fi

cd "${BASE_DIR}"
mkdir -p "${GEN_DATA_DIR}" "${PCA_DIR}" slurm/out slurm/err

source /storage/home/hcoda1/8/sverma87/miniconda3/etc/profile.d/conda.sh
GPU_ENV_PREFIX="/storage/scratch1/8/sverma87/conda-envs/GGD_gpu_env"
if [ -d "${GPU_ENV_PREFIX}" ]; then
    conda activate "${GPU_ENV_PREFIX}"
else
    conda activate GGD_env
fi

echo "Python : $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ===========================================================================
# Build command
# ===========================================================================
CMD="python -u src/conditional_wgan_pipeline.py \
    --data_input         ${INPUT_CSV} \
    --ref_input          ${REF_CSV} \
    --fam                ${FAM_FILE} \
    --n_synthetic        ${N_SYNTHETIC} \
    --output             ${GEN_DATA_DIR}/synthetic_genotypes_seed${SEED}.csv \
    --plot               ${PCA_DIR}/pca_overlay_seed${SEED}.png \
    --n_components       ${N_COMPONENTS} \
    --seed               ${SEED} \
    --wgan_epochs        ${WGAN_EPOCHS} \
    --wgan_latent_dim    ${WGAN_LATENT_DIM} \
    --batch_size         ${BATCH_SIZE} \
    --alpha_concentration ${ALPHA_CONCENTRATION} \
    --sample_name        ${SAMPLE_NAME} \
    --real_color         \"${REAL_COLOR}\""

if [ -n "${TARGET_ADMIXTURE}" ]; then
    CMD="${CMD} --target_admixture ${TARGET_ADMIXTURE}"
fi

echo "Running: ${CMD}"
echo ""
eval ${CMD}

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "=============================================="
ls -lh "${GEN_DATA_DIR}/"*"seed${SEED}"* 2>/dev/null
ls -lh "${PCA_DIR}/"*"seed${SEED}"* 2>/dev/null

exit ${EXIT_CODE}
