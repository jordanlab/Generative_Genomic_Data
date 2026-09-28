#!/bin/bash
#SBATCH --job-name=cwgan_ATQ3
#SBATCH --account=gts-jhamilton80
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data/slurm/out/cwgan_ATQ3_%j.out
#SBATCH --error=/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data/slurm/err/cwgan_ATQ3_%j.err

BASE_DIR="/storage/home/hcoda1/8/sverma87/scratch/Generative_Genomic_Data"

ATQ3_CSV="${BASE_DIR}/data/ATQ3/ATQ3_qc2_grch38_final_ldpruned_unphased_tensor.csv"
REF_CSV="${BASE_DIR}/data/reference/extractedChrAll_common_snps_forReference_qc_ldpruned_unphased_tensor.csv"
FAM_FILE="${BASE_DIR}/data/reference/extractedChrAll_common_snps_forReference.fam"

GEN_DATA_DIR="${BASE_DIR}/results/ATQ3/cWGAN_gen-data"
PCA_DIR="${BASE_DIR}/results/ATQ3/cWGAN_PCA"

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

# ===========================================================================
# Setup
# ===========================================================================
echo "=============================================="
echo "Conditional WGAN-GP  —  ATQ3 Dataset"
echo "SLURM Job  : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "=============================================="
echo "ATQ3 input  : ${ATQ3_CSV}"
echo "Ref input   : ${REF_CSV}"
echo "FAM file    : ${FAM_FILE}"
echo "Target admix: ${TARGET_ADMIXTURE:-BAQ-fitted Dirichlet (conservative)}"
echo "Seed        : ${SEED}"
echo ""

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
    --data_input         ${ATQ3_CSV} \
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
    --sample_name         ATQ3"

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
