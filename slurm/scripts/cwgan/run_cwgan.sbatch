#!/bin/bash
#SBATCH --job-name=cwgan
#SBATCH --account=gts-jhamilton80
#SBATCH --ntasks=1
#SBATCH --output=/storage/home/hcoda1/8/sverma87/ggd_results/cwgan_%x_%j.out
#SBATCH --error=/storage/home/hcoda1/8/sverma87/ggd_results/cwgan_%x_%j.err

# Fully self-contained: interpreter, source, inputs and outputs all live on the
# home filesystem. Nothing touches /storage/scratch1, which is mid-outage.
PY="${HOME}/ggd_env/bin/python"
SRC="${HOME}/ggd_src"
DATA="${HOME}/ggd_data"
OUT="${HOME}/ggd_results/${SAMPLE_NAME}"
mkdir -p "${OUT}"

SEED=${SEED:-42}
WGAN_EPOCHS=${WGAN_EPOCHS:-1000}
N_SYNTHETIC=${N_SYNTHETIC:-200}

INPUT_CSV="${DATA}/${SAMPLE_NAME}_refintersect_qc_ldpruned_unphased_tensor.csv"
REF_CSV="${DATA}/extractedChrAll_${SAMPLE_NAME}_intersect_qc_ldpruned_unphased_tensor.csv"
FAM="${DATA}/extractedChrAll_common_snps_forReference.fam"

echo "=== ${SAMPLE_NAME} seed ${SEED} | job ${SLURM_JOB_ID} | $(hostname) ==="
echo "start $(date)"
for f in "${INPUT_CSV}" "${REF_CSV}" "${FAM}"; do
    [ -f "$f" ] || { echo "ERROR missing: $f"; exit 1; }
done
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

cd "${SRC}" || exit 1
$PY -u conditional_wgan_pipeline.py \
    --data_input "${INPUT_CSV}" \
    --ref_input  "${REF_CSV}" \
    --fam        "${FAM}" \
    --n_synthetic ${N_SYNTHETIC} \
    --output     "${OUT}/synthetic_genotypes_seed${SEED}.csv" \
    --plot       "${OUT}/pca_overlay_seed${SEED}.png" \
    --n_components 50 \
    --seed       ${SEED} \
    --wgan_epochs ${WGAN_EPOCHS} \
    --wgan_latent_dim 32 \
    --batch_size 64 \
    --alpha_concentration 20.0 \
    --sample_name ${SAMPLE_NAME}
RC=$?
echo "exit=${RC}  end $(date)"
ls -lh "${OUT}"/*seed${SEED}* 2>/dev/null
exit ${RC}
