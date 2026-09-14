#!/bin/bash
#SBATCH --account=REPLACE_ME
#SBATCH --array=0-6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --output=output/logs/mip_default_%A_%a.txt
#SBATCH --time=0-02:00
#SBATCH --job-name=ogs_mip

# Default Mip-NeRF 360 (24 views, photometric-only densify grads).
#   sbatch --account=$OGS_ACCOUNT slurm/run_mipnerf_default.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

SCENES=(bicycle bonsai counter garden kitchen room stump)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SCENE=${SCENES[$IDX]}
DATA_SRC=${OGS_ROOT}/dataset/mipnerf360/${SCENE}
MODEL_OUT=${OGS_ROOT}/output/mip_default/${SCENE}
mkdir -p output/logs "${MODEL_OUT}"

python train.py \
    --source_path "${DATA_SRC}" \
    --model_path "${MODEL_OUT}" \
    --eval --n_views 24 \
    --use_error_densification \
    --error_densify_threshold 0.001 \
    --prune_from_iter 2000 \
    --prune_threshold 0.001 \
    --depth_weight 0.05 \
    --max_gaussians 250000 \
    --sample_pseudo_interval 10

python render.py --source_path "${DATA_SRC}" -m "${MODEL_OUT}" --iteration 10000
python metrics.py -s "${MODEL_OUT}" -m "${MODEL_OUT}"
