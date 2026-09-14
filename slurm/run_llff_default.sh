#!/bin/bash
#SBATCH --account=REPLACE_ME
#SBATCH --array=0-7
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --output=output/logs/llff_default_%A_%a.txt
#SBATCH --time=0-00:40
#SBATCH --job-name=ogs_llff

# Default LLFF-8 (3 views, images_8, photometric-only densify grads).
# Before sbatch: edit --account above OR export OGS_ACCOUNT and use:
#   sbatch --account=$OGS_ACCOUNT slurm/run_llff_default.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

SCENES=(fern flower fortress horns leaves orchids room trex)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SCENE=${SCENES[$IDX]}
DATA_SRC=${OGS_ROOT}/dataset/nerf_llff_data/${SCENE}
MODEL_OUT=${OGS_ROOT}/output/llff_default/${SCENE}
mkdir -p output/logs "${MODEL_OUT}"

python train.py \
    --source_path "${DATA_SRC}" \
    --model_path "${MODEL_OUT}" \
    --eval --n_views 3 \
    --images images_8 \
    --use_error_densification \
    --error_densify_threshold 0.0005 \
    --prune_from_iter 2000 \
    --prune_threshold 0.001 \
    --depth_weight 0.05 \
    --max_gaussians 32000 \
    --sample_pseudo_interval 10

python render.py --source_path "${DATA_SRC}" -m "${MODEL_OUT}" --iteration 10000
python metrics.py -s "${MODEL_OUT}" -m "${MODEL_OUT}"
