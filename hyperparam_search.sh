#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=output/hyperparam_search_%A_%a.txt
#SBATCH --time=0-02:40            # time (DD-HH-MM)
#SBATCH --array=0-7

# Source the user's bashrc to get the full interactive environment
source ~/.bashrc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.10.0

source /home/elrawy/scratch/FSGS/fsgs_env/bin/activate

nvidia-smi

# LLFF dataset = (fern, flower, fortress, horns, leaves, orchids, room, trex)
datasets=(fern flower fortress horns leaves orchids room trex)
dataset_idx=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$dataset_idx]}


echo "Running job for dataset: $dataset"

# --- Experiment Configurations ---

echo "Running Experiment 4: Hybrid Model v3"
python train.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset} --model_path output/ours_hybrid_v3_8/${dataset} --eval --n_views 3 --densify_grad_threshold 0.0005 --prune_from_iter 1000 --prune_threshold 0.003 --depth_weight 0.05 --sample_pseudo_interval 1
python render.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ours_hybrid_v3_8/${dataset} --iteration 10000
python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ours_hybrid_v3_8/${dataset} --iteration 10000
