#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8000M
#SBATCH --output=output/ablation_study_mipnerf_%A_%a.txt
#SBATCH --time=0-00:40            # time (DD-HH:MM)
#SBATCH --array=0-6

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.10.0

source /home/elrawy/scratch/FSGS/fsgs_env/bin/activate

nvidia-smi

# Mip-NeRF 360 dataset = (bicycle, bonsai, counter, garden, kitchen, stump, room)
datasets=(bicycle bonsai counter garden kitchen stump room)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Running job for dataset: $dataset"

# --- Experiment Configurations ---

# 1. Baseline (Standard ADC)
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset} --model_path output/mipnerf_baseline/${dataset} --eval --n_views 24 --densify_grad_threshold 0.0005 --prune_from_iter 500 --prune_threshold 0.005 --depth_weight 0.05 --depth_pseudo_weight 0.03
# python render.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_baseline/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_baseline/${dataset} --iteration 10000

# 2. Ours (Adjusted for Denser Datasets)
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset} --model_path output/mipnerf_ours_adjusted_final_4/${dataset} --eval --n_views 24 --use_error_densification --error_densify_threshold 0.0001 --prune_from_iter 2000 --prune_threshold 0.001 --max_gaussians 250000 --depth_weight 0.05 --depth_pseudo_weight 0.03
python render.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_ours_adjusted_final_4/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_ours_adjusted_final_4/${dataset} --iteration 10000

# 3. Ablation 1: No Error-Driven ADC
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset} --model_path output/mipnerf_ablation1_no_error_adc/${dataset} --eval --n_views 24 --densify_grad_threshold 0.0005 --prune_from_iter 2000 --prune_threshold 0.001 --depth_weight 0.05 --depth_pseudo_weight 0.03
# python render.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_ablation1_no_error_adc/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_ablation1_no_error_adc/${dataset} --iteration 10000

# 4. Ablation 2: No Delayed Pruning
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset} --model_path output/mipnerf_ablation2_no_delayed_pruning/${dataset} --eval --n_views 24 --use_error_densification --error_densify_threshold 0.0001 --prune_from_iter 500 --prune_threshold 0.001 --depth_weight 0.05 --depth_pseudo_weight 0.03
# python render.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_ablation2_no_delayed_pruning/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_ablation2_no_delayed_pruning/${dataset} --iteration 10000

# 5. Ablation 3: No Depth-Correlation Loss
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset} --model_path output/mipnerf_ablation3_no_depth_loss/${dataset} --eval --n_views 24 --use_error_densification --error_densify_threshold 0.0001 --prune_from_iter 2000 --prune_threshold 0.001 --depth_weight 0.0 --depth_pseudo_weight 0.03
# python render.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_ablation3_no_depth_loss/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/ --model_path output/mipnerf_ablation3_no_depth_loss/${dataset} --iteration 10000 