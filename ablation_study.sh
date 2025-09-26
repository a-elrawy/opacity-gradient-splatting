#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --output=output/ablation_study_%A_%a.txt
#SBATCH --time=0-05:40            # time (DD-HH:MM)
#SBATCH --array=0-7

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.10.0

source /home/elrawy/scratch/FSGS/fsgs_env/bin/activate

nvidia-smi

# LLFF dataset = (fern, flower, fortress, horns, leaves, orchids, room, trex)
datasets=(fern flower fortress horns leaves orchids room trex)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Running job for dataset: $dataset"

# --- Experiment Configurations ---

# 1. Baseline (Standard ADC)
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset} --model_path output/baseline/${dataset} --eval --n_views 3 --densify_grad_threshold 0.0005 --prune_from_iter 500 --prune_threshold 0.005 --depth_weight 0.05 --sample_pseudo_interval 1
# python render.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/baseline/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/baseline/${dataset} --iteration 10000

# 2. Ours (Full Model)
python train.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset} --model_path output/ours_final_4/${dataset} --eval --n_views 3 --use_error_densification --error_densify_threshold 0.0001 --prune_from_iter 2000 --prune_threshold 0.001 --depth_weight 0.05 --sample_pseudo_interval 1
python render.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ours_final_4/${dataset} --iteration 10000
python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ours_final_4/${dataset} --iteration 10000

# 3. Ablation 1: No Error-Driven ADC
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset} --model_path output/ablation1_no_error_adc/${dataset} --eval --n_views 3 --densify_grad_threshold 0.0005 --prune_from_iter 2000 --prune_threshold 0.001 --depth_weight 0.05 --sample_pseudo_interval 1
# python render.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ablation1_no_error_adc/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ablation1_no_error_adc/${dataset} --iteration 10000

# 4. Ablation 2: No Delayed Pruning
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset} --model_path output/ablation2_no_delayed_pruning/${dataset} --eval --n_views 3 --use_error_densification --error_densify_threshold 0.0001 --prune_from_iter 500 --prune_threshold 0.001 --depth_weight 0.05 --sample_pseudo_interval 1
# python render.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ablation2_no_delayed_pruning/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ablation2_no_delayed_pruning/${dataset} --iteration 10000

# 5. Ablation 3: No Depth-Correlation Loss
# python train.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset} --model_path output/ablation3_no_depth_loss/${dataset} --eval --n_views 3 --use_error_densification --error_densify_threshold 0.0001 --prune_from_iter 2000 --prune_threshold 0.001 --depth_weight 0.0 --sample_pseudo_interval 1
# python render.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ablation3_no_depth_loss/${dataset} --iteration 10000
# python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/ --model_path output/ablation3_no_depth_loss/${dataset} --iteration 10000
