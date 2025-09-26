#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=26000M
#SBATCH --output=output/train_%A_%a.txt
#SBATCH --time=0-4:40            # time (DD-HH:MM)
#SBATCH --array=0-7

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.10.0

source /home/elrawy/scratch/FSGS/fsgs_env/bin/activate

nvidia-smi


datasets=(bicycle bonsai counter garden kitchen room stump)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Running job for dataset: $dataset"





python train.py  --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset} --model_path output/${dataset} --eval  --n_views 24 --depth_pseudo_weight 0.03 --use_error_densification  --error_densify_threshold 0.0001
python render.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/  --model_path  output/${dataset} --iteration 10000
python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/mipnerf360/${dataset}/  --model_path  output/${dataset} --iteration 10000

