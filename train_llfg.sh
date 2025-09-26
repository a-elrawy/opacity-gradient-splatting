#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --output=output/train_%A_%a.txt
#SBATCH --time=0-25:40            # time (DD-HH:MM)
#SBATCH --array=0-7

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.10.0

source /home/elrawy/scratch/FSGS/fsgs_env/bin/activate

nvidia-smi


# LLFF dataset = (fern, flower, fortress, horns, leaves, orchids, room, trex)
datasets=(fern flower fortress horns leaves orchids room trex)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Running job for dataset: $dataset"

python train.py  --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset} --model_path output/${dataset} --eval  --n_views 3 --sample_pseudo_interval 1 --use_error_densification  --error_densify_threshold 0.0001 --prune_from_iter 2000 --prune_threshold 0.001
python render.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/  --model_path  output/${dataset} --iteration 10000
python metrics.py --source_path /home/elrawy/scratch/FSGS/dataset/nerf_llff_data/${dataset}/  --model_path  output/${dataset} --iteration 10000

