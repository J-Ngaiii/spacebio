#!/bin/bash
#SBATCH --job-name=performer-1L
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Activate environment
source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh
conda activate spacebio

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi

python -m scripts.attributions \
    --checkpoint data/results/sequence_models/hyena_42_20261109-193302/hyena.pt \
    --data data/processed/sequence/harmonized/harmonized_training_intersection.h5ad \
    --protein TP53 \
    --output-dir data/results/interpretability/hyena_depth3 \
    --num-samples 8 \
    --batch-size 2 \
    --internal-batch-size 8 \
    --n-steps 16 \
    --device cuda
