#!/bin/bash
#SBATCH --job-name=ig_performer
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/performer_ig_%j.out
#SBATCH --error=slurm/logs/performer_ig_%j.err

set -euo pipefail

# Activate environment
source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh
conda activate spacebio

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi

# Run Integrated Gradients
python -m scripts.attributions \
    --checkpoint data/results/performer_42_20261109-115613/performer.pt \
    --data data/processed/sequence/harmonized/harmonized_training_intersection.h5ad \
    --protein TP53 \
    --output-dir data/results/interpretability/performer \
    --num-samples 8 \
    --batch-size 2 \
    --internal-batch-size 8 \
    --n-steps 16 \
    --device cuda

echo "Integrated Gradients completed."
