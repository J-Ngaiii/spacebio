#!/bin/bash
#SBATCH --job-name=masking-pearson
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh
conda activate spacebio

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi

# Run from the SpaceBio project root: sbatch scripts/masking_pearson.sh
DATA=data/processed/sequence/harmonized/harmonized_training_intersection.h5ad
HYENA_CHECKPOINT=data/results/sequence_models/hyena_42_20261109-193302/hyena.pt
# Set this to the actual Performer checkpoint from your training run.
PERFORMER_CHECKPOINT=data/results/sequence_models/performer_42_YYYYMMDD-HHMMSS/performer.pt
OUTPUT_DIR=data/results/masking_pearson

# Hyena
srun python -m scripts.masking_pearson \
    --checkpoint "$HYENA_CHECKPOINT" \
    --data "$DATA" \
    --output-dir "$OUTPUT_DIR/hyena" \
    --split test \
    --percentages 0 10 20 30 50 70 90 \
    --batch-size 32 \
    --seed 42 \
    --device cuda

# Performer: uncomment after setting PERFORMER_CHECKPOINT above.
# srun python -m scripts.masking_pearson \
#     --checkpoint "$PERFORMER_CHECKPOINT" \
#     --data "$DATA" \
#     --output-dir "$OUTPUT_DIR/performer" \
#     --split test \
#     --percentages 0 10 20 30 50 70 90 \
#     --batch-size 32 \
#     --seed 42 \
#     --device cuda

echo "RNA masking evaluation complete."
