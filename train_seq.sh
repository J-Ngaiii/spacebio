#!/bin/bash
#SBATCH --job-name=spacebio-sequence
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/sequence_%j.out
#SBATCH --error=logs/sequence_%j.err

set -euo pipefail

# Run from the repository root
cd "$SLURM_SUBMIT_DIR"

# Activate Python environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spacebio

# Avoid CPU oversubscription
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

# W&B credentials should already be configured
# via `wandb login` or WANDB_API_KEY.
export WANDB_MODE=online

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"

nvidia-smi

# Train both models sequentially on the allocated GPU
srun python -m scripts.train_sequence \
    --data data/processed/sequence/harmonized/harmonized_training_gz.h5ad \
    --model both \
    --epochs 100 \
    --batch-size 16 \
    --lr 3e-4 \
    --wandb-project spacebio-sequence-models \
    --wandb-group harmonized-training \
    --wandb-mode online
