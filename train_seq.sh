#!/bin/bash
#SBATCH --job-name=test 
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Two times the number of GPUs for 2080ti in savio3_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1

#SBATCH --time=00:60:00

set -euo pipefail

# Run from the repository root
cd "$SLURM_SUBMIT_DIR"

# Activate Python environment
source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh
conda activate calypo
if [ $? -ne 0 ]; then
  echo "Error: Could not activate virtual environment. Exiting."
  exit 1
fi

if [ -f .env ]; then # store API keys in a local .env file then search for the variables
  export $(grep -v '^#' .env | xargs)
fi

# Avoid CPU oversubscription
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1


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
