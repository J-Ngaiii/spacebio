#!/bin/bash
#SBATCH --job-name=performer-1L
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # verify the CPU:GPU ratio for A40 in the Savio docs
#SBATCH --gres=gpu:A40:1
#SBATCH --time=03:00:00            # one model, one layer; adjust after a short test run
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh
conda activate spacebio || { echo "Could not activate conda env"; exit 1; }

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
nvidia-smi

srun python -m scripts.train_sequences \
    --data data/processed/sequence/harmonized/harmonized_training_gz.h5ad \
    --model performer \
    --depth 1 \
    --epochs 100 \
    --batch-size 16 \
    --lr 3e-4 \
    --wandb-project spacebio-sequence-models \
    --wandb-group harmonized-training \
    --wandb-mode online          # switch to offline if compute nodes lack internet