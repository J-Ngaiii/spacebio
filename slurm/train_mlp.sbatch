#!/bin/bash
# Train the MLP on one Savio GPU. Submit from the repository root:
#   sbatch slurm/train_mlp.sbatch [train_mlp.py options, e.g. --epochs 50]
# Command-line flags override the #SBATCH lines, e.g. sbatch --time=01:00:00 slurm/train_mlp.sbatch
#SBATCH --job-name=spacebio-mlp
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --cpus-per-task=2  # Savio requires 2 CPUs per GTX2080TI
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -eo pipefail

if [[ ! -f scripts/train_mlp.py ]]; then
    echo "Submit this job from the repository root." >&2
    exit 1
fi

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "${SPACEBIO_ENV:-spacebio}"

nvidia-smi
python scripts/run_pipe.py --stage interim --models simple-mlp
python scripts/train_mlp.py --device cuda --wandb-mode offline "$@"
