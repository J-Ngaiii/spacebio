#!/bin/bash
# Fine-tune BulkFormer on one Savio GPU. Submit from the repository root:
#   sbatch slurm/train_bulkformer.sbatch --checkpoint-path /path/to/checkpoint.pt [other train_bulkformer.py options]
# Command-line flags override the #SBATCH lines, e.g. sbatch --time=04:00:00 slurm/train_bulkformer.sbatch
#SBATCH --job-name=spacebio-bulkformer
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -eo pipefail

if [[ ! -f scripts/train_bulkformer.py ]]; then
    echo "Submit this job from the repository root." >&2
    exit 1
fi

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "${SPACEBIO_ENV:-spacebio}"

nvidia-smi
python scripts/run_pipe.py --stage interim --models bulkformer
python scripts/train_bulkformer.py --device cuda --wandb-mode offline "$@"
