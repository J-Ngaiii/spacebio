#!/bin/bash
# Fine-tune BulkFormer on one Savio GPU. Submit from the repository root:
#   sbatch slurm/train_bulkformer.sbatch [train_bulkformer.py options, e.g. --checkpoint zhfanrui/BulkFormer-147M]
# Download the checkpoint once on a login node first, in case compute nodes have no internet:
#   huggingface-cli download zhfanrui/BulkFormer-37M
# Command-line flags override the #SBATCH lines, e.g. sbatch --time=04:00:00 slurm/train_bulkformer.sbatch
#SBATCH --job-name=spacebio-bulkformer
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --cpus-per-task=8  # Savio requires 8 CPUs per A40
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
