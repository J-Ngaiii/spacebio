#!/bin/bash
#SBATCH --job-name=arch-ablation
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=48:00:00
#SBATCH --array=0-107%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ==========================================
# Experiment configurations
# ==========================================

MODELS=(hyena performer)
DEPTHS=(1 3 5 10 15 25 30 50)
WIDTHS=(64 128 256)
SEEDS=(42)

NM=${#MODELS[@]}
ND=${#DEPTHS[@]}
NW=${#WIDTHS[@]}
NS=${#SEEDS[@]}

# Map Slurm task ID to experiment configuration

ID=$SLURM_ARRAY_TASK_ID

DEPTH=${DEPTHS[$((ID % ND))]}
WIDTH=${WIDTHS[$(((ID / ND) % NW))]}
MODEL=${MODELS[$(((ID / (ND * NW)) % NM))]}
SEED=${SEEDS[$((ID / (ND * NW * NM)))]}

RUN_NAME="${MODEL}_d${DEPTH}_w${WIDTH}_s${SEED}"
OUTPUT_DIR="data/results/architecture_ablation/${RUN_NAME}"

# ==========================================
# Environment setup
# ==========================================

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

source /global/home/users/jonathanngai/miniconda3/etc/profile.d/conda.sh

conda activate spacebio || {
    echo "Could not activate conda env"
    exit 1
}

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export WANDB_NAME="$RUN_NAME"

# ==========================================
# Experiment information
# ==========================================

echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "Model: $MODEL"
echo "Depth: $DEPTH"
echo "Hidden Dimension: $WIDTH"
echo "Seed: $SEED"
echo "Run Name: $RUN_NAME"
echo "Output Directory: $OUTPUT_DIR"

nvidia-smi

# ==========================================
# Training
# ==========================================

srun python -m scripts.train_sequences \
    --data data/processed/sequence/harmonized/harmonized_training_intersection.h5ad \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --return-predictions \
    --depth "$DEPTH" \
    --d-model "$WIDTH" \
    --seed "$SEED" \
    --epochs 100 \
    --batch-size 16 \
    --sd-floor 0.01 \
    --lr 3e-4 \
    --patience 100 \
    --wandb-project spacebio-sequence-models \
    --wandb-group architecture-ablation \
    --wandb-mode online