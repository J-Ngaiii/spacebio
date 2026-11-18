#!/bin/bash
#SBATCH --job-name=ig-hyena-full
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

# Configuration
PROTEIN=TP53
MODEL=hyena_depth3

CHECKPOINT=data/results/sequence_models/hyena_42_20261109-193302/hyena.pt
DATA=data/space/osdr_gene_protein_intersection_intrain.h5ad

OUTPUT_DIR=data/results/interpretability/${MODEL}
VISUAL_DIR=data/visuals

# Step 1: Compute IG for all eligible test samples
srun python -m scripts.attributions \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --protein "$PROTEIN" \
    --output-dir "$OUTPUT_DIR" \
    --all-samples \
    --batch-size 4 \
    --internal-batch-size 8 \
    --n-steps 64 \
    --device cuda

# Step 2: Select top genes, then top samples
python -m scripts.visualize_attributions \
    --input "${OUTPUT_DIR}/${PROTEIN}_ig.npz" \
    --output-dir "$VISUAL_DIR" \
    --space-or-earth "space" \
    --top-genes 10 \
    --top-samples 25 \
    --sample-ranking mean_abs \
    --format png

echo "Attribution computation and visualization complete."
