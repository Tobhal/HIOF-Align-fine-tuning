#!/bin/bash
#SBATCH --job-name embeddings         # Job name
#SBATCH --output logs/embeddings/embeddings_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=8G                     # Memory (default units are megabytes)
#SBATCH --time=01:00:00              # Time limit hrs:min:sec (quick: a couple minutes)
#SBATCH --gres=gpu:1

mkdir -p logs/embeddings

source .venv/bin/activate

# Frozen (pretrained) vs. fine-tuned (best config, run 4) ALIGN image-embedding
# t-SNE, plus the separation-margin numbers behind it.
python embedding_visualization.py \
  --split_name fold_3_aug_square \
  --batch_size 32
