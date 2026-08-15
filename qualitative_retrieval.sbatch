#!/bin/bash
#SBATCH --job-name qualitative         # Job name
#SBATCH --output logs/qualitative/qualitative_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=8G                     # Memory (default units are megabytes)
#SBATCH --time=01:00:00              # Time limit hrs:min:sec (this is quick: ~1-2 min per model)
#SBATCH --gres=gpu:1

mkdir -p logs/qualitative

source .venv/bin/activate

# Best ALIGN config (run 4): 10 unseen query words, top-5 retrievals each.
python qualitative_retrieval.py \
  --split_name fold_3_aug_square \
  --batch_size 32 \
  --model_dir saved_models/align-fine-tune/fold_3_aug_square/4 \
  --out_dir results/qualitative

# Comparison: same 10 words (same seed), run 1 (word-mode) instead of PHOSC
# description -- see results/qualitative_writeup.md for why this comparison
# is worth having.
python qualitative_retrieval.py \
  --split_name fold_3_aug_square \
  --batch_size 32 \
  --model_dir saved_models/align-fine-tune/fold_3_aug_square/1 \
  --out_dir results/qualitative_run1_word_mode
