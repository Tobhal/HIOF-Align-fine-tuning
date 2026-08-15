#!/bin/bash
#SBATCH --job-name retrieval_metrics         # Job name
#SBATCH --output logs/retrieval_metrics/retrieval_metrics_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=8G                     # Memory (default units are megabytes)
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # inference over many checkpoints is much faster on GPU
# #SBATCH --nodelist=hpc7     # Replace with the desired node name

mkdir -p logs/retrieval_metrics

# Activate the venv that has torch/transformers/etc. installed
# (see requirements-retrieval.txt; created with: python3 -m venv .venv)
source .venv/bin/activate

# Runs inference over every already-completed ALIGN + CLIP checkpoint and writes
# results/retrieval_metrics.csv + results/retrieval_metrics.md
python retrieval_metrics_report.py \
  --split_name fold_3_aug_square \
  --align_nums 1 2 3 4 5 \
  --clip_nums 1 2 4 5 6 7 8 \
  --batch_size 32
