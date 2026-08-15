#!/bin/bash
#SBATCH --job-name negctrl_check     # Job name
#SBATCH --output logs/negctrl_epoch50/check_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=8G                     # Memory (default units are megabytes)
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # GPU speeds up the checkpoint loading + feature extraction a lot

source .venv/bin/activate

# Matched-epoch-50 retrieval comparison: run 4 (real prompts) vs runs 6/7
# (shuffled prompts, negative control) -- for results/findings.md item 2.
python retrieval_metrics_report.py \
  --align_nums 4 6 7 \
  --split_name "fold_3_aug_square" \
  --checkpoint_name "epoch 50.pt" \
  --skip_clip \
  --out_dir results/negctrl_epoch50 \
  --write_per_run_report
