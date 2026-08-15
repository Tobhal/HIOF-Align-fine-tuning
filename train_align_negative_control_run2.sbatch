#!/bin/bash
#SBATCH --job-name al_negctrl2       # Job name
#SBATCH --output logs/train/train_negctrl2_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=8            # More headroom for the threaded image loader below
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=240:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # Request 1 GPU
# #SBATCH --nodelist=hpc10     # Replace with the desired node name

source .venv/bin/activate

# Second, independent negative-control run -- identical to
# train_align_negative_control.sbatch (job 47844, still running) except:
#   --workers 8   parallelizes the per-batch PIL image loading (pure speed fix,
#                 no effect on the actual training math -- see align_fine_tune.py's
#                 train_epoch/validate_epoch). Measured ~5-10% overall, not more:
#                 GPU compute dominates per-batch time here, not data loading.
#   --cpus-per-task=8   gives that thread pool room to actually run in parallel
#                 under SLURM's cgroup (job 47844 only got 1).
# Still full FP32 (no mixed precision) -- kept numerically as close to how run 4
# was originally trained as job 47844 already is.
#
# Bonus: a second independent shuffled-prompt run also cross-checks the negative
# control result itself, given how much run-to-run variance showed up between
# nominally-identical configs in task 1 (see results/retrieval_metrics_writeup.md,
# finding 5).

srun python align_fine_tune.py \
--name "align-fine-tune" \
--batch_size 8 \
--accumulation_steps 16 \
--augmented \
--epochs 500 \
--lr 1e-5 \
--stop_patience 50 \
--save \
--save_every 10 \
--verbose \
--loss_func "supcon" \
--split_name "fold_3_aug_square" \
--validate \
--ignore_checkpoint \
--slurm_job_id $SLURM_JOB_ID \
--slurm_job_desc "negative control run 2: shuffled PHOSC descriptions, otherwise identical to run 4, threaded image loading for speed" \
--description "description" \
--lr_scheduler "cosine_warmup" \
--optimizer "adam" \
--lr_exploration_factor 1.5 \
--update_features \
--shuffle_descriptions \
--workers 8
