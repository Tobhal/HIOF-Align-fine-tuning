#!/bin/bash
#SBATCH --job-name al_negctrl        # Job name
#SBATCH --output logs/train/train_negctrl_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=240:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # Request 1 GPU
# #SBATCH --nodelist=hpc10     # Replace with the desired node name

source .venv/bin/activate

# Negative control for the best ALIGN configuration (PHOSC description + SupCon +
# Adam + cosine warmup + effective batch size 128 -- these hyperparameters are copied
# verbatim from saved_models/align-fine-tune/fold_3_aug_square/4/model_args.toml, the
# run identified as the best ALIGN setting). The ONLY difference from that run is
# --shuffle_descriptions, which randomly permutes the PHOSC description strings
# within each batch so image<->text correspondence is broken while the text side
# still sees the exact same real prompts. If retrieval collapses relative to run 4,
# that's evidence the model was exploiting image-attribute correspondence rather than
# just learning a separated embedding space.
#
# Run numbering is automatic (max existing + 1), so this lands in
# saved_models/align-fine-tune/fold_3_aug_square/<next number>/.

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
--slurm_job_desc "negative control: shuffled PHOSC descriptions, otherwise identical to run 4" \
--description "description" \
--lr_scheduler "cosine_warmup" \
--optimizer "adam" \
--lr_exploration_factor 1.5 \
--update_features \
--shuffle_descriptions
