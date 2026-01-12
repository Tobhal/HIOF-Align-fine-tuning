#!/bin/bash
#SBATCH --job-name des_train_align         # Job name
#SBATCH --output logs/train/train_%j.log    # Standard output and error log
# #SBATCH --partition a100          # Partition (queue) to submit to
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=240:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # Request 1 GPU – dette forteller SLURM at du vil bruke GPU-er
# #SBATCH --nodelist=hpc10     # Replace with the desired node name
# #SBATCH --nodelist=hpc6     # Replace with the desired node name

# Load any necessary modules or activate a virtual environment
# module load cuda # Example: Load CUDA module

# Commands to run your GPU job
# python align_fine_tuning.py --batch_size 32 --accumulation_steps 4 --augmented --epochs 500 --lr 0.0001 --maximize --stop_patience 50 --save --save_every 10 --verbose --loss_func contrastive --split_name fold_3_aug_square --validate --ignore_checkpoint --description description --slurm_job_id $SLURM_JOB_ID --slurm_job_desc "d phosc"

srun python align_fine_tuning.py \
--batch_size 32 \
--accumulation_steps 8 \
--augmented \
--epochs 500 \
--lr 1e-10 \
--maximize \
--stop_patience 50 \
--save \
--save_every 10 \
--verbose \
--loss_func contrastive \
--split_name fold_3_aug_square \
--validate \
--ignore_checkpoint \
--slurm_job_id $SLURM_JOB_ID \
--slurm_job_desc "d phosc" \
--description "description_long" \
--lr_scheduler "cosine" \
--lr_exploration_factor 1.5
