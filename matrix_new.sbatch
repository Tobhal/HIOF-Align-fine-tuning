#!/bin/bash
#SBATCH --job-name matrix_plot         # Job name
#SBATCH --output logs/matrix_new/matrix_new_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
# #SBATCH --nodelist=hpc7     # Replace with the desired node name

# Load Python environment or any other required modules
# module load python3

# Execute the script
python matrix_new.py --split_name fold_3_aug_square --nums 23 24 --heatmap