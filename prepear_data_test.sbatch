#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --output=logs/prepear_data/prep_data_%j.log
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00

set -euo pipefail

mkdir -p logs/prepear_data

# Load the same Python module/version used when creating ../align2
# Replace this with the correct module on your cluster:
module load python/3.8 || module load python3 || true

source ../align2/bin/activate

echo "Python path:"
which python || true
python --version || true
echo "Venv:"
echo "$VIRTUAL_ENV"

../align3/bin/python prepear_data.py -s --folds 0 2 4 -n aug_square -a --down_scale_factor 0.2