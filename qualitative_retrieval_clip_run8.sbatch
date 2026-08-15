#!/bin/bash
#SBATCH --job-name clip8_ool_check   # Job name
#SBATCH --output logs/qualitative/clip_run8_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=8G                     # Memory (default units are megabytes)
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1

source .venv/bin/activate

# Same OOL qualitative probe A4/run1 got (results/qualitative_writeup.md), run
# against CLIP run 8 (SupCon, eff. batch 64, cosine_warmup, word-mode -- the
# "CLIP beats ALIGN" run, currently only checked on aggregate mAP). Uses the
# *literal* 10 query words A4/run1 used (results/qualitative/query_words.txt),
# not a resample, so the three results are directly comparable.
python qualitative_retrieval.py \
  --model_family clip \
  --model_dir "saved_models/clip-fine-tune/fold_3_aug_square/8" \
  --checkpoint_name "best.pt" \
  --split_name "fold_3_aug_square" \
  --query_words_file "results/qualitative/query_words.txt" \
  --top_k 5 \
  --out_dir "results/qualitative_clip_run8"
