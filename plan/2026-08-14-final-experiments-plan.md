# Final thesis experiments — implementation plan

Date: 2026-08-14
Source: professor's 4-item plan (retrieval metrics for all runs, negative control on
best ALIGN config, qualitative retrieval examples, embedding visualization).

## What already exists in the repo (context, not to be re-derived later)

- Two fine-tuned model families, both under `saved_models/<family>/fold_3_aug_square/<run#>/`:
  - `align-fine-tune/` — runs `1..5`, trained via `align_fine_tune.py`.
  - `clip-fine-tune/` — runs `1,2,4,5,6,7,8`, trained via `clip_fine_tune.py`.
  - Each run folder has `model_args.toml` (full hyperparams), `metrics.csv` (train/val
    loss per epoch), checkpoints as **raw `state_dict`** files (`best.pt`, `latest.pt`,
    `epoch N.pt` — saved via `torch.save(model.state_dict(), path)` in
    `utils/early_stopping.py`), plus an image–image cosine-similarity matrix/heatmap
    from `matrix.py`/`matrix_new.py`.
  - New run folders are auto-numbered (`max(existing) + 1`) by `EarlyStopping.initialize_save_path`.
- `matrix_new.py` already implements query-by-string (text→image) retrieval metrics —
  `compute_qbs_map_recall()` gives mAP + Recall@1/5/10 — and logs them to a per-run
  `matrix_{num}_report.txt`. It also defines `compute_qbe_map_recall()` (image→image)
  but **never calls it**. Crucially, this report has only ever been generated for the
  old/legacy run 0 (`saved_models/old/{align,clip}-fine-tune/fold_3_aug_square/0/`) —
  **none of the current runs (align 1–5, clip 1,2,4–8) have retrieval metrics computed
  yet**, and there is no consolidated table across runs. This is exactly the gap the
  professor flagged as most important.
- The **best ALIGN configuration** (PHOSC description + SupCon + Adam + cosine warmup +
  effective batch size 128) is `saved_models/align-fine-tune/fold_3_aug_square/4/`:
  `batch_size=8`, `accumulation_steps=16` (→ effective batch 128), `optimizer=adam`,
  `lr_scheduler=cosine_warmup`, `loss_func=supcon`, `description="description"`.
  Confirmed directly from its `model_args.toml`, not guessed.
- Prompts/descriptions are generated deterministically per word by
  `get_phosc_description(word)` (`modules/utils/utils.py`), and built fresh every batch
  inside `train_epoch`/`validate_epoch` in `align_fine_tune.py` (lines ~386–399 and
  ~479–493) as `descriptions = [get_phosc_description(w) for w in words]`.
- The test/val loader (`get_test_loader`, `phase='val'`) draws from a word vocabulary
  disjoint from the training split (standard compositional-split fold), so it's the
  natural source of "unseen query words" for the qualitative section.
- No `scikit-learn` / `umap-learn` found in the local `align4/` venv (that venv looks
  unused/stale — no `torch` either — this project actually runs via SLURM `.sbatch`
  scripts on the cluster's real environment, not this local venv). Plan assumes
  `scikit-learn` needs to be confirmed/installed in the real training env before running
  item 4; `t-SNE` was chosen over UMAP specifically to avoid a new dependency.

## Decisions already confirmed with you

1. Negative control: **per-batch random shuffle** of the description list (not a fixed
   dataset-wide permutation).
2. Qualitative query words: **random sample of unseen words, fixed seed** (not hand-picked).
3. Embedding visualization: **t-SNE via scikit-learn**.
4. All new outputs go to a new top-level **`results/`** directory, separate from
   `saved_models/`.

## Planned changes

### 1. Retrieval metrics for all completed runs (highest priority)

New script: `retrieval_metrics_report.py` (repo root, follows the existing `matrix_new.py`
pattern so it can reuse `collect_image_features_align/clip`, `collect_text_features_align/clip`,
`compute_qbs_map_recall`, `compute_qbe_map_recall`).

- Loops over a config list: `align-fine-tune` runs `[1,2,3,4,5]` and `clip-fine-tune`
  runs `[1,2,4,5,6,7,8]` (skips missing/broken runs gracefully with a warning instead of
  crashing the whole batch).
- For each run: loads `best.pt`, rebuilds the test loader once per model family (reused
  across runs of that family to save time), computes **both** QbS (text→image) and QbE
  (image→image) mAP + Recall@1/5/10 — wiring in the already-written but unused
  `compute_qbe_map_recall`.
- Reads each run's `model_args.toml` to pull the hyperparameters that matter for the
  comparison table (description mode, loss function, optimizer, lr scheduler, effective
  batch size = `batch_size * accumulation_steps`).
- Writes one consolidated `results/retrieval_metrics.csv` (one row per run, both model
  families) and a companion `results/retrieval_metrics.md` (a ready-to-paste Markdown
  table for the thesis).
- Keeps writing the existing per-run `matrix_{num}_report.txt` too, so nothing already
  relied upon changes shape.
- New `retrieval_metrics_report.sbatch` mirroring `matrix_new.sbatch`'s resource
  footprint (CPU-only is likely fine since it's inference-only over saved checkpoints;
  will request 1 GPU anyway since ALIGN/CLIP forward passes are much faster on GPU).

### 2. Negative-control experiment on the best ALIGN config

- Add `--shuffle_descriptions` (store_true) to `parser/training_common_argparse.py`.
- In `align_fine_tune.py`, in both `train_epoch` and `validate_epoch`, right after
  `descriptions = [get_phosc_description(w) for w in words]` (and the `description_long`
  branch, for consistency): if `args.shuffle_descriptions`, apply a random permutation to
  the `descriptions` list only (images/words stay in their original order), breaking the
  image↔prompt correspondence while keeping the exact same distribution of real PHOSC
  description strings. This isolates "does the model use correspondence" from "did we
  just feed it garbage text."
- New `train_align_negative_control.sbatch`: clones `train_align.sbatch` but pins the
  exact best-config hyperparameters read from run 4's `model_args.toml`
  (`--batch_size 8 --accumulation_steps 16 --optimizer adam --lr_scheduler cosine_warmup
  --loss_func supcon --description description --augmented`) plus the new
  `--shuffle_descriptions` flag. Since numbering is automatic, this lands in
  `saved_models/align-fine-tune/fold_3_aug_square/6/`.
- After training, run `retrieval_metrics_report.py` against run 6 (same script as item 1,
  just add `6` to the align run list) so the comparison against run 4 is apples-to-apples
  and lands in the same `results/retrieval_metrics.csv`.

### 3. Qualitative retrieval examples from the best ALIGN model

New script: `qualitative_retrieval.py`.

- Loads `saved_models/align-fine-tune/fold_3_aug_square/4/best.pt`.
- Embeds all test/val-fold images once (reusing `collect_image_features_align`).
- Randomly samples 10 unseen words from the test/val vocabulary (fixed seed, e.g. 42) —
  falls back to fewer if the fold has <10 distinct words.
- For each query word: builds its PHOSC description with `get_phosc_description` (same
  function used at train time, so the query matches training-time prompt format),
  encodes it as text, ranks all test images by cosine similarity, takes top-5.
- Saves, per query, a small figure (query word + its PHOSC description text alongside
  top-5 retrieved image thumbnails, correct/incorrect marked by border color) to
  `results/qualitative/<word>.png`, plus one `results/qualitative/summary.csv` logging
  hit@1/5 and the rank of the first correct match per query — giving the discussion
  section something concrete (which of the 10 succeeded vs failed, and why, is easy to
  eyeball from the CSV + images together).

### 4. Embedding visualization (frozen vs fine-tuned ALIGN) — lower priority

New script: `embedding_visualization.py`.

- Extracts image embeddings for the test set from (a) frozen pretrained
  `kakaobrain/align-base` and (b) the fine-tuned best ALIGN checkpoint (run 4).
- Projects both to 2D with `sklearn.manifold.TSNE` (same fixed seed/perplexity for both,
  fit independently since the two embedding spaces aren't directly comparable).
- Saves a single side-by-side figure `results/embeddings/tsne_frozen_vs_finetuned.png`,
  points colored by word (or by first-character/class if the vocabulary is too large for
  a legend to be useful — will check unique word count in the test fold before deciding).
- Explicitly scoped as best-effort/last: only tackled once items 1–3 are done and
  working, per the professor's own ranking.

## Execution order

1. Item 1 (retrieval metrics for existing runs) — no training needed, just run inference
   over already-saved checkpoints. Fastest to validate the plan is on track.
2. Item 3 (qualitative examples) — also no training needed, reuses run 4's checkpoint.
3. Item 2 (negative control) — needs a full training run on the cluster (submitted via
   `sbatch`), so it's the slow step; kick it off early once the code change is written,
   let it run in the background while other items proceed.
4. Item 4 (visualization) — last, only after confirming `scikit-learn` is available in
   the real training environment.

## Open items to flag as I go (not blocking, but worth knowing)

- `matrix_new.py`'s CLIP path uses `import clip` (OpenAI's CLIP package) and
  `clip.tokenize` — I haven't verified this package is installed in the actual cluster
  training environment; `retrieval_metrics_report.py` will surface a clear error per-run
  rather than silently skipping if this import fails, and I'll fix forward if that error
  shows up on your first `sbatch` run of item 1.
- Run `3` doesn't exist under `clip-fine-tune/` (only `1,2,4,5,6,7,8` do) — the
  retrieval script will just skip missing run numbers rather than error.
