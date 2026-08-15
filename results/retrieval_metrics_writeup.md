# Retrieval metrics for all completed runs — results write-up

Run: `sbatch retrieval_metrics_report.sbatch` → SLURM job **47842**, completed in 1h18m
(`logs/retrieval_metrics/retrieval_metrics_47842.log`). Raw output:
`results/retrieval_metrics.csv` / `results/retrieval_metrics.md`.

## What was computed

For every completed fine-tuning run (ALIGN `1–5`, CLIP `1,2,4–8` — 12 runs, all
succeeded, no errors), on the same 384-image val-fold test set:

- **QbS (query-by-string, text→image)** — the main "word spotting" metric. Each
  test image's word is turned into a text query using **the same prompt convention
  that specific run was trained on** (plain word / PHOSC description / PHOSC
  number-description, read from that run's own `model_args.toml`), then ranked
  against all test images.
- **QbE (query-by-example, image→image)** — each test image ranked against all
  other test images. This metric already had scoring code in `matrix_new.py`
  (`compute_qbe_map_recall`) but was never actually wired into anything before now.

Both report mAP + Recall@1/5/10.

## Results

### ALIGN (`saved_models/align-fine-tune/fold_3_aug_square/`)

| run | description mode | QbS mAP | QbS R@1 | QbS R@5 | QbS R@10 | QbE mAP | QbE R@1 | QbE R@5 | QbE R@10 |
|----:|-------------------|--------:|--------:|--------:|---------:|--------:|--------:|--------:|---------:|
| 1 | word | **0.341** | 4.4% | 18.0% | 29.7% | 0.392 | 9.6% | 27.0% | 35.8% |
| 2 | word | 0.185 | 2.1% | 9.1% | 18.2% | 0.296 | 8.8% | 21.7% | 28.2% |
| 3 | phosc_number | 0.047 | 0.3% | 1.0% | 2.6% | 0.330 | 8.7% | 22.9% | 30.9% |
| **4** | **description** (best config) | 0.063 | 0.3% | 1.3% | 2.6% | **0.532** | 10.1% | 33.7% | 48.8% |
| 5 | description_long | 0.064 | 0.3% | 1.3% | 2.6% | 0.514 | 10.1% | 32.7% | 47.0% |

All 5 share identical hyperparameters otherwise (lr 1e-5, batch 8, accumulation 16 →
effective batch 128, Adam, cosine warmup, SupCon).

### CLIP (`saved_models/clip-fine-tune/fold_3_aug_square/`)

| run | loss | eff. batch | lr sched | QbS mAP | QbS R@1 | QbS R@5 | QbS R@10 | QbE mAP | QbE R@1 | QbE R@5 | QbE R@10 |
|----:|------|-----------:|----------|--------:|--------:|--------:|---------:|--------:|--------:|--------:|---------:|
| 1 | triplet | 64 | cosine | 0.046 | 0.3% | 1.6% | 2.3% | 0.178 | 6.0% | 13.9% | 18.0% |
| 2 | triplet | 64 | cosine | 0.044 | 0.3% | 1.0% | 2.9% | 0.039 | 0.2% | 1.1% | 2.4% |
| 4 | supcon | 52 | cosine | 0.047 | 0.5% | 2.1% | 2.3% | 0.057 | 0.7% | 3.4% | 5.6% |
| 5 | contrastive | 32 | cosine | 0.479 | 6.8% | 28.4% | 43.0% | 0.505 | 8.7% | 32.8% | 47.4% |
| 6 | contrastive | 32 | cosine | 0.526 | 7.0% | 29.4% | 49.5% | 0.544 | 8.7% | 35.4% | 52.5% |
| 7 | supcon | 64 | cosine_warmup | 0.044 | 0.3% | 1.3% | 2.6% | 0.068 | 1.4% | 4.4% | 7.2% |
| **8** | **supcon** | **64** | **cosine_warmup** | **0.614** | 6.5% | 32.8% | 56.8% | **0.788** | 10.3% | 47.3% | 74.5% |

All CLIP runs use `description=word` (no PHOSC-description CLIP configs exist yet).

## Key findings

**1. The coverage gap is closed.** Every completed ALIGN and CLIP run now has mAP +
Recall@1/5/10 for both QbS and QbE, reported consistently, backed by per-run
`matrix_{num}_report.txt` files next to each checkpoint.

**2. The single best retrieval model overall is CLIP run 8**, not any ALIGN
config — QbS mAP 0.614 / QbE mAP 0.789, clearly ahead of everything else in the
table. Worth foregrounding in the ALIGN-vs-CLIP comparison chapter.

**3. ALIGN's "best" config (run 4) is good at QbE but weak at QbS — a real gap,
not a measurement artifact.** Run 4 has the 2nd-highest QbE mAP among all 12 runs
(0.532: images cluster well by word identity), but its QbS mAP is only 0.063 —
*even though the text query was built in the exact PHOSC-description format run 4
was actually trained on* (this was a correctness fix I made to the evaluation
script; the old `matrix_new.py` always queried with the raw word regardless of
training format, which would have made this gap invisible). So the image encoder
learned a well-separated embedding space, but the text encoder isn't reliably
landing PHOSC-description queries near the right image. **This is precisely the
question the negative-control experiment (task 2, running as SLURM job 47844) is
designed to probe** — if the negative control's QbS collapses similarly while QbE
stays high, it doesn't distinguish "real" correspondence from just having a
separated space; if the real run 4 QbS is meaningfully above the shuffled-prompt
version, that's your evidence the model is doing more than chance.

**4. Within ALIGN, plain word-level prompts massively outperform PHOSC-description
prompts for QbS** (run 1: 0.341 vs runs 3/4/5: 0.047–0.064), while QbE tells the
opposite story (word-mode: 0.30–0.39, description-mode: 0.33–0.53). Runs 4/5 were
presumably picked as "best" on validation loss, not retrieval quality — this table
is the first place that tension becomes visible and is worth a sentence
reconciling the two selection criteria in the thesis.

**5. Large run-to-run variance between hyperparameter-identical configs.** CLIP
runs 7 and 8 have **byte-for-byte identical logged hyperparameters**
(supcon, Adam, cosine_warmup, effective batch 64, lr 1e-5) yet differ by
roughly **14×** in QbS mAP (0.044 vs 0.614) and QbE mAP (0.068 vs 0.788). ALIGN
runs 1 and 2 show the same pattern more mildly (word/word, otherwise identical,
0.341 vs 0.185 QbS mAP). This is worth flagging explicitly as a limitation —
either training is quite sensitive to something not captured in `model_args.toml`
(data shuffling order, augmentation sampling, initialization), or `best.pt` is
landing on very different points in training between nominal duplicates. Given
this spread, I'd treat any single run's number with some caution rather than as
a clean hyperparameter ablation.

## Methodology notes / caveats

- Test set is `fold_3_aug_square`'s val split, 399 images; the loader's
  `drop_last=True` with `--batch_size 32` keeps 384 of them. Same 384 for every
  run, so comparisons across runs are apples-to-apples, just not the full fold.
- QbS text queries are built per-run using that run's own training-time prompt
  convention (see finding 3) — a deliberate change from the original
  `matrix_new.py`, which always used the raw word.
- All CLIP runs are `description=word`; there's no PHOSC-description CLIP run to
  compare against ALIGN run 4 on equal footing.
- Encountered and fixed one real bug along the way: this `transformers` version
  changed `AlignModel.get_image_features()`/`get_text_features()` to return a
  `ModelOutput` instead of a plain tensor — handled in `retrieval_metrics_report.py`
  (`_unwrap_align_embeds`), doesn't affect training itself.
