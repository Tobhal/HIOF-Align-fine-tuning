# Qualitative retrieval examples — results write-up

Script: `qualitative_retrieval.py`. 10 unseen query words sampled with a fixed seed
(42) from the val-fold test set (384 images, 40 unique words) shared by every
ALIGN run. Each word is queried with its own real PHOSC description text (the
same prompt format that specific checkpoint was trained on), top-5 retrieved by
cosine similarity. Outputs: `results/qualitative/` (run 4, the best ALIGN
config) and `results/qualitative_run1_word_mode/` (run 1, added as a
comparison — see below), each with one PNG per query plus a `summary.csv`.

## Headline result: run 4 fails completely on unseen words — and always fails the same way

Run 4 (PHOSC description + SupCon + Adam + cosine warmup + effective batch 128 —
the config picked as "best ALIGN") scores **0/10 on hit@1 and 0/10 on hit@5**.
That alone would just be "poor recall," unremarkable next to task 1's already-low
QbS mAP (0.063). What's striking is *how* it fails: **all 10 queries retrieve the
exact same single word — কুচনী — in all 5 top slots**, regardless of what was
actually asked for.

| query word | hit@1 | hit@5 | rank of first correct hit | top-1 retrieved |
|---|---|---|---:|---|
| কাপাসদা | ✗ | ✗ | 274 | কুচনী |
| আনুলিয়া | ✗ | ✗ | 72 | কুচনী |
| খারিদুনগরী | ✗ | ✗ | 95 | কুচনী |
| কৈলাশপুর | ✗ | ✗ | 235 | কুচনী |
| কেশবচক | ✗ | ✗ | 181 | কুচনী |
| কালীঘাট | ✗ | ✗ | 170 | কুচনী |
| কাঁথি | ✗ | ✗ | 11 | কুচনী |
| কল্যানচাক | ✗ | ✗ | 308 | কুচনী |
| জধুবেড়িয়া | ✗ | ✗ | 26 | কুচনী |
| দেপাল | ✗ | ✗ | 6 | কুচনী |

(Full table with all top-5 words + similarity scores: `results/qualitative/summary.csv`.)

## Diagnosis: the text tower has collapsed for unseen PHOSC descriptions

I checked whether this is a bug in the eval script or a real property of the
model, by encoding the 10 different queries' PHOSC descriptions and measuring
pairwise cosine similarity between them directly (bypassing images entirely):

```
off-diagonal pairwise text-text cosine similarity across the 10 (genuinely
different) query descriptions: min=0.984, mean=0.993, max=0.9997
```

Ten different unseen words, with visibly different PHOS/PHOC content in their
descriptions (different shape counts, different character sequences — see the
`description` column in `summary.csv`), produce text embeddings that are almost
indistinguishable from each other (cosine ≥0.98). Once the text encoder maps
every unseen description to essentially the same point in embedding space, the
top-5 ranking is no longer answering "which image matches this word" — it's
just "which single image cluster happens to sit closest to that generic region,"
which is exactly the all-কুচনী pattern above. This is consistent with, and
explains, task 1's ALIGN result: run 4 has strong QbE (mAP 0.532 — images
separate well by class) but weak QbS (mAP 0.063 — text queries barely
discriminate between classes).

## Comparison: the same 10 queries against run 1 (word-mode) look completely different

Run 1 (identical hyperparameters, `description=word` instead of
`description=description`) was run through the exact same script, same seed, same
10 words (guaranteed identical since it's the same test set) for a clean
apples-to-apples comparison:

| query word | run 4 (description) hit@1 / hit@5 | run 1 (word) hit@1 / hit@5 | run 1 rank of first hit |
|---|---|---|---:|
| কাপাসদা | ✗ / ✗ | ✗ / ✗ | 10 |
| আনুলিয়া | ✗ / ✗ | **✓ / ✓** | 1 |
| খারিদুনগরী | ✗ / ✗ | **✓ / ✓** | 1 |
| কৈলাশপুর | ✗ / ✗ | ✗ / ✓ | 2 |
| কেশবচক | ✗ / ✗ | ✗ / ✓ | 4 |
| কালীঘাট | ✗ / ✗ | ✗ / ✗ | 24 |
| কাঁথি | ✗ / ✗ | ✗ / ✓ | 2 |
| কল্যানচাক | ✗ / ✗ | ✗ / ✓ | 3 |
| জধুবেড়িয়া | ✗ / ✗ | **✓ / ✓** | 1 |
| দেপাল | ✗ / ✗ | **✓ / ✓** | 1 |
| **totals** | **0/10, 0/10** | **4/10 (40%), 8/10 (80%)** | |

Run 1 is far from perfect (2 of 10 still miss the top-5 entirely), but it clearly
*is* discriminating between unseen words — most rank the correct image at 1–4,
a completely different regime from run 4's 6–308. This makes a strong case that
the collapse above is specific to the PHOSC-description text tower for this
particular checkpoint, not a ceiling on what ALIGN can do on this task.

## What this means for the thesis

- **For Chapter 4**: this pair of comparisons (0/10 vs 4/10 hit@1 on identical
  queries) is a concrete, visual "success vs. failure" story — probably more
  useful than run 4's numbers alone, since it shows word-mode prompts working
  reasonably on genuinely unseen words while description-mode prompts don't,
  on the exact same images.
- **For the negative-control experiment (task 2, in progress as job 47844)**:
  this raises the stakes on interpreting it. If a *correctly-trained* run 4
  already can't meaningfully separate unseen description prompts in text space,
  the negative control (shuffled prompts) needs to be read carefully — a
  collapsed QbS in both the real and shuffled versions wouldn't be strong
  evidence either way, since real run 4 may already be close to a text-embedding
  floor for unseen words. Worth explicitly comparing run 4's QbS mAP (0.063)
  against whatever the negative-control run scores, rather than only checking
  "did it collapse further."
- Given finding 5 from the task-1 write-up (large variance between nominally
  identical configs), it's worth a sentence checking whether this collapse is
  specific to run 4's particular checkpoint or a systematic property of
  description-mode training — a second description-mode run (run 5,
  `description_long`, same QbS/QbE pattern as run 4 in task 1's table) would be
  the cheapest way to check, if useful for the thesis.

## Files

- `results/qualitative/00_..09_*.png` + `results/qualitative/summary.csv` — run 4
  (best ALIGN config), 10 queries, full top-5 + similarity scores.
- `results/qualitative_run1_word_mode/00_..09_*.png` + `.../summary.csv` — run 1
  (word-mode), same 10 queries, for the comparison above.
- Each PNG shows the query word + (truncated) description text on the left, and
  the top-5 retrieved images with green/red borders for correct/incorrect word
  match, similarity score, and retrieved word underneath each thumbnail.
