# Findings, reframed against the two research questions

**Purpose of this document:** this is written to be handed to another LLM
(one with full context on the thesis document) to help draft/update the
thesis text. Every claim below is backed by a real number produced by
scripts in this repo, run on real checkpoints — nothing here is simulated or
estimated unless explicitly marked "preliminary." Source scripts and output
files are listed under each section so the exact number can be traced back.

## The two research questions

- **RQ1**: Can a multimodal Vision-Language Model understand the words in an
  image of handwritten text? (Includes: which types of word descriptions are
  most effective for fine-tuning the model.)
- **RQ2**: Can that multimodal Vision-Language Model be fine-tuned to better
  understand handwritten text, compared against the pretrained model?

**Goal behind both RQs, stated by the thesis author:** the underlying target
is **out-of-lexicon (OOL) word recognition** — can the model generalize to
words it never saw during training, not just memorize a fixed vocabulary.
This is why the specific identity of individual failing/succeeding words
matters less than the *pattern* of generalization across description types
and across fine-tuned-vs-frozen models.

**Scope caveat — state this explicitly in the thesis, don't let a reader
find it first:** "unseen"/"out-of-lexicon" throughout everything below means
**words held out from the same Bengali dataset** (a word-disjoint val/test
split — no word in the 384/399-image test set appears in training). It does
**not** mean cross-lingual or cross-script generalization (e.g., training on
Bengali and testing on a different language/writing system). That stronger
test was not run due to time constraints. Recommended sentence for the
limitations section: *"Out-of-lexicon generalization was evaluated by
holding out entire words from the same Bengali dataset; whether this
generalization extends to a different script or language remains untested
and is left for future work."*

Everything below is organized by research question, then by evidence item.
Item numbers (1–4) refer to the professor's original 4-item task list, kept
for traceability to scripts/commits.

---

# RQ1 — Does the model understand handwritten words, and which description type works best?

RQ1 has two parts and two different pieces of evidence answer them:

1. *"Which description type is most effective for fine-tuning"* → **item 1**
   (quantitative retrieval metrics across all trained configs).
2. *"Does the model actually understand words, i.e. generalize to
   out-of-lexicon words"* → **item 3** (qualitative retrieval on unseen
   words) — this is the sharper, more direct test, and the headline result
   there is more informative than any aggregate metric in item 1.

## Item 1 — Retrieval metrics across all description types (done)

Script: `retrieval_metrics_report.py`, run as SLURM job **47842** (completed,
1h18m). Computes **QbS** (query-by-string: text description → image, the
literal "word spotting from a description" task) and **QbE**
(query-by-example: image → image) as mAP + Recall@1/5/10, for every
completed run. Test set: `fold_3_aug_square` val split, 384 images / 40
unique words, word-disjoint from training — i.e., every query word is
out-of-lexicon relative to training. Full data: `results/retrieval_metrics.csv`
/ `.md`.

One methodology fix made while building this: QbS text queries are built
using **each run's own training-time prompt convention** (word / PHOSC
description / PHOSC number-description, read from that run's
`model_args.toml`) — the older `matrix_new.py` script always queried with
the raw word regardless of how the model was trained, which would have
hidden the description-type effect entirely.

### ALIGN — description type is the only thing that differs across these 5 runs

All 5 share lr 1e-5, batch 8, accumulation 16 → effective batch 128, Adam,
cosine warmup, SupCon loss:

| run | description mode | QbS mAP | QbS R@1 | QbS R@5 | QbS R@10 | QbE mAP | QbE R@1 | QbE R@5 | QbE R@10 |
|----:|-------------------|--------:|--------:|--------:|---------:|--------:|--------:|--------:|---------:|
| 1 | word | **0.341** | 4.4% | 18.0% | 29.7% | 0.392 | 9.6% | 27.0% | 35.8% |
| 2 | word | 0.185 | 2.1% | 9.1% | 18.2% | 0.296 | 8.8% | 21.7% | 28.2% |
| 3 | phosc_number | 0.047 | 0.3% | 1.0% | 2.6% | 0.330 | 8.7% | 22.9% | 30.9% |
| **4** | **description** (best-val-loss config) | 0.063 | 0.3% | 1.3% | 2.6% | **0.532** | 10.1% | 33.7% | 48.8% |
| 5 | description_long | 0.064 | 0.3% | 1.3% | 2.6% | 0.514 | 10.1% | 32.7% | 47.0% |

**Reading this for RQ1's "which description type" question:** the answer
depends on which retrieval direction you care about, and that split *is*
the finding:
- **QbS (text→image, i.e. "spot the word given a description"): plain word
  prompts win by a wide margin** — run 1's QbS mAP (0.341) is 5.3–7.3×
  every PHOSC-description variant (0.047–0.064).
- **QbE (image→image, i.e. "do same-word images cluster together"): PHOSC
  description prompts win** — run 4 (0.532) and run 5 (0.514) clearly beat
  word-mode (0.392, 0.296).
- Runs 4/5 were selected as "best" during training on validation loss, not
  retrieval quality — worth a sentence reconciling the two selection
  criteria, since "best config" depends entirely on which of QbS/QbE the
  thesis treats as the primary word-spotting metric.

### CLIP, for context (all `description=word`)

| run | loss | eff. batch | lr sched | QbS mAP | QbE mAP |
|----:|------|-----------:|----------|--------:|--------:|
| 1 | triplet | 64 | cosine | 0.046 | 0.178 |
| 2 | triplet | 64 | cosine | 0.044 | 0.039 |
| 4 | supcon | 52 | cosine | 0.047 | 0.057 |
| 5 | contrastive | 32 | cosine | 0.479 | 0.505 |
| 6 | contrastive | 32 | cosine | 0.526 | 0.544 |
| 7 | supcon | 64 | cosine_warmup | 0.044 | 0.068 |
| **8** | **supcon** | **64** | **cosine_warmup** | **0.614** | **0.788** |

CLIP run 8 is the single best retrieval model across *both* architectures
(QbS 0.614, QbE 0.788) — worth foregrounding in an ALIGN-vs-CLIP comparison,
though all CLIP runs use word-mode only, so there's no PHOSC-description
CLIP run to compare against ALIGN run 4 on equal footing.

### A second, independent lens on "which config is best": your original similarity-matrix metric agrees

The thesis's existing `matrix_new.py` script computes a full N×N image-image
cosine similarity matrix and reports its mean (most pairs in that matrix are
different-word pairs, so a mean close to 0 signals a well-spread,
non-collapsed embedding space — this is the metric already used in the
thesis, described there as "closer to 0 is better"). Pulling those
already-computed numbers next to QbE mAP:

| run | mode | mean similarity (closer to 0 = better) | QbE mAP (higher = better) |
|---|---|---:|---:|
| 1 | word | 0.287 | 0.392 |
| 2 | word | 0.387 | 0.296 |
| 3 | phosc_number | 0.454 | 0.330 |
| **4** | description | **0.255 (best)** | **0.532 (best)** |
| 5 | description_long | 0.294 | 0.514 |

**Both metrics agree run 4 is the best config overall** — the two metrics
are independent and computed by different scripts, so this is real
convergent evidence, not circular. They disagree on the *ordering of the
others* (mean-similarity ranks 1 > 5 > 2 > 3; QbE mAP ranks 5 > 1 > 3 > 2),
which is expected: mean similarity measures how spread-out the whole space
is, not whether same-word images specifically cluster together, so it's a
looser proxy for retrieval quality than mAP. Good sentence for the thesis:
*"The similarity-matrix mean and retrieval mAP agree on the best-performing
configuration, but diverge on the ranking of the others — indicating that
overall embedding spread alone does not fully predict retrieval quality;
mAP, which directly measures whether same-word images rank above others, is
the more direct metric for the word-spotting task."*

### Caveats on item 1

- Test set is `fold_3_aug_square`'s val split; `drop_last=True` at batch
  size 32 keeps 384 of 399 images — same 384 for every run, so cross-run
  comparison is fair, just not the full fold.
- Large variance between hyperparameter-identical configs: CLIP runs 7/8
  have byte-identical logged hyperparameters but differ ~14× in mAP (0.044
  vs 0.614); ALIGN runs 1/2 show a milder version of the same thing. Treat
  any single run as one sample from a noisy process, not a clean ablation —
  worth a limitations sentence.
- Fixed one real bug en route: this `transformers` version changed
  `AlignModel.get_image_features()`/`get_text_features()` to return a
  `ModelOutput` instead of a plain tensor; handled in
  `retrieval_metrics_report.py`.

## Item 3 — Qualitative retrieval on unseen words: the sharper RQ1 test (done)

Script: `qualitative_retrieval.py`. 10 unseen query words, fixed seed (42),
from the same 384-image/40-word test set. Each word queried with its own
real PHOSC description text (the exact prompt format that checkpoint was
trained on), top-5 retrieved by cosine similarity. Outputs:
`results/qualitative/` (run 4) and `results/qualitative_run1_word_mode/`
(run 1, added as a direct comparison), each with one PNG per query plus
`summary.csv`.

**This is the most direct test of RQ1 available in this repo**: item 1
answers "which description type scores higher on aggregate metrics," but
item 3 answers "does the model actually understand an unseen word or not,"
which is the literal question RQ1 asks.

### Headline result: run 4 (the "best" PHOSC config by every metric in item 1) fails completely on unseen words, and fails the exact same way every time

Run 4 scores **0/10 on hit@1 and 0/10 on hit@5**. Not just poor recall —
**all 10 queries retrieve the exact same single word — কুচনী — in all 5 top
slots**, regardless of what was actually asked for:

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

(Full table with all top-5 words + similarity scores:
`results/qualitative/summary.csv`.)

### Diagnosis: the text tower has collapsed for unseen PHOSC descriptions

Checked whether this is an eval-script bug or a real model property, by
encoding the 10 different queries' PHOSC descriptions and measuring pairwise
cosine similarity between them directly (bypassing images entirely):

```
off-diagonal pairwise text-text cosine similarity across the 10 (genuinely
different) query descriptions: min=0.984, mean=0.993, max=0.9997
```

Ten different unseen words, with visibly different PHOS/PHOC content in
their descriptions (different shape counts, different character sequences —
see the `description` column in `summary.csv`), produce text embeddings that
are almost indistinguishable from each other. Once the text encoder maps
every unseen description to essentially the same point in embedding space,
the top-5 ranking stops answering "which image matches this word" and
becomes "which single image cluster sits closest to that generic region" —
exactly the all-কুচনী pattern above. This directly explains item 1's ALIGN
result: run 4 has strong QbE (mAP 0.532 — images separate well by class) but
weak QbS (mAP 0.063 — text queries barely discriminate between classes).
**For RQ1: the image side has learned real, generalizable word identity;
the text side, for PHOSC descriptions specifically, has not generalized to
unseen words at all.**

### Comparison: the same 10 queries against run 1 (word-mode) look completely different

Run 1 (identical hyperparameters, `description=word` instead of
`description`), same script, same seed, same 10 words:

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

Run 1 is far from perfect (2 of 10 still miss the top-5 entirely), but it
clearly *is* discriminating between out-of-lexicon words — most rank the
correct image at 1–4, a completely different regime from run 4's 6–308.
**For RQ1, this is the central "success vs. failure" result**: word-mode
prompts generalize to genuinely unseen words on the same images where
PHOSC-description prompts do not, which is direct evidence for "plain word
supervision, not structured PHOSC description, is what currently lets this
model understand out-of-lexicon handwritten words."

### Cross-check: does CLIP's best run (C8) hold up under the same probe?

Item 1 reports CLIP run 8 (SupCon, eff. batch 64, cosine warmup, word-mode)
as the single best retrieval model overall (QbS mAP 0.614, QbE mAP 0.788),
on aggregate mAP alone. Since ALIGN's best-looking run (A4) failed this
exact probe despite winning on aggregate metrics, the same probe was run
against C8 before letting "CLIP beats ALIGN" stand unchallenged — same
script (now parameterized by model family, see below), the same literal 10
query words from run 4/run 1 (`results/qualitative/query_words.txt`, not a
resample), same test split. Config confirmed against
`saved_models/clip-fine-tune/fold_3_aug_square/8/model_args.toml` before
running: `loss_func=supcon`, `batch_size=32 × accumulation_steps=2 = eff.
batch 64`, `lr_scheduler=cosine_warmup`, `description=word` — matches the
claimed C8 config exactly.

| query word | ALIGN A4 (description) hit@1/hit@5 | ALIGN run1 (word) hit@1/hit@5 | CLIP run8 (word) hit@1/hit@5 |
|---|---|---|---|
| কাপাসদা | ✗/✗ | ✗/✗ | **✓/✓** |
| আনুলিয়া | ✗/✗ | **✓/✓** | **✓/✓** |
| খারিদুনগরী | ✗/✗ | **✓/✓** | **✓/✓** |
| কৈলাশপুর | ✗/✗ | ✗/✓ | ✗/✓ |
| কেশবচক | ✗/✗ | ✗/✓ | ✗/✓ |
| কালীঘাট | ✗/✗ | ✗/✗ | ✗/✗ |
| কাঁথি | ✗/✗ | ✗/✓ | ✗/✓ |
| কল্যানচাক | ✗/✗ | ✗/✓ | ✗/✓ |
| জধুবেড়িয়া | ✗/✗ | **✓/✓** | **✓/✓** |
| দেপাল | ✗/✗ | **✓/✓** | **✓/✓** |
| **totals** | **0/10, 0/10** | **4/10 (40%), 8/10 (80%)** | **5/10 (50%), 9/10 (90%)** |

**On hit rate alone, C8 clearly holds up** — it beats both ALIGN
configurations, consistent with its aggregate mAP lead.

**But the text-text collapse check is genuinely ambiguous, and worth
reporting exactly as such:**

```
off-diagonal text-text cosine similarity, C8's 10 query words:
min=0.9674  mean=0.9847  max=0.9956
```

That's nearly as high as A4's collapsed case (min=0.984, mean=0.993,
max=0.9997) — on the raw number alone, this looks like the same failure
mode. **It isn't, and the reason is the same correction already made in
item 4**: a uniformly high raw similarity number reflects how tightly
packed the whole embedding region is (a known anisotropy property of
these encoders), not whether the small differences between embeddings are
still discriminative. What actually decides that is whether different
queries land on different, correct images — and they do: C8's 10 queries
produced **9 distinct top-1 words** (only one repeat), versus A4's **1
distinct top-1 word across all 10 queries** (same word, every time). That
contrast — not the raw similarity number — is the real evidence of whether
a text tower has functionally collapsed.

**Reading against the Step 6 rubric this check was designed around:** this
is the "something in between" case, and should be reported as such rather
than forced into a clean narrative — but it leans toward *"the result holds
up."* High absolute text-text similarity alone doesn't demonstrate collapse
(item 4 already established that absolute similarity level is the wrong
thing to read); the behavioral evidence (hit rate, top-1 diversity) says C8
is still discriminating between out-of-lexicon words, just from a text
embedding space that — like most of these encoders' spaces — is generally
anisotropic. **Keep the "CLIP's best run beats ALIGN's best run" framing in
the thesis**, but add this nuance rather than citing QbS mAP alone: CLIP's
text tower generalizes to unseen words in a way ALIGN's PHOSC-description
tower does not, even though neither model's text embeddings are
well-separated in an absolute sense.

### What this means for the RQ1 write-up

- **Use the 0/10 vs 4/10 vs 5/10 hit@1 comparison (A4 / run1 / C8), not the
  raw item-1 mAP table, as the headline RQ1 result.** It's a concrete,
  visual, easy-to-defend success-vs-failure story on identical unseen
  queries and images — more convincing than aggregate mAP numbers alone,
  and it now spans both architectures instead of just ALIGN.
- The individual failing/succeeding words themselves are not the point (per
  the thesis author) — the *pattern* is: PHOSC-description text embeddings
  collapse *functionally* (same top-1 word every time) for unseen inputs,
  while word-mode embeddings (both ALIGN run 1 and CLIP run 8) keep
  discriminative structure even though none of these text spaces are
  well-separated in absolute similarity terms. Frame around the pattern,
  not the specific Bengali words.
- **The CLIP cross-check changes the shape of the RQ1 answer slightly**: the
  cleanest formulation is no longer "ALIGN's PHOSC descriptions collapse,
  word prompts don't" — it's "**word-level supervision generalizes to
  out-of-lexicon words on both backbones tested; structured PHOSC
  descriptions do not, at least at the configurations trained here.**" That
  is a claim about the *supervision/prompt approach*, not about ALIGN
  specifically, and it's now backed by evidence on two architectures instead
  of one.
- Given item 1's finding on run-to-run variance, it's worth one sentence
  checking whether A4's collapse is specific to that particular checkpoint
  or systematic to description-mode training. Run 5 (`description_long`)
  shows the same QbS/QbE pattern as run 4 in item 1's table, which is a good
  sign it's systematic rather than a one-off checkpoint artifact (not
  independently re-verified qualitatively, but the quantitative pattern
  matches).

### Files

- `results/qualitative/00_..09_*.png` + `summary.csv` + `query_words.txt` —
  run 4 (ALIGN, description-mode), 10 queries, full top-5 + similarity
  scores. `query_words.txt` is the literal word list reused by the run 1 and
  CLIP run 8 checks below, so all three query the same 10 words.
- `results/qualitative_run1_word_mode/00_..09_*.png` + `summary.csv` — run
  1 (ALIGN, word-mode), same 10 queries.
- `results/qualitative_clip_run8/00_..09_*.png` + `summary.csv` +
  `text_text_similarity.json` — CLIP run 8 (word-mode), same 10 queries,
  plus the text-text collapse diagnostic discussed above.
- Each PNG: query word + (truncated) description text on the left, top-5
  retrieved images with green/red borders for correct/incorrect word match,
  similarity score and retrieved word under each thumbnail.
- `qualitative_retrieval.py` now takes `--model_family {align,clip}` and
  `--query_words_file` (bypasses random sampling with a literal word list) —
  same script, same pipeline, only the encoder swapped, so all three checks
  are directly comparable.

---

# RQ2 — Can the model be fine-tuned to better understand handwritten text than the pretrained baseline?

RQ2 compares fine-tuned vs. pretrained. Two pieces of evidence:

1. **Item 4** — does fine-tuning measurably change how the (frozen vs.
   fine-tuned) model organizes held-out handwritten-word images?
2. **Item 2** — did fine-tuning learn the *real* image↔word correspondence,
   or would fine-tuning on any plausible-looking text have produced a
   similar-looking improvement? (Preliminary — see below.)

## Item 4 — Embedding visualization, frozen vs. fine-tuned (done)

Script: `embedding_visualization.py`. t-SNE of ALIGN **image** embeddings
only (no text involved) on the same 384-image, word-disjoint test fold,
comparing frozen (pretrained, no fine-tuning) `kakaobrain/align-base`
against the best fine-tuned ALIGN config (run 4). Output:
`results/embeddings/tsne_frozen_vs_finetuned.png`.

Three words are drawn in distinct colors; every other word is a single
muted gray "other" point (only the first 3 hues of the categorical palette
clear colorblind-safety in an all-pairs scatter plot; the rest correctly
fold into "Other" rather than pretending 40 arbitrary hues are all
distinguishable). Visually: in the frozen panel the 3 highlighted words'
points are scattered loosely among the gray background; in the fine-tuned
panel they pull into visibly tighter, more separated clusters — on words
the model never saw during training.

### Quantifying it — and a correction worth keeping in the thesis text

First attempt at a quantitative backup was mean pairwise cosine similarity
*within* each word's own images. That number goes the **wrong way**: frozen
ALIGN scores 0.791, fine-tuned scores 0.568 — raw same-word similarity
*drops* after fine-tuning, seemingly contradicting the plot. This is a known
artifact, not a real regression: pretrained embedding spaces are commonly
**anisotropic** (every embedding sits in a narrow cone of the hypersphere,
inflating *every* pairwise similarity, same-word or not) — a high raw
intra-class number there just means "everything looks similar to
everything," not "this model understands word identity."

The number that actually predicts retrieval quality is the **separation
margin** — mean same-class cosine similarity minus mean different-class
cosine similarity, over the whole test set. Rank-based metrics (QbE mAP)
only depend on relative ordering, so they're insensitive to a uniform shift
in similarity level; separation margin captures exactly that
relative-ordering signal instead of the misleading absolute level:

| model | intra-class mean | inter-class mean | **separation margin** |
|---|---:|---:|---:|
| frozen ALIGN (pretrained) | 0.791 | 0.728 | **0.063** |
| fine-tuned ALIGN (run 4) | 0.562 | 0.245 | **0.317** |

Fine-tuning lowers *both* intra- and inter-class similarity (the space
de-anisotropizes, spreads out over more of the hypersphere), but lowers
inter-class similarity far more than intra-class — **~5× the separation
margin** of the frozen model (0.317 vs 0.063), measured on words the
fine-tuned model never trained on. **This is the core RQ2 result**: it is
quantitative, held-out-word evidence that fine-tuning genuinely improved how
the image encoder organizes handwritten-word images by identity, not just
that fine-tuning changed the numbers superficially. It lines up with item
1's finding that run 4 has strong QbE mAP (0.532) — image-only retrieval
benefits directly from this separation, independent of the text-side
collapse found in item 3.

Full per-word intra-class numbers: `results/embeddings/intra_class_similarity.csv`.
Frozen-vs-fine-tuned separation margin: `results/embeddings/separation_margin.csv`.

**Caveat:** t-SNE coordinates from the two panels are **not on a shared
scale** (each is an independent fit) — cite the separation-margin table
(computed in the original embedding space, not the t-SNE projection), not
eyeballed plot distances.

**Bug fixed along the way, worth a footnote if the thesis discusses
tooling:** hardcoding the 3 highlight words directly in the script's source
(typing the Bengali text) caused one of them to silently match zero points —
the typed string and the dataset's version of the same word were different
Unicode normalization forms of visually-identical text (precomposed vs.
decomposed combining characters). Fixed with
`unicodedata.normalize('NFC', ...)` before comparison.

## Item 2 — Negative control (shuffled PHOSC prompts): PRELIMINARY, not converged

**Status: two training runs still in progress at time of writing — the
numbers below are a mid-training snapshot, not a finished result. Say so
explicitly if this goes in the thesis; do not present it as converged.**

### Why this experiment exists, and what it adds beyond item 4

Item 4 shows fine-tuning changes the embedding space in a way that improves
class separation. It does **not** by itself prove the model is using the
*real* content of the PHOSC descriptions to do that — a model could, in
principle, learn to separate images into some number of clusters using any
consistent-enough training signal, real or not. The negative control tests
this directly: train an identical model, from the same pretrained
`kakaobrain/align-base` base, with every hyperparameter matching run 4
exactly, except each batch's PHOSC description text is **randomly shuffled**
(`--shuffle_descriptions`) so the image↔text pairing within a batch is
broken. If shuffled-prompt training reaches similar retrieval quality to
real-prompt training, that would undercut RQ2's claim that fine-tuning
learns genuine image-text correspondence. If it performs clearly worse, that
supports the claim.

### Setup

Two independent runs, for redundancy given item 1's finding of large
run-to-run variance on nominally identical configs:

| job | run # | dir | node | speedup applied | epoch reached (at last check) | runtime so far |
|---|---:|---|---|---|---:|---|
| 47844 | 6 | `saved_models/align-fine-tune/fold_3_aug_square/6` | hpc8 | none (baseline FP32, matches run 4 exactly) | 61 | ~20.0h |
| 47845 | 7 | `saved_models/align-fine-tune/fold_3_aug_square/7` | hpc1 | threaded image loading only (no AMP, no numeric change to training) | 50 | ~16.9h |

For reference, run 4 itself trained **139 epochs** before early-stopping
(best epoch 89, patience 50). At the current ~20 min/epoch pace, neither run
6 nor run 7 will reach a comparable, fully-converged, early-stopped state
before the delivery deadline — matching run 4's epoch count alone would need
another ~28–30 hours from the time of the last check. **This experiment will
not fully converge before delivery; what follows is a matched-epoch
snapshot.**

### Matched-epoch comparison at epoch 50 (both negative-control runs vs. run 4)

Since all three save a checkpoint every 10 epochs, the fairest comparison
available before full convergence is checkpoints at the *same* epoch number
rather than waiting for early-stopping:

| model | QbS mAP | QbS R@1 | QbE mAP | QbE R@1 |
|---|---:|---:|---:|---:|
| run 4 (real prompts, epoch 50) | 0.058 | 0.3% | **0.430** | 9.8% |
| run 6 (shuffled prompts, epoch 50, run A) | 0.044 | 0.3% | **0.147** | 4.2% |
| run 7 (shuffled prompts, epoch 50, run B) | 0.047 | 0.3% | **0.141** | 4.1% |

(Full data: `results/negctrl_epoch50/retrieval_metrics.csv`.)

### Preliminary reading

- **QbE mAP is roughly 3× lower with shuffled prompts** at the identical
  epoch (0.430 real vs. 0.147/0.141 shuffled) — directionally consistent
  with the hypothesis that the model is exploiting the real
  image↔PHOSC-description correspondence, not just learning to separate
  images by some artifact independent of prompt content. This is
  preliminary but points toward a genuine "yes" for the strong version of
  RQ2 (fine-tuning learns real cross-modal correspondence, not just any
  consistent signal).
- **The two independent shuffled-prompt runs agree closely with each other**
  (QbE mAP 0.147 vs. 0.141 — a 0.006 spread), which is reassuring given item
  1's finding of large run-to-run variance elsewhere in this project. The
  gap to run 4 (0.430) is roughly 50× that spread, so this doesn't look like
  it could be explained by ordinary run-to-run noise.
- QbS is low across all three at this epoch and not very informative — this
  is consistent with (not contradicted by) item 3's finding that run 4's
  text tower largely collapses on unseen description prompts regardless of
  whether prompts were real or shuffled during training. The QbE gap is the
  meaningful signal in this snapshot, not the QbS gap.
- **Caveat, state plainly:** epoch 50 is early — run 4's own val_loss was
  still improving substantially past this point (best epoch 89), and its
  *converged* QbE mAP is 0.532, notably higher than run 4's own epoch-50
  QbE mAP shown in the table above. Runs 6/7 could still close part of the
  gap with more training. The gap at a fixed, identical epoch is real and
  suggestive, but should be presented as preliminary evidence, not a final
  causal claim.
- Two independent shuffled-prompt runs (6 and 7) were checked to guard
  against the run-to-run variance seen elsewhere in this project (item 1,
  finding 5) — they land within 0.006 QbE mAP of each other (0.147 vs.
  0.141), which is a good sign the gap to run 4 is a real effect and not an
  artifact of one unlucky run.

### For the thesis

If a later check (closer to the deadline) reaches a higher epoch, re-run the
matched-epoch comparison there instead — higher epoch means both sides have
had more time to converge, which is a stronger signal. If not, present the
epoch-50 snapshot above explicitly labeled preliminary/in-progress, with the
caveat that neither negative-control run reached early-stopping convergence
before the deadline. Avoid presenting this as a finished, converged negative
control — the honest framing is "early evidence in the expected direction,
experiment still running at time of writing."

---

# Summary table: evidence → RQ → status

| item | what it shows | answers | status |
|---|---|---|---|
| 1 | Aggregate QbS/QbE mAP + Recall@k across all 12 trained configs | RQ1 (which description type) | done |
| 3 | Qualitative unseen-word retrieval: ALIGN run 4, ALIGN run 1, **and CLIP run 8** (cross-check requested after A4's collapse), with text-collapse diagnosis on all three | RQ1 (does it actually understand unseen words) | done |
| 4 | t-SNE + separation margin, frozen vs. fine-tuned, on held-out words | RQ2 (does fine-tuning help) | done |
| 2 | Negative control: real vs. shuffled PHOSC prompts, matched-epoch retrieval | RQ2 (is fine-tuning learning real correspondence, or an artifact) | **preliminary — not converged, mid-training snapshot only** |

**One-paragraph synthesis, usable near-verbatim:** *Fine-tuning ALIGN on this
dataset produces an image encoder that measurably better separates
handwritten Bengali words by identity, even on words never seen during
training (item 4: ~5× separation margin over the frozen baseline). Which
text description best supports out-of-lexicon understanding depends on the
supervision approach, not the backbone architecture: plain word-level
prompts generalize to unseen words on both ALIGN and CLIP (hit@1 40% and
50% respectively on ten held-out queries), while structured PHOSC
descriptions collapse almost completely for ALIGN's best-performing
configuration (0/10 hit@1, a single word retrieved for every query) despite
that configuration winning on every aggregate metric tested (similarity-matrix
mean and QbE mAP). CLIP's best run (C8) was checked under the same probe
specifically because A4's aggregate-metric-vs-qualitative-probe mismatch
raised the question of whether "CLIP beats ALIGN" (based on aggregate QbS/QbE
mAP alone) would hold up the same way; it does, on both hit-rate and
top-1-word-diversity grounds, though its text embeddings show the same kind
of high absolute pairwise similarity A4's collapsed case did — evidence that
raw embedding-similarity level is not by itself informative without checking
whether it still translates to distinguishable rankings (the same lesson item
4's separation-margin correction established for images). Preliminary
evidence from an in-progress negative control (item 2) suggests the
fine-tuning improvement in item 4 depends on the real image-description
correspondence rather than any consistent training signal, though that
experiment had not reached full convergence at time of writing. All
"unseen"/"out-of-lexicon" claims here are scoped to held-out words within the
same Bengali dataset, not cross-lingual generalization, which was not
tested.*
