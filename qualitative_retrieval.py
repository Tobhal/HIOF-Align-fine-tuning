"""
Generate qualitative retrieval examples from the best ALIGN model.

For a fixed-seed random sample of unseen (val-fold) query words, build each
word's text query exactly the way that run was trained (read from its own
model_args.toml -- for the best ALIGN run, saved_models/align-fine-tune/
fold_3_aug_square/4, this is a PHOSC description), encode it, and show the
top-K retrieved images by cosine similarity: some hits, some misses, for
Chapter 4's discussion.

Outputs (default results/qualitative/):
  <NN>_<word>.png   one figure per query: query word + description text + top-K
                     thumbnails, green border = correct word, red = wrong word.
  summary.csv        one row per query: hit@1, hit@K, rank of the first correct
                     hit over the *full* ranking (not just top-K), and the
                     retrieved words -- for eyeballing successes/failures at a
                     glance without opening every image.
"""
from __future__ import annotations

import argparse
import json
import os
from os.path import join as ospj
from random import Random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from flags import DATA_FOLDER, device
from data.dataset_bengali import ImageLoader
from utils.get_dataset import get_test_loader
from modules.utils import set_phos_version, set_phoc_version

from parser import (
    dataset_argparse,
    phosc_net_argparse,
    aling_fine_tune_argparse,
    clip_fine_tune_argparse,
    loss_func_argparse,
    training_common_argparse,
    qualitative_retrieval_argparse,
)

from transformers import AlignModel, AutoProcessor, AutoTokenizer

# Reuse the already-verified helpers instead of re-implementing them:
#  - _unwrap_align_embeds handles the transformers-version return-type quirk
#  - build_text_queries mirrors align_fine_tune.py's word/description branching
#  - read_model_args/extract_hparams read a run's saved model_args.toml
from retrieval_metrics_report import (
    _unwrap_align_embeds,
    build_text_queries,
    read_model_args,
    extract_hparams,
)

FONT_PATH = ospj(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'NotoSansBengali-Regular.ttf')


def load_bengali_font() -> fm.FontProperties:
    """Bengali query words/descriptions render as empty boxes under matplotlib's
    default font. Noto Sans Bengali (assets/fonts/) fixes that."""
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        return fm.FontProperties(fname=FONT_PATH)
    print(f'[warn] Bengali font not found at {FONT_PATH}; Bengali text may render as boxes. '
          f'Download e.g. Noto Sans Bengali and place it there.')
    return fm.FontProperties()


@torch.no_grad()
def collect_image_features_and_paths_align(model, dataloader, image_loader: ImageLoader):
    processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
    model.eval()
    feats, labels, paths = [], [], []

    for batch in tqdm(dataloader, desc="Collecting image features"):
        *_, img_paths, _, words = batch
        pil_images = [image_loader(p) for p in img_paths]
        proc = processor(images=pil_images, return_tensors="pt")
        proc = {k: v.to(device) for k, v in proc.items()}

        f = _unwrap_align_embeds(model.get_image_features(**proc))
        f = F.normalize(f, dim=-1)
        feats.append(f)
        labels.extend(list(words))
        paths.extend(list(img_paths))

    return torch.cat(feats, dim=0), labels, paths


@torch.no_grad()
def encode_text_align(model, texts: List[str]) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
    tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    f = _unwrap_align_embeds(model.get_text_features(**tok))
    return F.normalize(f, dim=-1)


@torch.no_grad()
def collect_image_features_and_paths_clip(model, dataloader, image_loader: ImageLoader, preprocess):
    """Mirrors retrieval_metrics_report.collect_image_features_clip, plus paths (needed
    here to render thumbnails, which the aggregate-metrics script doesn't need)."""
    model.eval()
    feats, labels, paths = [], [], []

    for batch in tqdm(dataloader, desc="Collecting image features (CLIP)"):
        *_, img_paths, _, words = batch
        pil_images = [image_loader(p) for p in img_paths]
        images = torch.stack([preprocess(img) for img in pil_images]).to(device)

        f = model.encode_image(images)
        f = F.normalize(f, dim=-1)
        feats.append(f)
        labels.extend(list(words))
        paths.extend(list(img_paths))

    return torch.cat(feats, dim=0), labels, paths


@torch.no_grad()
def encode_text_clip(model, texts: List[str]) -> torch.Tensor:
    """Mirrors retrieval_metrics_report.collect_text_features_clip (single-chunk here,
    since queries are a handful of words, not the full test set)."""
    import clip as openai_clip
    model.eval()
    text_tokens = openai_clip.tokenize(texts, truncate=True).to(device)
    f = model.encode_text(text_tokens)
    return F.normalize(f, dim=-1)


def load_query_words(args, unique_words: List[str]) -> List[str]:
    """Either the literal word list from --query_words_file (guarantees an identical
    query set across model families / runs, since the sampled pool of usable images can
    differ slightly run to run even with the same --seed), or a fresh random sample."""
    if args.query_words_file:
        with open(args.query_words_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        missing = [w for w in words if w not in unique_words]
        if missing:
            print(f'[warn] {len(missing)} word(s) from {args.query_words_file} are not in this '
                  f"run's test set and will be skipped: {missing}")
            words = [w for w in words if w in unique_words]
        print(f'Using {len(words)} query words from {args.query_words_file} (fixed list, no sampling).')
        return words

    rng = Random(args.seed)
    n_queries = min(args.num_queries, len(unique_words))
    if n_queries < args.num_queries:
        print(f'[warn] only {len(unique_words)} unique words available, using all of them.')
    words = rng.sample(unique_words, n_queries)
    print(f'Sampled {n_queries} unseen query words (seed={args.seed}): {words}')
    return words


def sanitize_filename(word: str) -> str:
    cleaned = "".join(c for c in word if c not in '/\\:*?"<>|\n\r\t')
    return cleaned.strip() or "query"


def make_figure(query_word: str, description: str, top_words: List[str], top_paths: List[str],
                 top_scores: List[float], image_loader: ImageLoader, font_prop: fm.FontProperties,
                 out_path: str, top_k: int):
    fig, axes = plt.subplots(
        1, top_k + 1, figsize=(2.6 * top_k + 3.0, 3.4),
        gridspec_kw={'width_ratios': [1.15] + [1.0] * top_k},
    )

    # Left panel: the query itself (word + the description text actually fed to the model)
    ax0 = axes[0]
    ax0.axis('off')
    ax0.set_title('query', fontsize=10)
    ax0.text(0.5, 0.85, query_word, fontproperties=font_prop, fontsize=18, ha='center', va='center')
    shown_desc = description if len(description) <= 220 else description[:217] + '...'
    ax0.text(0.5, 0.45, shown_desc, fontproperties=font_prop, fontsize=7, ha='center', va='center', wrap=True)

    for i in range(top_k):
        ax = axes[i + 1]
        img = image_loader(top_paths[i])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        is_correct = (top_words[i] == query_word)
        color = 'tab:green' if is_correct else 'tab:red'
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(4)

        mark = '✓' if is_correct else '✗'  # check / cross
        ax.set_title(f'#{i + 1} {mark} sim={top_scores[i]:.2f}', fontsize=8.5, color=color)
        ax.set_xlabel(top_words[i], fontproperties=font_prop, fontsize=9)

    hit1 = 'yes' if top_words[0] == query_word else 'no'
    hitk = 'yes' if query_word in top_words else 'no'
    fig.suptitle(f'Query: {query_word}   (hit@1={hit1}, hit@{top_k}={hitk})',
                 fontproperties=font_prop, fontsize=13)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.03, right=0.99, wspace=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser = dataset_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = aling_fine_tune_argparse(parser)
    parser = clip_fine_tune_argparse(parser)
    parser = loss_func_argparse(parser)
    parser = training_common_argparse(parser)
    parser = qualitative_retrieval_argparse(parser)

    args = parser.parse_args(args) if args is not None else parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    root_dir = ospj(DATA_FOLDER, "BengaliWords_CroppedVersion_Folds")
    image_loader = ImageLoader(ospj(root_dir, args.split_name))

    set_phos_version(args.phosc_version)
    set_phoc_version(args.phosc_version)

    font_prop = load_bengali_font()

    model_args = read_model_args(args.model_dir)
    hparams = extract_hparams(model_args)
    description_mode = hparams.get('description_mode') or 'word'
    print(f'Model dir: {args.model_dir}')
    print(f'description_mode={description_mode}  loss_func={hparams.get("loss_func")}  '
          f'effective_batch_size={hparams.get("effective_batch_size")}')

    test_loader, _ = get_test_loader(args, None)

    ckpt_path = ospj(args.model_dir, args.checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    if args.model_family == 'clip':
        import clip as openai_clip
        model, preprocess = openai_clip.load("ViT-B/32", device=device)
        model = model.float().eval()
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=True)
        img_feats, img_labels, img_paths = collect_image_features_and_paths_clip(
            model, test_loader, image_loader, preprocess)
        encode_fn = lambda texts: encode_text_clip(model, texts)
    else:
        model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)
        img_feats, img_labels, img_paths = collect_image_features_and_paths_align(
            model, test_loader, image_loader)
        encode_fn = lambda texts: encode_text_align(model, texts)

    print(f'Collected {len(img_labels)} test images, {len(set(img_labels))} unique words.')

    unique_words = sorted(set(img_labels))
    query_words = load_query_words(args, unique_words)

    top_k = args.top_k
    rows = []
    query_text_feats = []  # collected for the text-text collapse diagnostic below

    for i, word in enumerate(query_words):
        (description,) = build_text_queries([word], description_mode)
        text_feat = encode_fn([description])  # [1, D]
        query_text_feats.append(text_feat)

        sims = (text_feat @ img_feats.T).squeeze(0).detach().cpu().numpy()  # [N]
        order = np.argsort(-sims)
        ranked_words = [img_labels[j] for j in order]
        rank_first_hit = next((r + 1 for r, w in enumerate(ranked_words) if w == word), None)

        top_idx = order[:top_k]
        top_words = [img_labels[j] for j in top_idx]
        top_paths = [img_paths[j] for j in top_idx]
        top_scores = [float(sims[j]) for j in top_idx]

        hit1 = top_words[0] == word
        hitk = word in top_words

        fname = f'{i:02d}_{sanitize_filename(word)}.png'
        out_path = ospj(args.out_dir, fname)
        make_figure(word, description, top_words, top_paths, top_scores,
                    image_loader, font_prop, out_path, top_k)

        row = {
            'query_word': word,
            'description': description,
            'hit@1': hit1,
            f'hit@{top_k}': hitk,
            'rank_of_first_correct_hit': rank_first_hit,
            'n_test_images_with_this_word': ranked_words.count(word),
        }
        for r, (w, s) in enumerate(zip(top_words, top_scores)):
            row[f'top{r + 1}_word'] = w
            row[f'top{r + 1}_sim'] = round(s, 4)
        row['figure'] = fname
        rows.append(row)

        print(f'[{i + 1}/{len(query_words)}] {word}: hit@1={hit1} hit@{top_k}={hitk} '
              f'rank_of_first_correct_hit={rank_first_hit}')

    summary_df = pd.DataFrame(rows)
    summary_path = ospj(args.out_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f'\nWrote {summary_path}')
    print(f'hit@1: {summary_df["hit@1"].mean():.2%}   hit@{top_k}: {summary_df[f"hit@{top_k}"].mean():.2%}')

    # Text-text collapse diagnostic (the check that actually explains *why* a hit-rate
    # result holds up or not -- see results/qualitative_writeup.md for the ALIGN run 4
    # case this was built for). Reuses the text embeddings already computed above, no
    # extra encode calls needed.
    all_text_feats = torch.cat(query_text_feats, dim=0)  # [n_queries, D]
    sim = (all_text_feats @ all_text_feats.T).detach().cpu()
    n = sim.shape[0]
    off_diag = sim[~torch.eye(n, dtype=torch.bool)]
    collapse_stats = {
        'model_family': args.model_family,
        'model_dir': args.model_dir,
        'n_queries': n,
        'off_diag_min': float(off_diag.min()),
        'off_diag_mean': float(off_diag.mean()),
        'off_diag_max': float(off_diag.max()),
    }
    collapse_path = ospj(args.out_dir, 'text_text_similarity.json')
    with open(collapse_path, 'w', encoding='utf-8') as f:
        json.dump(collapse_stats, f, ensure_ascii=False, indent=2)
    print(f"\n[text-text collapse check] off-diagonal cosine similarity across the "
          f"{n} query texts: min={collapse_stats['off_diag_min']:.4f}  "
          f"mean={collapse_stats['off_diag_mean']:.4f}  max={collapse_stats['off_diag_max']:.4f}")
    print(f'Wrote {collapse_path}')

    return summary_df


if __name__ == '__main__':
    main()
