"""
t-SNE visualization of ALIGN image embeddings: frozen (pretrained) vs. fine-tuned
(best ALIGN config, run 4), on the same test-fold images.

Colors: identity is only meaningful for a handful of words in a scatter plot with
~40 classes (any two points can sit side by side, so the CVD-safety budget for a
fully-saturated categorical palette is 3 colors here, not 8 -- see
results/embeddings_writeup.md for why). So a small set of --highlight_words get
distinct, validated colors; every other word is folded into a single muted "other"
gray, per the usual practice for identity encoding once a palette runs out of safe
slots. Defaults to 3 of the same words already discussed in the qualitative
write-up (results/qualitative_writeup.md), so a reader can cross-reference.

Output: results/embeddings/tsne_frozen_vs_finetuned.png
"""
from __future__ import annotations

import argparse
import os
import unicodedata
from os.path import join as ospj

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    embedding_visualization_argparse,
)

from transformers import AlignModel, AutoProcessor

from retrieval_metrics_report import _unwrap_align_embeds
from qualitative_retrieval import load_bengali_font

# First 3 slots of the validated default categorical palette (dataviz skill,
# references/palette.md) -- the only 3 that clear the all-pairs CVD/normal-vision
# floors required for a scatter plot, where any two points can be neighbors.
HIGHLIGHT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
OTHER_COLOR = "#c9c8c3"  # muted, recessive -- "everything else" bucket
DEFAULT_HIGHLIGHT_WORDS = ["খারিদুনগরী", "জধুবেড়িয়া", "দেপাল"]  # same 3 discussed in qualitative_writeup.md


@torch.no_grad()
def collect_image_features(model, dataloader, image_loader: ImageLoader, desc: str):
    processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
    model.eval()
    feats, labels = [], []

    for batch in tqdm(dataloader, desc=desc):
        *_, img_paths, _, words = batch
        pil_images = [image_loader(p) for p in img_paths]
        proc = processor(images=pil_images, return_tensors="pt")
        proc = {k: v.to(device) for k, v in proc.items()}

        f = _unwrap_align_embeds(model.get_image_features(**proc))
        f = F.normalize(f, dim=-1)
        feats.append(f)
        labels.extend(list(words))

    return torch.cat(feats, dim=0).detach().cpu().numpy(), labels


def mean_intra_class_cosine(feats: np.ndarray, labels) -> dict:
    """
    For every word with >=2 test images, mean pairwise cosine similarity among that
    word's own image embeddings (self-pairs excluded). `feats` rows must already be
    L2-normalized (dot product = cosine).

    NOTE: this alone is misleading -- pretrained embedding spaces are commonly
    "anisotropic" (all embeddings sit in a narrow cone of the hypersphere), which
    inflates *every* pairwise similarity, same-class or not. A high raw intra-class
    number can just mean "everything looks similar to everything," not "same-class
    images look similar to each other specifically." See separation_margin() for
    the number that actually predicts retrieval quality (rank-based metrics like
    QbE mAP are invariant to a uniform shift in similarity, they only care about
    relative ordering).
    """
    labels = np.array([unicodedata.normalize('NFC', w) for w in labels])
    out = {}
    for w in sorted(set(labels)):
        idx = np.where(labels == w)[0]
        if len(idx) < 2:
            continue
        sub = feats[idx]
        sim = sub @ sub.T
        off_diag = sim[~np.eye(len(idx), dtype=bool)]
        out[w] = float(off_diag.mean())
    return out


def separation_margin(feats: np.ndarray, labels) -> dict:
    """
    mean(same-class pairwise cosine) - mean(different-class pairwise cosine),
    over the *entire* test set at once (not per word) -- the embedding-space
    analogue of "how separable are the classes," robust to a uniform anisotropic
    shift in raw similarity level. Positive and large = same-class images sit
    measurably closer together than different-class images, which is what
    ranking-based retrieval metrics (QbE mAP/Recall) actually depend on.
    """
    labels = np.array([unicodedata.normalize('NFC', w) for w in labels])
    S = feats @ feats.T
    same = labels[:, None] == labels[None, :]
    eye = np.eye(len(labels), dtype=bool)
    same_mask = same & ~eye
    diff_mask = ~same
    intra_mean = float(S[same_mask].mean())
    inter_mean = float(S[diff_mask].mean())
    return {'intra_class_mean': intra_mean, 'inter_class_mean': inter_mean,
            'separation_margin': intra_mean - inter_mean}


def run_tsne(feats: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    n = feats.shape[0]
    perplexity = min(perplexity, max(5.0, (n - 1) / 3.0))  # keep sklearn happy on small N
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                init='pca', learning_rate='auto')
    return tsne.fit_transform(feats)


def plot_panel(ax, coords: np.ndarray, labels, highlight_words, font_prop, title: str):
    # Bengali text can arrive in different Unicode normalization forms (precomposed
    # vs. combining-mark sequences) that look identical but aren't `==` equal --
    # normalize both sides so word matching actually works regardless of source.
    labels = np.array([unicodedata.normalize('NFC', w) for w in labels])
    highlight_words = [unicodedata.normalize('NFC', w) for w in highlight_words]
    other_mask = ~np.isin(labels, highlight_words)

    ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
               s=14, c=OTHER_COLOR, alpha=0.6, linewidths=0, label='_nolegend_')

    for word, color in zip(highlight_words, HIGHLIGHT_COLORS):
        mask = labels == word
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=42, c=color, alpha=0.95, linewidths=0.6, edgecolors='white',
                   label='_nolegend_')
        # Relief label (aqua sits below 3:1 contrast on a white surface per the
        # palette doc -- a direct label keeps that cluster identifiable regardless).
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.annotate(word, (cx, cy), fontproperties=font_prop, fontsize=10,
                    color=color, ha='center', va='center',
                    xytext=(0, 12), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))

    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser = dataset_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = aling_fine_tune_argparse(parser)
    parser = clip_fine_tune_argparse(parser)
    parser = loss_func_argparse(parser)
    parser = training_common_argparse(parser)
    parser = embedding_visualization_argparse(parser)

    args = parser.parse_args(args) if args is not None else parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    highlight_words = args.highlight_words or DEFAULT_HIGHLIGHT_WORDS
    if len(highlight_words) > 3:
        print(f'[warn] {len(highlight_words)} highlight words given, but only the first 3 colors '
              f'({HIGHLIGHT_COLORS}) pass CVD-safety for a scatter plot (see module docstring). '
              f'Using only the first 3: {highlight_words[:3]}')
        highlight_words = highlight_words[:3]

    root_dir = ospj(DATA_FOLDER, "BengaliWords_CroppedVersion_Folds")
    image_loader = ImageLoader(ospj(root_dir, args.split_name))

    set_phos_version(args.phosc_version)
    set_phoc_version(args.phosc_version)

    font_prop = load_bengali_font()

    test_loader, _ = get_test_loader(args, None)

    print('Loading frozen (pretrained) ALIGN...')
    frozen_model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()
    frozen_feats, frozen_labels = collect_image_features(
        frozen_model, test_loader, image_loader, desc="Frozen ALIGN: collecting image features")
    del frozen_model
    torch.cuda.empty_cache()

    print(f'Loading fine-tuned ALIGN from {args.finetuned_model_dir}...')
    ft_model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()
    ckpt_path = ospj(args.finetuned_model_dir, args.checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=device)
    ft_model.load_state_dict(state, strict=True)
    ft_feats, ft_labels = collect_image_features(
        ft_model, test_loader, image_loader, desc="Fine-tuned ALIGN: collecting image features")
    del ft_model
    torch.cuda.empty_cache()

    frozen_intra = mean_intra_class_cosine(frozen_feats, frozen_labels)
    ft_intra = mean_intra_class_cosine(ft_feats, ft_labels)
    common_words = sorted(set(frozen_intra) & set(ft_intra))
    intra_df = pd.DataFrame([
        {'word': w, 'frozen_intra_class_cosine': frozen_intra[w], 'finetuned_intra_class_cosine': ft_intra[w]}
        for w in common_words
    ])
    intra_csv_path = ospj(args.out_dir, 'intra_class_similarity.csv')
    intra_df.to_csv(intra_csv_path, index=False)

    frozen_sep = separation_margin(frozen_feats, frozen_labels)
    ft_sep = separation_margin(ft_feats, ft_labels)
    sep_df = pd.DataFrame([
        {'model': 'frozen ALIGN (pretrained)', **frozen_sep},
        {'model': 'fine-tuned ALIGN (run 4)', **ft_sep},
    ])
    sep_csv_path = ospj(args.out_dir, 'separation_margin.csv')
    sep_df.to_csv(sep_csv_path, index=False)

    print(f'\nRaw mean intra-class cosine similarity (image embeddings only), '
          f'averaged over {len(common_words)} words with >=2 test images each:')
    print(f'  frozen ALIGN:     {intra_df["frozen_intra_class_cosine"].mean():.4f}')
    print(f'  fine-tuned ALIGN: {intra_df["finetuned_intra_class_cosine"].mean():.4f}')
    print(f'  (raw intra-class alone is misleading -- see separation margin below)')
    print(f'\nSeparation margin = mean(same-class cosine) - mean(different-class cosine), '
          f'over all {len(frozen_labels)} test images at once:')
    print(f'  frozen ALIGN:     intra={frozen_sep["intra_class_mean"]:.4f}  '
          f'inter={frozen_sep["inter_class_mean"]:.4f}  margin={frozen_sep["separation_margin"]:.4f}')
    print(f'  fine-tuned ALIGN: intra={ft_sep["intra_class_mean"]:.4f}  '
          f'inter={ft_sep["inter_class_mean"]:.4f}  margin={ft_sep["separation_margin"]:.4f}')
    print(f'Wrote {intra_csv_path}')
    print(f'Wrote {sep_csv_path}\n')

    print(f'Running t-SNE (perplexity={args.tsne_perplexity}, seed={args.seed}) on both embedding sets...')
    frozen_coords = run_tsne(frozen_feats, args.seed, args.tsne_perplexity)
    ft_coords = run_tsne(ft_feats, args.seed, args.tsne_perplexity)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    plot_panel(axes[0], frozen_coords, frozen_labels, highlight_words, font_prop,
               'Frozen ALIGN (pretrained, no fine-tuning)')
    plot_panel(axes[1], ft_coords, ft_labels, highlight_words, font_prop,
               'Fine-tuned ALIGN (best config, run 4)')

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=9,
               markerfacecolor=color, markeredgecolor='white', label=word)
        for word, color in zip(highlight_words, HIGHLIGHT_COLORS)
    ]
    legend_handles.append(
        Line2D([0], [0], marker='o', linestyle='', markersize=8,
               markerfacecolor=OTHER_COLOR, markeredgecolor='none', alpha=0.8,
               label='other test-fold words')
    )
    fig.legend(handles=legend_handles, prop=font_prop, loc='lower center',
               ncol=len(legend_handles), frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('ALIGN image-embedding t-SNE: frozen vs. fine-tuned', fontsize=14)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    out_path = ospj(args.out_dir, 'tsne_frozen_vs_finetuned.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
