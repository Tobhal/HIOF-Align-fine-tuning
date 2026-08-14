from __future__ import annotations

from utils.get_dataset import get_phoscnet, get_test_loader
from data.dataset_bengali import ImageLoader

import os
from os import PathLike
from os.path import join as ospj
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm import create_model

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

# (You keep these if other parts of your project import them)
from flags import DATA_FOLDER, device
from data.dataset_bengali import ImageLoader
from utils.dbe import dbe

# Optional/legacy imports kept for compatibility
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms

from transformers import (
    AlignProcessor,
    AlignModel,
    AutoTokenizer,
    AutoProcessor,
)

from utils.get_dataset import get_test_loader, get_phoscnet
from parser import (
    dataset_argparse,
    phosc_net_argparse,
    aling_fine_tune_argparse,
    matrix_new_argparse,
    clip_fine_tune_argparse,
    loss_func_argparse,
    training_common_argparse,
)


split = 'fold_0_t'
use_augmented = False

# ALIGN (HF): processor can batch images (and texts). We'll use AutoProcessor for images below.
# align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
# align_model = AlignModel.from_pretrained("kakaobrain/align-base")
# align_auto_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
# align_auto_processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

# (Kept for completeness; not used in this matrix script)
clip_preprocess_default = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def _downsample_matrix_avgpool(M: torch.Tensor, block: int = 1) -> torch.Tensor:
    """
Downsample a square matrix by average pooling (block x block).
Drops leftover rows/cols if N is not divisible by block.
    """
    if block <= 1:
        return M
    if torch.is_tensor(M):
        X = M
    else:
        X = torch.tensor(M, dtype=torch.float32)

    # keep square assumption; trim edges if needed
    N = X.shape[0]
    trim = N - (N // block) * block
    if trim > 0:
        X = X[:-trim, :-trim]

    X = X.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    X = F.avg_pool2d(X, kernel_size=block, stride=block, ceil_mode=False)
    return X.squeeze(0).squeeze(0)  # [H', W']


def make_tee_logger(txt_path: str):
    """
    Returns a function log(msg) that prints to stdout AND appends to txt_path.
    """
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    def log(msg: str = ""):
        print(msg)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log


def save_heatmap(
        S: torch.Tensor,
        out_path: str,
        title: str | None = None,
        vmin: float = -1.0,
        vmax: float = 1.0,
        cmap: str = "coolwarm",
        dpi: int = 300,
        cell_px: int = 3,
        max_width_px: int | None = 1800,
        tick_step: int | None = None,
        add_colorbar: bool = True,
        downsample_block: int = 1,
):
    """
Save a cosine-similarity heatmap for S (N x N), with crisp cells and small squares for N~300-400.

- cell_px sets size of each cell in pixels before capping by max_width_px.
- If max_width_px is set, cell size is reduced to keep figure at/below that width.
- vmin/vmax fix the color scale (important when comparing different runs).
- downsample_block>1 will average-pool the matrix (e.g., 2, 4, 8) for a smaller overview.

    """
    # Downsample if requested
    if downsample_block and downsample_block > 1:
        S = _downsample_matrix_avgpool(S, downsample_block)

    # Move to CPU numpy
    if torch.is_tensor(S):
        S_np = S.detach().cpu().numpy()
    else:
        S_np = np.asarray(S)
    N = S_np.shape[0]

    # Determine figure size from desired pixels per cell
    width_px = N * cell_px
    if max_width_px is not None and width_px > max_width_px:
        cell_px = max(1, max_width_px // max(1, N))
        width_px = N * cell_px

    fig_w_in = max(3.0, width_px / dpi)  # keep at least 3 inches for readability
    fig_h_in = fig_w_in

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), constrained_layout=True)
    # imshow is the recommended fast path for raster heatmaps on regular grids
    im = ax.imshow(
        S_np,
        vmin=vmin, vmax=vmax,  # fixed normalization for cosine [-1,1]
        cmap=cmap,
        interpolation="nearest",  # no smoothing; crisp cells
        aspect="equal",
    )

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("cosine similarity", rotation=270, labelpad=12)

    ax.set_title(title or f"Cosine similarity (N={N})", fontsize=10)
    ax.set_xlabel("image index")
    ax.set_ylabel("image index")

    # Ticks: off by default for large N; enable sparse ticks with tick_step
    if tick_step and tick_step > 0:
        idx = np.arange(0, N, tick_step)
        ax.set_xticks(idx)
        ax.set_yticks(idx)
        ax.tick_params(axis="both", which="both", labelsize=6, length=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved at: {out_path}")


@dataclass
class Result:
    model_number: int
    min_value: float
    max_value: float
    average_value: float


def save_matrix(matrix: torch.Tensor, _model_save_path: PathLike, csv_filename="matrix"):
    """
Save the given matrix as CSV and a small text summary next to it.
    """
    directory = os.path.dirname(_model_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    csv_path = ospj(directory, f'{csv_filename}.csv')
    txt_path = ospj(directory, f'{csv_filename}.txt')

    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()

    df = pd.DataFrame(matrix)
    df.to_csv(csv_path, index=False, header=False)

    print(f"Matrix saved at: {csv_path}")


@torch.no_grad()
def collect_image_features_align(model, dataloader, image_loader, device):
    align_auto_processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
    model.eval()
    feats, labels = [], []

    for batch in tqdm(dataloader, desc="Collecting image features"):
        # You already used img_paths at -3 and words at -1 in your main().
        # We'll mirror that assumption here.
        try:
            *_, img_paths, _, words = batch
        except Exception as e:
            raise RuntimeError(
                "Expected batch like (*_, image_paths, _, words). "
                "Adjust unpacking to your Dataset.__getitem__."
            ) from e

        pil_images = [image_loader(p) for p in img_paths]
        proc = align_auto_processor(images=pil_images, return_tensors="pt")
        proc = {k: v.to(device) for k, v in proc.items()}

        f = model.get_image_features(**proc)
        f = F.normalize(f, dim=-1)
        feats.append(f)

        labels.extend(list(words))

    return torch.cat(feats, dim=0), labels


@torch.no_grad()
def collect_image_features_clip(
        model: nn.Module,
        dataloader: DataLoader,
        image_loader: ImageLoader,
        preprocess,
        device: str
) -> torch.Tensor:
    model.eval()
    feats, labels = [], []

    for batch in tqdm(dataloader, desc="Collecting CLIP image features"):
        try:
            *_, img_paths, _, words = batch
        except Exception as e:
            raise RuntimeError("Expected batch like (*_, image_paths, _, words).") from e

        pil_images = [image_loader(p) for p in img_paths]
        images = torch.stack([preprocess(img) for img in pil_images]).to(device)

        f = model.encode_image(images)
        f = F.normalize(f, dim=-1)
        feats.append(f)

        labels.extend(list(words))

    feats = torch.cat(feats, dim=0)
    return feats, labels


@torch.no_grad()
def cosine_similarity_matrix(feats: torch.Tensor) -> torch.Tensor:
    """
Compute N×N cosine similarity matrix from L2-normalized features [N, D].
    """
    # feats must be unit-normalized along dim=-1
    return feats @ feats.T  # [N, N], values in [-1, 1]


@torch.no_grad()
def collect_text_features_align(
        model: nn.Module,
        texts: List[str],
        device: str
) -> torch.Tensor:
    """
Encode a list of texts with ALIGN, L2-normalize, return [N, D].
    """
    align_auto_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
    model.eval()
    out = []
    # Batch in chunks to avoid OOM if texts is huge
    B = 512
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        tok = align_auto_tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        f = model.get_text_features(**tok)  # [b, D]
        f = F.normalize(f, dim=-1)
        out.append(f)
    return torch.cat(out, dim=0)  # [N, D]


@torch.no_grad()
def collect_text_features_clip(
        model: nn.Module,
        texts: List[str],
        device: str
) -> torch.Tensor:
    """
Encode a list of texts with CLIP, L2-normalize, return [N, D].
    """
    model.eval()
    out = []
    B = 512
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        text_tokens = clip.tokenize(chunk).to(device)
        f = model.encode_text(text_tokens)
        f = F.normalize(f, dim=-1)
        out.append(f)
    return torch.cat(out, dim=0)


def average_precision_from_ranking(relevant_mask_sorted: np.ndarray) -> float:
    """
    relevant_mask_sorted: boolean array over the ranked list (True where item is relevant).
    Computes AP = mean precision at ranks where relevant occurs.
    """
    if relevant_mask_sorted.size == 0:
        return 0.0
    rel_idx = np.flatnonzero(relevant_mask_sorted)
    if rel_idx.size == 0:
        return 0.0

    # precision at each relevant hit
    precisions = []
    for r_i in rel_idx:
        # ranks are 1-based in definition
        k = r_i + 1
        precisions.append(relevant_mask_sorted[:k].sum() / k)
    return float(np.mean(precisions))


def recall_at_k_from_ranking(relevant_mask_sorted: np.ndarray, k: int) -> float:
    """
    Recall@k = (# relevant in top-k) / (total # relevant).
    """
    if relevant_mask_sorted.size == 0:
        return 0.0
    total_rel = relevant_mask_sorted.sum()
    if total_rel == 0:
        return 0.0
    k = min(k, relevant_mask_sorted.size)
    return float(relevant_mask_sorted[:k].sum() / total_rel)


@torch.no_grad()
def compute_qbe_map_recall(
    feats: torch.Tensor,
    labels: List[str],
    ks: Tuple[int, ...] = (1, 5, 10),
    exclude_self: bool = True,
) -> dict:
    """
    QbE: image->image retrieval.
    For each query i, rank all database images by cosine similarity to i.
    Relevant if same label. Optionally exclude the query itself from the ranked list.
    """
    feats = F.normalize(feats, dim=-1)
    S = (feats @ feats.T).detach().cpu().numpy()  # [N,N]
    N = S.shape[0]
    labels_np = np.array(labels)

    ap_list = []
    recall_lists = {k: [] for k in ks}

    for i in range(N):
        sims = S[i].copy()
        if exclude_self:
            sims[i] = -np.inf  # ensure self not retrieved

        order = np.argsort(-sims)  # descending
        rel = (labels_np[order] == labels_np[i])

        ap = average_precision_from_ranking(rel)
        ap_list.append(ap)

        for k in ks:
            recall_lists[k].append(recall_at_k_from_ranking(rel, k))

    out = {
        "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "AP_per_query": ap_list,  # optional, can remove if you don't want it
    }
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recall_lists[k])) if recall_lists[k] else 0.0
    return out


@torch.no_grad()
def compute_qbs_map_recall(
    text_feats: torch.Tensor,
    image_feats: torch.Tensor,
    text_labels: List[str],
    image_labels: List[str],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> dict:
    """
    QbS: text->image retrieval.
    For each query text q, rank all images by cosine similarity between text_feats[q] and image_feats[*].
    Relevant if labels match.
    """
    text_feats = F.normalize(text_feats, dim=-1)
    image_feats = F.normalize(image_feats, dim=-1)

    S = (text_feats @ image_feats.T).detach().cpu().numpy()  # [Nt, Ni]
    text_labels_np = np.array(text_labels)
    image_labels_np = np.array(image_labels)

    ap_list = []
    recall_lists = {k: [] for k in ks}

    for i in range(S.shape[0]):
        sims = S[i]
        order = np.argsort(-sims)
        rel = (image_labels_np[order] == text_labels_np[i])

        ap_list.append(average_precision_from_ranking(rel))
        for k in ks:
            recall_lists[k].append(recall_at_k_from_ranking(rel, k))

    out = {"mAP": float(np.mean(ap_list)) if ap_list else 0.0}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recall_lists[k])) if recall_lists[k] else 0.0
    return out


def print_results(results: List[Result]):
    for r in results:
        print(f"Model {r.model_number}: min={r.min_value:.6f}  max={r.max_value:.6f}  mean={r.average_value:.6f}")


def main(args=None, model=None, index=0) -> List[Result]:
    parser = argparse.ArgumentParser()

    parser = dataset_argparse(parser)
    parser = matrix_new_argparse(parser)
    parser = clip_fine_tune_argparse(parser)
    parser = aling_fine_tune_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = loss_func_argparse(parser)
    parser = training_common_argparse(parser)

    # Parse args
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    print(args.save_name)

    # Decide which model type we are using
    if args.save_name == 'clip-fine-tune':
        print("clip")
        model_type = 'CLIP'
    elif args.save_name == 'align-fine-tune':
        print("align")
        model_type = 'ALIGN'
    else:
        # Fallback or error; based on issue description, these are the two expected values.
        # We can default to ALIGN if it's not clip_fine_tune.
        model_type = 'ALIGN'

    # Dataset / loader
    # NOTE: Adapt this root_dir to your actual project layout
    root_dir = ospj(DATA_FOLDER, "BengaliWords_CroppedVersion_Folds")
    phosc_model = None
    test_loader, _ = get_test_loader(args, phosc_model)
    image_loader = ImageLoader(ospj(root_dir, args.split_name))

    for num in args.nums:
        model_dir = ospj(args.save_dir, args.save_name, args.split_name, str(num))

        report_path = ospj(model_dir, f"matrix_{num}_report.txt")
        # reset file each run
        if os.path.exists(report_path):
            os.remove(report_path)
        log = make_tee_logger(report_path)

        if model_type == 'ALIGN':
            # Load the fine-tuned ALIGN model
            fine_tuned_model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()

            if args.model_source == 'fine-tuned':
                ckpt_path = ospj(model_dir, args.checkpoint_name)
                print(f"Loading ALIGN model from {ckpt_path}")
                state = torch.load(ckpt_path, map_location=device)
                fine_tuned_model.load_state_dict(state, strict=True)

            preprocess = None # ALIGN uses its own processor inside collection funcs

        else:
            # Load the fine-tuned CLIP model
            fine_tuned_model, preprocess = clip.load("ViT-B/32", device=device)
            fine_tuned_model = fine_tuned_model.float().eval()

            if args.model_source == 'fine-tuned':
                ckpt_path = ospj(model_dir, args.checkpoint_name)
                print(f"Loading CLIP model from {ckpt_path}")
                state = torch.load(ckpt_path, map_location=device)

                # CLIP checkpoints often contain the state_dict directly or under a key
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']

                fine_tuned_model.load_state_dict(state, strict=True)

        all_words = []
        for batch in tqdm(test_loader, desc="Collecting words"):
            try:
                *_, _, _, words = batch
            except Exception as e:
                raise RuntimeError("Expected words in batch at position -1.") from e
            all_words.extend(list(words))

        if model_type == 'ALIGN':
            print(f"Collecting text features for ALIGN model")
            feats = collect_text_features_align(fine_tuned_model, all_words, device)
        else:
            print(f"Collecting text features for CLIP model")
            feats = collect_text_features_clip(fine_tuned_model, all_words, device)

        if model_type == 'ALIGN':
            print(f"Collecting image features for ALIGN model")
            img_feats, img_labels = collect_image_features_align(fine_tuned_model, test_loader, image_loader, device)
        else:
            print(f"Collecting image features for CLIP model")
            img_feats, img_labels = collect_image_features_clip(fine_tuned_model, test_loader, image_loader, preprocess, device)

        qbs_metrics = compute_qbs_map_recall(
            text_feats=feats,
            image_feats=img_feats,
            text_labels=all_words,
            image_labels=img_labels,
            ks=tuple(args.recall_ks) if hasattr(args, "recall_ks") else (1, 5, 10),
        )
        log(f"[QbS] mAP={qbs_metrics['mAP']:.4f}")
        log(f"R@1={qbs_metrics.get('Recall@1', 0):.4f}")
        log(f"R@5={qbs_metrics.get('Recall@5', 0):.4f}")
        log(f"R@10={qbs_metrics.get('Recall@10', 0):.4f}")

        # Cosine similarity matrix (vectorized)
        S = cosine_similarity_matrix(img_feats)  # [N, N], cosine in [-1, 1]
        # If you prefer cosine *distance*: D = 1 - S  (bounded in [0, 2] for normalized vecs)

        # Stats
        min_value = torch.min(S).item()
        max_value = torch.max(S).item()
        mean_value = torch.mean(S).item()

        log(f"min={min_value:.6f}")
        log(f"max={max_value:.6f}")
        log(f"mean={mean_value:.6f}")

        # Save next to model dir; use a stable file stem
        save_matrix(S, ospj(model_dir, "dummy_marker_path"), csv_filename=f'matrix_{num}')

        if args.heatmap:
            heatmap_path = ospj(model_dir, f"matrix_{num}_heatmap.png")
            save_heatmap(
                S,
                heatmap_path,
                title=f"{model_type} cosine similarity (model {num})",
                vmin=-1.0, vmax=1.0,
                cmap=args.cmap,
                dpi=300,
                cell_px=args.cell_px,
                max_width_px=args.max_width_px,
                tick_step=None,
                add_colorbar=True,
                downsample_block=args.downsample_block,
            )


if __name__ == '__main__':
    # Defaulting to no specific model here as main handles loading now
    main()
