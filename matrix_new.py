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
from parser import phosc_net_argparse, dataset_argparse, aling_fine_tune_argparse, matrix_new_argparse

# -----------------------
# Globals / Preprocessing
# -----------------------
split = 'fold_0_t'
use_augmented = False

# ALIGN (HF): processor can batch images (and texts). We'll use AutoProcessor for images below.
align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
align_model = AlignModel.from_pretrained("kakaobrain/align-base")
align_auto_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
align_auto_processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

# (Kept for completeness; not used in this matrix script)
clip_preprocess = Compose([
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


# -----------------------------
# Heatmap saver (vectorized)
# -----------------------------
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


# -----------------------
# Data classes / results
# -----------------------
@dataclass
class Result:
    model_number: int
    min_value: float
    max_value: float
    average_value: float


# -----------------------
# I/O: save the matrix
# -----------------------
def save_matrix(matrix: torch.Tensor, results: Result, _model_save_path: PathLike, csv_filename="matrix"):
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

    with open(txt_path, 'w') as f:
        f.write(f"Model number: {results.model_number}\n")
        f.write(f"Minimum value in matrix: {results.min_value}\n")
        f.write(f"Maximum value in matrix: {results.max_value}\n")
        f.write(f"Mean value in matrix: {results.average_value}\n")

    print(f"Matrix saved at: {csv_path}")


# ----------------------------------------
# Cosine matrix helpers (vectorized & fast)
# ----------------------------------------
@torch.no_grad()
def collect_image_features_align(
        model: nn.Module,
        dataloader: DataLoader,
        image_loader: ImageLoader,
        device: str
) -> torch.Tensor:
    """
Encode ALL images from dataloader with ALIGN, L2-normalize, and return [N, D] features.
Assumes batches yield something like: (*_, image_paths, _, _)
    """
    model.eval()
    feats = []
    for batch in tqdm(dataloader, desc="Collecting image features"):
        # Unpack; your loader previously used "... image_names ..." or similar
        try:
            *_, img_paths, _, _ = batch
        except Exception as e:
            raise RuntimeError(
                "Expected batch like (*_, image_paths, _, _). "
                "Adjust unpacking to your Dataset.__getitem__."
            ) from e

        # Load PIL images and batch-process via HF processor
        pil_images = [image_loader(p) for p in img_paths]
        proc = align_auto_processor(images=pil_images, return_tensors="pt")
        proc = {k: v.to(device) for k, v in proc.items()}

        f = model.get_image_features(**proc)  # [B, D]
        f = F.normalize(f, dim=-1)  # unit-norm => dot = cosine
        feats.append(f)

    feats = torch.cat(feats, dim=0)  # [N, D]
    return feats


@torch.no_grad()
def cosine_similarity_matrix(feats: torch.Tensor) -> torch.Tensor:
    """
Compute N×N cosine similarity matrix from L2-normalized features [N, D].
    """
    # feats must be unit-normalized along dim=-1
    return feats @ feats.T  # [N, N], values in [-1, 1]


# -----------------------------
# (Optional) text embeddings
# -----------------------------
@torch.no_grad()
def collect_text_features_align(
        model: nn.Module,
        texts: List[str],
        device: str
) -> torch.Tensor:
    """
Encode a list of texts with ALIGN, L2-normalize, return [N, D].
    """
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


# -----------------------
# Pretty print results
# -----------------------
def print_results(results: List[Result]):
    for r in results:
        print(f"Model {r.model_number}: min={r.min_value:.6f}  max={r.max_value:.6f}  mean={r.average_value:.6f}")


# -----------------------
# Main
# -----------------------
def main(args=None, model=None, index=0) -> List[Result]:
    parser = argparse.ArgumentParser()
    parser = matrix_new_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = aling_fine_tune_argparse(parser)

    # Parse args
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # Dataset / loader
    # NOTE: Adapt this root_dir to your actual project layout
    root_dir = ospj(DATA_FOLDER, "BengaliWords_CroppedVersion_Folds")
    phosc_model = None
    test_loader, _ = get_test_loader(args, phosc_model)
    image_loader = ImageLoader(ospj(root_dir, args.split_name))

    results = []

    for num in args.nums:
        model_dir = ospj(args.save_dir, args.name, args.split_name, str(num))
        ckpt_path = ospj(model_dir, args.checkpoint_name)

        # Load the fine-tuned ALIGN model
        align_fine_tuned_model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()
        if args.model_source == 'fine-tuned':
            state = torch.load(ckpt_path, map_location=device)
            align_fine_tuned_model.load_state_dict(state, strict=True)

        if args.evaluate == 'text':
            # If you really want text–text: collect all words from the loader
            all_words = []
            for batch in tqdm(test_loader, desc="Collecting words"):
                try:
                    *_, _, _, words = batch
                except Exception as e:
                    raise RuntimeError("Expected words in batch at position -1.") from e
                all_words.extend(list(words))

            feats = collect_text_features_align(align_fine_tuned_model, all_words, device)
        else:
            # Default: image–image matrix
            feats = collect_image_features_align(align_fine_tuned_model, test_loader, image_loader, device)

        # Cosine similarity matrix (vectorized)
        S = cosine_similarity_matrix(feats)  # [N, N], cosine in [-1, 1]
        # If you prefer cosine *distance*: D = 1 - S  (bounded in [0, 2] for normalized vecs)

        # Stats
        min_value = torch.min(S).item()
        max_value = torch.max(S).item()
        mean_value = torch.mean(S).item()

        res = Result(model_number=num, min_value=min_value, max_value=max_value, average_value=mean_value)

        # Save next to model dir; use a stable file stem
        save_matrix(S, res, ospj(model_dir, "dummy_marker_path"), csv_filename=f'matrix_{num}')

        if args.heatmap:
            heatmap_path = ospj(model_dir, f"matrix_{num}_heatmap.png")
            save_heatmap(
                S,
                heatmap_path,
                title=f"ALIGN cosine similarity (model {num})",
                vmin=-1.0, vmax=1.0,
                cmap=args.cmap,
                dpi=300,
                cell_px=args.cell_px,
                max_width_px=args.max_width_px,
                tick_step=None,
                add_colorbar=True,
                downsample_block=args.downsample_block,
            )

        results.append(res)

    return results


if __name__ == '__main__':
    results = main(model=align_model)
    print_results(results)
