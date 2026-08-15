"""
Compute retrieval metrics (mAP, Recall@1/5/10) for all already-completed fine-tuning
runs, for both ALIGN and CLIP, and write one consolidated summary table.

This reuses the checkpoints already saved under saved_models/<family>/<split>/<num>/ —
no retraining needed. For every run it computes:

  - QbS (query-by-string / text->image): for each test image's word, build the same
    kind of text query the run was actually *trained* on (plain word, PHOSC
    description, or PHOSC number-description -- read from that run's own
    model_args.toml), rank all test images by cosine similarity, and score against
    the true word labels.
  - QbE (query-by-example / image->image): for each test image, rank all other test
    images by cosine similarity and score against the true word labels. (The scoring
    logic already existed in matrix_new.py as compute_qbe_map_recall but was never
    wired into anything -- this script uses it.)

Output:
  - results/retrieval_metrics.csv  (one row per run, both families)
  - results/retrieval_metrics.md   (same table, as Markdown, ready to paste into the thesis)
  - saved_models/<family>/<split>/<num>/matrix_{num}_report.txt (per-run report, same
    convention as matrix_new.py, extended with QbE numbers)

A run that fails to load (missing checkpoint, missing optional dependency, shape
mismatch, ...) is skipped with a warning and recorded with an 'error' column in the
CSV instead of crashing the whole batch.
"""
from __future__ import annotations

import argparse
import os
import traceback
from os.path import join as ospj
from typing import List, Tuple

import numpy as np
import pandas as pd
import toml
import torch
import torch.nn.functional as F
from tqdm import tqdm

from flags import DATA_FOLDER, device
from data.dataset_bengali import ImageLoader
from utils.get_dataset import get_test_loader
from modules.utils import set_phos_version, set_phoc_version
from modules.utils.utils import get_phosc_description, get_phosc_number_description

from parser import (
    dataset_argparse,
    phosc_net_argparse,
    aling_fine_tune_argparse,
    clip_fine_tune_argparse,
    loss_func_argparse,
    training_common_argparse,
    retrieval_metrics_argparse,
)

from transformers import AlignModel, AutoProcessor, AutoTokenizer


# --------------------------------------------------------------------------------------
# Text-query construction, mirroring the branching in align_fine_tune.py's
# train_epoch/validate_epoch, so each run is evaluated with the same kind of prompt it
# was fine-tuned on.
# --------------------------------------------------------------------------------------

def build_text_queries(words: List[str], description_mode: str) -> List[str]:
    if description_mode == 'word':
        return list(words)
    elif description_mode == 'description':
        return [get_phosc_description(w) for w in words]
    elif description_mode == 'phosc_number':
        return [get_phosc_number_description(w) for w in words]
    elif description_mode == 'description_long':
        return [get_phosc_description(w) for w in words]
    else:
        raise ValueError(f'Unknown description mode: {description_mode!r}')


# --------------------------------------------------------------------------------------
# Feature collection (same approach as matrix_new.py: own PIL loading + own
# model-specific preprocessing, since the dataset loader only yields image paths).
# --------------------------------------------------------------------------------------

def _unwrap_align_embeds(output):
    """
    transformers' AlignModel.get_image_features()/get_text_features() have changed
    return type across versions: older releases return the pooled embedding tensor
    directly; newer releases (observed with transformers>=5) return a
    BaseModelOutputWithPooling(...) and the embedding is at `.pooler_output`. Handle
    both so this script isn't pinned to one transformers version.
    """
    if torch.is_tensor(output):
        return output
    if getattr(output, 'pooler_output', None) is not None:
        return output.pooler_output
    if getattr(output, 'image_embeds', None) is not None:
        return output.image_embeds
    if getattr(output, 'text_embeds', None) is not None:
        return output.text_embeds
    raise TypeError(f'Unrecognized ALIGN feature output type: {type(output)}')


@torch.no_grad()
def collect_image_features_align(model, dataloader, image_loader: ImageLoader) -> Tuple[torch.Tensor, List[str]]:
    processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
    model.eval()
    feats, labels = [], []

    for batch in tqdm(dataloader, desc="ALIGN: collecting image features"):
        *_, img_paths, _, words = batch
        pil_images = [image_loader(p) for p in img_paths]
        proc = processor(images=pil_images, return_tensors="pt")
        proc = {k: v.to(device) for k, v in proc.items()}

        f = _unwrap_align_embeds(model.get_image_features(**proc))
        f = F.normalize(f, dim=-1)
        feats.append(f)
        labels.extend(list(words))

    return torch.cat(feats, dim=0), labels


@torch.no_grad()
def collect_text_features_align(model, texts: List[str]) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
    model.eval()
    out = []
    B = 256
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        tok = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        f = _unwrap_align_embeds(model.get_text_features(**tok))
        f = F.normalize(f, dim=-1)
        out.append(f)
    return torch.cat(out, dim=0)


@torch.no_grad()
def collect_image_features_clip(model, dataloader, image_loader: ImageLoader, preprocess) -> Tuple[torch.Tensor, List[str]]:
    model.eval()
    feats, labels = [], []

    for batch in tqdm(dataloader, desc="CLIP: collecting image features"):
        *_, img_paths, _, words = batch
        pil_images = [image_loader(p) for p in img_paths]
        images = torch.stack([preprocess(img) for img in pil_images]).to(device)

        f = model.encode_image(images)
        f = F.normalize(f, dim=-1)
        feats.append(f)
        labels.extend(list(words))

    return torch.cat(feats, dim=0), labels


@torch.no_grad()
def collect_text_features_clip(model, texts: List[str]) -> torch.Tensor:
    import clip as openai_clip  # lazy: only required for CLIP runs
    model.eval()
    out = []
    B = 256
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        # CLIP's tokenizer truncates at 77 tokens; PHOSC descriptions can exceed that.
        text_tokens = openai_clip.tokenize(chunk, truncate=True).to(device)
        f = model.encode_text(text_tokens)
        f = F.normalize(f, dim=-1)
        out.append(f)
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------------------
# Retrieval scoring (ported from matrix_new.py, unmodified logic).
# --------------------------------------------------------------------------------------

def average_precision_from_ranking(relevant_mask_sorted: np.ndarray) -> float:
    if relevant_mask_sorted.size == 0:
        return 0.0
    rel_idx = np.flatnonzero(relevant_mask_sorted)
    if rel_idx.size == 0:
        return 0.0
    precisions = []
    for r_i in rel_idx:
        k = r_i + 1
        precisions.append(relevant_mask_sorted[:k].sum() / k)
    return float(np.mean(precisions))


def recall_at_k_from_ranking(relevant_mask_sorted: np.ndarray, k: int) -> float:
    if relevant_mask_sorted.size == 0:
        return 0.0
    total_rel = relevant_mask_sorted.sum()
    if total_rel == 0:
        return 0.0
    k = min(k, relevant_mask_sorted.size)
    return float(relevant_mask_sorted[:k].sum() / total_rel)


@torch.no_grad()
def compute_qbe_map_recall(feats: torch.Tensor, labels: List[str], ks: Tuple[int, ...]) -> dict:
    """Image->image: for each query image, rank all other images by similarity."""
    feats = F.normalize(feats, dim=-1)
    S = (feats @ feats.T).detach().cpu().numpy()
    N = S.shape[0]
    labels_np = np.array(labels)

    ap_list = []
    recall_lists = {k: [] for k in ks}

    for i in range(N):
        sims = S[i].copy()
        sims[i] = -np.inf  # exclude self
        order = np.argsort(-sims)
        rel = (labels_np[order] == labels_np[i])

        ap_list.append(average_precision_from_ranking(rel))
        for k in ks:
            recall_lists[k].append(recall_at_k_from_ranking(rel, k))

    out = {"mAP": float(np.mean(ap_list)) if ap_list else 0.0}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recall_lists[k])) if recall_lists[k] else 0.0
    return out


@torch.no_grad()
def compute_qbs_map_recall(text_feats: torch.Tensor, image_feats: torch.Tensor,
                            text_labels: List[str], image_labels: List[str],
                            ks: Tuple[int, ...]) -> dict:
    """Text->image: for each text query, rank all images by similarity."""
    text_feats = F.normalize(text_feats, dim=-1)
    image_feats = F.normalize(image_feats, dim=-1)

    S = (text_feats @ image_feats.T).detach().cpu().numpy()
    text_labels_np = np.array(text_labels)
    image_labels_np = np.array(image_labels)

    ap_list = []
    recall_lists = {k: [] for k in ks}

    for i in range(S.shape[0]):
        order = np.argsort(-S[i])
        rel = (image_labels_np[order] == text_labels_np[i])

        ap_list.append(average_precision_from_ranking(rel))
        for k in ks:
            recall_lists[k].append(recall_at_k_from_ranking(rel, k))

    out = {"mAP": float(np.mean(ap_list)) if ap_list else 0.0}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recall_lists[k])) if recall_lists[k] else 0.0
    return out


# --------------------------------------------------------------------------------------
# Per-run report (same convention as matrix_new.py's matrix_{num}_report.txt).
# --------------------------------------------------------------------------------------

def make_tee_logger(txt_path: str):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    def log(msg: str = ""):
        print(msg)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log


def read_model_args(model_dir: str) -> dict:
    args_path = ospj(model_dir, 'model_args.toml')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f'No model_args.toml found in {model_dir}')
    return toml.load(args_path)


def extract_hparams(model_args: dict) -> dict:
    """Flatten the bits of model_args.toml we want in the summary table."""
    dataset_args = model_args.get('Dataset arguments', {})
    dataloader_args = model_args.get('Dataloader arguments', {})
    common_args = model_args.get('Common training arguments', {})
    optim_args = model_args.get('Optimizer arguments', {})
    lr_args = model_args.get('Learning rate scheduler arguments', {})
    loss_args = model_args.get('Loss function arguments', {})

    batch_size = dataloader_args.get('batch_size')
    accumulation_steps = common_args.get('accumulation_steps')
    effective_batch_size = None
    if batch_size is not None and accumulation_steps is not None:
        effective_batch_size = batch_size * accumulation_steps

    return {
        'description_mode': common_args.get('description'),
        'loss_func': loss_args.get('loss_func'),
        'optimizer': optim_args.get('optimizer'),
        'lr_scheduler': lr_args.get('lr_scheduler'),
        'lr': model_args.get('Training arguments', {}).get('lr'),
        'batch_size': batch_size,
        'accumulation_steps': accumulation_steps,
        'effective_batch_size': effective_batch_size,
        'augmented': dataset_args.get('augmented'),
    }


# --------------------------------------------------------------------------------------
# Per-run evaluation
# --------------------------------------------------------------------------------------

def evaluate_align_run(model_dir: str, checkpoint_name: str, test_loader, image_loader: ImageLoader,
                        description_mode: str, ks: Tuple[int, ...], compute_qbe: bool) -> dict:
    model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()
    ckpt_path = ospj(model_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)

    img_feats, img_labels = collect_image_features_align(model, test_loader, image_loader)
    text_queries = build_text_queries(img_labels, description_mode)
    text_feats = collect_text_features_align(model, text_queries)

    result = {'n_test_images': len(img_labels), 'n_text_queries': len(text_queries)}
    qbs = compute_qbs_map_recall(text_feats, img_feats, img_labels, img_labels, ks)
    result.update({f'qbs_{k}': v for k, v in qbs.items()})

    if compute_qbe:
        qbe = compute_qbe_map_recall(img_feats, img_labels, ks)
        result.update({f'qbe_{k}': v for k, v in qbe.items()})

    del model
    torch.cuda.empty_cache()
    return result


def evaluate_clip_run(model_dir: str, checkpoint_name: str, test_loader, image_loader: ImageLoader,
                       description_mode: str, ks: Tuple[int, ...], compute_qbe: bool) -> dict:
    import clip as openai_clip  # lazy: whole run family is skipped cleanly if unavailable

    model, preprocess = openai_clip.load("ViT-B/32", device=device)
    model = model.float().eval()

    ckpt_path = ospj(model_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)

    img_feats, img_labels = collect_image_features_clip(model, test_loader, image_loader, preprocess)
    text_queries = build_text_queries(img_labels, description_mode)
    text_feats = collect_text_features_clip(model, text_queries)

    result = {'n_test_images': len(img_labels), 'n_text_queries': len(text_queries)}
    qbs = compute_qbs_map_recall(text_feats, img_feats, img_labels, img_labels, ks)
    result.update({f'qbs_{k}': v for k, v in qbs.items()})

    if compute_qbe:
        qbe = compute_qbe_map_recall(img_feats, img_labels, ks)
        result.update({f'qbe_{k}': v for k, v in qbe.items()})

    del model
    torch.cuda.empty_cache()
    return result


def write_per_run_report(model_dir: str, num: int, family: str, hparams: dict, metrics: dict):
    report_path = ospj(model_dir, f"matrix_{num}_report.txt")
    if os.path.exists(report_path):
        os.remove(report_path)
    log = make_tee_logger(report_path)

    log(f"Model family: {family}  |  run: {num}")
    log(f"description_mode={hparams.get('description_mode')}  loss_func={hparams.get('loss_func')}  "
        f"optimizer={hparams.get('optimizer')}  lr_scheduler={hparams.get('lr_scheduler')}  "
        f"effective_batch_size={hparams.get('effective_batch_size')}")
    log(f"n_test_images={metrics.get('n_test_images')}  n_text_queries={metrics.get('n_text_queries')}")
    log(f"[QbS text->image] mAP={metrics.get('qbs_mAP', float('nan')):.4f}  "
        f"R@1={metrics.get('qbs_Recall@1', float('nan')):.4f}  "
        f"R@5={metrics.get('qbs_Recall@5', float('nan')):.4f}  "
        f"R@10={metrics.get('qbs_Recall@10', float('nan')):.4f}")
    if 'qbe_mAP' in metrics:
        log(f"[QbE image->image] mAP={metrics.get('qbe_mAP', float('nan')):.4f}  "
            f"R@1={metrics.get('qbe_Recall@1', float('nan')):.4f}  "
            f"R@5={metrics.get('qbe_Recall@5', float('nan')):.4f}  "
            f"R@10={metrics.get('qbe_Recall@10', float('nan')):.4f}")


# --------------------------------------------------------------------------------------
# Consolidated table writers
# --------------------------------------------------------------------------------------

def write_markdown_table(df: pd.DataFrame, path: str):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f'{v:.4f}' if not np.isnan(v) else '')
            else:
                cells.append('' if v is None else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser()
    parser = dataset_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = aling_fine_tune_argparse(parser)
    parser = clip_fine_tune_argparse(parser)
    parser = loss_func_argparse(parser)
    parser = training_common_argparse(parser)
    parser = retrieval_metrics_argparse(parser)

    args = parser.parse_args(args) if args is not None else parser.parse_args()

    ks = tuple(args.recall_ks)

    os.makedirs(args.out_dir, exist_ok=True)

    root_dir = ospj(DATA_FOLDER, "BengaliWords_CroppedVersion_Folds")
    image_loader = ImageLoader(ospj(root_dir, args.split_name))

    # phosc description generation needs the right character/shape tables loaded
    set_phos_version(args.phosc_version)
    set_phoc_version(args.phosc_version)

    # test loader is model-agnostic (yields paths + words); build it once and reuse
    # across every run/checkpoint.
    test_loader, _ = get_test_loader(args, None)

    families = []
    if not args.skip_align:
        families.append(('ALIGN', args.align_save_name, args.align_nums, evaluate_align_run))
    if not args.skip_clip:
        families.append(('CLIP', args.clip_save_name, args.clip_nums, evaluate_clip_run))

    rows = []

    for family_label, save_name, nums, eval_fn in families:
        for num in nums:
            model_dir = ospj(args.save_dir, save_name, args.split_name, str(num))
            row = {'model_family': family_label, 'run_number': num, 'model_dir': model_dir}

            if not os.path.isdir(model_dir):
                print(f'[skip] {model_dir} does not exist')
                row['error'] = 'run directory not found'
                rows.append(row)
                continue

            try:
                model_args = read_model_args(model_dir)
                hparams = extract_hparams(model_args)
                row.update(hparams)

                print(f'\n=== {family_label} run {num} ({model_dir}) '
                      f'description={hparams.get("description_mode")} ===')

                metrics = eval_fn(
                    model_dir=model_dir,
                    checkpoint_name=args.checkpoint_name,
                    test_loader=test_loader,
                    image_loader=image_loader,
                    description_mode=hparams.get('description_mode') or 'word',
                    ks=ks,
                    compute_qbe=not args.skip_qbe,
                )
                row.update(metrics)

                if args.write_per_run_report:
                    write_per_run_report(model_dir, num, family_label, hparams, metrics)

            except Exception as e:
                print(f'[error] {family_label} run {num} failed: {e}')
                traceback.print_exc()
                row['error'] = str(e)

            rows.append(row)

    df = pd.DataFrame(rows)

    # stable, readable column order
    preferred_order = [
        'model_family', 'run_number', 'description_mode', 'loss_func', 'optimizer',
        'lr_scheduler', 'lr', 'batch_size', 'accumulation_steps', 'effective_batch_size',
        'augmented', 'n_test_images', 'n_text_queries',
        'qbs_mAP',
    ] + [f'qbs_Recall@{k}' for k in ks] + ['qbe_mAP'] + [f'qbe_Recall@{k}' for k in ks] + [
        'error', 'model_dir',
    ]
    ordered_cols = [c for c in preferred_order if c in df.columns] + \
                   [c for c in df.columns if c not in preferred_order]
    df = df[ordered_cols]

    csv_path = ospj(args.out_dir, 'retrieval_metrics.csv')
    md_path = ospj(args.out_dir, 'retrieval_metrics.md')
    df.to_csv(csv_path, index=False)
    write_markdown_table(df, md_path)

    print(f'\nWrote {csv_path}')
    print(f'Wrote {md_path}')
    print(df.to_string(index=False))

    return df


if __name__ == '__main__':
    main()
