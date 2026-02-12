import os
from os.path import join as ospj
from typing import List, Dict, Tuple, Optional
from enum import Enum
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip
from transformers import AlignModel, AlignProcessor, AutoTokenizer, AutoProcessor

from flags import DATA_FOLDER, device
from parser import phosc_net_argparse, dataset_argparse, aling_fine_tune_argparse, matrix_argparse
from utils.get_dataset import get_test_loader, get_phoscnet
from data.dataset_bengali import ImageLoader
from modules.utils.utils import get_phosc_description
from utils.utils import clip_text_features_from_description


class ModelType(Enum):
    CLIP = "CLIP"
    ALIGN = "ALIGN"


class EvalTask(Enum):
    TEXT_MATRIX = "text_matrix"   # debug: text-text similarity matrix
    QBS = "qbs"                   # text/prompt -> image retrieval
    QBE = "qbe"                   # image -> image retrieval


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def parse_ks(ks: str) -> List[int]:
    return [int(k.strip()) for k in ks.split(",") if k.strip()]


def average_precision(relevant: np.ndarray) -> float:
    """
    relevant: boolean array of length N, True where a relevant item occurs in the ranked list
    """
    if relevant.sum() == 0:
        return np.nan  # caller decides whether to skip or treat as 0
    idx = np.where(relevant)[0]
    precisions = [(relevant[:i + 1].sum() / (i + 1)) for i in idx]
    return float(np.mean(precisions))


def compute_map_and_recall(
    sim: np.ndarray,
    query_labels: List[str],
    cand_labels: List[str],
    ks: List[int],
    exclude_self: bool = False,
    query_ids: Optional[List[str]] = None,
    cand_ids: Optional[List[str]] = None,
    filter_singletons: bool = False,
) -> Dict[str, float]:
    """
    sim: [Q, N] similarity matrix
    query_labels: labels for queries
    cand_labels: labels for candidates
    exclude_self: for QbE (query in candidate pool), remove same id from ranking
    filter_singletons: skip queries where number of relevant candidates is 0 (after self removal if enabled)
    """
    Q, N = sim.shape
    cand_labels_arr = np.array(cand_labels)

    ap_list = []
    recall_at_k = {k: [] for k in ks}

    for qi in range(Q):
        scores = sim[qi]
        order = np.argsort(-scores)  # descending
        if exclude_self and query_ids is not None and cand_ids is not None:
            # remove the candidate with same id as query
            mask = np.array([cand_ids[j] != query_ids[qi] for j in order], dtype=bool)
            order = order[mask]

        # relevant set (by word identity)
        rel = (cand_labels_arr[order] == query_labels[qi])

        # singleton handling for QbE
        if filter_singletons and rel.sum() == 0:
            continue

        ap = average_precision(rel)
        if not np.isnan(ap):
            ap_list.append(ap)

        # Recall@k
        total_rel = rel.sum()
        for k in ks:
            k_eff = min(k, len(rel))
            if total_rel == 0:
                rk = np.nan
            else:
                rk = float(rel[:k_eff].sum() / total_rel)
            recall_at_k[k].append(rk)

    # aggregate
    metrics = {}
    metrics["mAP"] = float(np.nanmean(ap_list)) if len(ap_list) > 0 else float("nan")
    for k in ks:
        metrics[f"Recall@{k}"] = float(np.nanmean(recall_at_k[k])) if len(recall_at_k[k]) > 0 else float("nan")

    metrics["num_queries_used"] = int(len(ap_list))
    return metrics


# -----------------------
# Embedding extraction
# -----------------------
@torch.no_grad()
def extract_image_embeddings(
    model_type: ModelType,
    model,
    dataloader,
    image_loader: ImageLoader,
    clip_preprocess=None,
    align_processor=None,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Returns:
      feats: [N, D]
      labels: list of word labels (same length N)
      ids: list of image_names (same length N)
    """
    model.eval()
    feats = []
    labels = []
    ids = []

    for batch in tqdm(dataloader, desc="Extracting image embeddings"):
        *_, image_names, _, words = batch  # <-- your batch format

        if model_type == ModelType.CLIP:
            images = [clip_preprocess(image_loader(n)) for n in image_names]
            images = torch.stack(images, dim=0).to(device)
            img_feat = model.encode_image(images)
            img_feat = l2_normalize(img_feat)

        elif model_type == ModelType.ALIGN:
            # process as a batch (processor supports list of PIL images)
            pil_images = [image_loader(n) for n in image_names]
            inputs = align_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_feat = model.get_image_features(**inputs)
            img_feat = l2_normalize(img_feat)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        feats.append(img_feat.cpu())
        labels.extend(list(words))
        ids.extend(list(image_names))

    feats = torch.cat(feats, dim=0)
    return feats, labels, ids


@torch.no_grad()
def extract_text_embeddings(
    model_type: ModelType,
    model,
    dataloader,
    clip_model=None,
    align_tokenizer=None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    QbS queries: create a prompt from word and embed with the text tower.
    Returns:
      feats: [Q, D]
      labels: list of word labels for queries
    """
    model.eval()
    feats = []
    labels = []

    for batch in tqdm(dataloader, desc="Extracting text embeddings (QbS queries)"):
        *_, image_names, _, words = batch  # <-- your batch format
        # We treat each instance as a query by default (simple & consistent with your dataloader).
        # If you later want unique-word queries, we can add --unique_queries.
        if model_type == ModelType.CLIP:
            prompts = [get_phosc_description(w) for w in words]
            # your util likely supports list input; if it doesn't, we can loop.
            txt_feat = clip_text_features_from_description(prompts, clip_model)
            txt_feat = txt_feat.squeeze(1) if txt_feat.dim() == 3 else txt_feat
            txt_feat = l2_normalize(txt_feat)

        elif model_type == ModelType.ALIGN:
            prompts = [get_phosc_description(w) for w in words]
            inputs = align_tokenizer(prompts, padding=True, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            txt_feat = model.get_text_features(**inputs)
            txt_feat = l2_normalize(txt_feat)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        feats.append(txt_feat.cpu())
        labels.extend(list(words))

    feats = torch.cat(feats, dim=0)
    return feats, labels


def build_parser():
    parser = argparse.ArgumentParser()
    parser = matrix_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = aling_fine_tune_argparse(parser)

    parser.add_argument("--task", type=str, default="qbs",
                        choices=[t.value for t in EvalTask],
                        help="Which evaluation to run: qbs, qbe, or text_matrix.")
    parser.add_argument("--model_type", type=str, default="ALIGN",
                        choices=[m.value for m in ModelType],
                        help="Which backbone to evaluate.")
    parser.add_argument("--clip_backbone", type=str, default="ViT-B/32",
                        help="CLIP backbone name if --model_type CLIP.")
    parser.add_argument("--ks", type=str, default="1,5,10",
                        help="Comma-separated k values for Recall@k.")
    parser.add_argument("--exclude_self", action="store_true",
                        help="Exclude self-match for QbE when query images are in candidate pool.")
    parser.add_argument("--filter_singletons", action="store_true",
                        help="Skip queries with no relevant items (recommended for QbE).")
    parser.add_argument("--save_similarity", action="store_true",
                        help="Save similarity matrix as .npy (can be large).")
    parser.add_argument("--save_scores", action="store_true",
                        help="Save metrics to a text file next to the checkpoint.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    task = EvalTask(args.task)
    model_type = ModelType(args.model_type)
    ks = parse_ks(args.ks)

    root_dir = ospj(DATA_FOLDER, "BengaliWords_CroppedVersion_Folds")
    phosc_model = get_phoscnet(args, device)
    test_loader, _ = get_test_loader(args, phosc_model)
    image_loader = ImageLoader(ospj(root_dir, args.split_name))

    clip_model = None
    clip_preprocess = None

    align_processor = None
    align_tokenizer = None

    if model_type == ModelType.CLIP:
        clip_model, clip_preprocess = clip.load(args.clip_backbone, device=device)
        model = clip_model
    else:
        align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        align_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
        model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)

    results = []

    for num in args.nums:
        model_save_path = ospj(args.save_dir, args.name, args.split_name, str(num))
        ckpt_path = ospj(model_save_path, args.checkpoint_name)

        if model_type == ModelType.CLIP:
            state = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
        else:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state)

        model = model.to(device)
        model.eval()

        if task == EvalTask.TEXT_MATRIX:
            text_feats, text_labels = extract_text_embeddings(
                model_type=model_type,
                model=model,
                dataloader=test_loader,
                clip_model=model if model_type == ModelType.CLIP else None,
                align_tokenizer=align_tokenizer,
            )
            text_feats = l2_normalize(text_feats)
            sim = (text_feats @ text_feats.T).cpu().numpy()

            res = {
                "checkpoint": num,
                "min": float(sim.min()),
                "max": float(sim.max()),
                "mean": float(sim.mean()),
            }
            print(res)

            if args.save_similarity:
                np.save(ospj(model_save_path, f"text_matrix_{num}.npy"), sim)

            results.append(res)
            continue

        img_feats, img_labels, img_ids = extract_image_embeddings(
            model_type=model_type,
            model=model,
            dataloader=test_loader,
            image_loader=image_loader,
            clip_preprocess=clip_preprocess,
            align_processor=align_processor,
        )

        if task == EvalTask.QBS:
            txt_feats, txt_labels = extract_text_embeddings(
                model_type=model_type,
                model=model,
                dataloader=test_loader,
                clip_model=model if model_type == ModelType.CLIP else None,
                align_tokenizer=align_tokenizer,
            )
            txt_feats = l2_normalize(txt_feats)
            img_feats = l2_normalize(img_feats)

            sim = (txt_feats @ img_feats.T).cpu().numpy()  # [Q, N]
            metrics = compute_map_and_recall(
                sim=sim,
                query_labels=txt_labels,
                cand_labels=img_labels,
                ks=ks,
                exclude_self=False,
                filter_singletons=False,
            )

        elif task == EvalTask.QBE:
            img_feats = l2_normalize(img_feats)
            sim = (img_feats @ img_feats.T).cpu().numpy()  # [N, N]
            metrics = compute_map_and_recall(
                sim=sim,
                query_labels=img_labels,
                cand_labels=img_labels,
                ks=ks,
                exclude_self=args.exclude_self,
                query_ids=img_ids,
                cand_ids=img_ids,
                filter_singletons=args.filter_singletons,
            )

        else:
            raise ValueError(f"Unknown task: {task}")

        metrics["checkpoint"] = int(num)
        print(metrics)

        if args.save_similarity:
            np.save(ospj(model_save_path, f"sim_{task.value}_{num}.npy"), sim)

        if args.save_scores:
            out_path = ospj(model_save_path, f"metrics_{task.value}_{num}.txt")
            with open(out_path, "w") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")

        results.append(metrics)

    return results


if __name__ == "__main__":
    main()