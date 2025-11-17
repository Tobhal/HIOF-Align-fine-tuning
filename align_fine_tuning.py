# Standard library imports
import contextlib
import logging
import math
import os
import random
import sys
from enum import Enum
from os.path import join as ospj
from typing import Callable, Tuple
import math

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from torchvision import transforms
from timm import create_model
from tqdm import tqdm
from transformers import AlignProcessor, AlignModel, AutoTokenizer
import argparse
import pynvml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Module

# Local application/library specific imports
from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader
from flags import DATA_FOLDER, device
from modules import models, residualmodels
from modules.utils import set_phos_version, set_phoc_version, gen_shape_description, gen_shape_description_simple
# from train_clip.utils.clip_utils import gen_word_objs_embpeddings
from utils.dbe import dbe
from utils.early_stopping import EarlyStopping
from parser import phosc_net_argparse, dataset_argparse, early_stopper_argparse, aling_fine_tune_argparse, \
    optimizer_argparse, lr_scheduler_argparse, checkpoint_argparse, slurm_argparse
from utils.utils import load_args
from utils.get_dataset import get_training_loader, get_validation_loader, get_test_loader, get_phoscnet
from utils.loss_functions import compute_triplet_margin_loss, compute_contrastive_loss, simple_loss, \
    triplet_margin_from_similarity
from utils.lamb_optimizer import Lamb
from modules.utils.utils import get_phosc_description, get_phosc_number_description
from utils.lr_schedulers.exploration import ExplorationOptimizationScheduler
from utils.checkpoint import save_checkpoint, load_checkpoint

pynvml.nvmlInit()


def create_file_with_job_id(save_path: os.PathLike, job_id: int, job_description: str):
    with open(ospj(save_path, f'slurm_{job_id}.txt'), 'w') as file:
        file.write(job_description)


def get_gpu_memory_usage():
    """Get current and peak (max) memory usage of the GPU in gigabytes, including total system GPU memory usage."""
    device_id = 0  # Assuming we're querying the first GPU
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    allocated_memory = torch.cuda.memory_allocated(device_id)  # current memory allocated by PyTorch
    peak_memory = torch.cuda.max_memory_allocated(device_id)  # peak memory allocated by PyTorch
    total_memory = torch.cuda.get_device_properties(device_id).total_memory  # total memory of GPU

    # System-wide GPU memory usage
    system_allocated_memory = info.used  # total memory used by all processes

    # Convert bytes to gigabytes
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    peak_memory_gb = peak_memory / (1024 ** 3)
    total_memory_gb = total_memory / (1024 ** 3)
    system_allocated_memory_gb = system_allocated_memory / (1024 ** 3)

    return {
        'Allocated Memory (GB)': allocated_memory_gb,
        'Peak Memory (GB)': peak_memory_gb,
        'Total Memory (GB)': total_memory_gb,
        'System Allocated Memory (GB)': system_allocated_memory_gb
    }


def preprocess_align_batch(processor, images, texts):
    # Our on-disk images are already RGB, square, 224×224.
    # Disable extra resize/crop; keep ImageNet-style normalization.
    return processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
        # images_kwargs={
        #     "do_resize": False,
        #     "do_center_crop": False,
        #     "do_normalize": True,   # EfficientNet defaults
        # },
    )


def freeze_text_tower(model):
    if hasattr(model, "text_model"):
        for p in model.text_model.parameters():
            p.requires_grad = False
    if hasattr(model, "text_projection"):
        model.text_projection.requires_grad_(False)


def unfreeze_top_text_layers(model, n_blocks=4):
    tm = getattr(model, "text_model", None)
    if tm is None:
        return
    # Try common HF layouts
    blocks = None
    if hasattr(tm, "encoder") and hasattr(tm.encoder, "layer"):  # BERT/Roberta-like
        blocks = list(tm.encoder.layer)
    elif hasattr(tm, "transformer") and hasattr(tm.transformer, "h"):  # GPT2-like
        blocks = list(tm.transformer.h)

    # (re)freeze all, then unfreeze top-N
    if blocks:
        for p in tm.parameters():
            p.requires_grad = False
        for layer in blocks[-n_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True

    if hasattr(model, "text_projection"):
        model.text_projection.requires_grad_(True)
    # keep LayerNorms trainable (optional)
    for mod in tm.modules():
        if isinstance(mod, torch.nn.LayerNorm):
            for p in mod.parameters():
                p.requires_grad = True


def build_optimizer_with_param_groups(model, lr_img=5e-5, lr_text=1e-5, wd=0.01):
    img_params, txt_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(("text_model", "text_projection")):
            txt_params.append(p)
        else:
            img_params.append(p)
    return torch.optim.AdamW(
        [{"params": img_params, "lr": lr_img, "weight_decay": wd},
         {"params": txt_params, "lr": lr_text, "weight_decay": wd}],
        betas=(0.9, 0.98), eps=1e-8
    )


# Function to check if the model save path's directory exists
def verify_model_save_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        print(f"Model save directory does not exist, creating: {directory}")
        os.makedirs(directory)


def create_learning_rate_fn(
        optimizer: Optimizer,
        train_ds_size: int,
        train_batch_size: int,
        num_train_epochs: int,
        num_warmup_steps: int,
        learning_rate: float,
        linear=False
):
    """Returns a PyTorch learning rate scheduler."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if linear:
            return max(
                0.0, float(num_train_steps - current_step) / float(max(1, num_train_steps - num_warmup_steps))
            )
        else:  # Cosine decay
            return 0.5 * (1 + np.cos(np.pi * (current_step - num_warmup_steps) / (num_train_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def enable_logit_scale_training(model, max_temp=100.0):
    # ALIGN/CLIP usually expose a learnable logit_scale (log temperature).
    if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.requires_grad_(True)

        # optional: clamp logit_scale in-place after each optimizer.step()
        max_log = math.log(max_temp)  # e.g., log(100)

        def clamp_hook(_):
            with torch.no_grad():
                model.logit_scale.clamp_(0, max_log)

        return clamp_hook  # call this after each step
    return None


def create_cosine_annealing_lr_scheduler(
        optimizer: Optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1
):
    """Returns a PyTorch Cosine Annealing scheduler with Warm Restarts."""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
    )


def make_accum_scheduler(optimizer, len_train_loader, accumulation_steps, num_epochs,
                         warmup_steps=0, linear=False):
    """LambdaLR that expects one .step() per *optimizer step* (i.e., per accumulation boundary)."""
    steps_per_epoch = math.ceil(len_train_loader / accumulation_steps)
    num_train_steps = steps_per_epoch * num_epochs

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if linear:
            return max(0.0, float(num_train_steps - current_step) /
                       float(max(1, num_train_steps - warmup_steps)))
        # cosine
        progress = (current_step - warmup_steps) / float(max(1, num_train_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def custom_loss(image_features, text_features):
    # Assuming image_features and text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
    loss = torch.mean(1 - similarity)  # Penalize high similarity
    return loss


def normalize_features(features):
    return features / features.norm(dim=1, keepdim=True)


# Cross entropy helper function
def cross_entropy(logits, axis):
    logprobs = torch.log_softmax(logits, axis=axis)
    nll = torch.diag(logprobs)
    ce = -torch.mean(nll)
    return ce


def custom_loss_same_class(anchor_image_features, positive_text_features):
    # Ensure features are normalized
    image_features = F.normalize(anchor_image_features, dim=1)
    text_features = F.normalize(positive_text_features, dim=1)

    # Calculate similarity
    similarity = torch.matmul(image_features, text_features.T)

    # Compute CLIP loss
    loss = -((cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2)
    return loss


def custom_loss_different_class(anchor_image_features, negative_text_features):
    # Ensure features are normalized
    image_features = F.normalize(anchor_image_features, dim=1)
    text_features = F.normalize(negative_text_features, dim=1)

    # Calculate similarity
    similarity = torch.matmul(image_features, text_features.T)

    # Compute CLIP loss
    loss = 1 - ((cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2)
    return loss


def custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features, margin=1.0):
    # Calculate triplet loss
    loss = F.triplet_margin_loss(
        anchor_image_features,
        positive_text_features,
        negative_text_features,
        margin=margin
    )
    return loss


def supcon_infonce_from_logits(logits_per_image, logits_per_text, labels, symmetric=True):
    """
labels: LongTensor [B] with class/word ids. Supports multiple positives per class.
    """

    def _supcon(logits, labels):
        B = logits.size(0)
        device = logits.device
        # mask of positives (exclude self on diagonal)
        pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))  # [B,B] bool
        pos.fill_diagonal_(False)
        # log-softmax over rows
        log_prob = logits.log_softmax(dim=1)  # [B,B]
        # average log-prob over all positives per row (avoid div by 0)
        pos_counts = pos.sum(dim=1).clamp_min(1)
        loss = -(log_prob * pos).sum(dim=1) / pos_counts
        return loss.mean()

    # Temperature is already baked into ALIGN's logits; if you want a custom tau:
    # logits = logits / tau
    loss_i = _supcon(logits_per_image, labels)
    loss_t = _supcon(logits_per_text, labels) if symmetric else 0.0
    return 0.5 * (loss_i + loss_t) if symmetric else loss_i


class Loss_method(Enum):
    DIFFRENT_SAME = 1
    CUSTOM_TRIPLET_LOSS = 2


def calc_loss(anchor_image_features, positive_text_features, negative_text_features, is_same_class: bool,
              loss_method: Loss_method):
    if loss_method == Loss_method.DIFFRENT_SAME:
        if is_same_class:
            return custom_loss_same_class(anchor_image_features, positive_text_features)
        else:
            return custom_loss_different_class(anchor_image_features, negative_text_features)

    elif loss_method == Loss_method.CUSTOM_TRIPLET_LOSS:
        return custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features)


def train_epoch(
        epoch: int,
        train_loader: DataLoader,
        model: Module,
        processor,
        image_loader: ImageLoader,
        loss_func: str,
        optimizer: Optimizer,
        save_path: str,
        lr_scheduler: _LRScheduler = None,
        margin=1.0,
        accumulation_steps=4,
        description='word'
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_batches = len(train_loader)
    micro_loss_accum = 0.0
    running_loss = 0.0
    num_steps = 0

    for i, batch in enumerate(train_loader):
        *_, image_names, _, words = batch
        images = [image_loader(img_name) for img_name in image_names]

        if description == 'word':
            descriptions = words
        elif description == 'description':
            descriptions = [get_phosc_description(w) for w in words]
        elif description == 'phosc_number':
            descriptions = [get_phosc_number_description(w) for w in words]
        else:
            raise ValueError('Invalid description')

        # one-off example file
        example_path = ospj(save_path, 'description_example.txt')
        if not os.path.exists(example_path):
            with open(example_path, 'w') as f:
                f.write(f'{words[0]}\n{descriptions[0]}')

        uniq = sorted(set(words))
        w2i = {w: idx for idx, w in enumerate(uniq)}
        class_labels = torch.tensor([w2i[w] for w in words], device=device, dtype=torch.long)

        inputs = preprocess_align_batch(processor, images, descriptions)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        if loss_func == 'triplet':
            # loss = compute_triplet_margin_loss(outputs.logits_per_image, class_labels, margin)
            loss = triplet_margin_from_similarity(outputs.logits_per_image, class_labels, margin)
        elif loss_func == 'contrastive':
            loss = compute_contrastive_loss(outputs.logits_per_image, class_labels, margin)
        elif loss_func == 'simple':
            loss = simple_loss(outputs.logits_per_image)
        elif loss_func == 'supcon':
            loss = supcon_infonce_from_logits(outputs.logits_per_image, outputs.logits_per_text,
                                              class_labels, symmetric=True)
        else:
            raise ValueError('Invalid loss function')

        # accumulate micro-batch losses for accurate logging
        micro_loss_accum += float(loss.item())

        (loss / accumulation_steps).backward()

        boundary = ((i + 1) % accumulation_steps == 0) or ((i + 1) == total_batches)
        if boundary:
            optimizer.step()

            if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
                with torch.no_grad():
                    model.logit_scale.clamp_(0, math.log(100.0))

            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None:
                lr_scheduler.step()

            running_loss += micro_loss_accum / accumulation_steps  # average over the micro-batches we just consumed
            micro_loss_accum = 0.0
            num_steps += 1

    return running_loss / max(1, num_steps)


def validate_epoch(
        epoch: int,
        val_loader: DataLoader,
        model: Module,
        processor,
        image_loader: ImageLoader,
        loss_func: str,
        save_path: str,
        margin=1.0,
        description='word',
        use_amp: bool = False,  # optional speedup
):
    model.eval()
    model.to(device)

    total_loss = 0.0
    num_batches = 0

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp and torch.cuda.is_available()
        else contextlib.nullcontext()
    )

    with torch.no_grad(), autocast_ctx:
        for batch in val_loader:
            *_, image_names, _, words = batch

            # Build images + descriptions
            images = [image_loader(img_name) for img_name in image_names]

            if description == 'word':
                descriptions = words
            elif description == 'description':
                descriptions = [get_phosc_description(w) for w in words]
            elif description == 'phosc_number':  # <-- add parity with train
                descriptions = [get_phosc_number_description(w) for w in words]
            else:
                raise ValueError('Invalid description')

            # Deterministic label mapping (avoid unordered set)
            uniq = sorted(set(words))
            w2i = {w: i for i, w in enumerate(uniq)}
            class_labels = torch.tensor([w2i[w] for w in words], device=device, dtype=torch.long)

            # Forward
            inputs = preprocess_align_batch(processor, images, descriptions)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Loss
            logits_per_image = outputs.logits_per_image
            if loss_func == 'triplet':
                loss = compute_triplet_margin_loss(logits_per_image, class_labels, margin)
            elif loss_func == 'contrastive':
                loss = compute_contrastive_loss(logits_per_image, class_labels, margin)
            elif loss_func == 'simple':
                loss = simple_loss(logits_per_image)
            elif loss_func == 'supcon':
                loss = supcon_infonce_from_logits(outputs.logits_per_image,
                                                  outputs.logits_per_text,
                                                  class_labels,
                                                  symmetric=True)
            else:
                raise ValueError('Invalid loss function')

            total_loss += float(loss.item())
            num_batches += 1

    return total_loss / max(1, num_batches)


def main(_args=None):
    print(f'{device}')

    parser = argparse.ArgumentParser()

    parser = aling_fine_tune_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = early_stopper_argparse(parser)
    parser = optimizer_argparse(parser)
    parser = lr_scheduler_argparse(parser)
    parser = checkpoint_argparse(parser)
    parser = slurm_argparse(parser)

    # Parse arguments
    if _args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(_args)

    # align model
    align_model = AlignModel.from_pretrained("kakaobrain/align-base")
    align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
    align_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

    align_model.to(device)

    # Make logit_scale trainable (ALIGN/CLIP usually expose this as a Parameter)
    if hasattr(align_model, "logit_scale") and isinstance(align_model.logit_scale, torch.nn.Parameter):
        align_model.logit_scale.requires_grad_(True)

    # Load phosc model
    phosc_model = get_phoscnet(args, device)

    train_loader, train_set = get_training_loader(args, phosc_model)
    validation_loader, _ = get_validation_loader(args, phosc_model)
    # test_loader, _ = get_test_loader(args, phosc_model)

    image_loader = ImageLoader(ospj(DATA_FOLDER, args.data_dir, args.split_name))

    optimizer = None
    lr_scheduler = None

    save_path = ospj(args.save_dir, args.name, args.split_name)

    print(f'{args.maximize=}')

    # Select optimizer
    if args.optimizer == 'lamb' or args.optimizer == 'adam':
        optimizer = Lamb(
            align_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            adam=True if args.optimizer == 'adam' else False,
            maximize=args.maximize,
        )
    elif args.optimizer == 'none':
        optimizer = None
    else:
        raise ValueError('Invalid optimizer')

    # Select learning rate scheduler
    if args.lr_scheduler == 'cosine':
        lr_scheduler = create_cosine_annealing_lr_scheduler(optimizer, T_0=10)
    elif args.lr_scheduler == 'cosine_warmup':
        lr_scheduler = create_learning_rate_fn(
            optimizer,
            len(train_set),
            args.batch_size,
            args.epochs,
            args.warmup_steps,
            args.lr,
            linear=False,  # set False to activate cosine annealing
        )
    elif args.lr_scheduler == 'linear':
        lr_scheduler = create_learning_rate_fn(
            optimizer,
            len(train_set),
            args.batch_size,
            args.epochs,
            args.warmup_steps,
            args.lr,
            linear=True,
        )
    elif args.lr_scheduler == 'exploration':
        lr_scheduler = ExplorationOptimizationScheduler(
            optimizer,
            patience=args.lr_patience,
            threshold=args.lr_threshold,
            reduction_factor=args.lr_reduction_factor,
            exploration_factor=args.lr_exploration_factor,
        )
    elif args.lr_scheduler == 'none':
        lr_scheduler = None

    # Load checkpoint if there is any
    if args.ignore_checkpoint:
        start_epoch = 1
        best_loss = float('-inf') if args.maximize else float('inf')
    else:
        print('Loading checkpoint')
        checkpoint_path = save_path if args.checkpoint_path == None else args.checkpoint_path
        start_epoch, best_loss = load_checkpoint(checkpoint_path, align_model, optimizer, lr_scheduler,
                                                 maximize=args.maximize)

    early_stopping = EarlyStopping(
        save_path=save_path,
        loss=best_loss,
        patience=args.stop_patience,
        verbose=args.verbose,
        save_every=args.save_every,
        model_arguments=args,
        model_argument_parser=parser,
        save=args.save,
        maximize=args.maximize,
        validate=args.validate
    )

    # Save slurm job to model folder
    create_file_with_job_id(save_path, args.slurm_job_id, args.slurm_job_desc)

    # before training
    freeze_text_tower(align_model)
    optimizer = build_optimizer_with_param_groups(align_model, lr_img=5e-5, lr_text=1e-5, wd=args.weight_decay)

    # (re)build LR scheduler to point at the *current* optimizer and to account for accumulation
    if args.lr_scheduler in {'cosine_warmup', 'linear'}:
        lr_scheduler = make_accum_scheduler(
            optimizer,
            len_train_loader=len(train_loader),
            accumulation_steps=args.accumulation_steps,
            num_epochs=args.epochs - (start_epoch - 1),
            warmup_steps=args.warmup_steps,
            linear=(args.lr_scheduler == 'linear'),
        )
    elif args.lr_scheduler == 'cosine':
        # CosineAnnealingWarmRestarts works in "steps"; set T_0 in optimizer-steps:
        steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps_per_epoch * 10, T_mult=1, eta_min=0
        )
    elif args.lr_scheduler == 'exploration':
        lr_scheduler = ExplorationOptimizationScheduler(
            optimizer,
            patience=args.lr_patience,
            threshold=args.lr_threshold,
            reduction_factor=args.lr_reduction_factor,
            exploration_factor=args.lr_exploration_factor,
        )
    else:
        lr_scheduler = None

    warmup_epochs = 2  # keep text frozen at start
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == warmup_epochs:
            unfreeze_top_text_layers(align_model, n_blocks=4)
            optimizer = build_optimizer_with_param_groups(align_model, lr_img=5e-5, lr_text=1e-5, wd=args.weight_decay)

            if args.lr_scheduler in {'cosine_warmup', 'linear'}:
                lr_scheduler = make_accum_scheduler(
                    optimizer,
                    len_train_loader=len(train_loader),
                    accumulation_steps=args.accumulation_steps,
                    num_epochs=args.epochs - (epoch - 1),
                    warmup_steps=max(0, args.warmup_steps - epoch),  # simple carry-over
                    linear=(args.lr_scheduler == 'linear'),
                )
            elif args.lr_scheduler == 'cosine':
                steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=steps_per_epoch * 10, T_mult=1, eta_min=0
                )
            elif args.lr_scheduler == 'exploration':
                lr_scheduler = ExplorationOptimizationScheduler(
                    optimizer,
                    patience=args.lr_patience,
                    threshold=args.lr_threshold,
                    reduction_factor=args.lr_reduction_factor,
                    exploration_factor=args.lr_exploration_factor,
                )
            else:
                lr_scheduler = None

        train_loss = train_epoch(
            epoch=epoch,
            train_loader=train_loader,
            model=align_model,
            processor=align_processor,
            image_loader=image_loader,
            loss_func=args.loss_func,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            margin=args.margin,
            accumulation_steps=args.accumulation_steps,
            description=args.description,
            save_path=early_stopping.save_path,
        )

        val_loss = 0

        if args.validate:
            val_loss = validate_epoch(
                epoch=epoch,
                val_loader=validation_loader,
                model=align_model,
                processor=align_processor,
                image_loader=image_loader,
                loss_func=args.loss_func,
                margin=args.margin,
                description=args.description,
                save_path=early_stopping.save_path,
            )

        if early_stopping(train_loss, val_loss, align_model, epoch):
            return early_stopping.min_loss, early_stopping.best_model_path

    return early_stopping.min_loss, early_stopping.best_model_path


if __name__ == '__main__':
    try:
        best_score, best_model_path = main()

        print(f'Best model validation loss: {best_score}')
        print(f'Best model path: {best_model_path}')

    except KeyboardInterrupt:
        print('Ctrl-C exit')
