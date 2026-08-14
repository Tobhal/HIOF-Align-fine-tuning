import torch
import torch.nn.functional as F
from torch import Tensor
from utils.dbe import dbe


def compute_triplet_margin_loss(logits_per_image: Tensor, class_labels: Tensor, margin: float) -> Tensor:
    """
    Computes the triplet margin loss given logits, class labels, and a margin.

    Args:
        logits_per_image: Tensor of shape (batch_size, batch_size) representing
                          logits or similarity scores between all pairs.
        class_labels: Tensor of shape (batch_size,) representing class labels.
        margin: A float representing the margin value used in the loss calculation.

    Returns:
        A scalar tensor representing the mean triplet margin loss.
    """
    same_class_mask = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)

    # Positive pairs
    positive_scores = []
    for i in range(class_labels.size(0)):
        mask = (class_labels == class_labels[i]) & (torch.arange(class_labels.size(0)) != i)

        if mask.any():
            positive_scores.append(logits_per_image[i][mask].max())
        else:
            positive_scores.append(logits_per_image.sum() * 0.0)

    positive_pairs = torch.stack(positive_scores) if positive_scores else (logits_per_image.sum() * 0.0).unsqueeze(0)

    # Negative pairs
    negative_mask = ~same_class_mask

    negative_mask = negative_mask.to(logits_per_image.device)

    max_negative_logits = torch.where(negative_mask, logits_per_image,
                                      torch.tensor(float('-inf')).to(logits_per_image.device))
    negative_pairs = max_negative_logits.max(dim=1)[0]

    # Compute loss
    loss = F.relu((positive_pairs - negative_pairs) + margin).mean()
    return loss


def triplet_margin_from_similarity(S: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
S: (B, B) pairwise similarities (higher = more similar)
labels: (B,) class labels
    """
    B = S.size(0)
    device = S.device

    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))  # (B,B)
    eye = torch.eye(B, dtype=torch.bool, device=device)

    pos_mask = same & ~eye  # exclude self
    neg_mask = ~same

    # hardest positive: MIN similarity among positives
    pos_sim = torch.where(pos_mask, S, torch.full_like(S, float('inf')))
    hardest_pos = pos_sim.min(dim=1).values  # (B,)

    # hardest negative: MAX similarity among negatives
    neg_sim = torch.where(neg_mask, S, torch.full_like(S, float('-inf')))
    hardest_neg = neg_sim.max(dim=1).values  # (B,)

    # keep anchors that actually have at least 1 pos and 1 neg
    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if not valid.any():
        return S.sum() * 0.0

    # similarity-based triplet: max(0, margin + s_neg - s_pos)
    loss = F.relu(margin + hardest_neg[valid] - hardest_pos[valid]).mean() * 10
    return loss


def compute_contrastive_loss(logits_per_image: Tensor, class_labels: Tensor, margin: float) -> Tensor:
    """
    Computes the contrastive loss given logits, class labels, and a margin.

    Args:
        logits_per_image: Tensor of shape (batch_size, batch_size) representing
                          logits or similarity scores between all pairs.
        class_labels: Tensor of shape (batch_size,) representing class labels.
        margin: A float representing the margin value used in the loss calculation.

    Returns:
        A scalar tensor representing the mean contrastive loss.
    """
    same_class_mask = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)

    class_labels = class_labels.to(logits_per_image.device)
    same_class_mask = same_class_mask.to(logits_per_image.device)

    # Positive scores
    positive_scores = logits_per_image.masked_select(same_class_mask.fill_diagonal_(False))
    negative_scores = logits_per_image.masked_select(~same_class_mask)

    # Check if there are positive or negative scores and compute losses accordingly
    if positive_scores.numel() == 0:
        positive_loss = logits_per_image.sum() * 0.0
    else:
        positive_loss = F.relu(1.0 - positive_scores).mean()

    if negative_scores.numel() == 0:
        negative_loss = logits_per_image.sum() * 0.0
    else:
        negative_loss = F.relu(negative_scores - margin).mean()

    # Total loss
    loss = positive_loss + negative_loss

    return loss


def simple_loss(logits_per_image: Tensor) -> Tensor:
    """
    Computes the simple loss given logits. The simple loss is the mean of the top-5 logits for each image.

    Args:
        logits_per_image: Tensor of shape (batch_size, batch_size) representing
                          logits or similarity scores between all pairs.

    Returns:
        A scalar tensor representing the mean loss.
    """
    loss = logits_per_image.topk(5, dim=1).values.mean(dim=1).mean()

    return loss
