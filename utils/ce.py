from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .functional import label_smoothed_nll_loss

class SoftBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
    - Support of ignore_index value
    - Support of label smoothing
    """

    __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self, weight=None, ignore_index: Optional[int] = -100, reduction="mean", smooth_factor=None, pos_weight=None
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.smooth_factor is not None:
            soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
        else:
            soft_targets = target.type_as(input)

        loss = F.binary_cross_entropy_with_logits(
            input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction="none"
        )

        if self.ignore_index is not None:
            not_ignored_mask: Tensor = target != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss

def balanced_binary_cross_entropy_with_logits(
    logits: Tensor, targets: Tensor, gamma: float = 1.0, ignore_index: Optional[int] = None, reduction: str = "mean"
) -> Tensor:
    """
    Balanced binary cross entropy loss.

    Args:
        logits:
        targets: This loss function expects target values to be hard targets 0/1.
        gamma: Power factor for balancing weights
        ignore_index:
        reduction:

    Returns:
        Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
        shape of `logits` tensor.
    """
    pos_targets: Tensor = targets.eq(1).sum()
    neg_targets: Tensor = targets.eq(0).sum()

    num_targets = pos_targets + neg_targets
    pos_weight = torch.pow(neg_targets / (num_targets + 1e-7), gamma)
    neg_weight = 1.0 - pos_weight

    pos_term = pos_weight.pow(gamma) * targets * torch.nn.functional.logsigmoid(logits)
    neg_term = neg_weight.pow(gamma) * (1 - targets) * torch.nn.functional.logsigmoid(-logits)

    loss = -(pos_term + neg_term)

    if ignore_index is not None:
        loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)

    if reduction == "mean":
        loss = loss.mean()

    if reduction == "sum":
        loss = loss.sum()

    return loss

class BalancedBCEWithLogitsLoss(nn.Module):
    """
    Balanced binary cross-entropy loss.

    https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
    """

    __constants__ = ["gamma", "reduction", "ignore_index"]

    def __init__(self, gamma: float = 1.0, reduction="mean", ignore_index: Optional[int] = None):
        """

        Args:
            gamma:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return balanced_binary_cross_entropy_with_logits(
            output, target, gamma=self.gamma, ignore_index=self.ignore_index, reduction=self.reduction
        )

class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )