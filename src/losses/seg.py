# -*- coding: utf-8 -*-
"""Segmentation loss and metrics.

Dice + BCE only. Boundary-distance objectives are deliberately absent: the
supervising masks are produced by prompting a foundation model with a clinician
box, so their boundaries carry a component of the labelling procedure rather than
of the tissue.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0,
                 smooth: float = 1.0):
        super().__init__()
        self.dice_weight, self.bce_weight, self.smooth = dice_weight, bce_weight, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = 1.0 - ((2 * inter + self.smooth) / (denom + self.smooth))
        return self.bce_weight * bce + self.dice_weight * dice.mean()


@torch.no_grad()
def binary_metrics(logits: torch.Tensor, target: torch.Tensor,
                   threshold: float = 0.5) -> dict[str, torch.Tensor]:
    """Per-sample Dice, IoU, precision and recall."""
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)
    tp = (pred * target).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    fn = ((1 - pred) * target).sum(dims)
    eps = 1e-7
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "pred_area_frac": pred.mean(dims),
        "gt_area_frac": target.mean(dims),
    }


@torch.no_grad()
def longest_axis_px(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Longer side of the predicted mask's bounding box, in pixels.

    Feeds the comparison against the microscopically measured lesion extent.
    """
    pred = (torch.sigmoid(logits) > threshold).squeeze(1)
    out = []
    for m in pred:
        idx = m.nonzero()
        if idx.numel() == 0:
            out.append(torch.tensor(0.0, device=logits.device))
            continue
        h = (idx[:, 0].max() - idx[:, 0].min() + 1).float()
        w = (idx[:, 1].max() - idx[:, 1].min() + 1).float()
        out.append(torch.maximum(h, w))
    return torch.stack(out)
