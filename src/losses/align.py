# -*- coding: utf-8 -*-
"""Distribution-alignment loss between the two modality streams.

Why this exists
---------------
The published WLI/NBI fusion line (Wu et al., Med Image Anal 2026, the method
Jie et al. 2026 abbreviate ADFNet) aligns the two modalities by minimising the
Maximum Mean Discrepancy between their global feature distributions, and reports
that this improves fusion. This module reproduces that term alone, so the arm
that uses it differs from the plain concatenation arm by exactly one loss and
nothing else.

Two properties of the term are worth stating plainly, because they are what this
cohort can test:

* MMD is a two-sample statistic over the BATCH. It compares the set of WLI
  descriptors against the set of NBI descriptors and is invariant to which WLI
  frame was paired with which NBI frame — permuting the pairing inside a batch
  leaves it unchanged. So it aligns the two modalities as populations, not the
  two views of a lesion as a correspondence.
* The estimate is computed from `batch_size` samples per modality. At the batch
  size this cohort trains with, that estimate is noisy; the weight has to be
  small enough that the noise does not compete with the segmentation objective,
  which is consistent with the 1e-4 the original authors report.

This is a faithful reproduction of the alignment term, NOT of the full method,
which also carries progressive disentanglement and disentangle-aware contrastive
learning. Any arm using this must be reported as a distribution-alignment
ablation after Wu et al., never under the name of their network.
"""
from __future__ import annotations

import torch


def _pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2.0).pow(2)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor,
            sigmas: tuple[float, ...] = (1.0,),
            median_bandwidth: bool = True) -> torch.Tensor:
    """Unbiased-in-spirit MMD^2 with a Gaussian kernel.

    Args:
        x, y: (B, D) global descriptors, one row per sample.
        sigmas: kernel bandwidths, applied as multipliers on the base bandwidth.
        median_bandwidth: set the base bandwidth from the median pairwise squared
            distance of the pooled sample. Without it the kernel value depends on
            the feature scale, which drifts during training, and the loss would
            silently change meaning between early and late epochs.

    Returns a non-negative scalar; 0 when the two samples are identical.
    """
    if x.shape[0] < 2 or y.shape[0] < 2:
        return x.new_zeros(())
    x = x.float()
    y = y.float()

    d_xx = _pairwise_sq_dists(x, x)
    d_yy = _pairwise_sq_dists(y, y)
    d_xy = _pairwise_sq_dists(x, y)

    if median_bandwidth:
        pooled = torch.cat([d_xx.flatten(), d_yy.flatten(), d_xy.flatten()])
        base = pooled.detach().median().clamp_min(1e-8)
    else:
        base = x.new_tensor(1.0)

    total = x.new_zeros(())
    for s in sigmas:
        gamma = 1.0 / (2.0 * base * s)
        k_xx = torch.exp(-gamma * d_xx)
        k_yy = torch.exp(-gamma * d_yy)
        k_xy = torch.exp(-gamma * d_xy)
        total = total + k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return (total / len(sigmas)).clamp_min(0.0)


def stream_alignment_loss(out: dict, sigmas: tuple[float, ...] = (0.5, 1.0, 2.0)
                          ) -> torch.Tensor:
    """MMD between the reference-stream and auxiliary-stream global descriptors.

    Returns a zero scalar when the model produced no auxiliary stream, so the
    call site does not have to branch on the mode.
    """
    if "ref_global" not in out or "aux_global" not in out:
        ref = out["embedding"]
        return ref.new_zeros(())
    return mmd_rbf(out["ref_global"], out["aux_global"], sigmas=sigmas)
