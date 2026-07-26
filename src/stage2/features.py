# -*- coding: utf-8 -*-
"""Region-pooled features from a frozen Stage-1 encoder.

48 patients cannot fine-tune a ResNet-34. The trunk is loaded from the Stage-1
checkpoint for the *same* (config, fold, seed), frozen, and used only to produce
one 512-dimensional vector per patient; a regularised linear model is then fitted
on top. Because the fold is the same one the segmentation model was trained on,
no held-out patient has been seen by either stage.

Where the vector comes from
---------------------------
The bottleneck feature map (post-fusion for `ours`) is averaged over the lesion
region that Stage 1 itself predicted, using the sigmoid probability as a soft
weight. Nothing here needs a ground-truth mask, so the same code runs at test
time on a patient with no pathology yet.

Two details that decide whether the comparison is fair
-----------------------------------------------------
* **Augmentation is off.** With the trunk frozen there is nothing to make
  invariant, and deterministic features keep the linear fit reproducible.
* **One reference direction per call.** `ours` is bidirectional, so it could be
  given both the WLI-referenced and the NBI-referenced view and average them —
  but then it would see twice as many views as the single-modality arm and part
  of any gain would be view averaging rather than fusion. Each arm is therefore
  extracted under one fixed reference, and the two frames are reported
  separately, exactly as in the Stage-1 ablation.

Views within a direction (near and distant working distance) *are* averaged:
both arms have them, so that is symmetric.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EGCPairDataset, collate
from src.data.transforms import PairAugment
from src.models.net import EGCNet

# Below this predicted area the soft-weighted average is dominated by whatever
# noise survived the sigmoid, so fall back to global pooling and say so.
MIN_REGION_FRAC = 0.005


def load_checkpoint(path: str) -> dict:
    """Load one of our own checkpoints across torch versions.

    torch 2.6 flipped `weights_only` to True by default, and these checkpoints
    carry the RunConfig alongside the weights, so the strict path rejects them.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_frozen_model(ckpt_path: str, mode: str, device: torch.device,
                       use_role_embedding: bool = True,
                       aux_levels: tuple[int, ...] = (3, 4)) -> EGCNet:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Stage-1 checkpoint missing: {ckpt_path}\n"
            "Stage 2 reads the trunk from Stage 1; run scripts/train.py for this "
            "(config, fold, seed) first, or point --ckpt-dir at the Drive copy.")
    model = EGCNet(mode=mode, pretrained=False,
                   use_role_embedding=use_role_embedding, aux_levels=tuple(aux_levels))
    state = load_checkpoint(ckpt_path)
    model.load_state_dict(state.get("model", state))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _pool(embedding: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-region average pooling. Returns (B, C) vectors and the region fraction."""
    region = torch.sigmoid(logits)
    region = F.interpolate(region, size=embedding.shape[-2:], mode="bilinear",
                           align_corners=False)
    frac = region.mean(dim=(1, 2, 3))

    weight = region.sum(dim=(2, 3)).clamp_min(1e-6)
    pooled = (embedding * region).sum(dim=(2, 3)) / weight
    globally = embedding.mean(dim=(2, 3))
    degenerate = (frac < MIN_REGION_FRAC)[:, None]
    return torch.where(degenerate, globally, pooled), frac


@torch.no_grad()
def extract_split(model: EGCNet, npz: str, manifest: str, folds: str, fold: int,
                  split: str, mode: str, reference: str, device: torch.device,
                  batch_size: int = 8, num_workers: int = 2) -> dict[str, dict]:
    """case -> {'feature': (C,) float32, 'views': int, 'region_frac': float}."""
    dataset = EGCPairDataset(npz, manifest, folds, fold, split, mode=mode,
                             reference=reference, transform=PairAugment(train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=device.type == "cuda")

    acc: dict[str, dict] = {}
    for batch in loader:
        ref = batch["ref"].to(device, non_blocking=True)
        aux = batch["aux"].to(device, non_blocking=True) if "aux" in batch else None
        ref_id = batch["ref_id"].to(device, non_blocking=True)
        out = model(ref, aux, ref_id)
        pooled, frac = _pool(out["embedding"].float(), out["logits"].float())
        pooled = pooled.cpu().numpy()
        frac = frac.cpu().numpy()
        for i, case in enumerate(batch["case"]):
            entry = acc.setdefault(case, {"sum": np.zeros(pooled.shape[1], np.float64),
                                          "views": 0, "frac": 0.0})
            entry["sum"] += pooled[i]
            entry["frac"] += float(frac[i])
            entry["views"] += 1

    return {case: {"feature": (e["sum"] / e["views"]).astype(np.float32),
                   "views": e["views"],
                   "region_frac": e["frac"] / e["views"]}
            for case, e in acc.items()}


def to_matrix(features: dict[str, dict], labels: dict[str, int]
              ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Restrict to cases that have both a feature and a label, in a stable order."""
    cases = sorted(c for c in features if c in labels)
    if not cases:
        return [], np.zeros((0, 0), np.float32), np.zeros((0,), np.int64)
    x = np.stack([features[c]["feature"] for c in cases])
    y = np.array([labels[c] for c in cases], np.int64)
    return cases, x, y
