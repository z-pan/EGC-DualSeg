# -*- coding: utf-8 -*-
"""Dump single-modality predicted masks, so alignment can be driven without the answer.

    python scripts/predict_masks.py --fold 4 --seeds 0 1 2

This is step 1 of the designed pipeline: each modality is segmented on its own,
the two predicted masks give a coarse alignment, and only then are the frames
fused. `align.py` implements the ceiling of that idea using ground-truth masks;
this implements the real input to it.

The distinction matters more than it looks. An alignment fitted on *predicted*
masks can only be as good as the model's own segmentation, so the auxiliary
frame is placed according to what the model already believes — it can refine a
boundary but cannot correct a gross localisation error, because a gross error
would misplace the alignment too. An alignment fitted on *ground-truth* masks is
placed better than, and independently of, the model's belief, and therefore
injects localisation information this pipeline structurally cannot obtain at
inference. That is why the oracle number is an upper bound and not a forecast.

One honest asymmetry, which belongs in Methods. For patients in the training
split the predicted mask comes from a model that trained on them, so it is
better than it would be in deployment. For the held-out split it does not — the
fold that predicts a patient is the fold that excluded them. Metrics are read on
the held-out split, so the reported number is clean; what remains is a mild
train/test mismatch in alignment quality, in the direction of making training
slightly easier than inference.

Writes data/packaged/predmasks_fold{f}_seed{s}.npz with one uint8 mask per row
of the packaged image array, each produced by that row's own modality model.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import EGCPairDataset, collate, load_pairs   # noqa: E402
from src.data.transforms import PairAugment                        # noqa: E402
from src.stage2.features import build_frozen_model                 # noqa: E402

MODALITY_CONFIG = {"WLI": "configs/wli_only.yaml", "NBI": "configs/nbi_only.yaml"}


@torch.no_grad()
def predict_one_modality(raw: dict, modality: str, fold: int, seed: int,
                         ckpt_dir: str, device: torch.device,
                         num_workers: int) -> dict[tuple[str, str], np.ndarray]:
    name = raw["name"]
    model = build_frozen_model(os.path.join(ckpt_dir, f"{name}_fold{fold}_seed{seed}.pt"),
                               raw["mode"], device)
    out: dict[tuple[str, str], np.ndarray] = {}
    for split in ("train", "val"):
        ds = EGCPairDataset(raw["npz"], raw["manifest"], raw["folds"], fold, split,
                            mode="single", reference=modality,
                            transform=PairAugment(train=False))
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate,
                            num_workers=num_workers,
                            pin_memory=device.type == "cuda")
        for batch in loader:
            logits = model(batch["ref"].to(device), None,
                           batch["ref_id"].to(device))["logits"]
            probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
            for i, (case, scale) in enumerate(zip(batch["case"], batch["scale"])):
                out[(case, scale)] = (probs[i] > 0.5).astype(np.uint8) * 255
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raws = {m: yaml.safe_load(open(p, encoding="utf-8"))
            for m, p in MODALITY_CONFIG.items()}
    npz_path = raws["WLI"]["npz"]
    pairs, _ = load_pairs(raws["WLI"]["manifest"], raws["WLI"]["folds"])
    n_images = int(np.load(npz_path)["images"].shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in args.seeds:
        out_path = os.path.join(os.path.dirname(npz_path),
                                f"predmasks_fold{args.fold}_seed{seed}.npz")
        if os.path.isfile(out_path) and not args.force:
            print(f"  fold{args.fold} seed{seed}: already present, skipping")
            continue

        pred = np.zeros((n_images, 384, 384), np.uint8)
        covered = np.zeros(n_images, bool)
        for modality, raw in raws.items():
            table = predict_one_modality(raw, modality, args.fold, seed,
                                         args.ckpt_dir, device, args.num_workers)
            for pair in pairs:
                mask = table.get((pair.case, pair.scale))
                if mask is None:
                    continue
                idx = pair.idx[modality]
                pred[idx] = mask
                covered[idx] = True

        frac = (pred > 127).mean(axis=(1, 2))[covered]
        empty = int((frac < 1e-6).sum())
        np.savez_compressed(out_path, pred=pred, covered=covered)
        print(f"  fold{args.fold} seed{seed}: {int(covered.sum())}/{n_images} images, "
              f"median predicted area {np.median(frac):.4f}, {empty} empty -> {out_path}")
        if empty:
            print(f"    note: {empty} predictions are empty; those pairs fall back to "
                  "no alignment, which is the honest behaviour on a failed segmentation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
