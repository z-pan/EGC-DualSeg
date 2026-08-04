# -*- coding: utf-8 -*-
"""Sweep the decision threshold, so the comparison stops depending on one operating point.

    python scripts/threshold_sweep.py --configs configs/nbi_only.yaml configs/ours.yaml
    python scripts/threshold_sweep.py --configs configs/wli_only.yaml --folds 4

Every comparison in this project so far binarises at 0.5. That is fine for Dice,
which is roughly balanced around it, but the clinical read-out is not: missed
lesion and over-taken mucosa move in opposite directions as the threshold moves,
so an arm can look better on both simply by sitting at a luckier operating point.

The dual-modality arm currently misses 26% less lesion than the single-modality
one at 0.5 while taking no more normal tissue. The obvious objection is that
lowering the single-modality threshold would buy the same thing. This settles it:
sweep the threshold on every arm, and ask whether the dual arm lies outside the
single arm's whole achievable curve rather than at a better point on it.

Per image and per threshold it records only the two quantities the clinical
question turns on:

    missed_frac   lesion not covered, as a share of the lesion (residual disease)
    over_ratio    normal tissue taken, as a multiple of lesion area

plus Dice for continuity with everything else. The read-out
(scripts/summarise_sweep.py) does the matched-operating-point comparison; this
script makes no claim.

Writes results/sweep_{config}_fold{f}_seed{s}.csv. Cheap: one inference pass, and
thresholding a probability map 19 times costs nothing next to the forward pass.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine.trainer import RunConfig, build_loaders, build_model  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
FIELDS = ["case", "scale", "ref_modality", "config", "fold", "seed",
          "threshold", "missed_frac", "over_ratio", "dice", "pred_area_frac"]


@torch.no_grad()
def sweep_run(cfg: RunConfig, device: torch.device) -> list[dict]:
    tag = f"{cfg.name}_fold{cfg.fold}_seed{cfg.seed}"
    ckpt = os.path.join(cfg.ckpt_dir, f"{tag}.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"missing checkpoint {ckpt}")

    _, val_loader = build_loaders(cfg)
    model = build_model(cfg, device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()

    rows = []
    for batch in val_loader:
        ref = batch["ref"].to(device)
        aux = batch["aux"].to(device) if "aux" in batch else None
        probs = torch.sigmoid(
            model(ref, aux, batch["ref_id"].to(device))["logits"].float())
        probs = probs.cpu().numpy()[:, 0]
        gt = batch["mask"].numpy()[:, 0] > 0.5
        for i in range(probs.shape[0]):
            g = gt[i]
            g_area = float(g.sum())
            if g_area == 0:
                continue
            for t in THRESHOLDS:
                p = probs[i] > t
                tp = float((p & g).sum())
                rows.append(dict(
                    case=batch["case"][i], scale=batch["scale"][i],
                    ref_modality=batch["ref_modality"][i], config=cfg.name,
                    fold=cfg.fold, seed=cfg.seed, threshold=float(t),
                    missed_frac=float((g & ~p).sum()) / g_area,
                    over_ratio=float((p & ~g).sum()) / g_area,
                    dice=2 * tp / (float(p.sum()) + g_area + 1e-7),
                    pred_area_frac=float(p.mean())))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in args.configs:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        folds = args.folds if args.folds is not None else raw.pop("folds_to_run", [0, 1, 2, 3, 4])
        seeds = args.seeds if args.seeds is not None else raw.pop("seeds", [0, 1, 2])
        raw.pop("folds_to_run", None)
        raw.pop("seeds", None)
        if args.ckpt_dir:
            raw["ckpt_dir"] = args.ckpt_dir
        out_dir = args.out_dir or raw.get("out_dir", "results")
        raw["num_workers"] = args.num_workers

        for fold in folds:
            for seed in seeds:
                cfg = RunConfig(fold=fold, seed=seed, **raw)
                dest = os.path.join(out_dir, f"sweep_{cfg.name}_fold{fold}_seed{seed}.csv")
                if os.path.exists(dest) and not args.force:
                    print(f"  {os.path.basename(dest)} exists, skipping")
                    continue
                rows = sweep_run(cfg, device)
                os.makedirs(out_dir, exist_ok=True)
                with open(dest, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                at_half = [r for r in rows if abs(r["threshold"] - 0.5) < 1e-6]
                print(f"  {cfg.name} fold{fold} seed{seed}: "
                      f"{len(rows)} rows, at 0.5 missed "
                      f"{np.mean([r['missed_frac'] for r in at_half]):.3f} / over "
                      f"{np.mean([r['over_ratio'] for r in at_half]):.3f}")
    print("\nread it with: python scripts/summarise_sweep.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
