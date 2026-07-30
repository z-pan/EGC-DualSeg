# -*- coding: utf-8 -*-
"""Entry point: train one configuration over one or more folds and seeds.

    python scripts/train.py --config configs/ours.yaml
    python scripts/train.py --config configs/ours.yaml --folds 0 --seeds 0 --epochs 3
    python scripts/train.py --config configs/wli_only.yaml --resume-fold 2

Every (config, fold, seed) writes results/predictions_{name}_fold{f}_seed{s}.csv.
A run whose prediction CSV already exists is skipped, so an interrupted Colab
session can be restarted with the same command.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine.trainer import RunConfig, train_one       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--resume-fold", type=int, default=None,
                    help="skip folds below this index")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the prediction CSV exists")
    # Overrides so one config can be pointed at any synthetic level without
    # writing a yaml per level.
    ap.add_argument("--npz", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--folds-csv", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    folds = args.folds if args.folds is not None else raw.pop("folds_to_run", [0, 1, 2, 3, 4])
    seeds = args.seeds if args.seeds is not None else raw.pop("seeds", [0, 1, 2])
    raw.pop("folds_to_run", None)
    raw.pop("seeds", None)

    for key in ("epochs", "batch_size", "num_workers", "npz", "manifest", "name",
                "out_dir", "ckpt_dir"):
        value = getattr(args, key)
        if value is not None:
            raw[key] = value
    if args.folds_csv is not None:
        raw["folds"] = args.folds_csv

    summary = []
    for fold in folds:
        if args.resume_fold is not None and fold < args.resume_fold:
            continue
        for seed in seeds:
            cfg = RunConfig(fold=fold, seed=seed, **raw)
            tag = f"{cfg.name}_fold{fold}_seed{seed}"
            done = os.path.join(cfg.out_dir, f"predictions_{tag}.csv")
            if os.path.exists(done) and not args.force:
                print(f"[{tag}] already done, skipping")
                continue
            summary.append(train_one(cfg))

    if summary:
        print("\n=== summary ===")
        for s in summary:
            print(f"  {s['tag']:32s} best Dice {s['best_dice']:.4f} "
                  f"@ epoch {s['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
