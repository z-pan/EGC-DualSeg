# -*- coding: utf-8 -*-
"""Stage 2 entry point: histological grading on frozen Stage-1 features.

    python scripts/train_grade.py --seg-config configs/ours.yaml --ref WLI
    python scripts/train_grade.py --seg-config configs/wli_only.yaml --target macro

This is where the project's core claim is actually tested. Stage 1 compares a
dual-modality model against single-modality ones on SAM-derived masks, which
were produced from the clinician's own boxes — so a gain there is partly a
statement about the labelling procedure. Here the endpoint is the ESD pathology
report, and it owes nothing to any model.

The four arms of the comparison, in two independent frames (mirroring the
Stage-1 ablation, where cross-frame Dice is meaningless):

    WLI frame   wli_only + ref WLI   vs   ours + ref WLI
    NBI frame   nbi_only + ref NBI   vs   ours + ref NBI

Each writes results/grade_{config}_fold{f}_seed{s}.csv, where `config` is
`{seg}_{ref}` for the grade endpoint and `macro_{seg}_{ref}` for the positive
control — one file per fold and seed, columns case, config, fold, seed, y_true,
y_prob. Every patient appears in exactly one fold, so concatenating the five
folds gives one probability per patient per seed.

Why a linear probe rather than a fine-tuned head: 48 patients, ~38 of them in
any training split. The regularisation strength is the only free parameter and
it is chosen by stratified inner cross-validation *within* the training folds,
so the held-out fold is untouched by model selection.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stage2.features import (build_frozen_model, extract_split,  # noqa: E402
                                 to_matrix)
from src.stage2.labels import describe, load_labels                  # noqa: E402

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
FIELDS = ["case", "config", "fold", "seed", "y_true", "y_prob"]


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    """L2 logistic regression; C chosen by inner CV on the training folds only."""
    n_splits = min(5, int(np.bincount(y_train).min()))
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, solver="lbfgs"))
    if n_splits < 2:
        base.fit(x_train, y_train)
        return base, float("nan")
    search = GridSearchCV(
        base, {"logisticregression__C": C_GRID}, scoring="roc_auc",
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed), n_jobs=1)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_["logisticregression__C"]


def run_one(seg_name: str, seg_raw: dict, ref: str, target: str, fold: int, seed: int,
            ckpt_dir: str, out_dir: str, num_workers: int, device: torch.device) -> dict:
    npz, manifest = seg_raw["npz"], seg_raw["manifest"]
    folds_csv, mode = seg_raw["folds"], seg_raw["mode"]

    tag = f"{seg_name}_fold{fold}_seed{seed}"
    model = build_frozen_model(
        os.path.join(ckpt_dir, f"{tag}.pt"), mode, device,
        use_role_embedding=seg_raw.get("use_role_embedding", True),
        aux_levels=tuple(seg_raw.get("aux_levels", (3, 4))))

    common = dict(npz=npz, manifest=manifest, folds=folds_csv, fold=fold, mode=mode,
                  reference=ref, device=device, num_workers=num_workers)
    train_feats = extract_split(model, split="train", **common)
    val_feats = extract_split(model, split="val", **common)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    labels = load_labels(manifest, target)
    _, x_train, y_train = to_matrix(train_feats, labels)
    val_cases, x_val, y_val = to_matrix(val_feats, labels)

    config = f"{seg_name}_{ref.lower()}" if target == "grade" \
        else f"{target}_{seg_name}_{ref.lower()}"
    if len(val_cases) == 0 or len(np.unique(y_train)) < 2:
        print(f"  [{config} fold{fold} seed{seed}] skipped: "
              f"{len(val_cases)} labelled held-out patients, "
              f"{len(np.unique(y_train))} training classes")
        return dict(config=config, fold=fold, seed=seed, n_val=len(val_cases),
                    best_c=float("nan"), path="")

    probe, best_c = fit_probe(x_train, y_train, seed)
    y_prob = probe.predict_proba(x_val)[:, 1]

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"grade_{config}_fold{fold}_seed{seed}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for case, yt, yp in zip(val_cases, y_val, y_prob):
            writer.writerow(dict(case=case, config=config, fold=fold, seed=seed,
                                 y_true=int(yt), y_prob=float(yp)))

    # Degenerate pooling regions would quietly turn this into global average
    # pooling, so keep the evidence next to the run. logs/ is not versioned.
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"gradefeat_{config}_fold{fold}_seed{seed}.csv"),
              "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", "split", "views", "region_frac"])
        for split, feats in (("train", train_feats), ("val", val_feats)):
            for case in sorted(feats):
                writer.writerow([case, split, feats[case]["views"],
                                 f"{feats[case]['region_frac']:.5f}"])

    print(f"  [{config} fold{fold} seed{seed}] train {len(y_train)} "
          f"({int(y_train.sum())} pos) -> val {len(val_cases)} "
          f"({int(y_val.sum())} pos)  C={best_c}")
    return dict(config=config, fold=fold, seed=seed, n_val=len(val_cases),
                best_c=best_c, path=path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-config", required=True,
                    help="the Stage-1 config whose checkpoint supplies the trunk")
    ap.add_argument("--ref", default=None, choices=["WLI", "NBI"],
                    help="reference modality; defaults to the Stage-1 config's own")
    ap.add_argument("--target", default="grade", choices=["grade", "macro"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(args.seg_config, encoding="utf-8") as fh:
        seg_raw = yaml.safe_load(fh) or {}
    seg_name = seg_raw["name"]
    ref = args.ref or seg_raw.get("reference", "WLI")
    if ref == "random":
        raise SystemExit(
            f"{args.seg_config} trains bidirectionally; pass --ref WLI or --ref NBI "
            "so the held-out patients are scored in one fixed frame.")
    if seg_raw["mode"] == "single" and ref != seg_raw.get("reference"):
        raise SystemExit(
            f"{seg_name} only ever saw {seg_raw.get('reference')}; --ref {ref} "
            "would feed it the modality it was not trained on.")

    ckpt_dir = args.ckpt_dir or seg_raw.get("ckpt_dir", "checkpoints")
    out_dir = args.out_dir or seg_raw.get("out_dir", "results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = load_labels(seg_raw["manifest"], args.target)
    print(f"{seg_name} | ref {ref} | target {args.target} | {describe(labels)} "
          f"| device {device.type}")

    done = []
    for fold in args.folds:
        for seed in args.seeds:
            config = f"{seg_name}_{ref.lower()}" if args.target == "grade" \
                else f"{args.target}_{seg_name}_{ref.lower()}"
            path = os.path.join(out_dir, f"grade_{config}_fold{fold}_seed{seed}.csv")
            if os.path.exists(path) and not args.force:
                print(f"  [{config} fold{fold} seed{seed}] already done, skipping")
                continue
            done.append(run_one(seg_name, seg_raw, ref, args.target, fold, seed,
                                ckpt_dir, out_dir, args.num_workers, device))

    if done:
        written = [d for d in done if d["path"]]
        print(f"\nwrote {len(written)}/{len(done)} run files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
