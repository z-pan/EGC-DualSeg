# -*- coding: utf-8 -*-
"""Stage 2, lesion-level late fusion: does NBI carry *any* incremental signal?

    python scripts/train_grade_late.py --target macro
    python scripts/train_grade_late.py --target grade

Why this arm exists
-------------------
Every fusion result so far — Stage 1's segmentation ablation and Stage 2's
`ours_{wli,nbi}` arms — fuses the two modalities *in the image plane*, before
pooling. That architecture has to solve spatial correspondence, and the measured
lesion overlap between the two frames is IoU 0.28. So a null result from it is
ambiguous: it could mean NBI adds nothing, or it could mean the fusion operator
spent its capacity on a registration problem that grading does not require.

Late fusion removes the ambiguity. Each modality is encoded by its own
single-modality trunk, pooled over its *own* predicted lesion region, and the two
512-d vectors are concatenated. Nothing is ever required to line up spatially:
each vector is a description of one lesion as seen under one illumination, and
the probe is free to use both. If a patient-level NBI signal exists at all, this
is the arm that should find it — and if this arm is also null, coarse alignment
(handoff §一) is not worth building, because alignment fixes *usability*, not
*existence*.

The three arms this writes, and what each is for
------------------------------------------------
`late_fusion`       concat(WLI 512-d, NBI 512-d) -> 1024-d probe. The question.

`late_null`         the same probe fitted and evaluated with the NBI half taken
                    from a *different* patient (a derangement, drawn per split).
                    Same dimensionality, same marginal feature distribution,
                    patient-specific pairing destroyed throughout. This is the
                    matched null: it stops "1024-d beat 512-d" from being read as
                    fusion. `late_fusion − late_null` is the pairing;
                    `late_null − wli_only_wli` is what the extra 512
                    uninformative dimensions cost.

`late_aux_shuffled` the `late_fusion` probe itself, applied to held-out patients
                    whose NBI half has been permuted. Nothing is refitted. This
                    measures whether the fitted probe *uses* this patient's NBI
                    at all, rather than inferring it from a coefficient — the
                    same correction the Stage-1 gate ablation forced on the gate
                    reading, where a parameter that looked switched off turned
                    out to be in use with a negligible effect.

All three cost no extra GPU time: the features are extracted once per (fold,
seed) and reused. The 512-d single-modality arms are not recomputed here —
`wli_only_wli` and `nbi_only_nbi` from train_grade.py use the same checkpoints,
the same pooling and the same folds, so they *are* the halves of this
concatenation.

Everything else follows train_grade.py exactly — same folds, same frozen trunks,
same soft-region pooling, augmentation off, C chosen by inner CV inside the
training folds. Run the `macro` positive control first, for the same reason.

Fairness note: unlike the Stage-1 ablation there is no reference-frame issue
here. Each modality is read in its own frame and the label is the ESD pathology
report, which belongs to neither. All Stage-2 arms are therefore scored against
one gold standard on one set of patients and may be compared in a single table.
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
from scripts.train_grade import FIELDS, fit_probe                     # noqa: E402
from src.stage2.features import build_frozen_model, extract_split     # noqa: E402
from src.stage2.labels import describe, load_labels                   # noqa: E402

SPLIT_ID = {"train": 0, "val": 1}


def _derange(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed point, so no patient keeps its own partner."""
    if n < 2:
        return np.arange(n)
    identity = np.arange(n)
    for _ in range(64):
        perm = rng.permutation(n)
        if not np.any(perm == identity):
            return perm
    return np.roll(identity, 1)          # a cyclic shift is a derangement by construction


def build_matrix(wli: dict[str, dict], nbi: dict[str, dict], labels: dict[str, int],
                 rng: np.random.Generator
                 ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Per patient: [WLI 512-d | NBI 512-d], plus the same thing with the NBI
    half deranged. Both share the case order and the labels, because only the
    right-hand block is permuted."""
    cases = sorted(c for c in wli if c in nbi and c in labels)
    if not cases:
        empty = np.zeros((0, 0), np.float32)
        return [], empty, empty, np.zeros((0,), np.int64)
    left = np.stack([wli[c]["feature"] for c in cases])
    right = np.stack([nbi[c]["feature"] for c in cases])
    paired = np.concatenate([left, right], axis=1).astype(np.float32)
    broken = np.concatenate([left, right[_derange(len(cases), rng)]],
                            axis=1).astype(np.float32)
    y = np.array([labels[c] for c in cases], np.int64)
    return cases, paired, broken, y


def extract_arm(raw: dict, ref: str, fold: int, seed: int, ckpt_dir: str,
                num_workers: int, device: torch.device) -> tuple[dict, dict]:
    """Frozen features for one modality's trunk, for both splits of one fold."""
    name = raw["name"]
    model = build_frozen_model(
        os.path.join(ckpt_dir, f"{name}_fold{fold}_seed{seed}.pt"), raw["mode"], device,
        use_role_embedding=raw.get("use_role_embedding", True),
        aux_levels=tuple(raw.get("aux_levels", (3, 4))))
    common = dict(npz=raw["npz"], manifest=raw["manifest"], folds=raw["folds"],
                  fold=fold, mode=raw["mode"], reference=ref, device=device,
                  num_workers=num_workers)
    train = extract_split(model, split="train", **common)
    val = extract_split(model, split="val", **common)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return train, val


def write_run(out_dir: str, config: str, fold: int, seed: int,
              cases: list[str], y_true: np.ndarray, y_prob: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"grade_{config}_fold{fold}_seed{seed}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for case, yt, yp in zip(cases, y_true, y_prob):
            writer.writerow(dict(case=case, config=config, fold=fold, seed=seed,
                                 y_true=int(yt), y_prob=float(yp)))
    return path


def log_regions(out_dir: str, config: str, fold: int, seed: int,
                feats: dict[str, tuple[dict, dict]]) -> None:
    """Keep the pooled-region fractions next to the run.

    Below MIN_REGION_FRAC the soft pooling silently degrades to global average
    pooling, and for late fusion that can happen in one modality and not the
    other — which would be a real finding about the arm, not a nuisance.
    """
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"gradefeat_{config}_fold{fold}_seed{seed}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", "modality", "split", "views", "region_frac"])
        for modality, (train, val) in feats.items():
            for split, table in (("train", train), ("val", val)):
                for case in sorted(table):
                    writer.writerow([case, modality, split, table[case]["views"],
                                     f"{table[case]['region_frac']:.5f}"])


def run_one(wli_raw: dict, nbi_raw: dict, target: str, fold: int, seed: int,
            ckpt_dir: str, out_dir: str, num_workers: int, device: torch.device,
            controls: bool) -> list[dict]:
    wli_train, wli_val = extract_arm(wli_raw, "WLI", fold, seed, ckpt_dir,
                                     num_workers, device)
    nbi_train, nbi_val = extract_arm(nbi_raw, "NBI", fold, seed, ckpt_dir,
                                     num_workers, device)

    labels = load_labels(wli_raw["manifest"], target)
    prefix = "" if target == "grade" else f"{target}_"

    # One stream per (fold, seed, split): reproducible, not repeated across folds,
    # and the training split is deranged independently of the held-out split.
    rng = lambda split: np.random.default_rng([seed, fold, SPLIT_ID[split]])
    _, x_train, x_train_broken, y_train = build_matrix(
        wli_train, nbi_train, labels, rng("train"))
    cases, x_val, x_val_broken, y_val = build_matrix(
        wli_val, nbi_val, labels, rng("val"))

    if len(cases) == 0 or len(np.unique(y_train)) < 2:
        print(f"  [late fold{fold} seed{seed}] skipped: {len(cases)} labelled "
              f"held-out patients, {len(np.unique(y_train))} training classes")
        return [dict(config=f"{prefix}late_fusion", fold=fold, seed=seed, path="")]

    probe, best_c = fit_probe(x_train, y_train, seed)
    runs = [("late_fusion", probe.predict_proba(x_val)[:, 1])]
    if controls:
        # Same probe, permuted partner at evaluation time only: does it use NBI?
        runs.append(("late_aux_shuffled", probe.predict_proba(x_val_broken)[:, 1]))
        # Refitted on broken pairings: the matched-dimensionality null.
        null_probe, _ = fit_probe(x_train_broken, y_train, seed)
        runs.append(("late_null", null_probe.predict_proba(x_val_broken)[:, 1]))

    log_regions(out_dir, f"{prefix}late_fusion", fold, seed,
                {"WLI": (wli_train, wli_val), "NBI": (nbi_train, nbi_val)})

    done = []
    for arm, y_prob in runs:
        config = f"{prefix}{arm}"
        path = write_run(out_dir, config, fold, seed, cases, y_val, y_prob)
        done.append(dict(config=config, fold=fold, seed=seed, path=path))

    print(f"  [late fold{fold} seed{seed}] train {len(y_train)} "
          f"({int(y_train.sum())} pos) x {x_train.shape[1]}-d -> val {len(cases)} "
          f"({int(y_val.sum())} pos)  C={best_c}  arms: "
          f"{', '.join(a for a, _ in runs)}")
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wli-config", default="configs/wli_only.yaml")
    ap.add_argument("--nbi-config", default="configs/nbi_only.yaml")
    ap.add_argument("--target", default="grade", choices=["grade", "macro"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-controls", action="store_true",
                    help="write late_fusion only; the result is uninterpretable "
                         "without its controls, so this is for smoke tests")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    def read(path: str) -> dict:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    wli_raw, nbi_raw = read(args.wli_config), read(args.nbi_config)
    for raw, want, path in ((wli_raw, "WLI", args.wli_config),
                            (nbi_raw, "NBI", args.nbi_config)):
        if raw["mode"] != "single" or raw.get("reference") != want:
            raise SystemExit(
                f"{path} is mode={raw['mode']} reference={raw.get('reference')}; late "
                f"fusion needs a single-modality {want} trunk so each vector describes "
                "one modality on its own.")
    if wli_raw["npz"] != nbi_raw["npz"] or wli_raw["folds"] != nbi_raw["folds"]:
        raise SystemExit("the two trunks were trained on different data or folds; "
                         "their features cannot be concatenated per patient.")

    ckpt_dir = args.ckpt_dir or wli_raw.get("ckpt_dir", "checkpoints")
    out_dir = args.out_dir or wli_raw.get("out_dir", "results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = load_labels(wli_raw["manifest"], args.target)
    print(f"late fusion | {wli_raw['name']}+{nbi_raw['name']} | target {args.target} "
          f"| {describe(labels)} | device {device.type}")

    prefix = "" if args.target == "grade" else f"{args.target}_"
    arms = [f"{prefix}late_fusion"] + ([] if args.no_controls else
                                       [f"{prefix}late_aux_shuffled", f"{prefix}late_null"])

    done = []
    for fold in args.folds:
        for seed in args.seeds:
            paths = [os.path.join(out_dir, f"grade_{a}_fold{fold}_seed{seed}.csv")
                     for a in arms]
            if all(os.path.exists(p) for p in paths) and not args.force:
                print(f"  [fold{fold} seed{seed}] already done, skipping")
                continue
            done += run_one(wli_raw, nbi_raw, args.target, fold, seed, ckpt_dir,
                            out_dir, args.num_workers, device, not args.no_controls)

    if done:
        written = [d for d in done if d["path"]]
        print(f"\nwrote {len(written)}/{len(done)} run files to {out_dir}")
        print("read it with: python scripts/summarise_grade.py --target "
              f"{args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
