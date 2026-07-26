# -*- coding: utf-8 -*-
"""Stage 2 read-out: AUC per arm, and the paired test between arms.

    python scripts/summarise_grade.py
    python scripts/summarise_grade.py --results results --target grade

Reads results/grade_*.csv only — no GPU, no checkpoints, no packaged data. That
is deliberate: the figures and the reported numbers have to survive the Colab
subscription.

Every patient is held out in exactly one fold, so concatenating the five folds
gives one probability per patient per seed; those are averaged over seeds before
scoring, the same way the Stage-1 numbers were read. The comparison between two
arms is paired on patients and uses DeLong, which is the test the plan specifies
and the one that accounts for the two AUCs being estimated on the same 48 people.

The two frames are separate comparisons. A WLI-referenced model and an
NBI-referenced model are not competing on the same input, and their AUCs should
not be put in one ranking.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import math

import numpy as np
import pandas as pd

FRAMES = {
    "WLI": ("wli_only_wli", "ours_wli"),
    "NBI": ("nbi_only_nbi", "ours_nbi"),
}

# Normal tail and quantile in the standard library: this script is the one that
# has to keep working on a bare machine after the Colab subscription lapses, so
# it depends on nothing beyond numpy and pandas.
_Z_975 = 1.959963984540054


def _norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# --------------------------------------------------------------------------
# DeLong's test for two correlated ROC curves (Sun & Xu 2014, fast version).
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, float)
    out[order] = ranks
    return out


def _structural_components(scores: np.ndarray, n_pos: int):
    """scores: (k, n) with the positive cases first. Returns V10, V01 and AUCs."""
    pos, neg = scores[:, :n_pos], scores[:, n_pos:]
    k, n_neg = scores.shape[0], scores.shape[1] - n_pos

    tx = np.stack([_midrank(p) for p in pos])
    ty = np.stack([_midrank(n) for n in neg])
    tz = np.stack([_midrank(s) for s in scores])

    auc = (tz[:, :n_pos].sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    v10 = (tz[:, :n_pos] - tx) / n_neg
    v01 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    return v10, v01, auc


def delong_cov(scores: np.ndarray, n_pos: int):
    v10, v01, auc = _structural_components(scores, n_pos)
    n_neg = scores.shape[1] - n_pos
    cov = np.cov(v10) / n_pos + np.cov(v01) / n_neg
    return auc, np.atleast_2d(cov)


def delong_auc_ci(y_true: np.ndarray, y_prob: np.ndarray):
    order = np.argsort(-y_true, kind="stable")
    auc, cov = delong_cov(y_prob[order][None, :], int(y_true.sum()))
    se = float(np.sqrt(cov[0, 0]))
    lo, hi = auc[0] - _Z_975 * se, auc[0] + _Z_975 * se
    return float(auc[0]), float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1))


def delong_paired_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray):
    order = np.argsort(-y_true, kind="stable")
    auc, cov = delong_cov(np.stack([prob_a[order], prob_b[order]]), int(y_true.sum()))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = float(auc[0] - auc[1])
    if var <= 0:
        return delta, float("nan"), 1.0
    z = float(delta / np.sqrt(var))
    return delta, z, 2 * _norm_sf(abs(z))


# --------------------------------------------------------------------------
def load(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "grade_*.csv")))
    if not paths:
        raise SystemExit(
            f"no grade_*.csv in {results_dir}. Run scripts/train_grade.py first "
            "(it needs the Stage-1 checkpoints, so it runs where those live).")
    frames = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig")
        match = re.search(r"_fold(\d+)_seed(\d+)\.csv$", os.path.basename(path))
        if match:                       # trust the filename over the column
            df["fold"], df["seed"] = int(match.group(1)), int(match.group(2))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_patient(df: pd.DataFrame, config: str):
    sub = df[df.config == config]
    if sub.empty:
        return None
    folds = sub.groupby("case").fold.nunique()
    if (folds > 1).any():
        raise SystemExit(f"{config}: patients appear in more than one fold — "
                         "the fold assignment changed between runs")
    agg = sub.groupby("case").agg(y_true=("y_true", "first"),
                                  y_prob=("y_prob", "mean"),
                                  seeds=("seed", "nunique")).reset_index()
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--target", default="grade", choices=["grade", "macro"])
    args = ap.parse_args()

    df = load(args.results)
    prefix = "" if args.target == "grade" else "macro_"
    configs = sorted(c for c in df.config.unique()
                     if c.startswith("macro_") == (args.target == "macro"))
    if not configs:
        raise SystemExit(f"no arms found for target {args.target!r}")

    print(f"=== endpoint: {args.target} ===")
    tables = {}
    for config in configs:
        agg = per_patient(df, config)
        tables[config] = agg
        auc, lo, hi = delong_auc_ci(agg.y_true.to_numpy(), agg.y_prob.to_numpy())
        print(f"  {config:24s} n={len(agg):3d} ({int(agg.y_true.sum())} pos)  "
              f"seeds={agg.seeds.max()}  AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n=== paired comparison within each frame (DeLong) ===")
    for frame, (single, dual) in FRAMES.items():
        a, b = f"{prefix}{dual}", f"{prefix}{single}"
        if a not in tables or b not in tables:
            print(f"  {frame} frame: incomplete "
                  f"({'missing ' + a if a not in tables else 'missing ' + b})")
            continue
        left, right = tables[a], tables[b]
        shared = sorted(set(left.case) & set(right.case))
        left = left.set_index("case").loc[shared]
        right = right.set_index("case").loc[shared]
        assert (left.y_true.to_numpy() == right.y_true.to_numpy()).all()

        delta, z, p = delong_paired_test(left.y_true.to_numpy(),
                                         left.y_prob.to_numpy(),
                                         right.y_prob.to_numpy())
        verdict = "dual higher" if delta > 0 else "single higher"
        print(f"  {frame} frame  n={len(shared)}  "
              f"AUC(dual) - AUC(single) = {delta:+.3f}  z={z:+.2f}  p={p:.4f}"
              f"   [{verdict}]")

    print("\nThe two frames are independent comparisons; do not rank across them.")
    if args.target == "grade":
        print("Read the positive control before reading this: "
              "python scripts/summarise_grade.py --target macro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
