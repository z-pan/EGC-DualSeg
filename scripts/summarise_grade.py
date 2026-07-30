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

The two frames are separate comparisons *for the in-plane fusion arms*. A
WLI-referenced `ours` and an NBI-referenced `ours` are not competing on the same
input, and their AUCs should not be put in one ranking.

Late fusion (`late_fusion`) sits outside that split: it encodes each modality in
its own frame and concatenates, so there is no reference modality to condition
on. It is compared against every other arm directly — legitimate here, and only
here, because all Stage-2 arms are scored on the same patients against the same
pathology label. Read it next to its two controls: `late_null` is the same 1024-d
probe fitted and scored with the NBI half taken from another patient, and
`late_aux_shuffled` is the real probe with the NBI half permuted at evaluation
only. The first says what the extra dimensions are worth on their own; the
second says whether the probe ever used the second modality.
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

# Lesion-level late fusion, which has no reference frame. Each row is
# (arm, baseline, what the difference means). Order matters: the two controls
# come first, because the rest cannot be read without them.
LATE_COMPARISONS = [
    ("late_fusion", "late_null",
     "patient-specific NBI contribution (dimensionality held fixed)"),
    ("late_fusion", "late_aux_shuffled",
     "how much the fitted probe actually uses this patient's NBI"),
    ("late_null", "wli_only_wli", "price of 512 uninformative dimensions"),
    ("late_fusion", "wli_only_wli", "late fusion vs WLI alone"),
    ("late_fusion", "nbi_only_nbi", "late fusion vs NBI alone"),
    ("late_fusion", "ours_wli", "late vs in-plane fusion, WLI-referenced"),
    ("late_fusion", "ours_nbi", "late vs in-plane fusion, NBI-referenced"),
]

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


def compare(tables: dict, a: str, b: str):
    """Paired DeLong of arm `a` against arm `b` on the patients they share."""
    if a not in tables or b not in tables:
        return None
    left, right = tables[a], tables[b]
    shared = sorted(set(left.case) & set(right.case))
    if not shared:
        return None
    left = left.set_index("case").loc[shared]
    right = right.set_index("case").loc[shared]
    assert (left.y_true.to_numpy() == right.y_true.to_numpy()).all()
    delta, z, p = delong_paired_test(left.y_true.to_numpy(), left.y_prob.to_numpy(),
                                     right.y_prob.to_numpy())
    return len(shared), delta, z, p


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

    print("\n=== in-plane fusion: paired comparison within each frame (DeLong) ===")
    for frame, (single, dual) in FRAMES.items():
        a, b = f"{prefix}{dual}", f"{prefix}{single}"
        result = compare(tables, a, b)
        if result is None:
            missing = a if a not in tables else b
            print(f"  {frame} frame: incomplete (missing {missing})")
            continue
        n, delta, z, p = result
        verdict = "dual higher" if delta > 0 else "single higher"
        print(f"  {frame} frame  n={n}  "
              f"AUC(dual) - AUC(single) = {delta:+.3f}  z={z:+.2f}  p={p:.4f}"
              f"   [{verdict}]")

    late = [(a, b, meaning) for a, b, meaning in LATE_COMPARISONS
            if f"{prefix}{a}" in tables]
    if late:
        print("\n=== lesion-level late fusion (no reference frame; paired DeLong) ===")
        for a, b, meaning in late:
            result = compare(tables, f"{prefix}{a}", f"{prefix}{b}")
            if result is None:
                print(f"  {a} - {b:16s} incomplete (missing {prefix}{b})")
                continue
            n, delta, z, p = result
            # ASCII only: this prints to a cp936 console on the machine that
            # writes the paper, where a stray delta sign comes out as mojibake.
            print(f"  {a:17s} - {b:17s} n={n}  dAUC = {delta:+.3f}  z={z:+.2f}  "
                  f"p={p:.4f}   {meaning}")
        print("  Read the first two lines first. Without the null, a late-fusion gain "
              "cannot be\n  told apart from having 1024 dimensions; without the "
              "evaluation-time shuffle, a\n  null cannot be told apart from a probe "
              "that never looked at the NBI block.")

    print("\nThe two frames are independent comparisons; do not rank across them.")
    if late:
        print("Late fusion has no frame and is comparable with all of them: the "
              "label is the\npathology report, which belongs to neither modality.")
    if args.target == "grade":
        print("Read the positive control before reading this: "
              "python scripts/summarise_grade.py --target macro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
