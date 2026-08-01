# -*- coding: utf-8 -*-
"""Read out the oracle headroom probe and apply its pre-registered decision rule.

    python scripts/oracle_headroom.py
    python scripts/oracle_headroom.py --results results --fold 4

    ⚠️ EVERY NUMBER THIS PRINTS IS AN UPPER BOUND OBTAINED BY CHEATING. ⚠️

The probe aligns the auxiliary frame using the ground-truth masks of both frames
(`src/data/align.py`), at validation time as well as training time. None of it
is reportable. It answers one question and then gets out of the way:

    Is there enough headroom in an aligned high-resolution skip path to justify
    building the honest version of it?

Why the question is open. The decoder takes skips from the reference stream
only, because the auxiliary frame is unaligned — median lesion overlap 0.28 —
and routing it into the decoder injects misalignment noise silently. Fusion
therefore happens at 1/16 and 1/32 while boundary detail lives in the
high-resolution skips. Aligning the auxiliary frame lifts that constraint and
opens a path the proposed architecture structurally does not have. The
registration-free result (+0.001 / +0.009 Dice) does not bound it, and neither
does the lesion-level late-fusion null, which pools its features and so has
nothing to say about a spatial endpoint.

The three comparisons, in the order they should be read
------------------------------------------------------
1. `oracle_skip - oracle_noskip` — the skip path, with alignment and
   augmentation held identical. **This is the headroom.**
2. `oracle_skip - single` — the upper bound against the arm it would have to
   beat to matter. Carries the alignment and the augmentation change too.
3. `oracle_noskip - ours` — what oracle alignment alone buys the existing
   architecture. Expected to be small: the Kvasir misalignment study measured
   `ours` losing only 0.0051 Dice between perfect registration and 0.28.

Decision rule, fixed before the run:
    headroom <= 0.03 Dice  ->  close the aligned-fusion line; the honest version
                               can only do worse, since predicted masks reach
                               lesion IoU 0.60-0.64 against this oracle's 0.845
    headroom  > 0.03 Dice  ->  build the predicted-mask version and run 5 x 3

0.03 is the same threshold the Kvasir pilot used, and for the same reason: below
it the effect is the size of run-to-run variance and no amount of further
engineering will pull it clear.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = ["case", "scale", "ref_modality"]
HEADROOM_THRESHOLD = 0.03
SINGLE = {"WLI": "wli_only", "NBI": "nbi_only"}


def load(results_dir: str, fold: int) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "predictions_*.csv")))
    if not files:
        raise SystemExit(f"no prediction CSVs in {results_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df[df.fold == fold]


def paired(df: pd.DataFrame, a: str, b: str, frame: str) -> tuple[int, float] | None:
    """Mean per-image Dice difference between two configs, within one frame.

    Seeds are averaged per image first, exactly as in summarise.py, so this
    compares configurations on the same images rather than treating seeds as
    extra samples. With one fold there are too few images for a paired test to
    mean much, so this deliberately reports the effect size and no p value —
    the decision rule is about magnitude.
    """
    sub = df[(df.ref_modality == frame) & (df.config.isin([a, b]))]
    if sub.empty or sub.config.nunique() < 2:
        return None
    per_image = sub.groupby(KEY + ["config"]).dice.mean().reset_index()
    wide = per_image.pivot_table(index=KEY, columns="config", values="dice").dropna()
    if wide.empty:
        return None
    return len(wide), float((wide[a] - wide[b]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--fold", type=int, default=4,
                    help="the development fold the probe runs on")
    args = ap.parse_args()

    df = load(args.results, args.fold)
    present = set(df.config.unique())
    for needed in ("oracle_skip", "oracle_noskip"):
        if needed not in present:
            raise SystemExit(
                f"{needed} has not been run on fold {args.fold}.\n"
                f"  python scripts/train.py --config configs/{needed}.yaml")

    print("=" * 74)
    print("ORACLE HEADROOM PROBE: upper bound obtained with ground-truth masks.")
    print("Not a result. Do not report these numbers.")
    print(f"fold {args.fold}, seeds {sorted(df.seed.unique())}")
    print("=" * 74)

    print("\nmean Dice by config and frame")
    table = df.groupby(["config", "ref_modality"]).dice.mean().unstack()
    print(table.round(4).to_string())

    print("\n1. headroom: oracle_skip - oracle_noskip  (alignment and augmentation held fixed)")
    headroom = {}
    for frame in ("WLI", "NBI"):
        result = paired(df, "oracle_skip", "oracle_noskip", frame)
        if result is None:
            print(f"   {frame}: unavailable")
            continue
        n, delta = result
        headroom[frame] = delta
        print(f"   {frame} frame  n={n:3d} images   {delta:+.4f} Dice")

    print("\n2. oracle_skip - single-modality  (also carries alignment + augmentation)")
    for frame in ("WLI", "NBI"):
        result = paired(df, "oracle_skip", SINGLE[frame], frame)
        print(f"   {frame} frame  " + ("unavailable" if result is None else
                                       f"n={result[0]:3d} images   {result[1]:+.4f} Dice"))

    print("\n3. oracle_noskip - ours  (what oracle alignment alone buys the current model)")
    for frame in ("WLI", "NBI"):
        result = paired(df, "oracle_noskip", "ours", frame)
        print(f"   {frame} frame  " + ("unavailable" if result is None else
                                       f"n={result[0]:3d} images   {result[1]:+.4f} Dice"))

    print("\n" + "=" * 74)
    if not headroom:
        print("VERDICT: cannot be read; the headroom comparison is incomplete.")
        return 1
    best = max(headroom.values())
    print(f"largest headroom across frames: {best:+.4f} Dice "
          f"(threshold {HEADROOM_THRESHOLD:+.2f})")
    if best > HEADROOM_THRESHOLD:
        print("VERDICT: worth building. Implement the predicted-mask alignment\n"
              "(analytic 3-parameter fit, reachable lesion IoU 0.60-0.64) and run\n"
              "the full 5 folds x 3 seeds. Expect less than this: the oracle\n"
              "aligns to 0.845 and the honest version cannot.")
    else:
        print("VERDICT: close the line. This is the CEILING (the oracle aligns to\n"
              "median lesion IoU 0.845) and it does not clear the threshold, so the\n"
              "predicted-mask version, which reaches only 0.60-0.64, cannot. Aligned\n"
              "dual-modality fusion is done, and the paper says so with a measured\n"
              "bound rather than an argument.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
