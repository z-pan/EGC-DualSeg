# -*- coding: utf-8 -*-
"""Would this resection have taken the whole lesion? Read the error asymmetrically.

    python scripts/summarise_clinical.py
    python scripts/summarise_clinical.py --results results --margins 0 5 10 20 40

No GPU: reads the boundary_*.csv written by scripts/boundary_metrics.py.

Dice, boundary-F and hd95 all treat a missed pixel of lesion and a wrongly taken
pixel of normal mucosa as the same mistake. Endoscopic submucosal dissection does
not. Missed lesion is residual disease and a positive margin, and means further
treatment; extra margin is healthy mucosa, which the endoscopist removes on
purpose — the lesion is marked several millimetres outside its edge before
cutting. A metric that averages those two is answering a question nobody asks.

So this reads three things instead:

  margin adequacy    with a k-pixel safety margin added to the model's contour,
                     is the whole lesion inside it? One yes or no per image, so
                     the arms are compared with McNemar on the images where they
                     disagree. This is the number a clinician can act on.
  required margin    the margin that would have been needed — the largest
                     distance from an uncovered lesion pixel to the prediction.
                     Adequacy at k is just this thresholded, so it carries the
                     same information with more resolution.
  over-segmentation  normal tissue taken, as a multiple of lesion area, and how
                     far past the lesion it reaches. Reported so that a gain in
                     coverage bought purely by predicting everything is visible.

Seeds are averaged per image before anything is thresholded, matching
summarise.py, and the two reference frames stay separate as always.

On units: margins are pixels of the 384 canvas. The endoscopic field spans
roughly 11-23 mm across at near focus (§3.2), so one canvas pixel is very
approximately 30-60 um and a clinical 3 mm margin is of order 50-100 px. That
conversion is too loose to report as millimetres, which is why the sweep is in
pixels — but it does mean the interesting comparisons are at the larger k, not
at k = 0.

The ground truth is still the SAM-derived mask. Coverage is much less exposed to
that than a boundary metric — it asks whether a region is enclosed, not whether
two contours have the same shape — but it does not escape it, and the write-up
should say so.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

KEY = ["case", "scale", "ref_modality"]
FRAMES = {"WLI": "wli_only", "NBI": "nbi_only"}
DEFAULT_MARGINS = [0, 5, 10, 20, 40]


def load(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "boundary_*.csv")))
    if not files:
        raise SystemExit(
            f"no boundary_*.csv in {results_dir}. Run scripts/boundary_metrics.py "
            "where the checkpoints live — and if those files predate the clinical "
            "columns, re-run it with --force.")
    df = pd.concat((pd.read_csv(f, encoding="utf-8-sig") for f in files),
                   ignore_index=True)
    if "residual_px" not in df.columns:
        raise SystemExit(
            "these boundary_*.csv have no clinical columns; re-run "
            "scripts/boundary_metrics.py --force to add them")
    print(f"loaded {len(files)} files, {len(df)} rows, {df.config.nunique()} configs")
    return df


def mcnemar(dual: np.ndarray, single: np.ndarray):
    """Exact McNemar on paired yes/no outcomes. Returns (b, c, p)."""
    b = int(((dual == 1) & (single == 0)).sum())      # dual rescued it
    c = int(((dual == 0) & (single == 1)).sum())      # dual lost it
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--margins", type=int, nargs="+", default=DEFAULT_MARGINS)
    args = ap.parse_args()

    df = load(args.results)
    per_image = df.groupby(KEY + ["config"], as_index=False).agg(
        residual_px=("residual_px", "mean"),
        residual_area_frac=("residual_area_frac", "mean"),
        over_area_ratio=("over_area_ratio", "mean"),
        over_depth_p95_px=("over_depth_p95_px", "mean"),
        dice=("dice", "mean"))

    for frame, single in FRAMES.items():
        sub = per_image[per_image.ref_modality == frame]
        if single not in set(sub.config):
            print(f"\n=== {frame} frame: no {single} baseline ===")
            continue
        duals = sorted(c for c in sub.config.unique() if c != single)
        print(f"\n{'=' * 72}\n=== reference frame: {frame}   baseline {single} ===")
        print("    the two frames have different masks and are not comparable")

        print(f"\n  -- margin adequacy: whole lesion inside the prediction "
              f"dilated by k px --")
        header = "     " + f"{'arm':22s}" + "".join(f"k={k:<3d}   " for k in args.margins)
        print(header)
        wide_res = sub.pivot_table(index=KEY, columns="config", values="residual_px")
        for cfg in [single] + duals:
            if cfg not in wide_res:
                continue
            col = wide_res[cfg].dropna()
            cells = "".join(f"{100 * (col <= k).mean():5.1f}%  " for k in args.margins)
            print(f"     {cfg:22s}{cells}")

        for cfg in duals:
            pair = wide_res[[cfg, single]].dropna()
            if pair.empty:
                continue
            line = []
            for k in args.margins:
                b, c, p = mcnemar((pair[cfg] <= k).to_numpy().astype(int),
                                  (pair[single] <= k).to_numpy().astype(int))
                delta = 100 * ((pair[cfg] <= k).mean() - (pair[single] <= k).mean())
                star = "*" if p < 0.05 else " "
                line.append(f"k={k}: {delta:+5.1f}pp (+{b}/-{c}, p={p:.3f}){star}")
            print(f"\n     {cfg} - {single}, n = {len(pair)}")
            for entry in line:
                print(f"       {entry}")

        print(f"\n  -- required margin (px), lower is better --")
        for metric, label in (("residual_px", "required margin"),
                              ("residual_area_frac", "missed lesion / lesion"),
                              ("over_area_ratio", "normal tissue taken / lesion"),
                              ("over_depth_p95_px", "over-call reach p95 (px)")):
            wide = sub.pivot_table(index=KEY, columns="config", values=metric)
            medians = "   ".join(f"{c}={wide[c].median():.3f}"
                                 for c in [single] + duals if c in wide)
            print(f"     {label:28s} {medians}")
            for cfg in duals:
                pair = wide[[cfg, single]].dropna()
                if len(pair) < 5:
                    continue
                delta = float((pair[cfg] - pair[single]).median())
                p = wilcoxon_p(pair[cfg].to_numpy(), pair[single].to_numpy())
                star = "*" if p < 0.05 else " "
                print(f"       {cfg} - {single}: median {delta:+.3f}  "
                      f"p = {p:.4f}  n = {len(pair)}{star}")

    print("\n" + "=" * 72)
    print("Read adequacy at the larger k: a 3 mm clinical margin is of order 50-100 px\n"
          "on this canvas, so k = 0 asks for something the procedure never asks for.\n"
          "A gain in coverage that comes with a matching rise in normal tissue taken is\n"
          "a threshold moved, not a model improved — the last two rows are there to\n"
          "make that visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
