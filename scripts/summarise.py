# -*- coding: utf-8 -*-
"""Summarise segmentation results across configs, folds and seeds.

The unit of analysis is one (case, scale, reference modality) image. Seeds are
averaged first, so the paired tests compare configurations on the same images
rather than treating seeds as extra samples.

The two reference frames are reported separately and are NOT compared with each
other: their ground-truth masks live in different coordinate frames.

    python scripts/summarise.py
    python scripts/summarise.py --results results --min-seeds 3
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Which single-modality and early-fusion baseline belongs to which frame.
FRAMES = {
    "WLI": dict(single="wli_only", early="early_fusion_wli"),
    "NBI": dict(single="nbi_only", early="early_fusion_nbi"),
}
FUSION = "ours"
KEY = ["case", "scale", "ref_modality"]


def load(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "predictions_*.csv")))
    if not files:
        sys.exit(f"no prediction CSVs in {results_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"loaded {len(files)} files, {len(df)} rows, "
          f"{df.config.nunique()} configs, seeds {sorted(df.seed.unique())}\n")
    return df


def wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank; returns (median difference, p)."""
    d = a - b
    d = d[d != 0]
    if len(d) < 6:
        return float(np.median(a - b)), float("nan")
    try:
        from scipy.stats import wilcoxon as _w
        return float(np.median(a - b)), float(_w(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return float(np.median(a - b)), float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--metric", default="dice")
    args = ap.parse_args()

    df = load(args.results)
    m = args.metric

    # ---- seed-level means, then image-level means ------------------------
    per_seed = df.groupby(["config", "seed"])[m].mean().reset_index()
    print("=== per-configuration mean, spread across seeds ===")
    for cfg, g in per_seed.groupby("config"):
        print(f"  {cfg:20s} {g[m].mean():.4f} +/- {g[m].std(ddof=0):.4f}  "
              f"(seeds {list(np.round(g[m], 4))})")

    per_image = df.groupby(KEY + ["config"])[m].mean().reset_index()

    # ---- within each reference frame -------------------------------------
    for frame, names in FRAMES.items():
        sub = per_image[per_image.ref_modality == frame]
        wide = sub.pivot_table(index=KEY, columns="config", values=m)
        needed = [names["single"], names["early"], FUSION]
        have = [c for c in needed if c in wide.columns]
        wide = wide[have].dropna()
        if wide.empty:
            continue

        print(f"\n=== reference frame: {frame}   (n = {len(wide)} images) ===")
        print("    not comparable with the other frame: different ground-truth masks")
        for c in have:
            v = wide[c]
            print(f"  {c:20s} mean {v.mean():.4f}  median {v.median():.4f}  "
                  f"%<0.5 {100 * (v < 0.5).mean():.1f}%")

        if FUSION in have:
            for base in (names["single"], names["early"]):
                if base not in have:
                    continue
                diff, p = wilcoxon(wide[FUSION].values, wide[base].values)
                arrow = "+" if diff > 0 else ""
                print(f"  {FUSION} - {base:20s} median {arrow}{diff:+.4f}   "
                      f"Wilcoxon p = {p:.4f}" if not np.isnan(p) else
                      f"  {FUSION} - {base:20s} median {diff:+.4f}   p = n/a")

        # near / far split: the stratification behind the correspondence analysis
        print("  by working distance:")
        for scale in ("near", "far"):
            w = wide[wide.index.get_level_values("scale") == scale]
            if w.empty:
                continue
            line = "    " + f"{scale:5s} n={len(w):3d}  "
            line += "  ".join(f"{c.split('_')[0]}:{w[c].mean():.3f}" for c in have)
            if FUSION in have and names["single"] in have:
                line += f"   delta={w[FUSION].mean() - w[names['single']].mean():+.3f}"
            print(line)

    print("\nreminder: the WLI and NBI frames are two independent comparisons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
