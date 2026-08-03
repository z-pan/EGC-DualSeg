# -*- coding: utf-8 -*-
"""Did the second modality help the boundary, where Dice could not have seen it?

    python scripts/summarise_boundary.py
    python scripts/summarise_boundary.py --results results --metric bf3

Reads results/boundary_*.csv only: no GPU, no checkpoints. The unit of analysis
is one (case, scale, reference modality) image and seeds are averaged first, so
the paired tests compare configurations on the same images rather than treating
seeds as extra samples — identical to summarise.py, deliberately, so the two
read-outs differ in the metric and in nothing else.

The two reference frames stay separate. Their ground-truth masks live in
different coordinate frames and their boundaries do too.

What would change the project's conclusion
------------------------------------------
So far every dual-versus-single comparison has come back at roughly +0.001 to
+0.009 Dice — inside run-to-run variance. If the same arms separate on bf/nsd
while staying flat on Dice, then the effect was real all along and the metric
was hiding it, and "fusion buys nothing" has to be rewritten as "fusion buys a
boundary the area metric cannot resolve". If they stay flat here too, the
negative result gets considerably stronger, because it now covers the axis the
clinical rationale actually points at.

The size trap, restated because it has already cost this project once
------------------------------------------------------------------------
`naive_iou` correlated r = 0.575 with lesion size and was withdrawn from
correlation work; Dice itself sits at r = 0.562. hd95 is a distance and grows
with the object, so it is exposed the same way. bf and nsd are fractions of
contour length and should not be. This script prints the size correlation of
whichever metric is being read, so that is checked rather than assumed.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEY = ["case", "scale", "ref_modality"]
FRAMES = {"WLI": "wli_only", "NBI": "nbi_only"}
# Higher is better for every metric except hd95.
LOWER_IS_BETTER = {"hd95"}
DEFAULT_METRICS = ["dice", "hd95", "bf1", "bf2", "bf3", "bf5", "nsd3"]


def load(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "boundary_*.csv")))
    if not files:
        raise SystemExit(
            f"no boundary_*.csv in {results_dir}. These are recomputed from the "
            "weights, so run scripts/boundary_metrics.py where the checkpoints live.")
    df = pd.concat((pd.read_csv(f, encoding="utf-8-sig") for f in files),
                   ignore_index=True)
    print(f"loaded {len(files)} files, {len(df)} rows, "
          f"{df.config.nunique()} configs, seeds {sorted(df.seed.unique())}")
    degenerate = int(df.degenerate.sum())
    if degenerate:
        print(f"  {degenerate} rows have an empty predicted mask; they score 0 on "
              f"bf/nsd and NaN on hd95 rather than being dropped")
    return df


def paired(per_image: pd.DataFrame, a: str, b: str, metric: str):
    wide = per_image.pivot_table(index=KEY, columns="config", values=metric)
    if a not in wide or b not in wide:
        return None
    wide = wide[[a, b]].dropna()
    if len(wide) < 5:
        return None
    diff = wide[a] - wide[b]
    if np.allclose(diff, 0):
        return len(wide), 0.0, 1.0
    return len(wide), float(diff.median()), float(
        wilcoxon(wide[a], wide[b], zero_method="wilcox").pvalue)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--dual", nargs="+", default=None,
                    help="which arms to test against the single-modality baseline")
    args = ap.parse_args()

    df = load(args.results)
    configs = sorted(df.config.unique())
    duals = args.dual or [c for c in configs if c not in FRAMES.values()]

    for frame, single in FRAMES.items():
        sub = df[df.ref_modality == frame]
        if sub.empty or single not in set(sub.config):
            print(f"\n=== {frame} frame: no {single} baseline, skipping ===")
            continue
        print(f"\n=== reference frame: {frame} "
              f"(baseline {single}, n = {sub.groupby(KEY).ngroups} images) ===")
        print("    the two frames have different masks and are not comparable")

        for metric in args.metrics:
            per_image = sub.groupby(KEY + ["config"], as_index=False)[metric].mean()
            arrow = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
            means = per_image.groupby("config")[metric].mean()
            print(f"\n  -- {metric} ({arrow}) --")
            for cfg in sorted(means.index):
                print(f"     {cfg:22s} {means[cfg]:.4f}")
            for cfg in duals:
                if cfg not in set(per_image.config):
                    continue
                result = paired(per_image, cfg, single, metric)
                if result is None:
                    continue
                n, delta, p = result
                better = (delta < 0) if metric in LOWER_IS_BETTER else (delta > 0)
                flag = "dual better" if better else "single better"
                print(f"     {cfg} - {single}: median {delta:+.4f}  "
                      f"p = {p:.4f}  [{flag}]")

            # The lesson from naive_iou: a metric that tracks lesion size cannot
            # carry a comparison between arms that segment different amounts.
            base = per_image[per_image.config == single].set_index(KEY)
            size = sub.groupby(KEY).gt_area_frac.mean()
            joined = base[metric].to_frame().join(size.rename("size")).dropna()
            if len(joined) > 5 and joined[metric].std() > 0 and joined["size"].std() > 0:
                r = float(np.corrcoef(joined[metric], joined["size"])[0, 1])
                note = "  <-- size-dependent, read with care" if abs(r) > 0.4 else ""
                print(f"     [correlation with lesion size r = {r:+.3f}]{note}")

    print("\nRead bf/nsd before hd95: they are contour-length fractions and do not "
          "track lesion size,\nwhich hd95 does. A separation on bf/nsd with none on "
          "dice is the outcome that\nwould overturn 'fusion buys nothing'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
