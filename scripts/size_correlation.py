# -*- coding: utf-8 -*-
"""Does dual-modality segmentation agree better with the pathological ruler?

    python scripts/size_correlation.py
    python scripts/size_correlation.py --results results --scale near

No GPU, no checkpoints: everything comes from the committed prediction CSVs and
the manifest.

Why this endpoint and not another
--------------------------------
Every other comparison in this project is scored against the SAM-derived masks.
Those masks were reviewed case by case by the endoscopist, but they were still
produced by prompting a foundation model with a clinician's box, so their
contours carry a component of the labelling procedure — which is exactly why the
loss deliberately avoids boundary-distance objectives, and why a boundary metric
can only ever be a secondary read-out here.

The pathology report carries a different kind of number: the lesion's longer
dimension measured microscopically on the resected specimen, 2.5-33 mm, a
thirteen-fold span. It owes nothing to SAM, nothing to the endoscopist's box and
nothing to any model. If a dual-modality model tracks that ruler more closely
than a single-modality one does, that is a claim about the tissue rather than
about agreement with an annotation procedure.

It is also better powered than the grade endpoint: a correlation over ~42
lesions resolves a difference that an AUC over 48 patients cannot.

Conventions, taken unchanged from figures/fig4_segmentation.py so the numbers
are comparable with what is already in the plan (§3.3):

* only single-lesion specimen measurements in mm — '13x9' is kept, '18x8;12x8'
  and '直径1.2cm(内镜)' are not
* near focus only: at distance the working distance varies so much that a
  field-relative length stops meaning anything
* predicted size is the mask's longer bounding-box side as a percentage of the
  endoscopic field width, because the field width in mm is not known per image
* Spearman, because the two quantities are on different scales and the
  relationship need only be monotone

Comparing the two correlations needs care: they share the pathology variable, so
they are dependent. Williams' t is reported, and next to it a paired bootstrap
over patients, which assumes far less and is the one to trust if they disagree.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAMES = {"WLI": "wli_only", "NBI": "nbi_only"}
N_BOOT = 10000


def long_axis_mm(raw) -> float | None:
    """Longer dimension of the microscopic measurement, in mm, or None.

    The field mixes three things and only the first is usable:
      '13x9'            resection specimen, measured microscopically, mm
      '18x8;12x8'       multifocal — two lesions, one image, not attributable
      '直径1.2cm(内镜)'  an endoscopist's estimate, a different measurement
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if ";" in s or "(" in s:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[xX*]\s*(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    return max(float(m.group(1)), float(m.group(2)))


def gt_long_axis(npz_path: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """Longer bounding-box side of the ground-truth mask, same units as the model's.

    This is the ceiling row: what a segmenter that reproduced the annotation
    exactly would score. Without it a model correlation has no scale.
    """
    masks = np.load(npz_path)["masks"]
    rows = []
    for _, r in manifest.iterrows():
        m = masks[int(r["idx"])] > 127
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        side = max(int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1)
        rows.append(dict(case=r["case"], scale=r["scale"],
                         ref_modality=r["modality"], config="ground_truth",
                         long_frac=100 * side / r["content_w"]))
    return pd.DataFrame(rows)


def williams_t(r_dual: float, r_single: float, r_arms: float, n: int):
    """Williams' test for two dependent correlations sharing one variable."""
    if n < 6:
        return float("nan"), float("nan")
    det = (1 - r_dual ** 2 - r_single ** 2 - r_arms ** 2
           + 2 * r_dual * r_single * r_arms)
    rbar = (r_dual + r_single) / 2
    denom = (2 * ((n - 1) / (n - 3)) * det
             + rbar ** 2 * (1 - r_arms) ** 3)
    if denom <= 0:
        return float("nan"), float("nan")
    t = (r_dual - r_single) * math.sqrt((n - 1) * (1 + r_arms) / denom)
    from scipy.stats import t as tdist
    return t, 2 * tdist.sf(abs(t), df=n - 3)


def bootstrap_diff(path_mm: np.ndarray, dual: np.ndarray, single: np.ndarray,
                   seed: int = 0):
    """Paired bootstrap over lesions of (rho_dual - rho_single)."""
    rng = np.random.default_rng(seed)
    n = len(path_mm)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(path_mm[idx])) < 3:
            diffs[b] = np.nan
            continue
        diffs[b] = (spearmanr(path_mm[idx], dual[idx]).statistic
                    - spearmanr(path_mm[idx], single[idx]).statistic)
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p: how often the difference lands on the other side of 0.
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--manifest", default="data/packaged/manifest.csv")
    ap.add_argument("--npz", default="data/packaged/egc_dualseg_384.npz")
    ap.add_argument("--scale", default="near", choices=["near", "far", "both"])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "predictions_*.csv")))
    if not files:
        raise SystemExit(f"no prediction CSVs in {args.results}")
    pred = pd.concat((pd.read_csv(f, encoding="utf-8-sig") for f in files),
                     ignore_index=True)

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    ruler = (manifest.drop_duplicates("case")
             .assign(long_mm=lambda d: d.path_mm.map(long_axis_mm))
             .dropna(subset=["long_mm"]).set_index("case").long_mm)

    key = ["case", "scale", "ref_modality", "config"]
    img = pred.groupby(key, as_index=False).agg(
        pred_long_axis_px=("pred_long_axis_px", "mean"),
        content_w=("content_w", "first"))
    img["long_frac"] = 100 * img.pred_long_axis_px / img.content_w

    truth = gt_long_axis(args.npz, manifest)
    img = pd.concat([img[key + ["long_frac"]], truth], ignore_index=True)
    if args.scale != "both":
        img = img[img.scale == args.scale]
    img["path_long_mm"] = img.case.map(ruler)
    img = img.dropna(subset=["path_long_mm"])

    print(f"pathological ruler: {len(ruler)} patients with a single-lesion "
          f"specimen measurement, {ruler.min():.1f}-{ruler.max():.1f} mm")
    print(f"scale filter: {args.scale}\n")

    for frame, single in FRAMES.items():
        sub = img[img.ref_modality == frame]
        wide = sub.pivot_table(index="case", columns="config", values="long_frac")
        if single not in wide:
            print(f"=== {frame} frame: no {single} baseline ===")
            continue
        wide = wide.join(ruler.rename("path_mm"), how="inner")

        print(f"=== reference frame: {frame} ===")
        print("  Spearman rho between predicted lesion extent and the "
              "microscopically measured size")
        arms = [c for c in wide.columns if c != "path_mm"]
        rho = {}
        for cfg in sorted(arms):
            pair = wide[["path_mm", cfg]].dropna()
            if len(pair) < 6:
                continue
            r = spearmanr(pair.path_mm, pair[cfg])
            rho[cfg] = (float(r.statistic), float(r.pvalue), len(pair))
            mark = "  <- annotation ceiling" if cfg == "ground_truth" else ""
            print(f"    {cfg:22s} rho = {r.statistic:+.3f}  "
                  f"p = {r.pvalue:.4f}  n = {len(pair)}{mark}")

        def contrast(cfg: str, base: str, label: str) -> None:
            pair = wide[["path_mm", cfg, base]].dropna()
            if len(pair) < 8:
                return
            n = len(pair)
            r_a = spearmanr(pair.path_mm, pair[cfg]).statistic
            r_b = spearmanr(pair.path_mm, pair[base]).statistic
            r_ab = spearmanr(pair[cfg], pair[base]).statistic
            t, p_w = williams_t(r_a, r_b, r_ab, n)
            lo, hi, p_b = bootstrap_diff(pair.path_mm.to_numpy(),
                                         pair[cfg].to_numpy(),
                                         pair[base].to_numpy())
            print(f"    {label}:  d_rho = {r_a - r_b:+.3f}  n = {n}\n"
                  f"        Williams t = {t:+.2f}, p = {p_w:.4f}   |   "
                  f"bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], p = {p_b:.4f}")

        print(f"\n  against the {single} baseline (dependent correlations):")
        for cfg in sorted(arms):
            if cfg not in (single, "ground_truth") and cfg in rho:
                contrast(cfg, single, f"{cfg} - {single}")

        # The comparison that the ceiling row makes possible, and the one that
        # matters most given how these labels were made: does a model trained on
        # the SAM masks track the specimen better than those masks do? A model
        # cannot learn tissue truth that its target lacks, but it can average out
        # per-image annotation noise, and that would show up exactly here.
        if "ground_truth" in rho:
            print(f"\n  against the annotation itself (exploratory):")
            for cfg in sorted(arms):
                if cfg != "ground_truth" and cfg in rho:
                    contrast(cfg, "ground_truth", f"{cfg} - annotation")
        print()

    print("Trust the bootstrap over Williams' t where they disagree: Williams "
          "assumes\nbivariate normality that ranks do not satisfy. Neither is a "
          "substitute for the\nfact that this is one cohort of ~42 measurable "
          "lesions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
