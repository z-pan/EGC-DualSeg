# -*- coding: utf-8 -*-
"""Can WLI and NBI be aligned WITHOUT the ground-truth mask?

    python scripts/registration_feasibility.py --n 60

Background. Searching the optimal global affine with the true masks lifts lesion
IoU from 0.274 to 0.747, so the misalignment is largely rigid rather than a
non-rigid deformation. That number is an upper bound: at inference there is no
mask. This script asks what is reachable from the images alone, which decides
whether a coarse-alignment front end is worth building.

Three conditions, all scored the same way -- apply the estimated transform to
the NBI mask, then measure IoU against the WLI mask:

  none      no registration; the 0.274 baseline
  rigid     similarity transform (translation + rotation + isotropic scale)
            driven by Mattes mutual information on the images only
  bspline   free-form deformation on top of the rigid result, also image-only

Mutual information is the right similarity for this: the two modalities differ
mainly in colour mapping, so intensity-difference metrics are meaningless while
MI only needs a consistent statistical relationship.

The B-spline arm carries a warning. Given enough control points a non-rigid
registration can align almost anything, including things that should not be
aligned, so a higher IoU does not by itself mean a better transform. The grid
is therefore kept coarse and the Jacobian is reported: folding (negative
Jacobian) means the deformation is not physically plausible.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(REPO, "data", "packaged", "egc_dualseg_384.npz")
MANIFEST = os.path.join(REPO, "data", "packaged", "manifest.csv")


GRID = 256          # common canvas, matching the naive_iou convention


def norm_grid(arr, is_mask=False):
    from PIL import Image
    if is_mask:
        im = Image.fromarray(arr.astype(np.uint8) * 255).resize(
            (GRID, GRID), Image.NEAREST)
        return np.asarray(im) > 127
    im = Image.fromarray(arr.astype(np.float32)).resize((GRID, GRID), Image.BILINEAR)
    return np.asarray(im)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 1.0


def to_sitk(arr):
    return sitk.GetImageFromArray(arr.astype(np.float32))


def register(fixed_img, moving_img, mode, seed=0):
    """Estimate a transform from IMAGES ONLY. Returns the transform or None."""
    f, m = to_sitk(fixed_img), to_sitk(moving_img)
    f = sitk.Normalize(f)
    m = sitk.Normalize(m)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.20, seed)
    R.SetInterpolator(sitk.sitkLinear)

    init = sitk.CenteredTransformInitializer(
        f, m, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(init, inPlace=False)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    try:
        rigid = R.Execute(f, m)
    except RuntimeError:
        return None
    if mode == "rigid":
        return rigid

    # ---- free-form deformation on top of the rigid estimate ----------------
    # Deliberately coarse: a dense grid can align anything at all, which would
    # make a high IoU meaningless.
    mesh = [6, 6]
    bs = sitk.BSplineTransformInitializer(f, mesh, order=3)
    R2 = sitk.ImageRegistrationMethod()
    R2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    R2.SetMetricSamplingStrategy(R2.RANDOM)
    R2.SetMetricSamplingPercentage(0.20, seed)
    R2.SetInterpolator(sitk.sitkLinear)
    R2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                            numberOfIterations=120, maximumNumberOfCorrections=5)
    R2.SetInitialTransform(bs, inPlace=False)
    R2.SetMovingInitialTransform(rigid)
    R2.SetShrinkFactorsPerLevel([2, 1])
    R2.SetSmoothingSigmasPerLevel([1, 0])
    R2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    try:
        deform = R2.Execute(f, m)
    except RuntimeError:
        return rigid
    return sitk.CompositeTransform([rigid, deform])


def apply_to_mask(mask, ref_img, transform):
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    out = sitk.Resample(m, to_sitk(ref_img), transform, sitk.sitkNearestNeighbor,
                        0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out) > 0


def jacobian_folding(transform, ref_img):
    """Fraction of pixels where the deformation folds (Jacobian <= 0)."""
    try:
        jac = sitk.DisplacementFieldJacobianDeterminant(
            sitk.TransformToDisplacementField(
                transform, sitk.sitkVectorFloat64,
                to_sitk(ref_img).GetSize(), to_sitk(ref_img).GetOrigin(),
                to_sitk(ref_img).GetSpacing(), to_sitk(ref_img).GetDirection()))
        j = sitk.GetArrayFromImage(jac)
        return float((j <= 0).mean())
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="pairs to process")
    ap.add_argument("--out", default="results/registration_feasibility.csv")
    args = ap.parse_args()

    blob = np.load(NPZ)
    images, masks = blob["images"], blob["masks"]
    man = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    idx = {(r.case, r.scale, r.modality): int(r.idx) for r in man.itertuples()}
    geo = {(r.case, r.scale, r.modality): (int(r.pad_y), int(r.pad_x),
                                           int(r.content_h), int(r.content_w))
           for r in man.itertuples()}
    pairs = sorted({(r.case, r.scale) for r in man.itertuples()})[:args.n]

    rows = []
    for k, (case, scale) in enumerate(pairs, 1):
        kw, kn = (case, scale, "WLI"), (case, scale, "NBI")
        if kw not in idx or kn not in idx:
            continue
        # Crop the letterbox away (black borders dominate the MI estimate),
        # then put BOTH views on one common grid by resizing each cropped
        # field to GRID x GRID. This matches how naive_iou is defined -- each
        # mask normalised to the same canvas -- so the "none" row here is
        # directly comparable to the 0.274 baseline. Resampling NBI onto the
        # WLI grid by physical coordinates instead silently rescales it and
        # made the baseline read 0.171.
        gray = lambda a: a[..., :3].mean(-1)
        py, px, ch, cw = geo[kw]
        fi = norm_grid(gray(images[idx[kw]])[py:py + ch, px:px + cw])
        fm = norm_grid((masks[idx[kw]] > 127)[py:py + ch, px:px + cw], True)
        py2, px2, ch2, cw2 = geo[kn]
        mi_r = norm_grid(gray(images[idx[kn]])[py2:py2 + ch2, px2:px2 + cw2])
        mm_r = norm_grid((masks[idx[kn]] > 127)[py2:py2 + ch2, px2:px2 + cw2], True)
        if fm.sum() < 20 or mm_r.sum() < 20:
            continue

        rec = dict(case=case, scale=scale, none=iou(fm, mm_r))
        for mode in ("rigid", "bspline"):
            t = register(fi, mi_r, mode)
            if t is None:
                rec[mode] = np.nan
                continue
            rec[mode] = iou(fm, apply_to_mask(mm_r, fi, t))
            if mode == "bspline":
                rec["fold"] = jacobian_folding(t, fi)
        rows.append(rec)
        if k % 10 == 0:
            print(f"  {k}/{len(pairs)}", flush=True)

    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    d.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\npairs = {len(d)}")
    print(f"{'condition':10s} {'median':>8s} {'IQR':>18s} {'>=0.5':>7s} {'>=0.6':>7s}")
    for c in ("none", "rigid", "bspline"):
        v = d[c].dropna()
        if v.empty:
            continue
        print(f"{c:10s} {v.median():8.3f} "
              f"[{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]".rjust(0)
              + f" {(v>=0.5).mean()*100:6.0f}% {(v>=0.6).mean()*100:6.0f}%")
    if "fold" in d:
        print(f"\nB-spline folding (Jacobian<=0): median "
              f"{d['fold'].median()*100:.2f}% of pixels")
    print("\nreference: optimal affine using the TRUE masks reached 0.747 "
          "(upper bound, not attainable at inference)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
