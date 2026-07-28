# -*- coding: utf-8 -*-
"""Build the controlled-misalignment dataset from Kvasir-SEG.

    python scripts/make_synthetic.py --iou 1.0 0.28 --complement S --n 500

Why this exists: the real cohort gives ONE point on the "misalignment vs fusion
damage" curve (lesion IoU 0.28, early fusion -0.088). One point is not a
relationship. Here the misalignment is dialled to order on data with true masks,
so the curve can be traced and the real cohort placed on it.

The single design decision everything rests on
----------------------------------------------
If the auxiliary view were just a geometric transform of the reference, it would
be information-theoretically redundant -- I(aux; mask | ref) = 0 -- so fusion
could only ever hurt, never help. The "benefit" half of the curve would not
exist and there would be no crossover to find. So the second view must carry
information the first one lacks, in one of two flavours:

  S (spatial)  reference and auxiliary keep complementary halves of a random
               block partition; the other half is degraded. Recovering the full
               boundary requires combining them AT THE RIGHT PLACE.
               -> should be highly sensitive to misalignment.

  G (global)   the auxiliary view is degraded, then its tiles are shuffled on a
               fixed grid: spatial structure destroyed, texture statistics kept.
               -> a bag of features, usable only by a permutation-invariant
               operator. Should be insensitive to misalignment.

Misalignment is solved per image, not applied as a fixed shift: the IoU response
to a given translation depends on lesion size and shape, so a constant shift
would smear each nominal level across a wide range of actual overlaps and the
levels would stop being levels. Each image is bisected to its target IoU and the
achieved value is recorded for plotting.

Scale ratio is deliberately NOT fixed at the 1.9x measured in the real cohort:
the pilot showed that doing so caps achievable IoU at ~0.28 regardless of
translation (a 1.9x linear scale is 3.6x in area), which makes the
perfect-registration level unreachable and stops misalignment from being a
single controllable variable. Scale is a separate axis, left at 1.0 here.

Outputs data/synth/{tag}.npz + manifest, in the same schema the real dataset
uses, so the model, training loop and summariser run unchanged.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image

# Scale ratio defaults to 1.0, NOT the 1.9x measured in the real cohort.
# Found during the pilot: fixing scale at 1.9 makes the lesion 3.6x larger in
# area, which caps achievable IoU at ~1/3.6 = 0.28 no matter the translation.
# The "perfect registration" level then becomes unreachable and misalignment
# stops being a single controllable variable. Scale is therefore separated out
# and left at 1.0 for the sweep; it can be re-introduced as its own axis later.
SIZE = 256                 # synthetic study runs at 256 to keep the sweep cheap


def warp(arr, dx, dy, scale, order_nearest=False):
    """Translate + scale about the image centre, zero padded."""
    h, w = arr.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_y = (yy - cy - dy) / scale + cy
    src_x = (xx - cx - dx) / scale + cx
    iy = np.rint(src_y).astype(np.int32)
    ix = np.rint(src_x).astype(np.int32)
    ok = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
    out = np.zeros_like(arr)
    iyc, ixc = np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)
    if arr.ndim == 3:
        out[ok] = arr[iyc[ok], ixc[ok]]
    else:
        out[ok] = arr[iyc[ok], ixc[ok]]
    return out


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 1.0


def solve_shift(mask, target_iou, scale, tol=0.015, iters=40):
    """Bisect the translation magnitude until the warped mask hits target IoU.

    Direction is fixed (diagonal) so the only free parameter is magnitude and
    the search is monotone; a random direction per image would make the
    achieved IoU depend on lesion anisotropy in a way that is hard to control.
    """
    if target_iou >= 0.999:
        return 0.0, 0.0
    h, w = mask.shape
    ux, uy = 0.7071, 0.7071
    lo, hi = 0.0, float(max(h, w))
    for _ in range(iters):
        mid = (lo + hi) / 2
        m2 = warp(mask.astype(np.uint8), ux * mid, uy * mid, scale,
                  order_nearest=True).astype(bool)
        cur = iou(mask, m2)
        if abs(cur - target_iou) < tol:
            return ux * mid, uy * mid
        if cur > target_iou:      # still too aligned -> push further
            lo = mid
        else:
            hi = mid
    return ux * (lo + hi) / 2, uy * (lo + hi) / 2


def degrade(img, strength):
    """Blur + contrast loss. The 'missing' half of the complementary pair."""
    x = img.astype(np.float32)
    k = int(strength)
    if k >= 1:                                  # cheap box blur, repeated
        for _ in range(2):
            pad = np.pad(x, ((k, k), (k, k), (0, 0)), mode="edge")
            acc = np.zeros_like(x)
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    acc += pad[k + dy:k + dy + x.shape[0], k + dx:k + dx + x.shape[1]]
            x = acc / ((2 * k + 1) ** 2)
    m = x.mean()
    x = (x - m) * 0.45 + m                      # contrast crush
    return np.clip(x, 0, 255).astype(np.uint8)


def block_partition(h, w, block, rng):
    """Random complementary block mask: True where the reference stays sharp."""
    gh, gw = (h + block - 1) // block, (w + block - 1) // block
    grid = rng.random((gh, gw)) < 0.5
    return np.kron(grid, np.ones((block, block), bool))[:h, :w]


def shuffle_tiles(img, tile, rng):
    """Destroy spatial structure, keep texture statistics."""
    h, w = img.shape[:2]
    gh, gw = h // tile, w // tile
    out = img.copy()
    tiles = [(r, c) for r in range(gh) for c in range(gw)]
    perm = rng.permutation(len(tiles))
    src = out.copy()
    for i, (r, c) in enumerate(tiles):
        r2, c2 = tiles[perm[i]]
        out[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = \
            src[r2 * tile:(r2 + 1) * tile, c2 * tile:(c2 + 1) * tile]
    return out


def build(args):
    files = sorted(glob.glob(os.path.join(args.kvasir, "images", "*")))
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in rng.permutation(len(files))[:args.n]]
    print(f"Kvasir images: {len(files)}  target IoU levels: {args.iou}  "
          f"complement: {args.complement}")

    os.makedirs(args.out, exist_ok=True)
    for target in args.iou:
        tag = f"synth_{args.complement}_iou{int(round(target*100)):03d}"
        images, masks, rows = [], [], []
        achieved = []
        for i, fp in enumerate(files):
            im = np.asarray(Image.open(fp).convert("RGB").resize((SIZE, SIZE),
                                                                 Image.BILINEAR))
            mp = os.path.join(args.kvasir, "masks", os.path.basename(fp))
            mk = np.asarray(Image.open(mp).convert("L").resize((SIZE, SIZE),
                                                               Image.NEAREST)) > 127
            if mk.sum() < 50:
                continue

            r = np.random.default_rng(args.seed + i)
            if args.complement == "S":
                keep = block_partition(SIZE, SIZE, args.block, r)
                blurred = degrade(im, args.blur)
                ref = np.where(keep[..., None], im, blurred)
                aux_full = np.where(keep[..., None], blurred, im)   # complementary
            else:                                                   # G
                ref = degrade(im, max(1, args.blur // 2))
                aux_full = shuffle_tiles(degrade(im, max(1, args.blur // 2)),
                                         args.tile, r)

            dx, dy = solve_shift(mk, target, args.scale)
            aux = warp(aux_full, dx, dy, args.scale)
            mk2 = warp(mk.astype(np.uint8), dx, dy, args.scale,
                       order_nearest=True).astype(bool)
            achieved.append(iou(mk, mk2))

            case = f"k{i:04d}"
            for modality, arr, mm in (("WLI", ref, mk), ("NBI", aux, mk2)):
                rows.append(dict(idx=len(images), case=case, scale="near",
                                 modality=modality, fov_w=SIZE, fov_h=SIZE,
                                 content_w=SIZE, content_h=SIZE, pad_x=0, pad_y=0,
                                 target_iou=target,
                                 achieved_iou=round(achieved[-1], 4)))
                images.append(arr)
                masks.append((mm * 255).astype(np.uint8))

        np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                            images=np.stack(images), masks=np.stack(masks))
        with open(os.path.join(args.out, tag + "_manifest.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        # folds: one image = one case, patient-level split is trivially satisfied
        cases = sorted({r["case"] for r in rows})
        with open(os.path.join(args.out, tag + "_folds.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["case", "fold"])
            for j, c in enumerate(cases):
                wr.writerow([c, j % 5])

        ach = np.array(achieved)
        print(f"  {tag}: {len(achieved)} pairs | achieved IoU "
              f"median {np.median(ach):.3f} "
              f"[{np.percentile(ach,10):.3f}, {np.percentile(ach,90):.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvasir",
                    default=r"C:\Users\zpanp\projects\datasets\gastric_cancer\Kvasir-SEG")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--iou", type=float, nargs="+", default=[1.0, 0.28])
    ap.add_argument("--complement", choices=["S", "G"], default="S")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--blur", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="aux lesion scale ratio; 1.0 keeps misalignment a "
                         "single variable (see note at top of file)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
