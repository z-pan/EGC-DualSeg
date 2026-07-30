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

Two pilot findings are baked into this file.

Scale is not modelled. Fixing it at the 1.9x measured in the real cohort caps
achievable IoU at ~0.28 whatever the offset (1.9x linear is 3.6x in area),
making perfect registration unreachable and silently coupling scale to
misalignment.

Misalignment is produced by cropping TWO windows from the same original image,
not by translating one view with zero padding. Padding left 26% of the
auxiliary view pure black at IoU 0.28 versus 8% at IoU 1.0 -- a second variable
tracking the first. A permutation-invariant fusion operator exploited it hard
enough that misaligned beat perfectly-registered by 0.025 Dice at 6.7 standard
errors, which is physically impossible and was purely the padding. Cropping
twice keeps both views entirely real: black pixels now sit at 1.4% vs 1.6%
across levels.

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

SIZE = 256                 # synthetic study runs at 256 to keep the sweep cheap


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 1.0


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


def resize(arr, size, nearest=False):
    mode = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(Image.fromarray(arr).resize((size, size), mode))


def crop_window(arr, cy, cx, win, nearest=False):
    """Take a win x win window centred at (cy, cx); caller guarantees bounds."""
    return arr[cy - win // 2: cy - win // 2 + win,
               cx - win // 2: cx - win // 2 + win]


def window_geometry(mask, win):
    """Place the reference window on the lesion, and pick the slide direction
    that has the most room left in the original image.

    Anchoring the reference at the image centre splits the available room in
    half and the low-IoU levels then cannot be reached: the first attempt
    bottomed out at 0.28 when asked for 0.15. Centring on the lesion instead
    guarantees the reference actually contains it, and sliding away from the
    nearest border roughly doubles the usable travel.
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    cy = int(np.clip(round(ys.mean()), win // 2, H - win // 2 - 1))
    cx = int(np.clip(round(xs.mean()), win // 2, W - win // 2 - 1))
    up, down = cy - win // 2, H - (cy + win // 2)
    left, right = cx - win // 2, W - (cx + win // 2)
    sy = 1 if down >= up else -1
    sx = 1 if right >= left else -1
    room = min(down if sy > 0 else up, right if sx > 0 else left)
    return cy, cx, sy, sx, max(0, room)


def solve_window_shift(mask, win, target, size, tol=0.015, iters=40):
    """Bisect the offset between two crop windows until their masks hit target IoU.

    Two windows onto the SAME original image, rather than one window plus a
    padded translation. Padding was the previous approach and it was wrong: at
    IoU 0.28 it left 26% of the auxiliary view as pure black, which is not
    misalignment but a second variable, and a permutation-invariant fusion
    operator responded to it strongly enough to beat perfect registration.
    Cropping twice keeps both views entirely real.
    """
    cy, cx, sy, sx, room = window_geometry(mask, win)
    m8 = mask.astype(np.uint8)
    ref = resize(crop_window(m8, cy, cx, win), size, True) > 0
    if target >= 0.999 or room <= 0:
        return cy, cx, 0, 0, ref, ref.copy()

    lo, hi = 0.0, float(room) * 1.4142      # diagonal travel
    best = (0, 0, ref.copy(), 1.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        dy = int(round(sy * mid * 0.7071))
        dx = int(round(sx * mid * 0.7071))
        dy = int(np.clip(dy, -(cy - win // 2), mask.shape[0] - (cy + win // 2)))
        dx = int(np.clip(dx, -(cx - win // 2), mask.shape[1] - (cx + win // 2)))
        aux = resize(crop_window(m8, cy + dy, cx + dx, win), size, True) > 0
        cur = iou(ref, aux)
        best = (dy, dx, aux, cur)
        if abs(cur - target) < tol:
            break
        if cur > target:
            lo = mid
        else:
            hi = mid
    return cy, cx, best[0], best[1], ref, best[2]


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
            im_full = np.asarray(Image.open(fp).convert("RGB"))
            mp = os.path.join(args.kvasir, "masks", os.path.basename(fp))
            mk_full = np.asarray(Image.open(mp).convert("L")) > 127
            H, W = mk_full.shape
            if mk_full.sum() < 50:
                continue

            # Window small enough to leave room to slide.
            win = int(min(H, W) * args.win_frac)
            if win < 64:
                continue
            _cy, _cx, _sy, _sx, room = window_geometry(mk_full, win)
            if room < 8:
                continue

            cy, cx, dy, dx, mk, mk2 = solve_window_shift(mk_full, win, target, SIZE)
            got = iou(mk, mk2)
            if mk.sum() < 50 or mk2.sum() < 50:     # lesion slid out of view
                continue
            achieved.append(got)

            base_ref = resize(crop_window(im_full, cy, cx, win), SIZE)
            base_aux = resize(crop_window(im_full, cy + dy, cx + dx, win), SIZE)

            r = np.random.default_rng(args.seed + i)
            if args.complement == "S":
                keep = block_partition(SIZE, SIZE, args.block, r)
                ref = np.where(keep[..., None], base_ref, degrade(base_ref, args.blur))
                aux = np.where(keep[..., None], degrade(base_aux, args.blur), base_aux)
            else:                                                   # G
                ref = degrade(base_ref, max(1, args.blur // 2))
                aux = shuffle_tiles(degrade(base_aux, max(1, args.blur // 2)),
                                    args.tile, r)

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
    ap.add_argument("--win-frac", type=float, default=0.55,
                    help="crop window as a fraction of the short side; the "
                         "remainder is the room available to slide")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
