# -*- coding: utf-8 -*-
"""Oracle alignment of the auxiliary frame, for measuring headroom only.

    ⚠️ EVERYTHING HERE USES THE GROUND-TRUTH MASK OF BOTH FRAMES. ⚠️

    It is cheating by construction and no number produced with it may be
    reported as a result. Its only purpose is to put an upper bound on one
    specific open question, cheaply, before anyone builds the real thing.

The question
------------
The decoder's skip connections come from the reference stream only. That is a
deliberate constraint (`net.py`, `CLAUDE.md`): the auxiliary frame is not
spatially aligned with the target mask — median lesion overlap 0.28 — so routing
it into the decoder injects misalignment noise straight into the output, and it
fails silently. Cross-attention therefore happens at 1/16 and 1/32 only, while
boundary detail lives in the high-resolution skips.

If the auxiliary frame were aligned, that constraint would lift, and the model
would gain an information path the proposed architecture structurally does not
have. Nobody has measured whether that path is worth anything. Registration
from the images alone reaches lesion IoU 0.31; the analytic 3-parameter fit on
*predicted* masks reaches 0.60–0.64; the affine fitted on *true* masks is the
ceiling. This module implements that ceiling.

**Measured 2026-08-01: median lesion IoU 0.845 over all 300 directions**, at
full resolution, against 0.28 unaligned. That is higher than the 0.747–0.752
recorded earlier for "the optimal global affine", because this fit is a full
six-parameter affine from matched second moments and is then refined against
the IoU objective itself. A more generous ceiling is the right thing for a
headroom probe — it widens the gap to the honest 0.60–0.64 and so makes a
"close the line" verdict harder to argue with, not easier.

Read the achieved IoU that `optimal_affine` returns. If a future cohort comes
back far below this, the oracle is not an upper bound and the probe means
nothing.

Method
------
Moment matching, then a small refinement. The affine that carries the auxiliary
mask's ellipse of inertia onto the reference mask's is available in closed form
from the two centroids and covariances; lesions are not ellipses, so translation
and isotropic scale are then refined by direct search on the IoU itself, at
reduced resolution because the objective is smooth at this scale.

Rotation is left to the moment fit. The two frames are the same lesion seen
seconds apart through the same endoscope, so gross rotation is rare, and letting
a direct search chase it would find shape coincidences rather than pose.
"""
from __future__ import annotations

import numpy as np

# Below this many lesion pixels the second moments are not estimable and the
# oracle degenerates into noise; such a pair is left unaligned and reported.
MIN_MASK_PX = 32
REFINE_GRID = 96          # resolution at which the IoU objective is searched


def _moments(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Centroid and covariance of the foreground, in (row, col) pixel units."""
    ys, xs = np.nonzero(mask)
    n = ys.size
    if n < MIN_MASK_PX:
        return np.zeros(2), np.eye(2), n
    pts = np.stack([ys, xs]).astype(np.float64)
    mu = pts.mean(axis=1)
    cov = np.cov(pts) + np.eye(2) * 1e-6
    return mu, cov, n


def _sqrtm_spd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    return (v * np.sqrt(np.clip(w, 1e-12, None))) @ v.T


def _invsqrtm_spd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    return (v / np.sqrt(np.clip(w, 1e-12, None))) @ v.T


def warp_affine(img: np.ndarray, a_inv: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Resample `img` through `a_inv`, which maps destination (row, col) to source.

    Bilinear, zero outside. Same convention as `transforms._affine`, which this
    deliberately mirrors rather than reuses: that one is parameterised by angle
    and scale, this one by a general 2x3 matrix.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src_y = a_inv[0, 0] * yy + a_inv[0, 1] * xx + a_inv[0, 2]
    src_x = a_inv[1, 0] * yy + a_inv[1, 1] * xx + a_inv[1, 2]

    if nearest:
        iy = np.rint(src_y).astype(np.int32)
        ix = np.rint(src_x).astype(np.int32)
        valid = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
        out = np.zeros_like(img)
        out[valid] = img[np.clip(iy, 0, h - 1)[valid], np.clip(ix, 0, w - 1)[valid]]
        return out

    y0 = np.floor(src_y).astype(np.int32)
    x0 = np.floor(src_x).astype(np.int32)
    wy, wx = (src_y - y0), (src_x - x0)
    if img.ndim == 3:
        wy, wx = wy[..., None], wx[..., None]
    y1, x1 = y0 + 1, x0 + 1
    valid = (y0 >= 0) & (y1 < h) & (x0 >= 0) & (x1 < w)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)

    src = img.astype(np.float32)
    top = src[y0c, x0c] * (1 - wx) + src[y0c, x1c] * wx
    bot = src[y1c, x0c] * (1 - wx) + src[y1c, x1c] * wx
    out = top * (1 - wy) + bot * wy
    out[~valid] = 0
    return out.astype(img.dtype)


def _compose(m_inv: np.ndarray, mu_ref: np.ndarray, mu_aux: np.ndarray,
             scale: float, shift: np.ndarray) -> np.ndarray:
    """a_inv for `p_aux = M^-1 ((p_ref - shift - mu_ref) / scale) + mu_aux`."""
    linear = m_inv / scale
    offset = mu_aux - linear @ (mu_ref + shift)
    return np.concatenate([linear, offset[:, None]], axis=1)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def _downsample(mask: np.ndarray, size: int) -> np.ndarray:
    h, w = mask.shape
    ys = (np.arange(size) * h / size).astype(np.int32)
    xs = (np.arange(size) * w / size).astype(np.int32)
    return mask[np.ix_(ys, xs)]


def optimal_affine(ref_mask: np.ndarray, aux_mask: np.ndarray,
                   refine: bool = True) -> tuple[np.ndarray, float]:
    """Best affine carrying the auxiliary frame onto the reference frame.

    Returns (a_inv, achieved_iou). `a_inv` maps reference pixel coordinates to
    auxiliary ones and is what `warp_affine` consumes. `achieved_iou` is the
    overlap of the warped auxiliary mask with the reference mask, and is the
    number that says whether this oracle is worth its name.

    A pair whose masks are too small to estimate second moments is returned
    unaligned (identity), with its unaligned IoU, rather than warped by a fit
    that means nothing.
    """
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ref_mask, aux_mask = ref_mask.astype(bool), aux_mask.astype(bool)

    mu_r, cov_r, n_r = _moments(ref_mask)
    mu_a, cov_a, n_a = _moments(aux_mask)
    if n_r < MIN_MASK_PX or n_a < MIN_MASK_PX:
        return identity, _iou(ref_mask, aux_mask)

    m = _sqrtm_spd(cov_r) @ _invsqrtm_spd(cov_a)          # aux-centred -> ref-centred
    m_inv = np.linalg.inv(m)

    best = _compose(m_inv, mu_r, mu_a, 1.0, np.zeros(2))
    small_ref = _downsample(ref_mask, REFINE_GRID)
    ratio = ref_mask.shape[0] / REFINE_GRID

    def score(a_inv: np.ndarray) -> float:
        warped = warp_affine(aux_mask.astype(np.uint8), a_inv, nearest=True).astype(bool)
        return _iou(small_ref, _downsample(warped, REFINE_GRID))

    def full_res_iou(a_inv: np.ndarray) -> float:
        warped = warp_affine(aux_mask.astype(np.uint8), a_inv, nearest=True).astype(bool)
        return _iou(ref_mask, warped)

    best_iou = score(best)
    if not refine:
        return best, full_res_iou(best)

    # Coordinate descent on translation and isotropic scale, coarse to fine.
    # The moment fit already sets orientation and aspect; what it cannot know is
    # that lesions are not ellipses, and that residual is mostly a shift.
    scale, shift = 1.0, np.zeros(2)
    for step_px, step_s in ((8.0 * ratio, 0.08), (3.0 * ratio, 0.03), (1.0 * ratio, 0.01)):
        improved = True
        while improved:
            improved = False
            for delta in (np.array([step_px, 0.0]), np.array([-step_px, 0.0]),
                          np.array([0.0, step_px]), np.array([0.0, -step_px])):
                cand = _compose(m_inv, mu_r, mu_a, scale, shift + delta)
                value = score(cand)
                if value > best_iou + 1e-6:
                    best_iou, best, shift, improved = value, cand, shift + delta, True
            for ds in (step_s, -step_s):
                if scale + ds <= 0.2:
                    continue
                cand = _compose(m_inv, mu_r, mu_a, scale + ds, shift)
                value = score(cand)
                if value > best_iou + 1e-6:
                    best_iou, best, scale, improved = value, cand, scale + ds, True
    # The search objective is evaluated at REFINE_GRID, where downsampling hides
    # thin mismatches and reads a few points high. Report the honest number.
    return best, full_res_iou(best)


def build_affines(masks: np.ndarray, pairs, modalities=("WLI", "NBI")
                  ) -> tuple[dict, dict]:
    """Oracle affines for every (pair, reference modality). Slow; cache it.

    Returns (affines, achieved) keyed by (case, scale, reference). About 0.4 s
    per direction, so a few minutes for the cohort — worth doing once rather
    than once per run, and definitely rather than once per dataloader worker.
    """
    affines, achieved = {}, {}
    for pair in pairs:
        for ref in modalities:
            aux = modalities[1 - modalities.index(ref)]
            a_inv, iou = optimal_affine(masks[pair.idx[ref]] > 127,
                                        masks[pair.idx[aux]] > 127)
            affines[(pair.case, pair.scale, ref)] = a_inv
            achieved[(pair.case, pair.scale, ref)] = iou
    return affines, achieved


def load_or_build_affines(cache_path: str, masks: np.ndarray, pairs,
                          verbose: bool = True) -> tuple[dict, dict]:
    """Same, memoised to an .npz next to the packaged data (gitignored)."""
    import os

    keys = [(p.case, p.scale, ref) for p in pairs for ref in ("WLI", "NBI")]
    if os.path.isfile(cache_path):
        blob = np.load(cache_path, allow_pickle=True)
        cached_keys = [tuple(k) for k in blob["keys"]]
        if set(cached_keys) >= set(keys):
            affines = dict(zip(cached_keys, blob["affines"]))
            achieved = dict(zip(cached_keys, blob["achieved"]))
            return affines, achieved

    if verbose:
        print(f"  building oracle affines for {len(pairs)} pairs "
              f"(one-off, ~{0.8 * len(pairs):.0f}s) -> {cache_path}")
    affines, achieved = build_affines(masks, pairs)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    ordered = list(affines)
    np.savez_compressed(
        cache_path,
        keys=np.array([list(k) for k in ordered], dtype=object),
        affines=np.stack([affines[k] for k in ordered]),
        achieved=np.array([achieved[k] for k in ordered], np.float64))
    return affines, achieved
