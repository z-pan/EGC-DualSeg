# -*- coding: utf-8 -*-
"""Augmentation for unaligned modality pairs.

Two rules that are specific to this problem:

1. The reference frame and its mask share one geometric transform. The auxiliary
   frame gets its own, milder one. The two frames are already unregistered, so
   forcing a shared transform preserves nothing; giving the auxiliary frame an
   independent jitter augments the distribution of misalignment the model must
   tolerate.
2. Hue jitter is capped tightly. The colour difference between the two
   illumination modes is the signal — the enhanced mode differs from white light
   mainly in its green/blue balance — and a wide hue jitter would wash exactly
   that away.

Rule 1 has an exception, and it must be switched on deliberately:
`share_aux_geometry=True` gives the auxiliary frame the reference's transform
instead of its own. That is wrong for unaligned pairs — it preserves nothing and
throws away the misalignment augmentation — but it is *required* once the
auxiliary frame has been aligned to the reference (`dataset.oracle_align`),
because an independent jitter would immediately undo the alignment being
measured.

Implemented with numpy so the data pipeline has no OpenCV dependency.
"""
from __future__ import annotations

import numpy as np


def _rot90(img: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(img, k, axes=(0, 1)).copy()


def _affine(img: np.ndarray, angle: float, scale: float, order_nearest: bool) -> np.ndarray:
    """Rotate about the centre and scale, with zero padding."""
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta) / scale, np.sin(theta) / scale

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ys, xs = yy - cy, xx - cx
    src_y = cos_t * ys - sin_t * xs + cy
    src_x = sin_t * ys + cos_t * xs + cx

    if order_nearest:
        iy = np.rint(src_y).astype(np.int32)
        ix = np.rint(src_x).astype(np.int32)
        valid = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
        out = np.zeros_like(img)
        out[valid] = img[np.clip(iy, 0, h - 1)[valid], np.clip(ix, 0, w - 1)[valid]]
        return out

    y0 = np.floor(src_y).astype(np.int32)
    x0 = np.floor(src_x).astype(np.int32)
    wy = (src_y - y0)[..., None]
    wx = (src_x - x0)[..., None]
    y1, x1 = y0 + 1, x0 + 1
    valid = (y0 >= 0) & (y1 < h) & (x0 >= 0) & (x1 < w)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    src = img.astype(np.float32)
    top = src[y0c, x0c] * (1 - wx) + src[y0c, x1c] * wx
    bot = src[y1c, x0c] * (1 - wx) + src[y1c, x1c] * wx
    out = (top * (1 - wy) + bot * wy)
    out[~valid] = 0
    return out.astype(img.dtype)


def _photometric(img: np.ndarray, rng: np.random.Generator,
                 brightness: float, contrast: float, hue_deg: float) -> np.ndarray:
    x = img.astype(np.float32)
    x *= 1.0 + rng.uniform(-brightness, brightness)
    mean = x.mean()
    x = (x - mean) * (1.0 + rng.uniform(-contrast, contrast)) + mean
    if hue_deg > 0:
        # Small-angle hue rotation in RGB, cheap and adequate for +/- a few degrees.
        theta = np.deg2rad(rng.uniform(-hue_deg, hue_deg))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        one_third = 1.0 / 3.0
        sqrt_third = np.sqrt(one_third)
        a = cos_t + (1.0 - cos_t) * one_third
        b = one_third * (1.0 - cos_t) - sqrt_third * sin_t
        c = one_third * (1.0 - cos_t) + sqrt_third * sin_t
        m = np.array([[a, b, c], [c, a, b], [b, c, a]], np.float32)
        x = x @ m.T
    return np.clip(x, 0, 255).astype(np.uint8)


class PairAugment:
    """Geometric + photometric augmentation for a (reference, mask, auxiliary) triple."""

    def __init__(self, train: bool = True, seed: int = 0,
                 max_angle: float = 15.0, scale_range: tuple[float, float] = (0.85, 1.15),
                 aux_angle: float = 7.0, aux_scale: tuple[float, float] = (0.92, 1.08),
                 brightness: float = 0.20, contrast: float = 0.20, hue_deg: float = 5.0,
                 share_aux_geometry: bool = False):
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.max_angle, self.scale_range = max_angle, scale_range
        self.aux_angle, self.aux_scale = aux_angle, aux_scale
        self.brightness, self.contrast, self.hue_deg = brightness, contrast, hue_deg
        self.share_aux_geometry = share_aux_geometry

    def __call__(self, ref_img: np.ndarray, ref_mask: np.ndarray, aux_img: np.ndarray):
        if not self.train:
            return ref_img, ref_mask, aux_img
        rng = self.rng

        # --- reference frame and mask: one shared geometric transform ---------
        flip = rng.random() < 0.5
        if flip:
            ref_img, ref_mask = ref_img[:, ::-1].copy(), ref_mask[:, ::-1].copy()
        k = int(rng.integers(4))
        if k:
            ref_img, ref_mask = _rot90(ref_img, k), _rot90(ref_mask, k)
        angle = rng.uniform(-self.max_angle, self.max_angle)
        scale = rng.uniform(*self.scale_range)
        ref_img = _affine(ref_img, angle, scale, order_nearest=False)
        ref_mask = _affine(ref_mask.astype(np.uint8), angle, scale,
                           order_nearest=True).astype(bool)

        if self.share_aux_geometry:
            # The auxiliary frame has already been aligned to the reference; an
            # independent jitter here would undo exactly what is being measured.
            if flip:
                aux_img = aux_img[:, ::-1].copy()
            if k:
                aux_img = _rot90(aux_img, k)
            aux_img = _affine(aux_img, angle, scale, order_nearest=False)
        else:
            # --- auxiliary frame: its own, milder transform -------------------
            if rng.random() < 0.5:
                aux_img = aux_img[:, ::-1].copy()
            ka = int(rng.integers(4))
            if ka:
                aux_img = _rot90(aux_img, ka)
            aux_img = _affine(aux_img,
                              rng.uniform(-self.aux_angle, self.aux_angle),
                              rng.uniform(*self.aux_scale), order_nearest=False)

        ref_img = _photometric(ref_img, rng, self.brightness, self.contrast, self.hue_deg)
        aux_img = _photometric(aux_img, rng, self.brightness, self.contrast, self.hue_deg)
        return ref_img, ref_mask, aux_img
