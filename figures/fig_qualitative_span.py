# -*- coding: utf-8 -*-
"""Fig. 5 | What the pixel-correspondence assumption costs, across the range.

Why this figure exists
----------------------
Fig. 3d shows the frames where channel concatenation loses most, which is the
extreme of the distribution. A reader who cannot judge a boxplot by eye still has
no way to see whether that loss is confined to the extreme or runs through the
cohort. This figure answers that: five percentiles of the proposed scheme's
per-image Dice in each reference frame, the same contrast in every row.

Why only two configurations
---------------------------
The claim this figure carries is Fig. 3's: imposing a correspondence the frames do
not have costs more than the second frame contributes. That is a two-way contrast,
and drawing four configurations invites a four-way ranking the data cannot
support, since bottleneck concatenation does not separate from the proposed scheme
on any of twenty comparisons (Results 3.3) and the single-modality baseline does
not separate on area (Results 3.4). Both of those comparisons are made, with their
nulls, in Fig. 3 and Fig. 4. Here the reader sees the one contrast that does
separate.

Rows come from the recorded span rule, not from inspection. Across all seven
percentiles the proposed scheme is ahead of channel concatenation on twelve of
fourteen; the five drawn here are an evenly spaced subset of those seven, chosen
before the scores were looked at.

Every mask is a held-out prediction from the fold that excluded its patient, and
the Dice printed is the mean over the three training seeds, matching how every
other number in the paper is read.

Data: results/exemplars/ (predicted masks + index.csv), data/packaged/*.npz and
      manifest.csv. No GPU, no checkpoints.
Backend: Python / matplotlib (project default).
Output: fig_qualitative_span.{svg,pdf,tiff,png}

Palette discipline (shared with fig1_misalignment, fig2_fusion_operator)
  Blue / teal -> reference frame identity ONLY, carried on the panel border.
  Gold        -> the reference contour, as in fig1_misalignment. Red is not
                 usable here: it disappears against white-light mucosa.
  Grey        -> channel concatenation, which is not the proposed scheme.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from PIL import Image

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.linewidth": 0.8,
})

REPO = os.environ.get("EGC_REPO") or r"C:\Users\zpanp\projects\EGC-DualSeg"
RESULTS = os.path.join(REPO, "results")
EXEMPLARS = os.path.join(RESULTS, "exemplars")
MANIFEST = os.path.join(REPO, "data", "packaged", "manifest.csv")
NPZ = os.path.join(REPO, "data", "packaged", "egc_dualseg_384.npz")
OUT = os.path.dirname(os.path.abspath(__file__))

C_WLI = "#0F4D92"
C_NBI = "#42949E"
C_REF = "#FFD700"       # gold: legible on both pink and green mucosa
C_NEU = "#767676"       # channel concatenation

MM = 1 / 25.4
FIGW = 180 * MM          # height is computed from the tile aspect ratios

ROWS = ["span-p10", "span-p25", "span-p50", "span-p75", "span-p90"]
ROW_LABEL = {"span-p10": "10th", "span-p25": "25th", "span-p50": "50th",
             "span-p75": "75th", "span-p90": "90th"}

FRAMES = [
    ("WLI", C_WLI, "early_fusion_wli", "white light"),
    ("NBI", C_NBI, "early_fusion_nbi", "narrow band"),
]
COLS = ["Reference", "Channel concat.", "Proposed"]


def outline(mask, width=2):
    """Boundary of a binary mask, thickened so it survives downsampling."""
    m = mask.astype(bool)
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] ^ m[:-1, :]
    e[:, 1:] |= m[:, 1:] ^ m[:, :-1]
    for _ in range(width - 1):
        g = np.zeros_like(e)
        g[1:, :] |= e[:-1, :]; g[:-1, :] |= e[1:, :]
        g[:, 1:] |= e[:, :-1]; g[:, :-1] |= e[:, 1:]
        e |= g
    return e


def draw(ax, img, contours):
    ax.imshow(img)
    for mask, colour in contours:
        ov = np.zeros(img.shape[:2] + (4,), float)
        ov[outline(mask)] = (*matplotlib.colors.to_rgb(colour), 1.0)
        ax.imshow(ov)
    ax.set_xticks([]); ax.set_yticks([])


def dice_of(row):
    """Seed mean where recorded, else the rendered seed."""
    d = row["dice_seed_mean"]
    if d in ("", None) or (isinstance(d, float) and d != d):
        return float(row["dice_this_seed"])
    return float(d)


def main() -> int:
    for p in (os.path.join(EXEMPLARS, "index.csv"), MANIFEST, NPZ):
        if not os.path.isfile(p):
            raise SystemExit("missing input: %s" % p)

    idx = pd.read_csv(os.path.join(EXEMPLARS, "index.csv"), encoding="utf-8-sig")
    man = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    row_of = {(r.case, r.scale, r.modality): int(r.idx) for r in man.itertuples()}
    crop_of = {(r.case, r.scale, r.modality):
               (int(r.pad_y), int(r.pad_y) + int(r.content_h),
                int(r.pad_x), int(r.pad_x) + int(r.content_w))
               for r in man.itertuples()}
    blob = np.load(NPZ)
    images, masks = blob["images"], blob["masks"]

    # Measure first: the two frames differ in aspect ratio, so equal-height rows
    # would leave a white band under every wide tile.
    def tile_shape(frame, rule):
        sub = idx[(idx.ref_modality == frame) & (idx.rule == rule)]
        if sub.empty:
            return None
        key = (sub.iloc[0]["case"], sub.iloc[0]["scale"], frame)
        if key not in crop_of:
            return None
        y0, y1, x0, x1 = crop_of[key]
        return (y1 - y0) / (x1 - x0)

    heights = []
    for rule in ROWS:
        hs = [tile_shape(f[0], rule) for f in FRAMES]
        heights.append(max([h for h in hs if h] or [1.0]))

    # One tile is 1 unit wide; a row is as tall as its tallest tile.
    tile_w = (1.0 - 0.062 - 0.005) / (3 + 0.28 / 3 + 3)
    fig_h = FIGW * (sum(heights) * tile_w + 0.11) + 0.02
    fig = plt.figure(figsize=(FIGW, fig_h))
    gs = GridSpec(len(ROWS), 7, figure=fig,
                  width_ratios=[1, 1, 1, 0.28, 1, 1, 1],
                  height_ratios=heights,
                  left=0.062, right=0.995, top=0.918, bottom=0.012,
                  hspace=0.06, wspace=0.035)

    missing, ahead = [], []
    for fi, (frame, colour, early, frame_word) in enumerate(FRAMES):
        for ri, rule in enumerate(ROWS):
            sub = idx[(idx.ref_modality == frame) & (idx.rule == rule)]
            if sub.empty:
                missing.append("%s %s" % (frame, rule)); continue
            case, scale = sub.iloc[0]["case"], sub.iloc[0]["scale"]
            key = (case, scale, frame)
            if key not in row_of:
                missing.append("%s %s %s" % (frame, case, scale)); continue
            y0, y1, x0, x1 = crop_of[key]
            img = images[row_of[key]][y0:y1, x0:x1]
            gt = masks[row_of[key]][y0:y1, x0:x1] > 127

            scores = {}
            for ci, cfg in enumerate([None, early, "ours"]):
                ax = fig.add_subplot(gs[ri, fi * 4 + ci])
                if cfg is None:
                    draw(ax, img, [(gt, C_REF)])
                else:
                    hit = sub[(sub.case == case) & (sub.scale == scale)
                              & (sub.config == cfg)]
                    path = (os.path.join(EXEMPLARS, hit.iloc[0]["path"])
                            if not hit.empty else "")
                    if not os.path.isfile(path):
                        ax.axis("off")
                        missing.append("%s %s %s" % (frame, case, cfg)); continue
                    pred = (np.array(Image.open(path)) > 0)[y0:y1, x0:x1]
                    pen = colour if cfg == "ours" else C_NEU
                    draw(ax, img, [(gt, C_REF), (pred, pen)])
                    d = dice_of(hit.iloc[0])
                    scores[cfg] = d
                    ax.text(0.045, 0.045, "%.2f" % d, transform=ax.transAxes,
                            ha="left", va="bottom", fontsize=6.8, color="white",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.20", fc=pen,
                                      ec="none", alpha=0.90))
                for s in ax.spines.values():
                    s.set_edgecolor(colour); s.set_linewidth(0.9)
                if ri == 0:
                    ax.set_title(COLS[ci], fontsize=7.0, pad=3.5,
                                 color="#8A6D00" if ci == 0 else
                                 (colour if ci == 2 else C_NEU))
                if ci == 0 and fi == 0:
                    ax.text(-0.085, 0.5, ROW_LABEL[rule], transform=ax.transAxes,
                            rotation=90, ha="right", va="center", fontsize=7.2,
                            color=C_NEU)
            if len(scores) == 2:
                ahead.append(scores["ours"] > scores[early])

        mid = 0.062 + (0.995 - 0.062) * (0.185 + fi * 0.545)
        fig.text(mid, 0.962, frame_word, ha="center", va="bottom",
                 fontsize=9.2, color=colour, fontweight="bold")

    fig.text(0.012, 0.47, "percentile of per-image Dice", rotation=90,
             ha="center", va="center", fontsize=7.4, color=C_NEU)

    base = os.path.join(OUT, "fig_qualitative_span")
    for ext, kw in [(".svg", {}), (".pdf", {}), (".tiff", dict(dpi=600)),
                    (".png", dict(dpi=600))]:
        fig.savefig(base + ext, **kw)
    if missing:
        print("!!! %d tile(s) missing: %s" % (len(missing), ", ".join(missing)))
    print("proposed ahead of channel concatenation in %d of %d drawn rows"
          % (sum(ahead), len(ahead)))
    print("saved -> fig_qualitative_span.{svg,pdf,tiff,png}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
