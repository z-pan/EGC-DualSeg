# -*- coding: utf-8 -*-
"""Fig. 6 | Delineations across the performance range, three schemes side by side.

Why this figure exists
----------------------
Fig. 3d already shows the frames where channel concatenation loses most, which is
the extreme of the distribution. What the paper has nowhere is what a typical
delineation looks like, and a reader who cannot judge a boxplot by eye has no way
to see whether any of these models is usable. This figure fills that gap.

Rows are chosen by the recorded span rule, not by inspection: the 10th, 50th and
90th percentile of the proposed scheme's per-image Dice within each reference
frame. That is a deliberately unflattering selection, and it shows cases where the
single-modality baseline matches or beats the proposed scheme, which is what
Results 3.4 reports.

Columns are the three configurations the argument turns on: one frame alone, the
two frames stacked pixel-for-pixel, and the two frames fused without assuming any
correspondence. Bottleneck concatenation is absent because the exemplar dump did
not write its masks; it does not separate from the proposed scheme on any metric
(Results 3.3), so nothing in the comparison shown here would change.

Every mask is a held-out prediction from the fold that excluded its patient.

Data: results/exemplars/ (predicted masks + index.csv), data/packaged/*.npz and
      manifest.csv. No GPU, no checkpoints.
Backend: Python / matplotlib (project default).
Output: fig_qualitative_span.{svg,pdf,tiff,png}

Palette discipline (shared with fig1_misalignment, fig2_fusion_operator)
  Blue / teal -> reference frame identity ONLY, carried on the panel border.
  Gold        -> the reference contour, as in fig1_misalignment. Red is not
                 usable here: it disappears against white-light mucosa.
  Grey        -> a configuration that is not the proposed one.
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
C_REF = "#FFD700"       # gold: the reference contour, legible on pink and
                        # green mucosa. Same value as fig1_misalignment.
C_SIG = "#B64342"
C_NEU = "#767676"

MM = 1 / 25.4
FIGW, FIGH = 180 * MM, 218 * MM

# Rule -> plain reading for the row label. The percentile is of the proposed
# scheme's per-image Dice inside that reference frame.
ROWS = ["span-p10", "span-p50", "span-p90"]
ROW_LABEL = {"span-p10": "10th percentile",
             "span-p50": "median",
             "span-p90": "90th percentile"}

FRAMES = [
    ("WLI", C_WLI, "wli_only", "early_fusion_wli", "white light"),
    ("NBI", C_NBI, "nbi_only", "early_fusion_nbi", "narrow band"),
]
COLS = ["Reference contour", "Single modality",
        "Channel concatenation", "Proposed"]


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
    """One tile: the frame with zero or more coloured contours over it."""
    ax.imshow(img)
    for mask, colour in contours:
        ov = np.zeros(img.shape[:2] + (4,), float)
        ov[outline(mask)] = (*matplotlib.colors.to_rgb(colour), 1.0)
        ax.imshow(ov)
    ax.set_xticks([]); ax.set_yticks([])


def main():
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

    fig = plt.figure(figsize=(FIGW, FIGH))
    gs = GridSpec(len(FRAMES) * len(ROWS), len(COLS), figure=fig,
                  left=0.085, right=0.995, top=0.962, bottom=0.048,
                  hspace=0.05, wspace=0.03)

    missing = []
    for fi, (frame, colour, single, early, frame_word) in enumerate(FRAMES):
        for ri, rule in enumerate(ROWS):
            r = fi * len(ROWS) + ri
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

            for ci, cfg in enumerate([None, single, early, "ours"]):
                ax = fig.add_subplot(gs[r, ci])
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
                    d = float(hit.iloc[0]["dice_seed_mean"])
                    ax.text(0.035, 0.035, "Dice %.2f" % d, transform=ax.transAxes,
                            ha="left", va="bottom", fontsize=6.6, color="white",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.22", fc=pen,
                                      ec="none", alpha=0.88))
                for s in ax.spines.values():
                    s.set_edgecolor(colour); s.set_linewidth(1.0)
                if r == 0:
                    ax.set_title(COLS[ci], fontsize=7.4, pad=4.0,
                                 color="black" if ci else "#8A6D00")
                if ci == 0:
                    ax.text(-0.055, 0.5, ROW_LABEL[rule], transform=ax.transAxes,
                            rotation=90, ha="right", va="center", fontsize=7.0,
                            color=C_NEU)
        # one frame label per block, on the left margin
        y_top = 1 - 0.038 - fi * 0.4570
        fig.text(0.017, y_top - 0.2, frame_word, rotation=90, ha="center",
                 va="center", fontsize=9.2, color=colour, fontweight="bold")

    base = os.path.join(OUT, "fig_qualitative_span")
    for ext, kw in [(".svg", {}), (".pdf", {}), (".tiff", dict(dpi=600)),
                    (".png", dict(dpi=600))]:
        fig.savefig(base + ext, **kw)
    if missing:
        print("!!! %d tile(s) missing: %s" % (len(missing), ", ".join(missing)))
    print("saved -> fig_qualitative_span.{svg,pdf,tiff,png}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
