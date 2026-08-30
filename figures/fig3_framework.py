# -*- coding: utf-8 -*-
"""Fig. 1 | Two-stage dual-modal framework and cohort construction.

Panels
  a  Pipeline schematic. Stage 1 localises the lesion from the paired input;
     Stage 2 predicts histological grade from the localised representation.
     A reference-role embedding lets either modality act as the query stream,
     so one model covers both directions.
  b  Provenance and reliability of the two supervision signals side by side.
     No patient-identifying material is reproduced: the pathology report is
     rendered as a field summary, never as a scan.
  c  Cohort flow with explicit exclusions.

Backend: Python / matplotlib (project default).
Output: fig3_framework.{svg,pdf,tiff,png}

Palette discipline (shared with fig1_misalignment, fig2_fusion_operator)
  Blue / teal   -> modality identity (WLI / NBI) ONLY
  Neutral greys -> generic blocks and flow
  Red           -> exclusions and weak-supervision caution
  Green         -> the pathology reference standard

Plotting strings are ASCII-only on purpose so the source stays diff-friendly.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 10,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "legend.frameon": False,
})

# The endoscopic archive is NOT public: it holds identifiable patient
# images. Set EGC_RAW to a local copy to regenerate the panels that need it.
ROOT = os.environ.get("EGC_RAW", "")
MASKS = os.path.join(ROOT, "_sam_output", "masks")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import bbox_sam_tools as T                                   # noqa: E402

C_WLI = "#0F4D92"
C_NBI = "#42949E"
C_SIG = "#B64342"
C_GRN = "#2E6B45"
C_NEU = "#767676"
C_DARK = "#4D4D4D"
C_S1 = "#EDEDF2"
C_S2 = "#E7F0EF"
C_BOX = "#F4F4F4"

MM = 1 / 25.4
FINAL_WIDTH_MM = 180
FINAL_HEIGHT_MM = 150
FIGW, FIGH = FINAL_WIDTH_MM * MM, FINAL_HEIGHT_MM * MM

DEMO_CASE = "\u75c5\u4f8b15"          # median-correspondence exemplar, single lesion


# ---------- schematic helpers ---------------------------------------------
_BOXES = []                      # (ax, x, y, w, h, text_artist) for the fit check


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_DARK, tc="black", fs=8.2,
        lw=1.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, linespacing=1.30,
                fontweight=weight)
    _BOXES.append((ax, x, y, w, h, t))


def check_boxes(fig, margin=0.004):
    """Re-measure every label after the draw; overflow is a layout bug, not a
    styling preference, so say so loudly rather than shipping a clipped figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ax, x, y, w, h, t in _BOXES:
        bb = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
        if bb.width > w - margin or bb.height > h - margin:
            bad.append((t.get_text().replace("\n", "|"), w, bb.width, h, bb.height))
    if bad:
        print("\n!!! %d BOX(ES) TOO SMALL FOR THEIR LABEL:" % len(bad))
        for s, w, tw, h, th in bad:
            print("    %-34s box %.3f x %.3f   text %.3f x %.3f" % (s[:34], w, h, tw, th))
        print("!!! enlarge the box or rewrap the label; do not reduce the font.\n")
    else:
        print("box fit check: all %d labels fit" % len(_BOXES))
    return len(bad)


def arrow(ax, x0, y0, x1, y1, color=C_DARK, ls="solid", lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, ls=ls, zorder=1,
                                 connectionstyle="arc3,rad=%.3f" % rad,
                                 shrinkA=1.5, shrinkB=1.5))


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def plabel(ax, s, x=-0.01, y=1.00):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="left", va="bottom")


# ---------- load the demo case --------------------------------------------
demo = {}
for batch, case, cpath in T.find_case_dirs(ROOT):
    if case != DEMO_CASE:
        continue
    for jp, img, mod, sc, boxes in T.scan_case(cpath)["items"]:
        if img and sc == "near":
            mp = os.path.join(MASKS, batch, case,
                              os.path.splitext(os.path.basename(jp))[0] + ".png")
            demo[mod] = dict(img=img, mask=mp, boxes=boxes)


def fov_bbox(path):
    g = np.asarray(Image.open(path).convert("L"))
    bright = g > 25

    def span(f, thr=0.5):
        idx = np.where(f > thr)[0]
        if not len(idx):
            return 0, len(f)
        segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        s = max(segs, key=len)
        return int(s[0]), int(s[-1]) + 1

    x0, x1 = span(bright.mean(0))
    y0, y1 = span(bright.mean(1))
    return x0, y0, x1, y1


def demo_tile(mod, draw="none", size=320):
    d = demo[mod]
    bb = fov_bbox(d["img"])
    im = np.asarray(Image.open(d["img"]).convert("RGB").crop(bb)).copy()
    if draw == "mask":
        M = np.array(Image.open(d["mask"]))[bb[1]:bb[3], bb[0]:bb[2]] > 127
        er = np.zeros_like(M)
        er[1:-1, 1:-1] = (M[1:-1, 1:-1] & M[:-2, 1:-1] & M[2:, 1:-1]
                          & M[1:-1, :-2] & M[1:-1, 2:])
        edge = M & ~er
        k = max(1, int(round(min(im.shape[:2]) / 260)))
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                im[np.roll(np.roll(edge, dy, 0), dx, 1)] = (255, 215, 0)
    h, w = im.shape[:2]
    s = min(h, w)
    oy, ox = (h - s) // 2, (w - s) // 2
    im = im[oy:oy + s, ox:ox + s]
    rel = None
    if draw == "bbox":
        b = d["boxes"][0]
        rel = ((b[0] - bb[0] - ox) / s, (b[1] - bb[1] - oy) / s,
               (b[2] - b[0]) / s, (b[3] - b[1]) / s)
    return np.asarray(Image.fromarray(im).resize((size, size), Image.LANCZOS)), rel


# ---------- figure ---------------------------------------------------------
fig = plt.figure(figsize=(FIGW, FIGH))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.00, 1.10],
              width_ratios=[1.00, 1.02], hspace=0.14, wspace=0.12,
              left=0.035, right=0.985, top=0.955, bottom=0.025)

# =========================== a : pipeline =================================
axa = fig.add_subplot(gs[0, :]); blank(axa); plabel(axa, "a", x=-0.005, y=0.97)

axa.add_patch(Rectangle((0.035, 0.075), 0.615, 0.745, fc=C_S1, ec="none", zorder=0))
axa.add_patch(Rectangle((0.665, 0.075), 0.325, 0.745, fc=C_S2, ec="none", zorder=0))
axa.text(0.045, 0.855, "Stage 1  -  localisation", fontsize=8.6, color=C_DARK,
         fontweight="bold", va="bottom")
axa.text(0.045, 0.795, "supervision: clinician-reviewed masks", fontsize=7.8,
         color=C_DARK, va="bottom", style="italic")
axa.text(0.675, 0.855, "Stage 2  -  grading", fontsize=8.6, color=C_DARK,
         fontweight="bold", va="bottom")
axa.text(0.675, 0.795, "supervision: resection pathology", fontsize=7.8,
         color=C_GRN, va="bottom", style="italic")

box(axa, 0.048, 0.545, 0.088, 0.150, "WLI\nframe", fc="white", ec=C_WLI,
    tc=C_WLI, lw=1.3, weight="bold")
box(axa, 0.048, 0.245, 0.088, 0.150, "NBI\nframe", fc="white", ec=C_NBI,
    tc=C_NBI, lw=1.3, weight="bold")
box(axa, 0.163, 0.378, 0.100, 0.185, "crop to\nendoscopic\nfield", fs=7.6)
box(axa, 0.278, 0.354, 0.098, 0.232, "shared\nbackbone\n+ modality\nadapter", fs=7.4)
box(axa, 0.398, 0.378, 0.088, 0.185, "scale\nnormali-\nsation", fs=7.6)
box(axa, 0.505, 0.328, 0.098, 0.285, "cross-\nattention\n\nref = Q\naux = K/V", fs=7.4)
box(axa, 0.048, 0.098, 0.150, 0.128, "reference-role\nembedding",
    fc="white", ec=C_NEU, tc=C_NEU, fs=7.2)
box(axa, 0.690, 0.545, 0.105, 0.150, "decoder to\nlesion mask", fs=7.6)
box(axa, 0.690, 0.245, 0.105, 0.150, "grade head", fs=7.8)
box(axa, 0.845, 0.225, 0.125, 0.190, "low grade\nvs\nhigh grade+",
    fc="white", ec=C_GRN, tc=C_GRN, fs=7.8, lw=1.3, weight="bold")

arrow(axa, 0.136, 0.620, 0.163, 0.525)
arrow(axa, 0.136, 0.320, 0.163, 0.415)
arrow(axa, 0.265, 0.470, 0.276, 0.470)
arrow(axa, 0.376, 0.470, 0.398, 0.470)
arrow(axa, 0.488, 0.470, 0.503, 0.470)
arrow(axa, 0.603, 0.470, 0.688, 0.620, rad=-0.12)
arrow(axa, 0.200, 0.162, 0.505, 0.352, color=C_NEU, ls=(0, (3, 2)), lw=0.9, rad=-0.14)
arrow(axa, 0.742, 0.545, 0.742, 0.398)
arrow(axa, 0.795, 0.320, 0.843, 0.320)
axa.text(0.752, 0.470, "lesion region", fontsize=7.2, color=C_NEU,
         ha="left", va="center")

# =========================== b : supervision provenance ===================
axb = fig.add_subplot(gs[1, 0]); blank(axb); plabel(axb, "b", x=-0.01, y=0.99)

tile_bbox, rel = demo_tile("WLI", draw="bbox")
tile_mask, _ = demo_tile("WLI", draw="mask")

axb.text(0.040, 0.965, "Stage-1 target", fontsize=8.2, color=C_DARK,
         fontweight="bold", va="top")
axb.text(0.040, 0.900, "imaging-derived", fontsize=7.3, color=C_DARK,
         va="top", style="italic")
for i, (tl, cap) in enumerate([(tile_bbox, "clinician box"),
                               (tile_mask, "reviewed mask")]):
    ax_i = axb.inset_axes([0.040 + i * 0.245, 0.515, 0.205, 0.290])
    ax_i.imshow(tl); ax_i.set_xticks([]); ax_i.set_yticks([])
    for sp in ax_i.spines.values():
        sp.set_color(C_DARK); sp.set_linewidth(1.3)
    ax_i.set_xlabel(cap, fontsize=6.9, color=C_DARK, labelpad=2.5)
    if i == 0 and rel is not None:
        ax_i.add_patch(Rectangle((rel[0] * tl.shape[1], rel[1] * tl.shape[0]),
                                 rel[2] * tl.shape[1], rel[3] * tl.shape[0],
                                 fill=False, ec=C_DARK, lw=1.4))
arrow(axb, 0.252, 0.660, 0.277, 0.660, color=C_DARK)

axb.text(0.545, 0.965, "Stage-2 target", fontsize=8.2, color=C_GRN,
         fontweight="bold", va="top")
axb.text(0.545, 0.900, "pathology-derived", fontsize=7.3, color=C_GRN,
         va="top", style="italic")
box(axb, 0.545, 0.515, 0.420, 0.290,
    "Resection pathology\n\ngrade  |  depth\nextent  |  type",
    fc="white", ec=C_GRN, tc=C_DARK, fs=7.4, lw=1.4)

axb.add_patch(Rectangle((0.030, 0.085), 0.935, 0.290, fc="#F4F4F4",
                        ec=C_NEU, lw=0.9, zorder=0))
for j, line in enumerate(["Stage-1 masks are delineated within each",
                          "modality's own frame; the modality comparison",
                          "is therefore also evaluated on the Stage-2 target."]):
    axb.text(0.498, 0.300 - j * 0.075, line, fontsize=7.2, color=C_DARK,
             ha="center", va="center")

# =========================== c : cohort flow ==============================
axc = fig.add_subplot(gs[1, 1]); blank(axc); plabel(axc, "c", x=-0.01, y=0.99)

for y, txt in [(0.905, "81 case folders"),
               (0.755, "79 unique patients"),
               (0.605, "77 gastric-lesion patients")]:
    box(axc, 0.050, y - 0.050, 0.520, 0.100, txt, fc="white", ec=C_DARK, fs=7.8)
for y, txt in [(0.830, "-2 duplicated records"),
               (0.680, "-2 oesophageal lesions")]:
    arrow(axc, 0.310, y + 0.048, 0.310, y - 0.048)
    arrow(axc, 0.310, y, 0.588, y, color=C_SIG, lw=0.9)
    axc.text(0.600, y, txt, fontsize=7.1, color=C_SIG, va="center")

arrow(axc, 0.310, 0.552, 0.190, 0.482)
arrow(axc, 0.310, 0.552, 0.700, 0.482)
# stage headers sit ABOVE the boxes so the box text never overflows its frame
axc.text(0.190, 0.442, "Stage 1", fontsize=8.0, color=C_DARK,
         fontweight="bold", ha="center", va="bottom")
axc.text(0.700, 0.442, "Stage 2", fontsize=8.0, color=C_DARK,
         fontweight="bold", ha="center", va="bottom")
box(axc, 0.000, 0.265, 0.380, 0.165,
    "150 same-view pairs\n77 near / 73 distant\n77 patients",
    fc=C_S1, ec=C_DARK, fs=7.3)
box(axc, 0.510, 0.265, 0.390, 0.165,
    "48 gradeable resections\n21 low / 27 high+\n48 patients",
    fc=C_S2, ec=C_DARK, fs=7.3)
arrow(axc, 0.700, 0.263, 0.700, 0.202, color=C_NEU)
box(axc, 0.505, 0.100, 0.400, 0.115,
    "positive control\n45 with macroscopic type",
    fc="white", ec=C_NEU, tc=C_NEU, fs=7.0)
axc.text(0.020, 0.210,
         "10 multifocal patients; the 4\nwith a separate measurement\nper lesion are excluded from\nthe extent analysis, 2 retained",
         fontsize=6.9, color=C_SIG, va="top", linespacing=1.32)

check_boxes(fig)

base = os.path.join(OUT, "fig3_framework")
fig.savefig(base + ".svg")
fig.savefig(base + ".pdf")
fig.savefig(base + ".tiff", dpi=600)
fig.savefig(base + ".png", dpi=600)
plt.close(fig)
print("demo case:", DEMO_CASE, "| clinician boxes:", len(demo["WLI"]["boxes"]))
print("saved -> fig3_framework.{svg,pdf,tiff,png}")
