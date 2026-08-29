# -*- coding: utf-8 -*-
"""Fig. 2 | Effective spatial resolution delimits measurable pathology.

Panels
  a  Scale-calibrated near-focus NBI crops at a common physical field of view,
     each with a magnified inset placed on the highest-texture region inside the
     lesion. Exemplars are drawn from cases whose calibration sits inside the
     interquartile range, so the scale bar is representative rather than extreme.
  b  Per-lesion effective resolution (um per pixel) for both modalities.
  c  Structure size versus the morphological-classification limit implied by b,
     on one shared logarithmic axis in um. This is the panel that decides which
     pathology endpoints are admissible.

Calibration: pathological long axis (mm, from the resection report) divided by
the mask long axis (pixels, within the cropped endoscopic field), n = 42
single-lesion resections. The pathological extent is the microscopic tumour
extent, which is smaller than the endoscopically visible abnormality, so this
estimate is OPTIMISTIC: the true resolution is no better than reported, which
makes the exclusion argument in c conservative.

Backend: Python / matplotlib (project default).
Output: fig2_resolution.{svg,pdf,tiff,png}

Palette discipline (shared with Fig. 1)
  Blue / teal   -> modality identity (WLI / NBI) ONLY
  Neutral greys -> view type and generic distributions
  Red           -> reference lines, limits, exclusions
"""
import os
import re
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
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
LABELS = os.path.join(ROOT, "pathology_labels_clean.csv")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import bbox_sam_tools as T                                    # noqa: E402

C_WLI = "#0F4D92"
C_NBI = "#42949E"
C_SIG = "#B64342"
C_NEU = "#767676"
C_LIGHT = "#A8A8A8"

MM = 1 / 25.4
FINAL_WIDTH_MM = 180
FINAL_HEIGHT_MM = 132
FIGW, FIGH = FINAL_WIDTH_MM * MM, FINAL_HEIGHT_MM * MM

CROP_MM = 6.0          # physical field shown in each panel-a tile
INSET_MM = 1.5         # physical field of the magnified inset
MIN_PX_TO_CLASSIFY = 3     # permissive limit: 3 px across a structure
CONS_PX_TO_CLASSIFY = 5    # conservative limit


# --------------------------------------------------------------------------
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
    if (x1 - x0) < 0.3 * g.shape[1] or (y1 - y0) < 0.3 * g.shape[0]:
        return 0, 0, g.shape[1], g.shape[0]
    return x0, y0, x1, y1


def local_texture_peak(gray, mask, specular, win=48):
    """Centre of the highest local-contrast window inside the lesion mask.

    Specular highlights are the strongest local-contrast features in endoscopic
    images and would otherwise capture the search, so windows containing more
    than 1% saturated pixels are rejected and the contrast statistic itself is
    computed on non-specular pixels only.
    """
    g = gray.astype(float)
    step = max(8, win // 4)
    best, bxy = -1.0, None
    H, W = g.shape
    for y in range(0, H - win, step):
        for x in range(0, W - win, step):
            mw = mask[y:y + win, x:x + win]
            if mw.mean() < 0.98:
                continue
            sw = specular[y:y + win, x:x + win]
            if sw.mean() > 0.01:
                continue
            v = g[y:y + win, x:x + win][~sw].std()
            if v > best:
                best, bxy = v, (y + win // 2, x + win // 2)
    if bxy is None:                       # lesion too small / too much glare
        ys, xs = np.where(mask & ~specular)
        if not len(ys):
            ys, xs = np.where(mask)
        bxy = (int(ys.mean()), int(xs.mean()))
    return bxy


# ---- collect calibration --------------------------------------------------
labels = {r["case"]: r for r in csv.DictReader(open(LABELS, encoding="utf-8"))}
recs = []
for batch, case, cpath in T.find_case_dirs(ROOT):
    L = labels.get(case)
    if not L or L["type"] != "ESD" or L["exclude_reason"] or L["n_lesions"] != "1":
        continue
    m = re.match(r"([\d.]+)x([\d.]+)", L["size_mm"])
    if not m:
        continue
    path_long = max(float(m.group(1)), float(m.group(2)))
    d = {}
    for jp, img, mod, sc, boxes in T.scan_case(cpath)["items"]:
        if img:
            d[(mod, sc)] = (jp, img)
    for mod in ("WLI", "NBI"):
        if (mod, "near") not in d:
            continue
        jp, img = d[(mod, "near")]
        mp = os.path.join(MASKS, batch, case,
                          os.path.splitext(os.path.basename(jp))[0] + ".png")
        if not os.path.isfile(mp):
            continue
        bb = fov_bbox(img)
        M = np.array(Image.open(mp))[bb[1]:bb[3], bb[0]:bb[2]] > 127
        if M.sum() < 500:
            continue
        ys, xs = np.where(M)
        pix = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        recs.append(dict(case=case, mod=mod, img=img, mask=mp, bb=bb,
                         path_mm=path_long, umpp=path_long / pix * 1000,
                         field_frac=float(M.mean())))

um_wli = np.array([r["umpp"] for r in recs if r["mod"] == "WLI"])
um_nbi = np.array([r["umpp"] for r in recs if r["mod"] == "NBI"])
med_nbi = float(np.median(um_nbi))
q1, q3 = np.percentile(um_nbi, [25, 75])

# exemplars: NBI cases inside the IQR of the calibration, largest lesions first
# (a larger lesion gives more in-mask area for the texture inset)
cand = [r for r in recs if r["mod"] == "NBI" and q1 <= r["umpp"] <= q3
        and r["field_frac"] < 0.60]
cand.sort(key=lambda r: -r["path_mm"])
picks = cand[:3]

# ---- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(FIGW, FIGH))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.18, 1.0], width_ratios=[0.86, 1.30],
              hspace=0.34, wspace=0.34,
              left=0.085, right=0.985, top=0.925, bottom=0.115)

# -- a : scale-calibrated crops with magnified insets -----------------------
gs_a = gs[0, :].subgridspec(1, 3, wspace=0.05)
for col, r in enumerate(picks):
    ax = fig.add_subplot(gs_a[0, col])
    bb = r["bb"]
    im = np.asarray(Image.open(r["img"]).convert("RGB").crop(bb))
    M = np.array(Image.open(r["mask"]))[bb[1]:bb[3], bb[0]:bb[2]] > 127
    umpp = r["umpp"]
    half = int(CROP_MM * 1000 / umpp / 2)
    gray = np.asarray(Image.fromarray(im).convert("L"))
    specular = gray > 225
    cy, cx = local_texture_peak(gray, M, specular)
    y0 = int(np.clip(cy - half, 0, max(0, im.shape[0] - 2 * half)))
    x0 = int(np.clip(cx - half, 0, max(0, im.shape[1] - 2 * half)))
    crop = im[y0:y0 + 2 * half, x0:x0 + 2 * half]
    S = 640
    disp = np.asarray(Image.fromarray(crop).resize((S, S), Image.LANCZOS))
    ax.imshow(disp)

    # magnified inset taken at the texture peak of the displayed crop
    ins_half_disp = int(INSET_MM / CROP_MM * S / 2)
    iy = int(np.clip((cy - y0) / (2 * half) * S, ins_half_disp, S - ins_half_disp))
    ix = int(np.clip((cx - x0) / (2 * half) * S, ins_half_disp, S - ins_half_disp))
    sub = disp[iy - ins_half_disp:iy + ins_half_disp,
               ix - ins_half_disp:ix + ins_half_disp]
    ax.add_patch(Rectangle((ix - ins_half_disp, iy - ins_half_disp),
                           2 * ins_half_disp, 2 * ins_half_disp,
                           fill=False, ec="white", lw=1.2))
    ins = ax.inset_axes([0.615, 0.615, 0.375, 0.375])
    ins.imshow(np.asarray(Image.fromarray(sub).resize((320, 320), Image.LANCZOS)))
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color("white"); sp.set_linewidth(1.2)

    # 1 mm scale bar
    bar = S * (1.0 / CROP_MM)
    ax.add_patch(Rectangle((S * 0.045, S * 0.945), bar, S * 0.017,
                           fc="white", ec="black", lw=0.5))
    ax.text(S * 0.045 + bar / 2, S * 0.935, "1 mm", color="white",
            ha="center", va="bottom", fontsize=9)
    ax.text(S * 0.045, S * 0.055, f"{umpp:.0f} \u00b5m px$^{{-1}}$",
            color="white", ha="left", va="top", fontsize=9,
            bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.6))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.5); sp.set_color(C_NBI)
    if col == 0:
        ax.set_ylabel("NBI, near focus", color=C_NBI, fontweight="bold", labelpad=5)
        ax.text(-0.16, 1.06, "a", transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="left", va="bottom")

# -- b : paired per-lesion resolution, same idiom as Fig. 1d ---------------
# the two modalities are measured on the SAME lesion, so a paired scatter
# against y = x is more informative than two overlapping histograms.
axb = fig.add_subplot(gs[1, 0])
by_case = {}
for r in recs:
    by_case.setdefault(r["case"], {})[r["mod"]] = r["umpp"]
pw = np.array([v["WLI"] for v in by_case.values() if "WLI" in v and "NBI" in v])
pn = np.array([v["NBI"] for v in by_case.values() if "WLI" in v and "NBI" in v])
lo_b, hi_b = 0, max(pw.max(), pn.max()) * 1.06
axb.plot([lo_b, hi_b], [lo_b, hi_b], color=C_NEU, ls="--", lw=1.1, zorder=1)
finer = pn < pw
axb.scatter(pw[finer], pn[finer], s=16, color=C_NBI, alpha=0.8, lw=0, zorder=2)
axb.scatter(pw[~finer], pn[~finer], s=16, facecolor="none",
            edgecolor=C_LIGHT, lw=0.9, zorder=2)
axb.set_xlim(lo_b, hi_b); axb.set_ylim(lo_b, hi_b); axb.set_aspect("equal")
axb.set_xlabel("WLI resolution (\u00b5m px$^{-1}$)", color=C_WLI)
axb.set_ylabel("NBI resolution (\u00b5m px$^{-1}$)", color=C_NBI)
# stated above the axes so it can never collide with the point cloud
axb.set_title(f"{finer.mean()*100:.0f}% of lesions finer in NBI\n"
              f"median {np.median(pw):.1f} $\\rightarrow$ {np.median(pn):.1f} "
              f"µm px$^{{-1}}$",
              fontsize=8.5, color=C_NEU, pad=5, linespacing=1.35)
axb.text(-0.22, 1.02, "b", transform=axb.transAxes, fontsize=12,
         fontweight="bold", ha="left", va="bottom")

# -- c : structure size vs classification limit (shared log axis) ----------
axc = fig.add_subplot(gs[1, 1])
lim_perm = MIN_PX_TO_CLASSIFY * med_nbi          # permissive
lim_cons = CONS_PX_TO_CLASSIFY * med_nbi         # conservative

STRUCT = [
    ("Demarcation line",  1000, 10000, "#4D4D4D"),
    ("Gland opening\n(MS pattern)", 100, 200, C_NBI),
    ("Capillary\n(MV pattern)", 10, 50, C_SIG),
]
axc.axvspan(6, lim_cons, color=C_SIG, alpha=0.09, lw=0, zorder=0)
for i, (name, lo, hi, col) in enumerate(STRUCT):
    y = len(STRUCT) - 1 - i
    axc.barh(y, hi - lo, left=lo, height=0.40, color=col, alpha=0.8,
             edgecolor=col, lw=1.0, zorder=2)
axc.axvline(lim_perm, color=C_SIG, ls="--", lw=1.3, zorder=3,
            label=f"3 px  ({lim_perm:.0f} µm)")
axc.axvline(lim_cons, color=C_SIG, ls=":", lw=1.3, zorder=3,
            label=f"5 px  ({lim_cons:.0f} µm)")
axc.set_yticks(range(len(STRUCT)))
axc.set_yticklabels([s[0] for s in STRUCT][::-1], fontsize=9)
axc.tick_params(axis="y", length=0)
axc.set_xscale("log")
axc.set_xlim(6, 20000)
axc.set_ylim(-0.55, len(STRUCT) - 0.30)
axc.set_xlabel("Structure size (\u00b5m)")
# real Line2D handles guarantee the sample strokes match the drawn line styles
leg = axc.legend(loc="upper left", bbox_to_anchor=(0.02, 1.02), fontsize=8.5,
                 handlelength=1.8, labelspacing=0.35, borderaxespad=0.0,
                 title="Classification limit", alignment="left")
leg.get_title().set_fontsize(8.5)
leg.get_title().set_color(C_SIG)
for t in leg.get_texts():
    t.set_color(C_SIG)
# rotated so it fits inside the narrow shaded band
axc.text(0.030, 0.40, "not resolved", transform=axc.transAxes, rotation=90,
         color=C_SIG, fontsize=8.5, ha="center", va="center", style="italic")
axc.spines["left"].set_visible(False)
axc.text(-0.34, 1.10, "c", transform=axc.transAxes, fontsize=12,
         fontweight="bold", ha="left", va="bottom")

base = os.path.join(OUT, "fig2_resolution")
fig.savefig(base + ".svg")
fig.savefig(base + ".pdf")
fig.savefig(base + ".tiff", dpi=600)
fig.savefig(base + ".png", dpi=600)
plt.close(fig)

print(f"calibrated lesions: WLI {len(um_wli)}, NBI {len(um_nbi)}")
print(f"WLI  median {np.median(um_wli):.1f} um/px  IQR "
      f"{np.percentile(um_wli,25):.1f}-{np.percentile(um_wli,75):.1f}")
print(f"NBI  median {med_nbi:.1f} um/px  IQR {q1:.1f}-{q3:.1f}")
print(f"classification limit: 3 px = {lim_perm:.0f} um, 5 px = {lim_cons:.0f} um")
print(f"exemplar pool = {len(cand)} NBI lesions inside IQR")
for r in picks:
    print(f"  {r['case']:8s} {r['umpp']:.1f} um/px  path {r['path_mm']:.1f} mm  "
          f"field {r['field_frac']*100:.0f}%")
print("saved -> fig2_resolution.{svg,pdf,tiff,png}")
