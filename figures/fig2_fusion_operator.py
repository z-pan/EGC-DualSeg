# -*- coding: utf-8 -*-
"""Fig. 3 | The fusion scheme, not the second modality, decides the outcome.

Core conclusion this figure must defend
---------------------------------------
What a fusion scheme assumes about spatial correspondence decides the outcome.
Assuming it at every pixel costs accuracy, boundary quality and the margin a
resection would need; assuming it coarsely or not at all recovers them. The
proposed scheme is one instance of the second kind, not the unique best one --
panels a-c deliberately show bottleneck concatenation landing in the same place.

Rebuilt 2026-08-24: patient-level, three schemes
------------------------------------------------
The previous version drew two configurations at the image level (n = 150),
which claimed roughly twice the sample size the cohort has and could not show
the ladder that Results 3.3 rests on. Both are fixed:

  unit     seeds averaged into images, images averaged into patients, within a
           reference frame only. n = 77 in every panel.
  schemes  three, ordered by the scale at which each assumes correspondence:
           every pixel (channel concatenation), a 1/32 grid (bottleneck
           concatenation), none (cross-attention, proposed). The MMD-weighted
           and width-matched variants stay in Supplementary Table S2.
  p values read from outputs/patient_level.csv, the single source Results
           quotes, so the figure cannot drift from the text.

Panels, one claim question each
-------------------------------
  a  How much accuracy is won?          Per-patient Dice, with the share of
                                        failed delineations (Dice < 0.5) above
                                        each box.
  b  What does that mean clinically?    Share of patients whose whole lesion
                                        falls inside the prediction dilated by
                                        a k-pixel safety margin.
  c  Could it just be the threshold?    Each scheme's decision threshold swept
                                        over 19 values, drawn as the trade-off
                                        between missed lesion and normal tissue
                                        taken. One curve lying inside another
                                        means no threshold on the outer scheme
                                        reaches the inner one.
  d  What does it look like?            The frames where channel concatenation
                                        loses most, against ground truth and the
                                        proposed scheme.

A note on how panel c was chosen. The obvious drawing was a paired difference
with a "both improved" quadrant -- and it is the wrong one: only about a third
of cases improve on both axes at once, because the two medians are carried by
partly different cases. What the data do support is the operating-point-free
statement that the reachable SETS are nested, plus the robustness figure quoted
in the text: the proposed scheme is worse on both axes for 4% of white-light and
9% of narrow-band patients.

Data: EGC-DualSeg results/predictions_*.csv (Dice), results/boundary_*.csv
      (coverage and over-segmentation), results/sweep_*.csv (19 thresholds),
      outputs/patient_level.csv (published Holm p), results/exemplars/ plus the
      packaged manifest and .npz for panel d. No GPU, no checkpoints.
Backend: Python / matplotlib (project default).
Output: fig2_fusion_operator.{svg,pdf,tiff,png}

Palette discipline
  Blue / teal are EXCLUSIVELY frame identity -- white light versus narrow band.
  The three schemes are encoded by fill, hatch and line style inside that
  colour, never by a third hue. Red is reserved for reference lines, medians
  and thresholds; grey for neutral annotation.

The two frames are two separate comparisons: their ground-truth masks are
delineated in different coordinate systems, so a value in one is never ranked
against a value in the other. Panels a-c keep them side by side but never
pooled, and the caption says so.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

# ---- MANDATORY: editable SVG/PDF text (before any figure is created) -------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",     # text stays as <text> nodes, editable downstream
    "pdf.fonttype": 42,         # embedded TrueType, editable in PDF
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

# Repository root, resolved from this file so the scripts run from a clone.
# Override with EGC_REPO if the read-outs live elsewhere.
REPO = os.environ.get("EGC_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
EXEMPLARS = os.path.join(RESULTS, "exemplars")
MANIFEST = os.path.join(REPO, "data", "packaged", "manifest.csv")
NPZ = os.path.join(REPO, "data", "packaged", "egc_dualseg_384.npz")
OUT = os.path.dirname(os.path.abspath(__file__))

C_WLI = "#0F4D92"      # frame identity: white light
C_NBI = "#42949E"      # frame identity: image-enhanced
C_SIG = "#B64342"      # medians, thresholds, reference lines
C_NEU = "#767676"

FUSION = "ours"
FRAMES = {
    "WLI": dict(colour=C_WLI, single="wli_only", early="early_fusion_wli",
                label="WLI"),
    # Canonical per the manuscript terminology ledger. 18 of 250 enhanced frames
    # come from Fujifilm scopes, whose enhancement mode is not Olympus NBI; the
    # ledger's <TO CONFIRM> covers that and the caption carries the caveat.
    "NBI": dict(colour=C_NBI, single="nbi_only", early="early_fusion_nbi",
                label="NBI"),
}
# The three schemes drawn in panels a-c, ordered by the spatial scale at which
# each assumes the two frames correspond: every pixel, a 1/32 grid, none. That
# ordering is the argument of Results 3.3 and the panels read left to right in
# it. The MMD-weighted and width-matched variants stay in Supplementary Table S2.
SCHEMES = ["early", "mid", "ours"]
CONFIG = {"mid": "mid_fusion", "ours": FUSION}     # "early" is frame-specific
# Scheme inside a frame colour: fill, hatch and line style only, never a hue.
STYLE = {
    "early": dict(fill="white", hatch="///", ls="--",
                  label="Channel concatenation (every pixel)"),
    "mid": dict(fill="white", hatch=None, ls="-.",
                label="Bottleneck concatenation (1/32 grid)"),
    "ours": dict(fill=None, hatch=None, ls="-",
                 label="Cross-attention, no alignment (proposed)"),
}
KEY = ["case", "scale", "ref_modality"]
PKEY = ["case", "ref_modality"]         # the unit of analysis: the patient
MARGINS = np.arange(0, 61, 2)


def scheme_config(frame, key):
    """Resolve a scheme name to the config that ran it in this frame."""
    return FRAMES[frame]["early"] if key == "early" else CONFIG[key]


def p_text(p):
    if not np.isfinite(p):
        return "p n/a"
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.3f}"


C_REF = "#FFD700"        # gold reference contour, as in fig1_misalignment
C_HALO = "#141414"       # neutral rim; adds no hue, so the palette rule holds
PEN, RIM = 4, 3          # colour width, and how far the rim extends past it


def outline(mask, width=1):
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


# ---------------------------------------------------------------------------
def read_group(pattern):
    import glob
    files = glob.glob(os.path.join(RESULTS, pattern))
    if not files:
        raise SystemExit(f"no {pattern} in {RESULTS}")
    return pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in files],
                     ignore_index=True)


def to_patient(df, cols, extra=()):
    """Seeds averaged into images, images averaged into patients.

    The unit of analysis is the patient, not the image: 146 of the 154
    patient-by-frame combinations contribute both a near-focus and a distant
    pair, and those two are not independent observations. Averaging happens
    within a reference frame only, because the two frames carry ground truth
    drawn in different coordinate systems.
    """
    by = KEY + ["config"] + list(extra)
    per_image = df.groupby(by, as_index=False)[cols].mean()
    return per_image.groupby(PKEY + ["config"] + list(extra),
                             as_index=False)[cols].mean()


def load_dice():
    return to_patient(read_group("predictions_*.csv"), ["dice"])


def load_coverage():
    """Per-patient coverage quantities from the boundary pass.

    The margin indicators are built per image, before any averaging, because
    "is this lesion covered at k px" is only defined on an image. Averaged
    within a patient the indicator becomes a proportion in {0, 0.5, 1}, which
    is what panel b plots.
    """
    df = read_group("boundary_*.csv")
    for k in MARGINS:
        df[f"cov{k}"] = (df["residual_px"] <= k).astype(float)
    cols = ["residual_px", "residual_area_frac", "over_area_ratio"] + \
           [f"cov{k}" for k in MARGINS]
    out = to_patient(df, cols)
    return out.rename(columns={"residual_area_frac": "missed",
                               "over_area_ratio": "over"})


def load_coverage_by_seed():
    """Same, kept per seed so panel b can show the spread across seeds."""
    df = read_group("boundary_*.csv")
    for k in MARGINS:
        df[f"cov{k}"] = (df["residual_px"] <= k).astype(float)
    cols = [f"cov{k}" for k in MARGINS]
    per_image = df.groupby(KEY + ["config", "seed"], as_index=False)[cols].mean()
    return per_image.groupby(PKEY + ["config", "seed"],
                             as_index=False)[cols].mean()


def holm_lookup():
    """Adjusted p values as published, so the figure cannot drift from the text.

    outputs/patient_level.csv is the single source Results quotes. Recomputing a
    raw Wilcoxon here would print a different number next to the same box.
    """
    path = os.path.join(REPO, "outputs", "patient_level.csv")
    if not os.path.isfile(path):
        return {}
    d = pd.read_csv(path, encoding="utf-8-sig")
    return {(r.frame, r.metric, r.arm, r.baseline): r.p_holm
            for r in d.itertuples()}


DICE = load_dice()
COV = load_coverage()
COV_SEED = load_coverage_by_seed()
HOLM = holm_lookup()

# Every exclusion is counted before and after and printed at the end, so what
# the figure drops is auditable rather than implicit. The only exclusions here
# are images for which one of the two compared arms has no row — expected to be
# zero, since both arms ran on all five folds.
EXCLUSIONS = []

fig = plt.figure(figsize=(7.2, 4.5))
gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 0.58],
              hspace=0.55, wspace=1.05)


# ---- a | accuracy ---------------------------------------------------------
axa = fig.add_subplot(gs[0, 0:2])
# Panels a-c title from their own axes' left edge via loc="left"; panel d and the
# footnote are figure-level text and must be anchored to the same x, or they
# protrude past the plotted content.
TITLE_X = axa.get_position().x0
GROUP_W = 4.8        # three boxes per frame now, so the groups need more room
ticks = []
for i, (frame, spec) in enumerate(FRAMES.items()):
    sub = DICE[DICE.ref_modality == frame]
    cols = [scheme_config(frame, k) for k in SCHEMES]
    wide = sub.pivot_table(index="case", columns="config", values="dice")[cols]
    _before = len(wide)
    wide = wide.dropna()
    EXCLUSIONS.append((f"a/{frame}: patients missing a scheme", _before, len(wide)))
    for j, key in enumerate(SCHEMES):
        col = scheme_config(frame, key)
        pos = i * GROUP_W + j
        st = STYLE[key]
        bp = axa.boxplot([wide[col]], positions=[pos], widths=0.62,
                         showfliers=False, patch_artist=True,
                         medianprops=dict(color=C_SIG, lw=1.2),
                         whiskerprops=dict(color=spec["colour"], lw=0.9),
                         capprops=dict(color=spec["colour"], lw=0.9),
                         boxprops=dict(lw=0.9))
        patch = bp["boxes"][0]
        patch.set_facecolor(spec["colour"] if st["fill"] is None else st["fill"])
        patch.set_edgecolor(spec["colour"])
        if st["hatch"]:
            patch.set_hatch(st["hatch"])
        # Failed delineations: the share of patients the scheme simply loses.
        fail = 100 * float((wide[col] < 0.5).mean())
        axa.text(pos, 1.035, f"{fail:.0f}%", ha="center", va="bottom",
                 fontsize=5.8, color=C_NEU)
    # The contrast this panel is built around is Results 3.2: proposed against
    # channel concatenation. It is folded into the tick label because the
    # channel-concatenation whisker reaches the floor and an in-axes annotation
    # would sit on top of it. p is the published Holm value, not a fresh raw one.
    early = scheme_config(frame, "early")
    d = float((wide[FUSION] - wide[early]).median())
    p = HOLM.get((frame, "dice", FUSION, early))
    if p is None:
        p = float(wilcoxon(wide[FUSION], wide[early], zero_method="wilcox").pvalue)
    ticks.append((i * GROUP_W + 1.0,
                  f"{spec['label']}\nproposed − concat\n{d:+.3f}, {p_text(p)}"))

axa.axhline(0.5, color=C_SIG, lw=0.8, ls=":", zorder=0)
axa.text(GROUP_W + 2.7, 0.515, "failure threshold", fontsize=5.8, color=C_SIG,
         va="bottom", ha="right")
axa.set_xticks([t[0] for t in ticks])
axa.set_xticklabels([t[1] for t in ticks], fontsize=6.2)
axa.set_ylabel("Dice", fontsize=8)
axa.set_ylim(0, 1.13)
axa.set_xlim(-0.8, GROUP_W + 2.8)
axa.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axa.set_title("a  Segmentation accuracy", loc="left", fontsize=8, fontweight="bold", pad=9)
axa.text(-0.8, 1.105, "% below threshold", fontsize=6, color=C_NEU, va="bottom")


# ---- b | clinical coverage ------------------------------------------------
axb = fig.add_subplot(gs[0, 2:4])
for frame, spec in FRAMES.items():
    sub = COV[COV.ref_modality == frame]
    seedwise = COV_SEED[COV_SEED.ref_modality == frame]
    for key in SCHEMES:
        col = scheme_config(frame, key)
        g = sub[sub.config == col]
        _before = len(g)
        g = g.dropna(subset=[f"cov{k}" for k in MARGINS])
        EXCLUSIONS.append((f"b/{frame}/{key}: patients missing the scheme",
                           _before, len(g)))
        # Each patient contributes a proportion in {0, 0.5, 1} at every k, so
        # the curve is the mean of that proportion, not a count over images.
        rate = [100 * float(g[f"cov{k}"].mean()) for k in MARGINS]
        # Band = min-to-max across the three training seeds, the same
        # uncertainty definition used in panel a's underlying aggregation.
        per_seed = []
        for _, gs_ in seedwise[seedwise.config == col].groupby("seed"):
            per_seed.append([100 * float(gs_[f"cov{k}"].mean()) for k in MARGINS])
        if len(per_seed) > 1:
            arr = np.array(per_seed)
            axb.fill_between(MARGINS, arr.min(axis=0), arr.max(axis=0),
                             color=spec["colour"], alpha=0.11, lw=0, zorder=1)
        axb.plot(MARGINS, rate, color=spec["colour"], lw=1.5,
                 ls=STYLE[key]["ls"], zorder=3 if key == "ours" else 2)
axb.set_xlabel("Safety margin added to the prediction (px)", fontsize=7.5)
axb.set_ylabel("Lesion fully covered (%)", fontsize=8)
axb.set_xlim(0, 60)
axb.set_ylim(0, 82)
axb.set_title("b  Margin adequacy", loc="left", fontsize=8,
              fontweight="bold", pad=9)
# A clinical 3 mm margin is of order 50-100 px on this canvas, so the useful
# reading is at the larger k, not at zero.
axb.axvline(20, color=C_SIG, lw=0.8, ls=":", zorder=0)
axb.text(20.8, 74, "k = 20", fontsize=5.8, color=C_SIG)
axb.legend(handles=[Line2D([], [], color=C_NEU, ls=STYLE[k]["ls"], lw=1.5,
                           label=STYLE[k]["label"]) for k in SCHEMES]
           + [Line2D([], [], color=spec["colour"], lw=1.5, label=spec["label"])
              for spec in FRAMES.values()],
           fontsize=5.3, loc="upper left", handlelength=1.6, borderpad=0.15,
           labelspacing=0.12, ncol=2, columnspacing=0.7,
           bbox_to_anchor=(-0.02, 1.42))


# ---- c | the argument a threshold cannot make -----------------------------
# The per-image paired difference was the obvious way to draw this and it is the
# wrong one: only about a third of images improve on BOTH axes, because the two
# medians are carried by partly different images. What is genuinely
# operating-point-free is the trade-off CURVE — sweep the decision threshold on
# each operator and compare the reachable sets. If one curve lies inside the
# other, no threshold on the outer operator can reach the inner one, and that is
# the statement a threshold shift cannot manufacture.
def sweep_curves():
    import glob
    files = glob.glob(os.path.join(RESULTS, "sweep_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in files],
                   ignore_index=True)
    # Patient level, like every other panel: seeds into images, images into
    # patients, separately at each threshold.
    return to_patient(df, ["missed_frac", "over_ratio"], extra=["threshold"])


# Panel c carries no uncertainty band by design: four curves plus four bands
# would obscure the nesting that is the entire point, and the uncertainty for
# this comparison is already reported as paired tests in the text. Panel b,
# which makes a magnitude claim rather than a nesting claim, does show the
# seed-to-seed spread.
axc = fig.add_subplot(gs[0, 4:6])
SWEEP = sweep_curves()
worse_both = {}
for frame, spec in FRAMES.items():
    # Robustness figure quoted next to the curve: how often the proposed scheme
    # is worse on BOTH axes for the same patient. Small is the claim.
    sub = COV[COV.ref_modality == frame]
    w_m = sub.pivot_table(index="case", columns="config", values="missed")
    w_o = sub.pivot_table(index="case", columns="config", values="over")
    pair = pd.DataFrame({"dm": w_m[FUSION] - w_m[spec["early"]],
                         "do": w_o[FUSION] - w_o[spec["early"]]})
    _before = len(pair)
    pair = pair.dropna()
    EXCLUSIONS.append((f"c/{frame}: patients missing a scheme", _before, len(pair)))
    worse_both[frame] = 100 * float(((pair.dm >= 0) & (pair.do >= 0)).mean())

    if SWEEP is None:
        continue
    s = SWEEP[SWEEP.ref_modality == frame]
    for key in SCHEMES:
        col = scheme_config(frame, key)
        c = s[s.config == col].groupby("threshold")[
            ["missed_frac", "over_ratio"]].mean().sort_values("over_ratio")
        if c.empty:
            continue
        axc.plot(c.over_ratio, c.missed_frac, color=spec["colour"], lw=1.4,
                 ls=STYLE[key]["ls"], zorder=3 if key == "ours" else 2)
        half = s[(s.config == col) & (np.isclose(s.threshold, 0.5))][
            ["missed_frac", "over_ratio"]].mean()
        axc.scatter([half.over_ratio], [half.missed_frac], s=16,
                    facecolor=spec["colour"] if key == "ours" else "white",
                    edgecolor=spec["colour"], lw=1.0, zorder=4)

axc.set_xlim(0, 1.75)
axc.set_ylim(0.05, 0.45)
axc.set_xlabel("Normal tissue taken (× lesion area)", fontsize=7.5)
axc.set_ylabel("Lesion missed (fraction)", fontsize=8)
axc.set_title("c  Reachable set, all thresholds", loc="left", fontsize=8,
              fontweight="bold", pad=9)
axc.annotate("better", xy=(1.16, 0.235), xytext=(1.72, 0.345),
             fontsize=6.2, color=C_NEU, va="center",
             arrowprops=dict(arrowstyle="->", color=C_NEU, lw=0.8))
axc.text(1.72, 0.445, "filled dot = threshold 0.5", fontsize=5.8,
         color=C_NEU, ha="right", va="top")


# ---- d | qualitative ------------------------------------------------------
def qualitative_picks():
    """The images where naive early fusion loses most, by the recorded rule.

    The exemplar dump wrote one mask per (config, image) and stored which rule
    selected it; `contrast` is the rule that took the largest losses for early
    fusion, which is exactly the claim panels a-c quantify. Every mask shown is
    a held-out prediction from the fold that excluded its patient.
    """
    index_path = os.path.join(EXEMPLARS, "index.csv")
    if not os.path.isfile(index_path):
        return None
    idx = pd.read_csv(index_path, encoding="utf-8-sig")
    man = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    lookup = {(r.case, r.scale, r.modality): int(r.idx) for r in man.itertuples()}
    crop = {(r.case, r.scale, r.modality):
            (int(r.pad_y), int(r.pad_y) + int(r.content_h),
             int(r.pad_x), int(r.pad_x) + int(r.content_w))
            for r in man.itertuples()}
    picks = []
    for frame, spec in FRAMES.items():
        sub = idx[(idx.ref_modality == frame)
                  & (idx.rule.astype(str).str.contains("contrast"))]
        if sub.empty:
            continue
        wide = sub.pivot_table(index=["case", "scale"], columns="config",
                               values="dice_this_seed")
        if FUSION not in wide or spec["early"] not in wide:
            continue
        gap = (wide[FUSION] - wide[spec["early"]]).dropna()
        if gap.empty:
            continue
        case, scale = gap.idxmax()
        picks.append((frame, case, scale, sub))
    return picks, idx, lookup, crop


picks = qualitative_picks()
if picks:
    picks, idx, lookup, crop = picks
    blob = np.load(NPZ)
    images, masks = blob["images"], blob["masks"]
    from PIL import Image

    # A spacer column between the two frame groups: without it the second
    # group's frame label lands on top of the first group's last tile.
    n_col = 3 * len(picks) + (len(picks) - 1)
    inner = gs[1, 0:6].subgridspec(1, n_col, wspace=0.05,
                                   width_ratios=[1, 1, 1, 0.35, 1, 1, 1][:n_col])
    for pi, (frame, case, scale, sub) in enumerate(picks):
        spec = FRAMES[frame]
        row = lookup.get((case, scale, frame))
        if row is None:
            continue
        y0, y1, x0, x1 = crop[(case, scale, frame)]
        img = images[row][y0:y1, x0:x1]
        gt = masks[row][y0:y1, x0:x1] > 127

        panels = [("Ground truth", None, C_REF),
                  ("Channel concat.", spec["early"], C_NEU),
                  ("Proposed", FUSION, spec["colour"])]
        for ci, (title, cfg, colour) in enumerate(panels):
            ax = fig.add_subplot(inner[0, pi * 4 + ci])
            ax.imshow(img)
            if cfg is None:
                pred = gt
            else:
                hit = sub[(sub.case == case) & (sub.scale == scale)
                          & (sub.config == cfg)]
                if hit.empty:
                    ax.axis("off"); continue
                path = os.path.join(EXEMPLARS, hit.iloc[0]["path"])
                if not os.path.isfile(path):
                    ax.axis("off"); continue
                pred = np.array(Image.open(path)) > 0
                pred = pred[y0:y1, x0:x1]
            for w, c, alpha in ((PEN + RIM, C_HALO, 0.85), (PEN, colour, 1.0)):
                ov = np.zeros(img.shape[:2] + (4,), float)
                ov[outline(pred, w)] = (*matplotlib.colors.to_rgb(c), alpha)
                ax.imshow(ov)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(spec["colour"]); s.set_linewidth(0.9)
            ax.set_title(title, fontsize=5.6, color=colour, pad=2.0)
            if ci == 1:
                ax.text(0.5, 1.22, spec["label"], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7,
                        color=spec["colour"], fontweight="bold")
            if cfg is not None:
                hit = sub[(sub.case == case) & (sub.scale == scale)
                          & (sub.config == cfg)]
                if not hit.empty:
                    ax.text(0.5, -0.05, f"Dice {hit.iloc[0]['dice_this_seed']:.2f}",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=5.8, color=colour)
    fig.text(TITLE_X, 0.375, "d  Where channel concatenation loses most", fontsize=8,
             fontweight="bold", va="bottom")

base = os.path.join(OUT, "fig2_fusion_operator")
fig.savefig(base + ".svg", bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
fig.savefig(base + ".tiff", dpi=600, bbox_inches="tight")
print("saved ->", base + ".{svg,pdf,png,tiff}")
print("\nexclusions (before -> after):")
for what, a, b in EXCLUSIONS:
    flag = "" if a == b else "   <-- DROPPED"
    print(f"  {what}: {a} -> {b}{flag}")
print("\npanel c: fraction of images WORSE on both axes -> "
      + "  ".join(f"{k} {v:.1f}%" for k, v in worse_both.items()))
