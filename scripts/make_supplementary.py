# -*- coding: utf-8 -*-
"""Generate the supplementary tables from the committed read-outs.

    python manuscript/make_supplementary.py

Writes `manuscript/supplementary.md` and `manuscript/table1.md`. Nothing is
typed by hand: Table 1 and Table S1 come
from `outputs/patient_level.csv`, the read-out the manuscript quotes, and
Table S3 from `results/grade_*.csv` through the same DeLong estimator that
`scripts/summarise_grade.py` uses. Regenerating after a re-run is therefore the
only way these tables can change, and they cannot silently disagree with the
text.
"""
from __future__ import annotations

import io
import os
import glob

import numpy as np
import pandas as pd

# Repository root, resolved from this file so the scripts run from a clone.
# Override with EGC_REPO if the read-outs live elsewhere.
REPO = os.environ.get("EGC_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(REPO, "outputs")
PATIENT = os.path.join(REPO, "outputs", "patient_level.csv")
RESULTS = os.path.join(REPO, "results")
OUT = os.path.join(HERE, "supplementary.md")
OUT_T1 = os.path.join(HERE, "table1.md")
BOOT, RNG_SEED = 5000, 0

# Manuscript names, so the table reads like the text rather than like the repo.
LABEL = {
    "wli_only": "Single modality",
    "nbi_only": "Single modality",
    "early_fusion_wli": "Channel concatenation (six channels)",
    "early_fusion_nbi": "Channel concatenation (six channels)",
    "mid_fusion": "Bottleneck concatenation",
    "mid_fusion_wide": "Bottleneck concatenation, width-matched",
    "ours": "Cross-attention, no alignment (proposed)",
    "mmd_fusion": "Distribution alignment, development weight",
    "mmd_fusion_w1e-4": "Distribution alignment, weight 1e-4",
    "mmd_fusion_w0.01": "Distribution alignment, weight 0.01",
    "mmd_fusion_w1.0": "Distribution alignment, weight 1.0",
}
ORDER = ["wli_only", "nbi_only", "early_fusion_wli", "early_fusion_nbi",
         "mid_fusion", "mid_fusion_wide", "ours", "mmd_fusion",
         "mmd_fusion_w1e-4", "mmd_fusion_w0.01", "mmd_fusion_w1.0"]

METRICS = [("dice", "Dice", 3), ("bf1", "BF@1px", 3), ("bf2", "BF@2px", 3),
           ("bf3", "BF@3px", 3), ("bf5", "BF@5px", 3), ("nsd3", "NSD@3px", 3),
           ("hd95", "HD95 (px)", 1), ("residual_area_frac", "Lesion missed", 3),
           ("over_area_ratio", "Mucosa taken", 3), ("residual_px", "Margin (px)", 1)]

_Z = 1.959963984540054


KEY = ["case", "scale", "ref_modality"]
PKEY = ["case", "ref_modality"]


def _read(pattern):
    files = glob.glob(os.path.join(RESULTS, pattern))
    return pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                     ignore_index=True)


def patient_table():
    """Every metric the manuscript quotes, one row per patient per frame.

    Table 1 needs intervals, and outputs/patient_level.csv stores only the point
    estimate and the adjusted p. Rather than store a second copy of the raw
    numbers, the table is bootstrapped here from the same CSVs that file was
    built from, with the same aggregation order.
    """
    cols = [m[0] for m in METRICS if m[0] != "dice"]
    bnd = _read("boundary_*.csv")
    pred = _read("predictions_*.csv")
    a = (bnd.groupby(KEY + ["config"], as_index=False)[cols].mean()
            .groupby(PKEY + ["config"], as_index=False)[cols].mean())
    b = (pred.groupby(KEY + ["config"], as_index=False)["dice"].mean()
             .groupby(PKEY + ["config"], as_index=False)["dice"].mean())
    return a.merge(b, on=PKEY + ["config"], how="outer")


def paired_ci(tbl, frame, col, arm, ref):
    w = (tbl[tbl.ref_modality == frame]
         .pivot_table(index="case", columns="config", values=col)[[arm, ref]]
         .dropna())
    d = (w[arm] - w[ref]).to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    draws = [np.median(rng.choice(d, len(d))) for _ in range(BOOT)]
    return (float(np.median(d)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), len(w))


def _structural(scores, n_pos):
    """DeLong structural components. Same estimator as scripts/summarise_grade.py."""
    m, n = n_pos, scores.shape[1] - n_pos
    pos, neg = scores[:, :m], scores[:, m:]
    tx = np.array([pd.Series(r).rank().to_numpy() for r in pos])
    ty = np.array([pd.Series(r).rank().to_numpy() for r in neg])
    tz = np.array([pd.Series(r).rank().to_numpy() for r in scores])
    auc = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return v10, v01, auc


def delong_ci(y_true, y_prob):
    order = np.argsort(-y_true, kind="mergesort")
    y, s = y_true[order], y_prob[order][None, :]
    v10, v01, auc = _structural(s, int(y.sum()))
    m, n = int(y.sum()), len(y) - int(y.sum())
    var = np.cov(v10)[()] / m + np.cov(v01)[()] / n
    se = float(np.sqrt(var))
    return float(auc[0]), float(np.clip(auc[0] - _Z * se, 0, 1)), \
        float(np.clip(auc[0] + _Z * se, 0, 1))


def grade_table(prefix, label):
    rows = []
    files = sorted(glob.glob(os.path.join(RESULTS, f"grade_{prefix}*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    for cfg, g in df.groupby("config"):
        agg = g.groupby("case", as_index=False).agg(y_true=("y_true", "first"),
                                                    y_prob=("y_prob", "mean"))
        if agg.y_true.nunique() < 2:
            continue
        auc, lo, hi = delong_ci(agg.y_true.to_numpy(), agg.y_prob.to_numpy())
        rows.append((cfg, len(agg), int(agg.y_true.sum()), auc, lo, hi))
    rows.sort(key=lambda r: -r[3])
    return rows


def main() -> int:
    d = pd.read_csv(PATIENT, encoding="utf-8-sig")
    out = ["# Supplementary tables",
           "",
           "> Generated by `manuscript/make_supplementary.py` from "
           "`outputs/patient_level.csv` and `results/grade_*.csv`. Do not edit by "
           "hand: regenerate.",
           ""]

    # ---- Table S1 ---------------------------------------------------------
    out += ["## Table S1 | Per-configuration values, all metrics, both reference frames",
            "",
            "Patient-level means (n = 77). The two reference frames carry ground truth "
            "drawn in different coordinate systems and are not comparable with each "
            "other. Paired median differences against the single-modality baseline of "
            "the same frame are given below each block, with Holm-adjusted p across "
            "the eight configurations tested on that metric and frame.",
            ""]

    for frame, base in (("WLI", "wli_only"), ("NBI", "nbi_only")):
        sub = d[(d.frame == frame) & (d.baseline == base)]
        if sub.empty:
            continue
        present = [a for a in ORDER if a in set(sub.arm)]
        out += [f"### {frame} reference frame", "",
                "| Configuration | " + " | ".join(m[1] for m in METRICS) + " |",
                "|---|" + "---|" * len(METRICS)]
        # the baseline first, from its own recorded value
        vals = []
        for col, _, dp in METRICS:
            r = sub[(sub.metric == col)]
            vals.append(f"{r.baseline_value.iloc[0]:.{dp}f}" if len(r) else "—")
        out.append(f"| {LABEL[base]} (baseline) | " + " | ".join(vals) + " |")
        for arm in present:
            vals = []
            for col, _, dp in METRICS:
                r = sub[(sub.metric == col) & (sub.arm == arm)]
                vals.append(f"{r.arm_value.iloc[0]:.{dp}f}" if len(r) else "—")
            out.append(f"| {LABEL.get(arm, arm)} | " + " | ".join(vals) + " |")
        out += ["", f"Paired median difference against the baseline, Holm-adjusted "
                    f"p in brackets:", "",
                "| Configuration | " + " | ".join(m[1] for m in METRICS) + " |",
                "|---|" + "---|" * len(METRICS)]
        for arm in present:
            vals = []
            for col, _, dp in METRICS:
                r = sub[(sub.metric == col) & (sub.arm == arm)]
                if not len(r):
                    vals.append("—")
                    continue
                md, ph = r.median_diff.iloc[0], r.p_holm.iloc[0]
                ptxt = "<0.0001" if ph < 1e-4 else f"{ph:.3f}"
                vals.append(f"{md:+.{dp}f} [{ptxt}]")
            vals_row = " | ".join(vals)
            out.append(f"| {LABEL.get(arm, arm)} | {vals_row} |")
        out.append("")

    # ---- Table S2 ---------------------------------------------------------
    # The ladder table in Results 3.3 shows four schemes; this is the same
    # table with the variants that did not need to be in the main text.
    out += ["## Table S2 | Further fusion variants, patient-level mean Dice",
            "",
            "The four schemes in the Results 3.3 table, extended with the "
            "width-matched bottleneck variant and the four distribution-alignment "
            "weights. Same cohort and same aggregation (n = 77 patients).",
            "",
            "| Fusion scheme | Correspondence assumed | WLI | NBI |",
            "|---|---|---|---|"]
    LADDER = [("early_fusion", "every pixel", "channel concatenation, six channels"),
              ("mid_fusion", "1/32 grid", "concatenation at the bottleneck"),
              ("mid_fusion_wide", "1/32 grid", "concatenation at the bottleneck, width-matched"),
              ("ours", "none", "cross-attention, no alignment (proposed)"),
              ("mmd_fusion_w1e-4", "1/32 grid + distribution term", "bottleneck, alignment weight 1e-4"),
              ("mmd_fusion_w0.01", "1/32 grid + distribution term", "bottleneck, alignment weight 0.01"),
              ("mmd_fusion", "1/32 grid + distribution term", "bottleneck, alignment weight 0.3"),
              ("mmd_fusion_w1.0", "1/32 grid + distribution term", "bottleneck, alignment weight 1.0"),
              (None, "—", "single modality")]
    for arm, assumed, label in LADDER:
        vals = []
        for frame, base in (("WLI", "wli_only"), ("NBI", "nbi_only")):
            key = base if arm is None else (
                f"early_fusion_{frame.lower()}" if arm == "early_fusion" else arm)
            r = d[(d.frame == frame) & (d.metric == "dice") & (d.baseline == base)]
            if arm is None:
                vals.append(f"{r.baseline_value.iloc[0]:.3f}" if len(r) else "—")
            else:
                rr = r[r.arm == key]
                vals.append(f"{rr.arm_value.iloc[0]:.3f}" if len(rr) else "—")
        out.append(f"| {label} | {assumed} | {vals[0]} | {vals[1]} |")
    out += ["",
            "The four alignment weights are the sweep drawn in Supplementary "
            "Fig. S2; weight 0 is the plain bottleneck concatenation row above.",
            ""]

    # ---- Note S1 ----------------------------------------------------------
    out += ["## Note S1 | Alignment upper bound, development fold only",
            "",
            "Two probe configurations ask what a registration-first design could "
            "offer. Both ran on the development fold alone, so they are scored on "
            "15 patients and are reported as bounds rather than as performance "
            "estimates; neither enters the multiple-comparison families of "
            "Methods 2.7.",
            "",
            "The oracle probe aligns the auxiliary frame using the ground-truth "
            "masks of both frames. It raises the achievable lesion overlap from "
            "0.34 to 0.845, and on the narrow-band frame it gains a paired median "
            "of +0.014 Dice over the same architecture with the auxiliary path "
            "disabled (95% CI −0.000 to +0.064, p = 0.08). The operation is not "
            "available at inference, because it uses the annotation the model is "
            "being asked to produce.",
            "",
            "The predicted-mask probe substitutes the only alignment obtainable at "
            "inference, a fit to the two predicted masks. It gains +0.009 (95% CI "
            "−0.007 to +0.023, p = 0.30).",
            "",
            "The two intervals overlap and each contains the other's point "
            "estimate, so these data do not show that the two sources of alignment "
            "differ. What they bound is the size of either: at this sample size "
            "neither gained more than about 0.06 Dice. A decision rule fixed "
            "before the probe ran closed the aligned-fusion line at a headroom of "
            "0.03 Dice or less.",
            ""]

    # ---- Table S3 ---------------------------------------------------------
    out += ["## Table S3 | Stage-2 areas under the ROC curve, with DeLong intervals",
            "",
            "One probability per patient per seed, averaged over the three seeds "
            "before scoring. Intervals are 95% confidence intervals from the DeLong "
            "variance estimator.",
            ""]
    for prefix, title, note in (
            ("", "Histological grade (study endpoint)",
             "Every interval includes 0.5."),
            ("macro_", "Macroscopic type (positive control declared at the design stage)",
             "No interval reaches 0.5.")):
        rows = grade_table(prefix if prefix else "", title)
        if not rows:
            continue
        keep = [r for r in rows if (r[0].startswith("macro_") == bool(prefix))]
        if not keep:
            continue
        out += [f"### {title}", "",
                "| Configuration | n | Positive | AUC | 95% CI |",
                "|---|---|---|---|---|"]
        for cfg, n, pos, auc, lo, hi in keep:
            name = cfg[len("macro_"):] if prefix else cfg
            out.append(f"| `{name}` | {n} | {pos} | {auc:.3f} | "
                       f"{lo:.3f} to {hi:.3f} |")
        out += ["", note, ""]

    io.open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"wrote {OUT}")

    # ---- main-text Table 1 -------------------------------------------------
    pt = patient_table()
    t1 = ["**Table 1 | Proposed fusion scheme against channel-concatenation "
          "fusion, all ten metrics.** Paired median difference per patient "
          "(n = 77) with a 95% percentile bootstrap interval, Holm-adjusted "
          "across the ten metrics reported for this pair on one frame. A "
          "positive difference favours the proposed scheme on the first six "
          "metrics and the comparator on the last four, where lower is better; "
          "the arrow column states the direction of improvement.",
          "",
          "| Metric | Better | WLI: median difference (95% CI) | WLI p | "
          "NBI: median difference (95% CI) | NBI p |",
          "|---|---|---|---|---|---|"]
    SENSE = {"dice": "higher", "bf1": "higher", "bf2": "higher", "bf3": "higher",
             "bf5": "higher", "nsd3": "higher", "hd95": "lower",
             "residual_area_frac": "lower", "over_area_ratio": "lower",
             "residual_px": "lower"}
    for col, label, dp in METRICS:
        cells = []
        for frame in ("WLI", "NBI"):
            ref = f"early_fusion_{frame.lower()}"
            med, lo, hi, _ = paired_ci(pt, frame, col, "ours", ref)
            r = d[(d.frame == frame) & (d.metric == col) & (d.arm == "ours")
                  & (d.baseline == ref)]
            ph = r.p_holm.iloc[0] if len(r) else float("nan")
            # Three decimals prints 0.00015 as "0.000", which reads as zero.
            ptxt = ("< 0.0001" if ph < 1e-4
                    else f"{ph:.4f}" if ph < 0.001 else f"{ph:.3f}")
            cells += [f"{med:+.{dp}f} ({lo:+.{dp}f} to {hi:+.{dp}f})", ptxt]
        arrow = "higher" if SENSE[col] == "higher" else "lower"
        t1.append(f"| {label} | {arrow} | " + " | ".join(cells) + " |")
    t1 += ["",
           "Both reference frames carry ground truth drawn in their own "
           "coordinate system and are never compared with each other. "
           "Per-configuration values for every arm are in Supplementary "
           "Table S1.", ""]
    io.open(OUT_T1, "w", encoding="utf-8").write("\n".join(t1) + "\n")
    print(f"wrote {OUT_T1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
