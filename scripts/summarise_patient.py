# -*- coding: utf-8 -*-
"""Patient-level read-out. This is the one the manuscript quotes.

    python scripts/summarise_patient.py
    python scripts/summarise_patient.py --results results --out outputs/patient_level.csv

Why this exists
---------------
The 77 patients contribute 150 same-view pairs, because 146 of the 154
patient-by-frame combinations supply both a near-focus and a distant pair. Two
pairs from one patient are not independent observations, so a paired test over
150 images claims roughly twice the sample size the cohort actually has.

Methods 2.7 always specified per-patient testing; the earlier analyses did not
do it. Moving to the patient level changes exactly one verdict among the main
comparisons — `ours - nbi_only` on Dice goes from p = 0.024 to p = 0.108 — and
leaves the boundary and clinical results intact or stronger. That one flip is
load-bearing for Section 3.3, which is why this script exists rather than a
footnote saying the difference is negligible.

Aggregation order, and it matters:

    seeds -> image        a run is a sample of the training procedure, so the
                          three seeds are averaged before anything else
    images -> patient     within a reference frame only; the two frames have
                          different ground-truth masks and are never pooled

Multiplicity: p values are adjusted by Holm-Bonferroni within each
(reference frame, metric) family, that family being every arm compared against
its own single-modality baseline. Raw and adjusted values are both reported, so
a reader can see what the correction cost.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEY = ["case", "scale", "ref_modality"]
FRAMES = {"WLI": "wli_only", "NBI": "nbi_only"}
PROPOSED = "ours"


def is_probe(config: str) -> bool:
    """Alignment-probe arms, which do not belong in the main comparison family.

    They ran on the development fold only (15 patients rather than 77) and they
    answer a different question — what alignment could buy, not whether the
    second modality helps. Two reasons to keep them out of the Holm family:
    including them inflates the family and weakens the correction for the arms
    the claim rests on, and the oracle arms use ground-truth masks at inference,
    so a "significant" oracle result means nothing. They are reported separately
    and uncorrected, labelled as bounds.
    """
    return config.startswith("oracle") or config.startswith("pred_")

# Metrics quoted in the manuscript. `sense` is the direction of improvement.
METRICS = [
    ("dice", "higher", "Dice"),
    ("bf1", "higher", "boundary-F @1px"),
    ("bf2", "higher", "boundary-F @2px"),
    ("bf3", "higher", "boundary-F @3px"),
    ("bf5", "higher", "boundary-F @5px"),
    ("nsd3", "higher", "normalised surface dice @3px"),
    ("hd95", "lower", "95th-percentile Hausdorff"),
    ("residual_area_frac", "lower", "lesion missed (fraction)"),
    ("over_area_ratio", "lower", "normal tissue taken (x lesion)"),
    ("residual_px", "lower", "margin required (px)"),
]
MARGINS = [0, 5, 10, 20, 40]


def load(results: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    def read(pattern):
        files = glob.glob(os.path.join(results, pattern))
        if not files:
            raise SystemExit(f"no {pattern} under {results}")
        return pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                         ignore_index=True)

    pred = read("predictions_*.csv")
    bnd = read("boundary_*.csv")
    return pred, bnd


def to_patient(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Seeds averaged into images, images averaged into patients."""
    cols = [c for c in cols if c in df.columns]
    per_image = df.groupby(KEY + ["config"], as_index=False)[cols].mean()
    return per_image.groupby(["case", "ref_modality", "config"],
                             as_index=False)[cols].mean()


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down, returned in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def compare(tbl: pd.DataFrame, frame: str, arm: str, ref: str, col: str):
    sub = tbl[tbl.ref_modality == frame]
    wide = sub.pivot_table(index="case", columns="config", values=col)
    if arm not in wide or ref not in wide:
        return None
    w = wide[[arm, ref]].dropna()
    if len(w) < 5:
        return None
    d = w[arm] - w[ref]
    if np.allclose(d, 0):
        return dict(n=len(w), median=0.0, mean=0.0, p=1.0,
                    arm_value=float(w[arm].mean()), ref_value=float(w[ref].mean()))
    return dict(n=len(w), median=float(d.median()), mean=float(d.mean()),
                p=float(wilcoxon(w[arm], w[ref]).pvalue),
                arm_value=float(w[arm].mean()), ref_value=float(w[ref].mean()))


def coverage_table(bnd_patient: pd.DataFrame, frame: str, arms: list[str],
                   ref: str) -> pd.DataFrame:
    """Margin adequacy, kept on the patient scale.

    Per image the read-out is binary — is the whole lesion inside the prediction
    dilated by k px. Averaged within a patient it becomes a proportion in
    {0, 0.5, 1}, so the paired test is Wilcoxon on that proportion rather than
    McNemar on a binary. Reporting McNemar over images would reintroduce exactly
    the independence assumption this script exists to remove.
    """
    rows = []
    for k in MARGINS:
        sub = bnd_patient[bnd_patient.ref_modality == frame]
        wide = sub.pivot_table(index="case", columns="config", values=f"cov{k}")
        for arm in arms:
            if arm not in wide or ref not in wide:
                continue
            w = wide[[arm, ref]].dropna()
            d = w[arm] - w[ref]
            p = 1.0 if np.allclose(d, 0) else float(wilcoxon(w[arm], w[ref]).pvalue)
            rows.append(dict(frame=frame, k=k, arm=arm, n=len(w),
                             arm_pct=100 * float(w[arm].mean()),
                             ref_pct=100 * float(w[ref].mean()),
                             delta_pp=100 * float(d.mean()), p=p))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="outputs/patient_level.csv")
    args = ap.parse_args()

    pred, bnd = load(args.results)

    # Coverage indicators are built per image, before any averaging, because
    # "is this lesion covered at k px" is only defined on an image.
    for k in MARGINS:
        bnd[f"cov{k}"] = (bnd["residual_px"] <= k).astype(float)

    dice_pat = to_patient(pred, ["dice"])
    bnd_pat = to_patient(bnd, [m for m, _, _ in METRICS] + [f"cov{k}" for k in MARGINS])

    arms = sorted(set(bnd_pat.config) | set(dice_pat.config))
    records = []

    for frame, baseline in FRAMES.items():
        print(f"\n{'=' * 78}\n=== reference frame: {frame}   baseline {baseline} ===")
        print("    the two frames have different ground-truth masks and are "
              "not comparable")
        n_pat = dice_pat[(dice_pat.ref_modality == frame)
                         & (dice_pat.config == baseline)].case.nunique()
        print(f"    unit of analysis: patient (n = {n_pat})")

        for col, sense, label in METRICS:
            table = dice_pat if col == "dice" else bnd_pat
            if col not in table.columns:
                continue
            family = [a for a in arms if a != baseline and not is_probe(a)]
            results = {a: compare(table, frame, a, baseline, col) for a in family}
            results = {a: r for a, r in results.items() if r}
            if not results:
                continue
            names = list(results)
            adj = holm([results[a]["p"] for a in names])

            print(f"\n  -- {label}  ({sense} is better) --")
            print(f"     {baseline:22s} {results[names[0]]['ref_value']:8.4f}")
            for a, padj in sorted(zip(names, adj), key=lambda t: -abs(results[t[0]]['median'])):
                r = results[a]
                star = "*" if padj < 0.05 else " "
                note = "   <-- proposed" if a == PROPOSED else ""
                print(f"     {a:22s} {r['arm_value']:8.4f}   "
                      f"median {r['median']:+.4f}  p = {r['p']:.4f}  "
                      f"Holm = {padj:.4f}{star}  n = {r['n']}{note}")
                records.append(dict(frame=frame, metric=col, arm=a,
                                    baseline=baseline, n=r["n"],
                                    arm_value=r["arm_value"],
                                    baseline_value=r["ref_value"],
                                    median_diff=r["median"], mean_diff=r["mean"],
                                    p_raw=r["p"], p_holm=padj))

        # The operator contrast is a different question from the modality one,
        # so it is a separate family and gets its own correction.
        early = f"early_fusion_{frame.lower()}"
        print(f"\n  -- proposed versus input-level fusion ({early}) --")
        pairs = []
        for col, sense, label in METRICS:
            table = dice_pat if col == "dice" else bnd_pat
            if col not in table.columns:
                continue
            r = compare(table, frame, PROPOSED, early, col)
            if r:
                pairs.append((label, col, r))
        if pairs:
            adj = holm([r["p"] for _, _, r in pairs])
            for (label, col, r), padj in zip(pairs, adj):
                star = "*" if padj < 0.05 else " "
                print(f"     {label:32s} median {r['median']:+.4f}  "
                      f"p = {r['p']:.4f}  Holm = {padj:.4f}{star}  n = {r['n']}")
                records.append(dict(frame=frame, metric=col, arm=PROPOSED,
                                    baseline=early, n=r["n"],
                                    arm_value=r["arm_value"],
                                    baseline_value=r["ref_value"],
                                    median_diff=r["median"], mean_diff=r["mean"],
                                    p_raw=r["p"], p_holm=padj))

        # Alignment probes, separately and uncorrected. Anything using an oracle
        # alignment is an upper bound obtained with an annotation unavailable at
        # inference, and is never a performance claim.
        probes = [a for a in arms if is_probe(a)]
        if probes:
            print(f"\n  -- alignment probes (development fold only; oracle arms "
                  f"are upper bounds, not results) --")
            for a in probes:
                r = compare(dice_pat, frame, a, baseline, "dice")
                if r:
                    kind = "UPPER BOUND" if a.startswith("oracle") else "attainable"
                    print(f"     {a:22s} {r['arm_value']:8.4f}   "
                          f"median {r['median']:+.4f}  p = {r['p']:.4f}  "
                          f"n = {r['n']:2d}   [{kind}]")
                    records.append(dict(frame=frame, metric="dice", arm=a,
                                        baseline=baseline, n=r["n"],
                                        arm_value=r["arm_value"],
                                        baseline_value=r["ref_value"],
                                        median_diff=r["median"],
                                        mean_diff=r["mean"],
                                        p_raw=r["p"], p_holm=float("nan")))

        cov = coverage_table(bnd_pat, frame,
                             [a for a in arms if a != baseline and not is_probe(a)],
                             baseline)
        if not cov.empty:
            print(f"\n  -- margin adequacy: whole lesion inside the prediction "
                  f"dilated by k px --")
            piv = cov.pivot_table(index="arm", columns="k", values="arm_pct")
            piv.loc[baseline] = [100 * bnd_pat[(bnd_pat.ref_modality == frame)
                                               & (bnd_pat.config == baseline)][f"cov{k}"].mean()
                                 for k in MARGINS]
            print(piv.round(1).to_string())
            sig = cov[cov.p < 0.05]
            if len(sig):
                print("     significant at k (uncorrected):")
                for r in sig.itertuples():
                    print(f"       {r.arm:22s} k={r.k:<3d} {r.delta_pp:+.1f}pp  "
                          f"p = {r.p:.4f}")

    out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n{'=' * 78}\n{len(out)} comparisons -> {args.out}")
    print("Every p value above is on patient-level paired differences. Cohort "
          "means are\nprinted for description only and are not the basis of any "
          "test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
