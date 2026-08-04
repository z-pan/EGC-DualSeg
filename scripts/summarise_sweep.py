# -*- coding: utf-8 -*-
"""At equal cost in normal tissue, does the dual arm miss less lesion?

    python scripts/summarise_sweep.py
    python scripts/summarise_sweep.py --results results

No GPU: reads the sweep_*.csv written by scripts/threshold_sweep.py.

The clinical read-out compares arms at a threshold of 0.5 and finds the dual
model missing 26% less lesion on the enhanced frame while taking no more normal
mucosa. The objection that has to be answered before that means anything is
that lowering the single-modality threshold would buy the same trade. Missed
lesion and over-taken tissue move in opposite directions as the threshold moves,
so any single point on that trade-off is worth very little on its own.

So this asks the operating-point-free question instead: **does the dual arm sit
outside the single arm's achievable curve, or on it?**

Two matched comparisons, both paired on images:

  matched on over-segmentation   find the single arm's threshold whose mean
                                 over_ratio equals the dual arm's at 0.5, then
                                 compare missed lesion there. "At the same cost
                                 in healthy tissue, who misses less?"
  matched on missed lesion       the mirror image. "To miss as little as the
                                 dual arm does, how much extra tissue must the
                                 single arm take?"

If the dual arm wins both, it is a better model. If it wins neither, it was a
luckier operating point and the clinical result has to be withdrawn. Winning one
and not the other means the curves cross, and the honest statement is that the
advantage exists only over part of the range.

Matching is on the cohort mean, so the matched threshold is reported alongside
the residual mismatch — a comparison at 0.171 against 0.174 is fair, one at
0.171 against 0.210 is not, and the reader should be able to see which it is.
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


def load(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "sweep_*.csv")))
    if not files:
        raise SystemExit(
            f"no sweep_*.csv in {results_dir}. Run scripts/threshold_sweep.py "
            "where the checkpoints live.")
    df = pd.concat((pd.read_csv(f, encoding="utf-8-sig") for f in files),
                   ignore_index=True)
    print(f"loaded {len(files)} files, {len(df)} rows, "
          f"{df.config.nunique()} configs, "
          f"{df.threshold.nunique()} thresholds")
    return df


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def matched(per_image: pd.DataFrame, dual: str, single: str,
            match_on: str, compare: str, dual_threshold: float = 0.5):
    """Compare `compare` where the two arms are level on `match_on`."""
    d = per_image[(per_image.config == dual)
                  & (np.isclose(per_image.threshold, dual_threshold))]
    if d.empty:
        return None
    d = d.set_index(KEY)
    target = d[match_on].mean()

    curve = (per_image[per_image.config == single]
             .groupby("threshold")[[match_on, compare]].mean())
    tau = float((curve[match_on] - target).abs().idxmin())

    s = per_image[(per_image.config == single)
                  & (np.isclose(per_image.threshold, tau))].set_index(KEY)
    shared = d.index.intersection(s.index)
    if len(shared) < 5:
        return None
    d, s = d.loc[shared], s.loc[shared]
    return dict(tau=tau, n=len(shared),
                matched_dual=float(d[match_on].mean()),
                matched_single=float(s[match_on].mean()),
                cmp_dual=float(d[compare].mean()),
                cmp_single=float(s[compare].mean()),
                delta=float((d[compare] - s[compare]).median()),
                p=wilcoxon_p(d[compare].to_numpy(), s[compare].to_numpy()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="the dual arm's operating point being defended")
    args = ap.parse_args()

    df = load(args.results)
    per_image = df.groupby(KEY + ["config", "threshold"], as_index=False)[
        ["missed_frac", "over_ratio", "dice"]].mean()

    for frame, single in FRAMES.items():
        sub = per_image[per_image.ref_modality == frame]
        if single not in set(sub.config):
            continue
        duals = sorted(c for c in sub.config.unique() if c != single)
        print(f"\n{'=' * 74}\n=== reference frame: {frame}   baseline {single} ===")

        curve = (sub[sub.config == single]
                 .groupby("threshold")[["missed_frac", "over_ratio", "dice"]].mean())
        print(f"\n  {single} trade-off curve (cohort means)")
        print(f"    {'tau':>6} {'missed':>8} {'over':>8} {'dice':>8}")
        for tau, r in curve.iterrows():
            mark = "  <- 0.5" if abs(tau - 0.5) < 1e-6 else ""
            print(f"    {tau:6.2f} {r.missed_frac:8.3f} {r.over_ratio:8.3f} "
                  f"{r.dice:8.3f}{mark}")

        # The one comparison that needs no matching at all: an arm whose Dice at
        # its own operating point exceeds what the baseline reaches at ANY
        # threshold cannot be explained by threshold placement. It is a weaker
        # statement than dominating the trade-off curve, and it is immune to the
        # objection that sinks the fixed-threshold comparisons.
        best_single = float(curve.dice.max())
        best_tau = float(curve.dice.idxmax())
        print(f"\n  best Dice the baseline reaches at any threshold: "
              f"{best_single:.4f} (tau = {best_tau:.2f})")
        for cfg in [single] + duals:
            c = (sub[sub.config == cfg].groupby("threshold").dice.mean())
            at_half = float(c.loc[c.index[np.isclose(c.index, args.threshold)][0]]) \
                if any(np.isclose(c.index, args.threshold)) else float("nan")
            verdict = ""
            if cfg != single:
                verdict = ("  <- above the baseline's ceiling"
                           if at_half > best_single else "  <- within reach of the baseline")
            print(f"    {cfg:22s} own Dice at {args.threshold} = {at_half:.4f}   "
                  f"best over sweep = {c.max():.4f}{verdict}")

        for dual in duals:
            print(f"\n  --- {dual} at tau = {args.threshold} vs {single} anywhere "
                  f"on its curve ---")

            a = matched(sub, dual, single, "over_ratio", "missed_frac", args.threshold)
            if a:
                verdict = ("dual misses less at equal cost" if a["delta"] < 0
                           else "no advantage once cost is matched")
                print(f"    matched on normal tissue taken "
                      f"({a['matched_dual']:.3f} vs {a['matched_single']:.3f} "
                      f"at tau = {a['tau']:.2f}):")
                print(f"      missed lesion {a['cmp_dual']:.3f} vs "
                      f"{a['cmp_single']:.3f}   paired median {a['delta']:+.4f}  "
                      f"p = {a['p']:.4f}  n = {a['n']}   [{verdict}]")

            b = matched(sub, dual, single, "missed_frac", "over_ratio", args.threshold)
            if b:
                verdict = ("dual takes less tissue at equal coverage" if b["delta"] < 0
                           else "no advantage once coverage is matched")
                print(f"    matched on missed lesion "
                      f"({b['matched_dual']:.3f} vs {b['matched_single']:.3f} "
                      f"at tau = {b['tau']:.2f}):")
                print(f"      normal tissue taken {b['cmp_dual']:.3f} vs "
                      f"{b['cmp_single']:.3f}   paired median {b['delta']:+.4f}  "
                      f"p = {b['p']:.4f}  n = {b['n']}   [{verdict}]")

            if a and b:
                wins = (a["delta"] < 0 and a["p"] < 0.05,
                        b["delta"] < 0 and b["p"] < 0.05)
                # When both matchings land on the same baseline threshold they are
                # one comparison read two ways, not two axes, and calling a split
                # verdict "the curves cross" would overstate it.
                same_tau = abs(a["tau"] - b["tau"]) < 1e-6
                if all(wins):
                    print("      => outside the baseline's curve on both axes: "
                          "a better model, not a better threshold")
                elif any(wins) and same_tau:
                    print(f"      => both matchings land on the baseline at tau = "
                          f"{a['tau']:.2f}, so this is ONE comparison, not two.\n"
                          "         The cohort means are level; what differs is the "
                          "paired median, i.e.\n         the typical image is better "
                          "and the tail is worse. For a resection\n         margin the "
                          "tail is the part that matters, so this is not a claim.")
                elif any(wins):
                    print("      => wins on one axis only: the curves cross, and the "
                          "advantage holds\n         over part of the range rather "
                          "than everywhere")
                else:
                    print("      => on the baseline's curve: this was an operating "
                          "point, not a model\n         improvement, and the "
                          "clinical claim has to be withdrawn")

    print(f"\n{'=' * 74}")
    print("Matching is on cohort means, so check the matched pair printed on each line\n"
          "before reading its p value: the comparison is only fair if those two\n"
          "numbers are close.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
