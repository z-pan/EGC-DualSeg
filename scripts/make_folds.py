# -*- coding: utf-8 -*-
"""Assign patients to cross-validation folds. Run once; the result is permanent.

Splitting is at the PATIENT level: every image of a patient — both working
distances and both modalities — lands in the same fold. Anything else leaks.

Folds are stratified so that each fold carries a comparable share of
    low grade / high grade or above / not gradeable
which keeps both the Stage-2 class balance (21 : 27 over 48 patients) and the
graded-versus-ungraded split even. Stage 2 reuses these folds unchanged, so no
patient can be in a Stage-1 training fold and a Stage-2 test fold.

The output is committed to git. Once written it must not be regenerated: every
config, every seed, every ablation and both stages share this one assignment.

Usage
-----
    python scripts/make_folds.py
    python scripts/make_folds.py --k 5 --seed 20260726 --force
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys


def read_csv_dicts(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def stratum_of(row: dict) -> str:
    """Grade class when the patient is gradeable, otherwise a separate stratum."""
    if row.get("use_grade") != "1":
        return "ungraded"
    return "low" if row.get("grade_bin") == "低级别" else "high+"


def assign(strata: dict[str, list[str]], k: int, seed: int) -> dict[str, int]:
    """Deterministic stratified round-robin.

    Within each stratum the cases are shuffled with a fixed seed and then dealt
    out one at a time. Dealing continues across strata from a shared cursor, so
    that strata smaller than k do not all pile into fold 0.
    """
    import random

    rng = random.Random(seed)
    folds: dict[str, int] = {}
    cursor = 0
    for name in sorted(strata):
        cases = sorted(strata[name])          # sort first so the shuffle is reproducible
        rng.shuffle(cases)
        for case in cases:
            folds[case] = cursor % k
            cursor += 1
    return folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join("data", "packaged", "manifest.csv"))
    ap.add_argument("--out", default=os.path.join("configs", "folds.csv"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing folds.csv (do not use casually)")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"ERROR: {args.out} already exists.\n"
              f"Folds are fixed once written — every result in the project assumes "
              f"this assignment. Pass --force only if you intend to invalidate "
              f"everything already trained.", file=sys.stderr)
        return 1

    if not os.path.exists(args.manifest):
        print(f"ERROR: {args.manifest} not found. Run scripts/package_dataset.py first.",
              file=sys.stderr)
        return 1

    rows = read_csv_dicts(args.manifest)
    per_case: dict[str, dict] = {}
    for r in rows:
        per_case.setdefault(r["case"], r)

    strata: dict[str, list[str]] = collections.defaultdict(list)
    for case, r in per_case.items():
        strata[stratum_of(r)].append(case)

    folds = assign(strata, args.k, args.seed)

    # The committed file carries the fold assignment only. The stratum is a
    # per-patient clinical attribute and is recomputed locally from the manifest;
    # it is printed in the report below but never written to a versioned file.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "fold"])
        for case in sorted(folds, key=lambda c: (folds[c], c)):
            w.writerow([case, folds[case]])

    # --- report ------------------------------------------------------------
    print(f"patients {len(folds)}   k = {args.k}   seed = {args.seed}")
    print(f"written  {args.out}\n")

    table: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for case, f in folds.items():
        table[f][stratum_of(per_case[case])] += 1
    names = sorted({s for c in table.values() for s in c})
    print("fold  " + "".join(f"{n:>11s}" for n in names) + f"{'total':>8s}")
    for f in sorted(table):
        counts = [table[f][n] for n in names]
        print(f"{f:<6d}" + "".join(f"{c:11d}" for c in counts) + f"{sum(counts):8d}")

    pair_folds: collections.Counter = collections.Counter()
    for r in rows:
        pair_folds[folds[r["case"]]] += 1
    print("\nimages per fold: " + ", ".join(f"{f}:{n}" for f, n in sorted(pair_folds.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
