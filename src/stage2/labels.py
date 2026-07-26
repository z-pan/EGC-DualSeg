# -*- coding: utf-8 -*-
"""Patient-level endpoints for Stage 2, read from the packaged manifest.

Both endpoints come from the ESD pathology reports, not from a model, which is
the whole reason Stage 2 exists: Stage 1 is scored against SAM masks that were
themselves produced from clinician boxes, so a dual-versus-single comparison
there is partly circular. These labels are not.

grade  -- the primary endpoint. Low-grade intraepithelial neoplasia versus
          anything worse (high-grade, or invasive adenocarcinoma). n = 48, 21:27.

macro  -- the positive control, NOT a claim. Macroscopic Paris type, depressed
          (0-IIc) versus not (0-IIa, 0-IIb). n = 45, 23:22. A depressed lesion
          looks different on white light; a pipeline that cannot separate these
          is broken, and a null result on `grade` would mean nothing.
          The single 0-IIb case is grouped with 0-IIa: the axis is presence of a
          depressed component, and flat is not depressed. With n=1 the choice
          cannot move the AUC materially, but it is a choice, so it is written
          down here rather than left implicit.
"""
from __future__ import annotations

import csv

TARGETS = ("grade", "macro")


def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_labels(manifest_path: str, target: str = "grade") -> dict[str, int]:
    """case -> 0/1, containing only the patients eligible for this endpoint."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    flag = "use_grade" if target == "grade" else "use_macro"
    field = "grade_bin" if target == "grade" else "macro"

    labels: dict[str, int] = {}
    for row in _read_csv(manifest_path):
        if row.get(flag, "0").strip() != "1":
            continue
        value = (row.get(field) or "").strip()
        if not value:
            continue
        if target == "grade":
            labels[row["case"]] = 0 if value == "低级别" else 1
        else:
            labels[row["case"]] = 1 if "IIc" in value else 0
    return labels


def describe(labels: dict[str, int]) -> str:
    n1 = sum(labels.values())
    return f"n = {len(labels)}  ({len(labels) - n1} negative : {n1} positive)"
