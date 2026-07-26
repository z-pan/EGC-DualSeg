# -*- coding: utf-8 -*-
"""Package the raw gastroscopy archive into a de-identified, model-ready dataset.

Run once, locally. The output is the only artefact that leaves this machine.

What it does
------------
1. Selects the modelling cohort (77 patients: 81 folders minus 2 duplicate records
   and 2 oesophageal lesions).
2. Pairs WLI and NBI at the same working distance -> 150 pairs, 300 images.
3. Crops every frame to the valid endoscopic field. This removes the black border
   AND the burnt-in patient-information panel, so the package is de-identified by
   construction.
4. Letterboxes to 384 x 384 without stretching (the archive has 14 resolution
   combinations in two aspect-ratio families).
5. Writes images, masks and a manifest that already carries the per-pair
   correspondence statistics, so downstream analysis never needs to re-join.

Output
------
data/packaged/egc_dualseg_384.npz   images (M,384,384,3) uint8, masks (M,384,384) uint8
data/packaged/manifest.csv          M rows, one per image

Usage
-----
    python scripts/package_dataset.py
    python scripts/package_dataset.py --size 384 --out data/packaged
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

# --- the raw archive and its own tooling ------------------------------------
# The archive ships find_case_dirs/scan_case, which already handle the two
# inconsistent labelling conventions used by the two annotation batches.
# This script is a one-time bridge, so it reuses them rather than re-deriving.
RAW_ROOT = r"C:\Users\zpanp\projects\datasets\gastric_cancer"

# Modelling cohort exclusions, fixed by the design document (v2.3 section 3.5).
DUPLICATE_RECORDS = {"\u75c5\u4f8b76", "\u75c5\u4f8b51"}      # keep 75 and 50
OESOPHAGEAL = {"\u75c5\u4f8b41", "\u75c5\u4f8b47"}

MASK_THRESHOLD = 127
FIELD_BRIGHTNESS = 25       # pixel value above which a pixel counts as "lit"
FIELD_RUN_FRACTION = 0.5    # row/column must be this lit to belong to the field


# ---------------------------------------------------------------------------
def endoscopic_field(path: str) -> tuple[int, int, int, int]:
    """Bounding box of the valid endoscopic field.

    For each row and column, take the fraction of pixels above a low intensity
    threshold and keep the longest contiguous run above FIELD_RUN_FRACTION. This
    drops both the dark border and the sparse bright text of the information
    panel, which is why it also de-identifies the frame.
    """
    grey = np.asarray(Image.open(path).convert("L"))
    lit = grey > FIELD_BRIGHTNESS

    def longest_run(profile: np.ndarray) -> tuple[int, int]:
        idx = np.where(profile > FIELD_RUN_FRACTION)[0]
        if not len(idx):
            return 0, len(profile)
        segments = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        best = max(segments, key=len)
        return int(best[0]), int(best[-1]) + 1

    x0, x1 = longest_run(lit.mean(0))
    y0, y1 = longest_run(lit.mean(1))
    if (x1 - x0) < 0.3 * grey.shape[1] or (y1 - y0) < 0.3 * grey.shape[0]:
        return 0, 0, grey.shape[1], grey.shape[0]      # detection failed, keep all
    return x0, y0, x1, y1


def letterbox(arr: np.ndarray, size: int, nearest: bool = False) -> tuple[np.ndarray, dict]:
    """Resize preserving aspect ratio and pad to a square canvas.

    Returns the canvas and the geometry needed to map predictions back to a
    fraction of the endoscopic field (used by the pathology-extent analysis).
    """
    h, w = arr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resample = Image.NEAREST if nearest else Image.LANCZOS
    small = np.asarray(Image.fromarray(arr).resize((new_w, new_h), resample))
    if small.ndim == 2:
        canvas = np.zeros((size, size), small.dtype)
    else:
        canvas = np.zeros((size, size, small.shape[2]), small.dtype)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = small
    return canvas, dict(content_w=new_w, content_h=new_h, pad_x=pad_x, pad_y=pad_y)


def read_csv_dicts(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def mask_path_for(masks_root: str, batch: str, case: str, json_path: str) -> str:
    """Masks are named after the JSON stem, NOT the image stem.

    Batch 2 stores '016.jpg' alongside '016<modality>.json', and the mask is
    '016<modality>.png'. Resolving by image stem silently drops every batch-2
    case; this has already cost one debugging session.
    """
    stem = os.path.splitext(os.path.basename(json_path))[0]
    return os.path.join(masks_root, batch, case, stem + ".png")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=RAW_ROOT)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--out", default=os.path.join("data", "packaged"))
    args = ap.parse_args()

    raw_root = args.raw_root
    masks_root = os.path.join(raw_root, "_sam_output", "masks")
    sys.path.insert(0, raw_root)
    try:
        import bbox_sam_tools as T                                  # noqa: N811
    except ImportError:
        print(f"ERROR: bbox_sam_tools.py not importable from {raw_root}", file=sys.stderr)
        return 1

    labels = {r["case"]: r for r in
              read_csv_dicts(os.path.join(raw_root, "pathology_labels_clean.csv"))}
    corr = {(r["case"], r["scale"]): r for r in
            read_csv_dicts(os.path.join(raw_root, "_sam_output", "fov_consistency.csv"))}

    excluded = DUPLICATE_RECORDS | OESOPHAGEAL
    images, masks, rows = [], [], []
    skipped: list[str] = []

    for batch, case, case_dir in T.find_case_dirs(raw_root):
        if case in excluded or case not in labels:
            continue
        lab = labels[case]

        slots: dict[tuple[str, str], tuple[str, str]] = {}
        for json_path, img_path, modality, scale, boxes in T.scan_case(case_dir)["items"]:
            if img_path and boxes:
                slots.setdefault((modality, scale), (json_path, img_path))

        for scale in ("near", "far"):
            if ("WLI", scale) not in slots or ("NBI", scale) not in slots:
                continue
            pair_ok, staged = True, []
            for modality in ("WLI", "NBI"):
                json_path, img_path = slots[(modality, scale)]
                mp = mask_path_for(masks_root, batch, case, json_path)
                if not os.path.isfile(mp):
                    skipped.append(f"{case}/{scale}/{modality}: mask missing")
                    pair_ok = False
                    break
                x0, y0, x1, y1 = endoscopic_field(img_path)
                img = np.asarray(Image.open(img_path).convert("RGB"))[y0:y1, x0:x1]
                msk = (np.asarray(Image.open(mp))[y0:y1, x0:x1] > MASK_THRESHOLD)
                if msk.sum() == 0:
                    skipped.append(f"{case}/{scale}/{modality}: empty mask after crop")
                    pair_ok = False
                    break
                img_c, geom = letterbox(img, args.size)
                msk_c, _ = letterbox(msk.astype(np.uint8) * 255, args.size, nearest=True)
                staged.append((modality, img_c, msk_c, geom, x1 - x0, y1 - y0))
            if not pair_ok:
                continue

            c = corr.get((case, scale), {})
            for modality, img_c, msk_c, geom, fov_w, fov_h in staged:
                rows.append(dict(
                    idx=len(images), case=case, scale=scale, modality=modality,
                    fov_w=fov_w, fov_h=fov_h, **geom,
                    grade_bin=lab["grade_bin"], use_grade=lab["use_grade"],
                    macro=lab["macro"], use_macro=lab["use_macro"],
                    path_mm=lab["size_mm"], n_lesions=lab["n_lesions"], sites=lab["sites"],
                    naive_iou=c.get("naive_iou", ""), cdist=c.get("cdist", ""),
                    log2_ratio=c.get("log2_ratio", ""),
                    mi_same=c.get("mi_same", ""), mi_null=c.get("mi_null", ""),
                ))
                images.append(img_c)
                masks.append(msk_c)

    if not images:
        print("ERROR: nothing packaged", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, f"egc_dualseg_{args.size}.npz")
    np.savez_compressed(npz_path,
                        images=np.stack(images), masks=np.stack(masks))

    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cases = sorted({r["case"] for r in rows})
    pairs = {(r["case"], r["scale"]) for r in rows}
    near = sum(1 for _, s in pairs if s == "near")
    graded = len({r["case"] for r in rows if r["use_grade"] == "1"})
    size_mb = os.path.getsize(npz_path) / 1e6

    print(f"patients          {len(cases)}")
    print(f"pairs             {len(pairs)}  (near {near}, far {len(pairs) - near})")
    print(f"images            {len(images)}")
    print(f"gradeable cases   {graded}")
    print(f"package           {npz_path}  ({size_mb:.0f} MB)")
    print(f"manifest          {manifest_path}")
    if skipped:
        print(f"skipped           {len(skipped)}")
        for s in skipped:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
