# -*- coding: utf-8 -*-
"""Score the boundary, not the blob — the axis Dice cannot see.

    python scripts/boundary_metrics.py --configs configs/wli_only.yaml configs/nbi_only.yaml configs/ours.yaml
    python scripts/boundary_metrics.py --configs configs/pred_skip.yaml --folds 4

Why this exists. Every dual-versus-single comparison in this project has been
decided on Dice, and the lesions occupy 15-18% of the frame. At that size Dice
is dominated by gross overlap: moving the contour by a few pixels along the
whole perimeter changes it by about 0.005, which is the noise floor. Meanwhile
the clinical claim for the enhanced modality is specifically about the
**demarcation line** — where the lesion ends. So the effect that was looked for
lives almost entirely in a quantity the chosen metric is nearly blind to. The
v2 plan deleted the boundary axis; on the present evidence that was premature.

Four metrics per image, all against the same reference-frame mask the Dice used,
so nothing about the comparison changes except what is being measured:

    dice        recomputed, as a check that this pipeline reproduces
                results/predictions_*.csv exactly
    hd95        symmetric 95th-percentile Hausdorff distance, in pixels of the
                384 canvas. Sensitive to the worst part of the contour.
    bf{t}       boundary F-score at tolerance t px: the harmonic mean of
                "predicted contour within t of the truth" and "truth within t of
                the predicted contour"
    nsd{t}      normalised surface Dice at tolerance t px: the share of BOTH
                contours that lies within t of the other

hd95 grows with lesion size and so inherits the bias that already invalidated
`naive_iou` for correlation work (r = 0.575) and skews Dice itself (r = 0.562).
bf and nsd are ratios of contour length and are far less exposed; the read-out
reports the size correlation for each so the trap is visible rather than latent.

Writes results/boundary_{config}_fold{f}_seed{s}.csv, one row per image, with
the same key columns as predictions_*.csv so the two can be joined. No claim is
made here — scripts/summarise_boundary.py does the paired comparison.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine.trainer import RunConfig, build_loaders, build_model  # noqa: E402

TOLERANCES = (1, 2, 3, 5)
CLINICAL = ["residual_px", "residual_p95_px", "residual_area_frac",
            "over_depth_p95_px", "over_area_ratio", "fully_covered"]
FIELDS = (["case", "scale", "ref_modality", "config", "fold", "seed",
           "dice", "hd95", "gt_boundary_px", "pred_boundary_px",
           "gt_area_frac", "pred_area_frac", "content_w", "degenerate"]
          + [f"bf{t}" for t in TOLERANCES] + [f"nsd{t}" for t in TOLERANCES]
          + CLINICAL)

# 4-connectivity: the boundary is the ring of foreground pixels that touch
# background edge-on. An 8-connected erosion would thin diagonal contours and
# quietly shorten the denominators of bf and nsd.
_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def contour(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~binary_erosion(mask, structure=_STRUCT, border_value=1)


def boundary_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Distance-based agreement between two contours. All units are pixels."""
    cp, cg = contour(pred), contour(gt)
    n_p, n_g = int(cp.sum()), int(cg.sum())

    out = {"gt_boundary_px": n_g, "pred_boundary_px": n_p, "degenerate": 0}
    if n_p == 0 or n_g == 0:
        # An empty prediction (or, impossibly, an empty mask) has no contour to
        # score. Recording it as a failed image rather than dropping it keeps
        # the arms comparable on exactly the same set of images.
        out["hd95"] = float("nan")
        out["degenerate"] = 1
        for t in TOLERANCES:
            out[f"bf{t}"] = 0.0
            out[f"nsd{t}"] = 0.0
        return out

    # Distance from every pixel to the nearest contour pixel of the other mask.
    d_to_gt = distance_transform_edt(~cg)
    d_to_pred = distance_transform_edt(~cp)
    dp = d_to_gt[cp]          # predicted contour -> truth
    dg = d_to_pred[cg]        # truth -> predicted contour

    out["hd95"] = float(max(np.percentile(dp, 95), np.percentile(dg, 95)))
    for t in TOLERANCES:
        precision = float((dp <= t).mean())
        recall = float((dg <= t).mean())
        out[f"bf{t}"] = (2 * precision * recall / (precision + recall)
                         if precision + recall > 0 else 0.0)
        out[f"nsd{t}"] = float(((dp <= t).sum() + (dg <= t).sum()) / (n_p + n_g))
    return out


def clinical_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Split the error into the two halves the clinic does not treat alike.

    Dice, boundary-F and hd95 are all symmetric: a pixel of lesion missed and a
    pixel of normal mucosa taken cost the same. In endoscopic submucosal
    dissection they do not. Missed lesion is residual disease and a positive
    margin, which means further treatment; extra margin is healthy mucosa
    removed, which is the accepted price of the procedure — the endoscopist
    deliberately marks several millimetres outside the lesion before cutting.

    So the useful question is not "how well do the two contours agree" but
    "would resecting along this prediction, plus the usual safety margin, have
    taken the whole lesion" — a yes or no per image, and the margin that would
    have been required is the quantity behind it.

        residual_px         the largest distance from an uncovered lesion pixel
                            to the predicted mask: exactly the margin that would
                            have been needed for complete coverage, so
                            `residual_px <= k` IS adequacy at a k-pixel margin
        residual_p95_px     the same at the 95th percentile, for when one stray
                            component should not speak for the image
        residual_area_frac  missed lesion as a share of the lesion
        over_depth_p95_px   how far beyond the lesion the over-call reaches
        over_area_ratio     normal tissue taken, as a multiple of lesion area
        fully_covered       1 when nothing is missed at all

    The ground truth is still the SAM-derived mask, so this does not escape the
    annotation — but coverage is far less exposed to it than a boundary metric
    is, because it asks whether a region is enclosed rather than whether two
    contours have the same shape.
    """
    out = {k: float("nan") for k in CLINICAL}
    if not gt.any() or not pred.any():
        out["fully_covered"] = 0.0
        return out

    fn, fp = gt & ~pred, pred & ~gt
    d_to_pred = distance_transform_edt(~pred)
    d_to_gt = distance_transform_edt(~gt)

    out["residual_px"] = float(d_to_pred[fn].max()) if fn.any() else 0.0
    out["residual_p95_px"] = float(np.percentile(d_to_pred[fn], 95)) if fn.any() else 0.0
    out["residual_area_frac"] = float(fn.sum() / gt.sum())
    out["over_depth_p95_px"] = float(np.percentile(d_to_gt[fp], 95)) if fp.any() else 0.0
    out["over_area_ratio"] = float(fp.sum() / gt.sum())
    out["fully_covered"] = float(not fn.any())
    return out


@torch.no_grad()
def score_run(cfg: RunConfig, device: torch.device) -> list[dict]:
    tag = f"{cfg.name}_fold{cfg.fold}_seed{cfg.seed}"
    ckpt = os.path.join(cfg.ckpt_dir, f"{tag}.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"missing checkpoint {ckpt}\n"
            "  boundary metrics are recomputed from the weights, not from the "
            "prediction CSVs, because the CSVs never stored the masks.")

    _, val_loader = build_loaders(cfg)
    model = build_model(cfg, device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()

    rows = []
    for batch in val_loader:
        ref = batch["ref"].to(device)
        aux = batch["aux"].to(device) if "aux" in batch else None
        logits = model(ref, aux, batch["ref_id"].to(device))["logits"]
        pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()[:, 0]
        gt = batch["mask"].numpy()[:, 0] > 0.5
        for i in range(pred.shape[0]):
            p, g = pred[i].astype(bool), gt[i].astype(bool)
            tp = float((p & g).sum())
            eps = 1e-7
            row = dict(case=batch["case"][i], scale=batch["scale"][i],
                       ref_modality=batch["ref_modality"][i], config=cfg.name,
                       fold=cfg.fold, seed=cfg.seed,
                       dice=(2 * tp + eps) / (p.sum() + g.sum() + eps),
                       gt_area_frac=float(g.mean()), pred_area_frac=float(p.mean()),
                       content_w=int(batch["content_w"][i]))
            row.update(boundary_scores(p, g))
            row.update(clinical_scores(p, g))
            rows.append(row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    # Arms that share a yaml and differ only by a train-time override (the MMD
    # weight sweep) have no config file of their own. Without this they would all
    # resolve to the yaml's own name, collide on the output path, and be skipped
    # as already done — silently producing nothing.
    ap.add_argument("--name", default=None,
                    help="override the config's name; one config path only")
    args = ap.parse_args()
    if args.name is not None and len(args.configs) != 1:
        ap.error("--name applies to a single --configs entry")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in args.configs:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        folds = args.folds if args.folds is not None else raw.pop("folds_to_run", [0, 1, 2, 3, 4])
        seeds = args.seeds if args.seeds is not None else raw.pop("seeds", [0, 1, 2])
        raw.pop("folds_to_run", None)
        raw.pop("seeds", None)
        if args.ckpt_dir:
            raw["ckpt_dir"] = args.ckpt_dir
        if args.name:
            raw["name"] = args.name
        out_dir = args.out_dir or raw.get("out_dir", "results")
        raw["num_workers"] = args.num_workers

        for fold in folds:
            for seed in seeds:
                cfg = RunConfig(fold=fold, seed=seed, **raw)
                dest = os.path.join(out_dir,
                                    f"boundary_{cfg.name}_fold{fold}_seed{seed}.csv")
                if os.path.exists(dest) and not args.force:
                    print(f"  {os.path.basename(dest)} exists, skipping")
                    continue
                rows = score_run(cfg, device)
                os.makedirs(out_dir, exist_ok=True)
                with open(dest, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                bad = sum(r["degenerate"] for r in rows)
                finite = [r["hd95"] for r in rows if r["hd95"] == r["hd95"]]
                hd = f"{np.median(finite):.1f}px" if finite else "n/a"
                print(f"  {cfg.name} fold{fold} seed{seed}: {len(rows)} images, "
                      f"median hd95 {hd}, median bf3 "
                      f"{np.median([r['bf3'] for r in rows]):.3f}"
                      + (f", {bad} degenerate" if bad else ""))
                if bad == len(rows):
                    print("    every prediction is empty — check the checkpoint, "
                          "these rows carry no information")
    print("\nread it with: python scripts/summarise_boundary.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
