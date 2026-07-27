# -*- coding: utf-8 -*-
"""Write predicted masks for exemplar images, so qualitative panels can be
composed later without a GPU.

    python scripts/dump_exemplars.py --ckpt-dir /content/drive/MyDrive/EGC-DualSeg/checkpoints

Training saved per-image metrics and nothing else. A qualitative figure needs
the masks themselves, and after the compute subscription lapses they cannot be
regenerated — so this dumps a deliberate *superset* of what any one figure will
use. The masks are tiny (384 x 384, 1-bit PNG); the whole dump is a couple of
hundred kilobytes and belongs in the repository.

Two things this script exists to get right
------------------------------------------
**Every mask comes from the fold that held that patient out.** Each patient sits
in exactly one fold, so the checkpoint is chosen per patient. Rendering a lesion
with a model that trained on it would be a training-set picture presented as a
result, and nothing downstream would reveal it.

**Selection is by rule, stated up front, not by eye.** Panels chosen by
appearance are cherry-picking and read as such. Two rules are applied and each
selected image records which one picked it, so the figure legend can say so:

  span      per frame, the images at the 5/10/25/50/75/90/95th percentile of
            the proposed model's Dice — the honest range, weak cases included.
  contrast  per frame, the images where naive early fusion loses the most
            against the proposed model. This is the claim the figure is making,
            so the cases carrying it are shown rather than described.

Ranking uses the seed-averaged Dice, matching how every other number is read.
The dumped mask is one seed, so its own Dice is recorded alongside — a caption
must quote the number belonging to the image it sits under.

Output
    results/exemplars/index.csv
    results/exemplars/pred_{config}_{case}_{scale}_ref{FRAME}_seed{s}.png
Input images and ground-truth masks are NOT dumped: they are already in the
packaged .npz, which stays local, so the figure script reads them from there.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import EGCPairDataset, collate      # noqa: E402
from src.data.transforms import PairAugment               # noqa: E402
from src.stage2.features import build_frozen_model        # noqa: E402

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]
FRAMES = {"WLI": dict(single="wli_only", early="early_fusion_wli"),
          "NBI": dict(single="nbi_only", early="early_fusion_nbi")}
FUSION = "ours"
FIELDS = ["case", "scale", "ref_modality", "config", "fold", "seed", "rule",
          "dice_this_seed", "dice_seed_mean", "path"]


def load_predictions(results_dir: str):
    """Seed-averaged Dice per image, as a dict keyed by (case, scale, frame, config)."""
    paths = [f for f in os.listdir(results_dir) if f.startswith("predictions_")]
    if not paths:
        raise SystemExit(f"no prediction CSVs in {results_dir}; nothing to rank")
    acc: dict[tuple, list] = {}
    for name in paths:
        with open(os.path.join(results_dir, name), encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                key = (r["case"], r["scale"], r["ref_modality"], r["config"])
                acc.setdefault(key, []).append(float(r["dice"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def select(dice: dict, n_contrast: int) -> dict[tuple, set[str]]:
    """(case, scale, frame) -> set of rules that selected it."""
    chosen: dict[tuple, set[str]] = {}

    def mark(key, rule):
        chosen.setdefault(key, set()).add(rule)

    for frame, spec in FRAMES.items():
        images = sorted({(c, s) for (c, s, f, cfg) in dice
                         if f == frame and cfg == FUSION})
        if not images:
            continue

        ranked = sorted(images, key=lambda im: dice[(im[0], im[1], frame, FUSION)])
        for p in PERCENTILES:
            idx = int(round(p / 100 * (len(ranked) - 1)))
            mark((*ranked[idx], frame), f"span-p{p}")

        deltas = []
        for case, scale in images:
            early = dice.get((case, scale, frame, spec["early"]))
            ours = dice.get((case, scale, frame, FUSION))
            if early is not None and ours is not None:
                deltas.append((ours - early, case, scale))
        deltas.sort(reverse=True)
        for _, case, scale in deltas[:n_contrast]:
            mark((case, scale, frame), "contrast-early")

    return chosen


@torch.no_grad()
def dump_for(seg_name: str, seg_raw: dict, frame: str, fold: int, seed: int,
             targets: set[tuple], ckpt_dir: str, out_dir: str,
             device: torch.device, num_workers: int) -> list[dict]:
    tag = f"{seg_name}_fold{fold}_seed{seed}"
    model = build_frozen_model(
        os.path.join(ckpt_dir, f"{tag}.pt"), seg_raw["mode"], device,
        use_role_embedding=seg_raw.get("use_role_embedding", True),
        aux_levels=tuple(seg_raw.get("aux_levels", (3, 4))))

    dataset = EGCPairDataset(seg_raw["npz"], seg_raw["manifest"], seg_raw["folds"],
                             fold, "val", mode=seg_raw["mode"], reference=frame,
                             transform=PairAugment(train=False))
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate,
                        num_workers=num_workers)

    rows = []
    for batch in loader:
        ref = batch["ref"].to(device)
        aux = batch["aux"].to(device) if "aux" in batch else None
        out = model(ref, aux, batch["ref_id"].to(device))
        pred = (torch.sigmoid(out["logits"]) > 0.5).float().cpu()
        gt = batch["mask"]
        for i, (case, scale) in enumerate(zip(batch["case"], batch["scale"])):
            if (case, scale) not in targets:
                continue
            p, g = pred[i, 0], gt[i, 0]
            inter = float((p * g).sum())
            dice = (2 * inter + 1e-7) / (float(p.sum()) + float(g.sum()) + 1e-7)
            name = f"pred_{seg_name}_{case}_{scale}_ref{frame}_seed{seed}.png"
            Image.fromarray((p.numpy() * 255).astype(np.uint8)).convert("1").save(
                os.path.join(out_dir, name), optimize=True)
            rows.append(dict(case=case, scale=scale, ref_modality=frame,
                             config=seg_name, fold=fold, seed=seed,
                             dice_this_seed=round(dice, 6), path=name))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["configs/wli_only.yaml", "configs/nbi_only.yaml",
                             "configs/early_fusion_wli.yaml",
                             "configs/early_fusion_nbi.yaml", "configs/ours.yaml"])
    ap.add_argument("--seed", type=int, default=0,
                    help="which seed's masks to render; ranking always uses all seeds")
    ap.add_argument("--n-contrast", type=int, default=4,
                    help="images per frame where early fusion loses the most")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--folds", default="configs/folds.csv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results, "exemplars")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dice = load_predictions(args.results)
    chosen = select(dice, args.n_contrast)
    print(f"{len(chosen)} exemplar images selected by rule "
          f"({len(PERCENTILES)} span + {args.n_contrast} contrast per frame, "
          "minus overlaps)")

    with open(args.folds, encoding="utf-8-sig") as fh:
        folds = {r["case"]: int(r["fold"]) for r in csv.DictReader(fh)}

    rows: list[dict] = []
    for cfg_path in args.configs:
        with open(cfg_path, encoding="utf-8") as fh:
            seg_raw = yaml.safe_load(fh) or {}
        seg_name = seg_raw["name"]
        reference = seg_raw.get("reference", "WLI")
        directions = tuple(FRAMES) if reference == "random" else (reference,)

        for frame in directions:
            # Group by fold so each checkpoint is loaded once.
            by_fold: dict[int, set[tuple]] = {}
            for (case, scale, f), rules in chosen.items():
                if f != frame:
                    continue
                by_fold.setdefault(folds[case], set()).add((case, scale))
            for fold, targets in sorted(by_fold.items()):
                got = dump_for(seg_name, seg_raw, frame, fold, args.seed, targets,
                               args.ckpt_dir, out_dir, device, args.num_workers)
                for r in got:
                    key = (r["case"], r["scale"], frame)
                    r["rule"] = "|".join(sorted(chosen[key]))
                    r["dice_seed_mean"] = round(
                        dice.get((r["case"], r["scale"], frame, seg_name),
                                 float("nan")), 6)
                rows.extend(got)
                print(f"  {seg_name:18s} ref {frame}  fold {fold}: "
                      f"{len(got)} masks")

    index = os.path.join(out_dir, "index.csv")
    with open(index, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["ref_modality"], r["config"],
                                             r["case"], r["scale"])):
            writer.writerow({k: r.get(k, "") for k in FIELDS})

    total_kb = sum(os.path.getsize(os.path.join(out_dir, f))
                   for f in os.listdir(out_dir)) / 1024
    print(f"\nwrote {len(rows)} masks + index.csv to {out_dir}  ({total_kb:.0f} KB)")
    print("commit this directory: it cannot be regenerated without a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
