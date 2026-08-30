# -*- coding: utf-8 -*-
"""Dump the bottleneck-concatenation masks that Fig. 5 is missing.

    python scripts/dump_fig5_midfusion.py --ckpt-dir /path/to/checkpoints

The original exemplar dump ran before bottleneck concatenation was part of the
figure, so `results/exemplars/` has no mid_fusion masks and Fig. 5 shows three
configurations instead of four. Re-running the full dump would need a checkpoint
for every fold the selection touches. This one renders only the six images Fig. 5
actually draws, which needs three: folds 0, 3 and 4 at seed 0.

Nothing is re-selected. The six images are read back out of the existing
`index.csv` by their recorded span rule, so the figure keeps the selection it was
built on and this script cannot quietly change which cases are shown.

Requires torch and torchvision on CPU; no GPU. Roughly a minute per fold.
Download from the training Drive:
    mid_fusion_fold0_seed0.pt   mid_fusion_fold3_seed0.pt   mid_fusion_fold4_seed0.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
from dump_exemplars import dump_for, load_predictions, FIELDS   # noqa: E402

CONFIG = "configs/mid_fusion.yaml"
SEG_NAME = "mid_fusion"
ROWS_WANTED = ["span-p10", "span-p50", "span-p90"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(args.results, "exemplars")
    index_path = os.path.join(out_dir, "index.csv")
    with open(index_path, encoding="utf-8-sig") as fh:
        index = list(csv.DictReader(fh))

    # The six tiles Fig. 5 draws, taken from the rule already recorded.
    wanted = []
    for frame in ("WLI", "NBI"):
        for rule in ROWS_WANTED:
            hit = [r for r in index if r["ref_modality"] == frame and r["rule"] == rule]
            if not hit:
                print("  missing from index.csv: %s %s" % (frame, rule)); continue
            r = hit[0]
            if any(x["config"] == SEG_NAME and x["case"] == r["case"]
                   and x["scale"] == r["scale"] and x["ref_modality"] == frame
                   for x in index):
                print("  already dumped: %s %s %s" % (frame, r["case"], r["scale"]))
                continue
            wanted.append((frame, r["case"], r["scale"], int(r["fold"]),
                           int(r["seed"]), rule, r["dice_seed_mean"]))
    if not wanted:
        print("nothing to do"); return 0

    need = sorted({(w[0], w[3], w[4]) for w in wanted})
    missing = [t for t in need
               if not os.path.isfile(os.path.join(
                   args.ckpt_dir, "%s_fold%d_seed%d.pt" % (SEG_NAME, t[1], t[2])))]
    if missing:
        print("missing checkpoints in %s:" % args.ckpt_dir)
        for _, f, sd in sorted({(0, t[1], t[2]) for t in missing}):
            print("    %s_fold%d_seed%d.pt" % (SEG_NAME, f, sd))
        return 1

    with open(os.path.join(HERE, CONFIG), encoding="utf-8") as fh:
        seg_raw = yaml.safe_load(fh) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s | %d tile(s) over %d checkpoint(s)"
          % (device.type, len(wanted), len(need)))

    # Fig. 5 prints the seed-averaged Dice in every other column, so this one
    # must carry the same quantity rather than the single rendered seed.
    seed_mean = load_predictions(args.results)

    new_rows = []
    for frame, fold, seed in need:
        targets = {(w[1], w[2]) for w in wanted
                   if w[0] == frame and w[3] == fold and w[4] == seed}
        rows = dump_for(SEG_NAME, seg_raw, frame, fold, seed, targets,
                        args.ckpt_dir, out_dir, device, args.num_workers)
        for r in rows:
            src = next(w for w in wanted if w[1] == r["case"] and w[2] == r["scale"]
                       and w[0] == r["ref_modality"])
            r["rule"] = src[5]
            m = seed_mean.get((r["case"], r["scale"], r["ref_modality"], SEG_NAME))
            r["dice_seed_mean"] = "" if m is None else round(float(m), 6)
            new_rows.append(r)
            print("  %s %s %s  seed-mean %s (this seed %.3f)"
                  % (frame, r["case"], r["scale"], r["dice_seed_mean"],
                     r["dice_this_seed"]))

    with open(index_path, "a", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        for r in new_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("appended %d row(s) to %s" % (len(new_rows), index_path))
    print("now re-run: python figures/fig_qualitative_span.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
