# -*- coding: utf-8 -*-
"""Does the auxiliary stream contribute anything? Measure it, don't infer it.

    python scripts/gate_ablation.py --ckpt-dir checkpoints_local

The gate report says |tanh(gate)| settles near 0.014 across all 15 runs, which
*suggests* the model switched the auxiliary stream off. But that is an inference
from a parameter value, and the obvious rebuttal is that a small gate on a
large-magnitude attended signal could still matter. This measures the thing
directly: same checkpoint, same images, forward passes that differ only in what
the auxiliary stream is allowed to contribute.

Conditions
----------
normal        the model exactly as trained
gate_zero     fusion.gate forced to 0 -- the attended branch is removed entirely
aux_shuffled  each frame keeps its reference image but receives ANOTHER patient's
              auxiliary frame, so the pairing is destroyed while the input
              statistics are untouched
aux_zero      the auxiliary frame is replaced by a black frame

What each comparison answers
----------------------------
normal vs gate_zero     does the fusion branch change the output at all?
normal vs aux_shuffled  **the strong test.** Does the *content* of the paired
                        auxiliary frame matter? A model that merely exploits
                        the auxiliary stream's global statistics, rather than
                        this patient's actual second view, is indistinguishable
                        from one that ignores it — and only this comparison can
                        tell them apart. A non-zero gate would survive the first
                        test and still fail this one.

Reported at four levels of sensitivity, because they can disagree and the
disagreement is the most informative part:
  mean |dz|       does the BOTTLENECK move? This is upstream of the decoder and
                  is the only quantity that isolates whether the fusion branch
                  computes anything at all.
  mean |dp|       does the probability field move, even sub-threshold?
  mask IoU        does the thresholded prediction move?
  Dice            does the metric move?

The bottleneck column exists because of a trap found while testing this script:
with randomly initialised weights the fusion branch shifted the bottleneck by
14% while the logits moved by 2e-6, purely because an untrained decoder
attenuates it. Reading only the output would have said "the fusion branch does
nothing" when it demonstrably did something. Two outcomes to keep apart:

  dz ~ 0 and dp ~ 0     the branch genuinely contributes nothing; the gate is
                        shut and the negative result is about the modality.
  dz > 0 but dp ~ 0     the branch computes, and the decoder learned to ignore
                        it. Same Dice, different claim -- it would mean the
                        fusion OPERATOR is what failed, and a better one might
                        not.

Output: results/gate_ablation.csv, one row per (image, condition).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import EGCPairDataset, collate      # noqa: E402
from src.data.transforms import PairAugment               # noqa: E402
from src.stage2.features import build_frozen_model        # noqa: E402

CONDITIONS = ["normal", "gate_zero", "aux_shuffled", "aux_zero"]
FRAMES = ("WLI", "NBI")
FIELDS = ["case", "scale", "ref_modality", "fold", "seed", "condition",
          "dice", "iou_vs_normal", "mean_abs_prob_diff", "pixel_agree_vs_normal",
          "mean_abs_embed_diff", "rel_embed_diff"]


def dice_of(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    inter = (pred * gt).sum(dims)
    return (2 * inter + 1e-7) / (pred.sum(dims) + gt.sum(dims) + 1e-7)


@torch.no_grad()
def run_fold(seg_raw: dict, frame: str, fold: int, seed: int, ckpt_dir: str,
             device: torch.device, num_workers: int) -> list[dict]:
    tag = f"{seg_raw['name']}_fold{fold}_seed{seed}"
    model = build_frozen_model(
        os.path.join(ckpt_dir, f"{tag}.pt"), seg_raw["mode"], device,
        use_role_embedding=seg_raw.get("use_role_embedding", True),
        aux_levels=tuple(seg_raw.get("aux_levels", (3, 4))))
    gate_original = model.fusion.gate.detach().clone()

    dataset = EGCPairDataset(seg_raw["npz"], seg_raw["manifest"], seg_raw["folds"],
                             fold, "val", mode=seg_raw["mode"], reference=frame,
                             transform=PairAugment(train=False))
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                        collate_fn=collate, num_workers=num_workers)
    batch = next(iter(loader))          # one fold fits comfortably in memory

    ref = batch["ref"].to(device)
    aux = batch["aux"].to(device)
    mask = batch["mask"].to(device)
    ref_id = batch["ref_id"].to(device)
    n = ref.shape[0]

    # Deterministic derangement: every frame gets someone else's auxiliary view.
    # roll by 1 has no fixed point, so no image accidentally keeps its own pair.
    aux_variants = {
        "normal": aux,
        "gate_zero": aux,
        "aux_shuffled": torch.roll(aux, shifts=1, dims=0),
        "aux_zero": torch.zeros_like(aux),
    }

    probs, dices, embeds = {}, {}, {}
    for cond in CONDITIONS:
        model.fusion.gate.data.copy_(
            torch.zeros_like(gate_original) if cond == "gate_zero" else gate_original)
        out = model(ref, aux_variants[cond], ref_id)
        p = torch.sigmoid(out["logits"].float())
        probs[cond] = p
        embeds[cond] = out["embedding"].float()
        dices[cond] = dice_of((p > 0.5).float(), mask)
    model.fusion.gate.data.copy_(gate_original)

    base_bin = (probs["normal"] > 0.5).float()
    base_scale = embeds["normal"].abs().mean((1, 2, 3)).clamp_min(1e-7)
    rows = []
    for cond in CONDITIONS:
        cond_bin = (probs[cond] > 0.5).float()
        inter = (cond_bin * base_bin).sum((1, 2, 3))
        union = ((cond_bin + base_bin) > 0).float().sum((1, 2, 3))
        iou = torch.where(union > 0, inter / union.clamp_min(1e-7),
                          torch.ones_like(union))
        dp = (probs[cond] - probs["normal"]).abs().mean((1, 2, 3))
        agree = (cond_bin == base_bin).float().mean((1, 2, 3))
        dz = (embeds[cond] - embeds["normal"]).abs().mean((1, 2, 3))
        for i in range(n):
            rows.append(dict(
                case=batch["case"][i], scale=batch["scale"][i],
                ref_modality=frame, fold=fold, seed=seed, condition=cond,
                dice=float(dices[cond][i]), iou_vs_normal=float(iou[i]),
                mean_abs_prob_diff=float(dp[i]),
                pixel_agree_vs_normal=float(agree[i]),
                mean_abs_embed_diff=float(dz[i]),
                rel_embed_diff=float(dz[i] / base_scale[i])))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def wilcoxon(a, b):
    d = a - b
    d = d[d != 0]
    if len(d) < 6:
        return float(np.median(a - b)), float("nan")
    try:
        from scipy.stats import wilcoxon as _w
        return float(np.median(a - b)), float(_w(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return float(np.median(a - b)), float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-config", default="configs/ours.yaml")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="results/gate_ablation.csv")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    with open(args.seg_config, encoding="utf-8") as fh:
        seg_raw = yaml.safe_load(fh) or {}
    if seg_raw["mode"] != "dual":
        raise SystemExit(f"{args.seg_config} has no fusion branch to ablate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for fold in args.folds:
        for seed in args.seeds:
            for frame in FRAMES:
                rows += run_fold(seg_raw, frame, fold, seed, args.ckpt_dir,
                                 device, args.num_workers)
        print(f"  fold {fold} done ({len(rows)} rows)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---- read-out ---------------------------------------------------------
    import pandas as pd
    df = pd.DataFrame(rows)
    key = ["case", "scale", "ref_modality", "fold", "seed"]
    print(f"\n{len(df)} rows | {df.condition.nunique()} conditions | "
          f"device {device.type}")

    for frame in FRAMES:
        sub = df[df.ref_modality == frame]
        wide = sub.pivot_table(index=key, columns="condition", values="dice")
        print(f"\n=== {frame} frame  (n = {len(wide)} image-runs) ===")
        for cond in CONDITIONS:
            block = sub[sub.condition == cond]
            line = (f"  {cond:14s} Dice mean {block.dice.mean():.4f}  "
                    f"median {block.dice.median():.4f}")
            if cond != "normal":
                delta, p = wilcoxon(wide[cond].to_numpy(), wide["normal"].to_numpy())
                line += (f"   vs normal: median {delta:+.5f}"
                         f"{'' if not np.isfinite(p) else f', p = {p:.3g}'}")
            print(line)
        for cond in CONDITIONS[1:]:
            block = sub[sub.condition == cond]
            print(f"    {cond:14s} bottleneck shift {block.rel_embed_diff.mean():.3%}"
                  f" | mean |dp| {block.mean_abs_prob_diff.mean():.2e}"
                  f" | mask IoU {block.iou_vs_normal.median():.4f}"
                  f" | pixels moved {100*(1-block.pixel_agree_vs_normal.median()):.2f}%")

    shuffled = df[df.condition == "aux_shuffled"]
    dz = shuffled.rel_embed_diff.mean()
    dp = shuffled.mean_abs_prob_diff.mean()
    print("\ninterpretation -- read the bottleneck column FIRST")
    print(f"  Giving each frame another patient's auxiliary view shifts the bottleneck")
    print(f"  by {dz:.3%} and the probability field by {dp:.2e}.")
    print()
    if dz < 0.005:
        print("  -> The fusion branch contributes essentially nothing. The gate is shut,")
        print("     and the negative result is a statement about the second modality.")
    elif dp < 1e-4:
        print("  -> The branch DOES compute on the paired frame, but the decoder ignores")
        print("     it. That is a different claim: the fusion OPERATOR failed, not")
        print("     necessarily the modality. A better operator is not excluded, and")
        print("     the paper must say so rather than claim the modality is uninformative.")
    else:
        print("  -> The paired frame changes the prediction while leaving Dice flat:")
        print("     the stream is read and used, but what it carries does not help.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
