# -*- coding: utf-8 -*-
"""What did the fusion gate learn?

    python scripts/gate_report.py --ckpt-dir checkpoints
    python scripts/gate_report.py --ckpt-dir /content/drive/MyDrive/EGC-DualSeg/checkpoints

Stage 1 found that registration-free fusion beats naive early fusion decisively
but beats the single-modality baselines by roughly nothing. Two very different
explanations fit that:

  (a) the model learned to *ignore* the auxiliary stream, which would be direct
      evidence that the second modality carries nothing this task can use;
  (b) the model uses the auxiliary stream and the information genuinely is not
      enough.

`CrossAttentionFusion.gate` separates them. It is initialised at 0, applied as
tanh(gate), and multiplies the entire attended contribution — so the residual
branch starts as an exact identity and the fusion has to earn its way in. If the
learned |tanh(gate)| stays near zero across folds and seeds, that is (a). If it
settles somewhere clearly non-zero, that is (b), and the next thing to try is
the global-context variant rather than more capacity at the same fusion point.

Reading nothing but the checkpoints, so it costs seconds and no GPU. The role
and scale embeddings come along because they are free and they say whether the
two fusion directions and the two auxiliary pyramid levels were distinguished
at all.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.stage2.features import load_checkpoint  # noqa: E402

NEAR_ZERO = 0.05      # |tanh(gate)| below this is "the stream was switched off"


def summarise(path: str) -> dict:
    state = load_checkpoint(path)
    model = state.get("model", state)
    if "fusion.gate" not in model:
        return {}

    gate = float(model["fusion.gate"].reshape(-1)[0])
    row = {"checkpoint": os.path.basename(path),
           "gate_raw": gate,
           "gate_tanh": math.tanh(gate),
           "val_dice": float(state.get("val_dice", float("nan"))),
           "epoch": int(state.get("epoch", -1))}

    role = model.get("fusion.role")
    if role is not None:                      # (n_modalities, dim), zero-initialised
        row["role_norm_wli"] = float(role[0].norm())
        row["role_norm_nbi"] = float(role[1].norm())
        row["role_separation"] = float((role[0] - role[1]).norm())
    scale = model.get("fusion.scale_embed")
    if scale is not None:                     # one row per auxiliary pyramid level
        for i in range(scale.shape[0]):
            row[f"scale_embed_norm_{i}"] = float(scale[i].norm())
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--pattern", default="ours_fold*_seed*.pt")
    ap.add_argument("--out", default="results/gate_report.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, args.pattern)),
                   key=lambda p: [int(n) for n in re.findall(r"\d+", os.path.basename(p))])
    if not paths:
        raise SystemExit(
            f"no checkpoints matching {args.pattern} in {args.ckpt_dir}. "
            "Checkpoints are not versioned; point --ckpt-dir at the Drive copy.")

    rows = [r for r in (summarise(p) for p in paths) if r]
    skipped = len(paths) - len(rows)
    if not rows:
        raise SystemExit("none of the matched checkpoints contain a fusion gate — "
                         "single-modality and early-fusion models have no gate.")

    print(f"{'checkpoint':28s} {'gate':>9s} {'tanh(gate)':>11s} {'val Dice':>9s}")
    for r in rows:
        print(f"{r['checkpoint']:28s} {r['gate_raw']:9.4f} {r['gate_tanh']:11.4f} "
              f"{r['val_dice']:9.4f}")

    values = [abs(r["gate_tanh"]) for r in rows]
    mean = sum(values) / len(values)
    spread = (sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)) ** 0.5
    n_off = sum(v < NEAR_ZERO for v in values)

    print(f"\n|tanh(gate)|  mean {mean:.4f} +/- {spread:.4f}   "
          f"min {min(values):.4f}  max {max(values):.4f}")
    print(f"{n_off}/{len(values)} runs below {NEAR_ZERO}")
    if n_off >= 0.8 * len(values):
        print("\n-> (a) the auxiliary stream was switched off. The fusion gain over "
              "early fusion is then a statement about early fusion's harm, not "
              "about the second modality carrying signal. Report it that way.")
    elif mean >= NEAR_ZERO:
        print("\n-> (b) the auxiliary stream is in use and still yields no gain: the "
              "information is insufficient at this fusion point. The global-context "
              "variant (plan section 7bis) is the cheap next test.")
    else:
        print("\n-> mixed across runs. Report the spread rather than a single number, "
              "and do not build an argument on the gate alone.")

    if "role_separation" in rows[0]:
        seps = [r["role_separation"] for r in rows]
        print(f"role embedding separation  mean {sum(seps)/len(seps):.4f}   "
              "(0 means the two fusion directions were never distinguished)")

    if skipped:
        print(f"({skipped} matched checkpoints had no fusion gate and were skipped)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (k != "checkpoint", k))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
