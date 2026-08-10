# Two added fusion arms — what they are and how to run them

Added 2026-08-10, after the main sweep was already complete. They exist to close
the same gap: the paper compared the proposed operator against **input-level
fusion only**, which is the weakest rung of the ladder by construction. One
strong-baseline comparison is the first thing a referee will ask for.

| arm | operator | differs from |
|---|---|---|
| `mid_fusion` | concatenate auxiliary features onto the reference bottleneck, 1x1 conv | `ours`: fusion operator only |
| `mid_fusion_wide` | the same, widened until the parameter count matches `ours` | `mid_fusion`: operator width only |
| `mmd_fusion` | plain concatenation **plus** an MMD distribution-alignment loss | `mid_fusion`: one loss term only |

Everything else — fold assignment, seeds, schedule, optimiser, augmentation, the
pyramid levels the auxiliary stream supplies, the role embedding, the
identity-at-init gate — is copied from `ours.yaml` unchanged. `mmd_fusion -
mid_fusion` therefore isolates the alignment term exactly, and `ours -
mid_fusion` isolates the operator.

## Where the ladder now stands

| rung | correspondence assumed | arm |
|---|---|---|
| full pixel resolution | every pixel | `early_fusion_*` |
| 1/32 grid (~32 px cells) | coarse spatial | `mid_fusion` |
| population statistics only | none spatial, marginals matched | `mmd_fusion` |
| none | none | `ours` |

This also splits the two explanations offered for early fusion's failure.
`mid_fusion` leaves `conv1` and its ImageNet weights untouched, so if it still
fails, the six-channel dilution account is not the whole story.

## Step 1 — calibrate the MMD weight (do this first)

A loss weight is not portable across architectures. The authors of the method
being reproduced report 1e-4, but that value is only meaningful relative to their
feature scale and their objective. Measured here on fold 0, the raw MMD term is
~0.03 against a Dice+BCE loss of ~1.24, so 1e-4 contributes ~3e-6 and does
nothing at all. Running the full sweep at 1e-4 would return "indistinguishable
from plain concatenation" for a reason that has nothing to do with the method.

Run a short sweep and read the `train_mmd` column in `results/logs/`:

Fold 4 is this project's development fold: every hyperparameter has been chosen
there and then frozen (see the notebook, section 7). Calibrating anywhere else
would break that rule for the sake of a baseline.

```bash
for w in 0.0001 0.01 0.1 0.3 1.0; do
  python scripts/train.py --config configs/mmd_fusion.yaml \
    --mmd-weight $w --name mmdcal_$w \
    --folds 4 --seeds 0 --epochs 12 \
    --out-dir results/_mmdcal --ckpt-dir checkpoints/_mmdcal
done
```

Each run needs its own `--name`: a run whose prediction CSV already exists is
skipped, so two weights sharing a name would silently return the first one's
result. Then read the logs:

```bash
python - <<'PY'
import glob, os, pandas as pd
for p in sorted(glob.glob("results/_mmdcal/logs/mmdcal_*.csv")):
    d = pd.read_csv(p)
    w = os.path.basename(p).split("_")[1]
    print(f"w={w:8s} mmd_raw={d.train_mmd.iloc[-1]:.4f} "
          f"contribution={float(w)*d.train_mmd.iloc[-1]:.5f} "
          f"loss={d.train_loss.iloc[-1]:.4f} "
          f"share={100*float(w)*d.train_mmd.iloc[-1]/d.train_loss.iloc[-1]:.2f}% "
          f"best_dice={d.val_dice.max():.4f}")
PY
```

Pick the weight whose `share` is roughly 1–5% and whose `best_dice` has not
collapsed. Development fold, one seed, nothing else — calibrating across folds
would leak held-out patients into a hyperparameter.

**Report both.** State the calibrated weight, state that 1e-4 was the published
value, and state why it was not used. A referee who knows the source paper will
check this number; the defensible position is that the term's influence was
matched, not its literal value.

## Step 2 — the full sweep

```bash
python scripts/train.py --config configs/mid_fusion.yaml         # 5 folds x 3 seeds
python scripts/train.py --config configs/mid_fusion_wide.yaml    # 5 folds x 3 seeds
python scripts/train.py --config configs/mmd_fusion.yaml         # 5 folds x 3 seeds
```

Runs whose prediction CSV already exists are skipped, so an interrupted session
resumes with the same command.

## Step 3 — the analyses the new arms need

The summarise scripts glob `results/*.csv` and pick up new configs
automatically, but the two per-image passes must be run for the new arms first:

```bash
python scripts/boundary_metrics.py    # boundary_*.csv  -> Fig. 2b, Fig. 3c
python scripts/threshold_sweep.py     # sweep_*.csv     -> Fig. 2c, Fig. 3b
python scripts/summarise.py
python scripts/summarise_boundary.py
python scripts/summarise_clinical.py
python scripts/summarise_sweep.py
```

## Two things that are NOT automatic

**Figure styling.** `figures/fig2_fusion_operator.py` and
`figures/fig3_dual_vs_single.py` both carry a hard-coded `STYLE` dict with
exactly two configurations. Adding a third means allocating another fill and
line style **within the existing frame colours** — blue and teal encode frame
identity and nothing else, and a third hue would break the palette discipline
the figures are built on. Reserve red for reference lines.

**Parameter counts.** Measured with `pretrained=False`:

| arm | parameters | vs `ours` |
|---|---|---|
| `single` | 24,437,394 | −9.3% |
| `early_fusion_*` | 24,447,634 | −9.3% |
| `mid_fusion` | 25,097,746 | −6.8% |
| **`mid_fusion_wide`** | **26,936,082** | **−0.016%** |
| `ours` | 26,940,434 | — |

`mid_fusion` is the plain operator and is 1.84 M lighter, so `ours − mid_fusion`
confounds the operator with the budget. `mid_fusion_wide` removes that: same
data, folds, seeds, schedule, optimiser, augmentation, auxiliary levels and
parameter count, differing only in how the two streams are combined.

Report all three numbers. `mid_ffn_mult: 3.5` was solved for to match the count
and must not be tuned for accuracy — tuning it would reintroduce the asymmetry
the arm exists to remove.

Which arm carries the claim in the paper: **`mid_fusion_wide`**. `mid_fusion`
stays in as the un-widened reference so that the effect of width alone is
visible rather than assumed away.

## Pre-registration note

These arms were added **after** the main results were seen. `Methods 2.7` states
that the decision criteria were fixed in advance. The honest disclosure is that
these two comparisons were added for completeness of the operator ladder, and
that they are included in the multiple-comparison correction rather than
presented alongside the pre-specified arms as though they had been planned.
