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

## Results (all 45 runs complete, 2026-08-10)

Cohort mean Dice per reference frame. The two frames have different ground-truth
masks and are never compared with each other.

| arm | WLI | NBI |
|---|---|---|
| single modality | 0.7180 | 0.7850 |
| early fusion | 0.6301 | 0.7542 |
| `mid_fusion` | 0.7194 | 0.8016 |
| `mid_fusion_wide` | 0.7174 | 0.8011 |
| `mmd_fusion` | 0.7185 | 0.7795 |
| `ours` | 0.7158 | 0.7945 |

### 1. The parameter-count objection is dead, empirically

`mid_fusion_wide − mid_fusion`, paired over images:

| frame | metric | median | p |
|---|---|---|---|
| WLI | Dice | −0.0000 | 0.595 |
| NBI | Dice | −0.0006 | 0.762 |
| WLI | bf3 | −0.0066 | 0.143 |
| NBI | bf3 | −0.0001 | 0.495 |

1.84 M extra parameters buy nothing measurable. The gap between the plain
operator and the cross-attention operator was never a budget effect, and this
says so with data rather than with an argument.

### 2. Cross-attention does NOT beat bottleneck concatenation on region overlap

The cohort means favour `mid_fusion` on NBI by 0.0071, and the paired test does
not support it:

| comparison | frame | Dice median | p |
|---|---|---|---|
| `mid_fusion` − `ours` | NBI | +0.0033 | 0.34 |
| `mid_fusion_wide` − `ours` | NBI | +0.0015 | 0.33 |
| `mid_fusion` − `ours` | WLI | +0.0063 | 0.10 |
| `mid_fusion_wide` − `ours` | WLI | −0.0037 | 0.92 |

**"The proposed operator is more accurate than bottleneck concatenation" cannot
be claimed.** What separates the arms is the contour, not the region:

| comparison | frame | bf3 median | p |
|---|---|---|---|
| `mid_fusion_wide` − `ours` | WLI | −0.0072 | 0.023 |
| `mid_fusion_wide` − `ours` | NBI | −0.0090 | 0.052 |
| `mid_fusion` − `ours` | NBI | −0.0066 | 0.19 |

and the missed-lesion side, where `ours` is better than `mid_fusion` on NBI
(`residual_area_frac` +0.0079, p = 0.015) while `mid_fusion` takes less normal
tissue (`over_area_ratio` −0.0083, p = 0.008). For a resection margin the missed
side is the one that matters.

**These p values carry no multiple-comparison correction.** 0.023 does not
survive Bonferroni over four metrics. The boundary advantage is suggestive, not
established, and must be written that way.

### 3. The MMD alignment term is harmful here

| comparison | frame | metric | median | p |
|---|---|---|---|---|
| `mmd_fusion` − `mid_fusion` | NBI | Dice | −0.0097 | 0.0002 |
| `mmd_fusion` − `mid_fusion` | NBI | bf3 | −0.0109 | 0.056 |
| `mmd_fusion` − `ours` | NBI | Dice | −0.0056 | 0.0076 |
| `mmd_fusion` − `ours` | NBI | bf3 | −0.0144 | 0.0009 |

Same operator, same everything, one added loss term. On NBI it drops the arm
from above the single-modality ceiling (0.7850) to below it: `mid_fusion` 0.8016
and `mid_fusion_wide` 0.8011 clear it, `mmd_fusion` 0.7795 does not.

A batch-level distribution alignment that is invariant to which WLI frame was
paired with which NBI frame is not merely unhelpful on this cohort — it costs
accuracy. Report the calibrated weight alongside the published 1e-4.

### 4. What this does to the paper's framing

Three independent dual arms now beat the single-modality baseline on NBI —
`mid_fusion` +0.0108 (p = 0.0014), `mid_fusion_wide` +0.0112 (p = 0.0049),
`ours` +0.0089 (p = 0.024) — and all three clear the baseline's all-threshold
ceiling. The dual-versus-single claim is no longer one operator's result.

The operator claim changes shape. It is not "cross-attention is the right
operator". It is:

> What decides the outcome is whether pixel-level correspondence is imposed.
> Impose it at full resolution and the arm collapses; stop imposing it — at the
> 1/32 grid or not at all — and region-overlap accuracy is the same. The residual
> difference between those two is in contour quality, not in region overlap.

That claim is more transferable than the original one, and it is the claim the
misalignment measurement and the alignment-ceiling probe were already making.

On the WLI frame every arm is within noise of the single-modality baseline, and
the ceiling comparison there separates arms by less than 0.005 Dice. Read WLI as
a null throughout rather than reading its ordering.

## Pre-registration note

These arms were added **after** the main results were seen. `Methods 2.7` states
that the decision criteria were fixed in advance. The honest disclosure is that
these two comparisons were added for completeness of the operator ladder, and
that they are included in the multiple-comparison correction rather than
presented alongside the pre-specified arms as though they had been planned.
