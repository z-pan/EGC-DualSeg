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
python scripts/boundary_metrics.py    # boundary_*.csv  -> Fig. 3, Fig. 4
python scripts/threshold_sweep.py     # sweep_*.csv     -> Fig. 3c
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

### 3. The MMD alignment term at lambda = 0.3, and why one weight is not enough

Calibration on the development fold, 12 epochs, one seed:

| lambda | raw MMD | contribution | share of objective | dev Dice |
|---|---|---|---|---|
| 0.0001 | 0.0362 | 0.000004 | 0.00% | 0.6831 |
| 0.01 | 0.0356 | 0.000356 | 0.04% | 0.6885 |
| 0.1 | 0.0351 | 0.003514 | 0.37% | 0.6896 |
| **0.3** | 0.0363 | 0.010901 | **1.12%** | 0.6882 |
| **1.0** | 0.0519 | 0.051927 | **5.03%** | **0.7078** |

Two readings that must be carried into the write-up:

* **The published 1e-4 is inert here, measured rather than argued**: it
  contributes 4e-6, 0.00% of the objective. This is the direct evidence that a
  loss weight does not transfer across architectures.
* **The full sweep was trained at 0.3, which is not the value the stated
  selection rule picks.** Both 0.3 and 1.0 satisfy the 1-5% band and 1.0 has the
  better development Dice by ~0.02. A single run at 0.3 therefore cannot support
  "the alignment term is harmful" without inviting the reply that the baseline
  was under-weighted. Section 13.5 of the notebook runs 1e-4, 0.01 and 1.0 to
  settle it; until those land, the result below is stated **at lambda = 0.3**.
* **The term barely moves what it minimises.** Raw MMD sits near 0.035 from 1e-4
  through 0.3 and *rises* to 0.052 at 1.0. Pushing harder on the alignment does
  not achieve more alignment. That may be the more interesting finding.

### 3b. At lambda = 0.3 the term is harmful

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

### 3c. The weight sweep settles it: harmful at every weight, monotonically

Four orders of magnitude, the published value included. NBI cohort mean Dice:

| lambda | NBI Dice | vs the baseline's all-threshold ceiling (0.7850) |
|---|---|---|
| none (`mid_fusion`) | 0.8016 | above |
| 1e-4 (published) | 0.7956 | above |
| 0.01 | 0.7856 | above |
| 0.3 | 0.7795 | **below** |
| 1.0 | 0.7678 | **below** |

Monotone on NBI, and the trend is statistically real: Friedman across the five
arms gives chi2 = 41.1, p = 2.6e-08 (NBI) and chi2 = 26.7, p = 2.3e-05 (WLI);
the per-image Spearman of Dice against log10(lambda) has median rho = -0.300,
negative for 104/150 images on NBI and 93/150 on WLI. WLI is significant but not
monotone in its cohort means, which is what a null frame looks like.

Two consequences:

* **The pre-registered reading is met.** No weight beats `mid_fusion`, so this is
  a claim about the method rather than about a hyperparameter: batch-level
  distribution alignment that ignores the pairing costs accuracy here, and costs
  more the harder it is applied.
* **The "you under-weighted the baseline" objection reverses.** lambda = 1.0 is
  the value the selection rule picks, and it is the worst of the five:
  `w1.0 - mid_fusion` median -0.0142 on NBI and -0.0184 on WLI, both p < 0.0001.

### 3d. Two methodological findings that fell out of the sweep

**The noise floor is small, and paired medians are the reason.** `mmd_fusion_w1e-4`
contributes 4e-6 to the objective, so it and `mid_fusion` are functionally the
same configuration — an accidental near-replicate. Their paired difference is
median -0.0008 on NBI (p = 0.31) and -0.0017 on WLI (p = 0.36): essentially zero,
and an order of magnitude below the headline `ours - nbi_only` effect of +0.0089.

But their **cohort means** differ by -0.0061 and -0.0092, five times the paired
median. A handful of images carries the whole gap. This is direct evidence for
reporting paired medians with Wilcoxon rather than cohort means, and it is worth
stating in Methods as a measured justification rather than a stylistic choice.

**The threshold-sweep verdict is unstable near its category boundaries.**
`mmd_fusion_w1e-4` and `mid_fusion` are statistically indistinguishable on Dice
(p = 0.31), yet on NBI one earns *"outside the baseline's curve on both axes: a
better model, not a better threshold"* and the other only *"wins on one axis
only"*. The verdict is computed by matching **cohort means**, the very quantity
shown above to be fragile.

That does not overturn the ceiling argument, whose own panel was retired from
Fig. 4 on 2026-08-24 because the baseline’s best threshold is 0.5 anyway, but it
constrains how it may be written: the sweep returns a category label, not an
estimate with an interval, and adjacent categories can be separated by noise. Report the ceiling
comparison as the numeric gap it is, and do not lean on the verdict wording.

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
