# EGC-DualSeg — repo conventions

Dual-modal (WLI + NBI) segmentation and pathology-anchored grading of early gastric cancer.

- **Design rationale**: `Desktop\Research\08_2025_松江医院项目-早期胃癌内镜智能诊断\论文方案_双模态胃早癌分割.md` (v2.3)
- **Implementation plan**: same folder, `实现规划_双模态分割与病理分析.md`
- **Manuscript + figures**: same folder, `manuscript\` and `figures\`

Guiding constraint: **finish it**. The architecture is fixed (plan §4). Do not spend the
schedule redesigning it.

---

## Hard deadline

**Colab Pro+ expires 2026-08-20.** All GPU training must be complete by **8/17**, with 8/17–8/19
as buffer. After that only free-tier T4 or a local 4 GB laptop GPU is available. Anything that
needs re-running afterwards costs an order of magnitude more effort — so over-run now rather
than deferring ablations.

The end state to protect: **figures must be reproducible from committed CSVs alone, with no GPU.**

---

## Data

Raw archive (never modified, never leaves the machine):
`C:\Users\zpanp\projects\datasets\gastric_cancer`

| Step | Script | Output | In git? |
|---|---|---|---|
| Package | `scripts/package_dataset.py` | `data/packaged/egc_dualseg_384.npz`, `manifest.csv` | no |
| Folds | `scripts/make_folds.py` | `configs/folds.csv` | **yes** |

Cohort: 81 folders → minus 2 duplicate records (病例76, 病例51) and 2 oesophageal lesions
(病例41, 病例47) → **77 patients, 150 same-view pairs (77 near / 73 distant), 300 images,
48 gradeable**.

### Three traps that have already cost time

1. **Masks are named after the JSON stem, not the image stem.** Batch 2 stores `016.jpg`
   next to `016白光远景.json`, and the mask is `016白光远景.png`. Resolving by image stem
   silently drops all 35 batch-2 cases with no error.
2. **Mask-area fractions must use a consistent denominator.** Mask coverage is measured
   against the *cropped endoscopic field*; clinician-box coverage in the raw metadata is
   against the *full frame*, and the field is only ~42–47% of the frame. Comparing the two
   directly once produced a false "the model flooded" conclusion.
3. **Folds are permanent.** `configs/folds.csv` is written once and committed. Every config,
   seed, ablation and both stages share it. `make_folds.py` refuses to overwrite without
   `--force`; using `--force` invalidates every result already trained.

### What is deliberately not committed

`manifest.csv` and the `.npz` carry per-patient clinical attributes (grade, lesion extent,
site). They stay local and on Drive. `folds.csv` is committed because reproducibility requires
it. The raw archive is never reconstructible from this repo.

The packaged data is **de-identified by construction**: cropping to the endoscopic field
removes the burnt-in patient-information panel before anything is written.

---

## Layout

```
configs/    yaml, shared verbatim between local and Colab; folds.csv lives here
scripts/    one-shot and entry-point scripts
src/        data / models / losses / engine / utils
notebooks/  colab_train.ipynb — mount Drive, pull repo, run configs
data/       gitignored; packaged dataset only
results/    prediction CSVs (committed); checkpoints (not committed)
```

## Where things run

| Task | Where |
|---|---|
| Packaging, fold assignment | local |
| Smoke test (256², batch 2, 3 epochs, 1 fold) | local, RTX 3050 Ti 4 GB |
| All real training | Colab, background execution |
| Figures, analysis, writing | local, no GPU |

## Output contract — fixed before training, do not change

`results/predictions_{config}_fold{f}_seed{s}.csv`
```
case, scale, ref_modality, config, fold, seed, dice, iou, precision, recall,
pred_area_frac, gt_area_frac, pred_long_axis_px, content_w
```
`results/grade_{config}_fold{f}_seed{s}.csv`
```
case, config, fold, seed, y_true, y_prob
```
Stage-2 `config` encodes the arm, not just the Stage-1 config: `{seg}_{ref}` for the grade
endpoint (`ours_wli`, `wli_only_wli`, …) and `macro_{seg}_{ref}` for the positive control.
The lesion-level late-fusion arms have no reference modality and are named `late_fusion`,
`late_null` and `late_aux_shuffled` (with the `macro_` prefix for the control endpoint).
Each patient is held out in exactly one fold, so concatenating the five folds gives one
probability per patient per seed — that is what `summarise_grade.py` relies on.

`results/exemplars/` — `index.csv` plus one 1-bit PNG per selected (config, image), written by
`scripts/dump_exemplars.py`. Predicted masks are the one figure input that the metric CSVs
cannot reconstruct, so they are dumped while a runtime exists and committed. Each mask comes
from the fold that held its patient out, and `index.csv` records the rule that selected it —
a qualitative panel whose selection rule is not stated reads as cherry-picking.

These feed `figures/fig4_*.py`, `fig5_*.py`, `fig6_*.py` directly. Changing the schema means
re-running training.

## Modelling notes that are easy to get wrong

- **U-Net skips come from the reference stream only.** Auxiliary-stream features are not
  spatially aligned with the target mask; feeding them into skips injects misalignment noise
  straight into the decoder. Fails silently — metrics just get worse.
- **Augment the two streams separately.** Reference stream and mask share one geometric
  transform; the auxiliary stream gets its own mild one.
- **Cap hue jitter at ±5°.** The colour difference between modalities is the signal.
- **Backbone weights are shared across modalities**, with per-modality FiLM adapters. 77
  patients cannot support two full encoders.
- Three seeds per configuration, minimum. The expected effect (~+0.02 Dice) is plausibly the
  same size as run-to-run variance; single-seed numbers are not reportable.
- **Stage 2 extracts features under one fixed reference modality per arm.** `ours` is
  bidirectional and could be given both directions and averaged, but then it would see twice
  as many views as the single-modality arm and part of any gain would be view averaging rather
  than fusion. Views *within* a direction (near and distant) are averaged — both arms have
  those, so that is symmetric.
- **Stage 2 runs with augmentation off.** The trunk is frozen; there is nothing to make
  invariant, and deterministic features keep the linear fit reproducible.
- **Late fusion never gets read without its two controls.** `late_fusion` is a 1024-d probe and
  the single-modality arms are 512-d, so a bare comparison confounds the second modality with
  the extra dimensions. `late_null` fits and scores the same 1024-d probe with the NBI half
  drawn from another patient, so `late_fusion − late_null` is the patient-specific NBI
  contribution and `late_null − wli_only_wli` is what the dimensions cost.
  `late_aux_shuffled` permutes the NBI half at evaluation only, keeping the fitted probe, so a
  flat result there means the probe never used NBI rather than that NBI is uninformative —
  the same distinction the gate ablation forced on the Stage-1 gate reading. All three come
  from one script and cost no extra GPU time.
- Run the `macro` positive control before reading the `grade` result. Depressed lesions are
  plainly visible on white light; if that separation fails, a null on grade is a statement
  about the pipeline, not about the modalities.
