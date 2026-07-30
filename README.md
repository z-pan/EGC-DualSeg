# EGC-DualSeg

Dual-modal (white-light + narrow-band) segmentation of early gastric cancer, with a
pathology-anchored grading stage.

The premise is that the two illumination modes routinely acquired during a single
gastroscopy are **not spatially registered**, so fusion has to be registration-free — and how
much benefit fusion can give is bounded by how well the two views actually correspond, which
this project measures before it models.

## Pipeline

```
WLI ┐                                         ┌─ skip (reference stream only) ─┐
    ├─ shared ResNet-34 + modality adapter ───┤                                ├─ U-Net decoder ─ mask
NBI ┘                                         └─ multi-scale K/V ──┐           │
                                                                   ▼           │
                            reference-role embedding ─→ cross-attention (ref = Q)
                                                                   │
                                        frozen features ─→ grade head ─→ low / high+
```

Stage 1 is supervised by clinician-reviewed masks; Stage 2 by the resection pathology report.
The modality comparison is evaluated on both, because Stage-1 masks are delineated within each
modality's own frame.

## Setup

```bash
pip install -r requirements.txt
```

Training additionally needs PyTorch; install the build that matches your CUDA
(see https://pytorch.org). The two data-preparation scripts below need only numpy and Pillow.

## Getting started

```bash
# 1. Package the raw archive into a de-identified 384px dataset (run once, locally)
python scripts/package_dataset.py

# 2. Assign patients to folds (run once; the result is committed and permanent)
python scripts/make_folds.py
```

Expected output of step 1: 77 patients, 150 pairs (77 near / 73 distant), 300 images,
48 gradeable, ~90 MB.

## Cohort

| | n |
|---|---|
| Case folders in the archive | 81 |
| minus duplicate records | −2 |
| minus oesophageal lesions | −2 |
| **Patients** | **77** |
| Same-view WLI–NBI pairs | 150 |
| Gradeable resections | 48 (21 low grade / 27 high grade or above) |

## Running the experiments

Both stages train on Colab (`notebooks/colab_train.ipynb` is a thin driver over these
scripts); both read-outs run anywhere, from the committed CSVs, with no GPU.

```bash
# Stage 1 — segmentation, 5 configs x 5 folds x 3 seeds
python scripts/train.py --config configs/ours.yaml
python scripts/summarise.py

# Which way did the fusion gate go? Needs only the checkpoints.
python scripts/gate_report.py --ckpt-dir checkpoints

# Stage 2 — grading on the pathology endpoint, one arm per invocation
python scripts/train_grade.py --seg-config configs/ours.yaml --ref WLI --target macro
python scripts/train_grade.py --seg-config configs/ours.yaml --ref WLI

# Stage 2, lesion-level late fusion: each modality encoded and pooled on its own,
# the two 512-d vectors concatenated. Writes its shuffled-partner control alongside.
python scripts/train_grade_late.py --target macro
python scripts/train_grade_late.py --target grade

python scripts/summarise_grade.py --target macro     # positive control, read this first
python scripts/summarise_grade.py --target grade
```

Stage 2 freezes the Stage-1 trunk and fits a regularised linear probe on region-pooled
features, with the regularisation chosen inside the training folds. 48 patients cannot
fine-tune a ResNet-34, and reusing the Stage-1 folds keeps every held-out patient unseen by
both stages.

## Results so far

Stage 1 is complete: 75 runs, per-image predictions committed under `results/`.

| Frame | single modality | naive early fusion | registration-free fusion |
|---|---|---|---|
| WLI mask | 0.718 | 0.630 | 0.716 |
| NBI mask | 0.785 | 0.754 | 0.795 |

Mean Dice. The two frames have different ground-truth masks and are **not** comparable with
each other. Against the single-modality baseline the fusion model gains a median +0.001 on WLI
(p = 0.62) and +0.009 on NBI (p = 0.024) — statistically visible in one frame, but not a
clinically meaningful quantity. Against naive early fusion it gains +0.076 and +0.025, both
p < 0.0001, and early fusion nearly doubles the coarse failure rate on WLI (15.3% → 26.0%).

So the finding is that **how you fuse matters a great deal and fusion itself buys almost
nothing** — which is what the measured misalignment (median lesion overlap 0.28) predicts.
Stage 2 puts the same question to an imaging-independent endpoint: the Stage-1 masks are
clinician-reviewed, but their boundaries were refined within each modality's own frame, and
the pathology grade carries no such tie to either modality.

## Repository conventions

See `CLAUDE.md` — in particular the mask-naming trap, the permanence of `configs/folds.csv`,
and the output CSV contract that the figure scripts depend on.

## Status

Stage 1 complete and committed. Stage 2's four in-plane arms are run and committed: the
positive control clears (macroscopic type, AUC 0.79–0.83) and the grade endpoint is null in
every arm, all four 95% CIs spanning 0.5. Lesion-level late fusion is implemented and not yet
run — it is the arm that separates *NBI adds nothing* from *the in-plane operator was the
bottleneck*, and it needs the Stage-1 checkpoints, so it runs where those live.
