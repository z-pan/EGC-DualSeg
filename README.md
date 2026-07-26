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

## Repository conventions

See `CLAUDE.md` — in particular the mask-naming trap, the permanence of `configs/folds.csv`,
and the output CSV contract that the figure scripts depend on.

## Status

Data packaging and fold assignment complete. Model, training loop and Colab notebook in
progress.
