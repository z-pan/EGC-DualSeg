# Figure and table scripts

Every number in the manuscript is recomputed by these scripts from the CSVs
committed under `results/` and `outputs/`. No GPU and no model checkpoint is
needed: the training runs are already reduced to per-image read-outs, and each
script re-derives the patient-level statistics from them with the same
aggregation order (seeds into images, images into patients, within one reference
frame).

Run from anywhere; the repository root is resolved from the script's own path.
Set `EGC_REPO` only if the read-outs live outside this tree.

```bash
python figures/fig3_dual_vs_single.py
python scripts/make_supplementary.py
```

## What each script produces

| Script | Manuscript item | Runs from this repository alone |
|---|---|---|
| `fig1_misalignment.py` | Fig. 1, spatial correspondence | **no** — needs the image archive |
| `fig3_framework.py` | Fig. 2, framework and cohort flow | **no** — needs the image archive |
| `fig2_fusion_operator.py` | Fig. 3, fusion scheme comparison | yes |
| `fig3_dual_vs_single.py` | Fig. 4, dual versus single modality | yes |
| `fig5_pathology.py` | Fig. 5, resection-pathology endpoints | yes |
| `fig2_resolution.py` | Fig. S1, effective resolution | **no** — needs the image archive |
| `figS2_mmd_sweep.py` | Fig. S2, distribution-alignment sweep | yes |
| `fig6_correspondence.py` | Fig. S3, gain against correspondence | yes |
| `graphical_abstract.py` | Graphical abstract | yes |
| `../scripts/make_supplementary.py` | Table 1, Tables S1–S3 | yes |

Seven of the ten reproduce end to end from what is committed here. The three
that do not are the ones that display endoscopic frames: the exemplar pairs in
Fig. 1a, the pipeline schematic in Fig. 2, and the scale-calibrated views in
Fig. S1.

## The images are not in this repository

The endoscopic frames are identifiable patient data and are not public. The
scripts that need them read the archive from `EGC_RAW`:

```bash
EGC_RAW=/path/to/archive python figures/fig1_misalignment.py
```

Everything derived from those images that is *not* identifiable is committed:
the per-image segmentation and boundary read-outs under `results/`, the
patient-level comparison table in `outputs/patient_level.csv`, and the
cross-modal correspondence statistics that Fig. 1b–d summarise. A reader can
therefore check every statistic in the paper without access to the images, and
can reproduce every figure whose content is a statistic rather than a photograph.

## Palette

Blue `#0F4D92` and teal `#42949E` denote the reference frame, white light and
narrow band, and nothing else. Where a panel compares fusion schemes within one
frame, the scheme is encoded by fill, hatch and line style inside that colour,
never by a third hue. Red `#B64342` is reserved for reference lines, medians and
baselines. Grey is neutral annotation.
