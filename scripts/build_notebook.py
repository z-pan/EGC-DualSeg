# -*- coding: utf-8 -*-
"""Generate notebooks/colab_train.ipynb (avoids hand-writing notebook JSON)."""
import json, os

OUT = r"C:\Users\zpanp\projects\EGC-DualSeg\notebooks\colab_train.ipynb"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}

cells = []

cells.append(md(r"""
# EGC-DualSeg — Colab driver

This notebook is a **thin driver**, not where the code lives. Everything of substance is in
the repository, so Colab and the local machine run identical code and the training logic stays
reviewable in git.

**What it does**: mount Drive → pull the repo → stage the data package → point outputs at Drive
→ call `scripts/train.py` for a queue of configurations.

**Why outputs go to Drive**: sessions drop. `scripts/train.py` skips any run whose prediction
CSV already exists, so after a disconnect you re-run the *same* queue cell and it continues from
where it stopped. Nothing needs to be tracked by hand.

**Deadline**: Colab Pro+ expires 2026-08-20. All GPU work must finish by 8/17. The end state to
protect is that figures are reproducible from the committed CSVs alone, with no GPU.

## One-time setup on Drive

```
MyDrive/EGC-DualSeg/
├── data/
│   ├── egc_dualseg_384.npz      <- upload from local data/packaged/
│   └── manifest.csv             <- upload from local data/packaged/
├── results/                     <- created automatically
└── checkpoints/                 <- created automatically
```

The package is de-identified by construction: cropping to the endoscopic field removed the
burnt-in patient-information panel before it was written.
"""))

cells.append(md("## 1 — Runtime probe\n\nRecord the tier. Timings are meaningless without it."))
cells.append(code(r"""
import os, subprocess, sys, platform

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print("vCPU        ", os.cpu_count())
print("RAM (GB)    ", round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))
print("python      ", platform.python_version())

import torch
print("torch       ", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0))

# num_workers scales with vCPU, not with the GPU. On this workload the data pipeline is the
# more likely bottleneck, so this matters more than the GPU tier.
NUM_WORKERS = 2 if os.cpu_count() <= 4 else 6
print("NUM_WORKERS ", NUM_WORKERS)
"""))

cells.append(md("## 2 — Mount Drive and define paths"))
cells.append(code(r"""
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = "/content/drive/MyDrive/EGC-DualSeg"
REPO       = "/content/EGC-DualSeg"
# Staged inside the repo so the relative paths in configs/*.yaml resolve unchanged.
# It is on Colab's local SSD, not on Drive, and data/ is gitignored.
LOCAL_DATA = f"{REPO}/data/packaged"

DRIVE_DATA        = f"{DRIVE_ROOT}/data"
DRIVE_RESULTS     = f"{DRIVE_ROOT}/results"
DRIVE_CHECKPOINTS = f"{DRIVE_ROOT}/checkpoints"
for d in (DRIVE_RESULTS, DRIVE_CHECKPOINTS):
    os.makedirs(d, exist_ok=True)

assert os.path.isfile(f"{DRIVE_DATA}/egc_dualseg_384.npz"), \
    f"Upload the packaged dataset to {DRIVE_DATA} first (see the setup block above)."
print("drive ok")
"""))

cells.append(md("## 3 — Clone or update the repository\n\nPublic repo, so no credentials are needed."))
cells.append(code(r"""
if not os.path.isdir(REPO):
    !git clone --quiet https://github.com/z-pan/EGC-DualSeg.git {REPO}
else:
    !git -C {REPO} pull --quiet

os.chdir(REPO)
sys.path.insert(0, REPO)
!git -C {REPO} log --oneline -1
"""))

cells.append(md("## 4 — Dependencies\n\nColab ships torch and torchvision. Only fill the gaps, and record the version actually used."))
cells.append(code(r"""
!pip install --quiet pyyaml pandas

import torch, torchvision, yaml
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| pyyaml", yaml.__version__)
"""))

cells.append(md(r"""
## 5 — Stage the data package on local SSD

Reading a 90 MB `.npz` from Drive on every epoch goes over the network. Copy once.
"""))
cells.append(code(r"""
import shutil, time

os.makedirs(LOCAL_DATA, exist_ok=True)
for name in ("egc_dualseg_384.npz", "manifest.csv"):
    src, dst = f"{DRIVE_DATA}/{name}", f"{LOCAL_DATA}/{name}"
    if not os.path.exists(dst):
        t0 = time.time()
        shutil.copy(src, dst)
        print(f"copied {name}  ({os.path.getsize(dst)/1e6:.0f} MB, {time.time()-t0:.1f}s)")
    else:
        print(f"present {name}")

import numpy as np
blob = np.load(f"{LOCAL_DATA}/egc_dualseg_384.npz")
print("images", blob["images"].shape, blob["images"].dtype,
      "| masks", blob["masks"].shape)
"""))

cells.append(md(r"""
## 6 — Point outputs at Drive

`results/` and `checkpoints/` inside the repo become symlinks into Drive, so a dropped session
loses nothing and the queue can resume.
"""))
cells.append(code(r"""
for local, target in (("results", DRIVE_RESULTS), ("checkpoints", DRIVE_CHECKPOINTS)):
    path = os.path.join(REPO, local)
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        # move anything already written into Drive rather than discarding it
        for f in os.listdir(path):
            shutil.move(os.path.join(path, f), os.path.join(target, f))
        shutil.rmtree(path)
    os.symlink(target, path)
    print(f"{local} -> {os.readlink(path)}")

print("\nexisting prediction CSVs:", len([f for f in os.listdir(DRIVE_RESULTS)
                                          if f.startswith("predictions_")]))
"""))

cells.append(md(r"""
## 7 — Benchmark on the development fold

**This is the gate before the full queue.** Until now the pipeline has only been exercised on
CPU for one or two epochs; AMP, `GradScaler`, the CUDA path and a multi-worker `DataLoader` have
never run on real hardware.

Fold 4 is the **development fold**: every hyperparameter decision is made here and then frozen.
All five folds are still reported, and the Methods section discloses that selection used fold 4.
Tuning per fold would leak.

Read three things off this run:

1. **seconds/epoch** → schedules the whole 8/08–8/16 batch.
2. **is the validation Dice curve still climbing at epoch 80?** If it is, raise `epochs` to
   150–200 and `patience` to 25. 119 training pairs at batch 8 is only ~14 steps per epoch, so
   80 epochs is ~1100 steps, which may be short.
3. **GPU utilisation** (run `!nvidia-smi` in another cell mid-training). Low utilisation means
   the data pipeline is the bottleneck; raise `NUM_WORKERS` rather than the GPU tier.
"""))
cells.append(code(r"""
DEV_FOLD = 4      # development fold: hyperparameters are chosen here, then frozen

!python scripts/train.py --config configs/ours.yaml \
    --folds {DEV_FOLD} --seeds 0 --num-workers {NUM_WORKERS}
"""))
cells.append(code(r"""
# Validation curve of the benchmark run — decides `epochs` and `patience`.
import pandas as pd, matplotlib.pyplot as plt

log = pd.read_csv(f"{DRIVE_RESULTS}/logs/ours_fold{DEV_FOLD}_seed0.csv")
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(log.epoch, log.train_loss); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss")
ax[1].plot(log.epoch, log.val_dice);   ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val Dice")
best = log.val_dice.idxmax()
ax[1].axvline(log.epoch[best], ls="--", c="0.5")
plt.tight_layout(); plt.show()
print(f"best val Dice {log.val_dice.max():.4f} at epoch {int(log.epoch[best])} "
      f"of {int(log.epoch.max())}")
print("still climbing near the end?" ,
      log.val_dice.tail(10).mean() >= log.val_dice.iloc[-20:-10].mean())
"""))

cells.append(md(r"""
### Optional: learning-rate check on the development fold

Only if the benchmark looks unstable or clearly under-trained. Three points, fold 4 only.
Whatever is chosen here is frozen for every config, fold and seed.
"""))
cells.append(code(r"""
# for lr in (1e-4, 3e-4, 1e-3):
#     !python scripts/train.py --config configs/ours.yaml --folds {DEV_FOLD} --seeds 0 \
#         --num-workers {NUM_WORKERS} --force
#     # (set lr in the config, or add a --lr flag to scripts/train.py first)
"""))

cells.append(md(r"""
## 8 — Run queue

Sequential and idempotent: a run whose prediction CSV exists is skipped, so re-running this
cell after a disconnect resumes the queue. A failing config does not abort the rest.

Ablations are in the **same** queue rather than deferred — after 8/20 there is no Pro+ runtime
to come back to.
"""))
cells.append(code(r"""
CONFIGS = [
    "configs/wli_only.yaml",
    "configs/nbi_only.yaml",
    "configs/early_fusion_wli.yaml",
    "configs/early_fusion_nbi.yaml",
    "configs/ours.yaml",
    # ablations — keep in this queue, not after it
    # "configs/abl_no_scale_norm.yaml",
    # "configs/abl_no_role_embed.yaml",
    # "configs/abl_aux_skip.yaml",
]
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [0, 1, 2]

import subprocess, time
for cfg in CONFIGS:
    print(f"\n{'='*70}\n{cfg}\n{'='*70}", flush=True)
    t0 = time.time()
    cmd = ["python", "scripts/train.py", "--config", cfg,
           "--folds", *map(str, FOLDS), "--seeds", *map(str, SEEDS),
           "--num-workers", str(NUM_WORKERS)]
    r = subprocess.run(cmd)
    print(f"[{cfg}] exit {r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
"""))

cells.append(md(r"""
## 9 — Completion matrix

Run before archiving on 8/17. Every cell must be filled; a gap here after 8/20 cannot be
repaired on this hardware.
"""))
cells.append(code(r"""
import itertools, pandas as pd

names = [os.path.splitext(os.path.basename(c))[0] for c in CONFIGS]
rows = []
for name, fold in itertools.product(names, FOLDS):
    have = sum(os.path.exists(f"{DRIVE_RESULTS}/predictions_{name}_fold{fold}_seed{s}.csv")
               for s in SEEDS)
    rows.append(dict(config=name, fold=fold, seeds_done=have, expected=len(SEEDS)))
matrix = pd.DataFrame(rows).pivot(index="config", columns="fold", values="seeds_done")
print(matrix.to_string())

missing = [r for r in rows if r["seeds_done"] < r["expected"]]
print(f"\n{len(rows) - len(missing)}/{len(rows)} (config, fold) cells complete")
if missing:
    print("MISSING:")
    for m in missing:
        print(f"  {m['config']} fold {m['fold']}: {m['seeds_done']}/{m['expected']}")
else:
    print("all runs present — safe to archive")
"""))

cells.append(md(r"""
## 10 — Fusion gate diagnostic

Seconds, no GPU, and it decides what comes next. Stage 1 showed registration-free fusion
beating naive early fusion decisively and beating the single-modality baselines by nothing.
`fusion.gate` starts at 0 and multiplies the whole attended contribution, so where it ended up
separates *the model switched the auxiliary stream off* from *the model used it and the signal
was not there*. Only the second case justifies trying the global-context variant.
"""))
cells.append(code(r"""
!python scripts/gate_report.py --ckpt-dir {DRIVE_CHECKPOINTS} --out {DRIVE_RESULTS}/gate_report.csv
"""))

cells.append(md(r"""
## 11 — Stage 2: histological grading on the pathology endpoint

This is where the dual-versus-single claim is actually adjudicated. Stage 1 scores against SAM
masks derived from the clinician's own boxes, so a gain there is partly a statement about the
labelling procedure. Here the label is the ESD pathology report: low-grade versus high-grade or
worse, n = 48, 21:27.

The Stage-1 trunk is frozen and only supplies a region-pooled 512-d vector per patient; a
regularised linear probe is fitted on top, with the regularisation chosen by inner CV inside the
training folds. 48 patients cannot fine-tune a ResNet-34, and the folds are the same ones Stage 1
used, so no held-out patient has been seen by either stage.

**Run the positive control first.** Macroscopic type (depressed 0-IIc versus not, n = 45) is
plainly visible on white light. If the pipeline cannot separate *that*, a null result on grade
says nothing about the modalities and everything about the pipeline.
"""))
cells.append(code(r"""
import subprocess, time

# (Stage-1 config, reference modality) — two frames, two arms each, exactly as in Stage 1.
ARMS = [
    ("configs/wli_only.yaml", "WLI"),
    ("configs/ours.yaml",     "WLI"),
    ("configs/nbi_only.yaml", "NBI"),
    ("configs/ours.yaml",     "NBI"),
]

for target in ("macro", "grade"):          # positive control first, deliberately
    print(f"\n{'='*70}\nendpoint: {target}\n{'='*70}", flush=True)
    for cfg, ref in ARMS:
        t0 = time.time()
        r = subprocess.run(["python", "scripts/train_grade.py", "--seg-config", cfg,
                            "--ref", ref, "--target", target,
                            "--num-workers", str(NUM_WORKERS)])
        print(f"[{cfg} ref {ref}] exit {r.returncode} in {(time.time()-t0)/60:.1f} min",
              flush=True)
"""))
cells.append(code(r"""
!python scripts/summarise_grade.py --results {DRIVE_RESULTS} --target macro
!python scripts/summarise_grade.py --results {DRIVE_RESULTS} --target grade
"""))

cells.append(md(r"""
## 8/17 archive checklist

- [ ] every prediction CSV and grade CSV committed to the repository
- [ ] `results/gate_report.csv` committed
- [ ] checkpoints copied to Drive (they are not versioned)
- [ ] `configs/folds.csv`, all configs and the seed list committed
- [ ] training logs exported
- [ ] figures regenerate locally from CSVs alone, with no GPU
- [ ] `summarise.py` and `summarise_grade.py` both run offline against the committed CSVs
"""))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)

with open(OUT, encoding="utf-8") as fh:
    back = json.load(fh)
print(f"wrote {OUT}")
print(f"cells: {len(back['cells'])} "
      f"({sum(c['cell_type']=='code' for c in back['cells'])} code, "
      f"{sum(c['cell_type']=='markdown' for c in back['cells'])} markdown)")
