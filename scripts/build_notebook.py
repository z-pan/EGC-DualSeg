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

import torch, torchvision, yaml, sklearn
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| pyyaml", yaml.__version__, "| sklearn", sklearn.__version__)
# scikit-learn is Stage 2's linear probe; it ships with Colab, but assert it
# rather than discovering the gap an hour into a run.
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
import shutil

# Merge file-by-file rather than shutil.move on the directory: move() drops the
# source INSIDE an existing destination directory instead of merging, which
# silently produced results/exemplars/exemplars. Walking the tree and copying
# only what Drive lacks is idempotent and cannot nest.
def merge_into(src_dir, dst_dir):
    copied = skipped = 0
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        out = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(out, exist_ok=True)
        for f in files:
            s, d = os.path.join(root, f), os.path.join(out, f)
            if os.path.exists(d):
                skipped += 1
            else:
                shutil.copy2(s, d)
                copied += 1
    return copied, skipped

# Repair any nesting left behind by the previous version of this cell.
for sub in ("exemplars", "logs"):
    nested = os.path.join(DRIVE_RESULTS, sub, sub)
    if os.path.isdir(nested):
        c, s = merge_into(nested, os.path.join(DRIVE_RESULTS, sub))
        shutil.rmtree(nested)
        print(f"repaired nested {sub}/{sub}: recovered {c}, already present {s}")

for local, target in (("results", DRIVE_RESULTS), ("checkpoints", DRIVE_CHECKPOINTS)):
    path = os.path.join(REPO, local)
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        c, s = merge_into(path, target)
        shutil.rmtree(path)
        print(f"{local}: {c} new file(s) into Drive, {s} already there")
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
## 10 — Read-outs that need the checkpoints

Everything below extracts something from the trained weights that cannot be recovered from the
prediction CSVs. **Run it while there is still a runtime.** After 8/20 the checkpoints are still
on Drive but nothing can execute them.

### 10.0 Stage the checkpoints on local SSD — do not skip

Sections 10 and 11 reopen checkpoints many times: once per (config, fold) for the exemplar dump,
and once per (arm, fold, seed) for Stage 2, which is around 150 loads of a 108 MB file. Served
over the Drive mount that is well over 10 GB of network reads and dominates the runtime. Copied
once to local SSD it is a 8 GB copy and then every load is local.

The copy is skipped for files already staged, so re-running after a disconnect is cheap.
"""))
cells.append(code(r"""
import os, shutil, time

LOCAL_CKPT = f"{REPO}/checkpoints_local"
os.makedirs(LOCAL_CKPT, exist_ok=True)

t0 = time.time()
staged = skipped = 0
for name in sorted(os.listdir(DRIVE_CHECKPOINTS)):
    if not name.endswith(".pt"):
        continue
    src, dst = f"{DRIVE_CHECKPOINTS}/{name}", f"{LOCAL_CKPT}/{name}"
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        skipped += 1
        continue
    shutil.copy(src, dst)
    staged += 1

total_gb = sum(os.path.getsize(f"{LOCAL_CKPT}/{f}")
               for f in os.listdir(LOCAL_CKPT)) / 1e9
print(f"staged {staged}, already present {skipped}  "
      f"({total_gb:.1f} GB, {time.time()-t0:.0f}s)")
print("expected 75 checkpoints (5 configs x 5 folds x 3 seeds):",
      len([f for f in os.listdir(LOCAL_CKPT) if f.endswith('.pt')]))
"""))

cells.append(md(r"""
### 10.1 Fusion gate diagnostic

Seconds, and it decides what comes next. Stage 1 showed registration-free fusion beating naive
early fusion decisively and beating the single-modality baselines by nothing. `fusion.gate`
starts at 0 and multiplies the whole attended contribution, so where it ended up separates
*the model switched the auxiliary stream off* from *the model used it and the signal was not
there*. Only the second case justifies trying the global-context variant.
"""))
cells.append(code(r"""
!python scripts/gate_report.py --ckpt-dir {LOCAL_CKPT} --out {DRIVE_RESULTS}/gate_report.csv
"""))

cells.append(md(r"""
### 10.1b Gate ablation -- does the auxiliary stream contribute anything?

10.1 reports a parameter value and infers from it. This measures the thing directly: same
checkpoint, same images, forward passes that differ only in what the auxiliary stream may
contribute. Four conditions -- as trained, gate forced to 0, each frame given **another
patient's** auxiliary view, and the auxiliary view blanked.

The second of those is the strong test. A model that exploits only the auxiliary stream's
global statistics rather than *this patient's* actual second view is indistinguishable from one
that ignores it, and only breaking the pairing separates them.

**Read the bottleneck column first.** With random weights this branch shifted the bottleneck by
14% while the logits moved by 2e-6, purely because an untrained decoder attenuates it — reading
only the output would have said "does nothing" when it demonstrably did something. If the
bottleneck moves but the probabilities do not, the fusion *operator* failed rather than the
modality, and the paper has to say so.
"""))
cells.append(code(r"""
!python scripts/gate_ablation.py --ckpt-dir {LOCAL_CKPT} --out {DRIVE_RESULTS}/gate_ablation.csv
"""))

cells.append(md(r"""
### 10.2 Exemplar masks for the qualitative panel

Training saved per-image metrics and no masks, so a qualitative figure has nothing to draw.
This writes a deliberate superset — every mask picked by a stated rule, never by eye — at a
total cost of a few tens of kilobytes.

Two rules, and each mask records which one selected it so the legend can say so: `span` takes
the 5/10/25/50/75/90/95th percentile of the proposed model's Dice in each frame, weak cases
included; `contrast` takes the images where naive early fusion loses the most, since that is
the claim the figure makes.

Every mask is produced by the fold that held its patient out — the script picks the checkpoint
per patient, so no lesion is ever rendered by a model that trained on it.
"""))
cells.append(code(r"""
!python scripts/dump_exemplars.py --ckpt-dir {LOCAL_CKPT} --results {DRIVE_RESULTS}
"""))

cells.append(md(r"""
## 11 — Stage 2: histological grading on the pathology endpoint

The second endpoint, and the one with no tie to imaging. The Stage-1 masks were reviewed and
accepted case by case by the endoscopist, but their boundaries were still refined within each
modality's own frame, which leaves a residual asymmetry in a dual-versus-single comparison.
Here the label is the ESD pathology report: low-grade versus high-grade or worse, n = 48, 21:27.
The two endpoints corroborate each other rather than one propping up the other.

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
                            "--ckpt-dir", LOCAL_CKPT,
                            "--num-workers", str(NUM_WORKERS)])
        print(f"[{cfg} ref {ref}] exit {r.returncode} in {(time.time()-t0)/60:.1f} min",
              flush=True)
"""))
cells.append(md(r"""
### 11b Lesion-level late fusion — the arm that decides what comes next

Everything above fuses the two modalities **in the image plane**, before pooling, so it has to
solve spatial correspondence — and the measured lesion overlap between the frames is IoU 0.28.
That makes a null result ambiguous: NBI may add nothing, or the operator may have spent its
capacity on a registration problem that a patient-level label does not require.

This arm removes the ambiguity. Each modality is encoded by **its own** single-modality trunk,
pooled over **its own** predicted lesion region, and the two 512-d vectors are concatenated.
Nothing is ever asked to line up. If a patient-level NBI signal exists, this is the arm that
finds it.

Two controls come with it and neither is optional. Both are free — the features are extracted
once and reused.

* `late_null` — the same 1024-d probe **fitted and scored** with the NBI half taken from another
  patient. Without it, "1024-d beat 512-d" cannot be told apart from fusion.
* `late_aux_shuffled` — the real fitted probe, with the NBI half permuted **at evaluation only**.
  Without it, a null cannot be told apart from a probe that never looked at the NBI block. This
  is the same correction the gate ablation (10.1b) forced on the gate reading: measure the use,
  do not infer it from a coefficient.

**How to read the result:**

* `late_fusion` ≈ `late_null` **and** `late_fusion` ≈ `late_aux_shuffled` → the probe ignores
  NBI and NBI carries no patient-level increment. Coarse alignment (handoff §一) is then **not
  worth building**: alignment fixes usability, not existence, and the paper is written as a
  negative result with a mechanism.
* `late_fusion` > `late_null` → there is something to fuse, the in-plane operator was the
  bottleneck, and coarse alignment + mid fusion becomes worth the days it costs.
* `late_fusion` ≈ `late_null` but `late_fusion` ≫ `late_aux_shuffled` → the probe leans on NBI
  and gets nothing for it; that is a statement about the probe, not about the modality, and it
  needs saying before any of the above.

No checkpoint is trained here, so this reuses the same `wli_only` / `nbi_only` weights already
staged in 10.0.
"""))
cells.append(code(r"""
import subprocess, time

for target in ("macro", "grade"):          # positive control first, again
    t0 = time.time()
    r = subprocess.run(["python", "scripts/train_grade_late.py", "--target", target,
                        "--ckpt-dir", LOCAL_CKPT,
                        "--num-workers", str(NUM_WORKERS)])
    print(f"[late fusion {target}] exit {r.returncode} in {(time.time()-t0)/60:.1f} min",
          flush=True)
"""))
cells.append(code(r"""
!python scripts/summarise_grade.py --results {DRIVE_RESULTS} --target macro
!python scripts/summarise_grade.py --results {DRIVE_RESULTS} --target grade
"""))

cells.append(md(r"""
## 12 — Controlled-misalignment study: PILOT

The real cohort gives **one** point on the misalignment-versus-fusion-damage relation (lesion
IoU 0.28, early fusion −0.088). One point is not a relationship. This dials misalignment to
order on Kvasir-SEG, where true masks exist, so the curve can be traced and the real cohort
placed on it.

**Run this pilot before the full 44-run sweep.** It exists to answer one question:

> Is there enough headroom between the single-modality baseline and the perfectly-registered
> Oracle for the sweep to have any resolution at all?

If Oracle − single < 0.03 Dice, the complementary information is too weak, every level will sit
on top of the others, and the full sweep is wasted GPU time. Fix `--blur` / `--block` and re-run
the pilot rather than pressing on.

Two design faults this pilot already caught, both of which would have invalidated the sweep:

* Fixing the scale ratio at the 1.9x measured in the real cohort **caps achievable IoU at ~0.28**
  regardless of translation (1.9x linear is 3.6x in area), making perfect registration
  unreachable. Scale is now a separate axis, held at 1.0.
* Solving the shift **per image** rather than applying a constant one keeps each level within
  about ±0.02 of its target, so the levels stay distinguishable.

Kvasir-SEG is public and downloads to Colab's local disk in seconds — it never touches Drive.
"""))
cells.append(code(r"""
import os, subprocess, time

def find_kvasir(root="/content"):
    # Locate a directory holding images/ + masks/. The archive does not always
    # unpack to the name you expect, and hardcoding the path fails silently.
    for d, _subs, _files in os.walk(root):
        if os.path.basename(d) == "images":
            sib = os.path.join(os.path.dirname(d), "masks")
            if os.path.isdir(sib) and len(os.listdir(d)) > 100:
                return os.path.dirname(d)
    return None

KV = find_kvasir()

# 1) A 22 MB 500-image subset parked on Drive is the reliable path; the public
#    download has been flaky. Build it locally with the snippet in the repo if
#    it is not there yet.
if KV is None:
    subset = f"{DRIVE_ROOT}/kvasir_subset_500.zip"
    if os.path.isfile(subset):
        subprocess.run(["unzip", "-q", "-o", subset, "-d", "/content/"], check=False)
        KV = find_kvasir()
        if KV:
            print("unpacked the Drive subset")

# 2) Otherwise try the public archive, and report what actually came back
#    instead of swallowing the error behind -q.
if KV is None:
    z = "/content/kv.zip"
    for url in ("https://datasets.simula.no/downloads/kvasir-seg.zip",
                "https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip"):
        r = subprocess.run(["wget", "--no-check-certificate", "-O", z, url],
                           capture_output=True, text=True)
        size = os.path.getsize(z) if os.path.exists(z) else 0
        print(f"{url} -> exit {r.returncode}, {size/1e6:.2f} MB")
        if size > 10e6:
            subprocess.run(["unzip", "-q", "-o", z, "-d", "/content/"], check=False)
            KV = find_kvasir()
            if KV:
                break
        elif r.stderr:
            print("   ", r.stderr.strip().splitlines()[-2:])

assert KV, ("Kvasir-SEG unavailable. Upload kvasir_subset_500.zip to "
            f"{DRIVE_ROOT}/ and re-run this cell.")
print("Kvasir:", KV, "|", len(os.listdir(os.path.join(KV, "images"))), "images")

# Two levels only for the pilot: perfect registration, and the real cohort's 0.28
!python scripts/make_synthetic.py --kvasir {KV} \
    --out data/synth --iou 1.0 0.28 --complement S --n 500

PILOT = [
    ("configs/synth_single.yaml", "iou100", "single_S"),       # baseline, level-independent
    ("configs/synth_early.yaml",  "iou100", "early_S_iou100"), # Oracle upper bound
    ("configs/synth_early.yaml",  "iou028", "early_S_iou028"), # misaligned to 0.28
    ("configs/synth_ours.yaml",   "iou028", "ours_S_iou028"),  # registration-free
]
for cfg, lvl, name in PILOT:
    stem = f"data/synth/synth_S_{lvl}"
    t0 = time.time()
    r = subprocess.run(["python", "scripts/train.py", "--config", cfg,
                        "--npz", f"{stem}.npz", "--manifest", f"{stem}_manifest.csv",
                        "--folds-csv", f"{stem}_folds.csv", "--name", name,
                        "--num-workers", str(NUM_WORKERS)])
    print(f"[{name}] exit {r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
"""))
cells.append(code(r"""
# ---- pilot read-out: one decision, one number ----
import pandas as pd, glob, os

rows = {}
for f in glob.glob(f"{DRIVE_RESULTS}/predictions_*_S*.csv"):
    name = os.path.basename(f).replace("predictions_", "").replace("_fold0_seed0.csv", "")
    d = pd.read_csv(f)
    rows[name] = d.dice.mean()
for k in sorted(rows):
    print(f"  {k:22s} Dice {rows[k]:.4f}")

single = rows.get("single_S", float("nan"))
oracle = rows.get("early_S_iou100", float("nan"))
head = oracle - single
print(f"\nheadroom  Oracle(IoU 1.0) - single = {head:+.4f}")
if head >= 0.03:
    print("  >= 0.03  -> complementary information is strong enough; run the full sweep")
else:
    print("  <  0.03  -> too weak. Raise --blur or lower --block, re-run THIS pilot,")
    print("              do not start the 44-run sweep")

if "early_S_iou028" in rows:
    print(f"\nmisalignment damage  early(0.28) - Oracle(1.0) = "
          f"{rows['early_S_iou028'] - oracle:+.4f}")
if "ours_S_iou028" in rows and "early_S_iou028" in rows:
    print(f"registration-free gain  ours - early @0.28  = "
          f"{rows['ours_S_iou028'] - rows['early_S_iou028']:+.4f}   (first signal for P2)")
"""))

cells.append(md(r"""
## 8/17 archive checklist

- [ ] every prediction CSV and grade CSV committed to the repository
- [ ] `results/gate_report.csv` committed
- [ ] `results/exemplars/` committed — masks and index; not regenerable without a GPU
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
