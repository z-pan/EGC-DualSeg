# -*- coding: utf-8 -*-
"""Generate notebooks/colab_pilot.ipynb — a standalone notebook for the
controlled-misalignment pilot.

Deliberately separate from colab_train.ipynb. That one now carries five
different jobs (segmentation queue, gate diagnostics, ablation, exemplar dump,
Stage 2) and running the pilot from it means knowing which cells to skip. This
one runs top to bottom with nothing to skip, touches no gastric-cancer data,
and does not use the results/ symlink that caused the nested-directory failure.
"""
import json
import os

OUT = r"C:\Users\zpanp\projects\EGC-DualSeg\notebooks\colab_pilot.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


cells = []

cells.append(md(r"""
# 受控错位实验 · PILOT

**独立 notebook，从上到下依次运行，不需要跳过任何 cell。**
不读取胃癌数据，不需要 checkpoints，结果单独写入 `results_pilot/`，与主实验互不干扰。

---

### 这个 pilot 要回答的唯一问题

真实队列只给出了「错位 ↔ 融合损害」关系上的**一个点**（病灶 IoU 0.28，早融合 −0.088）。
一个点不构成关系。本实验在有真值掩膜的 Kvasir-SEG 上**把错位量调到指定值**，
从而把曲线画出来，再把真实队列放回曲线上验证。

但在铺开全部 44 次训练之前，必须先确认一件事：

> **单模态基线** 与 **完美配准的 Oracle 上界** 之间，是否有足够的 headroom？

若 `Oracle − single < 0.03 Dice`，说明人为构造的「互补信息」太弱，
各个错位档会挤在一起，整个扫描没有分辨率，44 次训练就是白跑。
**那种情况下应调大 `--blur` / 调小 `--block` 后重跑本 pilot，而不是继续铺开。**

### 设计上最要命的一点

如果第二个视图只是第一个视图的几何变换，它在信息论上是**冗余**的
（`I(aux; mask | ref) = 0`），融合就只可能有害、不可能有益——
曲线中「有益」的那一半、以及我们要找的临界点，都不会存在。
所以第二视图必须携带第一视图缺失的信息。本 pilot 用 **S 型（空间定位型）互补**：
参考视图与辅助视图各自保留互补的随机分块，另一半被模糊+降对比度处理，
完整边界只有把两者**在正确的空间位置上**结合才能恢复。

### 预计耗时

数据生成 3–5 分钟 + 4 次训练约 20 分钟，**合计约 25 分钟**。
"""))

cells.append(md("## 1 — 运行环境\n\n确认拿到 GPU。没有 GPU 就先去「代码执行程序 → 更改运行时类型」里改。"))
cells.append(code(r"""
import os, sys, platform, subprocess

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "没有 GPU：运行时 → 更改运行时类型 → GPU"

NUM_WORKERS = 2 if os.cpu_count() <= 4 else 6
print("vCPU", os.cpu_count(), "| NUM_WORKERS", NUM_WORKERS)
"""))

cells.append(md("## 2 — 挂载 Drive\n\n只用来存结果，防止 session 掉线丢失。不读取任何胃癌数据。"))
cells.append(code(r"""
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT   = "/content/drive/MyDrive/EGC-DualSeg"
REPO         = "/content/EGC-DualSeg"
PILOT_OUT    = f"{DRIVE_ROOT}/results_pilot"      # 与主实验的 results/ 分开
PILOT_CKPT   = "/content/ckpt_pilot"              # pilot 的权重不必保留，放本地盘
os.makedirs(PILOT_OUT, exist_ok=True)
os.makedirs(PILOT_CKPT, exist_ok=True)
print("结果将写入:", PILOT_OUT)
"""))

cells.append(md("## 3 — 拉取代码"))
cells.append(code(r"""
if not os.path.isdir(REPO):
    !git clone --quiet https://github.com/z-pan/EGC-DualSeg.git {REPO}
else:
    !git -C {REPO} pull --quiet

os.chdir(REPO)
sys.path.insert(0, REPO)
!git -C {REPO} log --oneline -1
"""))

cells.append(code(r"""
!pip install --quiet pyyaml pandas
import yaml, pandas
print("pyyaml", yaml.__version__, "| pandas", pandas.__version__)
"""))

cells.append(md(r"""
## 4 — 取得 Kvasir-SEG

按三条路依次尝试，每一步都会打印结果：

1. Colab 上已解压的目录（递归查找 `images/` + `masks/`，因为压缩包解出的目录名不固定）
2. **Drive 上的 500 张子集包** `kvasir_subset_500.zip`（22 MB，最可靠）
3. 官方公开下载（会打印退出码与实际字节数，失败时能看出是 URL 失效还是没网）

三条都失败会直接报错并告诉你要上传哪个文件。
"""))
cells.append(code(r"""
def find_kvasir(root="/content"):
    # 压缩包解出的目录名不一定叫 Kvasir-SEG，写死路径会静默失败
    for d, _subs, _files in os.walk(root):
        if os.path.basename(d) == "images":
            sib = os.path.join(os.path.dirname(d), "masks")
            if os.path.isdir(sib) and len(os.listdir(d)) > 100:
                return os.path.dirname(d)
    return None

KV = find_kvasir()
if KV:
    print("已存在:", KV)

if KV is None:
    subset = f"{DRIVE_ROOT}/kvasir_subset_500.zip"
    if os.path.isfile(subset):
        print("使用 Drive 子集包 ...")
        subprocess.run(["unzip", "-q", "-o", subset, "-d", "/content/"], check=False)
        KV = find_kvasir()

if KV is None:
    z = "/content/kv.zip"
    for url in ("https://datasets.simula.no/downloads/kvasir-seg.zip",
                "https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip"):
        r = subprocess.run(["wget", "--no-check-certificate", "-O", z, url],
                           capture_output=True, text=True)
        size = os.path.getsize(z) if os.path.exists(z) else 0
        print(f"{url}\n   exit {r.returncode}, 下载 {size/1e6:.2f} MB")
        if size > 10e6:
            subprocess.run(["unzip", "-q", "-o", z, "-d", "/content/"], check=False)
            KV = find_kvasir()
            if KV:
                break
        elif r.stderr:
            print("   ", r.stderr.strip().splitlines()[-2:])

assert KV, (f"Kvasir-SEG 获取失败。请把本地的 kvasir_subset_500.zip 上传到 "
            f"{DRIVE_ROOT}/ 后重跑本 cell。")
print("\nKvasir:", KV, "|", len(os.listdir(os.path.join(KV, "images"))), "张")
"""))

cells.append(md(r"""
## 5 — 生成合成数据

两档：**IoU 1.0**（完美配准，Oracle 上界）与 **IoU 0.28**（真实队列实测值）。

错位量是**逐图二分搜索**求解的，不是施加固定位移——同样的位移对不同大小/形状的病灶
造成的 IoU 变化差异很大，用固定位移会让每一档糊成一片宽分布，档位就失去意义。
留意输出里的 `achieved IoU median`，各档应精确命中目标且区间很窄（约 ±0.02）。
"""))
cells.append(code(r"""
!python scripts/make_synthetic.py --kvasir {KV} \
    --out data/synth --iou 1.0 0.28 --complement S --n 500
"""))

cells.append(md(r"""
## 6 — 训练 4 次

| 运行 | 含义 |
|---|---|
| `single_S` | 单模态基线（只用参考视图，与错位档无关） |
| `early_S_iou100` | 朴素早融合 @ 完美配准 → **Oracle 上界** |
| `early_S_iou028` | 朴素早融合 @ IoU 0.28 → 错位造成的损害 |
| `ours_S_iou028` | registration-free 融合 @ IoU 0.28 |

约 20 分钟。断了重跑本 cell 即可——已完成的运行会被自动跳过。
"""))
cells.append(code(r"""
import time

PILOT = [
    ("configs/synth_single.yaml", "iou100", "single_S"),
    ("configs/synth_early.yaml",  "iou100", "early_S_iou100"),
    ("configs/synth_early.yaml",  "iou028", "early_S_iou028"),
    ("configs/synth_ours.yaml",   "iou028", "ours_S_iou028"),
]
for cfg, lvl, name in PILOT:
    stem = f"data/synth/synth_S_{lvl}"
    t0 = time.time()
    r = subprocess.run(["python", "scripts/train.py", "--config", cfg,
                        "--npz", f"{stem}.npz", "--manifest", f"{stem}_manifest.csv",
                        "--folds-csv", f"{stem}_folds.csv", "--name", name,
                        "--out-dir", PILOT_OUT, "--ckpt-dir", PILOT_CKPT,
                        "--num-workers", str(NUM_WORKERS)])
    print(f"[{name}] exit {r.returncode} | {(time.time()-t0)/60:.1f} min", flush=True)
"""))

cells.append(md(r"""
## 7 — 判读

只看一个数：**headroom = Oracle(IoU 1.0) − 单模态**。
"""))
cells.append(code(r"""
import pandas as pd, glob

rows = {}
for f in glob.glob(f"{PILOT_OUT}/predictions_*_S*.csv"):
    name = os.path.basename(f).replace("predictions_", "").replace("_fold0_seed0.csv", "")
    rows[name] = pd.read_csv(f).dice.mean()

for k in sorted(rows):
    print(f"  {k:22s} Dice {rows[k]:.4f}")

single = rows.get("single_S")
oracle = rows.get("early_S_iou100")
if single is None or oracle is None:
    print("\n缺少基线或 Oracle 的结果，检查上一个 cell 是否全部成功")
else:
    head = oracle - single
    print(f"\nheadroom = Oracle(IoU 1.0) - single = {head:+.4f}")
    if head >= 0.03:
        print("  >= 0.03  ✅ 互补信息强度合适 —— 可以铺开全部 44 次扫描")
    else:
        print("  <  0.03  ❌ 互补信息太弱：各档会挤在一起，扫描没有分辨率")
        print("           调大 --blur（如 6）或调小 --block（如 16）后重跑本 pilot")

    if "early_S_iou028" in rows:
        print(f"\n错位损害   early(0.28) - Oracle(1.0) = {rows['early_S_iou028']-oracle:+.4f}")
        print("  为负且明显 -> 预测 P1 成立：早融合的损害随错位增大")
    if "ours_S_iou028" in rows and "early_S_iou028" in rows:
        d = rows["ours_S_iou028"] - rows["early_S_iou028"]
        print(f"registration-free  ours - early @0.28 = {d:+.4f}")
        print("  为正 -> 预测 P2 的第一个信号：该架构确实更抗错位")
"""))

cells.append(md(r"""
---

## 跑完之后

把本 notebook 第 1 节与第 7 节的输出贴回对话即可。需要重点看：

- `achieved IoU median` —— 确认合成在 Colab 上与本地一致
- `headroom` —— 决定是铺开扫描，还是回去调互补信息强度

pilot 结果在 `MyDrive/EGC-DualSeg/results_pilot/`，与主实验的 `results/` 完全分开。
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
      f"({sum(c['cell_type'] == 'code' for c in back['cells'])} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in back['cells'])} markdown)")
