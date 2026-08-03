# -*- coding: utf-8 -*-
"""Training and evaluation loop for one (config, fold, seed)."""
from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EGCPairDataset, collate
from src.data.transforms import PairAugment
from src.losses.seg import DiceBCELoss, binary_metrics, longest_axis_px
from src.models.net import EGCNet


@dataclass
class RunConfig:
    name: str = "ours"
    mode: str = "dual"                 # single | early | dual
    reference: str = "random"          # WLI | NBI | random
    npz: str = "data/packaged/egc_dualseg_384.npz"
    manifest: str = "data/packaged/manifest.csv"
    folds: str = "configs/folds.csv"
    out_dir: str = "results"
    ckpt_dir: str = "checkpoints"
    fold: int = 0
    seed: int = 0
    epochs: int = 80
    patience: int = 15
    batch_size: int = 8
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    amp: bool = True
    pretrained: bool = True
    use_role_embedding: bool = True
    aux_levels: tuple[int, ...] = (3, 4)
    init_from: str = ""                # e.g. a Kvasir-pretrained checkpoint
    # Oracle headroom probe only. `oracle_align` warps the auxiliary frame onto
    # the reference using the ground-truth masks of both, so anything trained
    # with it is an upper bound and not a result. The other two exist because
    # alignment is useless without them: the skips are the information path
    # being measured, and an independent auxiliary jitter would undo the
    # alignment before the model ever saw it.
    oracle_align: bool = False          # deprecated alias for align_mode='oracle'
    align_mode: str = "none"            # none | predicted | oracle
    predmask_npz: str = ""              # {fold}/{seed} placeholders are filled in
    aux_content: str = "self"           # self | shuffled
    aux_fill: str = "edge"              # edge | zero
    aux_skips: bool = False
    share_aux_geometry: bool = False
    extra: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: RunConfig) -> tuple[DataLoader, DataLoader]:
    # The predicted masks are per (fold, seed): the model that segments a patient
    # must be the one that held them out, or the alignment quietly sees the answer.
    predmask = cfg.predmask_npz.format(fold=cfg.fold, seed=cfg.seed) \
        if cfg.predmask_npz else ""
    align = dict(align_mode=cfg.align_mode, oracle_align=cfg.oracle_align,
                 predmask_npz=predmask, aux_content=cfg.aux_content,
                 aux_fill=cfg.aux_fill)
    train_ds = EGCPairDataset(cfg.npz, cfg.manifest, cfg.folds, cfg.fold, "train",
                              mode=cfg.mode, reference=cfg.reference,
                              transform=PairAugment(
                                  train=True, seed=cfg.seed,
                                  share_aux_geometry=cfg.share_aux_geometry),
                              seed=cfg.seed, **align)
    val_ds = EGCPairDataset(cfg.npz, cfg.manifest, cfg.folds, cfg.fold, "val",
                            mode=cfg.mode, reference=cfg.reference,
                            transform=PairAugment(train=False), seed=cfg.seed,
                            **align)
    common = dict(collate_fn=collate, num_workers=cfg.num_workers,
                  pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                       drop_last=len(train_ds) > cfg.batch_size, **common),
            DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common))


def build_model(cfg: RunConfig, device: torch.device) -> EGCNet:
    model = EGCNet(mode=cfg.mode, pretrained=cfg.pretrained,
                   use_role_embedding=cfg.use_role_embedding,
                   aux_levels=tuple(cfg.aux_levels), aux_skips=cfg.aux_skips)
    if cfg.init_from and os.path.isfile(cfg.init_from):
        state = torch.load(cfg.init_from, map_location="cpu")
        state = state.get("model", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  init_from {cfg.init_from}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    return model.to(device)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def run_epoch(model, loader, criterion, device, optimiser=None, scaler=None) -> float:
    train = optimiser is not None
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        aux = batch.get("aux")
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,
                                enabled=scaler is not None and device.type == "cuda"):
                out = model(batch["ref"], aux, batch["ref_id"])
                loss = criterion(out["logits"], batch["mask"])
        if train:
            optimiser.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
        total += loss.item() * batch["ref"].shape[0]
        n += batch["ref"].shape[0]
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, list[dict]]:
    model.eval()
    rows, dices = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch["ref"], batch.get("aux"), batch["ref_id"])
        m = binary_metrics(out["logits"], batch["mask"])
        axis = longest_axis_px(out["logits"])
        for i in range(batch["ref"].shape[0]):
            rows.append(dict(
                case=batch["case"][i], scale=batch["scale"][i],
                ref_modality=batch["ref_modality"][i],
                dice=float(m["dice"][i]), iou=float(m["iou"][i]),
                precision=float(m["precision"][i]), recall=float(m["recall"][i]),
                pred_area_frac=float(m["pred_area_frac"][i]),
                gt_area_frac=float(m["gt_area_frac"][i]),
                pred_long_axis_px=float(axis[i]),
                content_w=int(batch["content_w"][i]),
            ))
            dices.append(float(m["dice"][i]))
    return (sum(dices) / max(1, len(dices))), rows


def train_one(cfg: RunConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg, device)
    criterion = DiceBCELoss()

    backbone_ids = {id(p) for p in model.encoder.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if id(p) in backbone_ids],
         "lr": cfg.backbone_lr},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
         "lr": cfg.lr},
    ]
    optimiser = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None

    tag = f"{cfg.name}_fold{cfg.fold}_seed{cfg.seed}"
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{tag}.pt")

    best, best_epoch, history = -math.inf, -1, []
    print(f"[{tag}] train {len(train_loader.dataset)} / val {len(val_loader.dataset)} "
          f"| device {device.type}")
    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, criterion, device, optimiser, scaler)
        val_dice, _ = evaluate(model, val_loader, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_dice=val_dice))
        if val_dice > best:
            best, best_epoch = val_dice, epoch
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "epoch": epoch, "val_dice": val_dice}, ckpt_path)
        print(f"  epoch {epoch:3d}  loss {tr_loss:.4f}  val_dice {val_dice:.4f}"
              f"{'  *' if epoch == best_epoch else ''}  ({time.time() - t0:.1f}s)")
        if epoch - best_epoch >= cfg.patience:
            print(f"  early stop at epoch {epoch} (best {best:.4f} @ {best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    _, rows = evaluate(model, val_loader, device)
    pred_path = os.path.join(cfg.out_dir, f"predictions_{tag}.csv")
    fields = ["case", "scale", "ref_modality", "config", "fold", "seed", "dice", "iou",
              "precision", "recall", "pred_area_frac", "gt_area_frac",
              "pred_long_axis_px", "content_w"]
    with open(pred_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            r.update(config=cfg.name, fold=cfg.fold, seed=cfg.seed)
            writer.writerow({k: r[k] for k in fields})

    log_path = os.path.join(cfg.out_dir, "logs", f"{tag}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_dice"])
        writer.writeheader()
        writer.writerows(history)

    print(f"[{tag}] best val Dice {best:.4f} @ epoch {best_epoch} -> {pred_path}")
    return dict(tag=tag, best_dice=best, best_epoch=best_epoch,
                predictions=pred_path, checkpoint=ckpt_path)
