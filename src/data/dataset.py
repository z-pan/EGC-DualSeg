# -*- coding: utf-8 -*-
"""Paired WLI/NBI dataset.

One sample is one (patient, working distance) pair. Which modality plays the
reference role is decided by the config, or sampled per iteration when the model
is trained bidirectionally.

Input modes
-----------
single  : only the reference frame is returned (WLI-only / NBI-only baselines)
early   : reference and auxiliary are stacked channel-wise (resize-concat baseline)
dual    : reference and auxiliary are returned separately (the proposed fusion)

The mask always belongs to the reference frame. There is no aligned mask for the
auxiliary frame — that is the whole premise of the project.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("WLI", "NBI")


@dataclass
class Pair:
    case: str
    scale: str
    idx: dict          # modality -> row index into the packaged arrays
    meta: dict         # manifest row of the WLI image (labels are per patient)


def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_pairs(manifest_path: str, folds_path: str) -> tuple[list[Pair], dict[str, int]]:
    rows = _read_csv(manifest_path)
    folds = {r["case"]: int(r["fold"]) for r in _read_csv(folds_path)}

    grouped: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["case"], r["scale"])
        entry = grouped.setdefault(key, {"idx": {}, "meta": r})
        entry["idx"][r["modality"]] = int(r["idx"])

    pairs = []
    for (case, scale), entry in sorted(grouped.items()):
        if set(entry["idx"]) != set(MODALITIES):
            continue                       # incomplete pair, already rare after packaging
        pairs.append(Pair(case=case, scale=scale, idx=entry["idx"], meta=entry["meta"]))
    return pairs, folds


class EGCPairDataset(Dataset):
    def __init__(self, npz_path: str, manifest_path: str, folds_path: str,
                 fold: int, split: str, mode: str = "dual",
                 reference: str = "WLI", transform=None, seed: int = 0):
        """
        fold      : index of the held-out fold
        split     : 'train' (all other folds) or 'val' (this fold)
        mode      : 'single' | 'early' | 'dual'
        reference : 'WLI' | 'NBI' | 'random'   ('random' only meaningful in training)
        """
        assert split in {"train", "val"}
        assert mode in {"single", "early", "dual"}
        assert reference in {"WLI", "NBI", "random"}

        blob = np.load(npz_path)
        self.images = blob["images"]        # (M, H, W, 3) uint8
        self.masks = blob["masks"]          # (M, H, W)    uint8
        all_pairs, folds = load_pairs(manifest_path, folds_path)

        keep = (lambda f: f == fold) if split == "val" else (lambda f: f != fold)
        self.pairs = [p for p in all_pairs if p.case in folds and keep(folds[p.case])]

        self.split, self.mode, self.reference = split, mode, reference
        self.transform = transform
        self.rng = np.random.default_rng(seed)

        # A held-out fold must be scored under a fixed reference, otherwise the
        # metric depends on a coin flip. Resolve 'random' to both directions.
        self.eval_directions = MODALITIES if reference == "random" else (reference,)
        if split == "val":
            self.items = [(p, ref) for p in self.pairs for ref in self.eval_directions]
        else:
            self.items = [(p, None) for p in self.pairs]

    def __len__(self) -> int:
        return len(self.items)

    def _pick_reference(self, fixed: str | None) -> str:
        if fixed is not None:
            return fixed
        if self.reference == "random":
            return MODALITIES[int(self.rng.integers(2))]
        return self.reference

    def __getitem__(self, i: int) -> dict:
        pair, fixed_ref = self.items[i]
        ref = self._pick_reference(fixed_ref)
        aux = MODALITIES[1 - MODALITIES.index(ref)]

        ref_img = self.images[pair.idx[ref]]
        ref_msk = self.masks[pair.idx[ref]] > 127
        aux_img = self.images[pair.idx[aux]]

        if self.transform is not None:
            ref_img, ref_msk, aux_img = self.transform(ref_img, ref_msk, aux_img)

        to_chw = lambda a: torch.from_numpy(
            np.ascontiguousarray(a.transpose(2, 0, 1))).float().div_(255.0)
        sample = {
            "mask": torch.from_numpy(np.ascontiguousarray(ref_msk)).float().unsqueeze(0),
            "case": pair.case, "scale": pair.scale,
            "ref_modality": ref,
            "content_w": int(pair.meta["content_w"]),
            "content_h": int(pair.meta["content_h"]),
        }
        if self.mode == "single":
            sample["ref"] = to_chw(ref_img)
        elif self.mode == "early":
            sample["ref"] = torch.cat([to_chw(ref_img), to_chw(aux_img)], dim=0)
        else:
            sample["ref"] = to_chw(ref_img)
            sample["aux"] = to_chw(aux_img)
        # reference-role flag: 0 for WLI, 1 for NBI
        sample["ref_id"] = torch.tensor(MODALITIES.index(ref), dtype=torch.long)
        return sample


def collate(batch: list[dict]) -> dict:
    out: dict = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return out
