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

`oracle_align` suspends that premise on purpose. It warps the auxiliary frame
onto the reference frame using the ground-truth masks of *both*, which is
cheating, and exists only to put a ceiling on what an aligned auxiliary stream
could be worth (see `align.py`). Nothing produced with it is reportable.
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
                 reference: str = "WLI", transform=None, seed: int = 0,
                 oracle_align: bool = False, align_mode: str = "none",
                 predmask_npz: str = "", aux_content: str = "self",
                 aux_fill: str = "edge"):
        """
        fold         : index of the held-out fold
        split        : 'train' (all other folds) or 'val' (this fold)
        mode         : 'single' | 'early' | 'dual' | 'mid'
                       'mid' feeds the same two frames as 'dual'; the two differ
                       only inside the model, at the fusion operator.
        reference    : 'WLI' | 'NBI' | 'random'   ('random' only meaningful in training)
        align_mode   : 'none'      — the auxiliary frame is left where it was
                       'predicted' — aligned by the 3-parameter fit on the two
                                     single-modality PREDICTED masks. This is the
                                     designed pipeline and is honest.
                       'oracle'    — aligned by the 6-parameter fit on the two
                                     GROUND-TRUTH masks. Cheating; upper bound only.
        predmask_npz : required by align_mode='predicted'; written by
                       scripts/predict_masks.py for this (fold, seed)
        aux_content  : 'self'     — this patient's auxiliary frame
                       'shuffled' — ANOTHER patient's, warped by this pair's own
                                    transform. The control that separates "the
                                    second modality helped" from "the warp's own
                                    geometry gave the lesion away".
        aux_fill     : what warping leaves outside the source; see align.warp_affine
        oracle_align : deprecated alias for align_mode='oracle'
        """
        assert split in {"train", "val"}
        assert mode in {"single", "early", "dual", "mid"}
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
        if oracle_align and align_mode == "none":
            align_mode = "oracle"
        assert align_mode in {"none", "predicted", "oracle"}
        assert aux_content in {"self", "shuffled"}
        self.align_mode, self.aux_content, self.aux_fill = align_mode, aux_content, aux_fill
        self.align_iou: dict[tuple[str, str, str], float] = {}
        self._affines: dict[tuple[str, str, str], np.ndarray] = {}
        # (pair index, reference modality) -> warped auxiliary frame, filled on
        # first use. The affines themselves are computed once: the oracle search
        # is far too slow to repeat per run, let alone per dataloader worker.
        self._aligned: dict[tuple[int, str], np.ndarray] = {}

        if align_mode != "none" and mode != "single":
            from src.data.align import build_affines, load_or_build_affines
            if align_mode == "predicted":
                if not predmask_npz or not os.path.isfile(predmask_npz):
                    raise FileNotFoundError(
                        f"align_mode='predicted' needs {predmask_npz!r}.\n"
                        "  python scripts/predict_masks.py --fold "
                        f"{fold} --seeds {seed}")
                source = np.load(predmask_npz)["pred"]
                self._affines, self.align_iou = build_affines(
                    source, all_pairs, method="analytic")
                label = "predicted-mask (3-parameter)"
            else:
                stem = os.path.splitext(os.path.basename(npz_path))[0]
                cache = os.path.join(os.path.dirname(npz_path),
                                     f"oracle_affines_{stem}.npz")
                # Built over every pair, not just this split's: train and val are
                # separate instances and would otherwise invalidate each other's
                # cache and re-run the search.
                self._affines, self.align_iou = load_or_build_affines(
                    cache, self.masks, all_pairs, method="optimal")
                label = "ORACLE (ground-truth masks — upper bound, not a result)"
            reached = np.array(list(self.align_iou.values()))
            print(f"  alignment [{label}]: median overlap {np.median(reached):.3f} "
                  f"over {len(reached)} directions (unaligned baseline 0.28); "
                  f"fill={aux_fill}, aux content={aux_content}")

        # Which pair lends its auxiliary frame. A derangement, so no pair keeps
        # its own; fixed by seed so the control is reproducible.
        self._partner = list(range(len(self.pairs)))
        if aux_content == "shuffled" and len(self.pairs) > 1:
            rng = np.random.default_rng([seed, fold, 0 if split == "train" else 1])
            order = np.arange(len(self.pairs))
            for _ in range(64):
                perm = rng.permutation(len(self.pairs))
                if not np.any(perm == order):
                    break
            else:
                perm = np.roll(order, 1)
            self._partner = perm.tolist()

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

    def _aux_source(self, pair: Pair, aux: str) -> int:
        """Row of the auxiliary image to warp — this pair's, or a partner's."""
        if self.aux_content == "self":
            return pair.idx[aux]
        position = self.pairs.index(pair)
        return self.pairs[self._partner[position]].idx[aux]

    def _align_aux(self, i: int, pair: Pair, ref: str, aux: str) -> np.ndarray:
        """Warp the auxiliary frame into the reference frame.

        The transform always belongs to THIS pair, even when the content comes
        from another patient: the control has to hold the warp's geometry fixed
        and vary only what is being warped, or it measures nothing.
        """
        from src.data.align import warp_affine

        key = (i, ref)
        if key not in self._aligned:
            a_inv = self._affines[(pair.case, pair.scale, ref)]
            self._aligned[key] = warp_affine(self.images[self._aux_source(pair, aux)],
                                             a_inv, fill=self.aux_fill)
        return self._aligned[key]

    def __getitem__(self, i: int) -> dict:
        pair, fixed_ref = self.items[i]
        ref = self._pick_reference(fixed_ref)
        aux = MODALITIES[1 - MODALITIES.index(ref)]

        ref_img = self.images[pair.idx[ref]]
        ref_msk = self.masks[pair.idx[ref]] > 127
        aux_img = self.images[pair.idx[aux]]
        if self.align_mode != "none" and self.mode != "single":
            aux_img = self._align_aux(i, pair, ref, aux)
        elif self.aux_content == "shuffled" and self.mode != "single":
            aux_img = self.images[self._aux_source(pair, aux)]

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
