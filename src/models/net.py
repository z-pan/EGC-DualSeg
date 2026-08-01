# -*- coding: utf-8 -*-
"""Dual-stream segmentation network with registration-free cross-attention.

Design constraints, all forced by a 77-patient cohort and by the measured
misalignment between the two illumination modes:

* One ResNet-34 backbone, shared by both streams, with a small per-modality FiLM
  adapter at each stage. Two full encoders would roughly double the parameter
  count without adding capacity this cohort can support.
* The auxiliary stream contributes keys and values at TWO pyramid levels, because
  the lesion occupies a systematically larger share of the field in the enhanced
  modality (measured median ratio 1.9x). A single-scale key/value bank would have
  to absorb that offset implicitly.
* Skip connections come from the REFERENCE stream only. Auxiliary features are not
  spatially aligned with the target mask, so routing them into the decoder would
  inject misalignment noise directly into the output. This fails silently.
* A reference-role embedding is added to the query tokens, so one model serves
  both fusion directions instead of training two.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


# ---------------------------------------------------------------------------
class FiLM(nn.Module):
    """Per-modality channel-wise scale and shift."""

    def __init__(self, channels: int, n_modalities: int = 2):
        super().__init__()
        self.gamma = nn.Embedding(n_modalities, channels)
        self.beta = nn.Embedding(n_modalities, channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, modality_id: torch.Tensor) -> torch.Tensor:
        g = self.gamma(modality_id)[:, :, None, None]
        b = self.beta(modality_id)[:, :, None, None]
        return x * g + b


class SharedEncoder(nn.Module):
    """ResNet-34 trunk shared across modalities, with FiLM adapters per stage."""

    STAGE_CHANNELS = (64, 64, 128, 256, 512)

    def __init__(self, in_channels: int = 3, pretrained: bool = True,
                 n_modalities: int = 2):
        super().__init__()
        net = resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        if in_channels != 3:
            old = net.conv1
            net.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                repeats = (in_channels + 2) // 3
                net.conv1.weight.copy_(
                    old.weight.repeat(1, repeats, 1, 1)[:, :in_channels] * (3.0 / in_channels))
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.films = nn.ModuleList([FiLM(c, n_modalities) for c in self.STAGE_CHANNELS])

    def forward(self, x: torch.Tensor, modality_id: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.films[0](self.stem(x), modality_id)          # 1/2
        f1 = self.films[1](self.layer1(self.pool(f0)), modality_id)   # 1/4
        f2 = self.films[2](self.layer2(f1), modality_id)       # 1/8
        f3 = self.films[3](self.layer3(f2), modality_id)       # 1/16
        f4 = self.films[4](self.layer4(f3), modality_id)       # 1/32
        return [f0, f1, f2, f3, f4]


# ---------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """Reference tokens attend over a multi-scale bank of auxiliary tokens.

    Attention over a concatenated token bank is permutation invariant, so the
    module never assumes a spatial correspondence between the two frames.
    """

    def __init__(self, dim: int = 512, aux_dims: tuple[int, ...] = (256, 512),
                 heads: int = 8, n_modalities: int = 2, dropout: float = 0.1):
        super().__init__()
        self.role = nn.Embedding(n_modalities, dim)
        nn.init.zeros_(self.role.weight)
        self.aux_proj = nn.ModuleList([nn.Conv2d(d, dim, 1) for d in aux_dims])
        self.scale_embed = nn.Parameter(torch.zeros(len(aux_dims), dim))
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))     # start as identity: fusion learns in
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, ref_feat: torch.Tensor, aux_feats: list[torch.Tensor],
                ref_id: torch.Tensor) -> torch.Tensor:
        b, c, h, w = ref_feat.shape
        q = ref_feat.flatten(2).transpose(1, 2)                    # (B, HW, C)
        q = q + self.role(ref_id)[:, None, :]

        banks = []
        for i, (proj, feat) in enumerate(zip(self.aux_proj, aux_feats)):
            t = proj(feat).flatten(2).transpose(1, 2)
            banks.append(t + self.scale_embed[i][None, None, :])
        kv = torch.cat(banks, dim=1)                               # (B, sum(HW), C)

        attended, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                                need_weights=False)
        q = q + torch.tanh(self.gate) * attended
        q = q + self.ffn(q)
        return q.transpose(1, 2).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetDecoder(nn.Module):
    """Skips are taken from the reference stream only, by design.

    `aux_skips=True` breaks that rule and is only legitimate when the auxiliary
    frame has been spatially aligned to the reference: each skip is then the
    reference features concatenated with the aligned auxiliary features at the
    same resolution. Unaligned, this injects misalignment noise directly into
    the decoder and degrades silently — which is why it is off by default and
    why the only config that turns it on is the oracle headroom probe.
    """

    def __init__(self, enc_channels=(64, 64, 128, 256, 512), widths=(256, 128, 64, 32, 16),
                 aux_skips: bool = False):
        super().__init__()
        c0, c1, c2, c3, c4 = enc_channels
        w0, w1, w2, w3, w4 = widths
        self.aux_skips = aux_skips
        m = 2 if aux_skips else 1
        self.up4 = DecoderBlock(c4, c3 * m, w0)
        self.up3 = DecoderBlock(w0, c2 * m, w1)
        self.up2 = DecoderBlock(w1, c1 * m, w2)
        self.up1 = DecoderBlock(w2, c0 * m, w3)
        self.up0 = DecoderBlock(w3, 0, w4)
        self.head = nn.Conv2d(w4, 1, 1)

    def forward(self, feats: list[torch.Tensor], bottleneck: torch.Tensor,
                aux_feats: list[torch.Tensor] | None = None) -> torch.Tensor:
        def skip(level: int) -> torch.Tensor:
            f = feats[level]
            if not self.aux_skips:
                return f
            if aux_feats is None:
                raise ValueError("aux_skips is on but no auxiliary features were given")
            a = aux_feats[level]
            if a.shape[-2:] != f.shape[-2:]:
                a = F.interpolate(a, size=f.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([f, a], dim=1)

        x = self.up4(bottleneck, skip(3))
        x = self.up3(x, skip(2))
        x = self.up2(x, skip(1))
        x = self.up1(x, skip(0))
        x = self.up0(x, None)
        return self.head(x)


# ---------------------------------------------------------------------------
class EGCNet(nn.Module):
    """mode: 'single' | 'early' | 'dual'."""

    def __init__(self, mode: str = "dual", pretrained: bool = True,
                 heads: int = 8, dropout: float = 0.1,
                 use_role_embedding: bool = True, aux_levels: tuple[int, ...] = (3, 4),
                 aux_skips: bool = False):
        super().__init__()
        assert mode in {"single", "early", "dual"}
        if aux_skips and mode != "dual":
            raise ValueError("aux_skips needs a separate auxiliary stream (mode='dual')")
        self.mode = mode
        self.use_role_embedding = use_role_embedding
        self.aux_levels = aux_levels
        self.aux_skips = aux_skips

        in_ch = 6 if mode == "early" else 3
        self.encoder = SharedEncoder(in_channels=in_ch, pretrained=pretrained)
        self.decoder = UNetDecoder(aux_skips=aux_skips)
        if mode == "dual":
            dims = tuple(SharedEncoder.STAGE_CHANNELS[i] for i in aux_levels)
            self.fusion = CrossAttentionFusion(dim=512, aux_dims=dims,
                                               heads=heads, dropout=dropout)
        else:
            self.fusion = None

    def encode_reference(self, ref: torch.Tensor, ref_id: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(ref, ref_id)

    def forward(self, ref: torch.Tensor, aux: torch.Tensor | None = None,
                ref_id: torch.Tensor | None = None) -> dict:
        if ref_id is None:
            ref_id = torch.zeros(ref.shape[0], dtype=torch.long, device=ref.device)
        feats = self.encode_reference(ref, ref_id)
        bottleneck = feats[-1]

        aux_feats_all = None
        if self.mode == "dual":
            assert aux is not None, "dual mode requires the auxiliary frame"
            aux_id = 1 - ref_id
            aux_feats_all = self.encoder(aux, aux_id)
            aux_feats = [aux_feats_all[i] for i in self.aux_levels]
            role = ref_id if self.use_role_embedding else torch.zeros_like(ref_id)
            bottleneck = self.fusion(bottleneck, aux_feats, role)

        logits = self.decoder(feats, bottleneck,
                              aux_feats_all if self.aux_skips else None)
        if logits.shape[-2:] != ref.shape[-2:]:
            logits = F.interpolate(logits, size=ref.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return {"logits": logits, "embedding": bottleneck}


class GradeHead(nn.Module):
    """Stage 2. Trained on frozen Stage-1 features; 48 patients cannot fine-tune a trunk."""

    def __init__(self, dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, embedding: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """region: (B,1,H,W) soft or binary lesion mask, resampled to the embedding grid."""
        m = F.interpolate(region, size=embedding.shape[-2:], mode="bilinear",
                          align_corners=False)
        weight = m.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = (embedding * m).sum(dim=(2, 3)) / weight
        return self.net(pooled).squeeze(1)
