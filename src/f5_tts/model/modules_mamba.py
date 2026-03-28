"""
Mamba mixer blocks for HybridDiT.

Two concrete classes, same forward(x, mask) signature so HybridDiTBlock
can hot-swap them with zero other changes.

  BidirectionalMambaSubBlock  (DEFAULT, recommended)
    – Forward SSM scan  + Backward SSM scan in parallel.
    – Full-sequence receptive field at O(L) cost.
    – Non-causal: essential for non-autoregressive TTS (F5 / E2 flow-matching).

  MambaSubBlock  (legacy, kept for ablation only)
    – Original causal left→right scan.
    – Do NOT use in early layers for TTS — causal scan breaks look-ahead
      coarticulation even during training where all frames are available.

Architecture notes
------------------
Bidirectional pattern (Vision Mamba, VMamba, SAMBA):
    fwd  = Mamba(x)                   # shape (B, T, D)
    bwd  = flip_T( Mamba(flip_T(x)) ) # shape (B, T, D) — reverse time, scan, re-reverse
    out  = Linear([fwd ‖ bwd], D)     # merge 2D → D  (no information bottleneck)

sinpos injection
----------------
An optional learnable-scaled sinusoidal positional bias is added to x
*before* the Mamba scan, giving the SSM positional awareness without
the RoPE head-splitting mechanics used by DiT.

    x_in = x + pos_scale * sinpos(T, D)

pos_scale starts at 0 (identity init, safe for fine-tuning from DiT checkpoints)
and is learned during training.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal positional bias helper
# ---------------------------------------------------------------------------

def _sinusoidal_pos(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Standard sinusoidal encoding, shape (1, T, D).
    Odd-dim models: last dim is cos-filled to avoid size mismatch.
    """
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    div = torch.pow(
        10000.0,
        torch.arange(0, half, device=device, dtype=torch.float32) / half,
    )
    enc = torch.zeros(seq_len, dim, device=device)
    enc[:, 0::2] = torch.sin(pos / div)
    enc[:, 1::2] = torch.cos(pos / div)
    return enc.unsqueeze(0)  # (1, T, D)


# ---------------------------------------------------------------------------
# Bidirectional Mamba block  (PRIMARY — use this)
# ---------------------------------------------------------------------------

class BidirectionalMambaSubBlock(nn.Module):
    """
    Non-causal Mamba mixer.

    Parameters
    ----------
    dim          : model dimension (d_model)
    d_state      : SSM state size N.  Recommend 64 for TTS (↑ from legacy 16).
                   Higher d_state = richer temporal memory at modest VRAM cost.
    d_conv       : local conv window.  4 → ~42 ms @ 24 kHz / hop=256.
    expand       : inner-dim multiplier (inner = expand * dim).
    dropout      : applied after projection.
    inject_sinpos: add learnable-scaled sinusoidal pos bias before scan.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        inject_sinpos: bool = True,
    ) -> None:
        super().__init__()

        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise ImportError(
                "mamba_ssm is required for BidirectionalMambaSubBlock.\n"
                "Install: pip install mamba-ssm --no-build-isolation"
            ) from exc

        # Two independent SSM scanners — fwd and bwd do NOT share weights.
        # Separate weights let each direction specialise (onset vs offset, etc.)
        self.fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # Merge projection: 2D → D, no bias (AdaLN already handles affine shift)
        self.merge = nn.Linear(dim * 2, dim, bias=False)

        # Learnable scale for sinusoidal positional bias
        # init=0 means identity at checkpoint load — safe for fine-tuning
        self.inject_sinpos = inject_sinpos
        if inject_sinpos:
            self.pos_scale = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,          # (B, T, D)
        mask: torch.Tensor | None = None,  # (B, T) bool, True = valid
    ) -> torch.Tensor:

        # Positional injection before scan
        if self.inject_sinpos:
            sinpos = _sinusoidal_pos(x.shape[1], x.shape[2], x.device)
            x = x + self.pos_scale * sinpos

        # Forward scan
        fwd_out = self.fwd(x)                        # (B, T, D)

        # Backward scan: flip time → scan → flip back
        bwd_out = self.bwd(x.flip(1)).flip(1)        # (B, T, D)

        # Merge
        out = self.merge(torch.cat([fwd_out, bwd_out], dim=-1))  # (B, T, D)
        out = self.dropout(out)

        # Zero out padding positions to keep gradients clean
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)

        return out


# ---------------------------------------------------------------------------
# Legacy causal block  (DO NOT USE in early layers for TTS)
# ---------------------------------------------------------------------------

class MambaSubBlock(nn.Module):
    """
    Original causal (left→right) Mamba block.
    Kept for ablation studies only.  See module docstring for why this
    is unsuitable as an early-layer mixer in non-autoregressive TTS.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        inject_sinpos: bool = False,  # not used, kept for API compat
    ) -> None:
        super().__init__()

        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise ImportError(
                "mamba_ssm is required for MambaSubBlock. "
                "Install: pip install mamba-ssm --no-build-isolation"
            ) from exc

        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.mamba(x)
        out = self.dropout(out)
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out
