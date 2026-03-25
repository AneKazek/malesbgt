from __future__ import annotations

import torch
from torch import nn


class MambaSubBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        try:
            from mamba_ssm import Mamba
        except Exception as exc:  # pragma: no cover - explicit runtime guidance
            raise ImportError(
                "mamba_ssm is required for MambaSubBlock. Install a compatible wheel for your torch/cuda/abi."
            ) from exc

        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        out = self.mamba(x)
        out = self.dropout(out)
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out
