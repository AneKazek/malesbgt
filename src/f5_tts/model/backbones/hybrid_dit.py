"""
Hybrid DiT backbone that keeps baseline DiT behavior by default and swaps selected
mixer sublayers with Mamba blocks behind feature flags.
"""

from __future__ import annotations

import torch
from torch import nn

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import AdaLayerNorm, FeedForward
from f5_tts.model.modules_mamba import MambaSubBlock


class HybridDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.mixer = MambaSubBlock(
            dim=dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        del rope
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        mix_out = self.mixer(norm, mask=mask)
        x = x + gate_msa.unsqueeze(1) * mix_out

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_out
        return x


class HybridDiT(DiT):
    def __init__(
        self,
        *,
        use_mamba: bool = False,
        mamba_layers: list[int] | None = None,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        capture_hidden_for_distill: bool = False,
        **kwargs,
    ):
        self._ff_mult = int(kwargs.get("ff_mult", 4))
        self.use_mamba = bool(use_mamba)
        self.mamba_layers = sorted(set(mamba_layers or []))
        self.mamba_d_state = int(mamba_d_state)
        self.mamba_d_conv = int(mamba_d_conv)
        self.mamba_expand = int(mamba_expand)

        super().__init__(**kwargs)

        if self.use_mamba and self.mamba_layers:
            self._replace_selected_layers_with_mamba()

        self.capture_hidden_for_distill = bool(capture_hidden_for_distill)
        self.last_hidden_states: list[torch.Tensor] | None = None

    def set_capture_hidden_for_distill(self, enabled: bool):
        self.capture_hidden_for_distill = bool(enabled)

    def _replace_selected_layers_with_mamba(self):
        depth = len(self.transformer_blocks)
        valid_layers = [idx for idx in self.mamba_layers if 0 <= idx < depth]
        for idx in valid_layers:
            base_block = self.transformer_blocks[idx]
            hybrid_block = HybridDiTBlock(
                dim=self.dim,
                ff_mult=self._ff_mult,
                dropout=float(getattr(base_block.attn.to_out[1], "p", 0.0)),
                mamba_d_state=self.mamba_d_state,
                mamba_d_conv=self.mamba_d_conv,
                mamba_expand=self.mamba_expand,
            )

            # Safe initialization: keep FFN and AdaLN branch from base block; only mixer is new.
            hybrid_block.attn_norm.load_state_dict(base_block.attn_norm.state_dict())
            hybrid_block.ff_norm.load_state_dict(base_block.ff_norm.state_dict())
            hybrid_block.ff.load_state_dict(base_block.ff.state_dict())
            self.transformer_blocks[idx] = hybrid_block

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        if cfg_infer:
            x_cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache, audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache, audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache, audio_mask=mask
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        hidden_states = [] if self.capture_hidden_for_distill else None
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)
            if hidden_states is not None:
                hidden_states.append(x)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        if hidden_states is not None:
            self.last_hidden_states = hidden_states
        else:
            self.last_hidden_states = None

        x = self.norm_out(x, t)
        return self.proj_out(x)


def copy_shared_weights(student: nn.Module, teacher: nn.Module):
    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()
    updated = {}
    for k, v in teacher_sd.items():
        if k in student_sd and student_sd[k].shape == v.shape:
            updated[k] = v
    student_sd.update(updated)
    student.load_state_dict(student_sd)
    return len(updated)


def load_partial_state_dict_safely(model: nn.Module, state_dict: dict[str, torch.Tensor]):
    model_sd = model.state_dict()
    updated = {}
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            updated[k] = v
    model_sd.update(updated)
    model.load_state_dict(model_sd)
    return len(updated)


def init_hybrid_from_teacher(student: HybridDiT, teacher: DiT):
    copied = copy_shared_weights(student, teacher)
    student.eval()
    teacher.eval()
    return copied
