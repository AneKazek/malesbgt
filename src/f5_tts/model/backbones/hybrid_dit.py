"""
HybridDiT — F5-TTS backbone with early-layer Bidirectional Mamba.

Design philosophy
-----------------
  Layers 0 … (mamba_end-1)   →  BidirectionalMambaSubBlock
    • Processes low-level acoustic features (energy, formants, phoneme identity)
    • Bidirectional SSM: full-sequence receptive field at O(L) cost
    • Sinusoidal positional bias injected before each SSM scan
    • No self-attention overhead in these layers

  Layers mamba_end … (depth-1)  →  DiTBlock (full self-attention + RoPE)
    • Builds global context: word prosody, speaker identity, text-audio alignment
    • RoPE provides relative position encoding across the full sequence
    • Flow-matching denoising is dominated by these layers

Why early Mamba / late DiT?
  - Mamba is an order-recurrent sequence model — perfect for *local* acoustic
    signal processing (formant transitions, coarticulation within ~100 ms).
  - DiT attention with RoPE is quadratic but globally aware — essential for
    sentence-level prosody, speaker style, and diffusion step conditioning.
  - Placing Mamba at the *end* of the stack would contaminate the globally-aware
    representation with a causal or local-only scan right before output projection.
  - Early Mamba → late DiT keeps the information flow clean:
      raw mel → local acoustic encode (Mamba) → global context (DiT) → output

Additional features inherited from malesbgt
  - Knowledge distillation (teacher-student output + hidden)
  - CTC auxiliary head
  - Accent adversarial training (GRL)
  - Checkpoint audit + non-strict load
  - capture_hidden_for_distill flag
"""

from __future__ import annotations

import torch
from torch import nn

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import AdaLayerNorm, FeedForward
from f5_tts.model.modules_mamba import BidirectionalMambaSubBlock, MambaSubBlock


# ---------------------------------------------------------------------------
# Single hybrid block  (replaces one DiTBlock in the stack)
# ---------------------------------------------------------------------------

class HybridDiTBlock(nn.Module):
    """
    Drop-in replacement for DiTBlock that uses a Mamba mixer instead of
    multi-head self-attention.

    The AdaLayerNorm + FeedForward path is identical to DiTBlock so pretrained
    FFN / AdaLN weights can be copied across during _replace_selected_layers().

    Parameters
    ----------
    use_bidi      : use BidirectionalMambaSubBlock (True) or legacy causal (False)
    inject_sinpos : inject sinusoidal positional bias before each SSM scan
                    (only active when use_bidi=True)
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_bidi: bool = True,
        inject_sinpos: bool = True,
    ) -> None:
        super().__init__()

        # --- Norm + modulation (identical to DiTBlock) ---
        self.attn_norm = AdaLayerNorm(dim)

        # --- Mamba mixer ---
        mixer_cls = BidirectionalMambaSubBlock if use_bidi else MambaSubBlock
        self.mixer = mixer_cls(
            dim=dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout,
            inject_sinpos=inject_sinpos if use_bidi else False,
        )

        # --- Feed-forward (identical to DiTBlock) ---
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,           # (B, T, D)
        t: torch.Tensor,           # (B, D)  timestep embedding
        mask: torch.Tensor | None = None,  # (B, T) bool
        rope=None,                 # passed from HybridDiT.forward, not used here
    ) -> torch.Tensor:
        # rope is generated for the full stack by DiT's RotaryEmbedding and
        # forwarded to every block uniformly.  DiTBlocks use it; Mamba blocks
        # ignore it (Mamba gets positional info via inject_sinpos instead).
        _ = rope  # explicit no-op — do NOT del, keep it readable

        # AdaLN modulation
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # Mamba mixer (bidi or causal)
        mix_out = self.mixer(norm, mask=mask)
        x = x + gate_msa.unsqueeze(1) * mix_out

        # Feed-forward
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_out

        return x


# ---------------------------------------------------------------------------
# HybridDiT — main backbone
# ---------------------------------------------------------------------------

class HybridDiT(DiT):
    """
    F5-TTS backbone: early layers are BidirectionalMambaSubBlock,
    late layers are standard DiTBlock (full self-attention + RoPE).

    All DiT kwargs are forwarded to the parent constructor so configs are
    backward-compatible.  Mamba-specific kwargs are consumed here.

    Key parameters
    --------------
    mamba_layers     : list[int] of layer indices to swap → Mamba.
                       Recommended: [0,1,2,3,4,5,6,7]  (early 8 of 22)
    use_bidi         : bidirectional Mamba (True) or legacy causal (False).
                       MUST be True for non-autoregressive TTS.
    inject_sinpos    : inject sinusoidal positional bias in Mamba blocks.
    mamba_d_state    : SSM state size.  64 recommended (↑ from legacy 16).
    capture_hidden_for_distill : collect hidden states for distillation loss.
    """

    def __init__(
        self,
        *,
        # --- Mamba-specific ---
        use_mamba: bool = False,
        mamba_layers: list[int] | None = None,
        use_bidi: bool = True,
        inject_sinpos: bool = True,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        # --- Distillation / aux heads ---
        capture_hidden_for_distill: bool = False,
        # --- Everything else → DiT ---
        **kwargs,
    ) -> None:
        # Store Mamba params before super().__init__ builds transformer_blocks
        self._ff_mult = int(kwargs.get("ff_mult", 4))
        self.use_mamba = bool(use_mamba)
        self.mamba_layers = sorted(set(mamba_layers or []))
        self.use_bidi = bool(use_bidi)
        self.inject_sinpos = bool(inject_sinpos)
        self.mamba_d_state = int(mamba_d_state)
        self.mamba_d_conv = int(mamba_d_conv)
        self.mamba_expand = int(mamba_expand)

        super().__init__(**kwargs)

        if self.use_mamba and self.mamba_layers:
            self._replace_selected_layers_with_mamba()

        self.capture_hidden_for_distill = bool(capture_hidden_for_distill)
        self.last_hidden_states: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Layer replacement
    # ------------------------------------------------------------------

    def _replace_selected_layers_with_mamba(self) -> None:
        """
        Swap selected DiTBlocks → HybridDiTBlocks in-place.

        Weight transfer strategy:
          AdaLN   → copied from base block  (keeps timestep-conditioning intact)
          FFN     → copied from base block  (keeps learned transformations)
          ff_norm → copied from base block
          mixer   → freshly initialised     (only the SSM is new)

        This means the model is a valid continuation of any pretrained DiT
        checkpoint with ~(mamba_layers count / depth) fraction of new weights.
        """
        depth = len(self.transformer_blocks)
        valid = [i for i in self.mamba_layers if 0 <= i < depth]

        for idx in valid:
            base = self.transformer_blocks[idx]
            dropout = float(getattr(base.attn.to_out[1], "p", 0.0))

            hybrid = HybridDiTBlock(
                dim=self.dim,
                ff_mult=self._ff_mult,
                dropout=dropout,
                mamba_d_state=self.mamba_d_state,
                mamba_d_conv=self.mamba_d_conv,
                mamba_expand=self.mamba_expand,
                use_bidi=self.use_bidi,
                inject_sinpos=self.inject_sinpos,
            )

            # Copy AdaLN, ff_norm, FFN weights — only mixer is initialised fresh
            hybrid.attn_norm.load_state_dict(base.attn_norm.state_dict())
            hybrid.ff_norm.load_state_dict(base.ff_norm.state_dict())
            hybrid.ff.load_state_dict(base.ff.state_dict())

            self.transformer_blocks[idx] = hybrid

    # ------------------------------------------------------------------
    # Distillation flag
    # ------------------------------------------------------------------

    def set_capture_hidden_for_distill(self, enabled: bool) -> None:
        self.capture_hidden_for_distill = bool(enabled)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
    ) -> torch.Tensor:
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)

        if cfg_infer:
            x_cond = self.get_input_embed(
                x, cond, text,
                drop_audio_cond=False, drop_text=False,
                cache=cache, audio_mask=mask,
            )
            x_uncond = self.get_input_embed(
                x, cond, text,
                drop_audio_cond=True, drop_text=True,
                cache=cache, audio_mask=mask,
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x, cond, text,
                drop_audio_cond=drop_audio_cond, drop_text=drop_text,
                cache=cache, audio_mask=mask,
            )

        # RoPE is generated once for the whole stack.
        # DiTBlocks use it; HybridDiTBlocks ignore it (receive positional
        # info via inject_sinpos inside BidirectionalMambaSubBlock).
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        hidden_states: list[torch.Tensor] | None = (
            [] if self.capture_hidden_for_distill else None
        )

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, t, mask=mask, rope=rope)

            if hidden_states is not None:
                hidden_states.append(x)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        self.last_hidden_states = hidden_states  # None or list[tensor]

        x = self.norm_out(x, t)
        return self.proj_out(x)


# ---------------------------------------------------------------------------
# Utility helpers (distillation / partial weight copy)
# ---------------------------------------------------------------------------

def copy_shared_weights(student: nn.Module, teacher: nn.Module) -> int:
    """
    Copy all weight tensors from teacher → student where key and shape match.
    Returns the number of tensors copied.
    """
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    updated = {k: v for k, v in t_sd.items() if k in s_sd and s_sd[k].shape == v.shape}
    s_sd.update(updated)
    student.load_state_dict(s_sd)
    return len(updated)


def load_partial_state_dict_safely(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> int:
    """
    Load state_dict into model, skipping keys with shape mismatches.
    Safe for loading DiT checkpoints into HybridDiT (Mamba layers differ).
    """
    m_sd = model.state_dict()
    updated = {k: v for k, v in state_dict.items() if k in m_sd and m_sd[k].shape == v.shape}
    m_sd.update(updated)
    model.load_state_dict(m_sd)
    return len(updated)


def init_hybrid_from_teacher(student: HybridDiT, teacher: DiT) -> int:
    """Convenience: copy all compatible weights from a pretrained DiT."""
    n = copy_shared_weights(student, teacher)
    student.eval()
    teacher.eval()
    return n
