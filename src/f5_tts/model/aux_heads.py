from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from f5_tts.model.utils_grl import grad_reverse


class CTCAuxHead(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.proj = nn.Linear(input_dim, self.vocab_size + 1)
        self.ctc = nn.CTCLoss(blank=self.vocab_size, zero_infinity=True)

    def forward(
        self,
        hidden: torch.Tensor,
        text: torch.Tensor,
        input_lengths: torch.Tensor,
        generated_mask: torch.Tensor | None = None,
    ):
        logits = self.proj(hidden)

        # TODO: generated-only CTC can use generated_mask; patch-1 falls back to full sequence CTC.
        if generated_mask is not None:
            del generated_mask

        valid = text >= 0
        target_lengths = valid.sum(dim=1).to(dtype=torch.long)
        if int(target_lengths.sum().item()) == 0:
            return logits.new_tensor(0.0)

        targets = text.masked_select(valid).to(dtype=torch.long)
        input_lengths = input_lengths.to(dtype=torch.long).clamp(min=1, max=logits.shape[1])

        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return self.ctc(log_probs.float(), targets, input_lengths, target_lengths)


class AccentAdversarialHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, pooling: str = "masked_mean"):
        super().__init__()
        self.pooling = pooling
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_classes),
        )

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor | None = None):
        if self.pooling == "masked_mean" and mask is not None:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
            return (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom
        return hidden.mean(dim=1)

    def forward(self, hidden: torch.Tensor, lambda_adv: float, mask: torch.Tensor | None = None):
        pooled = self._pool(hidden, mask=mask)
        rev = grad_reverse(pooled, lambda_adv)
        return self.net(rev)
