from __future__ import annotations

import torch
from torch import nn


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor, lambda_: float | None = None):
        coeff = self.lambda_ if lambda_ is None else float(lambda_)
        return _GradientReversalFn.apply(x, coeff)


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0):
    return _GradientReversalFn.apply(x, float(lambda_))
