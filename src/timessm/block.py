"""
Gated SSM block (SSM_cours ch. 4.3), the MambaBlock of March 2025 with three
differences: the S4D layer replaces the divide scan, `delta_scale` is relayed
to it, and the selectivity - when asked for - sits in the READOUT, not in the
dynamics (an input-dependent gate on the SSM output, pointwise in time, so the
LTI rate transfer of the dynamics is untouched; ablation arm P-SSM.3).

    x -> LayerNorm -> in_proj -> (u, z)
      u -> [causal depthwise conv, d_conv > 0 only] -> SiLU -> S4D(u, delta_scale)
        -> [* sigmoid(W_r u), selective_readout only] -> * SiLU(z) -> out_proj
    -> dropout -> + x

The depthwise convolution is OFF by default (`d_conv: 0`). It is a filter in
STEPS, not in physical time: with it, running the decimated series at Delta
is no longer the same as running the full series at Delta / k (test 1 of
tests/test_ssm_layer.py shows the discrepancy), and that equivalence is the
whole point of the spike. Every other operation is pointwise in time.
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import S4DLayer

Scale = Union[float, torch.Tensor]


class GatedSSMBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        expand: int = 2,
        d_conv: int = 0,
        dropout: float = 0.0,
        selective_readout: bool = False,
        real: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = (
            nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
            if d_conv > 0 else None
        )
        self.ssm = S4DLayer(d_inner, d_state=d_state, real=real, dt_min=dt_min, dt_max=dt_max)
        self.readout_gate = nn.Linear(d_inner, d_inner) if selective_readout else None
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, delta_scale: Scale = 1.0) -> torch.Tensor:
        """x [B, L, D] -> [B, L, D]."""
        L = x.shape[1]
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        if self.conv is not None:
            u = self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2)   # causal
        u = F.silu(u)
        y = self.ssm(u, delta_scale=delta_scale)
        if self.readout_gate is not None:
            y = y * torch.sigmoid(self.readout_gate(u))
        y = y * F.silu(z)
        return x + self.dropout(self.out_proj(y))
