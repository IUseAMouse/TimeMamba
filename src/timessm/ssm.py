"""
S4D: a diagonal linear time-invariant state space layer with a rate knob.

Continuous system per channel h and state n (SSM_cours ch. 2-4):

    x'(t) = a x(t) + b u(t),   y(t) = Re(sum_n c_n x_n(t)) + d u(t),   Re(a) < 0

sampled every Delta with a zero-order hold (the input is held constant over
each interval), which is EXACT for piecewise-constant inputs:

    A_bar = exp(Delta a),   B_bar = (A_bar - 1) / a * b            (eq. 2.1)
    x_t = A_bar x_{t-1} + B_bar u_t,   y_t = Re(c . x_t) + d u_t

The same recurrence is a causal convolution y = K * u with the kernel
K_j = Re(c . A_bar^j B_bar), computed here in closed form (Vandermonde) and
applied by FFT - no divide-by-A_bar^i scan (the March 2025 code, unstable at
long lengths), no sequential loop at train time.

The rate knob. `delta_scale` multiplies Delta for the whole layer (per batch
or per item): Delta_eff = exp(log_dt) * delta_scale. Proposition 2.4 of the
course: running the series decimated by k at Delta is the same as running the
full series at Delta / k, so the eval passes w = 1 / k here where the
TimeJEPA harness would decimate and re-interpolate. That equivalence holds
because the system is LTI: a selective (input-dependent) Delta breaks it
(prop. 5.1), which is why this is S4D and not Mamba.

Stability by construction: Re(a) = -exp(log_a_re) < 0, so |A_bar| < 1 for any
positive Delta_eff.

Parameterization (all real tensors, so any optimizer works):
    log_a_re [H, N], a_im [H, N] (complex state only), B_re/B_im [H, N],
    C_re/C_im [H, N], log_dt [H], D [H].
Init 's4d-lin' (complex): a_n = -1/2 + i pi n. Init 's4d-real': a_n = -(n+1).
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

Scale = Union[float, torch.Tensor]


class S4DLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        real: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        init: str = "s4d-lin",
    ):
        super().__init__()
        H, N = d_model, d_state
        self.d_model, self.d_state, self.real = H, N, real
        n = torch.arange(N, dtype=torch.float32)
        if real or init == "s4d-real":
            self.real = True
            log_a_re = torch.log(n + 1.0).repeat(H, 1)
        elif init == "s4d-lin":
            log_a_re = torch.full((H, N), math.log(0.5))
        else:
            raise ValueError(f"unknown init {init!r} (s4d-lin, s4d-real)")
        self.log_a_re = nn.Parameter(log_a_re)
        if self.real:
            self.a_im = None
            self.B_im = None
            self.C_im = None
        else:
            self.a_im = nn.Parameter((math.pi * n).repeat(H, 1))
            self.B_im = nn.Parameter(torch.zeros(H, N))
            self.C_im = nn.Parameter(torch.randn(H, N) * math.sqrt(0.5))
        self.B_re = nn.Parameter(torch.ones(H, N))
        self.C_re = nn.Parameter(torch.randn(H, N) * math.sqrt(0.5))
        # Delta per channel, log-uniform over [dt_min, dt_max] (S4 recipe):
        # several decades of time scale in one layer.
        self.log_dt = nn.Parameter(
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.D = nn.Parameter(torch.ones(H))

    # ------------------------------------------------------------ pieces
    def _a(self) -> torch.Tensor:
        a_re = -torch.exp(self.log_a_re)
        if self.real:
            return a_re
        return torch.complex(a_re, self.a_im)

    def _b(self) -> torch.Tensor:
        return self.B_re if self.real else torch.complex(self.B_re, self.B_im)

    def _c(self) -> torch.Tensor:
        return self.C_re if self.real else torch.complex(self.C_re, self.C_im)

    def _scales(self, delta_scale: Scale, batch: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(unique scales [U], inverse index [B] or None when one scale)."""
        p = self.log_dt
        if not torch.is_tensor(delta_scale):
            return torch.tensor([float(delta_scale)], dtype=p.dtype, device=p.device), None
        s = delta_scale.to(device=p.device, dtype=p.dtype).reshape(-1)
        if s.numel() == 1:
            return s, None
        if s.numel() != batch:
            raise ValueError(f"delta_scale has {s.numel()} entries for a batch of {batch}")
        uniq, inv = torch.unique(s, return_inverse=True)
        return uniq, inv

    def discretize(self, delta_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ZOH for each scale in `delta_scale` [U]: (A_bar, B_bar, z) each
        [U, H, N] with z = Delta_eff * a (so A_bar = exp(z))."""
        dt = torch.exp(self.log_dt)                                # [H]
        delta = dt[None, :, None] * delta_scale.reshape(-1)[:, None, None]  # [U, H, 1]
        a = self._a()[None]                                        # [1, H, N]
        z = delta * a
        a_bar = torch.exp(z)
        b_bar = (a_bar - 1.0) / a * self._b()[None]
        return a_bar, b_bar, z

    def kernel(self, length: int, delta_scale: torch.Tensor) -> torch.Tensor:
        """Convolution kernel K [U, H, L], K_j = Re(c . A_bar^j B_bar)."""
        _, b_bar, z = self.discretize(delta_scale)
        j = torch.arange(length, device=z.device, dtype=self.log_dt.dtype)
        powers = torch.exp(z[..., None] * j)                       # [U, H, N, L]
        coef = (self._c()[None] * b_bar)[..., None]                # [U, H, N, 1]
        K = (coef * powers).sum(dim=2)                             # [U, H, L]
        return K.real if torch.is_complex(K) else K

    # ------------------------------------------------------------ forward
    def forward(self, u: torch.Tensor, delta_scale: Scale = 1.0) -> torch.Tensor:
        """u [B, L, H] -> y [B, L, H], causal FFT convolution + skip."""
        B, L, H = u.shape
        scales, inv = self._scales(delta_scale, B)
        K = self.kernel(L, scales)                                 # [U, H, L]
        K = K[inv] if inv is not None else K                       # [B|1, H, L]
        u_t = u.transpose(1, 2)                                    # [B, H, L]
        n_fft = 2 * L
        y = torch.fft.irfft(
            torch.fft.rfft(u_t, n=n_fft) * torch.fft.rfft(K, n=n_fft), n=n_fft
        )[..., :L]
        y = y + self.D[None, :, None] * u_t
        return y.transpose(1, 2)

    # ------------------------------------------------------------ recurrence
    def init_state(self, batch: int, dtype=None, device=None) -> torch.Tensor:
        dtype = dtype or self.log_dt.dtype
        device = device or self.log_dt.device
        if not self.real:
            dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        return torch.zeros(batch, self.d_model, self.d_state, dtype=dtype, device=device)

    def step(self, u_t: torch.Tensor, state: torch.Tensor,
             delta_scale: Scale = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """One step of the recurrence: u_t [B, H], state [B, H, N] ->
        (y_t [B, H], new state). Same numbers as `forward`, step by step."""
        B = u_t.shape[0]
        scales, inv = self._scales(delta_scale, B)
        a_bar, b_bar, _ = self.discretize(scales)                  # [U, H, N]
        if inv is not None:
            a_bar, b_bar = a_bar[inv], b_bar[inv]
        state = a_bar * state + b_bar * u_t.unsqueeze(-1).to(state.dtype)
        y = (self._c()[None] * state).sum(-1)
        y = y.real if torch.is_complex(y) else y
        return y + self.D[None] * u_t, state
