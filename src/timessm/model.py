"""
SSMForecaster: the TimeSSM model, duck-typed on TimeJEPA's JEPATST so the
TimeJEPA training module (FinetuneModule), its datamodule and its GIFT-Eval
harness (RateIN, mix, flip) run on it unchanged.

    x [B, L, 1] -> RobustScale (arcsinh, context stats) -> RevIN (context
    frame, pinned) -> Patching(1, 1) = one token per step -> n_layers gated
    S4D blocks over [context tokens ; n future tokens] -> LayerNorm
      context_embeddings [B, L, D], future_representations [B, n, D]
    -> QuantileHead (cross-attention on the context, patch 1 / stride 1:
       one latent per step) -> quantiles [B, n, Q] -> RevIN / RobustScale
       inverse for the denormalized fan.

Autonomous rollout. The sequence is extended by n copies of a LEARNED future
token (not a zero: the embedding has a bias and the gates must tell "future"
from "value 0"); the LTI state extrapolates through them and the outputs at
positions L..L+n-1 are the future representations. n is free (8 to 900 on
GIFT) with no autoregressive loop and no query table; nothing in the model
depends on the horizon it was trained at.

The rate knob. `forecast(x, n, w=s)` multiplies every layer's Delta by s
(float or [B]). w = 1 / k is the SSM's exact counterpart of decimating the
context by k (SSM_cours prop. 2.4). `rate_knob = 'delta'` tells the harness;
`predictor.w_film` is None so `+ratein_w` (the FiLM of the JEPA model) is
refused as it should be. There is no online_encoder: the JEPA-side inference
layers (+refine, +ttt, joint / critic terms) do not apply here.
"""

from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from timejepa.models.components.patching import Patching
from timejepa.models.components.revin import RevIN
from timejepa.models.components.robust_scale import RobustScale
from timejepa.models.decoders.linear_decoder import ForecastingHead

from .block import GatedSSMBlock

Scale = Union[float, torch.Tensor]


class SSMForecaster(nn.Module):
    rate_knob = "delta"
    # Read by TimeJEPA's load_checkpoint (P3.2 refusal): a mismatch on these
    # prefixes means the evaluation would score fresh weights.
    core_prefixes = ("blocks.", "patching.", "robust_scaler.", "future_token", "final_norm.")

    def __init__(
        self,
        input_length: int = 1024,
        prediction_length: int = 256,
        d_model: int = 192,
        n_layers: int = 6,
        d_state: int = 32,
        expand: int = 2,
        d_conv: int = 0,
        dropout: float = 0.1,
        selective_readout: bool = False,
        real: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        quantile_levels: Optional[Sequence[float]] = None,
        quantile_hidden_dim: Optional[int] = 1536,
        quantile_use_context: bool = True,
        robust_scale: bool = True,
        revin_affine: bool = False,
        name: str = "timessm",
    ):
        super().__init__()
        self.name = name
        self.input_length = int(input_length)
        self.prediction_length = int(prediction_length)
        self.d_model = int(d_model)
        # One token per step: Linear(1 -> D). No padding: prepare_context in
        # the harness aligns lengths on stride = 1, i.e. never truncates.
        self.patching = Patching(patch_size=1, d_model=d_model, num_features=1,
                                 stride=1, padding=False)
        self.revin = RevIN(num_features=1, affine=revin_affine)
        self.robust_scaler = RobustScale() if robust_scale else None
        self.blocks = nn.ModuleList([
            GatedSSMBlock(d_model, d_state=d_state, expand=expand, d_conv=d_conv,
                          dropout=dropout, selective_readout=selective_readout,
                          real=real, dt_min=dt_min, dt_max=dt_max)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.future_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.decoder = ForecastingHead(
            d_model=d_model, patch_size=1, stride=1,
            prediction_length=prediction_length, num_features=1,
            decoder_type="quantile", revin=self.revin,
            quantile_levels=quantile_levels,
            quantile_use_context=quantile_use_context,
            quantile_hidden_dim=quantile_hidden_dim,
        )
        # Duck-typing hooks read by the TimeJEPA module and harness.
        self.predictor = SimpleNamespace(w_film=None)
        self._pretrain_mode = False

    # ------------------------------------------------------------ core
    def encode(self, context_norm: torch.Tensor, n: int, delta_scale: Scale = 1.0):
        """context_norm [B, L, 1] (model frame) -> (context_embeddings
        [B, L, D], future_representations [B, n, D])."""
        B, L = context_norm.shape[0], context_norm.shape[1]
        tokens = self.patching(context_norm)                          # [B, L, D]
        future = self.future_token.reshape(1, 1, -1).expand(B, n, -1)
        h = torch.cat([tokens, future], dim=1)                        # [B, L+n, D]
        for block in self.blocks:
            h = block(h, delta_scale=delta_scale)
        h = self.final_norm(h)
        return h[:, :L], h[:, L:]

    def forward_finetune(
        self,
        context: torch.Tensor,
        n: Optional[int] = None,
        return_representations: bool = False,
        skip_revin: bool = False,
        w: Optional[Scale] = None,
    ) -> Dict[str, torch.Tensor]:
        """Same contract as JEPATST.forward_finetune, plus a free horizon n."""
        n = int(n or self.prediction_length)
        delta_scale = 1.0 if w is None else w
        if context.dim() == 2:
            context = context.unsqueeze(-1)
        if self.robust_scaler is not None and not skip_revin:
            self.robust_scaler.fit(context)
            context = self.robust_scaler.transform(context)
        context_norm = context if skip_revin else self.revin(context, mode="norm")

        ctx_emb, fut_repr = self.encode(context_norm, n, delta_scale)
        head = self.decoder.decoder
        quantiles = head(fut_repr, context_embeddings=ctx_emb, target_length=n)  # [B, n, Q]
        if skip_revin:
            quantiles_denorm = quantiles
        else:
            quantiles_denorm = self.revin.denormalize_target_space(quantiles)
            if self.robust_scaler is not None:
                quantiles_denorm = self.robust_scaler.inverse(quantiles_denorm)

        result = {
            "quantiles": quantiles,
            "quantiles_denorm": quantiles_denorm,
            "quantile_levels": head.quantile_levels,
            "forecast": head.median(quantiles),
            "forecast_denorm": head.median(quantiles_denorm),
        }
        if return_representations:
            result["context_embeddings"] = ctx_emb
            result["future_representations"] = fut_repr
            result["context_norm"] = context_norm
        return result

    def forecast(
        self,
        context: torch.Tensor,
        n: Optional[int] = None,
        return_representations: bool = False,
        skip_revin: bool = False,
        sample_paths: bool = True,
        w: Optional[Scale] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forecast n steps (any n: no rolling, the rollout is the state's
        own extrapolation). `w` = delta_scale, float or [B]."""
        return self.forward_finetune(context, n=n,
                                     return_representations=return_representations,
                                     skip_revin=skip_revin, w=w)

    def forward(self, context: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.forward_finetune(context, **kwargs)

    def forward_pretrain(self, *args, **kwargs):
        raise NotImplementedError(
            "SSMForecaster has no JEPA pretrain path (the spike trains from "
            "scratch on the pinball); lambda_anchor / lambda_joint must be 0.")

    # ------------------------------------------------------------ hooks
    def set_pretrain_mode(self, mode: bool = True):
        if mode:
            raise ValueError("SSMForecaster has no pretrain mode")
        self._pretrain_mode = False

    def is_pretrain_mode(self) -> bool:
        return False

    def _encoder_params(self):
        for block in self.blocks:
            yield from block.parameters()
        yield from self.final_norm.parameters()
        yield self.future_token

    def freeze_encoder(self):
        for p in self._encoder_params():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self._encoder_params():
            p.requires_grad = True

    def freeze_predictor(self):        # no predictor: the state extrapolates
        pass

    def unfreeze_predictor(self):
        pass

    def freeze_target_encoder(self):   # no target encoder
        pass

    def freeze_patching(self):
        for p in self.patching.parameters():
            p.requires_grad = False

    def unfreeze_patching(self):
        for p in self.patching.parameters():
            p.requires_grad = True

    def freeze_revin(self):
        for p in self.revin.parameters():
            p.requires_grad = False

    def unfreeze_revin(self):
        for p in self.revin.parameters():
            p.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    def get_num_params(self) -> Dict[str, int]:
        return {
            "blocks": sum(p.numel() for p in self.blocks.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "patching": sum(p.numel() for p in self.patching.parameters()),
            "total": self.count_parameters(trainable_only=False),
        }


def build_from_config(cfg) -> SSMForecaster:
    """Hydra config -> model. Read by scripts/train_ssm.py and, through
    `model.builder`, by TimeJEPA's create_model_from_config (eval)."""
    s = cfg.model.ssm
    return SSMForecaster(
        input_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        d_model=s.d_model, n_layers=s.n_layers, d_state=s.d_state,
        expand=s.get("expand", 2), d_conv=s.get("d_conv", 0),
        dropout=s.get("dropout", 0.1),
        selective_readout=bool(s.get("selective_readout", False)),
        real=bool(s.get("real", False)),
        dt_min=float(s.get("dt_min", 1e-3)), dt_max=float(s.get("dt_max", 1e-1)),
        quantile_hidden_dim=cfg.model.decoder.get("quantile_hidden_dim"),
        quantile_use_context=bool(cfg.model.decoder.get("quantile_use_context", True)),
        robust_scale=bool(cfg.model.get("robust_scale", True)),
        revin_affine=bool(cfg.model.get("revin_affine", False)),
        name=cfg.model.name,
    )
