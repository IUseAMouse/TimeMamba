"""TimeSSM: a linear time-invariant state space forecaster with a rate knob.

Trained and evaluated with the TimeJEPA stack (imported, never copied):
FinetuneModule, the LOTSA datamodule, RevIN / RobustScale, the quantile head,
the GIFT-Eval harness with RateIN.
"""

from .ssm import S4DLayer
from .block import GatedSSMBlock

__all__ = ["S4DLayer", "GatedSSMBlock"]
