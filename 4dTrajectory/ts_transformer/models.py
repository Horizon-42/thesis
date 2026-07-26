"""One interface over the two vendored architectures.

The vendored files are byte-identical to upstream, and upstream disagrees about how a
model is called: iTransformer keeps the four-argument Autoformer-family signature
``model(x_enc, x_mark_enc, x_dec, x_mark_dec)`` (ignoring the last three on the inverted
path), while PatchTST takes a bare ``model(x)``. Rather than teach the training loop to
branch on architecture — or edit the vendored code and forfeit the clean upstream diff —
each state forecaster is wrapped in a thin adapter with one signature::

    forecaster(x: Tensor[B, seq_len, C]) -> Tensor[B, N, C]

The replaceable output layer then adds ``final_time_s`` and returns a typed prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import TSConfig
from prediction_outputs import StateOutputLayer
from vendor.itransformer import Model as VendoredITransformer
from vendor.patchtst import Model as VendoredPatchTST


class ITransformerAdapter(nn.Module):
    """iTransformer: attention ACROSS variates (each channel is one token).

    The inverted design embeds a whole series per channel, so attention is computed between
    channels rather than between time steps. For trajectory data that means it can represent
    "east and north move together through a turn" directly — the coupling PatchTST cannot see.
    """

    def __init__(self, config: TSConfig):
        super().__init__()
        self.inner = VendoredITransformer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mark_enc=None: no calendar covariates. Time enters this project's data as the
        # uniform grid itself (dt is constant), not as an embedded timestamp feature.
        return self.inner(x, None, None, None)


class PatchTSTAdapter(nn.Module):
    """PatchTST: channel-independent, patched attention along TIME.

    Every channel is forecast by the same weights in isolation (``TSTiEncoder``), so
    cross-channel coupling is structurally unavailable — see vendor/patchtst/PROVENANCE.md.
    That is the documented design and the reason it generalises; here it is also the
    interesting contrast against iTransformer.
    """

    def __init__(self, config: TSConfig):
        super().__init__()
        # The vendored Model reads most knobs off `configs` but takes the activation as a
        # bare kwarg (upstream API) — plumb it, or TSConfig.activation would apply to
        # iTransformer only while the checkpoint records it for both.
        self.inner = VendoredPatchTST(config, act=config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


BUILDERS = {
    "itransformer": ITransformerAdapter,
    "patchtst": PatchTSTAdapter,
}


def build_state_forecaster(config: TSConfig) -> nn.Module:
    """The vendored state forecaster selected by ``config.model``."""
    return BUILDERS[config.model](config)


def build_model(config: TSConfig) -> StateOutputLayer:
    """Current state-output model with a separately replaceable prediction layer."""
    return StateOutputLayer(build_state_forecaster(config), config)


def resolve_device(spec: str) -> torch.device:
    """``"auto"`` -> cuda when available, else cpu; anything else passed through verbatim."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
