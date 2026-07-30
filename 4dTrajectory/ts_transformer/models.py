"""One interface over the two vendored architectures.

The vendored files are byte-identical to upstream, and upstream disagrees about how a
model is called: iTransformer keeps the four-argument Autoformer-family signature
``model(x_enc, x_mark_enc, x_dec, x_mark_dec)`` (ignoring the last three on the inverted
path), while PatchTST takes a bare ``model(x)``. Rather than teach the training loop to
branch on architecture — or edit the vendored code and forfeit the clean upstream diff —
each state forecaster is wrapped in a thin adapter with one signature::

    forecaster(x: Tensor[B, seq_len, C]) -> Tensor[B, N, C]

The state path attaches ``final_time_s`` to that forecast. The opt-in control path instead
retains the same encoder's ordered per-channel pre-projection features, conditions them on
the flight's mass and aerodynamic parameters, and decodes bounded controls plus a
non-uniform time partition.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    PREDICTION_CONTROL,
    PREDICTION_CONTROL_MIXTURE,
    PREDICTION_STATE,
    TSConfig,
)
from control_models import ControlMixtureOutputModel, ControlOutputModel
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

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten encoder tokens in channel-contract order before the state projector."""
        if self.inner.use_norm:
            x = x - x.mean(1, keepdim=True).detach()
            x = x / torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        encoded = self.inner.enc_embedding(x, None)
        encoded, _attentions = self.inner.encoder(encoded, attn_mask=None)
        # iTransformer has no channel-index embedding: averaging these permutation-
        # equivariant tokens erases which latent came from e/n/u or its derivative. Keep
        # the serialized CHANNELS order explicit in the flattened feature instead.
        return encoded.flatten(start_dim=1)

    def discard_state_head(self) -> None:
        """Remove the unused state projector when this adapter feeds a control head."""
        self.inner.projector = nn.Identity()


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

    @staticmethod
    def _backbone_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        z = x.permute(0, 2, 1)
        if backbone.revin:
            z = backbone.revin_layer(z.permute(0, 2, 1), "norm").permute(0, 2, 1)
        if backbone.padding_patch == "end":
            z = backbone.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=backbone.patch_len, step=backbone.stride)
        z = z.permute(0, 1, 3, 2)
        encoded = backbone.backbone(z)  # [B,C,d_model,patches]
        # Pool only time patches. PatchTST processes channels independently and has no
        # channel identity inside the backbone, so the ordered C axis must survive.
        return encoded.mean(dim=3).flatten(start_dim=1)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool channel-independent patch tokens before the state forecast head."""
        if not self.inner.decomposition:
            return self._backbone_features(self.inner.model, x)
        residual, trend = self.inner.decomp_module(x)
        return self._backbone_features(
            self.inner.model_res, residual
        ) + self._backbone_features(self.inner.model_trend, trend)

    def discard_state_head(self) -> None:
        """Remove unused flattened forecast heads from a control-only adapter."""
        if self.inner.decomposition:
            self.inner.model_res.head = nn.Identity()
            self.inner.model_trend.head = nn.Identity()
        else:
            self.inner.model.head = nn.Identity()


BUILDERS = {
    "itransformer": ITransformerAdapter,
    "patchtst": PatchTSTAdapter,
}


def build_state_forecaster(config: TSConfig) -> nn.Module:
    """The vendored state forecaster selected by ``config.model``."""
    return BUILDERS[config.model](config)


def _build_state_output(config: TSConfig) -> nn.Module:
    return StateOutputLayer(build_state_forecaster(config), config)


def _build_control_output(config: TSConfig) -> nn.Module:
    return ControlOutputModel(config, build_state_forecaster(config))


def _build_control_mixture_output(config: TSConfig) -> nn.Module:
    return ControlMixtureOutputModel(config, build_state_forecaster(config))


OUTPUT_MODEL_BUILDERS = {
    PREDICTION_STATE: _build_state_output,
    PREDICTION_CONTROL: _build_control_output,
    PREDICTION_CONTROL_MIXTURE: _build_control_mixture_output,
}


def build_model(config: TSConfig) -> nn.Module:
    """Build an explicitly registered output strategy."""
    return OUTPUT_MODEL_BUILDERS[config.prediction_output](config)


def resolve_device(spec: str) -> torch.device:
    """``"auto"`` -> cuda when available, else cpu; anything else passed through verbatim."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
