"""A latent intent on the control output (latent-intent design §六 L2).

The control head becomes the decoder of a conditional latent-variable model:

    q(z | future)        the POSTERIOR — reads the truth's future (the normalized target
                         rows and the true duration), training only, never at inference
    p(z | context)       the PRIOR — a mixture of K diagonal Gaussians from the same fused
                         features the decoder reads (the ego history + dynamics condition;
                         the traffic scene joins them in L4)
    decoder(z, context)  the existing bounded control head + the duration head, with z
                         reaching BOTH: Phase 0 measured the vectored spread to be largely
                         timing, so a latent that could not move the duration would leave
                         the largest mode of variation to the point estimate

Objective = the control path's own trajectory reconstruction (unchanged, dispatched on
the prediction's control fields) + ``latent_beta`` · KL(q ‖ p) with free bits. The KL is
analytic per dimension for a single Gaussian prior and a single-sample Monte-Carlo estimate
for a mixture; the per-dimension analytic KL against the most responsible component is
reported as a diagnostic either way, because posterior collapse (the decoder ignoring z)
is this model's one predictable failure and it looks exactly like "converged".

At inference the decoder receives the prior's most likely component mean — the top-1
prediction the record contract carries. Sampling K modes is ``sample_latents``.

z is never an output: it is not written to any record. The posterior lives in the
checkpoint (training resumes with it) but ``forecast`` cannot reach it — ``model_forward``
only hands the future to a model during training, and a model asked for a posterior
without one raises.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from batch_contract import LossComponents
from config import TSConfig
from control.heads import (
    ControlFeatureModel,
    _initialize_control_head,
    _initialize_final_time_head,
    control_head_for,
)
from prediction_outputs import ControlPrediction, FinalTimeHead

LATENT_KL_COMPONENT = "latent_kl"
# A latent dimension is ACTIVE when its KL exceeds what the free-bits budget gives away for
# nothing; with no budget, above this floor (nats). Read with the budget in mind: the count
# says "dimensions the objective is paying for", which is the collapse signal.
ACTIVE_UNIT_KL_NATS = 0.05
# Both log-variances are bare linear outputs; a transient excursion turns exp() into inf and
# the KL into NaN grads that the loop's divergence guard misdiagnoses as a learning-rate
# problem. Bounded here, and stated: σ² ∈ [e⁻⁸, e⁸].
LOGVAR_BOUND = 8.0
# The mixture prior's components start at N(0, I) with equal weights EXCEPT for a small
# random offset on each component's mean: identical components receive identical gradients
# forever (responsibilities are exactly 1/K at the symmetric point, and Adam preserves the
# tie), so a symmetric start trains one Gaussian wearing K labels.
PRIOR_MEAN_INIT_STD = 0.5
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class LatentControlPrediction(ControlPrediction):
    """A control prediction plus the latent it was decoded from and both densities' params.

    ``posterior_*`` are ``None`` outside training. Everything here is training/diagnostic
    state; the record contract reads only the inherited control fields.
    """

    latent: torch.Tensor                       # [B, Z]
    prior_logits: torch.Tensor                 # [B, K]
    prior_mean: torch.Tensor                   # [B, K, Z]
    prior_logvar: torch.Tensor                 # [B, K, Z]
    posterior_mean: torch.Tensor | None = None  # [B, Z]
    posterior_logvar: torch.Tensor | None = None


class PosteriorEncoder(nn.Module):
    """q(z | future): the normalized target rows and the true duration -> (mean, logvar)."""

    def __init__(self, config: TSConfig):
        super().__init__()
        self.scale_s = config.final_time_scale_s
        width = config.pred_len * len(config.channels) + 1
        self.network = nn.Sequential(
            nn.Linear(width, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 2 * config.latent_dim),
        )

    def forward(self, future: torch.Tensor, final_time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = torch.cat(
            [future.flatten(1), (final_time_s / self.scale_s).unsqueeze(-1).to(future.dtype)], dim=-1
        )
        mean, logvar = self.network(flat).chunk(2, dim=-1)
        return mean, logvar.clamp(-LOGVAR_BOUND, LOGVAR_BOUND)


class PriorNetwork(nn.Module):
    """p(z | context): a mixture of K diagonal Gaussians from the fused features."""

    def __init__(self, config: TSConfig):
        super().__init__()
        self.components = config.latent_prior_components
        self.latent_dim = config.latent_dim
        self.network = nn.Linear(config.d_model, self.components * (1 + 2 * self.latent_dim))
        with torch.no_grad():
            # Weights zero: an untrained prior does not depend on the context. Biases zero
            # (equal weights, unit variance) except each component's MEAN, offset at random
            # so the components are distinguishable from the first step (see
            # PRIOR_MEAN_INIT_STD); with K = 1 the offset is dropped and the prior is
            # exactly the N(0, I) the ELBO is usually written against.
            self.network.weight.zero_()
            self.network.bias.zero_()
            if self.components > 1:
                means = self.network.bias[self.components:].view(self.components, 2 * self.latent_dim)
                means[:, : self.latent_dim].normal_(std=PRIOR_MEAN_INIT_STD)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.network(features)
        logits = raw[:, : self.components]
        rest = raw[:, self.components :].reshape(-1, self.components, 2 * self.latent_dim)
        mean, logvar = rest.chunk(2, dim=-1)
        return logits, mean, logvar.clamp(-LOGVAR_BOUND, LOGVAR_BOUND)


def _gaussian_log_density(z: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sum over the last dim of log N(z; mean, exp(logvar))."""
    return -0.5 * (((z - mean) ** 2) / logvar.exp() + logvar + _LOG_2PI).sum(dim=-1)


def per_dimension_kl(
    q_mean: torch.Tensor, q_logvar: torch.Tensor, p_mean: torch.Tensor, p_logvar: torch.Tensor
) -> torch.Tensor:
    """Analytic KL(N(q) ‖ N(p)) per latent dimension, ``[B, Z]``."""
    return 0.5 * (
        (q_logvar - p_logvar).exp()
        + (q_mean - p_mean) ** 2 / p_logvar.exp()
        - 1.0
        + p_logvar
        - q_logvar
    )


def most_responsible_component(
    z: torch.Tensor, logits: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Index ``[B]`` of the mixture component with the highest posterior responsibility for z."""
    log_joint = F.log_softmax(logits, dim=-1) + _gaussian_log_density(
        z.unsqueeze(1), mean, logvar
    )
    return log_joint.argmax(dim=-1)


def latent_kl(
    prediction: LatentControlPrediction, *, free_bits_nats: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-flight KL term ``[B]`` (free bits applied) and the per-dim diagnostic ``[B, Z]``.

    Single Gaussian prior: exact per-dimension KL, free bits per dimension. Mixture prior:
    one-sample Monte-Carlo ``log q(z) − log p(z)`` with free bits on the total (a mixture
    has no per-dimension decomposition), and the diagnostic is the analytic KL against
    the component most responsible for the sampled z.
    """
    if prediction.posterior_mean is None or prediction.posterior_logvar is None:
        raise ValueError("the latent KL needs a posterior; this prediction was decoded from the prior")
    q_mean, q_logvar = prediction.posterior_mean, prediction.posterior_logvar
    z = prediction.latent
    if prediction.prior_logits.shape[-1] == 1:
        kl_dim = per_dimension_kl(
            q_mean, q_logvar, prediction.prior_mean[:, 0], prediction.prior_logvar[:, 0]
        )
        return (kl_dim - free_bits_nats).clamp(min=0.0).sum(dim=-1), kl_dim
    log_q = _gaussian_log_density(z, q_mean, q_logvar)
    log_p = torch.logsumexp(
        F.log_softmax(prediction.prior_logits, dim=-1)
        + _gaussian_log_density(z.unsqueeze(1), prediction.prior_mean, prediction.prior_logvar),
        dim=-1,
    )
    total = log_q - log_p
    if free_bits_nats > 0.0:
        # Free bits on the total; without a budget the SIGNED one-sample estimate passes
        # through — a single sample is negative a large share of the time when the true
        # KL is small, and clamping it at zero would overcharge exactly there, pushing
        # toward the collapse this term is meant to expose.
        budget = free_bits_nats * z.shape[-1]
        total = total.clamp(min=budget) - budget
    k = most_responsible_component(z, prediction.prior_logits, prediction.prior_mean, prediction.prior_logvar)
    index = k.view(-1, 1, 1).expand(-1, 1, z.shape[-1])
    kl_dim = per_dimension_kl(
        q_mean, q_logvar,
        prediction.prior_mean.gather(1, index).squeeze(1),
        prediction.prior_logvar.gather(1, index).squeeze(1),
    )
    return total, kl_dim


class LatentControlModel(ControlFeatureModel):
    """The control output with a latent intent between the context and the heads."""

    consumes_future = True   # model_forward hands it the future during training

    def __init__(self, config: TSConfig, feature_encoder: nn.Module):
        super().__init__(config, feature_encoder)
        if config.latent_dim < 1:
            raise ValueError("LatentControlModel needs latent_dim >= 1")
        self.latent_dim = config.latent_dim
        self.posterior = PosteriorEncoder(config)
        self.prior = PriorNetwork(config)
        self.latent_fusion = nn.Sequential(
            nn.Linear(config.d_model + config.latent_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.final_time_head = FinalTimeHead(config)
        # z's path to the duration: an additive term on the head's unconstrained logit, zero
        # at initialization so the duration starts exactly where the plain head would.
        self.latent_duration = nn.Linear(config.latent_dim, 1)
        with torch.no_grad():
            self.latent_duration.weight.zero_()
            self.latent_duration.bias.zero_()
        self.control_head = control_head_for(config)
        _initialize_control_head(self.control_head)
        _initialize_final_time_head(self.final_time_head)

    def top1_latent(self, logits: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """The prior's most likely component mean — the deterministic top-1 intent."""
        index = logits.argmax(dim=-1).view(-1, 1, 1).expand(-1, 1, self.latent_dim)
        return mean.gather(1, index).squeeze(1)

    def sample_latents(
        self, logits: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``samples`` draws from the mixture prior: latents ``[S, B, Z]`` and their component ``[S, B]``."""
        probabilities = F.softmax(logits, dim=-1)
        component = torch.multinomial(
            probabilities, samples, replacement=True, generator=generator
        ).transpose(0, 1)                                                  # [S, B]
        index = component.unsqueeze(-1).expand(-1, -1, self.latent_dim)
        chosen_mean = mean.unsqueeze(0).expand(samples, -1, -1, -1).gather(
            2, index.unsqueeze(2)
        ).squeeze(2)
        chosen_logvar = logvar.unsqueeze(0).expand(samples, -1, -1, -1).gather(
            2, index.unsqueeze(2)
        ).squeeze(2)
        noise = torch.randn(chosen_mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return chosen_mean + noise * (0.5 * chosen_logvar).exp(), component

    def decode(
        self, features: torch.Tensor, history: torch.Tensor, latent: torch.Tensor,
        dynamics: dict[str, torch.Tensor],
    ) -> ControlPrediction:
        decoder_features = self.latent_fusion(torch.cat([features, latent], dim=-1))
        if self.cta_given:
            # The given CTA is the duration; the head and z's duration path are inert.
            final_time_s = dynamics["cta_s"].to(features.dtype)
        else:
            raw_duration = self.final_time_head.raw(history) + self.latent_duration(latent).squeeze(-1)
            final_time_s = F.softplus(raw_duration) * self.final_time_head.scale_s
        return self.control_head(
            decoder_features, final_time_s,
            lower=dynamics["control_lower"], upper=dynamics["control_upper"],
        )

    def forward(
        self,
        history: torch.Tensor,
        dynamics: dict[str, torch.Tensor],
        future: tuple[torch.Tensor, torch.Tensor] | None = None,
        latent: torch.Tensor | None = None,
    ) -> LatentControlPrediction:
        """Training passes ``future = (target_rows, true_final_time_s)`` and decodes a posterior
        sample; inference passes nothing and decodes the prior's top-1; ``latent`` overrides
        both (K-sample forecasting, the shuffle-z diagnostic)."""
        features = self.fused_features(history, dynamics)
        logits, prior_mean, prior_logvar = self.prior(features)
        posterior_mean = posterior_logvar = None
        if latent is None:
            if future is not None:
                posterior_mean, posterior_logvar = self.posterior(*future)
                latent = posterior_mean + torch.randn_like(posterior_mean) * (0.5 * posterior_logvar).exp()
            else:
                latent = self.top1_latent(logits, prior_mean)
        controls = self.decode(features, history, latent, dynamics)
        return LatentControlPrediction(
            controls=controls.controls,
            segment_durations=controls.segment_durations,
            final_time_s=controls.final_time_s,
            latent=latent,
            prior_logits=logits,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
        )


def active_unit_threshold_nats(config: TSConfig) -> float:
    """Above this per-dimension KL the objective is paying for the dimension."""
    return max(config.latent_free_bits_nats, ACTIVE_UNIT_KL_NATS)


def with_latent_kl(
    components: LossComponents,
    prediction: LatentControlPrediction,
    config: TSConfig,
    flight_weights: torch.Tensor,
) -> LossComponents:
    """Add the weighted KL to a control objective and the collapse diagnostics.

    The KL term is flight-weighted like every other control term (``flight_weights`` are
    the airport-macro weights), so β means the same thing in a multi-airport run. The
    diagnostics are UNWEIGHTED sums plus the flight count they were summed over, so the
    epoch record divides like by like.
    """
    kl, kl_dim = latent_kl(prediction, free_bits_nats=config.latent_free_bits_nats)
    weights = flight_weights.to(dtype=kl.dtype, device=kl.device)
    diagnostics = dict(components.diagnostics)
    diagnostics["latent_flights"] = kl_dim.new_tensor(float(len(kl_dim)))
    # Two KLs: the one the objective CHARGES (free bits applied; the MC estimate for a
    # mixture) and the per-dimension analytic KL against the most responsible component,
    # which active_units is read from. For K = 1 without free bits they coincide.
    diagnostics["latent_kl_nats"] = kl.detach().sum()
    diagnostics["latent_component_kl_nats"] = kl_dim.detach().sum()
    diagnostics["latent_active_units"] = (
        kl_dim.detach().mean(dim=0) > active_unit_threshold_nats(config)
    ).sum().to(kl_dim.dtype) * len(kl_dim)
    return replace(
        components,
        extras={**components.extras, LATENT_KL_COMPONENT: config.latent_beta * (kl * weights).mean()},
        diagnostics=diagnostics,
    )
