#!/usr/bin/env python
"""L2.f: measure a latent checkpoint's prior and posterior DIRECTLY, on the split's flights.

Latent-intent design
(`4dTrajectory/ts_transformer/docs/2026-09-07_latent_intent_design.zh.md`) §六 L2.e' 结果 /
L2.f. Three 180-epoch arms were read as "the budget works, the information is just small"
because every number a training run wrote — total KL, active units, shuffled ΔADE — is
blind to WHERE the KL is spent. The answer came from asking the two densities directly:

    | arm                 | |q mean − p mean| / p sigma | q sigma | p sigma | KL/dim |
    | warm β=0.01 fb 0.05 | 0.05–0.11                   |    0.40 |    0.44 |   0.03 |
    | fb 0.5              | 0.10–0.20                   |    0.23 |    0.49 |   0.32 |
    | fb 1.0              | 0.11–0.19                   |    0.15 |    0.58 |   0.83 |

Every posterior mean sat ON the prior mean — under 0.2 prior sigma per flight — and the
whole budget went into narrowing the posterior. z was a denoised constant. This runner is
that measurement, as code beside the other replay runners::

    python run_ts_latent_probe.py \\
        --checkpoint L2d_warm=4dTrajectory/outputs/KRDU/experiments/l2_warm_posterior_20260907/L2d_warm_beta0p01/checkpoint.pt \\
        --checkpoint L2e_fb0p5=.../L2e_fb0p5/checkpoint.pt \\
        --out 4dTrajectory/outputs/KRDU/experiments/latent_probe_20260907

Training now records the same quantities per epoch (`history.json`'s `latent` block, read by
`run_ts_latent_readout.py --history`), so this runner is for a checkpoint whose history
predates them, for the exact per-dimension medians a summable epoch diagnostic cannot carry,
and for reading two arms side by side.

**Nothing here is a prediction result**: the posterior reads the truth's future by
construction, which is why `predict --z-from-posterior` wears `z=posterior` in its run name.
This is a diagnostic of the TRAINING-side densities and its numbers must never be quoted as
model accuracy.

The cohort is rebuilt the way every replay runner rebuilds one — the checkpoint's own
provenance and split, through `run_ts_anytime_curve.load_arm` / `cohort_series`, so the
roster rule (`data_provenance.checkpoint_data_provenance`) has one owner. That shared loader
also refuses a `cta_conditioning=given` or `intent_conditioning=truth-…` checkpoint (both
read the future in a second way); the refusal this runner adds is its own: a checkpoint
without a latent has no posterior to probe. The output directory is an immutable artifact
and is written only once the measurement has succeeded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR.parent, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ts_transformer.batch_contract import model_forward, unpack_batch  # noqa: E402
from ts_transformer.outputs.control.latent import (  # noqa: E402
    LatentControlPrediction,
    displacement_verdict,
    latent_kl,
    sigma_from_logvar,
)
from ts_transformer.dataset import FixedAnchorTrajectoryWindows  # noqa: E402
from ts_transformer.models import resolve_device  # noqa: E402
from ts_transformer.objective import move_dynamics  # noqa: E402
from run_ts_anytime_curve import Arm, Grid, cohort_series, load_arm, parse_arms  # noqa: E402

RESULT_SCHEMA = "ts-latent-probe-v1"
#: How the shared replay loader names this measurement in its refusals.
INSTRUMENT = "this posterior probe"
#: The outer-test split stays sealed (repo experiment rule): a diagnostic has no gate.
FORBIDDEN_SPLIT = "test"
SPLITS = ("val", "train", FORBIDDEN_SPLIT)
#: What `--limit` means, stated in the artifact as well as the flag: it is not a sample.
LIMIT_MEANING = (
    "--limit N takes the FIRST N flight keys of the checkpoint's split, in the split's own "
    "order — a prefix, not a random sample. Between 100 and 200 KRDU flights the displacement "
    "median moved 25 %, so a limited table is a smoke test and must never be quoted beside a "
    "full-split number."
)

#: Every median here is `torch.quantile(..., 0.5)` rather than `torch.median`, which returns
#: the LOWER of the two middle values on an even count — the training-side diagnostic uses
#: the same convention, and an outside reader reproducing it with numpy lands on this one.
MEDIAN = 0.5


def _quantile(values: torch.Tensor, quantile: float, dim: int | None = None) -> torch.Tensor:
    return (
        torch.quantile(values.flatten(), quantile) if dim is None
        else torch.quantile(values, quantile, dim=dim)
    )


def mixture_moments(
    logits: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-flight mean and variance of the MIXTURE prior itself, ``[B, Z]`` each.

    ``E[z] = Σ π_k μ_k`` and ``Var[z] = Σ π_k (σ_k² + μ_k²) − E[z]²`` — the law of total
    variance, i.e. the within-component width plus the spread BETWEEN components, which is
    where a K>1 prior keeps most of its range. With K = 1 the sum has one term and this
    reduces to ``(μ, σ²)``, the single component's own moments.
    """
    weights = F.softmax(logits, dim=-1).unsqueeze(-1)                      # [B, K, 1]
    expectation = (weights * mean).sum(dim=1)                              # [B, Z]
    second_moment = (weights * (logvar.exp() + mean.square())).sum(dim=1)
    return expectation, second_moment - expectation.square()


def densities(arm: Arm, series: list, device: torch.device) -> dict[str, torch.Tensor]:
    """Every flight's posterior and prior parameters, and the KL record built from them.

    One deterministic forward per batch through the SAME entry the training loop uses
    (`batch_contract.model_forward` with the truth's future), so the posterior here is the
    posterior the objective saw — not a second implementation of the encoder call.
    """
    windows = FixedAnchorTrajectoryWindows(series, arm.config, arm.normalizer)
    if len(windows) != len(series):
        raise SystemExit(
            f"{arm.label}: the fixed-anchor window set covers {len(windows)}/{len(series)} "
            "flights; the probe must cover the whole cohort"
        )
    model = arm.model.to(device).eval()
    collected: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(series), arm.config.batch_size):
        rows = np.arange(start, min(start + arm.config.batch_size, len(series)))
        x, y, _mask, final_time_s, _weights, dynamics, _dense = unpack_batch(windows.batch(rows))
        x, y, final_time_s = x.to(device), y.to(device), final_time_s.to(device)
        with torch.no_grad():
            prediction = model_forward(
                model, x, move_dynamics(dynamics, device), future=(y, final_time_s)
            )
        if not isinstance(prediction, LatentControlPrediction):
            raise SystemExit(f"{arm.label}: {type(prediction).__name__} carries no latent")
        kl = latent_kl(prediction, free_bits_nats=arm.config.latent_free_bits_nats)
        # The mixture's OWN moments, per flight, beside the one component the analytic
        # diagnostics are taken against: for K > 1 the prior's range lives mostly BETWEEN
        # the components, and the selected one cannot show it.
        prior_expectation, prior_variance = mixture_moments(
            prediction.prior_logits, prediction.prior_mean, prediction.prior_logvar
        )
        chunk = {
            "posterior_mean": prediction.posterior_mean,
            "posterior_sigma": sigma_from_logvar(prediction.posterior_logvar),
            "prior_mean": kl.prior_mean,
            "prior_sigma": sigma_from_logvar(kl.prior_logvar),
            "prior_expectation": prior_expectation,
            "prior_variance": prior_variance,
            "component_kl_per_dim": kl.per_dimension,
            "component_kl_mean_term_per_dim": kl.mean_term_per_dimension,
            "component_kl_variance_term_per_dim": kl.variance_term_per_dimension,
            "displacement_sigma": kl.displacement_sigma,
            "charged_kl": kl.charged.unsqueeze(-1),
        }
        for name, value in chunk.items():
            collected.setdefault(name, []).append(value.detach().cpu().double())
    return {name: torch.cat(values, dim=0) for name, values in collected.items()}


def probe_checkpoint(arm: Arm, series: list, device: torch.device) -> dict:
    """The reference probe's table for one checkpoint, over the whole split."""
    d = densities(arm, series, device)
    prior_mean, prior_sigma = d["prior_mean"], d["prior_sigma"]
    # What a prior SAMPLE spans across the cohort: the MIXTURE's own per-flight variance
    # (within-component width + between-component spread, `mixture_moments`) plus the spread
    # of its per-flight mean between flights. N(0, I) — what `predict --latent-random` draws
    # from — is 1.0 on this scale, and it beat the trained prior's best-of-6 in L2.d/e'.
    # Population spread (correction=0) everywhere a cohort is summarized: the cohort IS
    # the population here, and a one-flight `--limit 1` smoke test stays defined.
    prior_total_variance = (
        d["prior_variance"].mean(dim=0) + d["prior_expectation"].var(dim=0, correction=0)
    )
    displacement = d["displacement_sigma"]
    return {
        "checkpoint": str(arm.path),
        "latent_dim": int(arm.config.latent_dim),
        "prior_components": int(arm.config.latent_prior_components),
        "component": (
            "the single Gaussian" if arm.config.latent_prior_components == 1
            else "the component most responsible for the sampled z, per flight"
        ),
        "flights": len(series),
        "beta": arm.config.latent_beta,
        "free_bits_nats": arm.config.latent_free_bits_nats,
        "posterior_init_std": arm.config.latent_posterior_init_std,
        "beta_warmup_epochs": int(arm.config.latent_beta_warmup_epochs),
        "aux_duration_weight": arm.config.latent_aux_duration_weight,
        "prior_mean_across_flight_std": prior_mean.std(dim=0, correction=0).tolist(),
        "prior_sigma_median": _quantile(prior_sigma, MEDIAN, dim=0).tolist(),
        "posterior_mean_across_flight_std": d["posterior_mean"].std(dim=0, correction=0).tolist(),
        "posterior_sigma_median": _quantile(d["posterior_sigma"], MEDIAN, dim=0).tolist(),
        "displacement_sigma_median_per_dim": _quantile(displacement, MEDIAN, dim=0).tolist(),
        "displacement_sigma_median": float(_quantile(displacement, MEDIAN)),
        "displacement_sigma_p90": float(_quantile(displacement, 0.9)),
        "component_kl_per_dim_nats": d["component_kl_per_dim"].mean(dim=0).tolist(),
        "component_kl_mean_term_per_dim_nats":
            d["component_kl_mean_term_per_dim"].mean(dim=0).tolist(),
        "component_kl_variance_term_per_dim_nats":
            d["component_kl_variance_term_per_dim"].mean(dim=0).tolist(),
        "component_kl_nats_per_flight": float(d["component_kl_per_dim"].sum(dim=1).mean()),
        "component_kl_mean_term_nats_per_flight":
            float(d["component_kl_mean_term_per_dim"].sum(dim=1).mean()),
        "component_kl_variance_term_nats_per_flight":
            float(d["component_kl_variance_term_per_dim"].sum(dim=1).mean()),
        "charged_kl_nats_per_flight": float(d["charged_kl"].mean()),
        "prior_total_std": float(prior_total_variance.mean().sqrt()),
    }


def _vector(values: list[float], digits: int = 3) -> str:
    return " ".join(f"{value:.{digits}f}" for value in values)


def render(payload: dict) -> str:
    lines = [
        f"latent probe — split {payload['split']}, {payload['device']}"
        + (f", LIMIT {payload['limit']} (a prefix of the split, not a sample)"
           if payload["limit"] else ""),
    ]
    for label, block in payload["checkpoints"].items():
        share = (
            block["component_kl_mean_term_nats_per_flight"]
            / block["component_kl_nats_per_flight"]
        )
        lines += [
            "",
            f"== {label}  ({block['flights']} flights, latent_dim {block['latent_dim']}, "
            f"K={block['prior_components']}, beta {block['beta']:g}, free bits "
            f"{block['free_bits_nats']:g}/dim, q-std init {block['posterior_init_std']:g}"
            + (f", beta warm-up {block['beta_warmup_epochs']}" if block["beta_warmup_epochs"] else "")
            + (f", aux-T {block['aux_duration_weight']:g}" if block["aux_duration_weight"] else "")
            + ")",
            f"  prior  mean: across-flight std per dim   {_vector(block['prior_mean_across_flight_std'])}",
            f"  prior  sigma: median per dim             {_vector(block['prior_sigma_median'])}",
            f"  post   mean: across-flight std per dim   {_vector(block['posterior_mean_across_flight_std'])}",
            f"  post   sigma: median per dim             {_vector(block['posterior_sigma_median'])}",
            f"  |q mean − p mean| / p sigma, median/dim  {_vector(block['displacement_sigma_median_per_dim'], 2)}",
            f"  component KL per dim (nats, mean/flight) {_vector(block['component_kl_per_dim_nats'])}",
            f"    of which the MEAN term                 {_vector(block['component_kl_mean_term_per_dim_nats'])}",
            f"    of which the VARIANCE term             {_vector(block['component_kl_variance_term_per_dim_nats'])}",
            f"  totals per flight: component KL {block['component_kl_nats_per_flight']:.3f} "
            f"nats = mean {block['component_kl_mean_term_nats_per_flight']:.3f} + variance "
            f"{block['component_kl_variance_term_nats_per_flight']:.3f} ({share:.0%} in the "
            f"mean); charged {block['charged_kl_nats_per_flight']:.3f}",
            f"  {displacement_verdict(block['displacement_sigma_median'])} "
            f"(p90 {block['displacement_sigma_p90']:.3f})",
            f"  prior total std {block['prior_total_std']:.3f} (the mixture's own moments; "
            "N(0, I) = 1.0, and a narrower prior is why the random-latent control can beat "
            "the trained one)",
        ]
    lines.append("")
    if payload["limit"]:
        lines.append(LIMIT_MEANING)
    lines.append("The posterior reads the truth's future: these are TRAINING-side densities, "
                 "never a prediction result.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", action="append", metavar="LABEL=PATH",
                        help="a latent control checkpoint; repeat for more arms")
    parser.add_argument("--split", default="val", choices=SPLITS)
    parser.add_argument("--limit", type=int, default=0,
                        help="probe only the FIRST N flight keys of the split — a prefix, "
                             "not a sample, and the tables move with it (25 % on the "
                             "displacement median between 100 and 200 KRDU flights). A smoke "
                             "test; never quote a limited table beside a full-split one")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory (immutable; must not exist)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    if args.split == FORBIDDEN_SPLIT:
        parser.error(f"the {FORBIDDEN_SPLIT} split is sealed: this is a diagnostic with no gate")
    if args.limit < 0:
        parser.error("--limit must be non-negative (0 = the whole split)")

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; the latent probe is an immutable artifact")
    device = resolve_device(args.device)
    grid = Grid(split=args.split, bins_m=(), min_future_s=0.0, batch_size=None, limit=args.limit)
    # Load every checkpoint FIRST: the CTA/intent refusals, the missing split and the
    # provenance check live in the shared loader, and the latent refusal is here. A
    # rejected invocation must leave nothing behind.
    loaded = [
        load_arm(label, path, grid, device, instrument=INSTRUMENT)
        for label, path in arms.items()
    ]
    for arm in loaded:
        if arm.config.latent_dim < 1:
            raise SystemExit(
                f"{arm.label} ({arm.path}): latent_dim=0 — this checkpoint has no posterior "
                "to probe (its control head is deterministic)"
            )

    payload = {
        "schema": RESULT_SCHEMA,
        "split": args.split,
        "limit": args.limit,
        "limit_meaning": LIMIT_MEANING,
        "device": str(device),
        "measured": "the training-side densities q(z | future) and p(z | context) at the "
                    "fixed L-1 anchor; the posterior reads the future by construction",
        "checkpoints": {
            arm.label: probe_checkpoint(arm, cohort_series(arm, grid), device)
            for arm in loaded
        },
    }
    text = render(payload)
    # The artifact appears whole. The measurement is finished BEFORE anything is written,
    # so a run that dies mid-measurement leaves nothing at all; the staging directory covers
    # the write itself, whose failure leaves a `.partial-*` beside the artifact rather than
    # a half-written table under the name a reader will cite.
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=out.parent, prefix=f"{out.name}.partial-"))
    (staged / "latent_probe.json").write_text(json.dumps(payload, indent=2))
    (staged / "latent_probe.txt").write_text(text)
    staged.chmod(0o755)                     # mkdtemp is 0700; the artifact is readable
    staged.rename(out)
    print()
    print(text, end="")
    print(f"wrote {out / 'latent_probe.txt'} and {out / 'latent_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
