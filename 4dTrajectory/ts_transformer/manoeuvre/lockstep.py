"""Lockstep (plan §2.7): the executor re-asked every ``segment_s`` on its own flown rows, under
three protocols that differ only in WHERE the code comes from.

    C        the TRUTH's code at this absolute time (time-indexed: the code of the truth segment
             that starts when this round starts, `sequences.flight_sequences`) — the executor alone;
             gate X. Past the truth's last full segment the last code is HELD until the flight
             crosses or the budget ends (counted per flight, ``held_asks``).
    A        the prior's top-1 on the FLOWN history: every leg flown is tokenised back through the
             codebook (the flown segment's rows and the flown state at its start), the prior reads
             that code sequence with the flown boundary states, and its answer is the next z — the
             closed loop closed in code space; gate E. The prior's ``landed`` ends the flight.
    A-truth  the prior's top-1 on the TRUTH history (time-indexed prefix, truth codes and truth
             states) while the executor flies its own rows: what v2 §12.3 wanted — the prior's
             error alone, the coupling removed. Past the truth's last full segment there is no
             truth prefix to read: the flight ends (``truth-exhausted``).

One round = ``segment_s`` = the executor's horizon: the forecast is exactly one segment and is
flown whole (the one-shot variant, plan §2.6) unless it crosses the threshold ON THE FINAL
inside the segment (`inference.forecast.cut_at_threshold_crossing`: the flight ends,
``crossed``) or the flight's budget runs out (``horizon``). The budget is
`closing_horizon_s` of the truth's duration at the first ask — T₀ + max(30 s, 0.1·T₀), the
rule copied from the archived plan oracle so "established" is read over one budget in every
campaign — and is a CAP only: under A the prior's ``landed`` is the decision, the cap is what
stops a flight that never says so. Under A and A-truth every round ALSO flies the truth's code
from the same flown state, and ``e_plan`` is the distance between the two legs' end points
(plan §3.2) — the prior's error in the executor's currency, beside the step error ``e_track``
(the leg's end against the truth at that time).

The reference verdicts are the plan oracle's (`geometry.flyability`: fully flyable over the
flown geodetic states; established = crossed the threshold on the final), and the row carries
`observed_series_metrics`' ADE / FDE and the time-free geometry (chamfer, Fréchet).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.approach_difficulty import approach_difficulty
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import FlightSeries, Normalizer, truth_duration_s
from ts_transformer.data.time_grids import ROW_TOLERANCE_S
from ts_transformer.geometry import geometric_metrics
from ts_transformer.geometry.flyability import flyability_summary, required_controls
from ts_transformer.inference.export import observed_series_metrics
from ts_transformer.inference.forecast import Forecast, concatenate, cut_at_threshold_crossing, cut_rows
from ts_transformer.inference.receding import cut_at_lead, displacement_at, rolled_series
from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.manoeuvre.prior import ManoeuvrePrior, Step, collate, last_position_step
from ts_transformer.manoeuvre.segments import segment_rows, state_row
from ts_transformer.manoeuvre.sequences import CodeSequence, flight_sequences, state_token
from ts_transformer.manoeuvre.tokenizer import Codebook
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.plan_token import MANOEUVRE_Z_KEY
from ts_transformer.outputs.dynamics.context import dynamics_arrays

PROTOCOL_C = "C"
PROTOCOL_A = "A"
PROTOCOL_A_TRUTH = "A-truth"
PROTOCOLS = (PROTOCOL_C, PROTOCOL_A, PROTOCOL_A_TRUTH)
#: How a flight ended.
ENDED_CROSSED = "crossed"            # the threshold on the final, inside a leg
ENDED_LANDED = "landed"              # the prior said so (A / A-truth)
ENDED_HORIZON = "horizon"            # the budget
ENDED_TRUTH_EXHAUSTED = "truth-exhausted"   # A-truth past the truth's last full segment
#: The closing budget — the archived `plan_oracle.closing_horizon_s`, copied here as the plan
#: says (§2.7): T₀ + max(HORIZON_SLACK_S, HORIZON_SLACK_FRACTION·T₀).
HORIZON_SLACK_S = 30.0
HORIZON_SLACK_FRACTION = 0.1
#: The leads (s from the first ask) the displacement is read at (plan §3.2).
LEADS_S = (60.0, 120.0, 180.0, 300.0)
#: The prior says the flight lands within the next segment above this probability.
LANDED_THRESHOLD = 0.5


def closing_horizon_s(T_s: float) -> float:
    return T_s + max(HORIZON_SLACK_S, HORIZON_SLACK_FRACTION * T_s)


@dataclass(frozen=True)
class Executor:
    model: torch.nn.Module
    config: TSConfig
    normalizer: Normalizer


@dataclass(frozen=True)
class Prior:
    model: ManoeuvrePrior
    vocabulary: TypeVocabulary


@dataclass
class FlightRun:
    series: FlightSeries
    truth: CodeSequence
    horizon_s: float
    legs: list[Forecast] = field(default_factory=list)
    flown_s: float = 0.0
    ended: str | None = None
    truncated: bool = False
    asks: int = 0
    held_asks: int = 0
    codes: list[int] = field(default_factory=list)            # the code flown each round
    flown_codes: list[int] = field(default_factory=list)      # A: the flown legs tokenised back
    flown_states: list[np.ndarray] = field(default_factory=list)   # A: the boundary states, x_0 first
    asks_e: list[dict[str, Any]] = field(default_factory=list)
    landed_fraction: float | None = None                       # the prior's, when it ended the flight
    landed_probabilities: list[float] = field(default_factory=list)
    landing_this_round: float | None = None                    # A / A-truth: fly this fraction of a segment, then end

    @property
    def anchor(self) -> int:
        return self.truth.anchor


def _polyline(run: FlightRun) -> tuple[np.ndarray, np.ndarray]:
    """The flown path on the flight's clock: the anchor's observed row, then every leg's rows."""
    a0 = run.anchor
    times = [float(run.series.times[a0])]
    values = [np.asarray(run.series.values[a0], dtype=np.float64)]
    for leg in run.legs:
        times.extend(np.asarray(leg.times, dtype=np.float64).tolist())
        values.extend(np.asarray(leg.values, dtype=np.float64))
    return np.asarray(times), np.stack(values)


def _history(run: FlightRun, anchor: int, dt_s: float) -> FlightSeries:
    if not run.legs:
        return run.series
    return rolled_series(run.series, run.anchor, concatenate(run.legs, run.anchor, 0.0), anchor, dt_s)


def _dynamics(executor: Executor, histories: Sequence[FlightSeries], anchor: int, z: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    config = executor.config
    rows = [
        dynamics_arrays(history, anchor, parameterization=config.control_thrust_parameterization,
                        condition_features=config.control_condition_features)
        for history in histories
    ]
    dynamics = {name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device) for name in rows[0]}
    dynamics[MANOEUVRE_Z_KEY] = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device)
    return dynamics


def _fly(executor: Executor, histories: Sequence[FlightSeries], anchor: int, z: np.ndarray, device: torch.device,
         batch_size: int) -> list[Forecast]:
    out: list[Forecast] = []
    for start in range(0, len(histories), batch_size):
        chunk = list(histories[start : start + batch_size])
        out.extend(forecast_control_batch(
            executor.model, chunk, executor.config, executor.normalizer, anchor, device,
            dynamics=_dynamics(executor, chunk, anchor, z[start : start + batch_size], device),
        ))
    return out


def _end_point(forecast: Forecast) -> np.ndarray:
    return np.asarray(forecast.values[-1], dtype=np.float64)[list(POSITION_IDX)]


def _prior_step(prior: Prior, codes: Sequence[Sequence[int]], states: Sequence[np.ndarray], runs: Sequence[FlightRun],
                codebook: Codebook, device: torch.device) -> Step:
    sequences = [
        CodeSequence(
            dataset_id=run.truth.dataset_id, flight_id=run.truth.flight_id, anchor=run.anchor, segment_s=codebook.segment_s,
            start_times=float(run.series.times[run.anchor]) + codebook.segment_s * np.arange(len(flown)),
            codes=np.asarray(flown, dtype=np.int64), z=np.zeros((len(flown), codebook.z_dim), dtype=np.float32),
            states=np.asarray(boundary, dtype=np.float32), landed_fraction=0.0,
            typecode=run.truth.typecode, runway_course_rad=run.truth.runway_course_rad,
        )
        for run, flown, boundary in zip(runs, codes, states, strict=True)
    ]
    targets = [np.zeros((len(seq.codes), codebook.z_dim), dtype=np.float32) for seq in sequences] if prior.model.config.continuous else None
    batch = collate(sequences, prior.model.config, prior.vocabulary, continuous_targets=targets).to(device)
    return last_position_step(prior.model.eval(), batch)


def fly(
    executor: Executor, codebook: Codebook, series: Sequence[FlightSeries], protocol: str, *,
    prior: Prior | None, device: torch.device, batch_size: int, log=None,
) -> list[FlightRun]:
    """Every flight of ``series`` from the executor's fixed anchor, one round at a time for the
    whole cohort, under ``protocol``."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol is one of {PROTOCOLS}, got {protocol!r}")
    if (protocol != PROTOCOL_C) != (prior is not None):
        raise ValueError("protocols A and A-truth take the prior; protocol C flies the truth's codes and takes none")
    config = executor.config
    segment_s, dt_s = codebook.segment_s, codebook.dt_s
    if config.control_horizon_s != segment_s or config.dt_s != dt_s:
        raise ValueError(f"the executor's horizon {config.control_horizon_s:g} s / {config.dt_s:g} s is not the codebook's {segment_s:g} s / {dt_s:g} s")
    a0 = default_anchor(config)
    step_rows = int(round(segment_s / dt_s))
    truths = flight_sequences(series, codebook, a0)
    runs = [FlightRun(series=item, truth=truth, horizon_s=closing_horizon_s(truth_duration_s(item, a0)))
            for item, truth in zip(series, truths, strict=True)]
    for run in runs:
        run.flown_states.append(state_token(run.series.values[a0]))
    round_index = 0
    while any(run.ended is None for run in runs):
        active = [run for run in runs if run.ended is None]
        anchor = a0 + round_index * step_rows
        histories = [_history(run, anchor, dt_s) for run in active]
        # WHERE the code comes from
        z_rows: list[np.ndarray] = []
        codes: list[int] = []
        if protocol == PROTOCOL_C:
            for run in active:
                index = min(round_index, run.truth.length - 1)
                run.held_asks += int(round_index >= run.truth.length)
                codes.append(int(run.truth.codes[index]))
                z_rows.append(run.truth.z[index])
        else:
            if protocol == PROTOCOL_A_TRUTH:
                exhausted = [run for run in active if round_index > run.truth.length]
                for run in exhausted:
                    run.ended = ENDED_TRUTH_EXHAUSTED
                active = [run for run in active if run.ended is None]
                histories = [_history(run, anchor, dt_s) for run in active]
                if not active:
                    break
                prefix_codes = [run.truth.codes[:round_index].tolist() for run in active]
                prefix_states = [run.truth.states[: round_index + 1] for run in active]
            else:
                prefix_codes = [run.flown_codes for run in active]
                prefix_states = [np.stack(run.flown_states) for run in active]
            step = _prior_step(prior, prefix_codes, prefix_states, active, codebook, device)
            landed = step.landed_probability.cpu().numpy()
            fraction = step.landed_fraction.cpu().numpy()
            # the codebook's tokenizer lives on the CPU; the prior may be on the GPU
            step_codes = None if step.code is None else step.code.cpu()
            step_z = None if step.z is None else step.z.cpu()
            for row, run in enumerate(active):
                run.landed_probabilities.append(float(landed[row]))
                # the prior says the flight lands inside this segment: the top-1 code is flown for
                # that fraction of it (the last, partial segment), then the flight ends
                run.landing_this_round = float(fraction[row]) if landed[row] > LANDED_THRESHOLD else None
                if step_codes is not None:
                    codes.append(int(step_codes[row]))
                    z_rows.append(codebook.tokenizer.codes_to_z(step_codes[row : row + 1]).numpy()[0])
                else:
                    z_rows.append(step_z[row].numpy())
                    codes.append(int(codebook.tokenizer.z_to_codes(step_z[row : row + 1]).numpy()[0]))
        z = np.stack(z_rows)
        forecasts = _fly(executor, histories, anchor, z, device, batch_size)
        plan_legs: list[Forecast] | None = None
        if protocol != PROTOCOL_C:
            # the truth's code from the SAME state: e_plan's other leg
            truth_z = np.stack([run.truth.z[min(round_index, run.truth.length - 1)] for run in active])
            plan_legs = _fly(executor, histories, anchor, truth_z, device, batch_size)
        for row, (run, history, forecast) in enumerate(zip(active, histories, forecasts, strict=True)):
            run.codes.append(codes[row])
            _fly_leg(run, history, forecast, segment_s=segment_s, round_index=round_index)
            if plan_legs is not None and run.legs:
                run.asks_e[-1]["e_plan_m"] = float(np.linalg.norm(_end_point(run.legs[-1]) - _end_point(cut_at_lead(plan_legs[row], float(np.sum(run.legs[-1].sample_durations_s))))))
                run.asks_e[-1]["truth_code"] = int(run.truth.codes[min(round_index, run.truth.length - 1)])
            if protocol == PROTOCOL_A and run.legs and run.legs[-1] is not None:
                _tokenise_flown_leg(run, codebook, segment_s, dt_s, round_index)
        if log is not None:
            log(f"    {protocol} round {round_index}: {len(active)} flights at anchor {anchor}, "
                f"{sum(1 for run in runs if run.ended is None)} continue")
        round_index += 1
    return runs


def _fly_leg(run: FlightRun, history: FlightSeries, forecast: Forecast, *, segment_s: float, round_index: int) -> None:
    """Fly this round's segment whole; end at the crossing on the final or at the budget."""
    crossed = cut_at_threshold_crossing(forecast, history)
    remaining_s = run.horizon_s - run.flown_s
    landing_s = None if run.landing_this_round is None else run.landing_this_round * segment_s
    span_s = segment_s if landing_s is None else min(segment_s, landing_s)
    offsets = np.cumsum(forecast.sample_durations_s)
    if crossed.truncated_at_threshold and crossed.final_time_s <= min(span_s, remaining_s) + ROW_TOLERANCE_S:
        leg, run.ended, run.truncated = crossed, ENDED_CROSSED, True
    elif remaining_s <= span_s + ROW_TOLERANCE_S:
        rows = int(np.searchsorted(offsets, remaining_s + ROW_TOLERANCE_S, side="right"))
        leg, run.ended = (cut_rows(forecast, rows) if rows else None), ENDED_HORIZON
    elif landing_s is not None:
        rows = max(int(np.searchsorted(offsets, landing_s + ROW_TOLERANCE_S, side="right")), 1)
        leg, run.ended, run.landed_fraction = cut_rows(forecast, rows), ENDED_LANDED, run.landing_this_round
    else:
        leg = cut_at_lead(forecast, segment_s)
    if leg is None:
        return
    run.legs.append(leg)
    run.asks += 1
    run.flown_s += float(np.sum(leg.sample_durations_s))
    run.asks_e.append({
        "round": round_index, "lead_s": run.flown_s, "code": run.codes[-1],
        "e_track_m": displacement_at(run.series, leg, run.anchor, float(leg.times[-1])),
    })


def _tokenise_flown_leg(run: FlightRun, codebook: Codebook, segment_s: float, dt_s: float, round_index: int) -> None:
    """The leg just flown, read back as the prior will read it: its rows from the flown polyline
    in the start frame, through the codebook, and the flown state at its end."""
    times, values = _polyline(run)
    start = float(run.series.times[run.anchor]) + segment_s * round_index
    if times[-1] < start + segment_s - ROW_TOLERANCE_S:
        return   # the flight ended inside this segment; nothing whole to tokenise
    rows = segment_rows(times, values, start, segment_s, dt_s)
    state = np.array([np.interp(start, times, values[:, c]) for c in range(values.shape[1])])
    code, _z = codebook.encode(rows.astype(np.float32), state.astype(np.float32))
    run.flown_codes.append(int(code))
    end_state = np.array([np.interp(start + segment_s, times, values[:, c]) for c in range(values.shape[1])])
    run.flown_states.append(state_token(end_state))


# ── the rows ─────────────────────────────────────────────────────────────────

def whole_forecast(run: FlightRun) -> Forecast:
    """The flight's legs as one forecast from the anchor: its predicted end is the prior's
    arrival when it said landed (the flown span plus its fraction of a segment), else the span."""
    last = run.legs[-1]
    span = float(sum(np.sum(leg.sample_durations_s) for leg in run.legs))
    predicted = span + (run.landed_fraction * run.truth.segment_s if run.landed_fraction is not None else 0.0)
    whole = concatenate(run.legs, run.anchor, predicted)
    return replace(whole, truncated_at_threshold=run.truncated, horizon_capped=run.ended == ENDED_HORIZON)


def _forecast_geometry(series: FlightSeries, forecast: Forecast) -> dict[str, float]:
    # MUST match `experiments/support.forecast_geometry` (a runner module the package cannot
    # import): both paths [N, 4] (e, n, u, t) in the flight's chart, the truth after the anchor.
    anchor_time = float(series.times[forecast.anchor])
    future = series.supervision_times > anchor_time
    truth = np.column_stack([
        np.asarray(series.supervision_values, dtype=np.float64)[future][:, list(POSITION_IDX)],
        np.asarray(series.supervision_times, dtype=np.float64)[future] - anchor_time,
    ])
    predicted = np.column_stack([
        np.asarray(forecast.values, dtype=np.float64)[:, list(POSITION_IDX)], np.cumsum(forecast.sample_durations_s),
    ])
    return geometric_metrics.path_metrics(predicted, truth)


def reference_verdicts(series: FlightSeries, forecast: Forecast) -> dict[str, Any]:
    """The plan oracle's reading: fully flyable over the flown geodetic states, established =
    crossed the threshold on the final."""
    rows = [
        {"t": float(t), "lat": float(g[0]), "lon": float(g[1]), "alt": float(g[2]),
         "V": float(g[3]), "psi": float(g[4]), "gamma": float(g[5]), "m": float(g[6])}
        for t, g in zip(np.cumsum(forecast.sample_durations_s), forecast.geodetic_values, strict=True)
    ]
    controls = required_controls(rows, series.scenario.aircraft, aero=series.scenario.aero)
    summary = flyability_summary(controls, aircraft_code=str(series.scenario.aircraft.code))
    return {
        "fully_flyable": bool(summary["fully_flyable"]), "violations": dict(summary["violations"]),
        "established": bool(forecast.truncated_at_threshold),
    }


def flight_row(run: FlightRun, points: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One flight's row (and the record metrics): the reference verdicts, ADE / FDE, the
    geometry, the displacement at every lead, every ask's step error and e_plan, how it ended."""
    forecast = whole_forecast(run)
    metrics = observed_series_metrics(run.series, forecast, points=points)
    geometry = _forecast_geometry(run.series, forecast)
    origin = float(run.series.times[run.anchor])
    difficulty = approach_difficulty(run.series, run.anchor)
    row = {
        "flight_id": run.series.flight_id,
        "reference": reference_verdicts(run.series, forecast),
        "ended": run.ended, "truncated_at_threshold": run.truncated, "asks": run.asks, "held_asks": run.held_asks,
        "codes": list(run.codes), "truth_codes": run.truth.codes.tolist(),
        "flown_codes": list(run.flown_codes),
        "landed_fraction": run.landed_fraction, "landed_probabilities": list(run.landed_probabilities),
        "ade_m": float(metrics["ade_m"]), "fde_m": float(metrics["fde_m"]),
        "final_time_error_s": float(metrics["final_time_error_s"]),
        "chamfer_m": float(geometry["chamfer_m"]), "frechet_m": float(geometry["frechet_m"]),
        "at": {f"{lead:g}": displacement_at(run.series, forecast, run.anchor, origin + lead) for lead in LEADS_S},
        "asks_e": run.asks_e,
        "route_tortuosity": difficulty.route_tortuosity, "established_at_anchor": difficulty.established_at_anchor,
        "remaining_path_m": difficulty.remaining_path_m,
    }
    return row, metrics


__all__ = [
    "ENDED_CROSSED", "ENDED_HORIZON", "ENDED_LANDED", "ENDED_TRUTH_EXHAUSTED", "HORIZON_SLACK_FRACTION",
    "HORIZON_SLACK_S", "LANDED_THRESHOLD", "LEADS_S", "PROTOCOLS", "PROTOCOL_A", "PROTOCOL_A_TRUTH",
    "PROTOCOL_C", "Executor", "FlightRun", "Prior", "closing_horizon_s", "flight_row", "fly",
    "reference_verdicts", "whole_forecast",
]
