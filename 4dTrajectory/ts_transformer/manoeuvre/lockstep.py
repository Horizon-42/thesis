"""Lockstep (plan §2.7): the executor predicts again every ``segment_s`` on its own flown rows, under
three protocols that differ only in WHERE the code comes from.

    C        the TRUTH's code at this absolute time (time-indexed: the code of the truth segment
             that starts when this round starts, `sequences.flight_sequences`) — the executor alone;
             gate X. Past the truth's last full segment the last code is HELD until the flight
             crosses or the budget ends (counted per flight, ``held_predictions``).
    A        the prior's top-1 on the FLOWN history: every leg flown is tokenised back through the
             codebook (the flown segment's rows and the flown state at its start), the prior reads
             that code sequence with the flown boundary states, and its answer is the next z — the
             closed loop closed in code space; gate E. The prior's ``landed`` ends the flight.
    A-truth  the prior's top-1 on the TRUTH history (time-indexed prefix, truth codes and truth
             states) while the executor flies its own rows: what v2 §12.3 wanted — the prior's
             error alone, the coupling removed. Past the truth's last full segment there is no
             truth prefix to read: the flight ends (``truth-exhausted``).
    none     NO code and NO codebook: a no-token executor (``plan_conditioning = off``) predicting
             again every segment on its own flown rows — the baseline every protocol above is read
             against (2026-09-18: the command-vocabulary executor's closed-loop lead over the
             learned codebook turned out to be code-blindness, so "does the code help" needs the
             executor WITHOUT one under the same rounds and the same budget), and the two-tier
             v3 plan's stage A reading (§3.1). The segment is the executor's horizon; the row
             carries no code columns.

The first prediction is made at the executor's fixed anchor, L−1 (`config.default_anchor`: no floor under v3 —
the first row with a complete lookback, L seconds after entry). The second reading of v3 §3.1
starts from a REMAINING-PATH bin instead (`from_remaining_path`): each flight is first seen at the
row `anchor_grid.bin_anchor` places at the bin, so the same lockstep from L−1 of the cut flight
IS the lockstep from that row of the whole flight.

One round = ``segment_s`` = the executor's horizon: the forecast is exactly one segment and is
flown whole (the one-shot variant, plan §2.6) unless it crosses the threshold ON THE FINAL
inside the segment (`inference.forecast.cut_at_threshold_crossing`: the flight ends,
``crossed``) or the flight's budget runs out (``horizon``). The budget is
`closing_horizon_s` of the truth's duration at the first prediction — T₀ + max(30 s, 0.1·T₀), the
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

from ts_transformer.config import CONTROL_DURATION_UNIFORM, PLAN_CONDITIONING_OFF, TSConfig, default_anchor
from ts_transformer.data.anchor_grid import anchors_for_bin, remaining_path_profiles
from ts_transformer.data.approach_difficulty import approach_difficulty
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import FlightSeries, Normalizer, effective_min_future_s, series_from_row, truth_duration_s
from ts_transformer.data.time_grids import ROW_TOLERANCE_S
from ts_transformer.geometry import geometric_metrics
from ts_transformer.geometry.flyability import flyability_summary, required_controls
from ts_transformer.inference.export import observed_series_metrics
from ts_transformer.inference.forecast import Forecast, concatenate, cut_at_threshold_crossing, cut_rows
from ts_transformer.inference.receding import cut_at_lead, displacement_at, rolled_series
from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.manoeuvre.prior import ManoeuvrePrior, Step, collate, last_position_step
from ts_transformer.manoeuvre.segments import segment_rows
from ts_transformer.manoeuvre.sequences import CodeSequence, flight_sequences, state_token
from ts_transformer.manoeuvre.tokenizer import Codebook
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.plan_token import MANOEUVRE_Z_KEY
from ts_transformer.outputs.dynamics.context import dynamics_arrays

PROTOCOL_C = "C"
PROTOCOL_A = "A"
PROTOCOL_A_TRUTH = "A-truth"
PROTOCOL_NONE = "none"
PROTOCOLS = (PROTOCOL_C, PROTOCOL_A, PROTOCOL_A_TRUTH, PROTOCOL_NONE)
#: The code column of a protocol-``none`` prediction: no code was flown.
CODE_NONE = -1
#: How a flight ended.
ENDED_CROSSED = "crossed"            # the threshold on the final, inside a leg
ENDED_LANDED = "landed"              # the prior said so (A / A-truth)
ENDED_HORIZON = "horizon"            # the budget
ENDED_TRUTH_EXHAUSTED = "truth-exhausted"   # A-truth past the truth's last full segment
#: The closing budget — the archived `plan_oracle.closing_horizon_s`, copied here as the plan
#: says (§2.7): T₀ + max(HORIZON_SLACK_S, HORIZON_SLACK_FRACTION·T₀).
HORIZON_SLACK_S = 30.0
HORIZON_SLACK_FRACTION = 0.1
#: The leads (s from the first prediction) the displacement is read at (plan §3.2).
LEADS_S = (60.0, 120.0, 180.0, 300.0)
#: The prior says the flight lands within the next segment above this probability.
LANDED_THRESHOLD = 0.5


def closing_horizon_s(T_s: float) -> float:
    return T_s + max(HORIZON_SLACK_S, HORIZON_SLACK_FRACTION * T_s)


def required_positions(max_truth_duration_s: float, segment_s: float, longest_truth_segments: int) -> int:
    """How many positions a prior must hold to be queried at every round of a lockstep: the
    BUDGET's rounds (`closing_horizon_s` of the longest truth, in segments, rounded up) plus
    the BOS position and one spare — never fewer than the longest truth sequence needs. Sized
    from the truth alone, a prior overflowed `collate` under protocol A on the long flights
    (review 2026-09-18 B1: Δ = 20 s past T ≈ 400 s)."""
    rounds = int(np.ceil(closing_horizon_s(max_truth_duration_s) / segment_s))
    return max(rounds + 2, longest_truth_segments + 2)


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
    anchor: int                                  # the first prediction's row
    truth: CodeSequence | None                   # the truth's code sequence; None under protocol none
    horizon_s: float
    legs: list[Forecast] = field(default_factory=list)
    flown_s: float = 0.0
    ended: str | None = None
    truncated: bool = False
    predictions: int = 0                         # rounds flown
    held_predictions: int = 0                    # C: rounds past the truth's last full segment (the last code held)
    codes: list[int] = field(default_factory=list)            # the code flown each round
    flown_codes: list[int] = field(default_factory=list)      # A: the flown legs tokenised back
    flown_states: list[np.ndarray] = field(default_factory=list)   # A: the boundary states, x_0 first
    rounds: list[dict[str, Any]] = field(default_factory=list)       # one record per round: lead, code, e_track, e_plan
    landed_fraction: float | None = None                       # the prior's, when it ended the flight
    landed_probabilities: list[float] = field(default_factory=list)
    landing_this_round: float | None = None                    # A / A-truth: fly this fraction of a segment, then end


def from_remaining_path(series: Sequence[FlightSeries], config: TSConfig, target_m: float) -> tuple[list[FlightSeries], dict[str, int]]:
    """The cohort first seen at the remaining-path bin ``target_m``: every flight cut
    (`series_from_row`) so that the row `anchor_grid.bin_anchor` places at the bin — the sample
    nearest ``target_m`` of path left to fly, with a complete lookback before it and the
    executor's horizon of truth after it — becomes the executor's fixed anchor (`default_anchor`:
    L−1, or a floored config's floor). ``(the cut flights in cohort order, {dataset id: that row
    in the whole flight})``; a flight with no admissible row at the bin is absent from both (the
    caller counts it)."""
    anchors = anchors_for_bin(series, remaining_path_profiles(series), target_m, seq_len=config.seq_len,
                              min_future_s=effective_min_future_s(config), minimum_anchor_index=config.anchor_floor_index)
    # the bin's row lands on the executor's own fixed anchor (L−1, or its floor under a floored config)
    cut = [series_from_row(series[index], anchor - default_anchor(config)) for index, anchor in anchors.items()]
    return cut, {series[index].dataset_id: anchor for index, anchor in anchors.items()}


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


def _dynamics(executor: Executor, histories: Sequence[FlightSeries], anchor: int, z: np.ndarray | None,
              device: torch.device) -> dict[str, torch.Tensor]:
    """The executor's dynamics rows at ``anchor``; ``z`` (the given code vector) is handed over
    under `MANOEUVRE_Z_KEY`, or nothing is under protocol ``none`` (no key: a no-token executor)."""
    config = executor.config
    rows = [
        dynamics_arrays(history, anchor, parameterization=config.control_thrust_parameterization,
                        condition_features=config.control_condition_features)
        for history in histories
    ]
    dynamics = {name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device) for name in rows[0]}
    if z is not None:
        dynamics[MANOEUVRE_Z_KEY] = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device)
    return dynamics


def _fly(executor: Executor, histories: Sequence[FlightSeries], anchor: int, z: np.ndarray | None, device: torch.device,
         batch_size: int) -> list[Forecast]:
    out: list[Forecast] = []
    for start in range(0, len(histories), batch_size):
        chunk = list(histories[start : start + batch_size])
        out.extend(forecast_control_batch(
            executor.model, chunk, executor.config, executor.normalizer, anchor, device,
            dynamics=_dynamics(executor, chunk, anchor, None if z is None else z[start : start + batch_size], device),
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
    executor: Executor, codebook: Codebook | None, series: Sequence[FlightSeries], protocol: str, *,
    prior: Prior | None, device: torch.device, batch_size: int, log=None,
) -> list[FlightRun]:
    """Every flight of ``series`` from the executor's fixed anchor, one round at a time for the
    whole cohort, under ``protocol``. The codebook is the coded protocols' vocabulary; protocol
    ``none`` takes none."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol is one of {PROTOCOLS}, got {protocol!r}")
    if (protocol in (PROTOCOL_A, PROTOCOL_A_TRUTH)) != (prior is not None):
        raise ValueError("protocols A and A-truth take the prior; protocol C flies the truth's codes and protocol none no code: neither takes one")
    if (protocol == PROTOCOL_NONE) != (codebook is None):
        raise ValueError("protocol none takes no codebook (a no-token executor has no vocabulary); every coded protocol takes one")
    config = executor.config
    if (protocol == PROTOCOL_NONE) != (config.plan_conditioning == PLAN_CONDITIONING_OFF):
        raise ValueError(f"protocol none flies a no-token executor (plan_conditioning = off) and the coded protocols a "
                         f"manoeuvre-code one; got {protocol!r} with plan_conditioning {config.plan_conditioning!r}")
    segment_s, dt_s = config.control_horizon_s, config.dt_s
    if not segment_s:
        raise ValueError("the lockstep flies one fixed segment per round: the executor needs control_horizon_s > 0")
    if codebook is not None and (codebook.segment_s != segment_s or codebook.dt_s != dt_s):
        raise ValueError(f"the executor's horizon {segment_s:g} s / {dt_s:g} s is not the codebook's {codebook.segment_s:g} s / {codebook.dt_s:g} s")
    if config.control_duration_parameterization != CONTROL_DURATION_UNIFORM:
        # e_plan cuts the truth-code leg at the flown leg's span (`cut_at_lead` needs a row
        # there): the two legs share their query grid only under uniform segment durations
        raise ValueError("the lockstep's e_plan reading needs uniform control segment durations (the P1.4 recipe's)")
    a0 = default_anchor(config)
    step_rows = int(round(segment_s / dt_s))
    truths = flight_sequences(series, codebook, a0) if codebook is not None else [None] * len(series)
    runs = [FlightRun(series=item, anchor=a0, truth=truth, horizon_s=closing_horizon_s(truth_duration_s(item, a0)))
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
        if protocol == PROTOCOL_NONE:
            codes = [CODE_NONE] * len(active)
        elif protocol == PROTOCOL_C:
            for run in active:
                index = min(round_index, run.truth.length - 1)
                run.held_predictions += int(round_index >= run.truth.length)
                codes.append(int(run.truth.codes[index]))
                z_rows.append(run.truth.z[index])
        else:
            if protocol == PROTOCOL_A_TRUTH:
                for run in active:
                    if round_index > run.truth.length:
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
        z = np.stack(z_rows) if z_rows else None
        forecasts = _fly(executor, histories, anchor, z, device, batch_size)
        plan_legs: list[Forecast] | None = None
        if protocol in (PROTOCOL_A, PROTOCOL_A_TRUTH):
            # the truth's code from the SAME state: e_plan's other leg
            truth_z = np.stack([run.truth.z[min(round_index, run.truth.length - 1)] for run in active])
            plan_legs = _fly(executor, histories, anchor, truth_z, device, batch_size)
        for row, (run, history, forecast) in enumerate(zip(active, histories, forecasts, strict=True)):
            leg = _fly_leg(run, history, forecast, segment_s=segment_s, round_index=round_index, code=codes[row])
            if leg is None:
                continue   # the budget ended before this round's first row: nothing flown, nothing recorded
            if plan_legs is not None:
                run.rounds[-1]["e_plan_m"] = _plan_error(leg, plan_legs[row], history)
                run.rounds[-1]["truth_code"] = int(run.truth.codes[min(round_index, run.truth.length - 1)])
            # every whole leg is tokenised back — protocol A reads the flown codes at the next
            # round, and a protocol-C run on the TRAIN split is the closed-loop training's input
            # (plan §2.7 step 1: the flown history's codes, the truth's as the labels)
            if codebook is not None:
                _tokenise_flown_leg(run, codebook, segment_s, dt_s, round_index)
        if log is not None:
            log(f"    {protocol} round {round_index}: {len(active)} flights at anchor {anchor}, "
                f"{sum(1 for run in runs if run.ended is None)} continue")
        round_index += 1
    return runs


def _plan_error(leg: Forecast, plan_leg: Forecast, history: FlightSeries) -> float:
    """e_plan: the flown leg's end against the truth-code leg's end, the latter cut where it
    crosses the threshold on the final (as the flown one is) or at the flown leg's span."""
    plan_cut = cut_at_threshold_crossing(plan_leg, history)
    span = float(np.sum(leg.sample_durations_s))
    if not (plan_cut.truncated_at_threshold and plan_cut.final_time_s <= span + ROW_TOLERANCE_S):
        plan_cut = cut_at_lead(plan_leg, span)
    return float(np.linalg.norm(_end_point(leg) - _end_point(plan_cut)))


def _fly_leg(run: FlightRun, history: FlightSeries, forecast: Forecast, *, segment_s: float, round_index: int,
             code: int) -> Forecast | None:
    """Fly this round's segment whole; end at the crossing on the final, at the prior's landing
    or at the budget. Returns the leg appended, or None when the budget left no row to fly."""
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
        return None
    run.legs.append(leg)
    run.codes.append(code)
    run.predictions += 1
    run.flown_s += float(np.sum(leg.sample_durations_s))
    run.rounds.append({
        "round": round_index, "lead_s": run.flown_s, "code": code,
        "e_track_m": displacement_at(run.series, leg, run.anchor, float(leg.times[-1])),
    })
    return leg


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
    """The flight's legs as one forecast from the anchor. Its predicted end is the flown span:
    a flight the prior landed was already flown for its predicted fraction of the last
    segment (`_fly_leg`), so the span IS the arrival — adding the fraction again double
    counted it (review 2026-09-18 B2)."""
    span = float(sum(np.sum(leg.sample_durations_s) for leg in run.legs))
    whole = concatenate(run.legs, run.anchor, span)
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


def _code_columns(run: FlightRun) -> dict[str, Any]:
    """The truth's and the flown legs' code columns — a coded protocol's; a protocol-none row
    has none (no codebook labelled anything)."""
    if run.truth is None:
        return {}
    return {
        "truth_codes": run.truth.codes.tolist(), "flown_codes": list(run.flown_codes),
        "truth_length": run.truth.length, "landed_fraction_truth": run.truth.landed_fraction,
    }


def flight_row(run: FlightRun, points: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One flight's row (and the record metrics): the reference verdicts, ADE / FDE, the
    geometry, the displacement at every lead, every round's step error and e_plan, how it ended."""
    forecast = whole_forecast(run)
    metrics = observed_series_metrics(run.series, forecast, points=points)
    geometry = _forecast_geometry(run.series, forecast)
    origin = float(run.series.times[run.anchor])
    difficulty = approach_difficulty(run.series, run.anchor)
    row = {
        "flight_id": run.series.flight_id,
        "reference": reference_verdicts(run.series, forecast),
        "ended": run.ended, "truncated_at_threshold": run.truncated, "predictions": run.predictions, "held_predictions": run.held_predictions,
        "codes": list(run.codes), **_code_columns(run),
        "flown_states": [state.tolist() for state in run.flown_states],
        "landed_fraction": run.landed_fraction, "landed_probabilities": list(run.landed_probabilities),
        "ade_m": float(metrics["ade_m"]), "fde_m": float(metrics["fde_m"]),
        "final_time_error_s": float(metrics["final_time_error_s"]),
        "chamfer_m": float(geometry["chamfer_m"]), "frechet_m": float(geometry["frechet_m"]),
        # plan §3.1's absence rule: the forecast's last row is HELD past its end (a flight that
        # says it has arrived is at the threshold, and a reading past that claim measures the
        # claim); only the truth's end makes a reading absent — the rule B61's 768 / 1396 were
        # read under (review 2026-09-18 B3)
        "at": {f"{lead:g}": displacement_at(run.series, forecast, run.anchor, origin + lead, hold_forecast_end=True)
               for lead in LEADS_S},
        "rounds": run.rounds,
        "route_tortuosity": difficulty.route_tortuosity, "established_at_anchor": difficulty.established_at_anchor,
        "remaining_path_m": difficulty.remaining_path_m,
    }
    return row, metrics


__all__ = [
    "ENDED_CROSSED", "ENDED_HORIZON", "ENDED_LANDED", "ENDED_TRUTH_EXHAUSTED", "HORIZON_SLACK_FRACTION",
    "HORIZON_SLACK_S", "LANDED_THRESHOLD", "LEADS_S", "PROTOCOLS", "PROTOCOL_A", "PROTOCOL_A_TRUTH",
    "PROTOCOL_C", "PROTOCOL_NONE", "CODE_NONE", "Executor", "FlightRun", "Prior", "closing_horizon_s", "flight_row", "fly",
    "from_remaining_path", "reference_verdicts", "required_positions", "whole_forecast",
]
