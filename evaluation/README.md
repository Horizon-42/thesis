# evaluation — judge optimized trajectories against their targets

File-based seam at the **end** of the optimization pipeline: inputs are
per-trajectory record files, the output is one JSON report. The package depends
only on `geokit` + stdlib — it never imports the optimizer, so anything that can
write the record format (the collocation batch, a future data-driven model, …)
can be evaluated identically.

```
scenario_optimization (batch)                    evaluation (this package)
  ├─ references/<flight>_reference_eval.json ◄─┐
  ├─ <flight>_states.json   (viz/comparison)   │ reference_file
  ├─ <flight>_eval.json ───────────────────────┴►  python -m evaluation --input <dir>
  └─ summary.json                                    └─► evaluation_report.json
```

One-shot runner (scenarios → references → optimization → selected tails, per airport):
`python run_scenario_pipeline.py --airport KRDU --target-type runway [--with-constraint] [--outputs eval]`
(`--outputs czml,eval` default: also rebuilds the frontend comparison CZML from the same solve)

## Input contract (one JSON per trajectory)

```jsonc
{
  "source":        { "id": "AFR074", "runway": "05L", ... },   // provenance, free-form
  "initial_state": { "lat", "lon", "alt", "V", "psi", "gamma", "m" },
  "target_state":  { same keys },              // null allowed on unsolved records
  "final_time_s":  380.5,                      // null when unsolved
  "states":   [ { "t", "lat", "lon", "alt", "V", "psi", "gamma", "m" }, ... ],
  "controls": [ { "thrust", "bank_rad", "load_factor" }, ... ],  // SAME length as states
  "reason":   "ValueError: infeasible"         // unsolved records only
}
```

- **Alignment.** `controls[i]` is the control **active at** `states[i].t`
  (zero-order hold): the optimizer's sparse piecewise-constant segments sampled
  onto the state grid. The terminal entry repeats the last segment's control.
- **Reference records** — an observed ADS-B track expressed in this SAME
  contract: per-sample kinematics derived from the track
  (`flight_scenarios.state_samples_from_track`, times rebased to 0, the same
  target the optimizer flew to) with `controls == []` (observed data has no
  control inputs).
- **Unsolved configurations** keep their boundary conditions but have
  `states == controls == []` — that is how the batch **solve rate** is computed
  from the file set alone.
- **`reference_file`** (optional) points a record at its reference, resolved
  relative to the record's own directory (the batch writes
  `references/<identity>_reference_eval.json` and sets the pointer on every
  record, failed ones included).
- Units: metres / seconds / kg; lat/lon degrees; ψ/γ radians (model convention:
  ψ = 0 East, CCW); alt metres MSL.

The optimization side produces these via
`4dTrajectory/optimization/evaluation_export.py`. There is nothing to
re-derive: the true-dynamics rollout (`aerodynamic_model.rollout_piecewise_constant`)
already carries the active control on every sample; the export just maps it.
Both `scenario_optimization` batch modes write a `*_eval.json` next to every
`*_states.json` — **including failed scenarios** (empty record).

## What is judged

Per trajectory (`evaluate_record`): the **final state vs the target state** —
"did it arrive?". Lateral = great-circle distance (`geokit.haversine_m`),
vertical = signed altitude difference. Speed / heading deltas are reported for
context but not gated. A trajectory is **successful** iff it is solved AND
passes both positional gates.

## Regulation-derived gates (`thresholds.py`, overridable)

| Gate | Default | Source (in `docs/regulation/Order_8260.58D.pdf`) |
|---|---|---|
| lateral | ≤ 106.75 m | §3-1-5.c(3) "Course width at threshold" (pp. 3-7/3-8), Formula 3-1-1 — the course width at the LTP is `greater of 350 ft or tan(1.5°)·d_GARP`, a **one-sided** value (Figure 3-1-7 draws it ±350 ft about the centerline — the order says "course width" where this package says semiwidth). 350 ft = 106.68 m → **106.75 m** via the order's own "round to the nearest 0.25-meter increment" rule (so the number never appears verbatim); it is the tightest full-scale deflection any LPV final can have at the LTP |
| vertical (low) | ≥ −3.05 m | §1-3-1.f(2)(b) "Threshold crossing height" (p. 1-27), item 1: default TCH ⇒ 30 ft wheel crossing height, minimum WCH 20 ft ⇒ 10 ft below the target (derived offset) |
| vertical (high) | ≤ +6.10 m | same — maximum WCH 50 ft ⇒ 20 ft above (derived offset) |

The vertical window assumes the target altitude sits at the published TCH point
over the threshold — true for this project's CIFP-anchored targets. For targets
that are not thresholds (e.g. the unconstrained batch's end-of-track states) the
gates still measure "reached the target to approach-guidance accuracy"; override
them if a looser criterion is wanted (e.g. `--lateral-max-m 556` for RNP 0.3
containment).

## Observed-track comparison (`reference.py`)

When a record carries `reference_file`, the report adds per-trajectory and batch
comparisons against the flight as actually flown:

- **flight-time delta** — `optimized − observed` (negative = the optimizer is
  faster over the same start→target journey; the scenario's initial state is
  derived from the track start, so durations compare directly). Unsolved records
  still report the observed baseline duration.
- **path deviation** — both paths resampled at 101 fractions of their own
  horizontal arc length, then compared point-by-point (horizontal great-circle
  distance + signed altitude difference; mean/p95/max). A path-SHAPE comparison:
  the two fly different speed profiles by design, so time-matching would
  conflate timing with geometry.

## Batch metrics (`evaluate_batch` / the report JSON)

- `solve_rate` — non-empty solutions / all records
- `success_rate` — passed both gates / all records (+ `success_rate_among_solved`)
- `lateral_m` — mean / **p95** / max over solved records. p95 is reported because
  RNP containment is itself a 95 % statistic (8260.58D: position within the
  leg's RNP radius 95 % of the time), so it compares directly with RNP limits.
- `vertical_m` — signed mean + abs mean / p95 / max
- `final_time_s` — mean / min / max flight time over solved records
- `trajectories` — one row per record (deviations, violations, failure reason)

## HTML report (`visualize.py`)

```bash
python -m evaluation.visualize --input outputs/run1 --output outputs/run1/evaluation_report.html
```

Recomputes the evaluation from the record files (same flags as `python -m evaluation`)
and renders a single HTML page: summary cards + aggregate tables, per-flight final
lateral/vertical deviation charts against the gates, optimized-vs-observed flight
times + Δtime distribution + path-shape deviation (when references exist), a
per-flight track overlay (plan view + altitude profile behind a flight selector,
`--max-tracks` evenly-sampled overlays embedded — the cap is stated on the page),
and the full verdict table. Plotly loads from its CDN (same convention as the
project's other interactive docs); all data is embedded.

## Usage

```bash
# batch-optimize scenarios (writes *_states.json + *_eval.json + summary.json;
# --reference-tracks additionally writes references/ and points every record at its reference)
python 4dTrajectory/optimization/scenario_optimization.py \
    --scenarios scenarios.json --reference-tracks landings.json \
    --output-dir outputs/run1 --constrained-iaf

# evaluate the run
python -m evaluation --input outputs/run1 --output outputs/run1/evaluation_report.json

# HTML report (tables + charts)
python -m evaluation.visualize --input outputs/run1 --output outputs/run1/evaluation_report.html

# custom gates
python -m evaluation --input outputs/run1 --lateral-max-m 556 --vertical-above-max-m 30
```

Tests: `python -m pytest evaluation/tests -v` (also wired into `run_all_tests.sh`).
