# flight_scenarios — who gets into a dataset (reference)

Full text behind the index lines of `flight_scenarios/CLAUDE.md`, moved here VERBATIM on 2026-09-16 so that
file stays a short index (it is injected into every session that touches the tree). Only the
`### <ID>` headings and the dated **Correction / Note** paragraphs are new; every heading has
exactly one index line in that CLAUDE.md ending in its ID (`grep -n '^### # ·' flight_scenarios/docs/population_reference.md`).

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.**

## Population: who gets into a dataset

### FS1 · `build_scenarios_from_arrivals` is strict; `dataset.build_scenario_dataset` is the batch layer

- **`build_scenarios_from_arrivals` is strict; `dataset.build_scenario_dataset` is the batch
  layer.** The strict builder raises on any flight it cannot build, which is right for a
  library and wrong for a batch: **35 flights of the 42,725 rostered arrivals (0.08 %) have
  no usable fitted final approach** (KMSY 1, KRDU 1, KSJC 25, KSMF 8, KSTL 0) and aborted the
  fitted-ADS-B dataset for 4 of the 5 airports. **That count is against the 2026-08-19 roster
  generation; the roster is now 42,650 and the 35 has NOT been re-measured** — the numerator
  needs an uncapped rebuild, so quote it as a ~0.08 % rate, not as a current tally. What IS
  current: over the capped selections on disk, 20 flights are dropped (KMSY 1, KRDU 0, KSJC 13,
  KSMF 6, KSTL 0), same shape. They now raise `UnusableFittedApproach` (its own type, so a
  broad `except ValueError` cannot swallow a real contract violation next to it) and the
  batch layer drops and NAMES them.

### FS2 · `--max-per-runway` caps the population, and the cap is written down

- **`--max-per-runway` caps the population, and the cap is written down.** Default 2000 via
  `prepare_scenario_inputs.py`; at that value the fleet is 23,429 flights / 70,287 solves
  instead of 42,650 / 127,950 (both re-read off the `.selection.json` files on disk
  2026-09-06; this used to say 23,453 / 70,359 against 42,725 / 128,175). Selection is evenly
  spaced over landing time within each runway, so a capped runway still spans the whole
  harvest window. It is derived from the arrival ROSTER alone — no source track is opened
  for a discarded flight, which is also why
  a capped KRDU build costs 0.9 GB / 7 s instead of 2.4 GB / 24 s — and therefore does not
  depend on the target type: **both prepared datasets select the same flights**, which is
  what keeps the per-flight comparison between `fitted_adsb` and `runway` paired. Every
  decision lands in `<scenarios>.selection.json` (`flight-scenarios-selection-v1`): a bounded
  population that is not stated reads as a full one, and every rate computed from it is then
  quietly wrong.

### FS3 · `source["dynamics_typecode"]` is what the v9 speed gate keys on

- **`source["dynamics_typecode"]` is what `evaluation`'s v9 threshold speed gate keys
  on** — the ICAO type the model FLEW (the fallback type when the identity's type has
  no dynamics; `resolved_typecode` keeps the identity), looked up in the published
  approach-speed table `aircraft/reference_speeds.json`; a computed record without it
  grades speed-indeterminate (loud, by design). Both record producers (optimizer batch
  and ts_transformer export) copy `scenario.source`, so this one write covers both.
  `source["landing_aero"]` (`{wing_area_m2, cl_max_landing}`, the same `AeroParams` the
  optimizer/replay fly) is still written as provenance of the model's own velocity
  floor but is no longer read by evaluation (v6–v8 anchored on it).
  → `evaluation/docs/THRESHOLD_SPEED_GATE.md`

### FS4 · `source["flight_key"]` is populated here

- **`source["flight_key"]` is populated here.** Optimizer evaluation rows used to report
  `flight_key: null` while observed rows carried it, because `build_scenario` never copied
  it. It does now — but only when the flight has an `id`, since `flight_key`'s fallback is
  the caller's list index and this function does not have one; a key built on the wrong index
  would disagree with the record filename `_scenario_filename` derives.
