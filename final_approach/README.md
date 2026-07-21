# final_approach

The single geometry of a flown final-approach segment. Pure functions over `geokit` +
stdlib; no I/O, no config, no regulation constants.

```
trajectory_data_process  ──┐
                           ├──>  final_approach  ──>  geokit + stdlib
evaluation/arrival.py    ──┘
```

## Why it exists

Two planes need the same geometry and must not each grow their own copy: the harvest
asks **which runway** a track landed on, evaluation asks **how good** the approach was.
Both answers come from one least-squares fit of the flown segment, extrapolated to the
landing threshold.

## The load-bearing separation

This package returns **facts only** — no `established` flag, no pass/fail, no gate
constant. That is not tidiness:

> If the harvest filtered tracks on the quality criterion evaluation later reports,
> every surviving track would pass by construction. The "established rate" would be
> 1.0 — manufactured by the selection rather than measured.

So `assign_runway` makes a **relative** comparison (arg-min over runways) and may never
reject a track for flying badly. `evaluation/arrival.py` applies the **absolute** FAA
8260.58D gates. Same fit, different question.

## Why extrapolate at all

Crowd-sourced ADS-B stops before touchdown. Measured on 996 KRDU arrivals: **970 end
short of the threshold, a median 325 m out and still ~135 ft up**, at a clean 1 Hz
cadence right to the cut — the receivers lose the aircraft, the aircraft keeps flying.
The last sample records where reception ended, not where the aircraft crossed.

Per-airport, the last sample sits at (median along-track): KSMF +3 m, KMSY −12 m,
KSTL −122 m, KRDU −325 m, KSJC −759 m. It is a property of receiver siting, not of
the airport or the approach.

## Why a straight line, and not a Gaussian process

A stabilised final approach *is* straight in both planes, so OLS on that form is the
correct estimator, not an approximation. A kernel GP is an **interpolator** — extrapolated
past its data it reverts to the prior mean, dragging the crossing toward zero. You reach
for a GP when the functional form is unknown; here it is known, and the fitted glidepath
coming out **3.02–3.11° across five airports** is the evidence.

## Why the fit is mandatory, not cosmetic

OpenSky `geoaltitude` is quantised to **25 ft = 7.62 m** (all 482 distinct altitudes in
the KRDU set lie on that lattice). The vertical gate window is **9.15 m** wide, so one
sample carries ±3.81 m of rounding — it cannot resolve the window even in principle.
Averaging a few hundred quantised samples pins the crossing to under 2 m. `states[-1]`
never could.

## Why sigma is autocorrelation-corrected

The aircraft crosses one 7.62 m quantisation step every ~2 samples (measured 3.81 m of
descent per 1 Hz sample), so **54.8 % of consecutive samples report an identical raw
altitude** and residuals are strongly correlated: lag-1 ρ ≈ 0.43, i.e. n_eff ≈ 0.40 n.

Both variance terms are deflated — `Sxx` sums over the same correlated samples as `1/n`.
Correcting only the first gives a 1.15× inflation where the honest figure is **1.58×**
(σ 1.06 → 1.67 m at KRDU). On a 9.15 m gate that changes verdicts, so the naive number
is not offered as an option. ρ is clamped at 0, so the correction can only widen.

## Why the window is [−5000, −300] m

On a 3° path that spans ~900 ft down to ~107 ft above threshold elevation: below the
1000 ft stabilisation gate every airline SOP requires, above flare initiation (~50 ft).

Measured sensitivity (KRDU, published TCH):

| window (m) | n | median crossing | median σ |
|---|---|---|---|
| [−8000, −300] (≈ from the FAF) | 330 | +5.43 m | 1.63 m |
| [−5000, −300] | 397 | **+3.66 m** | **1.68 m** |
| [−4000, −300] | 406 | +3.33 m | 1.75 m |
| [−2000, −300] | 410 | +4.04 m | 2.50 m |

Starting at the FAF biases the crossing **high** — aircraft are still intercepting the
glidepath from above out there. Shrinking below ~3 km leaves too short a baseline to pin
the slope and σ climbs. The window is a real methodological choice: report the
sensitivity, do not pick one silently.

## What discriminates a runway, and what cannot

| signal | separates parallels? | separates the two ends? |
|---|---|---|
| distance to threshold | **no** (763 m correct vs 791 m wrong) | no |
| runway heading | **no** (parallels share it) | yes |
| median abs cross-track | **yes** (13.9 m vs 230.5 m at KSJC) | **no** (shared centreline) |
| along-track direction of travel | no | **yes** |

Distance is worst where the threshold is displaced: at KSJC 30L every track ends ~775 m
from the landing point, so the measure says nothing about which runway was flown.

## One track, one runway — by construction

The predecessor classified per threshold and then tried to undo double-assignment with a
pairwise guard. The guard's logic was right, but a guard can be bypassed: the shipped
artifacts still had **232 of KSJC's 319 unique landings (72.7 %) in two runways' files**
(169 in 30L+30R, 63 in 12L+12R), plus 32 at KSTL.

Here the track is fitted against every candidate once and the arg-min is taken. No code
path emits two runways, so the bug is unrepresentable rather than prevented.

Validated against the real harvest — the arg-min reproduces the independently derived
ground truth and leaves the clean airports alone:

| | new | independent ground truth |
|---|---|---|
| KSJC 30L+30R → 30L | 162 | 160 |
| KSJC 12L+12R → 12R | 61 | 61 |
| KSTL 30L+30R → 30R | 32 | 32 |
| KSMF (no parallel bug) | 199/201/198/115 | 200/200/200/114 |

## Only the final inbound run is fitted

Selecting by along-track range alone is not enough. A real arrival can occupy the same
band twice — downwind leg, vectoring, a go-around, or a track exported against the wrong
runway end so it contains the approach *and* the landing roll. One shipped KSJC track
ranged over −23.5 km to +18.7 km yet ended at +2.6 km; a range filter mixed downwind
samples into the fit and produced a **median cross-track of 8.7 km**, which then decided
a runway assignment.

`_final_inbound_run` walks backward from the last sample at or inside the window's inner
edge and stops on a reversal. This also subsumes direction: an outbound track yields no
fit at all.

## The four outcomes

| outcome | meaning |
|---|---|
| `assigned` | one runway wins outright |
| `ambiguous` | two runways within `AMBIGUITY_MARGIN_M` (50 m) — returned, never guessed |
| `unassignable` | it landed, but no runway yielded a fittable segment — a **coverage** statement, not a judgement of the approach |
| `not_landing` | never descended to near a threshold |

`unassignable` and `not_established` must stay distinct: the first is the receiver's
fault, the second the approach's. Conflating them charges a reception gap to the pilot.
Every outcome carries `scores` so a disputed assignment is auditable without a re-run.

## Datum contract

`TrackPoint.alt_m` and `RunwayFrame.elevation_m` **must share a vertical datum**. The
harvest works in ellipsoidal height (HAE, as the sensor reported); the modeling plane
works in MSL. This package never converts — it subtracts, so mixing the two shifts
results by the geoid undulation (~33 m over the US) with no warning. Assignment reads
only cross-track and is datum-free; the landing screen's height test is not.

## Heading convention

`course_deg` in is a **compass** bearing (0 = North, clockwise) — what
`runway_thresholds.json` publishes and what a CIFP plate prints. The project's dynamics
model uses math-ENU (0 = East, CCW). `RunwayFrame` converts once at construction and
never re-exposes the raw angle, so a caller cannot pick the wrong one.

## Tests

```bash
python -m pytest final_approach/tests -q --import-mode=importlib
```

Synthetic fixtures only — exact, fast, no data dependency. The real-harvest validation
above is a one-off check, not a test, because the artifacts it reads are due to be
re-harvested.
