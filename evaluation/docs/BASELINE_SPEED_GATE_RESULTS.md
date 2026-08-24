# Baseline speed-gate results — first three-gate measurement of the observed fleet

Measured 2026-08-24, on the five airports' observed evaluation reports republished at
commit `cbb09e0` (the baseline speed gate) over the 2026-08-24 fleet reclassify (every
stored event carries `crossing_ground_speed_m_s`). Design and sources:
`THRESHOLD_SPEED_GATE.md`; §5 there records the owner decision this measurement
implements — the baseline runs the SAME three gates as its modeled twins, with the
crossing GROUND speed judged as a stated proxy
(`vref_1p23_vs1g_to_vref_plus_20kt_ground_speed_proxy` on every observed row).

**Headline caution, in one line:** the fail structure clusters by AIRFRAME FAMILY,
not by weather — read §5 before quoting any airport-level speed-fail rate as flight
behaviour.

## 1. Headline: two-gate → three-gate

| | arrivals | 2-gate pass (pre) | 3-gate pass | of decided | speed pass / graded | speed fail | ungraded |
|---|---|---|---|---|---|---|---|
| KMSY | 4,150 | 93.8 % | **57.8 %** | 68.1 % | 2,565 / 3,501 (73.3 %) | 936 | 649 |
| KRDU | 14,439 | 98.1 % | **46.0 %** | 64.0 % | 6,700 / 10,194 (65.7 %) | 3,494 | 4,245 |
| KSJC | 11,157 | 99.9 % | **45.8 %** | 68.2 % | 5,106 / 7,481 (68.3 %) | 2,375 | 3,676 |
| KSMF | 4,231 | 99.8 % | **64.4 %** | 71.1 % | 2,725 / 3,825 (71.2 %) | 1,100 | 406 |
| KSTL | 8,769 | 96.8 % | **59.3 %** | 72.0 % | 5,377 / 7,190 (74.8 %) | 1,813 | 1,579 |
| fleet | 42,746 | — | — | — | 22,473 / 32,191 (69.8 %) | 9,718 | 10,555 |

Two DIFFERENT effects pull the composite down, and they must never be quoted as one:

- **Speed fails** compose the verdict to `fail` (a graded flight crossed outside its
  stall-anchored window).
- **Ungraded flights** compose to `indeterminate`, not fail — they deflate the
  "of all" rate mechanically. Quote the "of decided" column alongside it.

### Where "ungraded" comes from

Fleet-wide: 10,555 ungraded = **10,306 unresolvable airframes + 0 speedless events**
(every graded-subject event carried a fitted crossing speed — the harvest's speed fit
achieved full coverage) + the balance from rows with no estimable crossing at all.
Unresolvable means the icao24 maps to no OpenAP-supported type
(`flight_scenarios.resolve_landing_aero`, no fallback by design): overwhelmingly GA
and unregistered traffic, which is why KRDU (29 %) and KSJC (33 %) are hit hardest
while the airline-dominated KMSY/KSMF/KSTL barely are.

## 2. Slow or fast?

Fleet-wide: 9,718 fails = **6,461 too slow (66 %) + 3,257 too fast (34 %)** — but the
split inverts by airport:

| | fails | too slow | too fast | slow margin kt (med/p90/p95/max) | fast margin kt (med/p90/p95/max) |
|---|---|---|---|---|---|
| KMSY | 936 | 360 (38 %) | **576 (62 %)** | 3.1 / 9.8 / 13.7 / 24.5 | 2.8 / 7.5 / 9.4 / 22.6 |
| KRDU | 3,494 | **2,411 (69 %)** | 1,083 (31 %) | 5.2 / 13.7 / 16.4 / 26.9 | 3.2 / 10.4 / 13.0 / 26.1 |
| KSJC | 2,375 | **2,087 (88 %)** | 288 (12 %) | 4.0 / 13.4 / 17.3 / 41.8 | 2.4 / 8.6 / 10.1 / 19.7 |
| KSMF | 1,100 | **839 (76 %)** | 261 (24 %) | 4.3 / 12.3 / 14.6 / 26.9 | 2.7 / 6.7 / 7.7 / 24.8 |
| KSTL | 1,813 | 764 (42 %) | **1,049 (58 %)** | 3.7 / 12.2 / 15.2 / 27.0 | 3.0 / 7.9 / 9.6 / 64.4 |

Margins are small: the median fail sits 3–5 kt beyond a bound of a 20 kt window.
The wind-explainable share (margin within a plausible surface-wind magnitude):

- slow fails within 5 kt of the bound: 49–70 % per airport; within 10 kt: 79–90 %.
- fast fails within 5 kt: 65–79 %; within 10 kt: 89–97 %.

A headwind lowers ground speed, so wind can only manufacture SLOW fails (aircraft
land into wind by procedure; tailwind components are operationally capped ≈10 kt).
Two structural corollaries: the ground-speed proxy has a systematic slow bias
everywhere, and fast fails are the stronger evidence of genuinely fast crossings —
or of a window anchored too low (§5).

## 3. Per-runway split (graded n ≥ 30)

| airport | runway | graded | slow | fast | pass |
|---|---|---|---|---|---|
| KMSY | 02 | 246 | 9.3 % | 7.3 % | 83.3 % |
| KMSY | 11 | 2,187 | 10.7 % | 17.6 % | 71.7 % |
| KMSY | 29 | 1,066 | 9.6 % | 16.2 % | 74.2 % |
| KRDU | 05L | 2,624 | 27.1 % | 6.9 % | 66.0 % |
| KRDU | 05R | 729 | 16.7 % | 13.2 % | 70.1 % |
| KRDU | 23L | 1,355 | 13.6 % | 16.5 % | 70.0 % |
| KRDU | 23R | 5,486 | 25.4 % | 10.6 % | 64.0 % |
| KSJC | 12R | 341 | 37.8 % | 3.2 % | 58.9 % |
| KSJC | 30L | 6,488 | 28.5 % | 3.7 % | 67.8 % |
| KSJC | 30R | 647 | 16.1 % | 5.7 % | 78.2 % |
| KSMF | 17L | 1,024 | 22.9 % | 6.2 % | 70.9 % |
| KSMF | 17R | 2,258 | 22.0 % | 7.4 % | 70.6 % |
| KSMF | 35L | 543 | 20.1 % | 5.3 % | 74.6 % |
| KSTL | 11 | 484 | 3.5 % | 18.6 % | 77.9 % |
| KSTL | 12L | 3,306 | 11.8 % | 13.2 % | 75.0 % |
| KSTL | 12R | 221 | 11.3 % | 16.7 % | 71.9 % |
| KSTL | 29 | 86 | 12.8 % | 10.5 % | 76.7 % |
| KSTL | 30L | 97 | 14.4 % | 4.1 % | 81.4 % |
| KSTL | 30R | 2,979 | 10.1 % | 15.8 % | 74.0 % |

A weak wind signature exists (reciprocal ends differ — e.g. KSTL 11 fast-heavy vs 30L
slow-heavy), but the dominant runway-level differences track which CARRIERS use which
runway — i.e. fleet mix, which §5 shows is the primary effect. The same lesson as the
KSJC route-mix/ADE analysis: a per-runway or per-airport speed rate without its type
mix is not a comparison.

## 4. The wind caveat, quantified

The judged value is ground speed; the window is airspeed-anchored. A 10 kt headwind —
entirely ordinary on final — moves a flight 10 kt toward the slow bound, half the
window. From §2: **≈ 79–90 % of slow fails lie within 10 kt of the bound** and are
therefore individually indistinguishable from wind without crossing-time weather
data. The per-flight attribution needs METAR surface wind at the landing time
(unimplemented; see §6). Until then, quote observed speed rates only with this
caveat attached — the report carries it in
`methodology.terminal_speed.observed_proxy_caveat`, and the proxy criterion id on
every row is what keeps observed and computed speed rates from being averaged
together by accident.

## 5. The dominant effect is the per-type window anchor, not weather

Fleet-wide speed results by resolved airframe (graded ≥ 200 flights):

| type | graded | pass | slow | fast |
|---|---|---|---|---|
| B38M | 5,864 | 75.6 % | 6.5 % | 17.8 % |
| B737 | 5,537 | 79.1 % | 16.7 % | 4.3 % |
| B738 | 4,962 | 75.7 % | 5.9 % | 18.4 % |
| E75L | 3,401 | 74.3 % | 22.1 % | 3.5 % |
| B739 | 2,249 | 82.0 % | 5.9 % | 12.0 % |
| A319 | 1,895 | 46.6 % | 53.2 % | 0.2 % |
| CRJ9 | 1,650 | 84.1 % | 5.1 % | 10.8 % |
| A321 | 1,202 | 41.5 % | 58.2 % | 0.2 % |
| A320 | 1,000 | 54.8 % | 44.7 % | 0.5 % |
| A21N | 881 | **8.1 %** | **91.9 %** | 0.0 % |
| B39M | 713 | 83.6 % | 4.9 % | 11.5 % |
| A20N | 535 | 33.3 % | 66.4 % | 0.4 % |
| C56X | 314 | 45.2 % | 1.3 % | 53.5 % |
| B763 | 239 | 64.9 % | 33.9 % | 1.3 % |

**Wind cannot tell Airbus from Boeing.** Every A320-family type slow-fails at
45–92 % (A21N: 91.9 % slow, 0 fast) while every 737-family type passes 75–84 % with
a mild FAST skew — on the same days, at the same airports, through the same weather.
A manufacturer-clustered split of this size is a systematic bias in the per-type
window anchor: the model's `Vs1g = sqrt(2mg / (ρ₀ S Cl_max_landing))` uses OpenAP's
landing mass and the `aircraft` package's landing `Cl_max`, and for the A320 family
that combination evidently places 1.23·Vs1g ABOVE the speeds the real fleet crosses
at (an A21N window whose floor excludes 92 % of real crossings is measuring the
window, not the fleet). The mirror image — B738/B38M ≈18 % fast, C56X 53 % fast —
suggests those anchors sit slightly LOW.

This also explains the airport-level inversion in §2 without invoking climate:
KSJC/KRDU/KSMF arrivals are Airbus/E75L-heavy (slow-dominated), KMSY/KSTL are
737-family-heavy (fast-skewed). **Rule: do not read airport-level or per-runway
speed-fail rates as flight behaviour until the A320-family (at minimum A21N, A20N,
A321, A319, A320) stall facts are reviewed; quote per-type rates instead.** The same
window is the optimizer's velocity-floor anchor (`aircraft.aero_params`, one
formula), so a corrected Cl_max/landing-mass would move BOTH the baseline gate and
the solver's floor — by design, together.

## 6. Follow-ups this measurement opens

1. **Review A320-family stall facts** in the `aircraft` package / OpenAP mapping
   (Cl_max_landing, landing mass) against published Vref tables; re-derive this
   document after any change (the republish is `--evaluate-only` × 5, ~10 min).
2. **METAR crossing-time wind** would convert the proxy into a defensible
   airspeed-equivalent measurement and attribute the wind-explainable band in §2
   per flight (data acquisition, not an evaluation change).
3. When the optimizer batch reruns (first v6 computed reports), compare per-type
   speed rates baseline-vs-optimized on the SAME window — the shared anchor makes
   the comparison exact; expect floor-riding solves near 1.10·Vs to fail low.

## 7. Reproduction

```bash
# the reports these numbers come from
python -m trajectory_data_process.harvest --airport <ICAO> --evaluate-only
# per-row inputs: trajectories[].speed_result, .crossing_ground_speed_ms,
#                 .bounds.speed_lower_ms/.speed_upper_ms, .runway, .reason;
#                 per-type via the record's source.aircraft_type (row.file →
#                 <harvest>/<ICAO>/approach/records/)
```

Margins are `(bound − v)` resp. `(v − bound)` in knots; "of decided" =
pass / (total − indeterminate); type table pools all five airports.

## 8. After the anchor calibration (2026-08-24, commits `6e31f2d` + republish)

The fixes §5 called for landed the same day: A320-family landing Cl_max 2.7 → 3.0
(calibrated from Airbus's definitional VLS = 1.23·Vs1g + published VLS figures, pinned
by `aircraft/tests/test_aero_anchors.py`), and the C56X airframe restored from the
C550 surrogate's masses to certificated values. Republished results:

| | 3-gate pass (2.7 era) | 3-gate pass (calibrated) | of decided | speed pass / graded | slow | fast |
|---|---|---|---|---|---|---|
| KMSY | 57.8 % | **61.1 %** | 72.1 % | 2,718 / 3,501 (77.6 %) | 195 | 588 |
| KRDU | 46.0 % | **53.8 %** | 74.9 % | 7,825 / 10,194 (76.8 %) | 1,299 | 1,070 |
| KSJC | 45.8 % | **47.8 %** | 71.2 % | 5,331 / 7,481 (71.3 %) | 1,863 | 287 |
| KSMF | 64.4 % | **67.3 %** | 74.3 % | 2,848 / 3,825 (74.5 %) | 714 | 263 |
| KSTL | 59.3 % | **62.4 %** | 75.8 % | 5,651 / 7,190 (78.6 %) | 494 | 1,045 |

Per type (graded ≥ 200), after calibration:

| type | graded | pass | slow | fast | | type | graded | pass | slow | fast |
|---|---|---|---|---|---|---|---|---|---|---|
| B38M | 5,864 | 75.6 % | 6.5 % | 17.8 % | | A321 | 1,202 | **80.3 %** | 17.1 % | 2.7 % |
| B737 | 5,537 | 79.1 % | 16.7 % | 4.3 % | | A320 | 1,000 | **85.4 %** | 11.1 % | 3.5 % |
| B738 | 4,962 | 75.7 % | 5.9 % | 18.4 % | | A21N | 881 | **33.6 %** | 66.4 % | 0.0 % |
| E75L | 3,401 | 74.3 % | 22.1 % | 3.5 % | | B39M | 713 | 83.6 % | 4.9 % | 11.5 % |
| B739 | 2,249 | 82.0 % | 5.9 % | 12.0 % | | A20N | 535 | **71.0 %** | 27.3 % | 1.7 % |
| A319 | 1,895 | **78.9 %** | 19.2 % | 2.0 % | | C56X | 314 | **73.6 %** | 5.7 % | 20.7 % |
| CRJ9 | 1,650 | 84.1 % | 5.1 % | 10.8 % | | B763 | 239 | 64.9 % | 33.9 % | 1.3 % |

Reading:

- **The manufacturer cluster is gone**: A319/A320/A321 now pass 78.9–85.4 %, in line
  with the Boeing family — confirming §5's diagnosis that the anchor, not the fleet,
  produced the old 45–58 % slow-fail rates. C56X's fast-fail collapsed 53.5 → 20.7 %.
- **The "of decided" composite is now nearly uniform across airports (71.2–75.8 %)**
  where the 2.7-era spread was wider — airport-level variation was largely anchor ×
  fleet-mix, as predicted.
- **Remaining residuals, deliberate**: A21N (33.6 % pass, all-slow — the fleet lands
  far below the MLW the window is anchored at; needs an operational landing-mass
  model, not another Cl tweak), A20N (27 % slow, same direction, milder), B763
  (33.9 % slow, n=239, heavy-bucket Cl 2.4 uncertain), E75L (22 % slow). These stay
  documented rather than patched — each needs its own evidence.
