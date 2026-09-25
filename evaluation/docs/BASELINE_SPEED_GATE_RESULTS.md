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
(`flight_scenarios.resolve_airframe`, then `resolve_landing_aero`; no fallback by design): overwhelmingly GA
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

## 9. Measured load factor and the METAR headwind (2026-09-07, report v8)

Measured on the same five observed batches, evaluated to a scratch directory with the
v8 code (`--metar-root` = the five airports' IEM ASOS archives, 2026-04-30 … 07-23; the
reports on disk were not overwritten). Two things changed for observed rows: the
crossing load factor is inverted from the flight's own final 20 s of kinematics
(`THRESHOLD_SPEED_GATE.md` §3.5), and the ground speed is corrected by the field's
headwind component into an airspeed estimate (§3.6). The load factor alone moved
211 verdicts fleet-wide (all within a knot of the floor); the wind is what matters.

### 9.1 Coverage and the wind itself

| airport | speed-graded rows | METAR estimate / proxy | headwind kt p5 / p50 / p95 |
|---|---|---|---|
| KRDU | 10,194 | 13,289 / 1,148 | −2.3 / +4.9 / +12.0 |
| KSJC | 7,481 | 10,341 / 810 | −1.5 / +8.0 / +14.0 |
| KSTL | 7,190 | 8,550 / 215 | −2.9 / +3.7 / +9.9 |
| KSMF | 3,825 | 3,922 / 307 | −1.6 / +6.1 / +14.2 |
| KMSY | 3,501 | 3,928 / 221 | −2.6 / +2.8 / +10.0 |

A report within 30 min exists for 91–97 % of the rows; the rest (stale or variable
wind) stay on the proxy and say so. The median headwind is 3–8 kt — a quarter to a
third of the 20 kt window, on every airport, in the direction that makes a
ground-speed proxy read SLOW.

### 9.2 What the correction did to the verdicts

| airport | speed pass / fail, proxy → estimate | fails slow / fast, proxy → estimate | rows that moved (fail→pass, pass→fail) | marginal (±5 kt) |
|---|---|---|---|---|
| KRDU | 7,741 / 2,453 → 7,253 / 2,941 | 1,383 / 1,070 → 585 / 2,356 | 2,318 (915, 1,403) | 4,777 |
| KSJC | 5,272 / 2,209 → 6,041 / 1,440 | 1,922 / 287 → 357 / 1,083 | 2,525 (1,647, 878) | 3,403 |
| KSTL | 5,614 / 1,576 → 5,113 / 2,077 | 531 / 1,045 → 192 / 1,885 | 1,497 (498, 999) | 3,665 |
| KSMF | 2,831 / 994 → 2,904 / 921 | 731 / 263 → 154 / 767 | 1,107 (590, 517) | 1,794 |
| KMSY | 2,704 / 797 → 2,460 / 1,041 | 209 / 588 → 89 / 952 | 626 (191, 435) | 1,724 |

The "slow" cluster §2 and §5 discussed was mostly wind: it shrinks 3–5× at every
airport (KSJC 1,922 → 357). What is left is a "fast" cluster, and it is NOT weather
either — it sits on one manufacturer, again:

| KSJC type | graded | proxy slow / fast | estimate slow / fast / pass | median V_est / Vs1g(MLW) |
|---|---|---|---|---|
| B737 | 1,765 | 566 / 24 | 70 / 106 / 1,589 | 1.332 |
| B38M | 1,619 | 207 / 96 | 19 / 435 / 1,165 | 1.375 |
| B738 | 839 | 118 / 56 | 24 / 212 / 603 | 1.375 |
| E75L | 1,063 | 479 / 0 | 55 / 33 / 975 | 1.310 |
| A319 | 280 | 128 / 2 | 13 / 6 / 261 | 1.305 |
| A21N | 153 | 138 / 0 | 61 / 0 / 92 | 1.236 |

KSTL and KRDU repeat it: B38M / B738 / B739 cross at a median 1.37–1.41 × the model's
1-g stall speed at MLW (KSTL B38M 748 of 1,783 fast; KRDU B738 515 of 1,114), the
Airbus family (Cl_max 3.0 after §8) at 1.31–1.35, mid-window, and A21N at 1.24, still
slow (the landing-mass residual of §8). The 737 family's upper edge is 1.23 + 20 kt ≈
1.42 × Vs1g at these masses, so its crossings sit ON that edge — which is why the
marginal count (estimate within its ±5 kt of a bound; `speed_margin_ms` /
`speed_uncertainty_ms` on every row) is 35–50 % of the estimate-judged rows — a
20 kt window with ±5 kt on both edges leaves only half of it unambiguous.

### 9.3 Reading, and what it opens

- §8 calibrated the A320 family's Cl_max from 2.7 to 3.0 on the PROXY, i.e. on
  ground speeds that were reading 5–8 kt slow because of the wind. With the wind
  corrected the family sits mid-window, so the calibration landed at roughly the right
  place for the wrong reason, and the 737 bucket (kept at 2.7) is now the outlier in
  the other direction. **The per-type anchors must be re-derived on the corrected
  airspeed**, with the wind in the model, before any speed-fail rate is quoted as
  flight behaviour (follow-up O4 in `2026-09-07_observed_speed_gate_plan.zh.md`).
- Three candidate explanations for the 737 "fast" cluster, none decidable from the
  gate alone: the bucket's landing Cl_max 2.7 is low for the 737NG/MAX (the same
  correction §8 made for Airbus), 737 crews fly V_ref + 15–20 kt to the threshold
  (well above what AC 91-79B §5.2.2 asks — V_ref plus wind and gust additives to 50 ft),
  or the tower's 10 m wind understates the headwind at 50 ft (a log-profile factor of
  ~1.1–1.2 pushes the same way). Type-specific published V_ref tables at typical
  landing mass would separate the first from the other two.
- The A21N residual is unchanged by the wind: a mass problem, as §8 said.
- The proxy-vs-estimate split is on every row (`bounds.speed_criterion`, `wind`);
  never pool the two, and quote the marginal count next to any observed speed rate.

Reproduction: `python -m evaluation --input trajectory_data_process/outputs/harvest/<ICAO>/approach --output <scratch>` with and without `--metar-root /nonexistent`; the comparison scripts lived in the session scratchpad and are two dozen lines over the two reports' `trajectories` rows joined on `flight_key`.

## 10. The published-V_ref window (v9, 2026-09-07): the observed fleet passes

Re-evaluated on the same five observed batches, same METAR tables, same records —
only the window changed (`THRESHOLD_SPEED_GATE.md` §3.1: the FAA Aircraft
Characteristics Database approach speed per type, scaled by √(m/MALW), over the type's
published mass range for observed rows). Reports:
`trajectory_data_process/outputs/evaluation_reports/observed_v9_2026-09-07/<ICAO>_observed_report.json`
(a separate folder; nothing an experiment reads was rewritten).

### 10.1 Speed verdicts, v8 → v9

| airport | graded | pass | fail | **v9 pass** | v8 pass (§9, stall anchor + wind) | estimate / proxy rows | indeterminate |
|---|---|---|---|---|---|---|---|
| KRDU | 10,030 | 9,567 | 463 | **95.4 %** | 71.1 % | 13,289 / 1,148 | 4,409 |
| KSJC | 7,308 | 7,212 | 96 | **98.7 %** | 80.8 % | 10,341 / 810 | 3,849 |
| KSTL | 7,151 | 6,946 | 205 | **97.1 %** | 71.1 % | 8,550 / 215 | 1,618 |
| KSMF | 3,814 | 3,708 | 106 | **97.2 %** | 75.9 % | 3,922 / 307 | 417 |
| KMSY | 3,484 | 3,355 | 129 | **96.3 %** | 70.3 % | 3,928 / 221 | 666 |
| fleet | 31,787 | 30,788 | 999 | **96.9 %** | 73.8 % | 40,030 / 2,701 | 10,959 |

The indeterminate rows are named, not hidden (`speed_indeterminate_reasons`): 10,541
"airframe could not be resolved from icao24" (the same 24.7 % as before — no type, no
window) and 404 whose type publishes no minimum operating mass (GLF5 147, C25A 119,
LJ45 103, C525 35; `docs/reference_speeds/README.md` "Types with no published minimum
mass"). Every graded row has a determinate pass/fail.

### 10.2 By type (fleet, graded rows)

Every type's published speeds, its observed and MALW windows, and its per-airport pass
rates are tabulated in `2026-09-07_reference_speed_windows_by_type.zh.md`; the rows below
are the fleet's main types.

| type | pass | fail | rate | margin to nearest edge, p50 / p05 (kt) | note |
|---|---|---|---|---|---|
| B38M | 5,733 | 131 | 97.8 % | 12.9 / 2.3 | FAA 140/145 (dual flap values) |
| B737 | 5,316 | 221 | 96.0 % | 11.0 / 0.8 | FAA single value 130 kt; Eurocontrol Vat 137 |
| B738 | 4,827 | 135 | 97.3 % | 11.7 / 1.9 | FAA 140/144 |
| E75L | 3,099 | 302 | 91.1 % | 9.0 / −2.1 | FAA single value 126 kt (max flap); no published flaps-5 V_ref |
| B739 | 2,213 | 36 | 98.4 % | 13.8 / 3.8 | FAA 140/149 |
| A319 | 1,836 | 59 | 96.9 % | 11.6 / 1.4 | FAA single value 126 |
| CRJ9 | 1,650 | 0 | 100 % | 18.8 / 10.2 | FAA 132/141; MFW published |
| A321 | 1,189 | 13 | 98.9 % | 15.0 / 5.0 | |
| A320 | 991 | 9 | 99.1 % | 14.7 / 6.2 | |
| A21N | 873 | 8 | 99.1 % | 17.4 / 6.9 | the v8 "slow" residual is gone (§9) |
| B39M | 703 | 10 | 98.6 % | 15.0 / 4.5 | |
| A20N | 533 | 2 | 99.6 % | 19.0 / 9.9 | FAA 137 vs Airbus 131.5 at the same MLW: the higher value is the anchor |
| C56X | 308 | 6 | 98.1 % | 14.8 / 3.0 | |
| E170 / E190 | 156 / 134 | 19 / 20 | 89 % / 87 % | 7.4 / 6.2 | FAA single value 124 vs Eurocontrol 130/131 |
| C172 | 47 | 16 | 74.6 % | 7.3 / −10.4 | GA training traffic; 62 kt anchor, a 10 kt wind is 16 % of it |

The 737 family's v8 "fast" cluster (§9.2, 40 % of the 737NG/MAX rows) is gone: with the
published 140–150 kt anchors the family sits 11–14 kt inside its window. The Airbus
family, mid-window under v8, is now 15–19 kt inside. The A21N residual vanished with
the mass assumption.

### 10.3 Anatomy of the 999 residual fails

- **992 fast, 7 slow.** The fast overshoot past the upper edge is small: p50 **2.2 kt**,
  p90 7.2 kt; 773 of the 992 are within 5 kt of the edge, 947 within 10 kt. The seven
  slow rows are four C172s (41–50 kt CAS against a 51 kt edge), two C56X (88–94 kt vs
  97) and one GL5T (101 vs 103) — GA and bizjet rows 1–9 kt under the lower edge.
- **747 of the 992 fast fails have their RAW ground speed inside the window**: the METAR
  headwind correction pushed them over. Fast fails carry a median headwind of 7.7 kt
  (p90 15.7) against 4.9 kt (p90 11.0) on passing rows; 313 of them had more than
  10 kt on the nose, 355 a gust in the report. They cluster on windy days — KSMF
  2026-05-18 (headwind p50 21.5 kt): 23 fails of 124 graded; KRDU 2026-07-21 (15.9 kt):
  26 of 170; KSJC 2026-05-18 (12.1 kt): 29 of 152. This is the stated limit of the
  correction (§3.6): the tower's 10 m sustained wind is not the wind at 50 ft, and on a
  gusty day a crew flies V_ref + the wind additive, which the ALAR +20 kt already
  admits — a 2 kt overshoot of that edge on a 20 kt headwind day is the report, not the
  flight.
- **The type clusters are single-valued FAA rows.** E75L (302 fails, 8.9 %), B737 (221,
  4.0 %), A319 (59), E170/E190 (39): each has ONE FAA approach speed — the maximum-flap
  value — where the dual-value rows (B738 140/144, B39M 140/150, CRJ9 132/141) carry the
  reduced-flap speed on the upper edge. An E175 landed at flaps 5 or a 737-700 at flaps
  30 has a higher V_ref than the row's 126 / 130 kt, and its legitimate V_ref + additive
  crossing reads "fast" against an edge built from the other flap setting. Eurocontrol's
  Vat for the same types (E170 130, E190 131, B737 137) is 6–7 kt higher, i.e. the
  other configuration. Rule A keeps the FAA value (`docs/reference_speeds/README.md`
  "Where the three sources disagree"); a PUBLISHED reduced-flap V_ref for these types
  would move their upper edges and is the one open source item
  (`docs/code-health-followups.md` #21).
- **C172 (63 rows, 16 fast + 4 slow)** is GA training traffic on a 62 kt anchor: the
  window is 51–82 kt CAS, so the ±5–10 kt the tower wind and the 25 ft altitude quantum
  put on a 65 kt crossing is a quarter of the window. Quote it separately.

### 10.4 Reading

All these flights landed. Under v8 the window failed 26 % of them and the failures
followed the airframe family, which was the anchor's signature; under v9 it fails
3.1 %, the failures follow the day's wind and the single-flap-value rows, and the
median overshoot is two knots. That is what a ground truth is supposed to show a
correct window: residuals that are small, explainable row by row, and traceable to a
stated limit of the method (the tower wind) or of a source (a one-flap FAA row) —
not to the flights. Quote the observed speed rate as **96.9 % of graded rows (fleet),
95.4–98.7 % by airport**, with the 24.7 % unresolved-airframe rows and the 404
no-minimum-mass rows stated beside it, never pooled with computed subjects (whose
window is framed at a known mass and is 20 kt plus the flap spread wide). Read the
observed rate as a one-sided test in practice: the lower edge sits at the type's
lightest published mass (A320: 109 kt against 136 kt at MALW), so it binds only on
genuinely slow GA/bizjet rows (7 of 999 fails); the observed verdict is almost
entirely a "not too fast" verdict.

Reproduction: `python -m evaluation --input trajectory_data_process/outputs/harvest/<ICAO>/approach --output <folder>/<ICAO>_observed_report.json --metar-root data/metar` from the v9 tree; the per-type and fail-anatomy tallies are two short scripts over the reports' `trajectories` rows (`speed_result`, `speed_margin_ms`, `bounds`, `wind`, `crossing_ground_speed_ms`).

### 10.5 After the airframe-identity fix and the 172-type table (2026-09-08)

Two changes, same gate: the observed writer now names the ICAO type whenever the identity
resolves (it used to require OpenAP dynamics for a states mass the gate never reads —
9,056 rows lost their type to that), and the published table grew from 40 to 172 types
(63 with a published minimum mass; `docs/reference_speeds/README.md`). The rebuild also
brought the two LNAV/VNAV runways into the observed batch (KRDU 32 +1,604 rows of mostly
GA traffic, KSMF 35R +259 — see the KSMF 35R threshold item in `docs/open-items.md`:
those 259 lateral fails are a configured-threshold error, not the flights').

| airport | rows | speed graded | speed pass | speed indeterminate | composite pass / total | of decided | indeterminate share |
|---|---|---|---|---|---|---|---|
| KRDU | 16,043 | 13,934 | 94.0 % | 2,109 (13.1 %) | 79.8 % | 90.6 % | 12.0 % |
| KSJC | 11,157 | 8,707 | 98.3 % | 2,450 (22.0 %) | 76.7 % | 98.3 % | 22.0 % |
| KSTL | 8,769 | 8,230 | 96.9 % | 539 (6.1 %) | 88.0 % | 93.6 % | 6.0 % |
| KSMF | 4,490 | 4,282 | 95.9 % | 208 (4.6 %) | 86.1 % | 90.2 % | 4.5 % |
| KMSY | 4,150 | 3,895 | 96.4 % | 255 (6.1 %) | 84.8 % | 90.1 % | 5.9 % |
| fleet | 44,609 | 39,048 | **96.0 %** | 5,561 (**12.5 %**, was 24.7 %) | | | |

What is still indeterminate, by cause (fleet): 1,657 rows whose icao24 resolves to no
identity (followups #24/#25), and ~3,900 rows whose type has a FAA speed but NO published
minimum operating mass — bizjets almost entirely (E55P 719, CL30 196, E545 180, C680 167,
C750 157, GLF5 147, G280 127, H25B 126, C25A 119, GLF4 117, C560 105, …): Embraer publishes
no weights in any public brochure, Textron retired the older Citation cards, Gulfstream and
Honda refuse robots (the README lists each attempt). KSJC's 22 % is that bizjet share.
The GA rows now graded (P28A 1,339, C172, SR22, C208, BE36 …) bring the fleet's slow
fails from 7 to 147 (P28A 98 of them: a 62 kt lower edge on a 70 kt anchor, and the tower
wind is a quarter of that speed) — quote GA types separately, as §10.3 says.
