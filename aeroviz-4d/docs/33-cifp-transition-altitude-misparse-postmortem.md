# CIFP Transition-Altitude Misparse Postmortem

## Summary

Selecting some RNAV **initial fixes (IF)** as the trajectory-optimizer
start point produced an immediately-infeasible problem (IPOPT
`Maximum_Iterations_Exceeded` / `Infeasible_Problem_Detected`). The most
visible case was **KRDU RNAV (GPS) RWY 32**, IFs **CONCA** and **SINNO**,
which were placed at **18000 ft (5486.4 m)** only ~29 km from the runway —
a ~10.4° descent that exceeds the optimizer's flight-path-angle limit.

This was **not** a real procedure altitude. It came from the CIFP parser
reading the **Transition Altitude** field (a procedure-wide constant,
18000 ft across the US dataset) as if it were the leg's **crossing
altitude**, for IF legs that actually have *no* altitude constraint.

This is the same family as the earlier
[runway profile altitude anomaly](13-runway-profile-altitude-anomaly-postmortem.md):
an unknown altitude being filled from the wrong source. That one rendered
the affected fixes **too low** (unknown → `0`) in the *profile renderer*;
this one makes them **too high** (unknown → transition altitude) in the
*parser* itself, so it poisons every downstream consumer.

## What We Observed

`public/data/airports/KRDU/procedure-details/KRDU-R32-RW32.json` before the
fix:

| IF    | dist to thr | altitude.valueFt | geometryAltitudeFt | direct glideslope |
|-------|-------------|------------------|--------------------|-------------------|
| NOSIC | 22.2 km     | 3400             | 3400               | 2.3° (OK)         |
| CONCA | 29.2 km     | **18000**        | **18000**          | **10.4° (infeasible)** |
| SINNO | 28.9 km     | **18000**        | **18000**          | **10.5° (infeasible)** |

NOSIC is a normal final-approach IF; CONCA/SINNO are high TAA/feeder entry
fixes that, in the source data, carry **no** Altitude 1/2 constraint.

## Root Cause

ARINC 424 SIAP leg altitude fields (1-indexed columns):

```
col 83        Altitude Description   (+ / - / B / blank)
cols 85-89    Altitude 1             (the binding crossing altitude)
cols 90-94    Altitude 2             (window second bound, rare)
cols 95-99    Transition Altitude    (procedure-wide constant, e.g. 18000)
```

Column-aligned raw records (KRDU R32):

```
CONCA-IF :  ...IF                                             18000   altDesc=[ ] alt1=[     ] alt2=[     ] transAlt=[18000]
SINNO-IF :  ...IF                                             18000   altDesc=[ ] alt1=[     ] alt2=[     ] transAlt=[18000]
NOSIC-HF :  ...HF        ...    + 03400     18000             altDesc=[+] alt1=[03400] alt2=[     ] transAlt=[18000]
```

`18000` appears in the **same column on every leg** — including legs that
*also* have a real Altitude 1 (NOSIC `+03400`). That is the signature of a
procedure-wide field, not a per-leg crossing altitude.

Both parser paths fell back to it when Altitude 1/2 were empty:

* **Production path** — `cifp_parser.parse_procedure_legs` (the `cifparse`
  library adapter):
  ```python
  altitude_ft = normalize_int(primary.get("alt_1"))
  if altitude_ft is None:
      altitude_ft = normalize_int(primary.get("alt_2"))
  if altitude_ft is None:
      altitude_ft = normalize_int(primary.get("trans_alt"))   # ← BUG
  ```
* **Legacy fixed-width path** — `cifp_parser.parse_leg_altitude_ft`:
  ```python
  secondary_altitude = line[94:99].strip()   # ← cols 95-99 = Transition Altitude
  ```

For CONCA/SINNO IF legs (`alt_1=None, alt_2=None`), both returned the
transition altitude 18000.

### Why the parser-package cross-check did not catch it

We *do* have ready-made CIFP parsers available — `cifparse` (used in
production) and `arinc424` (cross-check) — wired up in
`validate_cifp_parser_packages.py`. But that evaluation harness extracts
altitude with `first_int_field([... , "Transition Altitude"])` /
`["alt_1", "alt_2", "trans_alt"]`, i.e. it bakes in the **same** wrong
fallback. So all three parsers "agreed" on 18000. Switching parser library
would **not** have fixed this; the bug is conceptual — *transition altitude
is never a crossing altitude* — so the fix is to stop using it as one.

## Downstream Chain

```
parse_procedure_legs → leg.altitude_ft = 18000 (transition altitude)
  → preprocess_procedures: constraints.altitude = {qualifier:"at", valueFt:18000}
                           geometryAltitudeFt   = 18000
  → KRDU-R32-RW32.json
  → frontend altitudeFtForInitialFix() → IF placed at 5486.4 m
  → optimizer IF→runway-threshold ≈ 10.4° descent → infeasible (max-iter)
```

## The Fix

### A. Parser stops reading transition altitude (`cifp_parser.py`)

* Production `parse_procedure_legs`: dropped the `trans_alt` fallback; use
  only `alt_1`/`alt_2`. Set the altitude **qualifier** from the `alt_desc`
  descriptor (`+`→atOrAbove, `-`→atOrBelow, `B`→block) instead of
  hardcoding `"at"`.
* Legacy `parse_leg_altitude_ft`: scan only `line[70:94]` (through
  Altitude 2) and stop strictly before the Transition Altitude field; the
  `line[94:99]` fallback is removed. Added `parse_leg_altitude_qualifier`.

Result: CONCA/SINNO → `altitude_ft = None`; NOSIC → `3400` (atOrAbove).

### B. Frontend derives an unpublished IF's altitude (`rnavInitialFixCandidates.ts`)

`altitudeFtForInitialFix` returns a leg's **own** altitude only when it is a
finite, positive published `geometryAltitudeFt`/`altitude.valueFt` — never
the transition altitude or the fix's terrain elevation. A feeder/transition
IF with no own altitude (CONCA/SINNO) is **not discarded**: a new
`derivedInitialFixAltitudeFt` interpolates an altitude from the nearest
published fix(es) on the branch (by along-track distance when bracketed,
else the single available neighbour). So CONCA derives ~3400 ft from the
downstream NOSIC leg and stays usable as an initial state. An IF is dropped
only when *no* neighbour has a published altitude. (The earlier
"skip the IF" behaviour was changed to this on request — feeder IFs should
remain selectable with a sensible derived altitude.)

### C. Regression tests

* `tests/test_preprocess_procedures.py` — the two tests that previously
  *asserted* the buggy `== 18000` now assert the transition altitude is
  ignored, plus a real `at or above 3400` case with qualifier.
* `tests/test_cifp_fix_record_consistency.py` — production-path test:
  KRDU R32 CONCA/SINNO IF → `altitude_ft is None`, NOSIC → 3400 /
  atOrAbove.
* `src/data/__tests__/rnavInitialFixCandidates.test.ts` — an unpublished IF
  derives its altitude from the nearest published neighbour (downstream, or
  interpolated when bracketed); never uses a zero placeholder / terrain
  elevation; dropped only when no neighbour has an altitude.

## Regenerate the Data (procedures only — NOT a full airport rebuild)

The parser fix only changes coded data, so only the per-airport
**procedure** assets need rebuilding (`procedures.geojson`,
`procedure-details/*.json`, chart manifest). This does **not** touch
terrain / DSM / runway / waypoint builds.

```bash
cd aeroviz-4d/python
python preprocess_procedures.py \
  --cifp-root ../../data/CIFP/CIFP_260319 \
  --airport KRDU \
  --procedure-type SIAP \
  --include-all-rnav \
  --include-transitions
```

Repeat per affected airport (those with `procedure-details/` already
generated), e.g.:

```bash
for ap in CYLW CYVR CYYC KMSY KRDU KSJC KSMF KSTL; do
  python preprocess_procedures.py --cifp-root ../../data/CIFP/CIFP_260319 \
    --airport "$ap" --procedure-type SIAP --include-all-rnav --include-transitions
done
```

## Guarantee / Boundary

* Transition altitude can no longer leak into any leg's crossing altitude,
  in either parser path; covered by tests.
* IFs lacking their own published altitude get one **derived** from the
  nearest published neighbour(s) on the branch (so feeder IFs stay usable),
  and are dropped only if no neighbour has an altitude. Note the derived
  altitude is a *synthesised* starting value, not a published crossing
  altitude.
* Out of scope here: a feasibility pre-check that reports "required
  glideslope X° exceeds the −γ limit" before handing the problem to IPOPT
  (tracked separately).
```
