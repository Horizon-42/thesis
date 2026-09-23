# The vocabulary's reference entries, moved verbatim

Cut on 2026-09-23 from the live `docs/reference/contracts.md` (C30, C29) and
`docs/reference/defaults.md` (H5–H12), byte for byte; the live index no longer names them.

## From `docs/reference/contracts.md`

### C30 · instruction design: issuing heading and airport elevation

The [instruction vocabulary design](../2026-09-21_box_vocabulary_design.zh.md) defines heading
words as signed turns relative to the model `psi` at issue time, positive left and negative right.
The issue reference remains fixed throughout the command, including event splits caused by other
dimensions. Equal turn tokens issued twice are distinct commands; their continuation/reissue
encoding, target set and transient envelope still need specification. Signed turns of +180 and
−180 are different, even though their final directions coincide.

The point-mass dynamics' `psi` is the physical velocity direction in moving local geographic ENU
(east zero, increasing toward north, radians), not an independent body-yaw attitude state.
`states_from_channels` undoes chart transport factors before recovering it. The fixed-airport-ENU
turn statistics provide evidence, not a bit-identical substitute for model `psi` labels.

Height words use `altitude_msl - airport_reference_point(airport)["elevation_m"]`. The runway gate
retains its own MSL elevation plus TCH; its height relative to the airport is generally nonzero.
The dynamics altitude remains MSL. Integer height targets and envelope coverage require analysis
in the new datum; the old gate-relative candidate lists are not silently reinterpreted.

These are design decisions, not changes to the live tokenizer, dynamics or saved artifacts.

### C29 · instruction signal coordinates and distribution evidence

`manoeuvre.instructions.course_frame` measures signed cross-track distance to the extended final
approach course: right-positive when looking inbound, left-negative. Relative ground-track angle
uses the opposite sign convention: left-positive. Height is the chart vertical coordinate relative
to `FlightSeries.target_chart`; the target constructed by `flight_scenarios.runway_target` includes
threshold elevation **plus threshold crossing height**, so negative relative height does not imply
being below the runway surface. Preserve these distinctions when defining explicit target words.

The five-airport train/validation distributions, signed cross-track histograms, approximately level
height segments and proposed integer target sets are in
[指令目标分布与整数词表建议](../2026-09-22_instruction_dictionary_distribution.zh.md).
Those sets are pending user review, not an adopted dictionary or a model evaluation.

## From `docs/reference/defaults.md`

### H5 · the altitude word's 1000 ft bin is indistinguishable from random rounding (2026-09-21)

Measured after the user challenged the bin ("不会太粗吗?"). Every number below is off the five
airports' full arrival manifests — **42,604 flights, 75,534 altitude instructions** (the labeller
refused 1 track, a KRDU record holding a stopped row; counted, not dropped). Wider than stage B's
8,257-flight cohort, which is stated because the cohort's own numbers differ.

- **D51's stated justification does not hold.** It reads "目标离档中心 p95 都在半档以内 (128 m)".
  `altitude_bin` rounds to nearest, so the distance from any value to its assigned centre is ≤ half
  a bin **by construction**. The check cannot fail; it measured nothing. (Repo rule: a bound that
  can never bind is worse than no bound.)
- **56 % of "altitude instructions" are not instructions.** The labeller reads the threshold
  crossing as a plateau and emits an altitude target for it: on the KRDU cohort, **8,212 of 8,257
  flights have exactly one** such event below 200 ft, 45 have none, and **none has two** — so it is
  structural, not an averaging artefact. Fleet-wide it is 42,621 events with p25/p50/p75 =
  **42 / 62 / 94 ft** above the threshold. That is the landing, which the terminal word (D72)
  already states.
- **On the real levels that remain (33,124 events), the current bin is random rounding**:
  mean 235.3 / p50 240.9 / p95 472.0 ft, against a uniform-random reference of 250 / 250 / 475.
  500 ft → 112.4, 250 ft → 62.4, **200 ft → 50.9 (51 words)**, 100 ft → 26.3 (101 words). The real
  levels sit at 1600–3700 ft above the threshold with only 24.7 % on a round 500 ft and 15.3 % on
  a round 1000 ft, because the bins are anchored at the threshold while the assignments are MSL.
- **Anchoring the bins on MSL is NOT the fix** (measured, since it was proposed): fleet-wide it is
  a wash (87.6 → 83.2 ft at a 500 ft bin). It rescues KRDU (205.6 → 63.2) and KSTL and damages
  KSJC (73.8 → 93.7) and KSMF. Per-airport the current bin runs 73.8 (KSJC) to 205.6 (KRDU), and
  the spread tracks threshold elevation.

### H6 · the vertical angle is a criterion, not a word — and the two angles are different (2026-09-21)

The user proposed a vertical-angle word mirroring the heading word ("就像 LPV 规定一样"). Measured
on 13,043 altitude instructions / 7,486 flights (first 1,500 per airport; coverage stated).
**Two distinct quantities, kept apart because an earlier readout of mine conflated them:**

- **Position angle** = `atan(height above threshold / along-track distance to it)` — "am I on the
  published path". On the final segment (course ±30°, cross-track < 1 NM, before the threshold):
  p5 2.29 / p25 2.96 / **p50 3.05** / p75 3.13 / p95 3.35°; minus that runway's published
  glidepath, p50 **+0.04°**, and **88.5 % within ±0.5°**.
- **Flight path angle** = the aircraft's own descent gradient. Same instants: p5 −0.00 / p25 1.75 /
  **p50 2.52** / p75 3.47 / p95 4.73°, with **9.4 % level** and 89.9 % descending. Real descents are
  built from level and steep segments alternating around the path (a sample: 4131 ft at 12.8 NM
  descending 5.53° to regain it; 3043 ft at 12.8 NM level at 0.00° waiting for it).

**Conclusions, both from the same numbers:**
1. The position angle is an excellent **criterion** for D61's vertical check: tight, per runway,
   and referenced to a published value (KRDU 32 is **3.50°**, so it must be read per runway and
   never hard-coded to 3°). The flight path angle is not — it is broad and noisy.
2. The position angle is a **bad word**: 88.5 % of its mass lands in three 0.1° bins, so a model
   learns to always emit bin 0 — the identical pathology to always emitting altitude word 0, in
   different units.
3. **One angular scale cannot serve the whole approach.** Off the final course (40.1 % of altitude
   instructions) the same 0.1° resolution needs **598 words** to cover p1–p99 (p95 is +14.69° above
   the published path), because on downwind and base the straight line to the threshold is not the
   path the aircraft will fly. So the "angular everywhere, far field reads as large angles" variant
   is refused by measurement, and with it the cost of a word whose unit changes by segment
   (`altitude_centre_m` stays one conversion; the gate, `plan_conditioning=instruction`, the
   frontend colouring and the conditioning scaling are all untouched).

### H7 · what the altitude word is FOR: an anchor against drift, not a carrier of precision (2026-09-21)

Three measurements, in the order the user pushed for them. Together they change what the bin
width is chosen to do, so they belong with H5 rather than inside it.

**(a) Non-uniform bins win, once the gradient points at the levels.** The user asked twice for
non-uniform bins; my first two attempts measured the wrong shapes (a hand ladder whose coarse
segment's centres drifted off the round numbers, then one that was uniform-250 with a coarse tail)
and I twice reported that non-uniform buys nothing. Fitting the centres to the data instead
(Lloyd-Max on the fleet's real levels, one bin reserved for the degenerate final-descent event):

| bins | uniform | fitted |
|---|---|---|
| 11 | 235.3 ft | **134.4** |
| 21 | 112.4 | **72.7** |
| 26 | — | **53.3** |
| 41 | 62.4 | **35.5** |

So **26 fitted bins beat 41 uniform ones** (53.3 vs 62.4 ft) with the median class holding 1,275
examples instead of 242. The fitted centres are dense over 1500–3200 ft and sparse elsewhere —
exactly the shape the user described, pointed at the band the levels actually occupy (1600–3700 ft),
not at the ground. Cost: the centres become part of the data, so they freeze with the sha and a
cohort change (a v6 rebuild) would want refitting; `altitude_centre_m` becomes a 26-entry lookup.

**(b) The descent angle DOES hold, and is bimodal.** I had dismissed it as noise; that was wrong.
Read with the labeller's own plateau rule (10 s smoothing, ≥ 20 s, ±0.5°) over ~2,000 flights, the
flight path angle holds **11,199 plateaus, median 32 s** — the same order as the height plateaus.
But 13.0 % of them are level, and the 9,723 genuinely descending ones have an interquartile of only
**2.70–3.22°** (p50 3.01). So the aircraft does two things — level, or descend at ~3° — across the
WHOLE approach, not just the final segment. As a word that is one bit, and the height word already
carries it (a lower target means descend, the same target means level). D77's conclusion stands;
its stated reason ("散且带噪") does not and was replaced.

**(c) Removing the altitude word entirely is refused by its tail, not its centre.** The user
proposed dropping it: with descent pinned at 3° the altitude follows from the duration word, and
the model learns it implicitly. Reconstructing every flight's vertical profile from
`level/descend + duration + TRUE ground speed`:

- per segment: median descent 473 ft, median error **−2.1 ft**, median |error| **43.5 ft** — the
  3° constant is genuinely good, better than expected;
- per approach, accumulated: median |error| **153.4 ft**, p75 299, p95 1,189.

**Correction (2026-09-21, same day): those three per-approach figures were mislabelled.** They are
the sum of the errors made INSIDE each plateau; the transitions between plateaus were not counted
at all, so they are not the end-to-end profile error they were called. Re-measured with the
instructions tiling the track — each one in force from its own event until the next, so the
transitions belong to the preceding instruction, which is what a closed loop actually does — the
end-to-end height error at the runway with no altitude word is **median 218.1 ft, p95 1,442.6,
p99 2,327.7**. The direction of the finding is unchanged and slightly strengthened; the three
numbers above are not. Also visible only end-to-end: the worst case does not depend on any of
these choices, because **a flight that never levels off never triggers an anchor at all**.

**Method note, because this was the third instance in one day.** Three readouts were wrong in the
same way — a local quantity used as if it were a global one: an angle computed from a future
TARGET height against the present distance; a flight path angle taken as `arctan2(-dh, -ds)`,
which wraps to ±180° on downwind where the along-track distance grows; and this one, segments that
do not tile what they claim to summarise. Each was caught by the numbers being implausible, not by
reading the code. Before quoting a per-flight or per-approach aggregate built from segments, check
that the segments cover the flight.

The centre passes and the tail fails, for a structural reason: an absolute target's error is
bounded by half a bin, a rate's is unbounded and compounds over segments. The median flatters
itself because the signed errors cancel (median +4.7 ft). And this is a LOWER bound — it used the
true ground speed, while in closed loop the speed is a predicted word too, so the two errors
multiply.

**What this changes:** the altitude word's job is to bound drift, not to carry precision, because
the segment shape is already accurate to 43 ft without it. A bin chosen as an anchor can be
coarser than one chosen as a measurement — which also relieves H5's class-count problem
(200 ft leaves a median of 9 examples per class on KRDU alone). Pending numbers to settle
together: bin width under the anchor reading, uniform vs fitted, and the cohort (KRDU alone vs
five airports, which moves the per-class counts by an order of magnitude).

### H8 · the vocabulary those measurements produced — `2b8bf25c2a36` / `segment-v14` (2026-09-21)

What H5–H7 replaced the altitude word with, decided line by line with the user and read over the
five airports. **Six kinds**, one event sequence per flight (a row only where something changed):

| kind | values | classes |
|---|---|---|
| runway | `AIRPORT:ident` — qualified, because idents collide across airports | the COHORT's thresholds: **22** on the pooled cohort (the manifests hold 23 — KRDU 4 / KSJC 4 / KSTL 8 / KSMF 3 / KMSY 4 — and no KSTL 06 flight is in it) |
| heading | 72 DIRECTIONS relative to the final approach course, **5°** a bin, plus ONE POSITION — *track the centreline* (v14; see H11) | 73 |
| vertical | **flight path angle**, six modes: climb 3.0, level 0, descent 1.4 / 2.4 / 3.1 / 4.4° (**descent POSITIVE**) | 6 |
| speed | ground-speed centres fitted to the fleet: 44 56 63 68 74 79 86 93 99 107 114 121 129 138 147 157 m/s | 16 |
| duration | since the previous event, 2 s a bin, 0–300 s | 151 |
| terminal | continue / landed / go-around | 3 |

- **The tolerances are NOT in the sha** (they are decoding and read-back parameters): vertical
  level mode ±0.1° absolute, every other vertical mode ±7 % of its own angle; speed ±3 %. The
  runway classes are not in the sha either — they are per airport and travel beside the spec, so
  one vocabulary reads every airport.
- **Two segmentation regimes, because the quantities differ in kind.** Absolute targets (heading,
  speed) are TILED from the data's own 2 s rows (`tile_segments`, v14 — plateaus until then; see
  H11). The vertical is a RATE, and neither a plateau nor a value-merge can see it, so it is read
  by optimal piecewise-linear fitting of height against cumulative horizontal distance (dynamic
  programming over `VERTICAL_SEGMENTS` breakpoints, then a bottom-up merge to a fixed point).
  Measured cost 7.0 ms a flight (≈3.1 min over 26,382); greedy segmentation is 29.9 % worse, and
  the DP is not the bottleneck — `build_series` is.
- **The climb mode exists for the go-around post-training** and it is USED in the data (a segment
  must climb past 1.5° to reach it). The executor's climb clamp was 2°, under the 3° mode plus its
  band, and was raised to 4°; an invariant test now requires every mode **plus its tolerance** to
  sit inside the executor's limits.
- Read over the five-airport cohort: train 21,890 flights / 143,434 events (p50 6 a flight, gap
  p50 34 s / p95 110 s), val 4,492 / 29,412; every heading, vertical, speed and runway word is used
  at least once, duration 137/151, terminal 2/3 (the go-around value is never read — no cohort
  flight goes around).

### H9 · a sentence has to be able to LAND, and two things stopped it (2026-09-21)

The user's acceptance rule: before anything is trained on a sentence, that sentence must fly to
the runway. `run_ts.py instruction_replay` is the gate — it flies what the ARTEFACT says
(`Reading.from_dict`), never a re-reading, because a gate that re-derives its own input cannot see
the file drift from the code that wrote it. On the first v12 artefacts **36.7 %** of KRDU sentences
reached the threshold on the final; after the two fixes below, **99.6 %** (val split, 1,399 of
1,404 — straight-in 100 %, vectored 99.2 %), and the gap to the observed track fell from a p95
median of 2,035–2,586 m to **1,085 m**.

**A heading word cannot say which way round to turn.** It names a direction, so a half circle is a
coin toss and `wrap_deg(180)` is −180 on every implementation — the reconstruction turned the same
way every time and mirrored the whole track (up to 18 km). On the five-airport train split 13.4 %
of the 28,221 heading changes exceed 150° and **9.9 % are exactly a half circle**, so 17.1 % of
flights carry at least one. The reading DOES know the direction — it reads the course unwrapped,
where +178° and −182° are different numbers — so v13 spends that on intermediate targets the
aircraft actually passed through (`turn_split_deg`, `_split_long_turns`), each said when the
aircraft reaches the one before. That is also how a controller says it.

**A heading word cannot hold a LINE.** This is what stopped the vectored flights. Measured on 150
KRDU arrivals, at the moment a track first lines up with the course inside 10 km: the real tracks
are **13 m** from the centreline, the reconstructions that land 33 m, and the ones that fail
**2,464 m** — aligned with the course and flying a parallel line forever, so `to_go ≤ 0` never
coincides with "on the final". Word 0 names the COURSE, and flying the final approach course means
tracking the centreline (`target_course_deg`, intercept capped at 30°). That one change took the
landing rate 36.7 % → 94.7 % on its own. It adds nothing to the vocabulary and changes no
sentence: it is what flying the word means.

**The preview's airframe was also wrong, and it is now measured, not borrowed.** Plateau to
plateau the fleet turns at 0.67 of what a 20° bank gives (p50 over 280 turns of more than 20°),
i.e. 14°; |dV/dt| where the speed is changing is p50 0.19 / p90 0.54 / p99 0.99 m/s² over 62,383
samples, so the old 1.0 cap was the p99. A preview that turns half again too fast finishes each
turn early and flies straight while the aircraft is still turning. Worth 36.7 → 48.7 % alone —
real, but an order less than the two above.

**What the gap is NOT.** Un-quantising the words barely helps: replacing the speed WORD with the
exact plateau value the reader measured moves the gap p95 median only 2,035 → 1,669 m (−18 %),
while replacing it with the continuously observed speed gives 907 m (−55 %). The residual distance
is the piecewise-constant flying model and the missing wind, not the vocabulary's resolution.

### H10 · what the vocabulary can and cannot say — measured per SIGNAL (2026-09-21, on `segment-v13`)

**These are v13's numbers, and the speed row is why v14 exists** (H11): the tiling reader was built
to answer it, so the table below is the BEFORE. Re-measure on v14 before quoting it as current.


The replay gate answers "can a sentence reach the runway" (99.1 % pooled), but its gap mixes what
the words cannot say with what the preview cannot fly. This isolates the vocabulary: each signal is
reconstructed from the words alone and compared to the observed one, the vertical integrated over
the OBSERVED ground speed so the speed word's error cannot leak into it. Five-airport val split,
4,492 flights, every observed row; "settled" = rows at least 30 s after the event that issued the
word in force.

| signal | all rows p50 | p90 | p99 | settled p50 | settled p90 |
|---|---|---|---|---|---|
| heading (deg) | 0.57 | 25.16 | 87.74 | **0.41** | **2.35** |
| speed (m/s) | 2.86 | 26.99 | 55.66 | **2.23** | **18.04** |
| height (m) | 35.42 | 150.51 | 328.23 | 37.10 | 151.18 |

Share of the approach the aircraft spends INSIDE the word's own band: **heading 80.4 %, speed
48.5 %**.

- **Exact by construction**: the runway (a name), the terminal word (the landing is an event since
  v13), and the duration (2 s bins on the ADS-B row grid — the duration words sum to the flight's
  own span for 99.9 % of flights, the rest being the counted 300 s clamps).
- **Strong: the heading.** Settled, it is inside half a bin; every plateau-to-plateau turn is
  reproduced to p50 1° / p90 3° / max 4°. The p90 of 25° over ALL rows is the turns themselves —
  a word names a target and the aircraft takes time to reach it.
- **Middling: the vertical.** The five-segment fit itself is RMS 11 m (p50, from the artefact);
  quantising its angles to the six modes and integrating gives p50 35 m / p90 150 m of accumulated
  height error. So the six modes cost about 24 m at the median, and the error compounds because a
  rate's does.
- **Weak, and it is the READING RULE rather than the resolution: the speed.** Settled p90 is
  18 m/s and the aircraft is inside the band less than half the time. The cause is mechanical:
  only **51.3 %** of an approach is covered by a speed plateau (heading: 84.1 %) and **61 %** of it
  has |dV/dt| > 0.1 m/s² — an approach speed is a continuous deceleration, and a plateau reader can
  only describe holds. This is the same structural problem the altitude word had before the
  vertical moved to segment fitting, and the speed was left on plateaus.

**Three independent findings say this one thing.** H9's decomposition: replacing the speed word
with the continuously observed speed halved the replay gap (2,035 → 907 m) while un-quantising the
word bought 18 %. The prior's readings: recall on speed CHANGES is 0.17, missing 58 % of them. And
this: the signal is a ramp read as holds. More speed classes would not fix any of them.

**Not measured here**: the closed loop (every figure above is against the truth's own states),
multi-aircraft, and the go-around — its terminal value never occurs in this cohort.

### H11 · `segment-v14` — a position word, a tiling reader, and the runway out of the model's input (2026-09-21)

Three changes, all from auditing what v13 could not say. The English specification of the
vocabulary as it now stands is `docs/instruction_vocabulary.en.md`.

**A POSITION word, because every other word was a velocity.** The heading, the vertical and the
speed all constrain a velocity; nothing constrained a position, so a sentence had no mechanism by
which a lateral error could correct itself, and an open-loop replay landed 36.7 %. The morning's
fix put the correction in the DECODER by flying direction word 0 as a tracking law. That was wrong
twice over: the READING never said to join — measured on the v13 artefacts, 8.2 % of the rows
carrying word 0 were outside the 500 m corridor, 4.1 % beyond 2 km, and at the moment the word was
first issued the median displacement was 535 m (p90 5,019 m) — and the reading and the flying then
disagreed about what one word meant, which no artefact could have shown. The heading kind now has
`heading_established_word`, read off `course_frame`'s own `established` column, and direction word
0 is a direction again. **The landing rate is now the vocabulary's** (99.1 % KRDU val) rather than
58 points of it being the decoder's.

**A TILING reader for the absolute kinds** (the user's design, and the answer to H10's speed row).
Merging the data's own 2 s rows while the tolerance allows it, then absorbing whatever is shorter
than one instruction, gives both behaviours from one rule: a turn's slivers fold back onto its two
ends (0 → 90° over 30 s = 2 segments), a deceleration's survive (140 → 70 m/s = 8 segments, tiling
exactly). Plateaus gave the first and not the second, which is why half of every speed profile had
no word responsible for it. `departure_row` survives as the boundary refinement, so an instruction
is still issued where the aircraft STARTED moving; `settled_s` is now where it arrives at the
target, which under a tiling is no longer the segment's end.

Sentences get longer, which is the cost of describing a ramp: KRDU events per flight p50 7 → **11**
(p95 15 → 21), speed instructions p50 3 → **5** (p95 6 → 10). **And the replay is closer to the
track**: gap p95 median 1,085 → **821 m**, mean gap median 651 → **453 m** on the same KRDU val
split — a quarter to a third better, against a landing rate that fell only 99.6 → 99.1 % while
becoming honest.

**The runway leaves the model's INPUT** (`InstructionPrior.INPUT_KINDS`). The labeller writes it
constant per flight, so while the same column was also an input, "predict the runway at k+1" was
"copy the runway at k": NLL 0.001, top-1 1.000, and one of the six cross-entropies in the loss was
identically zero. Out of the input it must be inferred from the trajectory — the state tokens are
in the AIRPORT frame while the heading and vertical words are relative to the runway's course, so
the pair identifies it, by inference. Every prior reading taken before this scored a copy.

**The conditioning gains a fifth column**, because `72 × 5° = 360°` has the same cosine and sine as
direction word 0: without it an executor would be told to fly a bearing where the sentence said to
hold a line.

### H12 · if the vertical word were an ALTITUDE, how it would have to be divided (2026-09-21, KRDU only)

A design check, not a decision — the full measurement is `docs/2026-09-21_vertical_word_method.zh.md`
§4e (KRDU train, 6,851 flights, `tile_segments` with `min_instruction_s = 0`).

- **Under the tiling reader an altitude word has ONE design lever, the bin width.** The tolerance
  is nearly free (0.45 → 0.03 of a bin moves instructions a flight 5.7 → 6.5 and the level error
  not at all), because two neighbouring segments reading as one word emit one instruction: a
  flight's altitude instructions are its BIN CROSSINGS. Uniform bins therefore sit on a hyperbola —
  error p50 ≈ 0.3 × bin, instructions ≈ total descent / bin, so error × instructions ≈ 0.3 × the
  descent ≈ 360 m. Matching the angle word's 12 m profile RMS costs a 50 m bin and 32 instructions
  a flight against the angle's 4.
- **The right grading is GEOMETRIC** — bins uniform in `ln(1 + h/h0)`, the third appearance of the
  reason behind H-percentage tolerances — and it needs no change to the merge: `tile_segments`
  takes one SCALAR tolerance, which a graded grid in metres cannot express (the "tolerance under
  half the narrowest gap" invariant would force the ground's bin on the whole climb), while tiling
  the WARPED signal makes the tolerance constant there and metric-growing in height.
- **The only altitudes that are instructions are the LEVEL-OFFS** (0.97 a flight, 16.4 % of the
  time, p50 816 m over the threshold) and they sit on the 1000 ft **MSL** grid: 55.9 % within
  100 ft against a 20 % null. **Height above the THRESHOLD destroys that structure** (7.5 %, below
  the null): the threshold elevation shifts the grid by about a third of a bin, differently per
  airport. So an altitude word would have to live in MSL — the only word here not measured against
  the runway.
- **The offset from the round grid is a COMPUTABLE term, not noise** (corrected later the same
  day; the first reading quoted a pooled "+97 ft" and called it weather). It grows in proportion to
  height — 50 ft at 600–900 m, 74 at 900–1,200, 101 at 1,200–1,500, 163 at 1,500–2,000, 300 at
  2,000–3,000 — while the implied ISA deviation stays at 6.4–9.9 °C: control assigns a BAROMETRIC
  altitude and this data carries a GEOMETRIC one, which differ by about `height × ΔT/273`.
  **Removing one number a day takes the offset from a median 86 ft to 29 ft** (6,458 level
  stretches, 57 days, 69.7 % inside 50 ft). So the ladder is NOT limited to 500 ft — the limit is
  of order 100 ft — and no re-harvest is called for: the correction is a read-time scalar from the
  field's surface temperature. The vertical datum itself is correct and was verified
  (`flight_scenarios/datum.py` converts HAE→MSL once per flight on the CIFP threshold elevation;
  KRDU's four CIFP thresholds match the published values to within 0.6 ft).
- **The division the three regimes argue for**: geometric below 300 m (h0 = 50 m, step 0.30 → 0 /
  17.5 / 41 / 73 / 116 / 174 / 252 m, threshold-relative) plus the 1000 ft MSL ladder above it —
  17 words, 9.7 instructions a flight, level 21.8 / 76.2 m, under 300 m 14.2 / 38.8, end
  7.7 / 20.3, profile RMS 34.4 / 70.7. Against the angle word (6 words, 4 instructions, RMS
  12.0 / 41.4, terminal error that ACCUMULATES to 34.7 / 194.6 / 844 m): **the angle says the shape
  and pays at the end, the altitude pins the end and cannot say the shape, at 2.4× the words** —
  and about 8 of its 9.7 words a flight are staircase bookkeeping on a continuous descent that no
  controller ever spoke.
- **Reconciles with H5 rather than contradicting it**: H5 measured UNCORRECTED MSL anchoring over
  five airports and found a wash (it rescued KRDU 205.6 → 63.2 ft, damaged KSJC and KSMF). What
  decides it is the per-airport, per-season offset above, which H5 did not remove. **Only KRDU was
  read here**, and no altitude sentence has been flown through `instruction_replay` — the landing
  gate (H9) is unmeasured for this word, as is the executor contract it would need (an altitude
  target does not say a rate).
