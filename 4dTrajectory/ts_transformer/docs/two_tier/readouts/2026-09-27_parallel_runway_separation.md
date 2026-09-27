# Separation on parallel and crossing runways in the recorded arrivals

*Readout, 2026-09-27. Multi-aircraft design §3.2 and §3.4, M0 step 4 (design §6.4).*

- **Census:** `4dTrajectory/outputs/POOLED/traffic/census_20260927/census.json`, schema `ts-traffic-census-v2`, written
  at `eb583c80` on a clean tree by `experiments/traffic_census.py`. It covers the training operating days and reads the
  instruction artefact `instruction_language/v5_20260926`.
- **Examples:** [`figures/parallel_runway_separation/`](figures/parallel_runway_separation/), drawn by
  `experiments/traffic_separation_examples.py`; `examples.json` holds every number quoted in §4 and §5.

**Decision (user, 2026-09-27, provisional).** The multi-aircraft closed loop's separation checks and its reward use the
**visual reading** (§2), and the **IFR reading** is reported beside every separation readout. §7 asks one follow-up
question: the visual reading, as it stands, also assumes visual separation, which is looser than 7-4-4 c.

---

## 1 The question

The multi-aircraft closed loop ends an aircraft that loses separation, with reward 0 (design §3.4). That needs a rule for
"lost separation". We first encoded FAA JO 7110.65BB literally: 3 NM radar or 1,000 ft vertical, the wake minima, and the
parallel-runway rules. Under that rule, the **recorded** traffic loses separation 0.6–1.9 times per hour of traffic at four
of the five airports.

So is the problem in the data or in our rule? Both, in different ways:

- **The geometry is real.** The two aircraft really are within 3 NM horizontally and within 1,000 ft vertically.
- **Almost all of it happens on parallel and crossing runways.** In good weather these runways run visual approaches
  (7-4-4) and visual separation (7-2-1), which permit these distances. ADS-B does not record either.
- **Our "established" is stricter than a controller's.** It is the labeller's capture corridor, and the aircraft must
  stay inside it to the threshold. Some pairs that are established in practice are therefore judged as still turning
  in, and fall under the stricter turn-on rule.

## 2 The two readings

| Situation | IFR reading (the order as written) | Visual reading (checks and reward) |
|---|---|---|
| One runway, both established, in trail | Radar 3 NM and the directly-behind wake minimum (TBL 5-5-1); horizontal only (5-9-6 a5) | The same |
| One runway, one aircraft still joining | Radar 3 NM or 1,000 ft | The same |
| Two runways under 2,500 ft apart (KSJC 30L/30R; KSTL 12L/12R and 30L/30R) | Judged as one runway (5-5-4 h NOTE; that radar applies across the pair is our reading) | No minimum between the two runways, established or not |
| Dependent parallels, 2,500–4,300 ft (KRDU; KSTL 11/12R) | 1,000 ft or 3 NM during turn-on; diagonal 1.0 or 1.5 NM once both are established (5-9-6) | No minimum between the two runways, established or not |
| Independent parallels, ≥ 4,300 ft (KSMF) | 1,000 ft or 3 NM during turn-on (5-9-7 a1); none once both are established | No minimum between the two runways, established or not |
| Runways of other directions (KRDU 32 × 5/23, KMSY) | Radar or vertical | Both established on their finals: not judged (the crossing-runway gate, 3-10-4, is not modelled). While either is still vectored: radar or vertical |
| Leader over its threshold | TBL 5-5-2 to the established aircraft next behind, on the same runway or a pair judged as one | Same runway only |

Runway spacings and regimes are from the separation literature, `docs/literature/arrival_separation/README.md` §7. Code:
`inference/separation.py` (`IFR`, `VISUAL`).

**What the visual reading assumes.** It is looser than the conditions 7-4-4 c sets for visual approaches, which are
(JO 7110.65BB Change 3, pp. 439–440):

- **c2 (a) and c3 (a):** dependent and independent parallels, "one aircraft is turning to final and another aircraft is
  established on the extended centerline for the adjacent runway". Approved separation is provided until the aircraft
  are on a heading or course "which will intercept the extended centerline of the runway at an angle not greater than
  30 degrees" and a visual approach clearance has been acknowledged.
- **c1 (as amended by N JO 7110.805):** parallels under 2,500 ft. "Approved separation must be provided until the
  preceding aircraft is established on its extended runway centerline". Aircraft must not overtake where wake separation
  is required.
- **c4:** intersecting and converging runways. Approved separation until the visual approach clearance is
  acknowledged; "When aircraft flight paths intersect, approved separation must be maintained until visual separation is
  provided".
- **b1:** "Do not permit the respective aircrafts' primary radar targets/fusion target symbols to touch unless visual
  separation is being applied."

Before those conditions are met, the only way to be closer than radar or vertical is **visual separation** (7-2-1). A
pilot reports the traffic in sight and is told to maintain visual separation, or the tower sees both aircraft. The visual
reading drops parallel pairs whether or not the conditions are met, so wherever they are not, **it assumes visual
separation**. It also does not encode the no-overtaking condition of c1, nor visual separation between aircraft in trail
on one runway.

## 3 What the census finds

The census covers the training operating days and every arrival, with or without a sentence. The aircraft are judged at
every 2 s scene step. A pair's loss over consecutive steps is one episode. The table counts **pairs of aircraft** with at
least one loss, per hour with traffic in the scene.

| Airport | Flights | Hours with traffic | Pairs with a loss, IFR | per hour | Pairs with a loss, visual | per hour |
|---|---|---|---|---|---|---|
| KMSY | 4,882 | 360 | 11 | 0.03 | 10 | 0.03 |
| KRDU | 14,499 | 812 | 1,518 | 1.87 | 370 | 0.46 |
| KSJC | 10,057 | 587 | 526 | 0.90 | 243 | 0.41 |
| KSMF | 5,650 | 396 | 290 | 0.73 | 45 | 0.11 |
| KSTL | 9,615 | 615 | 361 | 0.59 | 249 | 0.40 |
| All | 44,703 | 2,771 | 2,706 | 0.98 | 917 | 0.33 |

At every landing, the census also judges the established aircraft next behind the leader as the leader crosses its
threshold:

| Airport | Judged (IFR / visual) | Below the required distance (IFR / visual) | Of which a wake loss, TBL 5-5-2 (IFR / visual) |
|---|---|---|---|
| KMSY | 1,396 / 1,396 | 0 / 0 | 0 / 0 |
| KRDU | 4,131 / 4,131 | 25 / 25 | 4 / 4 |
| KSJC | 5,018 / 4,641 | 322 / 133 | 36 / 27 |
| KSMF | 921 / 921 | 24 / 24 | 7 / 7 |
| KSTL | 3,573 / 3,472 | 218 / 172 | 10 / 8 |

Under IFR, the episodes group by the two aircraft's runways as follows (2,993 episodes in all):

- **KRDU:** crossing runways 809; dependent parallels 758, of which 228 are below the diagonal minimum at some step;
  same runway 143.
- **KSJC:** the close pair 30L/30R 302, same runway 268.
- **KSMF:** independent parallels 253, same runway 48, crossing 1.
- **KSTL:** same runway 269, the close pairs 64, dependent parallels 67.
- **KMSY:** same runway 10, crossing 1.

## 4 Where the IFR losses are: the two aircraft at the start of each episode

The table below describes the IFR episodes of the relations that carry most of the losses, at each episode's first step
(`examples.json`, `episodes_at_their_start`). The degree and metre rows are medians.

| | KSMF independent | KSJC close pair | KRDU dependent | KRDU crossing | KSTL same runway |
|---|---|---|---|---|---|
| Episodes | 253 | 302 | 758 | 809 | 269 |
| Both established / one / neither | 0 / 164 / 89 | 85 / 197 / 20 | 142 / 447 / 169 | 544 / 238 / 27 | 211 / 42 / 16 |
| The one ahead not yet established | 150 | 159 | 352 | 107 | 20 |
| Both headings within 10° of their courses | 12 % | 41 % | 43 % | 88 % | 88 % |
| Larger heading more than 30° off its course | 70 % | 53 % | 30 % | 9 % | 8 % |
| Larger heading off its course | 44° | 40° | 16° | 0.7° | 0.4° |
| Larger distance off its centreline | 1,882 m | 858 m | 310 m | 8.5 m | 11 m |
| Height difference | 196 m | 282 m | 292 m | 99 m | 296 m |
| Closest approach / required distance | 0.71 | 0.63 | 0.66 | 0.75 | 0.94 |

Reading each column:

- **KSMF, independent parallels (5,982 ft).** Never both established. Typically one aircraft is on its final and the
  other turns in beside it, 44° off its course and 1.9 km off its centreline, with about 200 m of height between them.
  Under IFR that is a turn-on without 1,000 ft or 3 NM, which 5-9-7 a1 does not permit. In 70 % of the episodes the
  turn is steeper than the 30° of 7-4-4 c3 (a), so even for visual approaches these need visual separation.
- **KSJC, 30L/30R (699 ft).** The order separates the pair as one runway for wake. Under IFR, one aircraft following
  another across the pair at 4–5 km is therefore "in trail under 3 NM". Visual approaches (7-4-4 c1) are how these
  runways are flown side by side. In 159 of the 302 episodes the preceding aircraft is not yet established at the first
  step, which c1 as amended does not allow without visual separation.
- **KRDU, dependent parallels (3,498 ft).** 43 % of the episodes start with both aircraft pointing down their finals,
  the farther one a median 310 m off its centreline.
  - The capture corridor (20 m + d·tan 0.45°, 2°) must hold for the rest of the approach before an aircraft counts as
    established. So these aligned aircraft are still judged by the turn-on rule (3 NM or 1,000 ft), not the 1.0 NM
    diagonal.
  - The closest approach, a median 0.66 × 3 NM ≈ 3.7 km, is well above the diagonal.
- **KRDU, crossing runways.** Runway 32, mostly general aviation, against 5L/5R/23L/23R.
  - In 544 of the 809 episodes both aircraft are established on their own finals, converging near the field with about
    100 m of height between them.
  - The crossing-runway rule (3-10-4) gates each aircraft at the threshold or the other's flight path, not by distance
    on the final. It is not modelled.
- **KSTL, same runway.** Two aircraft on one final, the one behind a median 0.94 × 3 NM ≈ 5.2 km back, just inside the
  radar minimum. These stay losses under both readings (§6).

## 5 Recorded examples

Each figure draws one TYPICAL episode of its kind: the one whose closest approach is nearest its kind's median, the
earliest of equals (rule, not choice).

- **Left panel:** the plan view in the airport frame, north up.
  - Both flights' full tracks are faint. Three minutes either side of the episode are solid, and the episode itself is
    thick.
  - Runways are drawn with their extended centrelines, and ○ marks where each aircraft is captured (established).
- **Right panel:** the two aircraft's horizontal distance (black, against 3 NM) and their height difference (green,
  against 1,000 ft) over time, with the episode shaded.

### 5.1 KSMF, turning in beside an independent parallel final

![KSMF turn-on](figures/parallel_runway_separation/turn_on_independent.svg)

- **Aircraft.** SWA3892 is established on the 17L final: 0.2° off its course, 3 m off the centreline, 7.6 km out.
  SKW3209 turns onto 17R from the west: 75° off its course, 3.2 km off its centreline, 10.1 km out.
- **The episode.** 11 September 2026, 17:20:36Z, 70 s. The distance falls from 5.55 to 4.16 km (0.75 × 3 NM), and the
  height difference stays at 186–209 m.
- **Verdict.** IFR: a turn-on loss (5-9-7 a1). Visual reading: none. By 7-4-4 c3 (a), a 75° turn beside a parallel
  final needs visual separation. It is one of 164 such episodes at KSMF.

### 5.2 KSJC, in trail across 30L/30R

![KSJC close pair](figures/parallel_runway_separation/close_parallel_pair.svg)

- **Aircraft.** SWA4310 on 30R and EJA976 on 30L, both established, 7.6 and 12.5 km out: one follows the other across
  the pair.
- **The episode.** 14 May 2026, 22:11:16Z, 106 s. The distance falls from 4.88 to 4.17 km (0.75 × 3 NM), and the
  height difference from 369 to 228 m.
- **Verdict.** IFR, which judges the pair as one runway: in trail under 3 NM. Visual reading: none. The preceding
  aircraft is established, so 7-4-4 c1 is met. It is one of 194 such episodes at KSJC.

### 5.3 KRDU, dependent parallels, both aligned but one not yet captured

![KRDU dependent](figures/parallel_runway_separation/dependent_aligned.svg)

- **Aircraft.** CXK634 is established on 23R, 2.8 km out.
- **EJA270 on 23L.** It is 0.5° off its course and 10 m off its centreline, 8.0 km out, but not yet captured. It leaves
  the corridor again before the threshold, so its capture comes later (○).
- **The episode.** 2 May 2026, 00:11:58Z, 50 s. The distance falls from 4.12 to 3.79 km (0.68 × 3 NM), and the height
  difference from 296 to 256 m.
- **Verdict.**
  - IFR judges the pair by the turn-on rule (3 NM or 1,000 ft), because one aircraft counts as still joining. Under the
    1.0 NM diagonal the same geometry would be clear.
  - Visual reading: none. The 0.5° intercept meets 7-4-4 c2 (a).
  - It is one of 186 episodes like it at KRDU.

### 5.4 KRDU, finals of crossing runways

![KRDU crossing](figures/parallel_runway_separation/crossing_established.svg)

- **Aircraft.** N181FA is established on the runway 32 final, 3.0 km out. SWA2431 is established on 05R, 1.6 km out.
- **The episode.** 1 May 2026, 15:56:02Z, 18 s. They converge near the field: the distance falls from 5.42 to 4.19 km
  (0.75 × 3 NM), with 99–113 m of height between them. The episode ends when SWA2431 lands.
- **Verdict.** IFR: radar or vertical broken. Visual reading: not judged (3-10-4 not modelled). It is one of 544 such
  episodes at KRDU.

### 5.5 KSTL, in trail on one runway (a loss under both readings)

![KSTL same runway](figures/parallel_runway_separation/same_runway_in_trail.svg)

- **Aircraft.** SWA142 is on the 30R final, 3.3 km out. SWA22 turned in behind it and follows, 8.9 km out.
- **The episode.** 8 July 2026, 04:26:06Z, 48 s. The distance falls from 5.53 to 5.28 km, against the 5.56 km minimum
  (0.95 × 3 NM), with about 285 m of height between them.
- **How it ends.** The episode ends when SWA142 lands, 5.28 km ahead of SWA22, so that landing is also below the
  required distance.
- **Verdict.** A loss under both readings. It is one of 211 at KSTL.

## 6 What the visual reading leaves

The visual reading leaves 917 pairs with a loss (1,004 episodes), 0.33 per hour with traffic. In episodes:

- **KSTL:** 269 on one runway: 229 in trail with both established, 58 with one still joining (an episode may be both).
- **KSJC:** 268 on one runway: 122 in trail, 151 joining.
- **KRDU:** 265 on crossing runways with at least one aircraft still being vectored, and 143 on one runway (30 in
  trail).
- **KSMF:** 48 on one runway (22 in trail), 1 crossing.
- **KMSY:** 10 on one runway, all joining.

Most of the same-runway losses are close calls. Where 3 NM was required, the aircraft came no closer than 2.5 NM in 239
of 268 KSTL episodes, 123 of 264 at KSJC, 68 of 143 at KRDU and 31 of 46 at KSMF.

2.5 NM is the reduced in-trail minimum of 5-5-4 j. It applies within 10 NM of the runway, and needs a documented
average runway occupancy of 50 s or less and other conditions. None of our airports is known to hold that
authorization, so it is not encoded.

Other sources of these losses are:

- visual separation between aircraft following each other (7-2-1);
- aircraft joining a final from the traffic pattern. At KRDU the same-runway joiners start a median 73° off course and
  2.6 km off the centreline. A probe during the code review of the preview census found most of KSJC's same-runway
  joiners were piston types; the runners do not measure this.

## 7 Consequences and decision

- **The IFR reading as the check would end, and zero-reward, aircraft for what the recorded controllers do routinely.**
  - The "labelled words" setting (design §2.2) would very likely fail its step-0 pass line (at most 3 % of flights
    ended by the check, design §3.4).
  - The model would learn to keep parallel arrivals further apart than the recorded operation does.
- **The visual reading matches the recorded operation, but it is more permissive than the instrument rules.**
  - In instrument weather it is unsafe.
  - The data carry no weather, and the prior deliberately takes none (prior design §2), so we cannot say which days
    were visual.
- **Decision (user, 2026-09-27, provisional).**
  - Checks and reward: visual.
  - Every separation readout also reports IFR, next to the recorded traffic's own rate under the same check (design
    §1.3: being "safer than the data" leaves the data).

### 7.1 A question for the user: should the visual reading follow 7-4-4 c's conditions?

The visual reading as it stands (§2) grants **visual separation** wherever 7-4-4 c's conditions are not met.

- Visual separation takes a traffic-in-sight report and an instruction to keep visual separation (7-2-1).
- Our vocabulary has neither word, so in the closed loop the model would be credited with an instruction it never
  gave.

A stricter visual reading keeps everything in §2, except that on parallels it assumes only visual approach clearances,
not visual separation. It would keep radar or vertical in two cases:

- **Dependent and independent parallels** (7-4-4 c2 a, c3 a): while the aircraft turning in is more than 30° off its
  course.
- **A close pair** (7-4-4 c1 as amended): while the aircraft ahead is not yet established.

**What it would change.** The census does not measure this reading, but the first step of each episode bounds it from
below. At that step:

- the turn is steeper than 30° in 70 % of the KSMF independent episodes, 30 % of the KRDU dependent ones and 21 % of the
  KSTL dependent ones;
- the aircraft ahead is not yet established in 159 of 302 KSJC close-pair episodes and 13 of 64 at KSTL.

That is about 590 episodes, so the stricter reading would land between the two readings measured: IFR 2,993, visual
1,004.

**To build it:**

- `Traffic` gains each aircraft's heading off its runway's course: the census measures it from the track, and the
  closed loop takes it from the executor's state.
- The judge gains one condition.
- The census reruns (about a minute).

**Recommendation: adopt the stricter reading for the checks and the reward.** It is what 7-4-4 c says. The one thing it
does not grant, visual separation, is the one thing the model cannot ask for. The present visual reading stays in force
until the user decides.

### 7.2 Open questions

1. Visual separation in trail on one runway (7-2-1) is not encoded under either reading; it may account for much of §6.
2. The no-overtaking condition of 7-4-4 c1 is not encoded.
3. "Established" is the labeller's capture. It is shared with the labeller and the executor, and is left as it is.
4. Weather (METAR) could separate visual days from instrument days. That would be a new data source, not planned.
5. Which approaches each airport's charts authorise was not checked (literature §7).

## 8 Reproduce

```bash
python run_ts.py traffic_census --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \
    --out 4dTrajectory/outputs/POOLED/traffic/census_20260927
python run_ts.py traffic_separation_examples --census 4dTrajectory/outputs/POOLED/traffic/census_20260927 \
    --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \
    --out 4dTrajectory/ts_transformer/docs/two_tier/readouts/figures/parallel_runway_separation
```

The census takes about a minute. The code is on branch `dev-multi-aircraft`, at `eb583c80`:

- the judge: `inference/separation.py`;
- the rules and wake tables: `inference/runway_schedule.py`;
- the scene steps and samples: `prior/scene.py`;
- the two runners above.
