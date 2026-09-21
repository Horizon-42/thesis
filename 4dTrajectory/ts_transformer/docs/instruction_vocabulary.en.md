# The instruction vocabulary

**In force:** sha `2b8bf25c2a36`, reading rule `segment-v14` (2026-09-21).
Code: `4dTrajectory/ts_transformer/manoeuvre/instructions.py`.
Chinese design document, with the reasoning and the campaign plan:
`2026-09-21-instruction_vocabulary_plan.zh.md`. This file is the vocabulary alone — what the words
are and how a track becomes a sequence of them.

An arrival is read as a **sentence**: a sequence of **events**, one at each moment something
changed, plus one for the landing. Each event carries six words.

## 1 The words

| kind | what it names | classes |
|---|---|---|
| **runway** | the landing threshold, as `AIRPORT:ident` | the cohort's thresholds (22 over five airports) |
| **heading** | 72 **directions** of the ground track relative to the final approach course, 5° apart, plus **one position**: *track the centreline* | 73 |
| **vertical** | the **flight path angle**: climb 3.0, level 0, descend 1.4 / 2.4 / 3.1 / 4.4 degrees. **Descent is positive** | 6 |
| **speed** | ground speed, 16 centres fitted to the fleet: 44 56 63 68 74 79 86 93 99 107 114 121 129 138 147 157 m/s | 16 |
| **duration** | time since the previous event, 2 s a bin, 0–300 s | 151 |
| **terminal** | continue / landed / go-around | 3 |

**Tolerances** — the band a word allows whatever flies it. They are **not** part of the sha,
because they change no word: read the same track under two tolerances and the sentence is
identical.

| kind | tolerance |
|---|---|
| vertical, level mode | ±0.1° absolute (a percentage of zero is no tolerance) |
| vertical, every other mode | ±7 % of the mode's own angle |
| speed | ±3 % of the word's centre |

**The runway classes are not in the sha either.** They are per cohort and travel beside the spec,
so one vocabulary reads every airport. They carry their own digest (`runway_sha256`), because two
artefacts with the same spec can disagree about what runway word 1 means.

### 1.1 Why each kind

**Runway.** Every geometric word is measured against a runway, so a sentence without one cannot
say where it is going; and in real control the runway is something the controller says out loud.
The label is qualified by airport because idents collide — KSJC and KSTL both have 12L/12R/30L/30R,
KSTL and KMSY both have 11 and 29.

**Heading.** Relative to the final approach course, so direction word 0 means "aligned with the
course" — the single commonest instruction on an approach. 5° a bin gives each class a median 497
samples on the five-airport cohort.

**The established word is a POSITION, and it is the only one.** Every other word in this vocabulary
constrains a *velocity* — the direction of the ground track, the flight path angle, the speed. A
sentence built only of velocity targets has no mechanism by which a lateral error can correct
itself: an aircraft told to fly the course's direction while 2 km abeam flies a line 2 km abeam for
ever. Measured: replaying v13 sentences open-loop, only **36.7 %** reached the threshold on the
final, and the failures arrived aligned with the course but a median **2,464 m** to the side, where
the real tracks are **13 m**. Real control says both kinds of thing and they are different kinds —
"turn left heading 270" names a direction, "cleared for the approach" names a **line** and asks the
aircraft to join and hold it. The established word is the second kind. It is still an absolute
target; the target is the centreline rather than a bearing.

**Vertical is an angle, not a height.** On an approach an aircraft does two things — hold level, or
descend near 3° — and that angle barely changes from twenty thousand feet to the threshold, while a
height word spends its resolution on a number the duration and speed words already imply. The
climb mode exists for the go-around that post-training will construct; it occurs in this data, so
it is not a placeholder, but the *terminal* word's go-around value never does.

**Speed is GROUND speed.** ADS-B carries no airspeed and the METAR on disk is surface only, so the
wind is inside this number. The centres are fitted to the distribution rather than to a round unit,
because ground speed has no whole-knot structure.

**Duration.** The issue times come off the ADS-B row grid, so a 2 s bin is lossless. The ceiling
exists because the class set must be finite; a longer gap clamps to it and is counted.

**Terminal.** One question with one answer — does the sentence stop here — rather than a separate
head.

### 1.2 What the vocabulary does not have

No "hold" word: a moment at which nothing changed is not an event and costs no token. No intercept
angle: it was the heading word's shadow, since every intercept was issued at the same instant as a
heading word, and an intercept angle is a consequence of the heading a controller assigns rather
than a target they state. No altitude word: see above. No word for another aircraft — this
vocabulary describes one arrival.

## 2 Reading a track into words

### 2.1 The runway frame

Every row of the observed track is converted into: along-track distance to the threshold,
cross-track offset (positive right of the course), the ground track relative to the final approach
course (kept **unwrapped**, which is what the rates and the segmentation are read from), height
above the threshold, ground speed, and whether the aircraft is **established** (inside the
centreline corridor **and** tracking the course). The threshold is the published landing threshold,
not the pavement end. A row slower than the minimum ground speed has no course and is refused —
a padded or corrupt row must not fabricate a turn.

### 2.2 Smoothing

The course over 6 s, the height and the speed over 10 s. Enough to remove the row-to-row noise
that would otherwise cut a segment, short enough not to move a real manoeuvre.

### 2.3 Segmentation — two kinds of quantity, two methods

**The absolute targets (heading, speed) are TILED** (`tile_segments`). Starting from the data's own
2 s rows, a segment extends while every row in it stays within the kind's tolerance of the
segment's midrange, and closes when the next row would break that; then, repeatedly, the shortest
segment under one instruction's length is folded into whichever neighbour its value is closer to,
until none is left.

One rule, two behaviours, which is the point:

- a **turn** sweeps through many bins in a few seconds, so its slivers fold away and the heading
  collapses back onto the two ends it was commanded between (0 → 90° over 30 s reads as 2
  segments);
- a **deceleration** crosses a band slowly — 0.19 m/s² across 4 m/s is 21 s — so its segments
  survive and the ramp is described (140 → 70 m/s reads as 8 segments, tiling exactly).

This replaced a plateau reader, which asked a signal to *hold* before it would call anything a
target. That is the right question for a signal that settles and the wrong one for a signal that
ramps, and an approach speed ramps: speed plateaus covered only **51.3 %** of an approach against
the heading's 84.1 %, while **61 %** of it has |dV/dt| > 0.1 m/s². The half the plateaus missed had
no word responsible for it at all.

**The vertical is a RATE, and is FITTED.** A rate describes the shape of the whole curve, so its
segments must tile it by construction. Height is taken against *cumulative horizontal distance* —
the flight path angle is that curve's slope — and cut into at most five straight pieces by dynamic
programming over the optimal breakpoints, then merged bottom-up until no piece is shorter than one
instruction. The segment count is a **ceiling, not a quota**: what it costs is reported per flight
as the RMS height error of the words' own profile against the track, so a cap that binds cannot be
silent.

### 2.4 From segments to instructions

- A segment's value is its **median**; its word is that value's class.
- An instruction is **issued** at the segment's start — the row at which the signal *left* the band
  before it, because the controller speaks and then the aircraft moves — and **settled** where the
  signal first comes within the tolerance of the new value, which is the aircraft arriving.
- A segment reading as the word already in force, or moving the signal by less than the kind's
  **minimum change** (half a bin for the heading; 3 % of the speed in force), continues the
  instruction in force and is **recorded** as absorbed, never dropped.
- **A turn wider than 150° is split.** A heading word names a direction and a direction cannot say
  which way *round*; at a half circle the short way is a coin toss and past it is simply wrong.
  The reading does know — it read the course unwrapped, where +178° and −182° are different
  numbers — so it spends that on intermediate targets the aircraft actually passed through, each
  said when the aircraft reaches the one before, which is how a controller says it too. Measured:
  13.4 % of heading changes exceed 150° and 9.9 % are exactly a half circle, so 17.1 % of flights
  carry at least one, and guessing wrong mirrors the whole track.
- **The established word replaces the direction words from the moment the aircraft joined the
  line**, read off the `established` column — the last run of it, the one that reaches the end of
  the record, because an approach joins once and lands. A leg flown *parallel* to the course while
  displaced keeps its direction word, which is what it was doing.
- **The runway** is said once, at the first event. The sentence's structure allows a change — it is
  a column at every event — but this data cannot resolve one (a late change is 1 flight in 44,622).

### 2.5 The event sequence

The instructions of every kind are sorted by issue time, and every moment at which something
changed becomes one event carrying the six words **in force** at it, plus the gap to the previous
event. No row ever repeats the one before it.

**The landing is an event.** Without it the last row is the last *change*, which on the
five-airport cohort sits a median **136 s** — about 11 km — before the record ends, and since the
duration word is the gap to the *previous* event, the final leg's length would be in no word at
all: a generated sentence could not say when it lands. With it, the duration words sum to the
flight's own span.

### 2.6 Refusing rather than making do

A track too short to hold a manoeuvre, a row with no course, a value the class set cannot name —
each is refused or clamped **and counted**, and the counts are printed with the artefact. A bounded
coverage that is not stated reads as though it never bound.

## 3 What the vocabulary can and cannot say

Every figure below is measured, and separately from the model that flies the words, so it is the
vocabulary's own error rather than a decoder's. Reconstructing each signal from the words alone
(the vertical integrated over the *observed* ground speed, so the speed word cannot leak into it),
five airports, 4,492 validation flights, every observed row; "settled" means at least 30 s after
the event that issued the word in force. **These numbers are `segment-v13`'s and are being
re-measured on v14, whose tiling reader exists because of the speed row.**

| signal | all rows p50 | p90 | settled p50 | settled p90 |
|---|---|---|---|---|
| heading (deg) | 0.57 | 25.16 | 0.41 | 2.35 |
| speed (m/s) | 2.86 | 26.99 | 2.23 | 18.04 |
| height (m) | 35.42 | 150.51 | 37.10 | 151.18 |

- **Exact by construction:** the runway (a name), the terminal word, and the duration (2 s bins on
  the ADS-B grid — the duration words sum to the flight's own span for 99.9 % of flights, the rest
  being counted clamps).
- **Strong: the heading.** Settled, inside half a bin; every plateau-to-plateau turn reproduced to
  p50 1° / p90 3° / max 4°. The p90 of 25° over *all* rows is the turns themselves — a word names a
  target and the aircraft takes time to reach it.
- **Middling: the vertical.** The five-segment fit is RMS 11 m; quantising its angles to the six
  modes and integrating gives p50 35 m of accumulated height error. A rate's error compounds.
- **The speed was the weak kind under v13**, and it was the reading rule rather than the
  resolution — hence §2.3.

**The acceptance test** (`run_ts.py instruction_replay`) flies what the artefact says and requires
the sentence to reach the threshold **on the final**. A sentence nothing can fly to the runway is
not worth training on. It flies the *file*, never a re-reading, because a gate that re-derives its
own input cannot see the artefact drift from the code that wrote it.

**Not measured:** the closed loop (every figure is against the truth's own states), multiple
aircraft, and the go-around — its terminal value never occurs in this cohort.
