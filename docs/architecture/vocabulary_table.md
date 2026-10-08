# Instruction vocabulary of the two-tier model

Source: `4dTrajectory/ts_transformer/docs/two_tier/design/vocabulary.md` §3, §8. LaTeX version: `vocabulary_table.tex`.

One row of a sentence (Δ = 4 s) says five words, one per column, in the order shown. Every column also has the word
"unchanged" (the column says nothing new); it is counted in the class numbers.

## Summary

| Column | Says | Words | Classes |
|---|---|---|---|
| Runway | expected runway, go-around | candidate runways, *go-around* | N_cand + 2 |
| Heading | track to fly | relative to runway course, 5° grid | 73 |
| Altitude | level to hold | 40 levels above airport, *no level-off* | 42 |
| Angle | steepness of the change | level, descent 1–4, climb | 7 |
| Speed | ground speed to hold | 5 m/s grid, *unspecified* | 49 |

## Table 1 — Columns

| # | Column | Words | Classes | Meaning and reference frame |
|---|---|---|---|---|
| 1 | Runway | candidate runway *k*; *go-around* | N_cand + 2 | "Expect runway *k*"; sets the runway in force *R*. *Go-around* abandons the approach and leaves *R* unchanged; the next runway word ends it. The first predicted step must name a candidate. |
| 2 | Heading | relative heading, 5° grid (72 values) | 73 | "Fly track θ = course(*R*) + 5*k*° and keep it". Relative to the course of *R*, so the same manoeuvre has the same word at every airport; class 0 is the final, class 36 the downwind. A turn is a series of words, one per 5°. |
| 3 | Altitude | target level *T* (40 levels, three segments, Table 2); *no level-off* | 42 | "Descend (climb) to *T* and keep it"; *T* is a geometric height above the airport elevation *E* (flown at *T* + *E* MSL). *No level-off*: descend at the angle in force to the threshold; not permitted while the go-around state is true. |
| 4 | Angle | level; descent 1–4; climb (Table 2) | 7 | How steep the vertical change to the altitude target is. A climb word carries no angle: the executor climbs at 1.25°, or at 1.885°–3° after a go-around. |
| 5 | Speed | target ground speed, 5 m/s grid, 20–250 m/s (47 values); *unspecified* | 49 | "Hold ground speed *V*". A change of speed is said in 5 m/s steps, flown at a_max = 1.4 m/s², so the rate comes from the words. *Unspecified*: the pilot flies the approach speed of the type; it starts at the capture row. |

**Grammar** (checked by the labeller, applied as masks when the prior decodes):

1. The first predicted step says a value in every column.
2. A runway word ends the go-around state *G*; *go-around* needs *G* false.
3. A level more than ε below the aircraft needs a descent class; more than ε above it needs the climb class.
4. *No level-off* needs a descent class.
5. *No level-off* is not permitted while *G* is true.
6. A *go-around* row with *no level-off* in force also says a level above the aircraft.

## Table 2 — Grids of the altitude and angle columns

Altitude levels (height above airport elevation *E*; 40 levels):

| Segment | Levels (m) | Step (m) | Max. rounding (m) | Band ε (m) |
|---|---|---|---|---|
| 1 | 0, 60, …, 1260 | 60 | 30 | 40 |
| 2 | 1380, …, 2700 | 120 | 60 | 70 |
| 3 | 3150, …, 5400 | 450 | 225 | 235 |

Angle classes:

| Class | Nominal angle | Range |
|---|---|---|
| Level | 0° | holds a reached level |
| Descent 1 | 1.5° | −0.5° to 2.0° |
| Descent 2 | 2.5° | 2.0° to 2.75° |
| Descent 3 | 3.0° | 2.75° to 3.75° |
| Descent 4 | 4.5° | 3.75° to 10° |
| Climb | 1.25° (go-around: 1.885°–3°) | 0.5° to 15° (labelled pieces) |

## Table 3 — Envelope (tolerance) of each word

How far the flown path may deviate and still count as following the word.

| Column | Tolerance |
|---|---|
| Runway | None; the word is exact. The aircraft must land on the runway in force. |
| Heading | Track within ±4.5° of the commanded heading. |
| Altitude | Height within a band around the level: ±40 m up to 1,200 m, ±70 m up to 2,580 m, ±235 m above. |
| Angle | Climb or descent angle within the range of its class (Table 2). |
| Speed | Ground speed within ±5 m/s of the commanded speed. |
