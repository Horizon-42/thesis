# CRJ900 VREF read off Figure 3 (chart reading, not a printed number)

Source: Bombardier **CRJ900 Airport Planning Manual, CSP C-020, Rev 11**
(`data/reference_speeds/bombardier/CRJ900APMR11.pdf`,
sha256 `987d2962bd5ac61714a631aef93dd09b582301c25821c65266b81701b9c23ef8`).

Section **00-03-03 Landing Speed Restrictions**, *Figure 3 — Landing Speed – Flaps at 45
Degrees/Slats Extended*, **PDF page 118** (printed as `00-03-03 Page 6`, dated Oct 20/2010).
Effectivity of the paragraph that calls out Figure 3: `**ON A/C 15001−15035, 15038−15039, 15042`.
The same chart for the remaining serials (`15036−15037, 15040−15041, 15043−15990`) is *Figure 4*
on **PDF page 120**; it is read here as a cross-check.

The figure is a printed graph, so this is a **measured reading, not a quoted value.**

## Weight read at

Maximum Landing Weight (MTLW) from the same manual, page `00-02-01`:
**73,500 lb (33,340 kg)** — see `bombardier_CRJ900_00-02-01.txt`.

## Axes and calibration

The chart is printed rotated 90°: on the page, the **x axis is VREF** (labelled 155 kt at the left
down to 115 kt at the right in Figure 3, 155 → 120 kt in Figure 4) and the **y axis is gross
weight** (40 ×1000 kg at the top down to 24 ×1000 kg at the bottom; the outer scale is 87 → 55
×1000 lb).

The page was rendered to greyscale with `pdftoppm -gray -r 300 -f 118 -l 118` (2550 × 3301 px)
and the printed grid located by counting dark pixels per row/column:

| axis | gridlines found | pixel range | spacing | value per interval |
|---|---|---|---|---|
| VREF (x), Fig 3 | 17 | 632 → 1702 | 66.9 px | 40 kt / 16 = **2.5 kt** |
| gross weight (y) | 33 | 422 → 2563 | 66.9 px | 16 t / 32 = **0.5 t** |
| VREF (x), Fig 4 | 15 | 716 → 1702 | 70.4 px | 35 kt / 14 = **2.5 kt** |

Interval counts agree exactly with the printed axis ranges, so the outermost gridlines are the
frame at 155 kt / 115 kt and 40 t / 24 t. Cross-check against the printed weight labels: the
calibration `y = 422 + (40 − w)·133.81` predicts y = 1225 / 1359 / 1493 px for 34 / 33 / 32 t;
the labels "34", "33", "32" render at y ≈ 1226 / 1358 / 1490 px.

Dotted mid-gridlines halve the solid spacing, so **the finest printed division is 1.25 kt.**

## Which line is which

Two nearly parallel lines are drawn. A 600 dpi crop of the label area
(`pdftoppm -png -r 600 -f 118 -l 118 -x 1500 -y 1800 -W 1000 -H 900`) shows the string
`10,000 ft` running parallel to and to the **left** of the left line, and `5000 ft & below`
running parallel to and to the **right** of the right line. So:

* left line (higher KIAS) = **10,000 ft**
* right line (lower KIAS) = **5000 ft & below** ← the sea-level line

(In Figure 4 the corresponding labels read `1000 ft` and `5000 ft & below`.)

## Reading

Row for 33,340 kg is y = 1313.2 px. That row is clear of both the horizontal gridlines and the
in-plot label text, and contains exactly two 2-px marks:

| line | x (px) | VREF |
|---|---|---|
| 10,000 ft | 1027.5 | 140.2 kt |
| **5000 ft & below** | **1046.9** | **139.5 kt** |

Cross-checks, all in KIAS at 33,340 kg:

| method | 10,000 ft | 5000 ft & below |
|---|---|---|
| single clean row, Figure 3 | 140.21 | 139.50 |
| least squares over 30 clean rows y ∈ [1150, 1500] (≈ 33.9 → 32.0 t), max residual 0.96 px / 0.36 px | 140.21 | 139.49 |
| least squares over 161 rows y ∈ [1550, 2300] (≈ 31.6 → 26.0 t), extrapolated up to MLW | 140.49 | 139.56 |
| Figure 4 (later serials), fit over y ∈ [1150, 1500] | 140.18 | 139.44 |

## Result

**VREF ≈ 139.5 KIAS at 33,340 kg (MLW), flaps 45 / slats extended, "5000 ft & below" line.**

Reading resolution: the finest printed gridline is 1.25 kt and the four independent readings above
span 0.12 kt, so the honest uncertainty is the chart's own line width — **quote as 139.5 ± 1 kt**
(i.e. ≈ 140 kt).

For comparison, the FAA Aircraft Characteristics Database gives CRJ9 `Approach_Speed_knot` = 141 kt
(min 132, max 141) at MALW 73,500 lb, and the Eurocontrol APD gives Landing Vat 135 kt.
