# Metric definitions

Definitions of the metrics as the code computes them. Source of each definition is given in its section. No interpretation
or thresholds beyond the definition itself.

Notation: a flight has a **true path** and a **predicted path**. A path is a polyline of nodes `(e, n, u, t)`: east and north
in metres in a local chart, height `u` in metres, time `t` in seconds from the **anchor** (the first node, `t = 0`). All
distances are Euclidean, in metres.

| Symbol | Meaning |
|---|---|
| $\mathbf{p}(t)$ | the true position `(e, n, u)` at time $t$ (linear interpolation between nodes) |
| $\hat{\mathbf{p}}(t)$ | the predicted position at time $t$ (linear interpolation; held at its last node after its end) |
| $T$ | the true duration of the flight (the true final time, $T > 0$) |
| $\hat{T}$ | the predicted final time |
| $K$ | the number of grid points, $K = 64$ |

---

## 1. ADE and FDE (average and final displacement error)

Source: `4dTrajectory/ts_transformer/geometry/metrics.py`, `common_physical_time_flight_metrics`.

Both errors are measured on **one grid of true physical time**: the true flight, $(0, T]$, is cut into $K$ equal steps,

$$t_k = \frac{k}{K}\,T, \qquad k = 1, \dots, K.$$

The anchor ($k = 0$) is not scored, because both paths start there. At each grid time, the displacement is the 3D distance
between the predicted and the true position:

$$d_k = \bigl\lVert \hat{\mathbf{p}}(t_k) - \mathbf{p}(t_k) \bigr\rVert_2 .$$

$$\mathrm{ADE} = \frac{1}{K}\sum_{k=1}^{K} d_k, \qquad \mathrm{FDE} = d_K .$$

- $d_K$ is the displacement at $t_K = T$: the two positions at the true landing time.
- A predicted path that ends before $T$ is held at its last node. It is charged over the whole true horizon (no truncation).
- The two paths are compared **at the same time**. A prediction that flies the right path at the wrong speed is charged the
  along-path displacement. So ADE and FDE contain the timing error (section 3 separates it).
- Per flight, ADE and FDE are one number each. Over a set of flights, the code summarises them as the mean, the median
  (p50) and the 95th percentile (p95) of the per-flight values.

**The true path in the readouts.** The exported states hold the observed rows only, and the observed rows stop a median of
380 m (6 s) short of the threshold at KRDU. The readouts therefore close the true path by a straight line to the threshold
at the true final time $T$ (`GEOMETRY_TRUTHS` = `closed`, the default), so that ADE, FDE and the arc-aligned ADE end at the
same point. The readouts state which truth they used.

**Arrival endpoint error.** Computed in the same function, from the same two paths, and reported beside the FDE. The FDE
compares the two paths at the **true** final time $T$. The arrival endpoint error lets the prediction finish first: it
compares the position of the prediction at **its own** final time $\hat{T}$ with the true position at $T$ (the true landing
point):

$$\mathrm{EPE} = \bigl\lVert \hat{\mathbf{p}}(\hat{T}) - \mathbf{p}(T) \bigr\rVert_2 .$$

(Code name: `arrival_endpoint_error_m`.) The time at which the prediction arrives does not enter: a prediction that arrives
earlier or later than the true flight, but at the same point, has $\mathrm{EPE} = 0$. The FDE of the same prediction is
not zero in that case, because at $t = T$ the prediction is not yet at the point (or has gone past it). The arrival time
error itself is reported separately as `final_time_error_s` $= \hat{T} - T$.

Quantities computed on the same grid in the same function (not part of ADE or FDE): the horizontal ADE (the same mean over
the horizontal distance only), the along-track, cross-track and vertical components of the displacement (along-track
and cross-track are taken in the direction of the true path), and `final_time_error_s` $= \hat{T} - T$.

---

## 2. Landing rate

Source: the judge, `4dTrajectory/ts_transformer/docs/two_tier/design/vocabulary.md` §5.8 (`autopilot/judge.py`).

The judge gives **one outcome** to each flight (the first event that occurs). The landing rate is the share of flights whose
outcome is `landed`:

$$\text{landing rate} = \frac{\#\{\text{flights with outcome } \texttt{landed}\}}{\#\{\text{flights in the set}\}} .$$

The denominator is **all** flights of the set. A flight that does not reach a threshold (`timeout`) or ends in any other
outcome is a flight that did not land.

**What `landed` means.** A flight is `landed` when, with the go-around state false (G false):

1. **An approach crossing happens.** The aircraft crosses the threshold plane of the runway in force R, lined up: its track
   is within 30° of the course of R, and its lateral offset is inside the landing screen (at most 1,000 m, and at most
   half the spacing to a parallel runway).
2. **Height.** At the crossing the aircraft is at most 100 m above the threshold.
3. **Lateral offset.** At the crossing the lateral offset is inside the runway limit: the half-width of the final approach
   segment cone at the threshold (106.7 m), and at most half the spacing to a parallel runway.
4. **The decision-altitude (DA) check passed.** At the DA point (the first flown row where the aircraft, on the final of R
   and with G false, descends through the DA): the height is within ±22 m of the published glidepath of R, and the lateral
   offset is inside the final approach segment cone at the distance of the DA point.

**The other outcomes** (the flight is not counted as landed). The first event wins. `dynamics_failure` (a state that is not
finite, or the stall cut-off); `ground_contact` (below the threshold elevation before the threshold);
`crossed_too_high` (a crossing higher than 100 m); `crossed_off_runway` (a crossing outside the runway limit);
`unstable_at_minimums` (a crossing inside the limits, after a DA check that failed, or without a DA point);
`crossed_other_runway` (G false and a lined-up crossing of another candidate runway); `timeout` (none of these within the
time limit: 1.5 times the remaining observed time, plus 900 s for each go-around). In a window of traffic, a loss of
separation that the commanded aircraft answers for also ends the flight and is not a landing.

A go-around (G true) is not an outcome: while G is true, no crossing is an event, and the flight continues. A flight that
goes around and then lands has the outcome `landed`. The landing rate counts it as a landing. (The post-training reward
treats it separately: $0.9^n$ for $n$ go-arounds.)

---

## 3. Arc-aligned ADE: the deviation without the landing-time error

Formal name in the code: **arc-length-aligned ADE** (`arc_aligned_ade_m`). It is one of the **time-free path metrics**
(also called the geometric metrics). Source:
`4dTrajectory/ts_transformer/geometry/geometric_metrics.py`, `arc_aligned_ade_m`.

It compares the two paths **by position along the path, not by time**. Each path is parametrised by its **own** horizontal
arc length. For a path, let $s \in [0, L]$ be the horizontal distance flown from the anchor ($L$ the total horizontal
length). The path is read at the same $K = 64$ **fractions** of its own length,

$$u_k = \frac{k}{K}, \qquad k = 1, \dots, K,$$

giving the points $\mathbf{q}(u_k L)$ (the 3D position where the aircraft has flown the fraction $u_k$ of its own horizontal
path). Then

$$\mathrm{ArcADE} = \frac{1}{K}\sum_{k=1}^{K} \bigl\lVert \hat{\mathbf{q}}(u_k \hat{L}) - \mathbf{q}(u_k L) \bigr\rVert_2 .$$

- The distance is 3D: the height difference is included.
- The anchor ($u = 0$) is not scored, as in the ADE.
- The time never enters. A prediction that flies the right path at the wrong speed, or that lands earlier or later, has
  the same ArcADE as the one that flies it at the right speed. Only the **shape and the height profile** are scored.
- Because the two paths have different lengths, the fractions fall at different distances. A predicted path that is
  shorter or longer than the true one shifts the compared points.

**The final point: arc-aligned FDE (terminal position error).** The last fraction $u_K = 1$ is the end of each path, so the
arc-aligned counterpart of the FDE is the 3D distance between the **last points** of the two paths:

$$\mathrm{ArcFDE} = \bigl\lVert \hat{\mathbf{q}}(\hat{L}) - \mathbf{q}(L) \bigr\rVert_2 .$$

The code names it `terminal_position_m` (reported as `arc_length_terminal_position_m`). It is computed by
`arc_length_geometry_metrics` in `geometry/arc_length_geometry.py`, next to the mean distance, the horizontal and vertical
errors and the length ratio, and it is the arc-length metric that the training validation reports. It does not depend on
the time. In `geometric_metrics.py` this value is not a separate output: it is the last term of the sum of the ArcADE.

**The timing that it leaves out** is reported by two companions, computed on the same fractions $u_k$:

- the **along-path lag** $\ell_k = \hat{t}(u_k) - t(u_k)$: the time at which the prediction reaches the fraction minus the
  time at which the truth does (positive: late); the code reports its median and the mean of its absolute value;
- the **duration error** $\ell_K = \hat{t}_{\text{end}} - t_{\text{end}}$: the lag at the last fraction, that is the
  difference of the two end times.

**Validity.** The metric needs each path to be a route, because it uses the arc length of the path as the parameter. A path
whose consecutive steps reverse their heading by more than 90° at more than 5 % of its nodes (a saw-tooth) has no
meaningful arc length; its arc-based values are not used (`arc_family_valid`).

**Related time-free metrics** (same file, horizontal path, both paths resampled every 100 m):

- **Chamfer distance**: the symmetric mean nearest-point distance between the two paths. It ignores the order of the
  points.
- **Discrete Fréchet distance**: the order-preserving "dog-leash" distance, the shortest leash that lets both walkers
  traverse their paths from start to end without going back.
