# Approach Trajectory Optimization — Constraint Design

> **Target minima:** **LPV** (Localizer Performance with Vertical Guidance).
> The final segment therefore uses **angular (localizer-like) lateral containment** and a
> **published glidepath**, not a constant RNP 0.3 NM box.
>
> This document is the single source of truth for the optimizer's constraints and for the
> labels used later in model training. **Part A** is the design in English; **Part B** is the
> Chinese translation. A glossary of every acronym is in §1 / §B1.

---

# Part A — Design (English)

## 0. Scope and what the optimized trajectory represents

We optimize the **arrival/approach trajectory** of one aircraft from a start point through the
published RNAV (GPS) approach down to the runway, on an **LPV** procedure. The decision
variables are the aircraft state (geodetic position, speed `V`, heading `psi`, flight-path
angle `gamma`, mass `m`) and the controls, discretized by direct collocation (control `N`
nodes, state `N·M` nodes — see §7).

**Interpretation (important, drives how tight the corridors are):** the optimized path is the
**flyable optimal reference trajectory *inside* the legal containment tube**. It is *not* the
nominal procedure centerline (which has zero cross-track error by definition), and it is *not*
a Monte-Carlo "flown" path with navigation error. The optimizer is allowed to deviate from the
centerline **up to a design fraction of the containment half-width** in order to minimize the
objective (time / fuel / energy), while still being a path a crew could legally and stably fly.
Consequently:

- The lateral bound is `k · (containment half-width)` with a **design-margin fraction `k ≤ 1`**
  (default `k = 0.5`), not the full RNP/FSD half-width. This keeps the result off the edge of
  containment, where no real crew flies.
- All fix positions (IAF/IF/FAF, runway) come from real **CIFP** data, so the **turn angles are
  already fixed by geometry** — the optimizer does not choose fix locations. Turn-angle limits
  (§4.4) are therefore *validation checks on the data*, not optimizer constraints.

## 1. Terminology (glossary)

| Term | Meaning |
|---|---|
| **RNAV (GPS)** | The U.S. title for an RNP APCH procedure flown with GPS/SBAS. |
| **PBN / RNP / RNAV** | Performance-Based Navigation; *Required Navigation Performance* (with on-board monitoring/alerting) / *Area Navigation*. |
| **RNP value (e.g. RNP 1.0)** | Lateral navigation accuracy: the total system error stays within ± that many NM of centerline for 95% of the time. The **containment half-width** used for protection is conventionally **2× RNP**. |
| **LPV** | *Localizer Performance with Vertical Guidance.* A WAAS/SBAS approach giving ILS-like **angular** lateral + vertical guidance; supports DA as low as 200 ft HAT ("LPV-200"). |
| **LNAV / LNAV+V / LNAV-VNAV** | Other RNAV (GPS) minima lines (lateral-only / advisory vertical / barometric vertical). We do **not** target these here. |
| **IAF / IF / FAF (PFAF)** | *Initial / Intermediate / Final Approach Fix.* For vertically-guided approaches the FAF is the **PFAF (Precise FAF)** — the point where the glidepath is intercepted. |
| **MAPt** | *Missed Approach Point* (out of scope here; we optimize to the runway/DA). |
| **TAA** | *Terminal Arrival Area* — the sectoring that provides **multiple IAFs** (typically left-base, right-base, straight-in). |
| **TF leg** | *Track-to-Fix* leg — a geodesic from one fix to the next; the centerline our corridors are measured against. |
| **Cross-track error (XTE)** | Perpendicular distance from the aircraft to the leg centerline. This is what the lateral corridor bounds. |
| **Along-track distance** | Distance of the aircraft's projection measured **along** the leg centerline; used to index step-down floors and the glidepath. |
| **LTP / FTP** | *Landing / Fictitious Threshold Point* — the runway threshold reference for the FAS. Elevation = **TDZE**. |
| **FPAP** | *Flight Path Alignment Point* — defines the lateral course direction; located at/near the runway stop end. |
| **GARP** | *GNSS Azimuth Reference Point* — `FPAP + 305 m (1000 ft)` along the extended course; the apex from which the **angular** lateral deviation is measured (the "localizer antenna" analog). |
| **FAS data block** | *Final Approach Segment* data block: the coded set {LTP/FTP, FPAP, TCH, GPA, course width} that defines the LPV final geometry. |
| **FSD (full-scale deflection)** | The ±1.0 CDI limit. For LPV the **lateral FSD is ±350 ft (106.7 m) at the LTP** and **converges (narrows) toward the runway** because it is angular. |
| **CDI** | *Course Deviation Indicator* — shows deviation in fractions of FSD. |
| **GPA / VPA** | *Glide Path Angle / Vertical Path Angle* — the published descent angle. Nominal **3.0°**; standard range 2.5–3.5°; **> 3.5° (or > 1000 ft/min) is non-standard**. |
| **TCH / RDH** | *Threshold Crossing Height / Reference Datum Height* — height of the glidepath above the LTP at the threshold (typically ~50 ft). |
| **GPIP** | *Glide Path Intercept Point* — where the glidepath meets the runway, `TCH / tan(GPA)` beyond the LTP (e.g. 50 ft @ 3° → ~954 ft). |
| **DA** | *Decision Altitude* — the LPV decision gate (LPV-200 → 200 ft above TDZE). |
| **HAT** | *Height Above Touchdown* — height above TDZE; the unit DA/MDA minima are quoted in. |
| **MOC** | *Minimum Obstacle Clearance* — the clearance baked into a segment's minimum altitude. |
| **SDF / step-down** | *Step-Down Fix* — a fix imposing an "at or above" minimum crossing altitude inside a segment. |
| **Block / WINDOW altitude** | An "at-or-above **and** at-or-below" altitude constraint (a vertical window) coded on a leg. |
| **Descent gradient** | Vertical descent per horizontal distance, usually ft/NM (318 ft/NM ≈ 3.0° ≈ 5.2%). |

## 2. Reference geometry and conventions — use the optimizer's own metric frame

We optimize with a **`*Normalized`** scheme (`trapezoidalNormalizedFullTransport`), so the decision
state **already carries horizontal position as metres in one single frame anchored at the target
(the LTP)**:

```
n = (lat − lat_t) · R                 # north metres   (R = geokit WGS84_A)
e = (lon − lon_t) · R · cos(lat_t)    # east  metres   (lat_t, lon_t = the optimization target = LTP)
```

(`h, V, psi, gamma, m` are unchanged.) This is a flat equirectangular / ENU-like frame with its
**origin at the LTP** and standard parallel at `lat_t`; over a TMA (tens of NM) the projection
distortion is sub-metre. The `Normalized` change of variables is exact (the dynamics still uses the
full geodetic RHS); it exists purely to condition the NLP. Consequences for the constraints:

- **No per-leg coordinate transform is needed.** Each state node's horizontal position *is*
  `(n_k, e_k)` — a decision variable. Do **not** build a separate ENU frame per leg.
- **Convert every fix once into this same frame**, as constants, using the *identical* `R`, `lat_t`,
  `cos(lat_t)`:
  `n_fix = (lat_fix − lat_t)·R`, `e_fix = (lon_fix − lon_t)·R·cos(lat_t)`, for IAF, IF, PFAF,
  **LTP (= the origin (0,0))**, FPAP, GARP. Using a different radius or anchor would misalign the
  fixes from the aircraft state. (Reuse the optimizer's `_normalization_cb` / `geokit.WGS84_A`; do
  not re-derive.)
- **Write the constraints directly in `(n, e)`** (the normalized decision variables), **not** on
  reconstructed lat/lon radians. This keeps the constraint Jacobian metric and well-conditioned —
  the entire reason the `Normalized` scheme exists; computing corridors in radians would reintroduce
  the ill-conditioning the scheme was built to remove.
- The **LTP/target is the origin**, so horizontal distance from the threshold is simply `‖(n, e)‖`,
  and along-/cross-track reduce to dot / perpendicular products against constant fix vectors (§4).
- Heading `psi` and bearings are consistent with this frame: `+n` is north, `+e` is east, so a leg
  bearing is `atan2(Δn, Δe)` and the intercept/alignment checks compare `psi` to it directly.
- **`FullTransport` is irrelevant to coordinates** — it only adds the exact `psi` cross-term to the
  *dynamics RHS*; it does not change the constraint frame. Only the **`Normalized`** part matters
  for geometry.

For a leg from fix `A` to fix `B` (now constant `(n,e)` points) and a state node `P = (n_k, e_k)`:
- along-track unit `û = (B−A)/‖B−A‖`, **along-track** `s = (P−A)·û` (clamp to `[0, ‖B−A‖]`),
  **cross-track** `e_xt = ‖(P−A) − s·û‖` (signed if turn-side logic is needed).
All planar in the one frame; geodesic-vs-planar error is sub-metre over a single leg.
Altitudes are SI metres internally; criteria are quoted in ft/NM with SI in parentheses.

## 3. Segment definitions

The approach is split into four legs (a leg may further contain step-downs):

```
Start ──feeder──▶ IAF ──initial──▶ IF ──intermediate──▶ PFAF ──final(LPV)──▶ LTP(+TCH)
```

Each leg gets: (a) a **lateral corridor**, (b) **vertical constraints**, (c) the leg endpoints'
**fix passage** condition. Constraints are enforced at **every state node** on the leg
(§7), not only at the control nodes.

### 3.1 Final segment — PFAF → LTP (LPV; the defining change)

This is the segment that LPV changes the most. The final approach geometry comes from the
**FAS data block**: `LTP/FTP`, `FPAP`, `TCH`, `GPA`, course width.

**(a) Lateral — angular converging corridor (NOT a constant box).**
Let `D_GARP(P)` = along-course distance from the **GARP** to the aircraft's along-track
projection, and `D_GARP(LTP)` = distance from GARP to LTP = `‖LTP − FPAP‖ + 305 m`. The lateral
**full-scale half-width** at `P` is

```
w_FS(P) = 106.7 m · D_GARP(P) / D_GARP(LTP)        # 106.7 m = 350 ft at the LTP
```

i.e. a cone that is widest at the PFAF and narrows to ±350 ft at the threshold. The constraint:

```
|e_final(P)| ≤ k · w_FS(P)        for every state node P on the final segment
```

with design margin `k` (default 0.5). This is the key LPV difference: the corridor **shrinks
linearly toward the runway** instead of staying at RNP 0.3.

**(b) Vertical — ride the published glidepath.**
The glidepath is anchored to cross the LTP at `TCH` with angle `GPA`:

```
h_GP(d) = TDZE + TCH + d · tan(GPA)        # d = horizontal distance back from LTP along course
```

Constrain altitude to a thin window around it (vertical guidance, not "dive and drive"):

```
h_GP(d) − δ_v⁻ ≤ h(P) ≤ h_GP(d) + δ_v⁺
```

with a small `δ_v` (and **never below**: `δ_v⁻` tight — sub-glidepath is the dangerous side).
Equivalently, constrain the flight-path angle to a tight window around the GPA:
`gamma(P) ∈ [GPA − ε_γ, GPA + ε_γ]`. Use the altitude-window form as primary (it pins position,
not just slope); keep the `gamma` window as a smoothness aid.

**(c) Course alignment + PFAF passage.**
- Final approach course `= bearing(LTP → FPAP)`. Heading should track it; combined with (a) this
  enforces both position and direction on the course.
- The aircraft must cross the **PFAF** at its coded altitude with small tolerance (it is the
  glidepath-intercept anchor): `|e| ≤ ε_xy`, `|h − h_PFAF| ≤ ε_z`. Use a tight tolerance, not a
  hard equality, to keep the NLP feasible.

**(d) Target / termination.** Aim the path at `LTP + TCH` (the point the glidepath targets), **not**
the bare runway elevation (else you arrive ~50 ft low). The operational decision gate is the
**DA** (LPV-200 → TDZE + 200 ft); the optimization may terminate at DA or continue to LTP+TCH.

> **Auxiliary idea (kept from the original draft):** bounding the angle between the velocity
> vector and the 3-D PFAF→LTP chord is elegant but **insufficient alone** — equal-direction paths
> can be laterally/vertically offset (parallel offset has the same direction). Use it only as a
> secondary smoothness term; (a)+(b) are what actually contain the path.

### 3.2 Intermediate segment — IF → PFAF

- **Lateral:** RNP 1.0 box. `|e| ≤ k · halfwidth_intermediate`, with `halfwidth_intermediate =
  2 × RNP = 2 × 1.0 NM = 3704 m` (or use 1× RNP = 1852 m if you bound the *reference* path more
  tightly — pick one and record it in §8). The **angular LPV corridor begins only at the PFAF**;
  before it, this is a constant box.
- **Intercept angle onto the final course at the PFAF ≤ 30°** (FAA standard for establishing on a
  course between IF and FAF). Since fixes are from CIFP this is a data check (§4.4).
- **Vertical:** prefer **level** flight; cap descent at **≤ 318 ft/NM (3.0%)**. Enforce
  `h ≥ MOC_floor(s)` (step-downs) at all nodes; apply any coded WINDOW upper bound.
- **PFAF passage:** as in §3.1(c).

### 3.3 Initial segment — IAF → IF

- **Lateral:** RNP 1.0 box (same formulation as §3.2).
- **Vertical:** cap descent at **≤ 500 ft/NM**; `h ≥ MOC_floor(s)` at all nodes; coded WINDOWs.
- **Course change at the IF ≤ 90°** (data check).
- **Endpoints:** the segment runs between a *chosen* IAF and the IF — see §5 for how the IAF is
  chosen **without** a non-convex `min`.

### 3.4 Feeder segment — Start → IAF

- **Lateral:** RNP 1.0 (some feeder routes are RNP 2.0 — record the value used).
- **Vertical:** `h ≥ MOC_floor(s)`; descent ≤ 500 ft/NM.
- **Endpoint:** the *chosen* IAF (§5). Course change at the IAF ≤ ~110° for T/Y-bar offset IAFs
  (data check).

## 4. Lateral containment — formulation details

### 4.1 What is bounded
The **cross-track distance `e`** from each state node to its leg's centerline (§2), bounded by
`k ·` (that leg's containment half-width). Final segment uses the angular `w_FS(P)` (§3.1a);
the others use the per-segment RNP box.

### 4.2 Enforce at every state node
The corridor must hold **continuously**, so evaluate `e` at **all `N·M` state nodes**, not at the
2 control points (see §7 for why control-node-only checks let the path bulge out between nodes).

### 4.3 Fix passage vs corridor
The FAF/PFAF gets a **tight** `ε_xy`; the IAF/IF get the **corridor** tolerance (they may be
crossed with deviation). This matches the original asymmetric intent.

### 4.4 Turn-angle checks (data validation, not optimizer constraints)
Because fixes come from CIFP, verify (and reject bad data) rather than constrain:
`≤ 30°` intercept onto final at PFAF; `≤ 90°` course change at IF; `≤ ~110°` at offset IAFs.

## 5. Multiple IAFs — solve per-IAF, do **not** use `min`

A TAA exposes several IAFs. The original draft bounded distance to **`min` over all IAFs**. That
makes the feasible set a **union of disks → non-convex**, and `min(·)` is **non-smooth**, which
breaks the gradient-based NLP (IPOPT): the result depends on the seed (which IAF "basin" it
starts in) and the reference centerline for the initial/feeder legs becomes ambiguous.

**Adopted approach:** enumerate. For each candidate IAF, build and solve **one independent NLP**
with that IAF (and its fixed centerline) pinned, then **pick the solution with the best
objective**. This is globally optimal over the IAF choice, keeps every NLP smooth and convex-ish,
and removes the ambiguity. It maps directly onto the existing parallel scenario infrastructure
(`scenario_optimization.py`) — run the per-IAF problems as parallel scenarios.

*(If a single NLP is ever required, use a smooth soft-min `−(1/β)·log Σ exp(−β dᵢ)`; it is still
non-convex but at least differentiable. A binary-per-IAF MINLP is exact but needs a different
solver. Both are inferior to enumeration here.)*

## 6. Vertical constraints — details

### 6.1 Step-down fixes as an along-track floor (replaces "constrain one control point")
A step-down fix sits at a specific along-track location; a generic "extra control point" will not
land on it, so constraining that point's altitude is wrong. Instead build the **minimum-altitude
staircase** `MOC_floor(s)` (a function of along-track distance, rising backward through each SDF's
minimum crossing altitude) and enforce, at **every state node**:

```
h(P) ≥ MOC_floor( s(P) )
```

This covers all step-downs robustly without node-placement tricks, and reuses the same idea as the
project's obstacle/OCS minimum-altitude surfaces.

### 6.2 Block / WINDOW altitudes
Where the procedure codes an at-or-above **and** at-or-below window (the block-altitude case the
CIFP parser now decodes), apply **both** bounds: `low ≤ h(P) ≤ high`. Using only the lower bound
lets the optimizer sit too high and become unable to descend in time.

### 6.3 Maximum descent gradient (was missing)
Lower bounds alone permit "dive-and-drive". Add per-segment **descent caps** (as bounds on
`gamma` or on `Δh/Δs_horizontal`): final = the GPA itself (3.0°); intermediate ≤ 318 ft/NM
(prefer level); initial/feeder ≤ 500 ft/NM. Verify exact numbers against 8260.3/8260.58 (§9).

### 6.4 Glidepath anchoring
Final-segment altitude follows `h_GP(d)` anchored at `LTP + TCH` (§3.1b), not the runway
elevation.

## 7. Discretization and where constraints live

Distinguish three things the original draft conflated:

- **Control nodes (`N`):** where the piecewise-constant controls live (can be coarse).
- **State nodes (`N·M`):** the collocation nodes where the state is represented (dense; `M`
  auto-selected, see CLAUDE.md dense-state scheme).
- **Constraint-evaluation points:** **all state nodes**. Path constraints (lateral corridor,
  `MOC_floor`, WINDOW, gamma caps, glidepath window) are imposed on every state node.

**Procedure-leg → node mapping:** each CIFP leg spans a contiguous range of state nodes; a node
belongs to the leg whose along-track interval contains its projection. Apply that leg's corridor /
floors to those nodes. "One control point per segment start + one extra" is fine for *control*
resolution but must **not** be used as the set of points where containment is checked.

## 8. Parameter table (single source of truth)

| Symbol | Meaning | Default | Source / note |
|---|---|---|---|
| `k` | lateral design-margin fraction | 0.5 | keeps path off containment edge |
| RNP feeder | feeder lateral accuracy | 1.0 NM (some 2.0) | record per procedure |
| RNP initial | initial lateral accuracy | 1.0 NM | RNP APCH |
| RNP intermediate | intermediate lateral accuracy | 1.0 NM | RNP APCH |
| containment factor | half-width = factor × RNP | 2× (or 1× for reference path) | pick & record |
| `w_FS(LTP)` | LPV lateral FSD half-width at threshold | 106.7 m (350 ft) | LPV/WAAS |
| `D_GARP(LTP)` | GARP→LTP distance | `‖LTP−FPAP‖ + 305 m` | FAS data block |
| `GPA` | glidepath angle | 3.0° | range 2.5–3.5; >3.5° non-standard |
| `TCH` | threshold crossing height | ~50 ft (15.24 m) | FAS data block |
| `δ_v⁻ / δ_v⁺` | vertical window (below/above glidepath) | tight; below ≪ above | sub-glidepath is dangerous |
| `ε_xy, ε_z` | PFAF passage tolerance | tight | tolerance, not hard equality |
| desc. cap intermediate | max descent gradient | 318 ft/NM (3.0%) | verify 8260.x |
| desc. cap initial/feeder | max descent gradient | 500 ft/NM | verify 8260.x |
| DA (LPV-200) | decision altitude | TDZE + 200 ft | operational gate |
| intercept @ PFAF | onto final course | ≤ 30° | data check |
| course change @ IF | initial→intermediate | ≤ 90° | data check |
| course change @ IAF | offset IAF (T/Y-bar) | ≤ ~110° | data check |

## 9. Open items / to verify against primary criteria

- Exact **descent-gradient maxima** and **intermediate level-flight** rules: FAA Order **8260.3**
  (TERPS) and **8260.58** (PBN). The values above are the commonly-cited ones; pin them to the
  edition you target.
- Exact **lateral/vertical FSD** angular definitions and any **course-width floor** close-in:
  RTCA **DO-229** / the **FAS data block** definition. The ±350 ft-at-LTP cone is the working
  model; confirm the vertical FSD angle you adopt for `δ_v`.
- Whether the optimization terminates at **DA** or continues to **LTP+TCH**.

## 10. Notes for later model training

- **Consistent labels:** the per-IAF enumeration (§5) yields a single globally-optimal label per
  scenario; the `min`-based formulation would yield seed-dependent labels (different IAF basins),
  which poisons supervised/imitation learning. This is a second reason to drop `min`.
- **Smooth surrogates in the loss:** `MOC_floor` (staircase) and any `min` are non-smooth. If using
  a physics-/constraint-informed loss, replace them with `softplus`/soft-min surrogates in the
  **loss** even if the optimizer uses exact forms.
- **Constraints as features:** for a policy/imitation model, feed distance-to-corridor-edge,
  along-track margin to the next `MOC_floor` step, glidepath altitude error, and chord angle as
  inputs, so the model can learn to respect containment.

## References

- FAA ATC — establishing on a course between IF and FAF at **≤ 30°**:
  <https://www.faa.gov/air_traffic/publications/atpubs/atc_html/chap4_section_8.html>
- FAA AIM 1-2-2 — PBN / RNAV, RNP APCH segment values:
  <https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap1_section_2.html>
- AOPA — RNP APCH segment RNP values (1.0 NM init/intermediate/missed; 0.3 NM or angular final):
  <https://www.aopa.org/news-and-media/all-news/2023/june/pilot/instrument-tip-pbn-bingo>
- IVAO — final segment 3.0° = 318 ft/NM; >3.5° or >1000 ft/min non-standard:
  <https://wiki.ivao.aero/en/home/training/documentation/IFR_Approach_procedure_-_Final_approach_segment>
- SKYbrary — LPV (angular lateral+vertical, DA to 200 ft HAT):
  <https://skybrary.aero/articles/localiser-performance-vertical-guidance-lpv>
- LPV lateral course width **±350 ft at threshold**, converging:
  <https://www.boldmethod.com/learn-to-fly/navigation/what-is-the-difference-between-lpv-and-lnav-vnav-and-plus-v-gps-approaches/>
- FAA Order 8260.3 (TERPS) and 8260.58 (PBN) — primary criteria:
  <https://www.faa.gov/regulations_policies/orders_notices/index.cfm/go/document.current/documentNumber/8260.3> ·
  <https://skybrary.aero/sites/default/files/bookshelf/3605.pdf>

---

# Part B — 设计（中文）

> **目标 minima：LPV**（带垂直引导的航向性能，Localizer Performance with Vertical Guidance）。
> 因此**最终段采用角度型（类 localizer）横向包容**与**已公布的下滑道**，而不是固定的 RNP 0.3 NM 矩形走廊。
>
> 本文档是优化器约束、以及后期模型训练所用标签的**唯一权威来源**。所有缩写见 §B1 术语表。

## B0. 范围与"这条优化轨迹代表什么"

我们优化单架飞机从起点、经已公布的 RNAV (GPS) 进近、直到跑道的**进场/进近轨迹**，程序类型为 **LPV**。
决策变量是飞机状态（大地坐标位置、速度 `V`、航向 `psi`、航迹角 `gamma`、质量 `m`）与控制量，用直接配点法离散
（控制 `N` 个结点、状态 `N·M` 个结点，见 §B7）。

**含义（重要，决定走廊收得多紧）：** 这条优化轨迹是**在合法包容管道（containment tube）内、可飞的最优参考航迹**。
它**不是**程序标称中心线（中心线的 cross-track error 按定义为 0），也**不是**含导航误差的蒙特卡洛"实飞"航迹。
为了最小化目标（时间/油耗/能量），优化器**可以偏离中心线，但偏离量不超过包容半宽的某个设计比例**，同时仍然是机组可以
合法且稳定飞出的航迹。由此：

- 横向边界取 `k ·（包容半宽）`，其中**设计裕度比例 `k ≤ 1`**（默认 `k = 0.5`），不是用满 RNP/FSD 半宽。
  这样结果不会贴在包容边缘——真实机组不会在那里飞。
- 所有定位点（IAF/IF/FAF、跑道）来自真实 **CIFP** 数据，因此**转弯角已由几何固定**——优化器并不选择定位点位置。
  所以转弯角限制（§B4.4）是**对数据的校验**，不是优化器约束。

## B1. 术语表

| 术语 | 含义 |
|---|---|
| **RNAV (GPS)** | 美国对用 GPS/SBAS 飞的 RNP APCH 程序的命名。 |
| **PBN / RNP / RNAV** | 基于性能的导航；*所需导航性能*（带机载监控/告警）/ *区域导航*。 |
| **RNP 值（如 RNP 1.0）** | 横向导航精度：总系统误差在 95% 时间内保持在中心线 ± 该 NM 内。用于保护的**包容半宽**惯例取 **2× RNP**。 |
| **LPV** | *带垂直引导的航向性能*。WAAS/SBAS 进近，提供类 ILS 的**角度型**横向 + 垂直引导；DA 可低至 TDZE 上 200 ft（"LPV-200"）。 |
| **LNAV / LNAV+V / LNAV-VNAV** | RNAV (GPS) 的其它 minima 行（仅横向 / 咨询性垂直 / 气压垂直）。本方案**不**针对它们。 |
| **IAF / IF / FAF（PFAF）** | *起始 / 中间 / 最终进近定位点*。带垂直引导时 FAF 即 **PFAF（精密 FAF）**——截获下滑道之点。 |
| **MAPt** | *复飞点*（本方案不含；我们优化到跑道/DA）。 |
| **TAA** | *终端到达区*——提供**多个 IAF** 的扇区划分（通常左基线、右基线、直入）。 |
| **TF leg** | *Track-to-Fix 航段*——从一个定位点到下一个的测地线；走廊就以它为中心线度量。 |
| **横向偏差 / cross-track error (XTE)** | 飞机到航段中心线的垂直距离。横向走廊约束的就是它。 |
| **沿航迹距离 (along-track)** | 飞机投影沿中心线方向的距离；用于索引 step-down 下限与下滑道。 |
| **LTP / FTP** | *着陆/虚拟跑道入口点*——FAS 的跑道入口基准；高程 = **TDZE**。 |
| **FPAP** | *飞行航径对准点*——定义横向航道方向，位于跑道停止端附近。 |
| **GARP** | *GNSS 方位基准点*——沿延长航道 `FPAP + 305 m (1000 ft)`；**角度型**横向偏差从该顶点度量（相当于"localizer 天线"）。 |
| **FAS data block** | *最终进近段*数据块：{LTP/FTP, FPAP, TCH, GPA, 航道宽度} 的编码集合，定义 LPV 最终几何。 |
| **FSD（满刻度偏转）** | CDI 的 ±1.0 限。LPV 的**横向 FSD 在 LTP 处为 ±350 ft（106.7 m）**，并因角度型而**向跑道收窄**。 |
| **CDI** | *航道偏离指示器*——以 FSD 的比例显示偏差。 |
| **GPA / VPA** | *下滑道角 / 垂直航径角*——公布的下降角。标称 **3.0°**；标准范围 2.5–3.5°；**> 3.5°（或 > 1000 ft/min）为非标准**。 |
| **TCH / RDH** | *入口穿越高度 / 基准面高度*——下滑道在入口处高出 LTP 的高度（通常约 50 ft）。 |
| **GPIP** | *下滑道截获点*——下滑道与跑道相交处，位于 LTP 外 `TCH / tan(GPA)`（如 50 ft @ 3° → 约 954 ft）。 |
| **DA** | *决断高度*——LPV 的决断关口（LPV-200 → TDZE 上 200 ft）。 |
| **HAT** | *接地区上高度*——高出 TDZE 的高度；DA/MDA minima 用此单位。 |
| **MOC** | *最小越障余度*——已计入某段最低高度中的余度。 |
| **SDF / step-down** | *下降定位点*——在段内施加"等于或高于"的最低穿越高度。 |
| **Block / WINDOW 高度** | 同时"等于或高于"**且**"等于或低于"的高度约束（垂直窗口）。 |
| **下降梯度** | 单位水平距离的下降量，常用 ft/NM（318 ft/NM ≈ 3.0° ≈ 5.2%）。 |

## B2. 参考几何与约定 —— 直接用优化器自己的米制坐标系

我们用 **`*Normalized`** 方案（`trapezoidalNormalizedFullTransport`）优化，因此决策状态**本身就把水平位置存成
"以目标（LTP）为原点的单一坐标系下的米"**：

```
n = (lat − lat_t) · R                 # 北向米   （R = geokit WGS84_A）
e = (lon − lon_t) · R · cos(lat_t)    # 东向米   （lat_t, lon_t = 优化目标 = LTP）
```

（`h, V, psi, gamma, m` 不变。）这是一个**原点在 LTP**、标准纬线在 `lat_t` 的平面等距/类 ENU 坐标系；在一个 TMA
（几十 NM）内投影畸变是亚米级。`Normalized` 这个变量替换是**精确的**（动力学仍用完整大地 RHS），它存在的唯一目的是
**改善 NLP 条件数**。这对约束的含义：

- **不需要逐段做坐标变换。** 每个状态结点的水平位置**就是** `(n_k, e_k)`——一个决策变量。**不要**再逐段建 ENU 系。
- **把每个定位点一次性转换到这同一个坐标系**（作为常数），用**完全相同的** `R`、`lat_t`、`cos(lat_t)`：
  `n_fix = (lat_fix − lat_t)·R`，`e_fix = (lon_fix − lon_t)·R·cos(lat_t)`，对 IAF、IF、PFAF、
  **LTP（= 原点 (0,0)）**、FPAP、GARP 都这样。用了不同的半径或锚点，定位点就会与飞机状态错位。
  （复用优化器的 `_normalization_cb` / `geokit.WGS84_A`，不要重新推导。）
- **直接在 `(n, e)` 上写约束**（即用归一化的决策变量），**不要**用重建出的弧度 lat/lon。这样约束 Jacobian 保持米制、
  良态——这正是 `Normalized` 方案存在的全部理由；若改用弧度算走廊，会重新引入该方案本要消除的病态。
- **LTP/目标即原点**，所以"到入口的水平距离"就是 `‖(n, e)‖`，沿航迹/横向也退化为对常数定位点向量做点积/取垂直分量（§B4）。
- 航向 `psi` 与各航道方位与本坐标系一致：`+n` 为北、`+e` 为东，故航段方位为 `atan2(Δn, Δe)`，切入/对准校验直接拿 `psi` 比。
- **`FullTransport` 与坐标无关**——它只是给*动力学 RHS* 加上精确的 `psi` 交叉项，不改变约束坐标系。对几何而言只有
  **`Normalized`** 这部分有意义。

对从定位点 `A` 到 `B`（现在是常数 `(n,e)` 点）、状态结点 `P = (n_k, e_k)` 的航段：
- 沿航迹单位方向 `û = (B−A)/‖B−A‖`；**沿航迹** `s = (P−A)·û`（截断到 `[0, ‖B−A‖]`）；
  **横向偏差** `e_xt = ‖(P−A) − s·û‖`（如需判断转弯侧则取带符号）。
全部在这一个坐标系内做平面运算；单段内测地线与平面的误差为亚米级。内部高度用 SI 米；准则用 ft/NM 给出并在括号内附 SI。

## B3. 各段定义

进近分为四段（段内还可含 step-down）：

```
起点 ──feeder──▶ IAF ──initial──▶ IF ──intermediate──▶ PFAF ──final(LPV)──▶ LTP(+TCH)
```

每段给出：(a) **横向走廊**，(b) **垂直约束**，(c) 段端点的**定位点穿越**条件。约束在该段的
**每个状态结点**上施加（§B7），而非仅在控制结点上。

### B3.1 最终段 —— PFAF → LTP（LPV，改动最大）

LPV 改变最大的就是这一段。最终进近几何来自 **FAS data block**：`LTP/FTP`、`FPAP`、`TCH`、`GPA`、航道宽度。

**(a) 横向 —— 角度型收敛走廊（不是固定矩形）。**
设 `D_GARP(P)` = 从 **GARP** 沿航道到飞机沿航迹投影的距离，`D_GARP(LTP)` = GARP 到 LTP 的距离 =
`‖LTP − FPAP‖ + 305 m`。则在 `P` 处的横向**满刻度半宽**为

```
w_FS(P) = 106.7 m · D_GARP(P) / D_GARP(LTP)        # 106.7 m = LTP 处的 350 ft
```

即一个在 PFAF 处最宽、向入口收窄到 ±350 ft 的锥形走廊。约束：

```
|e_final(P)| ≤ k · w_FS(P)        最终段上每个状态结点 P
```

`k` 为设计裕度（默认 0.5）。这正是 LPV 的关键差异：走廊**沿向跑道线性收窄**，而非维持 RNP 0.3。

**(b) 垂直 —— 沿公布下滑道。**
下滑道以"在 LTP 处以 `TCH` 高度、按 `GPA` 角穿越"为锚：

```
h_GP(d) = TDZE + TCH + d · tan(GPA)        # d = 沿航道从 LTP 回算的水平距离
```

把高度约束到围绕它的窄窗内（垂直引导，而非"俯冲—平飞"）：

```
h_GP(d) − δ_v⁻ ≤ h(P) ≤ h_GP(d) + δ_v⁺
```

`δ_v` 取小（且**绝不可偏低过多**：`δ_v⁻` 收紧——低于下滑道是危险侧）。等价地，把航迹角约束到 GPA 附近的窄窗
`gamma(P) ∈ [GPA − ε_γ, GPA + ε_γ]`。以**高度窗**为主（它钉住的是位置而不仅是坡度），`gamma` 窗作为平滑辅助。

**(c) 航道对准 + PFAF 穿越。**
- 最终进近航道 `= bearing(LTP → FPAP)`。航向应跟随它；与 (a) 合用即可同时约束位置与方向。
- 飞机须以小容差在编码高度穿越 **PFAF**（它是下滑道截获锚点）：`|e| ≤ ε_xy`、`|h − h_PFAF| ≤ ε_z`。
  用收紧的容差而非硬等式，以保持 NLP 可行。

**(d) 目标/终止。** 把航迹瞄准 `LTP + TCH`（下滑道所指之点），**不是**裸跑道高程（否则到入口会低约 50 ft）。
运行上的决断关口是 **DA**（LPV-200 → TDZE + 200 ft）；优化可在 DA 终止，或继续到 LTP+TCH。

> **辅助想法（保留自初稿）：** 约束速度矢量与三维 PFAF→LTP 弦的夹角很优雅，但**单独不够**——同方向的航迹可以在横/纵向
> 平移（平行偏移方向相同）。只把它作为次要平滑项；真正约束航迹的是 (a)+(b)。

### B3.2 中间段 —— IF → PFAF

- **横向：** RNP 1.0 矩形。`|e| ≤ k · halfwidth_intermediate`，其中 `halfwidth_intermediate = 2 × RNP =
  2 × 1.0 NM = 3704 m`（若想把*参考*航迹收得更紧，也可用 1× RNP = 1852 m——二选一并记入 §B8）。
  **角度型 LPV 走廊只从 PFAF 开始**；在此之前是固定矩形。
- **在 PFAF 处切入最终航道的角度 ≤ 30°**（FAA 标准：在 IF 与 FAF 之间建立在某航道上的切入角）。
  因定位点来自 CIFP，这是数据校验（§B4.4）。
- **垂直：** 优先**平飞**；下降封顶 **≤ 318 ft/NM (3.0%)**。在所有结点施加 `h ≥ MOC_floor(s)`（step-down）；
  应用任何编码的 WINDOW 上界。
- **PFAF 穿越：** 同 §B3.1(c)。

### B3.3 起始段 —— IAF → IF

- **横向：** RNP 1.0 矩形（同 §B3.2）。
- **垂直：** 下降封顶 **≤ 500 ft/NM**；所有结点 `h ≥ MOC_floor(s)`；编码 WINDOW。
- **在 IF 处的航道改变 ≤ 90°**（数据校验）。
- **端点：** 本段在*被选中的* IAF 与 IF 之间——如何**不用非凸 `min`** 来选 IAF，见 §B5。

### B3.4 进场衔接段（feeder）—— 起点 → IAF

- **横向：** RNP 1.0（部分 feeder 航路为 RNP 2.0——记录所用值）。
- **垂直：** `h ≥ MOC_floor(s)`；下降 ≤ 500 ft/NM。
- **端点：** *被选中的* IAF（§B5）。T/Y 型偏置 IAF 处航道改变 ≤ 约 110°（数据校验）。

## B4. 横向包容 —— 公式细节

### B4.1 约束对象
每个状态结点到其所在段中心线的**横向偏差 `e`**（§B2），上界为 `k ·`（该段包容半宽）。最终段用角度型 `w_FS(P)`
（§B3.1a），其余段用各自的 RNP 矩形。

### B4.2 在每个状态结点上施加
走廊必须**连续**成立，故在**全部 `N·M` 个状态结点**上评估 `e`，而非在 2 个控制点上
（仅查控制结点会让航迹在结点之间鼓出走廊，原因见 §B7）。

### B4.3 定位点穿越 vs 走廊
FAF/PFAF 用**收紧的** `ε_xy`；IAF/IF 用**走廊**容差（允许带偏差穿越）。这与初稿的非对称意图一致。

### B4.4 转弯角校验（数据校验，非优化器约束）
因定位点来自 CIFP，做校验（并剔除坏数据）而非约束：PFAF 处切入最终 `≤ 30°`；IF 处航道改变 `≤ 90°`；
偏置 IAF 处 `≤ 约 110°`。

## B5. 多 IAF —— 逐 IAF 求解，**不要**用 `min`

TAA 暴露多个 IAF。初稿把距离约束在**对所有 IAF 取 `min`** 上。这会让可行域成为**多圆盘的并集 → 非凸**，且
`min(·)` **不光滑**，会破坏梯度型 NLP（IPOPT）：结果依赖初值（落入哪个 IAF 的"盆地"），且起始/feeder 段的参考中心线
变得歧义。

**采用方案：枚举。** 对每个候选 IAF，钉死该 IAF（及其固定中心线），各**独立建一个 NLP 求解**，再**取目标最优的解**。
这在 IAF 选择维度上是全局最优，保持每个 NLP 光滑且较凸，并消除歧义。它可直接套用现有的并行 scenario 框架
（`scenario_optimization.py`）——把逐 IAF 问题作为并行 scenario 跑。

*（若确需单个 NLP，用光滑软最小 `−(1/β)·log Σ exp(−β dᵢ)`；它仍非凸但至少可微。每 IAF 一个二元变量的 MINLP 精确但需换
求解器。在此场景两者都不如枚举。）*

## B6. 垂直约束 —— 细节

### B6.1 用沿航迹下限表示 step-down（取代"约束某个控制点"）
step-down 定位点位于具体的沿航迹位置；泛泛的"段中额外控制点"不会正好落在其上，故约束那个点的高度是错的。应构造
**最低高度阶梯** `MOC_floor(s)`（沿航迹距离的函数，向后逐级抬升到每个 SDF 的最低穿越高度），并在**每个状态结点**施加：

```
h(P) ≥ MOC_floor( s(P) )
```

这能稳健覆盖所有 step-down，无需结点摆放技巧，并与项目里 obstacle/OCS 的最低高度面同源。

### B6.2 Block / WINDOW 高度
凡程序编码了"等于或高于"**且**"等于或低于"的窗口（即 CIFP 解析器现已正确解码的 block 高度），同时施加**两个**界：
`low ≤ h(P) ≤ high`。只用下界会让优化器停得过高、来不及下降。

### B6.3 最大下降梯度（原方案缺失）
只有下界会放纵"俯冲—平飞"。加入逐段**下降封顶**（对 `gamma` 或 `Δh/Δs水平` 设界）：最终 = GPA 本身（3.0°）；
中间 ≤ 318 ft/NM（优先平飞）；起始/feeder ≤ 500 ft/NM。精确数值以 8260.3/8260.58 为准（§B9）。

### B6.4 下滑道锚定
最终段高度沿 `h_GP(d)`，锚在 `LTP + TCH`（§B3.1b），而非跑道高程。

## B7. 离散化与"约束放在哪里"

区分初稿混为一谈的三件事：

- **控制结点（`N`）：** 分段常值控制量所在处（可稀疏）。
- **状态结点（`N·M`）：** 表示状态的配点（密；`M` 自动选取，见 CLAUDE.md 的 dense-state 方案）。
- **约束评估点：** **全部状态结点**。路径约束（横向走廊、`MOC_floor`、WINDOW、gamma 封顶、下滑道窗）施加在每个状态结点上。

**程序航段 → 结点映射：** 每条 CIFP 航段覆盖一段连续的状态结点；某结点归属于"其投影落在该航段沿航迹区间内"的那条航段。
对这些结点施加该航段的走廊/下限。"每段起点一个控制点 + 额外一个"对*控制*分辨率没问题，但**绝不可**当作检查包容的点集。

## B8. 参数表（唯一权威来源）

| 符号 | 含义 | 默认 | 来源/备注 |
|---|---|---|---|
| `k` | 横向设计裕度比例 | 0.5 | 使航迹不贴包容边缘 |
| RNP feeder | feeder 横向精度 | 1.0 NM（部分 2.0） | 按程序记录 |
| RNP initial | 起始横向精度 | 1.0 NM | RNP APCH |
| RNP intermediate | 中间横向精度 | 1.0 NM | RNP APCH |
| 包容系数 | 半宽 = 系数 × RNP | 2×（参考航迹可用 1×） | 二选一并记录 |
| `w_FS(LTP)` | LPV 入口处横向 FSD 半宽 | 106.7 m (350 ft) | LPV/WAAS |
| `D_GARP(LTP)` | GARP→LTP 距离 | `‖LTP−FPAP‖ + 305 m` | FAS data block |
| `GPA` | 下滑道角 | 3.0° | 范围 2.5–3.5；>3.5° 非标准 |
| `TCH` | 入口穿越高度 | 约 50 ft (15.24 m) | FAS data block |
| `δ_v⁻ / δ_v⁺` | 垂直窗（下滑道下/上） | 收紧；下界 ≪ 上界 | 低于下滑道为危险侧 |
| `ε_xy, ε_z` | PFAF 穿越容差 | 收紧 | 容差，非硬等式 |
| 中间段下降封顶 | 最大下降梯度 | 318 ft/NM (3.0%) | 以 8260.x 核 |
| 起始/feeder 下降封顶 | 最大下降梯度 | 500 ft/NM | 以 8260.x 核 |
| DA (LPV-200) | 决断高度 | TDZE + 200 ft | 运行关口 |
| PFAF 处切入 | 切入最终航道 | ≤ 30° | 数据校验 |
| IF 处航道改变 | 起始→中间 | ≤ 90° | 数据校验 |
| IAF 处航道改变 | 偏置 IAF（T/Y 型） | ≤ 约 110° | 数据校验 |

## B9. 待办 / 需对照一手准则核实

- 精确的**下降梯度上限**与**中间段平飞**规则：FAA Order **8260.3**（TERPS）与 **8260.58**（PBN）。上文为常被引用的数值，
  请钉到你针对的版本。
- 精确的**横向/垂直 FSD** 角度定义及近端的**航道宽度下限**：RTCA **DO-229** / **FAS data block** 定义。
  ±350 ft @ LTP 的锥形是工作模型；确认你为 `δ_v` 采用的垂直 FSD 角。
- 优化在 **DA** 终止还是继续到 **LTP+TCH**。

## B10. 对后期模型训练的提示

- **一致的标签：** §B5 的逐 IAF 枚举为每个 scenario 给出唯一的全局最优标签；基于 `min` 的写法会给出依赖初值的标签
  （不同 IAF 盆地），毒化监督/模仿学习。这是放弃 `min` 的第二个理由。
- **损失里用光滑代理：** `MOC_floor`（阶梯）与任何 `min` 都不光滑。若用物理/约束知情的损失，即使优化器用精确形式，
  也应在**损失**里换成 `softplus`/软最小代理。
- **把约束当特征：** 对策略/模仿模型，把"到走廊边缘的距离""到下一级 `MOC_floor` 的沿航迹余量""下滑道高度误差""弦角"
  作为输入，让模型学会遵守包容。

## 参考资料

- FAA ATC —— 在 IF 与 FAF 之间以 **≤ 30°** 建立在航道上：
  <https://www.faa.gov/air_traffic/publications/atpubs/atc_html/chap4_section_8.html>
- FAA AIM 1-2-2 —— PBN / RNAV，RNP APCH 各段值：
  <https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap1_section_2.html>
- AOPA —— RNP APCH 各段 RNP 值（起始/中间/复飞 1.0 NM；最终 0.3 NM 或角度型）：
  <https://www.aopa.org/news-and-media/all-news/2023/june/pilot/instrument-tip-pbn-bingo>
- IVAO —— 最终段 3.0° = 318 ft/NM；>3.5° 或 >1000 ft/min 为非标准：
  <https://wiki.ivao.aero/en/home/training/documentation/IFR_Approach_procedure_-_Final_approach_segment>
- SKYbrary —— LPV（角度型横向+垂直，DA 低至 200 ft HAT）：
  <https://skybrary.aero/articles/localiser-performance-vertical-guidance-lpv>
- LPV 横向航道宽度 **入口处 ±350 ft**、收敛：
  <https://www.boldmethod.com/learn-to-fly/navigation/what-is-the-difference-between-lpv-and-lnav-vnav-and-plus-v-gps-approaches/>
- FAA Order 8260.3（TERPS）与 8260.58（PBN）—— 一手准则：
  <https://www.faa.gov/regulations_policies/orders_notices/index.cfm/go/document.current/documentNumber/8260.3> ·
  <https://skybrary.aero/sites/default/files/bookshelf/3605.pdf>
