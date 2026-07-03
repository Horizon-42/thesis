# `approach_constraints` — approach-procedure constraints

Turns an **LPV** approach (segments + FAS geometry) into a set of inequality constraints
`g(z) ≤ 0` over the optimizer's state nodes. Started as a teaching scaffold; the numbered
TODOs ①–⑩ are now **implemented** (the numbering is kept below as a map of the math). It is the
**single source** of the corridor / glidepath / floor / course math — the optimizer
(`collocation.optimizer`) and the backend (`aeroviz_backend.procedure_segments`) consume these
functions and constants; nothing is re-derived elsewhere.

Design sources: [`../../docs/optimization_constraint_design.md`](../../docs/optimization_constraint_design.md)
and [`../../docs/lpv_final_segment.en.html`](../../docs/lpv_final_segment.en.html).

---

## 1. The one idea you must keep in mind: the `(n, e)` frame

We optimize with the **`trapezoidalNormalizedFullTransport`** scheme, so the decision state at
each node is **already metric**, in one frame anchored at the **target = LTP**:

```
z = (n, e, h, V, psi, gamma, m)
n = (lat − lat_t)·R ,   e = (lon − lon_t)·R·cos(lat_t)     # R = geokit.WGS84_A, LTP = origin (0,0)
```

Consequences (this is *why* the math below is simple):

- The aircraft horizontal position **is** `(n, e)` — a decision variable. No per-node transform.
- Convert every **fix** (lat/lon) into this same frame **once** with `TargetFrame` (`frame.py`,
  provided). The LTP is `(0, 0)`; FPAP/GARP/PFAF/IF/IAF become constant `(n, e)` points.
- Write all constraints **directly in `(n, e)`** so the constraint Jacobian stays metric and
  well-conditioned — the whole reason `Normalized` exists.
- `FullTransport` only changes the dynamics RHS; it does **not** affect any of these geometry
  constraints.

## 2. The constraint convention

Every `*_violation` function returns `g` with:

```
g ≤ 0   ⇔   satisfied            (standard NLP inequality g(x) ≤ 0)
```

Arrays of nodes → arrays of violations. Violation rows are **metres**, except the
descent-gradient rows which are **radians** — `ConstraintReport` keeps the two families apart
(`max_violation()` / `max_angular_violation()`, `is_feasible(tol_m, tol_rad)`), so an angular
violation can never hide under a metre tolerance.

**Heading convention** (the one thing that has actually caused a bug — KRDU RW32): the state
`psi` and every course from `geometry.course_bearing` are in the DYNAMICS MODEL's convention —
**0 = East (+e), counter-clockwise toward North (+n)** (`V_east = V·cosγ·cos ψ`,
`V_north = V·cosγ·sin ψ`). NOT the compass bearing; the two agree only at 45°/225°.

## 2b. Backend-agnostic — one implementation, NumPy **and** CasADi (写完即可用)

The functions you write must run **as-is** inside the CasADi NLP, not just in NumPy tests. Two
rules make that automatic:

- **Pass coordinates as scalar components** (`n, e, h, s, gamma`), not packed `(2,)` points. Each
  may be a NumPy array of node values *or* a CasADi symbol. The operators `+ − × ÷` are overloaded
  for both, so the linear parts of a constraint are backend-agnostic for free.
- **Route nonlinear ops through `mathx`** (`mathx.fabs/sqrt/atan2/sin/cos/tan/fmax/if_else`) — it
  picks `ca.*` if any argument is a CasADi type, else `np.*`. Never call `np.abs/np.hypot/
  np.arctan2/np.maximum/np.where` on a node coordinate.
- **Fixes are constants** (`A`, `B`, GARP, LTP, course width, GPA, …) — computing with plain
  NumPy/`math` on them is fine; they fold in as constants. `geometry.leg_unit` does exactly this.

`test_backend.py` asserts that the same function yields identical numbers in NumPy and CasADi —
so once your TODO passes that test, it is genuinely optimizer-ready.

## 3. Module map

| File | Role | Status |
|---|---|---|
| `frame.py` | `TargetFrame`: fixes (lat/lon) → `(n, e)` | provided (matches the optimizer) |
| `mathx.py` | NumPy/CasADi op dispatch (`fabs`, `atan2`, `if_else`, …) | provided |
| `state.py` | decision-state column layout (`north`/`east`/`altitudes`/…) | provided |
| `geometry.py` | along/cross-track, course bearing, intercept angle | implemented (①②③④) |
| `lateral.py` | RNP box corridor + LPV angular corridor | implemented (⑤⑥⑦) |
| `vertical.py` | glidepath window, step-down floor, descent cap | implemented (⑧⑨⑩) |
| `segments.py` | `SegmentSpec` + per-segment assembly | provided (composes the primitives) |
| `builder.py` | `ConstraintSet` + `ConstraintReport` (NumPy path) | provided |
| `examples.py` | a synthetic straight-in LPV approach | provided |

## 4. The constraint functions (numbering kept as a map of the math)

| # | Function (signature) | Formula (full derivation in the docstring) | Test |
|---|---|---|---|
| ① | `geometry.along_track(n,e,A,B)` | `s = (n−A_n)·u_n + (e−A_e)·u_e` | `test_geometry.py` |
| ② | `geometry.cross_track(n,e,A,B)` | signed `e_xt = (n−A_n)·u_e − (e−A_e)·u_n` | `test_geometry.py` |
| ③ | `geometry.course_bearing(A,B)` | `atan2(Δn, Δe)` — model convention (0 = E, CCW) | `test_geometry.py` |
| ④ | `geometry.intercept_angle_deg(t,c)` | `|wrap_to_pi(t−c)|·RAD2DEG`, via `mathx` | `test_geometry.py` |
| ⑤ | `lateral.box_corridor_violation(n,e,…)` | two smooth rows `±e_xt − k·halfwidth` | `test_lateral.py` |
| ⑥ | `lateral.lpv_course_halfwidth(n,e,lpv)` | `course_width · d_GARP(node)/d_GARP(LTP)` | `test_lateral.py` |
| ⑦ | `lateral.lpv_corridor_violation(n,e,lpv)` | two smooth rows `±e_xt − k·halfwidth(node)` | `test_lateral.py` |
| ⑧ | `vertical.glidepath_altitude(d,lpv)` | `TDZE + TCH + d·tan(GPA)` | `test_vertical.py` |
| ⑨ | `vertical.moc_floor(s,…)` | staircase via `mathx.if_else`/`fmax` | `test_vertical.py` |
| ⑩ | `vertical.descent_gradient_violation(gamma,…)` | `(−max_descent) − gamma` (radians) | `test_vertical.py` |

The window/floor *wrappers* (`glidepath_window_violation`, `moc_floor_violation`) call ⑧ / ⑨.
`test_backend.py` asserts NumPy/CasADi equivalence, so everything here is optimizer-ready as-is.

## 5. Run it

```bash
# from the repo root (uses the conda `aviation` env)
python -m pytest 4dTrajectory/optimization/approach_constraints/tests/ -q

# the end-to-end demo (feasible vs. deliberately-infeasible trajectory)
PYTHONPATH=4dTrajectory/optimization python -m approach_constraints
```

`python -m approach_constraints` prints a feasible report for the on-path trajectory and flags
the final-leg corridor on the off-path one.

## 6. How the CasADi optimizer consumes this package (the live wiring)

Because the functions are backend-agnostic (§2b), the bridge is direct — **no rewrite, no
`ca.Function` callback, no finite differences**. The consumer is
`collocation.optimizer.CollocationOptimizer(segments=[...])`:

1. The backend (`aeroviz_backend.procedure_segments.build_constraint_segments`) converts every
   fix once with `TargetFrame` (frame anchored at the **target = the LTP**) and builds one
   `SegmentSpec` per procedure leg. The optimizer validates the anchoring at construction (the
   final segment must end at the `(n, e)` origin).
2. The optimizer models **one PHASE per segment** (plus an unconstrained start→first-fix
   transition phase when the start is away from the procedure), and feeds each phase's symbolic
   node columns into `segment_violations_from_components`:

   ```python
   viol = ac.segment_violations_from_components(seg, n_vec, e_vec, h_vec, gamma_vec,
                                                include_lateral=(seg.lpv is not None))
   for expr in viol.values():
       g.append(expr)          # bounds: -inf ≤ g ≤ 0  (the package's sign convention)
   ```

   The FAF position + intercept angle are pinned on the phase leading into the final
   (`geometry.course_bearing` supplies the course — same `psi` convention, one source).
3. **Multiple IAFs:** build one segment list per candidate IAF and solve each separately, then
   keep the best objective — do **not** use a non-convex `min` (design-doc §5).

The `*_violation` sign convention (`g ≤ 0`) is already the IPOPT inequality form, so the upper
bound is simply `0` and the lower bound `−inf`.

## 7. References

- `../../docs/optimization_constraint_design.md` — the full constraint design (§2 frame, §3
  segments, §4 lateral, §5 multi-IAF, §6 vertical, §7 discretization).
- `../../docs/lpv_final_segment.en.html` — interactive LPV final-segment explainer (FAS data
  block, angular course, glidepath, OCS, DA).
- FAA Order 8260.58D, Chapter 3 — the primary criteria (FAS data block, LPV/GLS final, DA).
