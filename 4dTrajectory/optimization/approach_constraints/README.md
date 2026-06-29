# `approach_constraints` — approach-procedure constraints (teaching scaffold)

Turns an **LPV** approach (segments + FAS geometry) into a set of inequality constraints
`g(z) ≤ 0` over the optimizer's state nodes. The architecture is fully wired and tested; the
**core constraint math is left for you** as numbered TODOs with formulas, hints, and tests.

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

Arrays of nodes → arrays of violations. `ConstraintReport.max_violation() ≤ tol` ⇒ feasible.

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
| `frame.py` | `TargetFrame`: fixes (lat/lon) → `(n, e)` | **provided** (matches the optimizer) |
| `mathx.py` | NumPy/CasADi op dispatch (`fabs`, `atan2`, `if_else`, …) | **provided** |
| `state.py` | decision-state column layout (`north`/`east`/`altitudes`/…) | **provided** |
| `geometry.py` | along/cross-track, bearing, intercept angle | **TODO ①②③④** |
| `lateral.py` | RNP box corridor + LPV angular corridor | **TODO ⑤⑥⑦** |
| `vertical.py` | glidepath window, step-down floor, descent cap | **TODO ⑧⑨ (⑩ bonus)** |
| `segments.py` | `SegmentSpec` + per-segment assembly | **provided** (composes your primitives) |
| `builder.py` | `ConstraintSet` + `ConstraintReport` | **provided** |
| `examples.py` | a synthetic straight-in LPV approach | **provided** |

You only edit `geometry.py`, `lateral.py`, `vertical.py`.

## 4. Your TODO checklist (suggested order)

Do them bottom-up; each unlocks the next. Replace the `raise NotImplementedError` body, then run
that module's test — the matching `xfail` will start to **xpass**.

| # | Function (signature) | Formula (full derivation in the docstring) | Test |
|---|---|---|---|
| ① | `geometry.along_track(n,e,A,B)` | `s = (n−A_n)·u_n + (e−A_e)·u_e` | `test_geometry.py` |
| ② | `geometry.cross_track(n,e,A,B)` | signed `e_xt = (n−A_n)·u_e − (e−A_e)·u_n` | `test_geometry.py` |
| ③ | `geometry.course_bearing(A,B)` | `atan2(Δe, Δn)` | `test_geometry.py` |
| ④ | `geometry.intercept_angle_deg(t,c)` | `|wrap_to_pi(t−c)|·RAD2DEG`, via `mathx` | `test_geometry.py` |
| ⑤ | `lateral.box_corridor_violation(n,e,…)` | `mathx.fabs(cross_track) − k·halfwidth` | `test_lateral.py` |
| ⑥ | `lateral.lpv_course_halfwidth(n,e,lpv)` | `course_width · d_GARP(node)/d_GARP(LTP)` | `test_lateral.py` |
| ⑦ | `lateral.lpv_corridor_violation(n,e,lpv)` | `mathx.fabs(cross_track) − k·halfwidth(node)` | `test_lateral.py` |
| ⑧ | `vertical.glidepath_altitude(d,lpv)` | `TDZE + TCH + d·tan(GPA)` | `test_vertical.py` |
| ⑨ | `vertical.moc_floor(s,…)` | staircase via `mathx.if_else`/`fmax` | `test_vertical.py` |
| ⑩ | `vertical.descent_gradient_violation(gamma,…)` *(bonus)* | `(−max_descent) − gamma` | `test_vertical.py` |

The window/floor *wrappers* (`glidepath_window_violation`, `moc_floor_violation`) are already
written — they call ⑧ / ⑨, so they light up for free.

**Tip:** `u_n, u_e, _ = geometry.leg_unit(A, B)` gives the constant leg unit-vector. Write the
node math with `+ − ×` and `mathx.*` only (see §2b) — then `test_backend.py` confirms it works in
CasADi too. No NumPy broadcasting tricks needed: a column of node values **and** a CasADi symbol
both flow through the same scalar-component code.

## 5. Run it

```bash
# from the repo root (uses the conda `aviation` env)
python -m pytest 4dTrajectory/optimization/approach_constraints/tests/ -q

# the end-to-end demo (feasible vs. deliberately-infeasible trajectory)
PYTHONPATH=4dTrajectory/optimization python -m approach_constraints
```

Before you start: **6 pass, 13 xfail**. As you finish TODOs the xfails become **xpass**; when all
are done, `python -m approach_constraints` prints a feasible report for the on-path trajectory and flags the
final-leg corridor on the off-path one.

## 6. Wiring into the CasADi optimizer

Because the functions are backend-agnostic (§2b), the bridge is direct — **no rewrite, no
`ca.Function` callback, no finite differences**. Pass the optimizer's symbolic state columns where
the tests pass NumPy columns:

```python
# state nodes are decision variables: X has shape (7, K) of SX/MX (one column per node)
n, e = X[N, :], X[E, :]                      # symbolic rows -> scalar components
g_lat = lateral.lpv_corridor_violation(n, e, lpv, k)     # an SX/MX expression
opti.subject_to(g_lat <= 0)                  # IPOPT differentiates it analytically
```

Steps in `casadi_direct_collocation_optimizer.py`:

1. Pre-convert every fix once with `TargetFrame` (lat/lon → `(n, e)`); build a `SegmentSpec` per
   leg (design-doc §7 maps CIFP legs → state nodes by along-track position).
2. Slice each leg's symbolic node columns and call `segment_violations` (or the individual
   functions) — the returned expressions go straight into `g` with upper bound `0`.
3. **Multiple IAFs:** build one `ConstraintSet` per candidate IAF and solve each separately, then
   keep the best objective — do **not** use a non-convex `min` (design-doc §5).

The `*_violation` sign convention (`g ≤ 0`) is already the IPOPT inequality form, so the upper
bound is simply `0` and the lower bound `−inf`.

## 7. References

- `../../docs/optimization_constraint_design.md` — the full constraint design (§2 frame, §3
  segments, §4 lateral, §5 multi-IAF, §6 vertical, §7 discretization).
- `../../docs/lpv_final_segment.en.html` — interactive LPV final-segment explainer (FAS data
  block, angular course, glidepath, OCS, DA).
- FAA Order 8260.58D, Chapter 3 — the primary criteria (FAS data block, LPV/GLS final, DA).
