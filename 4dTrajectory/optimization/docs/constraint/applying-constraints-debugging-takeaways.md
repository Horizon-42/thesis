# Applying procedure path-constraints to the NLP — debugging take-aways

What I learned wiring the `approach_constraints` package into the direct-collocation optimiser
(`casadi_direct_collocation_optimizer.py`) so an approach is enforced as IPOPT path constraints.
Read this before adding or changing any NLP constraint — most of these cost real debugging time.

## TL;DR

1. **Constraints fed to IPOPT must be smooth.** A `|·|` (abs) in a constraint has a kink, and if
   the optimum sits on that kink the gradient-based solver will not converge. Encode `|x| ≤ c` as
   **two linear rows** `x − c ≤ 0` and `−x − c ≤ 0`. This was the single biggest bug.
2. **Constraints only go on the normalized full-transport schemes.** They are written in the
   metric `(n, e)` decision state; on any other scheme the state isn't metric and the rows are
   ill-conditioned. The optimiser *raises* if you try.
3. **Node→segment membership is a fixed partition**, decided up front from the published
   along-track distances — never from the (variable) node positions.

---

## 1. The big one — non-smooth constraints kill IPOPT

The lateral corridor was first written the "obvious" way:

```python
violation = mathx.fabs(cross_track(n, e, A, B)) - k * halfwidth      # |e| - margin ≤ 0
```

This **looks** correct (and its numeric value is right), but the LPV optimum rides the centerline,
where `cross_track ≈ 0` — **exactly the non-differentiable kink of `|·|`**. IPOPT hit
`Maximum_Iterations_Exceeded` (it runs the full 3000 default iters — it's genuine non-convergence,
not a low cap), **even when seeded with a feasible solution** (homotopy didn't help — the proof it
was a smoothness problem, not a seed problem).

The fix is the standard NLP encoding of an absolute value / box — **two smooth linear rows**:

```python
e_xt   = cross_track(n, e, A, B)
margin = k * halfwidth
return e_xt - margin, -e_xt - margin        # both ≤ 0  ⇔  |e_xt| ≤ margin
```

Their `max` is still `|e_xt| − margin`, so feasibility semantics are unchanged, but the feasible
region's boundary is now smooth everywhere. With this, the same problem converges from a **cold**
start and returns a feasible trajectory.

**Rule:** no `abs`, `min`, `max`, or `if/else` whose switch point can coincide with the optimum.
Re-express as multiple smooth inequalities (or a smooth surrogate). The glidepath *window* is
already two-sided (`low`, `high`) for the same reason; the step-down floor is built as **one
constant floor per segment** so its `if_else` never switches inside a segment.

## 2. Constraints belong in the same metric frame as the decision state

The whole reason the constraints are gated to `*NormalizedFullTransport`: in those schemes the
decision state node is `z = (n, e, h, V, psi, gamma)` with `n, e` in **metres from the target
(LTP)** — i.e. exactly the `approach_constraints` package's `(n, e)` frame. So the corridor / glidepath /
floor expressions are written **directly on the decision variables** and stay well-conditioned.

- Convert every **fix** into that same frame once (`TargetFrame` anchored at the target), with the
  same `R = WGS84_A` and target latitude the optimiser's `_normalization_cb` uses. A different
  radius or anchor silently misaligns the fixes from the aircraft state.
- On a plain-radian geodetic scheme the position rows would be ~7 orders smaller than the altitude
  rows — the optimiser would be back to the conditioning problem the normalized scheme was built to
  fix. Hence the hard guard, not a silent fallback.

## 3. Node→segment membership must be a FIXED partition

A constraint is "this node is in the intermediate leg, that one is in the final leg". You **cannot**
decide membership from the node's position — the positions are decision variables, so the
membership (and therefore which corridor applies) would be symbolic and discontinuous.

`partition_node_indices(num_nodes, spans)` assigns node `i` to a segment by its **nominal**
along-track fraction `(i+0.5)/num_nodes`, using the published per-segment distances. It is a
constant computed once at build time. It's approximate (nodes are time-spaced, not distance-spaced)
but constant — which is what the NLP needs.

## 4. Feasibility / tolerance gotchas (these read as "solver is broken")

- **Glidepath window can't be too tight.** A 15 m below-tolerance made otherwise-fine problems
  `Infeasible`. Defaults are now ±60/120 m for the *optimised reference* (tunable). Distinguish the
  reference-path tolerance from a real FTE tracking tolerance.
- **Anchor the glidepath at the target.** The terminal node is pinned to the target, so the
  glidepath must pass through the target altitude at the threshold: set `tdze = target_alt − tch`
  so `glidepath(0) = target_alt`. Otherwise the terminal-equality and the glidepath window fight.
- **The target must be dynamically reachable.** A hand-picked `(pos, V, ψ, γ)` is usually *not* on
  the dynamics manifold → `Infeasible` even with no path constraints. Generate test targets by
  rolling the dynamics forward.
- **Watch for knife-edge caps.** A descent-gradient cap exactly equal to the path's gradient puts
  the whole segment on the constraint boundary. Real intermediate segments are shallow, so the 3°
  cap isn't binding there; a synthetic uniform-3° path made it knife-edge. Don't tune to synthetic
  stress cases.
- **`Maximum_Iterations` vs `Infeasible` mean different things.** Max-iter ⇒ usually conditioning /
  non-smoothness (it *could* converge but can't make progress). Infeasible ⇒ the feasible set is
  genuinely empty (too-tight tolerance, unreachable target, conflicting constraints). Read the
  IPOPT status before changing anything.

## 5. The debugging method that actually localised it

An isolation ladder, cheapest first — each step rules a class of cause in or out:

1. **Does the unconstrained problem converge?** (Yes ⇒ the base setup is fine; the constraints are
   the cause. No ⇒ fix the scenario/target first.)
2. **Add a trivially-satisfied constraint** (e.g. a corridor with half-width `1e9`). Converges ⇒
   the *wiring* (g-append, bounds, shapes) is correct; the problem is constraint *content*.
3. **Check the constraint residuals on a known path** (the rollout). If the path you expect to be
   feasible has large residuals, your geometry/derivation is wrong — not the solver.
4. **Seed with the unconstrained optimum (homotopy).** If it *still* fails from a feasible-ish
   seed, it's not a seed problem — suspect non-smoothness (this pointed straight at the `abs`).
5. **Loosen one constraint at a time** to separate "too tight" (infeasible) from "non-smooth"
   (max-iter that persists even when loose).

## 6. Checklist for adding a new constraint

- [ ] Write it as `g ≤ 0`, **smooth** (no kink at a plausible optimum; split `abs`/`min` into rows).
- [ ] Express it on the **metric `(n, e, h, V, psi, gamma)`** decision state; pre-convert constants
      (fixes) into that frame once.
- [ ] If it's per-segment, decide membership with the **fixed** partition, not node positions.
- [ ] Append to `g` with `lb = -inf, ub = 0` in **both** NLP builders (fixed-time and free-time).
- [ ] Add a wiring test (g grows; rejected on the wrong scheme) **and** one real convergent solve
      that checks the returned path actually satisfies it.
- [ ] Sanity-check tolerances against a reachable target before blaming the solver.

---

*See also:* `4dTrajectory/optimization/approach_constraints/README.md` (the package),
`4dTrajectory/docs/optimization_constraint_design.md` (the design),
`4dTrajectory/docs/lpv_final_segment.en.html` (LPV geometry).
