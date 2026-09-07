# Hard procedure constraints in training — survey and integration plan (2026-09-08)

Companion to `2026-09-04_constraint_methods_survey.zh.md` (the first method survey),
`2026-09-04_procedure_constraints_design.zh.md` (what the constraint is and what the data
supports), `2026-09-05_final_constraint_results.zh.md` (state path), `2026-09-05_control_penalty_results.zh.md`
and `2026-09-06_control_hooks_results.zh.md` (control path), and the pre-registered L1.c arm in
`2026-09-07_latent_intent_design.zh.md` §六. The papers behind §3 are downloaded and annotated
in `docs/literature/procedure_hard_constraints/` (repo root; `README.md` is the index).

The question this document answers: **how do we put the final-approach procedure constraint
into the TRAINING of both prediction paths as a hard constraint, given what has already been
measured here, and what does the literature say about the two problems we hit (a penalty that
trades accuracy, a filter that makes the network lazy)?** §1 states the problem in this repo's
terms, §2 says what "hard" can honestly mean here, §3 is the survey (one family per
subsection, with the core formula written out), §4 is the plan.

## 0. Status (read this first after a compaction)

| item | state | where |
|---|---|---|
| Literature folder (three clusters: constructive maps / projection layers; safety filters + CBF + the lazy-policy problem; constrained training + aviation + two-stage + generative) | **collected 2026-09-08**: 83 entries, 80 PDFs (Shi 2021, Pang 2021 and Gao 2018 paywalled), every formula in the notes read off the PDF; re-fetch with `download.sh` | `docs/literature/procedure_hard_constraints/` (`README.md` index, `notes_{A,B,C}.md`) |
| This survey + plan | **written 2026-09-08**; nothing in §4 is built or trained | this file |
| H0 gate readout (no training) | planned | §4.1 |
| H1 state path: learned commitment gate + bounded output | planned; depends on H0 | §4.2 |
| H2 control path: filter-aware training (score the UNFILTERED rollout, penalise the intervention) | planned; runs after L1.c (`l1c_procedure_arms.json`), which is queued and untouched | §4.3 |
| H3 vertical barrier on the load factor (+ speed hold) | planned; predict-time first | §4.4 |
| H4 Lagrangian redo (feasible ε, PID multiplier) | conditional on L1.c passing its gate | §4.5 |
| H5 solver stage (predictor → casadi tracking OCP) | deployment route, separate project | §4.6 |
| H6 constrained generation (guidance / projection inside a sampler) | with the C line only | §4.7 |

Decision state: **proposal**. No config field, loss term or CLI flag exists yet for H0–H6. Code
lands in a sibling worktree (`../thesis-hc`, branch `dev-hard-constraints`) because the
`a0_random_20260907` campaign runs from the main tree.

## 1. The problem in this repo's terms

**The constraint.** Only the final segment is data-consistent (0 % of observed flights fly an
off-axis IAF; 15–62 % join inside the FAF). In the threshold-anchored `enu` chart, with `ψ`
the runway course and `d` the along-course distance back from the threshold:

```
d      = −(e·cos ψ + n·sin ψ)                 xt = e·sin ψ − n·cos ψ     (+ = right of course)
hw(d)  = cw · (d + d_GARP) / d_GARP           full-scale LPV half-width, ≈107 m at the threshold
GP(d)  = d · tan GPA                          glidepath height in the chart (GPA = 3°)

C(d) = { (xt, u) :  |xt| ≤ k·hw(d),   GP(d) − 60 m ≤ u ≤ GP(d) + 120 m },   k = 0.5
```

Per row the feasible set is a **box in runway axes** `(xt, u − GP(d))`. Observed tracks, once
established, sit inside it 87–99 % of the time; the ±22 m evaluation gate does not (14–69 %),
so C(d) is the constraint and ±22 m stays a metric.

**The gate.** The constraint applies only once the flight is established on final. Training
knows this from the truth (`truth_final_gate`: the tail of rows that stays inside the k-cone to
the threshold, ignoring the last 300 m; per-flight join distance `d_join`, KRDU median
15–18 km, KSJC 30L 24 km, 12R 7 km). Inference does not. The deployable "self gate"
(`final_approach_geometry.soft_on_final`) reads the prediction's own geometry: inside the
full-scale cone floored at 500 m, path within 30° of the course. Its failure mode is
measured: a vectored flight crossing the extended centreline opens it, and a hard filter then
flies the flight down the corridor while the truth still has 30–60 km to go (v1 hard barrier:
49 KRDU flights worse by > 1 km; the FAF-distance gate is worse still, vectored ADE
2599 → 4665 m). The remaining violations after the best arms are mostly gate-closed rows on
vectored flights (KRDU: ~60 % of the residual violation rows).

**The two output paths.**

- `state`: the network emits chart positions per row. A per-row output constraint can be
  imposed BY CONSTRUCTION (that is `corridor-bounded`).
- `control`: the network emits N bounded controls `(thrust fraction, bank, load factor)` held
  for segment durations, plus a duration; the RK4 point-mass rollout with a first-order bank
  actuator (`τ = 2 s`) turns them into the path. The corridor is a **state** constraint
  reached only through the dynamics, so "hard" here means an invariance argument on the
  closed loop, never a clamp on the output.

**What is already measured (validation splits, paired per flight; details in the cited docs).**

| method | path | result | verdict |
|---|---|---|---|
| hinge² penalty, fixed λ = 1e-3 | state | violation 77 → 58 %, ADE flat / +42 m, entry \|xt\| p95 unchanged | vetoed (trades accuracy for violation) |
| hinge² penalty + primal-dual, ε = 0.05 | state | λ ×50 in 74 epochs, ADE 1383 → 2421 m | diverged: ε unreachable (base 0.77) |
| tanh-bounded output in runway axes, self gate | state | violation 77 → 48 % (KRDU) / 34 → 20 % (KSJC), FDE −51…−91 m, entry \|xt\| p95 492 → 305 m, vectored not worse | **adopted** (candidate default) |
| post-hoc projection, self gate | state | most of the FDE gain, not the violation / tail; kinks | deployment fallback |
| hinge² penalty through the rollout, λ = 1e-3 / 5e-3 (teacher 64) | control | FDE −101 m but vectored ADE +581 m; 5e-3 collapses bank skill 0.73 → 0.28 | vetoed; **re-test on the teacherless base pre-registered (L1.c)** |
| barrier filter on the bank at PREDICT time (v2, soft) | control | FDE −201 m / ADE −55 m (KRDU), −66 / −17 m (KSJC), zero flights worse by > 1 km, straight-in entry \|xt\| p95 1821 → 70 m | **adopted** as the deployed safety layer |
| the same filter INSIDE the training loop (6 arms: barrier v1/v2, nominal law v1/v2, 2 airports) | control | every arm worse than its predict-time counterpart; clipped steps 15–24 %, bank skill −0.08…−0.26, vectored mid-path worse | not adopted: **the network learns to rely on the filter** |
| nominal tracking law + bounded residual (archived) | control | best vertical (violation 47 → 29 %), lateral and accuracy worse when trained through | archived; vertical half worth reviving as a predict-time hook |

Two facts drive everything below: **(i) by-construction beat penalties on the state path
without a weight to tune; (ii) on the control path the only thing that worked is a filter the
network never trained with**, and the diagnosed cause is a gradient sink — once the filter
guarantees the corridor, the position loss no longer pushes the network's own bank on gated
segments (`2026-09-06_control_hooks_results.zh.md` conclusion 4).

## 2. What "hard" can honestly mean here

Four levels, from strongest to weakest, and every one is conditional on the gate:

| level | mechanism | guarantee | where it applies |
|---|---|---|---|
| **H-a construction** | the output map's range IS the feasible set | exact, every row, differentiable, no weight | state path (done: `corridor-bounded`); control path only for the envelope box, not for the corridor |
| **H-b invariance through dynamics** | a per-step filter keeps `h(x_k) ≥ 0` for the next step given the model | exact under the model used by the filter (here: lagged point mass, hold Δt, second-order lead approximation) — measured p95 overshoot 65 m, not zero | control path (done at predict time) |
| **H-c post-hoc projection** | clamp the finished prediction | exact, but the path is no longer the model's and has kinks | state path fallback (`--project-final`) |
| **H-d solver** | the prediction seeds / references a constrained OCP whose rows are the optimizer's | exact and flyable; ~4 s per flight, IPOPT in an isolated process | deployment (both paths) |

And the honest statement of any of them: `P(violation | the model says "committed") = 0` by the
mechanism, plus a **measured** gate error (how often, and how early / late, the model commits
relative to the truth). A hard constraint under a wrong gate is the FAF-projection disaster;
so the gate is not a detail of the constraint, it is half of it. The plan therefore treats the
gate as a first-class prediction with its own loss and its own metric (§3.5, §4.1, §4.2).

## 3. Method families

Each subsection: the idea in plain words, the core formula, what it guarantees and costs,
and the verdict for our three sub-problems (state output, control output, gate). Citations
are `[key]` entries of `docs/literature/procedure_hard_constraints/README.md`.

### 3.1 Constructive output maps (the range of the network is the feasible set)

**Idea.** Do not ask the loss to respect the constraint; make it impossible to leave the set.
For a box this is a squashing function per axis; for a general convex set it is a map from a
ball onto the set; for a curve it is a basis whose convex hull property turns "every point
inside" into "every control point inside".

**Core formulas.**

*Box (ours).* With half-width `b(d) = k·hw(d)` and the asymmetric window `[−60, +120]`:

```
xt_out = b · tanh(xt_net / b)
u_out  = GP(d) + r(u_net − GP(d)),   r(s) = −60·tanh(−s/60) if s < 0 else 120·tanh(s/120)
```

slope one at zero (an in-corridor prediction is untouched), saturating at the edge; blended by
the gate weight `w ∈ [0, 1]`: `xt = xt_net + w·(xt_out − xt_net)` (`final_approach_geometry.bound_to_final`).

*HardNet-Aff — input-dependent two-sided affine constraints, closed form* [HardNet]. For
`b_l(x) ≤ A(x) f(x) ≤ b_u(x)` with `A(x)` full row rank, the repair layer is

```
P(f)(x) = f(x) + A(x)⁺ [ relu(b_l(x) − A(x) f(x)) − relu(A(x) f(x) − b_u(x)) ],   A⁺ = Aᵀ(AAᵀ)⁻¹
```

It is the *parallel* projection, not the Euclidean one: every violated row is snapped to its own
bound and every satisfied row is left exactly alone (`a_iᵀ P(f) = a_iᵀ f` when feasible, Prop. 7),
and the paper proves the composed class keeps universal approximation. With `A` a selection
matrix (our runway axes) `A⁺ = Aᵀ` and the layer IS an elementwise clamp in `(xt, u − GP)`. So
HardNet is the closed-form justification of `corridor-bounded`, with one difference worth an
arm: the clamp leaves an in-corridor prediction untouched, the tanh distorts it (slope < 1 away
from zero); the clamp has zero gradient outside, the tanh keeps a small one. HardNet's Remark 4
also gives the smooth way to gate: move `b_l, b_u` continuously to `∓∞` as the gate weight goes
to 0, instead of switching the map on and off.

*Gauge map — a bijection from the box onto a state-dependent polytope* [GaugeMap]. For a
convex set `Q` containing the origin, the gauge function is `γ_Q(v) = inf{λ ≥ 0 : v ∈ λQ}`,
closed form for a polytope `Q = {w : F_iᵀ w ≤ g_i}`: `γ_Q(v) = max_i F_iᵀ v / g_i`. Then

```
G(v | Q) = ( ‖v‖_∞ / γ_Q(v) ) · v ,   v ∈ [−1, 1]ⁿ      is a BIJECTION of the hypercube onto Q
policy:  π(x) = G( ψ_θ(x) | Ω(x) − K x ) + K x        (a known safe feedback Kx supplies the interior point)
```

RAYEN [RAYEN] is the projection-flavoured relative — `y = z₀ + min(1/κ(v̄), ‖v‖)·v̄` walks
from a fixed interior point along the ray and caps at the boundary (inside stays, outside is
pulled to the boundary); Πnet [Pinet] does the true Euclidean projection with a batched
Douglas–Rachford solve and implicit differentiation. The distinction that matters for us:
**a saturating filter or projection is many-to-one** (every command past the edge maps to the
edge, the gradient there dies), **the gauge map is one-to-one** (every latent value keeps a
distinct consequence). That is a mechanistic account of the lazy network measured on the
control path, and the gauge map is its structural cure (§3.3, §4.3).

*Bézier / B-spline corridor* [Bernstein-corridor, EGO-Planner]. A curve
`p(t) = Σ_i B_{i,n}(t)·c_i` with Bernstein weights `B_{i,n} ≥ 0, Σ B = 1` is a convex
combination of its control points, so if every `c_i` is inside a convex cell, `p(t)` is inside
for all `t`. A corridor becomes a per-control-point linear constraint (and then a gauge map
or a QP handles it).

**Guarantee / cost.** Exact, every sample, differentiable, no weight, negligible cost.

**Verdict here.** The per-row feasible set of the state path is a box in runway axes, so the
per-axis tanh (or HardNet's clamp) IS the exact constructive map; a gauge map adds nothing
there unless a coupled row appears (a curvature limit between rows). Two things the cluster
says we should still do: (i) test the clamp against the tanh (HardNet's Prop. 7 — the tanh
distorts in-corridor rows, which is a bias in exactly the rows that matter); (ii) parameterise
the lateral offset by along-track distance `xt(d)` rather than by time step (Werling's Frenét
argument, PBP), which decouples the corridor from the unknown speed profile. On the control
path the output set is the envelope box (already sigmoid-bounded) and the corridor is a state
set reached through the rollout, so a constructive map reaches it only through the barrier's
state-dependent bank interval — that is the gauge-map arm of §4.3. Construction does nothing
about the gate: the map is applied where `w > 0`, and `w` is the problem (§3.5).

### 3.2 Differentiable projection / optimization layers

**Idea.** Put `argmin` inside the network: the layer returns the feasible point nearest the
network's proposal, and gradients flow back through the optimality conditions, so upstream
learns to propose things that project well.

**Core formulas.** OptNet [OptNet] solves per sample

```
y* = argmin_y  ½ yᵀ Q y + qᵀ y    s.t.  A y = b,   G y ≤ h
```

and differentiates through the KKT system: with multipliers `λ` (inequalities) and `ν`
(equalities), the implicit derivative solves one linear system in `(dy, dλ, dν)`,

```
[ Q        Gᵀ            Aᵀ ] [ dy ]     [ −∂ℓ/∂y ]
[ diag(λ) G   diag(G y − h)   0 ] [ dλ ] =  [    0    ]
[ A          0              0 ] [ dν ]     [    0    ]
```

which gives `∂ℓ/∂Q, ∂ℓ/∂q, …` in closed form. cvxpylayers [cvxpylayers] generalises this to
any disciplined convex program (differentiating the cone program's solution map). DC3 [DC3]
avoids the solve for large problems: the network emits a partial vector `z_p`, a *completion*
`z_c = φ(z_p)` solves the equalities exactly, then a few *correction* steps
`z ← z − η ∇_z Σ_j relu(g_j(z))²` push toward the inequalities, and training back-propagates
through completion and correction. KKT-hPINN [KKT-hPINN] is the equality-only closed form
for PINNs.

**Guarantee / cost.** OptNet / cvxpylayers: exact feasibility of `y*`, one QP (or cone solve)
per sample per step — for 300 rows × batch this needs a throughput test before use. DC3:
equalities exact, inequalities approximately satisfied (correction steps), cheap.

**Verdict here.** For a per-row box the projection is the clamp, and the tanh of §3.1 is a
better-conditioned version of it (no dead zone). A QP layer earns its cost only when
constraints couple rows (a turn-rate or acceleration limit across consecutive state rows,
which the state path currently violates with its node-scale saw-tooth). On the control path
the rollout already IS the coupled constraint, and the per-segment CBF-QP of §3.3 is the
projection layer that fits it.

### 3.3 Safety filters in the training loop (control path), and the lazy-policy problem

**Idea.** The policy proposes a command, a thin layer minimally changes it so that a barrier
function stays non-negative one step ahead. The layer is a small QP with a closed-form
solution for one constraint. Our `BarrierFilter` is exactly this on the bank command.

**Core formulas.** Control barrier function [CBF-theory]: a set `S = {x : h(x) ≥ 0}` is forward
invariant under any controller satisfying

```
ḣ(x, u) + α(h(x)) ≥ 0,   α a class-K function (often α(h) = α·h)
```

and the CBF-QP safety filter is

```
u* = argmin_u ‖u − u_net‖²   s.t.  ∇h(x)ᵀ f(x, u) + α h(x) ≥ 0,   u ∈ U
```

Discrete time [DCBF] (what a held command needs): `h(x_{k+1}) − h(x_k) ≥ −γ h(x_k)`, `0 < γ ≤ 1`,
which gives `h(x_k) ≥ (1 − γ)^k h(x_0) ≥ 0`; `γ Δt ≤ 1` is why every gain in
`barrier_filter.py` is `min(gain, 1/Δt)`. Relative degree two (position constrained through
heading through bank; height through path angle through load factor) needs either a
two-layer condition (ours: position layer → heading interval → bank interval) or an
exponential CBF `ḧ + k₁ ḣ + k₂ h ≥ 0` [ECBF]. Single-constraint closed form [Dalal2018]:
`u* = u_net − max(0, (g(x)ᵀ u_net + c(x)) / ‖g(x)‖²) · g(x)` for a linearised constraint
`g(x)ᵀ u + c(x) ≤ 0`.

BarrierNet [BarrierNet] makes the QP a differentiable layer AND lets the network output the
class-K parameters: the constraint becomes `∇h f + p₁(z)·h ≥ 0` (relative degree 2:
`ḧ + p₁(z) ḣ + p₂(z) h ≥ 0`) with `p₁, p₂ > 0` emitted by the network from the observation `z`,
so how conservatively the filter acts is learned per situation while the invariance argument
still holds for any positive `p`. Gradients go through the QP by OptNet's KKT differentiation.

**The lazy-policy problem: known, named, measured, and curable.** The literature does not say
"training through a filter is harmful"; it says training through a filter **with naive loss
bookkeeping** is, and it measures the dependence and its cure.

- *Measured directly* [PizarroBejarano2025] (model-predictive safety filter around an RL agent,
  quadrotor sim + hardware): the trained policy's return WITHOUT the filter is the laziness
  metric ("return when uncertified"). With a correction penalty

  ```
  R_α(x, u_uncert, u_cert) = R(x, u_applied) − α · ‖u_uncert − u_cert‖²
  ```

  the uncertified return goes 11 → 31 → 211 → 202 for α = 0.1, 1, 10, 100 while the certified
  return stays 213–214 (Table I). Dependence is a monotone function of the penalty weight and
  is removed by tuning it; too large a weight starts to cost the certified return. Their third
  modification, *safe reset* (reject start states from which the filter is infeasible),
  "significantly improves convergence".
- *The loss trick* [OptLayer]: four strategies compared. `CC` (execute the projected action,
  learn on it) is what our six arms did and is the weakest; the winner `CPC` updates the network
  TWICE per batch — first on the RAW action with a violation-penalised reward
  `r̃ = r − c`, `c = ‖A ũ − b‖ + ‖relu(G ũ − h)‖` (rows normalised), then on the PROJECTED
  action with the true reward. The raw output is charged for needing the projection and
  credited for what the projection achieved.
- *Benchmark* [Krasowski2023]: our hook is "action projection"; the projection + intervention
  ("adaption") penalty tuple gives both the highest return and the LOWEST intervention rate,
  and the survey notes projection filters misbehave when the projection is infeasible.
- *Distillation* [Cheng2019]: the RL policy absorbs the CBF compensator by regressing an MLP
  onto the previous iterations' corrections, `ū_k(s) ≈ Σ_{j<k} u^CBF_j(s)`, the QP staying on top.
- *Learned conservativeness* [BarrierNet]: `α` as a network output makes the filter part of the
  policy; safety holds for any `p > 0`.

Three results reframe our six arms rather than fix them:

1. **Permissiveness is the cost driver** [OhFisac2026, Hsu2023]. If the filter is
   least-restrictive (intervenes iff the action would leave the maximal safe set) and the SAME
   filter is deployed, an ε-optimal policy of the filtered MDP is ε-optimal among all safe
   policies — no asymptotic performance penalty for training through it. A CBF corridor with
   hand-picked `(k, α)` is strictly inside the true viability kernel, so "accuracy dropped"
   reads as "the filter is too restrictive": sweep `α` and `k` before adding machinery.
2. **Infeasible gate entry is the theorem's excluded case** [OhFisac2026 Assumption 1.2,
   PizarroBejarano2025 safe reset, Krasowski2023]. A vectored flight enters the gate already
   outside the corridor (`h < 0`); the filter then demands a return at rate `α|h|`, the
   projection is at or beyond the envelope corner, and every guarantee above is void there.
   That is the most plausible mechanism for "the vectored flights got worse", and the cheapest
   fix is to open the gate only from a feasible state.
3. **In an imitation setting, training through the filter is provably the better option**
   [GeigerStraehle2022]: with the safety layer at train AND test the imitation error is linear
   in the horizon, `|v^I − v^D| ≤ 2εT‖c‖_∞`; with the layer at test time only it is quadratic,
   `|v^O − v^D| ≤ (4ε/ν)T²‖c‖_∞` (and a matching lower bound), because the layer drives the
   policy into states it never saw. With 64 segments this is the argument AGAINST settling for
   the predict-time-only filter — the current adopted form is the quadratic case. Their caveat:
   a discontinuous layer complicates training (our soft form exists for that reason).

**Guarantee / cost.** Invariance under the filter's model (ours: lagged point mass with a
second-order lead approximation; measured p95 overshoot 65 m at predict time, i.e. near-hard,
not exact). Cost: one closed-form per segment; ≈ 2× if the unhooked schedule is rolled out
beside the hooked one (the `needs_reference` machinery already does that).

**Verdict here.** The control path's "hard" is H-b. What has NOT been tried is the filter in
the loop with the bookkeeping the literature uses: a correction penalty swept in dose, the
raw rollout scored beside the filtered one (CPC), a gate that opens only from a feasible state,
a permissiveness sweep first — and, structurally, the gauge-map bijection of §3.1 in place of
the saturation. Those are the arms of §4.3. `α` as a head output [BarrierNet] is the follow-up.

### 3.4 Constrained training: Lagrangian, PID multiplier, feasible ε, log barrier

**Idea.** Write training as `min_θ L(θ) s.t. C(θ) ≤ ε` and let a multiplier find the weight.

**Core formulas.** Primal-dual [Chamon2020, Fioretto2020]:

```
θ ← θ − η_θ ∇_θ [ L(θ) + λ (C(θ) − ε) ]
λ ← max(0, λ + η_λ (C(θ) − ε))            (dual ascent, once per epoch here)
```

**Why ours diverged, in print** [GallegoPosada2022 App. G, Chamon2023]: the dual player's best
response to a fixed primal is a linear program — `λ = 0` if the constraint is satisfied, any
value at equality, **`λ = +∞` if it is violated** — and the dual optimum is `+∞` whenever the
empirical problem is infeasible. With ε = 0.05 against a base rate of 0.77 the constraint was
violated at every step, so ascent was correctly walking toward an unbounded dual. Chamon's
PACC guarantees assume strict feasibility with margin. Nothing about the optimizer needed
fixing; the level did. Three ways out, cheapest first:

1. *Anneal the level* — the constraint enters at full strength from step zero in our run;
   SafeDiffuser [SafeDiffuser] and the log-barrier schedule [Kervadec2022] tighten on a
   schedule (`γ(t)` from a loose value to the target). The same "start soft, tighten"
   principle appears in annealed winner-takes-all [AnnealedWTA] (§3.5).
2. *Learn the level* — resilient constrained learning [Hounie2023] adds a relaxation `u ≥ 0`
   per constraint with a strictly convex cost `h(u)` and updates three variables:

   ```
   θ ← θ − η ∇_θ [ L_0 + Σ_i λ_i L_i ]
   u ← [ u − η_u ( ∇h(u) − λ ) ]_+              relax exactly where the dual price exceeds the relaxation cost
   λ_i ← [ λ_i + η_λ ( C_i(θ) − u_i ) ]_+       dual against the LEARNED level
   ```

   with `h(u) = α‖u‖²` (a linear `h` is plain fixed-level learning). A runaway λ is absorbed
   into a growing `u`, and the converged `u*` IS the measured reachable violation rate.
3. *Drop the level* — the solver stage (§3.6), where feasibility is not a training target.

If a fixed level is kept, do not use plain ascent. PID Lagrangian [Stooke2020] (integral
control is what ascent is; add P and D):

```
Δ = C − ε;   I ← (I + Δ)_+;   ∂ ← (C − C_prev)_+;   λ ← (K_P Δ + K_I I + K_D ∂)_+
```

and νPI [nuPI] (ships in the Cooper library): `ξ_t = ν ξ_{t−1} + (1 − ν) e_t`,
`λ_{t+1} = λ_0 + κ_p ξ_t + κ_i Σ_{τ≤t} e_τ`, which cuts the multiplier when the violation is
improving even while still infeasible (plain ascent keeps raising it) — the damping our run
lacked. Dual restarts [GallegoPosada2022] zero λ the moment the constraint is satisfied, so
compliant flights stop being over-regularised. Stooke's `1/(1 + λ)` rescaling of the primal
objective keeps the effective learning rate honest under a large multiplier.

**Guarantee / cost.** Soft: the level is set (or learned) and λ is found, not swept. Cost: a
few scalars. Prerequisite for a fixed ε: reachable, i.e. set relative to the base's OWN rate
(control base 0.55 on the truth gate; observed floor 0).

**Verdict here.** Only as the comparison arm of the pre-registered L1.c (fixed λ): if L1.c
passes, the resilient / νPI version is how its dose stops being hand-picked and how the
reachable violation rate gets measured (§4.5). It cannot make a constraint hard, and on the
control path the penalty's mid-path cost (the hinge's gradient flows back into every earlier
segment) is a property of the rollout, not of the multiplier.

### 3.5 A discrete decision plus a constrained regime (the gate)

**Idea.** Motion forecasting learned long ago that a multimodal "which goal / which mode"
decision should be a classification trained jointly with the continuous regression, not a
threshold on the regression's own geometry. TNT [TNT] factorises

```
p(traj | x) = Σ_τ p(τ | x) · p(traj | τ, x)
L = CE(target classification) + L_reg(trajectory of the MATCHED target)   (winner-takes-all)
```

TNT samples its candidate targets from the map, labels the candidate closest to the truth,
teacher-forces the trajectory decoder on the truth target in training, and scores candidates at
inference; DenseTNT [DenseTNT] manufactures the unobservable label offline by optimisation and
supervises against it. Winner-takes-all collapses modes when hypotheses are few; annealed WTA
[AnnealedWTA] replaces the argmin by a softmin at temperature `T(t) = T_0 ρ^t`,
`q_k ∝ exp(−ℓ_k / T(t))`, `L = Σ_k q_k ℓ_k` (stop-gradient on `q`), starting as a mean and ending
as exact WTA.

Our gate is the same object with one binary mode that is **monotone in time**: "not yet
established" → "established", never back. Two parameterisations:

```
join-row distribution:   p(J = j | x) = softmax_j(ℓ_j),   g_k = P(J ≤ k) = Σ_{j≤k} p(J = j)
hazard form:             g_k = 1 − Π_{j≤k} (1 − σ(ℓ_j))                                   (monotone by construction)
loss:                    BCE(g_k, truth gate_k) over rows, or CE on the truth join row
```

At inference the hard constraint applies to rows `k ≥ Ĵ` (or `g_k ≥ ½`); at training the
bound is blended by `g_k` (soft, differentiable) with the truth gate as an auxiliary target
(teacher forcing on the gate, like `truth-…` intent conditioning but as a LABEL, not an input).
The threshold on `g` is then a calibratable number: choose it on the validation split so that
the false-open rate on vectored flights is ≤ α (a Neyman–Pearson / split-conformal choice,
the machinery of `calibration.py`), which turns "hard after commitment" into "hard after a
commitment with a certified false-open rate".

Precedents for supervising the gate rather than thresholding it: the map-adaptive goal-based
predictor [MapAdaptiveGoal] builds its mode-classifier TARGETS from a cross-track test ("the
future stays within a cross-track band of this reference path") — our truth gate is that test
with the corridor half-width as the band; PBP [PBP] classifies over candidate reference paths
before decoding in the path's Frenét frame, which is the template for a *procedure* classifier
(which approach / runway, hence which corridor) if the gate is ever more than binary. The
aviation default is the opposite: a hard phase indicator that silently disables the constraint
when its inputs are missing [ZhangChen2022]. Smooth substitutes for a hard state gate: HardNet's
bounds moved continuously to `∓∞` (Remark 4), SafeDiffuser's constraint margin annealed in along
the horizon, Fisac's model-confidence gate.

**Verdict here.** This is the missing half of both paths. The self gate is a heuristic on the
prediction's geometry and its false-opens (centreline crossings) are the whole vectored-flight
cost of a hard filter; a learned monotone gate reads the history (heading trend, height vs
glidepath, distance, speed), is trained on the truth gate's rows, and is scored on its own. On
the control path it must also refuse to open from an infeasible state (§3.3, point 2).

### 3.6 Two-stage: predict decisions, let a solver enforce (amortized optimization)

**Idea.** The network outputs what the solver cannot know (duration, join distance, a
reference path); a constrained OCP produces the final trajectory with the optimizer's own
rows [Amos2023, Sambharya2023, DiffMPC].

```
min_{x, u, T}  ∫ ‖x − x_ref(t)‖²_W + ‖u‖²_R dt   s.t. dynamics, envelope, corridor rows, glidepath rows, terminal pin
```

with `x_ref` the prediction (and its duration as the fixed `T`), warm-started from it.

Two ways to train the predictor for the solver rather than for the track: score it by the
solver's fixed-point residual after a fixed number of iterations from the predicted warm start
[Sambharya2023] (no offline solution library needed), or by the decision loss of the solved
problem [SPO]. Amos's taxonomy [Amos2023] classifies our current setup as fully-amortised,
regression-based — "distillation" — which is why the constraint has to be bolted on
afterwards. Constraint-informed warm starts for trajectory optimisation [Briden2023,
Guffanti2024] predict which constraints will be active (the join point is such a prediction)
and the solver does the rest.

**Guarantee / cost.** Exact and flyable (H-d); ~4 s per solve (56 s on a failure), IPOPT in an
isolated subprocess (casadi is not thread-safe); needs a tracking objective the collocation
optimizer does not have yet. Differentiable MPC at this scale is not worth it.

**Verdict here.** The deployment-grade "hard" and the natural place of the optimizer in the
scheduler; not a training-time method (§4.6).

### 3.7 Constrained generation (when the output becomes a sampler)

**Idea.** In a diffusion / iterative sampler the constraint enters at every denoising step:
as guidance (a gradient of the violation, soft), as a projection (`x ← Π_C(x)` after each
step, hard) [ProjectedDiffusion], or as a CBF-QP inside the step [SafeDiffuser]:

```
guidance [Diffuser, MotionDiffuser]:  x_{t−1} = μ_θ(x_t, t) − s · ∇_x Σ relu(g(x))² + σ_t z
projection [ProjectedDiffusion]:      x_t^{i+1} = P_C( x_t^i + γ_t ∇ log q(x_t^i) + √(2γ_t) ε ),   P_C(x) = argmin_{y∈C} ‖y − x‖²
SafeDiffuser:                         a CBF-QP inside each denoising step, with the constraint
                                      margin γ(j) annealed from loose at step N to 0 at step 0
```

Projected diffusion is hard on `C` at every iterate (convergence proved for convex `C`); our
`P_C` is a closed-form clip in runway axes, so it is within reach — but it must be time-masked
to the post-join rows, which is the gate again.

**Verdict here.** Relevant the day the C line (a conditional diffusion over the 96 operating
numbers, guided THROUGH the rollout) exists: the corridor hinge through the rollout is the
guidance, the barrier filter applied inside the sampler is the projection. Not before.

### 3.8 Aviation instances

Twelve aviation trajectory-prediction papers were read (ten in full; Shi, Xu & Pan 2021 and
Pang et al. 2021 are paywalled with no preprint and are quoted from abstracts only). The field
splits into two families: *physics-as-generator* — the network emits BADA or neural-ODE
parameters that an integrator turns into the path [Hodgkin2026, PepperThomas2023, NODE-FDM],
hard on energy feasibility but en-route / vertical only, with Pepper restricting to
FL150–FL325 "where trajectories are less likely to be affected by local operational
procedures" — and *learned-sequence* in terminal airspace [ASCENT, Patrikar2022, MAIFormer,
XiangChen2024, ZhangChen2022], which enforce nothing (Zhang & Chen rejection-sample against
dataset-maximum turn rates at inference, with a hard phase gate the authors admit is disabled
when heading is missing). The only published-regulation hard constraint found is Hodgkin's
post-hoc rejection of samples under the UK 500 ft/min climb minimum — a scalar bound. **None
enforces a lateral corridor or a glidepath window, and none predicts the establishment
point.** The nearest neighbour of the gate problem observes the ATC decision from spoken
instructions instead of inferring it [Guo2023] — and even with the instruction handed to it,
feature conditioning leaves a 300 m altitude undershoot, which is the argument for inferring
the decision and then ENFORCING its consequence geometrically. Yoon & Lee (T-ITS 2025) list
"flight procedures and airspace constraints" as future work. That gap is what §4 builds, and a
positioning claim the thesis can make on the record.

## 4. Integration plan (prioritised; each item has a gate before it is built or run)

Conventions: KRDU first, one seed, validation split, paired to the base; readouts are the
existing `compare_constraint_arms.py` (violation rates on truth-established rows, entry
|xt| p95, stratified ADE/FDE) and `score_control_arms.py` (bank skill, shared profile,
clipped-step share); seed noise for a state arm is 5–22 m pooled ADE. Nothing below touches
`tracks/`, the arrival rosters, the test split or a running campaign.

### 4.1 H0 — the gate readout (no training, one day)

Measure, on existing prediction directories (`B_corridor_bounded_pred_val`, `A_control_v3`
+ barrier, the L1.b base), the self gate against the truth gate per flight:

```
open_pred  = first row with membership ≥ ½ that stays ≥ ½ to the end (predicted commitment)
open_truth = first truth-gate row (d_join)
timing error  Δd = d(open_pred) − d_join        (p10 / p50 / p90, per stratum)
false-open    = open_pred exists and the truth never establishes before the horizon, or Δd > +5 km
missed        = open_truth exists and open_pred does not
residual violation split: rows violating with gate open vs gate closed
```

Decision rule: if false-opens on vectored flights are below 3 % and Δd p90 within 3 km, the
gate is not the bottleneck and H1/H2 run with the self gate made monotone (cumulative max);
otherwise H1's learned gate goes first. (The 09-06 bins — flights hurt when the gate opened at
d < 8 km or ≥ 16 km — say the second branch is likely.)

### 4.2 H1 — state path: learned commitment gate + bounded output

- A `gate` head on the shared backbone features emits row logits `ℓ_k`; the gate is the
  hazard form `g_k = 1 − Π_{j≤k}(1 − σ(ℓ_j))` (monotone). The bounded output blends with
  `w_k = g_k · m_k` where `m_k` is the geometric membership (both must agree: the network says
  "committed" AND the row is physically in the cone) — a false commitment far from the
  centreline then has no effect, which is the safety of the current design kept.
- Loss component `gate`: BCE against the truth gate rows (registered in
  `objective.loss_component_names`); weight calibrated like every other term (parity with the
  position term at the converged base), and the objective otherwise unchanged.
- Inference: `w_k = 1[g_k ≥ θ_g]·m_k`, hard clamp (H-c inside the model), `θ_g` chosen on
  the validation split for a false-open rate ≤ 5 % on vectored flights and written to the
  checkpoint metadata like a conformal table.
- Config: `corridor_gate="committed"` (a third value beside `on-final` / `faf`),
  `gate_loss_weight`, CLI flags named after the fields, `run_naming` meta, `TSConfig`
  refusals (a `committed` gate without the loss weight is refused).
- The gate weight is annealed in: the bound is applied with `w_k = g_k · m_k` from the start
  but the gate loss weight ramps over the first 20 epochs (the "start soft, tighten"
  principle of §3.4 / §3.5); the geometric membership alone (today's B arm) is the ablation.
- Two cheap side arms on the same base: **clamp vs tanh** (HardNet's parallel projection —
  an in-corridor row untouched — against the tanh that distorts it), and the lateral offset
  regressed as a function of along-track distance `xt(d)` rather than of the time index.
- Gates: violation on truth-established rows below B's 48 % / 20 % by ≥ 10 points; entry
  |xt| p95 ≤ B; vectored ADE not worse than B by more than seed noise on two seeds; gate
  timing p50 within 2 km; false-open rate on vectored flights ≤ 5 % at the calibrated
  threshold. Veto: vectored ADE worse on both seeds.
- Cost: 2 airports × 2 seeds ≈ 4 state trainings (≈ 1 h each), plus the two side arms.

### 4.3 H2 — control path: the filter in the loop, with the literature's bookkeeping

Notation: `u` the network's commands `[N×3]`, `R(u)` the unhooked rollout, `R_F(u)` the rollout
through the soft barrier, `u*_k = F(x*_k, u_k)` the command actually flown at segment `k`,
`g_k` the gate weight there, `‖·‖_box` each channel in half-envelope units (the imitation
term's unit). Every trained arm is predicted TWICE — with and without the hook — and the pair
is reported (the "return / return when uncertified" reading): the without-hook number is the
dependence metric, beside the clipped-step share and bank skill already used as the lazy veto.

**H2-0 — permissiveness and feasible entry, predict-time only (no training, hours).**
On the L1.c base checkpoint, sweep the filter's `α ∈ {0.1, 0.2, 0.5}` and `k ∈ {0.5, 0.75, 1.0}`
(nine predict-only runs) and read violation / FDE / clipped share per cell; and count, per
flight, whether the gate opens from an infeasible state (`h_R < 0` or `h_L < 0` at the first
gated segment) and how the FDE change splits between feasible and infeasible entries. Decides
the `(α, k)` and the entry rule the trained arms use.

**H2-a — correction penalty (Pizarro Bejarano form), three doses.**

```
L = L_traj(R_F(u))  +  λ_int · (1/N) Σ_k g_k ‖u*_k − u_k‖²_box        gradient of the 2nd term into the raw head
```

`λ_int` at the control-path parity dose and 10×, 100× (the dependence is monotone in the
dose and the sweep is the experiment, not a detail). Gate: **committed** (cumulative max of the
soft membership over segments) and **feasible-entry** (a segment may open the gate only if
both barrier margins are ≥ 0 at its lead point; an infeasible row keeps the gate closed).

**H2-b — CPC form (OptLayer): the raw rollout scored beside the filtered one.**

```
L = L_traj(R(u)) + L_traj(R_F(u)) + λ_int · (1/N) Σ_k g_k ‖u*_k − u_k‖²_box
```

Uses the second rollout the `needs_reference` path already integrates (≈ 2× per epoch). One
dose (H2-a's best).

**H2-c — bijective interval map (gauge form) instead of saturation.** The barrier already
computes the safe bank interval `[μ_min(x), μ_max(x)]`; instead of saturating the head's bank
into it (many-to-one), the head's pre-squash logit is mapped onto it one-to-one, blended by
the gate:

```
lo = g·μ_min + (1 − g)·(−μ_max_env),   hi = g·μ_max + (1 − g)·μ_max_env
μ  = ½(lo + hi) + ½(hi − lo) · tanh(z_bank)               (1-D gauge map; identity to the envelope box when g = 0)
n' = n · cos μ_head / cos μ                                (load coordination, unchanged)
```

No intervention term is needed in principle (there is no intervention); the imitation and
heading-rate terms are applied to the mapped bank. One arm.

- Base: the L1.c base (teacherless `hr8+TV`, N = 32) so every reading is against
  `L1c_base_barrier_infer` on ONE base; queue position: after L1.c.
- Gates (paired to `L1c_base_barrier_infer`, both predicted with the hook): violation on
  truth-established rows ≤ the hook arm's; straight-in FDE ≤ the hook arm's; vectored ADE ≤
  hook arm + 30 m; bank skill ≥ 0.70; clipped-step share < 20 % throughout training.
  Dependence: the without-hook prediction must not be worse than the base's own without-hook
  numbers by more than seed noise (the "return when uncertified" clause). Veto: the six 09-06
  signatures (clipped ≥ 20 % with skill below base − 0.02; vectored ADE > +100 m).
- Code: `control_filter_correction_weight` (λ_int), `control_raw_rollout_weight` (H2-b's
  first term, 0 or 1), `control_gate_commitment` ∈ {`instant`, `committed`,
  `committed-feasible`}, `control_hook_form` ∈ {`saturate`, `gauge`} on `TSConfig`; registered
  under `true-time-position` only (where the imitation and heading-rate terms live), refused off
  it and refused with `control_command_hook="off"`; components `filter_correction` and
  `raw_state` in `loss_component_names`; run-name items for each. The retired name
  `control_hook_gate` is dropped silently by `from_dict` and must not be reused.
- Follow-up if an H2 arm passes: `α` as a head output clamped to `(0, 1/Δt]` [BarrierNet].
- Cost: H2-0 nine predictions; H2-a three trainings (≈ 65 min each); H2-b one training at
  ≈ 2×; H2-c one training; every arm two predictions.

### 4.4 H3 — the vertical half (load factor barrier + speed hold), predict time first

`h_lo = u − (GP(d) − 60) ≥ 0`, `h_hi = (GP(d) + 120) − u ≥ 0`; with `u̇ = V sin γ` and
`γ̇ = g (n cos μ − cos γ) / V` the constraint is relative degree two in the load factor, so
the same two-layer scheme as the lateral filter: a height barrier gives an allowed
path-angle interval, a path-angle barrier over the hold gives a load-factor interval; the
thrust holds the unhooked rollout's speed (`T' = T + k m (V_ref − V)`, the archived nominal
law's third rule, `k ≤ 1/Δt`). Predict-time arm on the L1.c base first (the open item
"combined hook"); if it is a net gain like the lateral one, it enters H2's filter.

### 4.5 H4 — the Lagrangian redo (only if L1.c passes)

Same hinge, two arms. **Resilient** [Hounie2023]: the relaxation `u` learned with
`h(u) = α u²`, `α` at two values, νPI on the multiplier, λ annealed from 0 over 20 epochs; the
converged `u*` is reported as the measured reachable violation rate. **Fixed level**:
`ε_lat = base rate − 0.05` measured on the base's first epoch, νPI multiplier
(`κ_p, κ_i, ν` from the Cooper defaults), dual restarts on, λ capped at 10× the parity dose,
the same ramp. Both with `lr_plateau_metric=selection` (the objective refuses the plateau
scheduler on a moving objective). Reads whether the dose can be found instead of picked, and
what the fleet can actually reach.

### 4.6 H5 — solver stage (deployment)

A tracking objective in `collocation/optimizer.py` (reference = the prediction on the
solver's own node times, `dense_node_times`; fixed `T` = predicted duration), the existing
corridor / glidepath / alignment rows, batch driver in the `run_scenario_optimization.py`
style, isolated subprocess. Output: hard, flyable, and the natural scheduler interface. A
separate project; listed so the survey's H-d level has an owner.

### 4.7 H6 — constrained generation

Only with the C line: the corridor hinge through the rollout as guidance and the barrier
filter inside the sampler as projection. No work now.

### Order and dependencies

```
H0 (1 day, no GPU) ──► H1 (state, 4 trainings)        after the a0_random campaign, in the worktree
                   └─► H2-lite / H2-full (control)     after L1.c, on its base
H3 predict-time arm   (no training)                    any time the GPU is free
H4                    only if L1.c passes its gate
H5, H6                separate projects
```

## 5. What not to do (each already measured or ruled out)

- Any hook inside the training loop WITHOUT a correction charge or the raw rollout scored
  beside it (six arms, none beat its predict-time counterpart; the literature calls that the
  `CC` strategy and measures it as the weakest).
- Dropping the filter at deployment after training with it (every paper that measures it
  keeps the filter; the without-hook prediction is a dependence METRIC, not a delivery form).
- A hard gate, or a gate that opens from an infeasible state (hard saturation is fine; the
  gate must stay soft in training, feasible at entry, and calibrated at inference).
- The penalty as the mechanism of hardness, a dual step with an absolute ε, or any constraint
  level applied at full strength from epoch 0.
- Constraints the data does not fly: IAF legs, the pre-FAF join window, fix discs, ±22 m as
  a window.
- Rebuilding the closure tracker (retired 2026-09-07) as "constraint by definition".
- Procedure geometry as a model INPUT (H3 of the frame ablation: the backbone ignores it).

## 6. Files

- Reading list, notes with per-paper formulas, `download.sh`:
  `docs/literature/procedure_hard_constraints/` (repo root).
- Geometry and gates every method reads: `final_approach_geometry.py`,
  `control/constraints/{gates,barrier_filter}.py`, `control/dynamics/hooks.py`.
- Existing terms this plan extends: `objective.procedure_loss`, `ProcedureMultipliers`,
  `prediction_outputs.StateOutputLayer._bound_to_final`, `config.PROCEDURE_LOSS_FIELDS`,
  `config.CONTROL_HOOK_FIELDS`.
- Pre-registered neighbour: `docs/experiments/l1c_procedure_arms.json` (untouched).
