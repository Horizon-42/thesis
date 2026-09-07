# Cluster A — Hard constraints by construction, and differentiable projection layers

Reading notes for the procedure-constraint question: how to make a learned arrival predictor
satisfy the final-approach corridor (lateral `|xt| <= k*hw(d)`, glidepath `u in [d*tan3 - 60,
d*tan3 + 120]`) **by construction** rather than by penalty, on either the `state` path (ENU
positions per step) or the `control` path (N=64 bounded controls + duration through a
differentiable RK4 point-mass rollout).

PDFs live in `papers/` (gitignored); `download_A.sh` re-fetches all of them.
Formulas are transcribed from the PDFs, in the paper's own notation, in plain-text math.
Anything **not** verified from the paper itself is marked `[UNVERIFIED]`.

Two facts about our setting recur below and are worth stating once, because they decide which
of these mechanisms is even applicable:

- **The corridor is a box in runway axes, per timestep, and is affine in the state.** After the
  runway-frame transform, `hw(d)` and `d*tan(3deg)` are affine (or piecewise-affine) in
  along-track distance `d`, so both clauses are two-sided affine constraints
  `bl(x) <= A(x) y <= bu(x)` with `A` diagonal-ish and `nc <= nout`. That puts our problem in the
  *easiest* class in this literature — the one with closed-form projections (HardNet-Aff,
  KKT-hPINN, RAYEN-linear) rather than the one needing solver-in-the-loop.
- **The gate ("established on final") is the hard part, not the geometry.** Every method here
  assumes the feasible set `C(x)` is *known at inference from the input `x`*. Ours is known only
  up to a latent binary/soft indicator. Which is why the recurring practical answer in this
  cluster is *not* "project" but "reparameterise": Frenet / runway-relative coordinates make the
  constraint a coordinate bound that can be enforced with a squashing function whose tightness is
  a smooth function of `d`, so a gate error degrades gracefully instead of flipping a projection
  on and off. Notes flag this per paper.

---

## A1. Hard-constraint output layers (closed form; no solve in the loop)

### 1. RAYEN — Imposition of Hard Convex Constraints on Neural Networks

- Tordesillas, J., Klemm, V., How, J. P., Hutter, M. arXiv:2307.08336 (v1 2023; v2 2026,
  journal-formatted, SAGE/IJRR style). File: `papers/rayen_2307.08336.pdf`.
- **NOTE on attribution**: the brief said "Tordesillas, How, Carlone". Carlone is *not* an
  author; the fourth author is Marco Hutter (v2 adds Victor Klemm).

**Mechanism.** RAYEN turns constraint satisfaction into a *ray-shooting* problem. Offline it
computes the affine hull of the feasible set `Y`, reduces to a lower-dimensional coordinate `z`
in that hull, and finds one strictly interior point `z0` by a small convex program. Online the
network emits an unconstrained direction/magnitude vector `v`; RAYEN walks from `z0` along the
unit direction `v̄ = v/||v||` by a step that is capped at the distance to the boundary. Because
the cap is computed in closed form (a `max` of per-constraint inverse distances), the output is
feasible for *any* weights and *any* input, at train and test time, with no projection and no
iteration. It supports linear, convex-quadratic, SOC and LMI constraints.

**Core formulas.**

```
affine-hull lift (Eq. 7):     y = f(z) := N z + A_E^+ b_E
        N: orthonormal basis of null(A_E); A_E^+ = pseudoinverse; aff(Y) = {y : A_E y = b_E}

inverse distance to boundary: kappa := inf{ lambda^-1 : z0 + lambda*v̄ in Z, lambda > 0 }

THE MAP (Eq. 10):             z1 := z0 + min( 1/kappa , ||v|| ) * v̄   in Z
                              y  := f(z1)                              in Y
                              (if v = 0, take z1 := z0)

linear part, closed form:     kappa_L = relu( max( A_p (/) ((b_p - A_p z0) 1_{1xn}) v̄ ) )
        with A_p z <= b_p the reduced polytope, (/) elementwise division;
        D := A_p (/) ((b_p - A_p z0) 1) is precomputable offline.

per-face form:                kappa_{L,j} = relu( A_p[j,:] v̄ / (b_p[j] - A_p[j,:] z0) )

quadratic:  kappa_{Q,i} = nonneg root of  g_i( f(z0 + (1/kappa)*v̄) ) = 0   (a scalar quadratic)
SOC:        kappa_{S,j} similarly from  h_j( f(z0 + (1/kappa)*v̄) ) = 0
LMI:        kappa_M = relu( max eig( L^T (-S) L ) ),  H^-1 = L L^T (Cholesky)
overall:    kappa = max( kappa_L, kappa_Q, kappa_S, kappa_M )

interior point z0 (offline):  (z0, eps*) = argmax_{z,eps} eps
                              s.t. all inequalities hold with margin eps,  0 <= eps <= delta
                              (delta = 0.5 in the paper); require eps* > 0.
```

**Guarantee.** Hard, for all inputs and all weights, during training and testing. Requires `Z` to
have non-empty interior and the feasible set to be *input-independent* in the base formulation
(the paper's `Y` is fixed; input-dependent sets need `z0` and the affine hull recomputed).

**Cost.** Closed form, no solve. Reported: 1K quadratic constraints on a 1K-dim variable costs
8 ms of overhead; a 300x300 LMI on a 10K-dim variable, 12 ms. Offline cost is the affine-hull
computation plus one convex program for `z0`.

**Maps onto us.** *Control path:* excellent fit and the closest match to what we already do — our
box `[thrust_lo, thrust_hi] x [bank_lo, bank_hi] x [n_lo, n_hi]` per step is a polytope with a
trivially known interior point (the box centre), so the linear `kappa_L` alone suffices and is a
one-line elementwise formula; RAYEN would be a *drop-in principled replacement for the tanh
squash*, differing in that it preserves the direction of `v` rather than squashing each
coordinate independently. *State path:* our corridor is input-dependent (it depends on `d`, hence
on the predicted state), which RAYEN in its published form does not handle — the interior point
`z0` and the affine hull would have to be recomputed per sample. Feasible for our geometry
(`z0` = corridor centreline at each `d`, closed form) but it is an extension, not the paper.
*Gating:* RAYEN gives no help; it needs `Y` fixed before the ray is shot. If the gate is wrong,
RAYEN enforces the wrong set just as hard.

---

### 2. HardNet — Hard-Constrained Neural Networks with Universal Approximation Guarantees

- Min, Y. & Azizan, N. (MIT). arXiv:2410.10807 (v4, Oct 2025); NeurIPS 2025 Constrained-ML
  workshop. File: `papers/hardnet_2410.10807.pdf`.

**Mechanism.** For *input-dependent two-sided affine* constraints, HardNet appends a closed-form,
almost-everywhere-differentiable repair layer to any backbone. The layer measures each
constraint's violation with a ReLU and pushes the output back through the pseudoinverse of the
constraint matrix. Crucially it is not the Euclidean projection: it is the *parallel* projection —
each violated constraint is snapped exactly to its own bound, and each satisfied constraint is
left untouched (`a_i^T P(f)(x) = a_i^T f(x)`). Because the repair is applied inside the forward
pass, the network trains *through* it. The paper's headline theoretical result is that this
preserves universal approximation over the constrained function class, i.e. the hard layer costs
no expressivity.

**Core formulas.**

```
constraints:   b_l(x) <= A(x) f(x) <= b_u(x)  for all x,   A(x) in R^{nc x nout}
assumptions:   (i) feasible for every x;  (ii) A(x) has full row rank (so nc <= nout)

HardNet-Aff (Eq. 5):
   P(f_theta)(x) = f_theta(x)
                 + A(x)^+ [  ReLU( b_l(x) - A(x) f_theta(x) )
                           - ReLU( A(x) f_theta(x) - b_u(x) ) ]
   with  M^+ := M^T (M M^T)^{-1}

single-constraint special case (Eq. 3), recovered with nc = 1, b_l = -inf:
   P(f)(x) = f(x) - a(x)/||a(x)||^2 * ReLU( a(x)^T f(x) - b(x) )

what it actually solves (Prop. 6):
   P(f)(x) = argmin_z ||z - f(x)||_2
             s.t. z in argmin_y || A(x)(y - f(x)) ||_2  s.t. b_l <= A(x) y <= b_u

parallel-projection property (Prop. 7), per row a_i of A(x):
   a_i^T P(f)(x) =  b_l(i)(x)        if a_i^T f(x) <  b_l(i)(x)
                    b_u(i)(x)        if a_i^T f(x) >  b_u(i)(x)
                    a_i^T f(x)       otherwise

HardNet-Cvx (Eq. 8), general convex C(x), no closed form:
   P(f)(x) = argmin_z ||z - f(x)||_2  s.t.  z in C(x)
   (implemented with OptNet / cvxpylayers)
```

**Guarantee.** Hard, input-dependent, for any weights, train and test; plus a universal
approximation theorem (Thm 8, Cor. 9): `{P(f_NN)}` universally approximates
`{f : b_l(x) <= A(x) f(x) <= b_u(x)}` if the backbone class does, for width `w >= max(nin+1, nout)`.

**Cost.** Closed form: one `A A^T` solve of size `nc x nc` per sample (trivial when `A` is
diagonal or block-diagonal, as ours is). Paper reports "a modest increase in training time".

**Maps onto us.** This is the single best structural match in the cluster. *State path:* our two
clauses are exactly `b_l(x) <= A y <= b_u` with, per timestep, `A` selecting the cross-track and
vertical runway-frame coordinates — `A` is a selection matrix, `A A^T = I`, so `A^+ = A^T` and the
whole layer collapses to an elementwise clamp in runway axes. **Note that this is the closed-form
justification for the tanh-bounded runway-axis output we already adopted**: HardNet's parallel
projection and a per-coordinate clamp coincide when `A` is a selection matrix; the difference is
that HardNet clamps (gradient 0 outside) where tanh squashes (gradient small but non-zero, and it
also distorts *inside* the corridor). Worth an arm: HardNet-clamp vs tanh-squash on identical
splits, since HardNet's Prop. 7 says the clamp leaves already-feasible outputs untouched, which
tanh does not. *Control path:* applies to the box on the controls in the same way, but says
nothing about the rollout's *state* constraints — the corridor is a nonlinear function of the
controls through RK4, so `A(x)` is not available in closed form. *Gating:* nothing; `b_l, b_u`
must be known from `x`. The natural hack is to let the gate move `b_l/b_u` continuously
(`b_u -> +inf` when the gate probability is 0), which HardNet supports natively via Remark 4
(`b_l = -inf` / `b_u = +inf` component-wise) and which degrades smoothly rather than switching.

---

### 3. Pi-net (ΠNET) — Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers

- Terpin, A., Grontas, P. D., Balta, E. C., Lygeros, J., D'Andrea, R. arXiv:2508.10480;
  ICLR 2026. File: `papers/pinet_2508.10480.pdf`. (Not on the brief's list; found while
  searching for RAYEN and kept because it is the current head of this line.)

**Mechanism.** Pi-net does the *true* Euclidean projection onto a convex set `C(x)` as an output
layer, but makes it cheap by (a) splitting `C(x)` into an affine part `A` and a "simple" cone/box
part `K` whose individual projections are closed form, (b) running Douglas–Rachford operator
splitting in the forward pass on GPU (batched, in JAX), and (c) backpropagating with the implicit
function theorem instead of unrolling the iterations. It positions itself explicitly against
RAYEN (which scales a ray from a fixed interior point, so it is not the nearest feasible point)
and against DC3 (whose correction is unrolled gradient descent).

**Core formulas.**

```
layer:          y = Pi_{C(x)}( y_raw ) = argmin_{z in C(x)} || z - y_raw ||

split:          C(x) = Pi_d( A(x) cap K(x) ),   Pi_d = projection onto first d coordinates
                A affine, K a set whose projection is closed-form (box, cone, ...)

composite form (Eq. 3):
   (Pi_C(y_raw), y_aux*) = argmin_{y,y_aux} || y - y_raw ||^2
                                            + I_A([y; y_aux]) + I_K([y; y_aux])
   split as   g([y;y_aux]) = I_A(.)   and   h([y;y_aux]) = ||y - y_raw||^2 + I_K(.)

Douglas-Rachford fixed-point iteration (Eq. 4):
   z_{k+1} = prox_{sigma g}( s_k )
   t_{k+1} = prox_{sigma h}( 2 z_{k+1} - s_k )
   s_{k+1} = s_k + omega ( t_{k+1} - z_{k+1} )

backward: implicit function theorem at the fixed point (no unrolling)
```

**Guarantee.** Hard by design ("the output `y` of the projection layer always lies in `C(x)`") for
convex `C(x)`, up to solver tolerance on the DR iteration.

**Cost.** Iterative but batched and GPU-native; claims orders-of-magnitude better training time
than prior learning approaches at similar inference time. Still `K` DR iterations per forward
pass — heavier than HardNet, lighter than a QP solver.

**Maps onto us.** *State path:* if we ever want the *nearest* feasible trajectory rather than a
clamp (e.g. to avoid the clamp's zero gradient), this is the method — our corridor is a box in
runway axes, so `K` is a box and `prox_K` is a clamp, and DR converges in a handful of iterations.
*Control path:* same, on the control box. *Gating:* none. Practical read: Pi-net is the "if
HardNet's parallel projection distorts too much, do the real projection" fallback, not a first
choice for us.

---

### 4. Gauge map — Computationally Efficient Safe RL for Power Systems

- Tabas, D. & Zhang, B. arXiv:2110.10333; ACC 2022. File: `papers/tabas_zhang_gaugemap_2110.10333.pdf`.

**Mechanism.** The gauge (Minkowski) function of a convex compact set containing the origin
measures how far along a direction the set extends. Tabas & Zhang use the ratio of two gauge
functions to build a **closed-form bijection from the unit hypercube `B_inf` onto an arbitrary
polytope `Q`**, preserving direction and mapping level sets to level sets. A network with a
`tanh` output (hence living in `B_inf`) is composed with this map, so its action is always inside
the safe action set — and, unlike a projection, the map is a *bijection*, so no volume of the
network's output space is wasted and no two distinct latent actions collapse. Because the safe
set `Omega(x_t)` generally does not contain the origin, they first shift it by a known safe
linear feedback `K x_t`.

**Core formulas.**

```
gauge function of C-set Q:      gamma_Q(v) = inf{ lambda >= 0 : v in lambda Q }
polytope Q = {w : F_i^T w <= g_i}:
                                gamma_Q(v) = max_i { F_i^T v / g_i }      (Eq. 11)
                                (derivation gives max{0, max_i(.)}; the max is >= 0 for a C-set)

GAUGE MAP, hypercube -> polytope (Eq. 12):
        G(v | Q) = ( ||v||_inf / gamma_Q(v) ) * v ,      v in B_inf
   Lemma 1: G : B_inf -> Q is a bijection; w = G(v|Q)  iff  w, v share a direction
            and gamma_Q(w) = ||v||_inf.

general P -> Q form (appendix):  G(v | P, Q) = ( gamma_P(v) / gamma_Q(v) ) * v

shift to make the safe set a C-set (Lemma 2):  Omega_hat_t := Omega(x_t) - K x_t
policy (Eq. 13):                 pi_theta(x_t) := G( psi_theta(x_t) | Omega_hat_t ) + K x_t
                                 with psi_theta : S -> B_inf  (e.g. a tanh-output network)
```

**Guarantee.** Hard: the action is in `Omega(x_t)` for every state in the invariant set `S` and
every weight vector. Requires a known robust control-invariant set `S` and a safe backup policy
`pi_0 = Kx` (this is the strong assumption).

**Cost.** Closed form: one `max` over the polytope's `r` rows plus an `inf`-norm. Differentiable
a.e. Cheaper than RAYEN (no interior-point program; the interior point is `Kx` by construction).

**Maps onto us.** *Control path:* this is the *most directly reusable* piece in the cluster and it
is essentially "what the tanh squash should have been". Our controls already come out of a tanh
into a box; the gauge map says how to send that box bijectively onto the *state-dependent* safe
polytope (e.g. bank limited by load factor and by the CBF condition) without ever leaving it and
without a filter. Because it is a bijection, the "laziness" failure we measured when training
*through* the CBF filter has a different character here: a projection/filter is many-to-one (many
latent actions map to the same clipped action, so the network gets no gradient signal to be
right), whereas the gauge map is one-to-one, so every latent action still has a distinct
consequence and the gradient never vanishes. **This is the single most promising lead for
replacing our vetoed filter-in-training arm.** *State path:* less natural — the per-step corridor
is a box already, so the gauge map degenerates to a rescale. *Gating:* the safe set `Omega(x_t)`
must be known, same limitation.

---

### 5. Gauge-map MPC — Safe and Efficient Model Predictive Control Using Neural Networks: An Interior Point Approach

- Tabas, D. & Zhang, B. arXiv:2203.12196; CDC 2022. File: `papers/tabas_zhang_mpc_interiorpoint_2203.12196.pdf`.
  Code: `github.com/dtabas/gauge_networks`.

**Mechanism (brief — companion to #4).** Same gauge-map layer, now used to parameterise an MPC
*policy*: the network explores the interior of the MPC feasible set in an unsupervised
(policy-gradient-free, cost-minimising) paradigm rather than imitating a solver. The interior
point of the feasible set `V(x0)` is found by minimising the maximum constraint violation. Reports
better policies found faster than projection-based baselines and substantially shorter solve
times, applied to a robust MPC problem.

**Guarantee / cost.** Hard constraint satisfaction inherited from the gauge map; closed-form
online, one small optimisation offline (or per `x0`) to get the interior point.

**Maps onto us.** The relevant lesson is methodological: *unsupervised* training inside a
hard-feasible parameterisation beats *imitating* a constrained solver then projecting. Our
`control` path is exactly that setup (a differentiable rollout gives us a cost, so we do not have
to imitate the IPOPT solution). Worth citing when arguing why the control path deserves a
by-construction layer rather than an inference-time filter.

---

### 6. Homeomorphic Projection — Liang, Chen & Low

- Liang, E., Chen, M., Low, S. H. "Homeomorphic Projection to Ensure Neural-Network Solution
  Feasibility for Constrained Optimization", JMLR 25 (2024) 1-55. Earlier/short version:
  "Low Complexity Homeomorphic Projection ...", ICML 2023, PMLR 202:20623-20649.
  File: `papers/homeomorphic_projection_jmlr25.pdf`.
- **NOTE**: the brief said "NeurIPS 2023". It is ICML 2023 (+ JMLR 2024), not NeurIPS.

**Mechanism.** Learn an invertible neural network (INN) `Phi_theta` that is a *minimum-distortion
homeomorphism* between the feasible set `K_theta` and the unit ball `B`. At inference, map an
infeasible prediction into ball coordinates, then simply shrink it toward the origin — a scalar
bisection on `alpha in [0,1]` — until it maps back inside `K_theta`. Because the ball is
star-shaped about its centre and the map has bounded distortion, shrinking is a cheap surrogate
for a real projection, and the optimality loss is bounded by the distortion. Covers all compact
convex sets and some non-convex ones (anything homeomorphic to a ball).

**Core formulas.**

```
Assumption 1:   K_theta is homeomorphic to the unit ball B = {x : ||x|| <= 1}

distortion (Def. 4):  D(psi, Z) = kappa_2 / kappa_1 >= 1  with
   kappa_1 = inf_{z1 != z2 in Z} ||psi(z1) - psi(z2)|| / ||z1 - z2||
   kappa_2 = sup_{z1 != z2 in Z} ||psi(z1) - psi(z2)|| / ||z1 - z2||

MDH mapping (Def. 5):  min_{psi in H_n}  log D(psi^-1, X_theta)   s.t.  K_theta = psi(B)

validity (Def. 6):     Phi_theta is valid for K_theta if Phi_theta(0) in K_theta

HOMEOMORPHIC BISECTION (Eq. 5):
   x_hat_theta = Phi_theta( alpha* . z_tilde_theta )
   where  z_tilde_theta = Phi_theta^{-1}( x_tilde_theta )
   and    alpha* = sup { alpha in [0,1] : Phi_theta(alpha . z_tilde_theta) in K_theta }

ideal case (INN learns the MDH exactly): bisection reduces to the closed form
   z_hat_theta = z_tilde_theta / || z_tilde_theta ||

k-step guarantee (Thm):  x_hat^k in K_theta, and
   || x_hat^k - x*_theta ||  <=  eps_pre  +  D(Phi^-1, .) * ( ... )  + eps_bis
   with eps_bis shrinking exponentially in k.

complexity: O( k ( m n^2 + G ) ) for k bisection steps
            (m n^2 = INN forward/inverse, G = constraint check)
```

**Guarantee.** Hard (feasible output guaranteed as long as the INN is *valid*, i.e. maps the ball
centre inside `K_theta`), with a *bounded* optimality loss rather than exact projection.

**Cost.** `k` bisection steps, each an INN forward + inverse + a constraint evaluation. Much
cheaper than a QP but not closed form; needs an INN trained per constraint family.

**Maps onto us.** Overkill for our geometry: our corridor is a box, so the homeomorphism to the
ball is trivial and known analytically — we do not need an INN to learn it. Its real relevance is
*negative evidence for a design choice*: the paper shows that "shrink toward a known interior
point" (which is also what RAYEN does) is enough to guarantee feasibility with bounded optimality
loss, i.e. our corridor centreline is a legitimate anchor and we do not need a true projection.
*Gating:* no help. `[UNVERIFIED]` whether the ICML and JMLR versions differ in the bisection
statement — I read the JMLR version only.

---

## A2. Differentiable projection / optimisation layers

### 7. DC3 — A Learning Method for Optimization with Hard Constraints

- Donti, P. L., Rolnick, D., Kolter, J. Z. ICLR 2021. arXiv:2104.12225.
  File: `papers/dc3_2104.12225.pdf`.

**Mechanism.** Two stages, both differentiable. **Completion**: the network outputs only `m` of
the `n` decision variables; the remaining `n-m` are *solved for* from the equality constraints
(explicitly for linear systems, or by Newton's method), and gradients flow through that solve via
the implicit function theorem. **Correction**: the completed point may still violate the
inequalities, so DC3 takes a fixed number of gradient-descent steps on the squared inequality
violation, *in the partial variables `z`*, so the correction moves along the equality manifold.
Both are applied at training time (with few steps, so backprop is affordable) and test time (with
more steps). Training minimises the soft loss anyway — the soft loss is *required*, because the
correction is only guaranteed to land in the feasible region if it starts near it.

**Core formulas.**

```
problem:        min_y f_x(y)  s.t.  g_x(y) <= 0,  h_x(y) = 0
soft loss (Eq. 2, still used as the training loss):
   L_soft(y_hat) = f_x(y_hat) + lambda_g ||ReLU(g_x(y_hat))||_2^2 + lambda_h ||h_x(y_hat)||_2^2

COMPLETION:  network emits z in R^m; phi_x : R^m -> R^{n-m} with h_x([z; phi_x(z)]) = 0
             y_tilde = [ z ; phi_x(z) ]

  implicit differentiation (Eq. 3), J^h = Jacobian of h_x wrt y:
     0 = d/dz h_x([z; phi_x(z)]) = J^h_{:,0:m} + J^h_{:,m:n} * d phi_x(z)/dz
     => d phi_x(z)/dz = - ( J^h_{:,m:n} )^{-1} J^h_{:,0:m}
     => dL/dz = dL/dz|_explicit - ( dL/d phi_x(z) ) ( J^h_{:,m:n} )^{-1} J^h_{:,0:m}

CORRECTION:  one step, learning rate gamma:
     rho_x( [z; phi_x(z)] ) = [ z - gamma*Delta_z ; phi_x(z) - gamma*Delta_{phi_x(z)} ]
     Delta_z          = grad_z || ReLU( g_x([z; phi_x(z)]) ) ||_2^2
     Delta_{phi_x(z)} = ( d phi_x(z)/dz ) Delta_z

train: y_hat = rho_x^{(t_train)}( y_tilde ),  t_train small (5-10 in the paper)
test:  y_hat = rho_x^{(t_test)}( y_tilde )
```

**Guarantee.** Equalities: hard (by completion). Inequalities: **soft in general** — feasibility
holds only in the limit `t -> inf` and is *proved* only for linear inequality constraints
("for linear constraints as in 4.1-4.2, the correction process is mathematically guaranteed to
converge"). At finite `t` it is approximate.

**Cost.** `t` extra gradient steps per sample per forward pass (each needing `d phi_x/dz`, i.e. a
linear solve of size `(n-m)`), at both train and test time. Substantially more than a closed-form
layer.

**Maps onto us.** *State path:* our inequalities are linear in the runway frame, which is exactly
the case DC3 can prove; but with a closed-form projection available (HardNet), `t` unrolled
gradient steps buys nothing. *Control path:* the completion idea is the interesting transfer —
our `control` head already emits a partial parameterisation (controls + duration) and the rollout
*is* `phi_x`, i.e. the "equality constraints" (the dynamics) are satisfied by construction and
differentiated through. Our control path is structurally a DC3 without the correction stage.
**The relevant warning from DC3 is that the correction stage needs the soft loss to work** — and
we vetoed the soft hinge on both paths, which means a DC3-style correction is not available to us
as-is. *Gating:* none.

---

### 8. OptNet — Differentiable Optimization as a Layer in Neural Networks

- Amos, B. & Kolter, J. Z. ICML 2017. arXiv:1703.00443. File: `papers/optnet_1703.00443.pdf`.

**Mechanism.** A layer whose forward pass *is* a QP solve, with all of `Q, q, A, b, G, h`
allowed to depend differentiably on the previous layer. The backward pass never re-solves and
never unrolls: it takes matrix differentials of the KKT conditions at the optimum and solves one
linear system whose matrix is the (already factorised) KKT matrix, so the backward pass is
quadratic where the forward was cubic. This is the ancestor of every "optimisation as a layer"
method in this cluster.

**Core formulas.**

```
layer (Eq. 1-2):
   z_{i+1} = argmin_z  (1/2) z^T Q(z_i) z + q(z_i)^T z
             s.t.  A(z_i) z = b(z_i),   G(z_i) z <= h(z_i),   Q >= 0

Lagrangian:  L(z,nu,lambda) = (1/2) z^T Q z + q^T z + nu^T(Az - b) + lambda^T(Gz - h)

KKT at the optimum (Eq. 4):
   Q z* + q + A^T nu* + G^T lambda* = 0
   A z* - b = 0
   D(lambda*) ( G z* - h ) = 0            D(.) = diag(.)

matrix differentials (Eq. 6):
   [ Q          G^T             A^T ] [ dz     ]     [ dQ z* + dq + dG^T lambda* + dA^T nu* ]
   [ D(l*)G     D(G z* - h)     0   ] [ dlambda ]  = -[ D(lambda*) dG z* - D(lambda*) dh    ]
   [ A          0               0   ] [ dnu    ]     [ dA z* - db                           ]

BACKWARD PASS (Eq. 7) — one solve with the transposed KKT matrix:
   [ d_z      ]       [ Q        G^T D(lambda*)   A^T ]^{-1}  [ (dL/dz*)^T ]
   [ d_lambda ] = -   [ G        D(G z* - h)      0   ]       [ 0          ]
   [ d_nu     ]       [ A        0                0   ]       [ 0          ]

gradients (Eq. 8):
   grad_Q L = (1/2)( d_z z*^T + z* d_z^T )      grad_q L = d_z
   grad_A L = d_nu z*^T + nu* d_z^T             grad_b L = -d_nu
   grad_G L = D(lambda*) d_lambda z*^T + lambda* d_z^T
   grad_h L = -D(lambda*) d_lambda
```

**Guarantee.** Hard, exactly (to solver precision) — the output *is* the constrained argmin.

**Cost.** One QP solve per sample per forward pass (a batched primal-dual interior-point method
on GPU in the paper), plus one backsubstitution with the cached KKT factorisation on backward.
This is the expensive end of the cluster; the reason the closed-form papers exist.

**Maps onto us.** We should **not** put a QP in our training loop — with `N=64` steps and a
64-sample batch that is 64 QPs per forward pass, and our constraints are affine anyway, where
HardNet's closed form is exact enough. OptNet's value to the thesis is as the citation for *why*
a closed form is preferred, and as the fallback if we ever need a genuinely coupled constraint
(e.g. corridor + a terminal CTA + a wake-separation constraint jointly). *Gating:* none.

---

### 9. cvxpylayers — Differentiable Convex Optimization Layers

- Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S., Kolter, J. Z. NeurIPS 2019.
  arXiv:1910.12430. File: `papers/cvxpylayers_1910.12430.pdf`.

**Mechanism.** Generalises OptNet from QPs to *any* disciplined convex program, and removes the
requirement to hand-write the canonical form. They introduce **DPP** (disciplined parametrized
programming), a grammar restricting DCP just enough that the canonicalisation map from parameters
to cone-program data `(A,b,c)` is *affine* — hence trivially differentiable and representable as a
fixed sparse matrix whose transpose is the backward pass. The cone program itself is
differentiated by implicit differentiation of its optimality conditions (via `diffcp`).

**Core formulas.**

```
solution map factorisation:   S = R o s o C
   C : parameters theta  ->  cone-program data (A, b, c)      [affine under DPP => "ASA form"]
   s : (A,b,c)           ->  cone solution x~*
   R : x~*               ->  solution x* of the original problem   [affine; here just a slice]

ADJOINT OF THE DERIVATIVE (the whole method in one line):
   D^T S(theta) = D^T C(theta) . D^T s(A,b,c) . D^T R(x~*)

cone program standard form:
   minimize c^T x  s.t.  b - A x in K
D^T s: implicit function theorem on the cone program's optimality conditions;
       at non-differentiable points a least-squares solution of the (singular) linear
       system is used as a heuristic derivative.
solution retrieval: x~* = (x*, s*), R(x~*) = x*  (a slice, hence linear)
```

**Guarantee.** Hard (the output is the exact convex-program solution) for any DPP-representable
`C(x)`; this is the implementation HardNet-Cvx and others call.

**Cost.** One cone solve per sample per forward pass, plus a linear solve on backward. Batched in
their PyTorch/TF layers, but still solver-in-the-loop.

**Maps onto us.** Same verdict as OptNet — a reference implementation and a citation, not a
training-loop candidate at `N=64`. It is however the right tool for a *one-off* offline study,
e.g. "what is the exact projection distance of our unconstrained predictions onto the corridor,
as a distribution over the test set?" — that is a diagnostic we can run once with cvxpylayers to
quantify how much the corridor would have to move the model's outputs, which is exactly the
number that decides whether a hard layer is worth it.

---

### 10. KKT-hPINN — Physics-Informed Neural Networks with Hard Linear Equality Constraints

- Chen, H., Constante Flores, G. E., Li, C. (Purdue ChE). arXiv:2402.07251;
  *Computers & Chemical Engineering* 2024. File: `papers/kkt_hpinn_2402.07251.pdf`.
  Code: `github.com/li-group/KKThPINN`.

**Mechanism.** For *linear equality* constraints only, the Euclidean projection onto the feasible
affine set has a closed form obtained by solving the KKT system of the projection QP. The authors
bake that closed form into two fixed, non-trainable layers appended to any backbone. Their
emphasised point is Remark 3: putting the projection *inside* the architecture (so training sees
projected outputs and the gradient direction changes every step) is materially different from
post-hoc projection at test time, and empirically more accurate.

**Core formulas.**

```
prior knowledge:   A x_hat + B y = b,   A in R^{m x N0}, B in R^{m x NL}, rank(B) = m

projection QP:     y_tilde = argmin_y (1/2) || y - y_hat ||^2   s.t.  A x_hat + B y = b

KKT system:        [ I   B^T ] [ y_tilde  ]   [ y_hat        ]
                   [ B   0   ] [ lambda*  ] = [ b - A x_hat  ]

CLOSED FORM (Thm 1, Eq. 2):   y_tilde = A* x_hat + B* y_hat + b*
                              A* = - B^T (B B^T)^{-1} A
                              B* =   I - B^T (B B^T)^{-1} B
                              b* =   B^T (B B^T)^{-1} b
```

**Guarantee.** Hard, exact, for *linear equality* constraints. No inequality support (the
follow-ups arXiv:2606.10682 and arXiv:2507.08124 extend to piecewise-linear / nonlinear and to
inequalities — `[UNVERIFIED]`, found in search results only, not read).

**Cost.** Two fixed matrix multiplies; `(B B^T)^{-1}` is precomputed once. Cheapest layer in the
cluster.

**Maps onto us.** Our procedure constraints are *inequalities*, not equalities, so KKT-hPINN does
not apply to the corridor directly. Two genuine uses: (a) a **terminal equality** — "the
trajectory ends at the threshold at the published TCH and runway heading" is `B y = b` exactly,
and KKT-hPINN would enforce it in closed form on the `state` path, which is a cleaner endpoint
treatment than a loss term; (b) Remark 3 is the citable evidence for a claim we need anyway —
that training *through* a hard layer differs from applying it at inference. Note the contrast
with our CBF finding: KKT-hPINN says training-through *helps*, our filter experiment says
training-through *hurts*. The difference is that KKT-hPINN's layer is a fixed affine map
(injective on the feasible directions, gradient never vanishes), whereas a CBF filter is
many-to-one and saturating. That distinction is the thesis argument.

---

## A3. Hard vs soft — the evidence, and architectural constraint enforcement

### 11. Beucler et al. — Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems

- Beucler, T., Pritchard, M., Rasp, S., Ott, J., Baldi, P., Gentine, P.
  *Phys. Rev. Lett.* 126, 098302 (2021). arXiv:1909.00912v5.
  File: `papers/beucler_analytic_constraints_1909.00912.pdf`.

**Mechanism.** The cleanest statement of "constrain the architecture, not the loss". Given `n`
analytic constraints `C [x; y] = 0` on `p` outputs, the network is made to emit only `p - n`
"direct" outputs; the remaining `n` are computed as exact **residuals** by solving the constraint
system (in row-echelon form) inside a fixed, non-trainable layer. The loss is still the plain MSE
on the *full* output vector, so gradients pass through the constraint layer and the learned
weights depend on `C`. They contrast this (ACnet) against a soft penalty (LCnet) with weight
`alpha`, and find architectural constraints hold to machine precision *without degrading*
performance, and reduce errors in the outputs most affected by the constraints. The free choice
of *which* outputs are residuals is `n` extra hyperparameters.

**Core formulas.**

```
error vector:              y_Err = y_NN - y_Truth

SOFT (LCnet), Eq. 5-6:
   P(x, y_NN) = || C [x ; y_NN] ||^2
              = (1/n) sum_i ( sum_j C_ij x_j + sum_k C_{i(k+m)} y_NN,k )^2
   L(alpha)   = alpha * P(x, y_NN) + (1 - alpha) * MSE( y_Truth, y_NN ),   alpha in [0,1]

HARD (ACnet), Eq. 7:
   min MSE   s.t.  C [ x  y_NN ]^T = 0
   implementation: network emits (p - n) "direct" outputs; the remaining n "residual"
   outputs are solved from C (row-echelon, bottom row upward) in a fixed layer;
   concatenate -> y_NN satisfies C to machine precision.
```

**Guarantee.** Hard for *equality* constraints (machine precision). Requires `n <= p` and a valid
choice of residual outputs. HardNet's Remark 3 notes the failure mode: a chosen residual subset
can be invalid for some inputs (e.g. `[x, x-1] f(x) = 1` at `x = 0` and `x = 1`).

**Cost.** One fixed triangular solve per forward pass; negligible.

**Maps onto us.** Direct structural analogue for our *vertical* clause if we ever express the
glidepath as an equality-plus-window rather than a window: emit `d` and a bounded offset, compute
`u` as the residual `u = d*tan(3deg) + offset`. This is the **residual-from-a-reference-path**
pattern in its purest form and, together with Werling (#14) and PBP (#21), is the strongest
architectural argument for the runway-relative parameterisation we already adopted. Also the
citation of record for "hard beat soft at equal accuracy in a physics emulator", which we need to
justify having vetoed the hinge. *Gating:* none — the constraints are always active there.

---

### 12. Márquez-Neila, Salzmann & Fua — Imposing Hard Constraints on Deep Networks: Promises and Limitations

- EPFL, 2017. arXiv:1706.02025. File: `papers/marquezneila_hard_constraints_1706.02025.pdf`.
- This is the cluster's **negative result**, and the most important one for us to cite honestly.

**Mechanism.** They train a deep network under hard *equality* constraints using a proper
Lagrangian/KKT method rather than a penalty. The obstacle is scale: the KKT linear system has the
dimension of the weight vector (millions), so it cannot be formed. They solve it with a Krylov
subspace method that needs only matrix-vector products (available through R-op/L-op), plus a
per-iteration subsampling of the active constraint set to keep cost bounded. Applied to human
pose estimation.

**Core formulas.**

```
Lagrangian (Eq. 4):    min_w max_Lambda L(w, Lambda),   L = R(w) + Lambda^T C(w)

linearised KKT step (Eq. 5), solved for the increment dw and multipliers Lambda:
   [ eta I        (dC/dw)^T ] [ dw     ]   [ - dR/dw  ]
   [ dC/dw        0         ] [ Lambda ] = [ - C(w)   ]
   (this is projected gradient descent: the projection is onto the hyperplanes of
    the linearised constraints)

solved by a Krylov subspace method on B v = b without forming B.
```

**Result and guarantee.** Hard constraints *are* trainable at deep-network scale — "but not better
than using a soft-constraint approach as we had hoped". The authors attribute the negative result
to having to re-select a subset of active constraints every iteration (forgetting previously
enforced ones), which successful hard-constraint methods do not do.

**Cost.** Explicitly "slow": a Krylov solve per iteration.

**Maps onto us.** Read this as the boundary of the cluster: **imposing hard constraints on the
weights is a losing move; imposing them on the outputs by construction is not.** Every method that
works in A1/A2 constrains the *output map*, leaving the weights unconstrained. Our own vetoed soft
hinge and our vetoed train-through-filter are, in this light, the two ways *not* to do it, and
Márquez-Neila is the citation showing the Lagrangian third way is also unattractive. `[UNVERIFIED]`
whether later work has revisited their subsampling explanation.

---

### 13. POLICE — Provably Optimal Linear Constraint Enforcement for Deep Neural Networks

- Balestriero, R. & LeCun, Y. arXiv:2211.01340v3 (2023); ICASSP 2023.
  File: `papers/police_2211.01340.pdf`.

**Mechanism.** A continuous-piecewise-affine (ReLU / leaky-ReLU / abs) network is affine on each
region of its input-space partition. POLICE forces the network to be *exactly affine* on a chosen
convex input region `R` by ensuring all vertices of `R` share the same pre-activation sign pattern
at every layer — achieved by adding a per-layer bias shift `c` computed from the vertices. It is a
training-time algorithm with zero inference cost (fold `c` into the biases once, post-training),
no extra parameters, and no sampling.

**Core formulas.**

```
region in V-representation (Eq. 7):
   R = { V^T alpha : alpha_p >= 0 for all p,  alpha^T 1_P = 1 },   V = [v_1, ..., v_P]^T

Theorem 1: a CPA DNN stays affine on R  iff  all vertices v_p share the same
           pre-activation sign pattern, at every layer.

POLICE per-layer bias shift (Algorithm 1), for layer l:
   H = V^{(l)} (W^{(l)})^T + 1_P (b^{(l)})^T
   s = MajorityVote( sign(H), axis = 0 )
   c = ReLU( -H diag(s) ).max(axis = 0) * s
   X^{(l+1)} = sigma( X^{(l)} (W^{(l)})^T + 1_N ( b^{(l)} + c )^T )
   V^{(l+1)} = sigma( V^{(l)} (W^{(l)})^T + 1_P ( b^{(l)} + c )^T )

test time: run the same procedure once with no data and set b^{(l)} <- b^{(l)} + c.
```

**Guarantee.** Hard, provable: the network *is* affine on `R`. Note this constrains the network's
*functional form on an input region*, not the output's membership in a set. Cost is `P` extra
forward rows per layer per batch at train time, zero at inference.

**Maps onto us.** **Marginal relevance — flagging that clearly.** POLICE constrains behaviour over
an *input* polytope; our constraint is on the *output*. It would apply only in an inverted framing
("for all lookbacks in this set, the map must be affine"), which we do not want. The one place it
could matter is a *smoothness/monotonicity* guarantee on the glidepath segment (forcing the
network to be affine in `d` once established would make the predicted vertical profile exactly a
straight line, which is arguably the right inductive bias on final). Worth one paragraph in the
thesis as "considered and rejected, for this reason", not an experiment.

---

## A4. Trajectory-side constructive feasibility

### 14. Werling, Ziegler, Kammel & Thrun — Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenét Frame

- IEEE ICRA 2010, pp. 987-993. DOI 10.1109/ROBOT.2010.5509799. No arXiv.
  File: `papers/werling2010_frenet_icra.pdf` (open copy from the Lund ICRA-2010 proceedings mirror).

**Mechanism.** The origin of the reference-path-plus-bounded-offset parameterisation. Rather than
planning in Cartesian coordinates, express the trajectory as arc length `s(t)` along a reference
centreline plus a perpendicular offset `d(t)`. Then plan `s` and `d` *independently* as
one-dimensional jerk-optimal polynomials — quintics between fully specified boundary states,
quartics when the terminal position is free — and only afterwards map back to Cartesian. The
key structural fact is that the road constraint becomes a *bound on one scalar coordinate*.

**Core formulas.**

```
FRENET PARAMETERISATION (Eq. 1):
   x(s(t), d(t)) = r(s(t)) + d(t) * n_r(s(t))
   r  = reference centreline,  n_r = its normal,  s = arc length,  d = lateral offset

jerk cost:      J_t(p(t)) = integral_{t0}^{t1} p'''(tau)^2 d tau

Proposition 1: for start state P0 = [p0, p0', p0''] and end state P1 = [p1, p1', p1''] over
   T = t1 - t0, the minimiser of  C = k_j J_t + k_t g(T) + k_p h(p1)   (k_j,k_t,k_p > 0,
   g,h arbitrary) is a QUINTIC POLYNOMIAL in t.   (quartic when p1 is left free)

low-speed mode (Eq. after 1):  x(s(t), d(t)) = r(s(t)) + d(s(t)) * n_r(s(t))
   i.e. lateral offset parameterised by arc length, not time, to avoid the singularity
   as s' -> 0.
```

**Guarantee.** Not a learning method: the *parameterisation* makes the corridor constraint
one-dimensional and trivially checkable; feasibility is then obtained by generating a family and
checking, not by construction.

**Cost.** Closed-form polynomial coefficients.

**Maps onto us.** This is the mathematical justification for our runway-frame output. Our
`(xt, u, d)` runway coordinates *are* Frenét coordinates on a straight reference (the extended
runway centreline plus the 3-degree glidepath), which is the degenerate-curvature case where
`r(s)` is a line and `n_r` is constant — so the transform is exact and cheap, with none of the
curvature singularities Werling has to handle. **Consequence worth stating in the thesis: in the
runway frame the corridor is a per-timestep interval on two scalars, so a bounded output
(tanh/clamp) is not a heuristic — it is the exact feasible set.** The low-speed remark (parameterise
`d` by `s`, not `t`) is directly relevant to us: our corridor half-width `hw(d)` is a function of
distance, not time, so predicting the offset *as a function of along-track distance* rather than
of timestep is the structurally correct choice, and would make the constraint a fixed box
independent of the (unknown) speed profile.

---

### 15. Zhou, Gao, Wang, Liu & Shen — Robust and Efficient Quadrotor Trajectory Generation for Fast Autonomous Flight

- IEEE RA-L 2019 (Fast-Planner). arXiv:1907.01531. File: `papers/fastplanner_bspline_1907.01531.pdf`.
- **Stand-in for Gao et al., ICRA 2018** ("Online Safe Trajectory Generation for Quadrotors Using
  Fast Marching Method and Bernstein Basis Polynomial", DOI 10.1109/ICRA.2018.8462878), which is
  **not downloadable: closed access** (Semantic Scholar `openAccessPdf` status CLOSED; the
  HKUST-Aerial-Robotics/Btraj repository hosts code only). Same group, same convex-hull mechanism.

**Mechanism.** Represent the trajectory as a uniform B-spline over control points `Q_i`. The
**convex-hull property** says a span of the curve lies inside the convex hull of its `p_b + 1`
governing control points; and the `k`-th derivative of a B-spline is a B-spline of order
`p_b - k` whose control points are finite differences of the originals. So *constraining the
control points constrains the whole continuous curve, and its velocity/acceleration/jerk, without
sampling* — a finite set of linear constraints implies an infinite set.

**Core formulas.**

```
uniform B-spline of degree p_b, control points {Q_0..Q_N}, knot spacing Delta t.

CONVEX-HULL PROPERTY: the span on (t_i, t_{i+1}) lies inside
   conv{ Q_{i - p_b}, Q_{i - p_b + 1}, ..., Q_i }
   => if all governing control points lie in a convex safe region, so does the curve.

derivative control points (Eq. 7 / Eq. 2 of EGO-Planner):
   V_i = ( Q_{i+1} - Q_i ) / Delta t
   A_i = ( V_{i+1} - V_i ) / Delta t
   J_i = ( A_{i+1} - A_i ) / Delta t
   => dynamic-feasibility bounds ||V_i|| <= v_max, ||A_i|| <= a_max are LINEAR in Q,
      and imply the bound holds for the whole continuous trajectory.

only the interior N + 1 - 2 p_b control points are optimised; the first and last p_b are
fixed by the boundary conditions.
```

**Guarantee.** Hard and *continuous-time* (not just at sample points), but **conservative** — the
convex hull is larger than the curve, so satisfiable trajectories can be rejected.

**Cost.** Free: the constraints are linear in the decision variables; the hull property is a
theorem, not a computation.

**Maps onto us.** Strong idea, applicable to the `state` path if we ever move from per-timestep
positions to a spline parameterisation. Our corridor is a convex region *per distance band*, so a
B-spline whose control points are constrained to lie in the corridor would satisfy the corridor
*between* our 64 sample points as well — closing a real gap, since right now we only constrain
where we sample. The cost is conservatism (roughly, the hull is wider than the curve, so we would
enforce a slightly tighter corridor than the real one) and a change of output head. Also gives us
dynamic feasibility (bounded speed/acceleration) as *linear* constraints on the same control
points, which the `state` path currently has no way to express at all.

---

### 16. Zhou, Wang, Ye, Xu & Gao — EGO-Planner: An ESDF-free Gradient-based Local Planner for Quadrotors

- IEEE RA-L 2021. arXiv:2008.08835. File: `papers/egoplanner_2008.08835.pdf`.

**Mechanism (brief — companion to #15).** Same B-spline convex-hull machinery, but the collision
term is built per control point from `{p, v}` pairs (a point on the obstacle surface and the unit
vector to it) discovered by comparing the naive trajectory against the map, avoiding a global
ESDF. Dynamic feasibility is handled by *lengthening the time allocation* when the derivative
control-point bounds are violated, then refitting — i.e. the duration is the slack variable that
restores feasibility.

**Core formulas.** As #15 (Eq. 2 for `V_i, A_i, J_i`); objective
`min_Q J = lambda_s J_s + lambda_c J_c + lambda_d J_d` (smoothness, collision, dynamic).

**Maps onto us.** Two transfers. (a) The convex-hull-on-control-points argument, as in #15.
(b) **Time allocation as the feasibility slack** — our `control` path already predicts a *duration*
alongside the controls; EGO-Planner is the precedent for treating that duration as the variable
that absorbs infeasibility (if the corridor cannot be met at the predicted speed, stretch time
rather than violate geometry). That is a concrete, cheap arm we have not run.

---

### 17. Cui et al. — Deep Kinematic Models for Kinematically Feasible Vehicle Trajectory Predictions

- Cui, H., Nguyen, T., Chou, F.-C., Lin, T.-H., Schneider, J., Bradley, D., Djuric, N.
  ICRA 2020. arXiv:1908.00219. File: `papers/cui_deep_kinematic_1908.00219.pdf`.

**Mechanism.** The most direct analogue of our `control` path in the driving literature. Instead
of regressing positions, the network's head emits *controls* (longitudinal acceleration `a` and
steering angle `gamma`) at each horizon, and a fixed, differentiable **kinematic layer** — a
bicycle model integrated forward — turns them into positions. Controls are clipped to physical
ranges, so every predicted trajectory is kinematically feasible by construction. The model still
outputs positions and is trained with exactly the same position loss as the unconstrained
baseline; no ground-truth controls are needed.

**Core formulas.**

```
rollout (Eq. 4-5):   s_{i(j+h+1)} = f( s_{i(j+h)}, a_{i(j+h)}, gamma_{i(j+h)}, kappa_i )
                     s_{i(j+h+1)} = s_{i(j+h)} + s'_{i(j+h)} * Delta t
state s = (x, y, psi, v);  kappa_i = (l_r, l_f, a_max, gamma_max)

state derivatives (Eq. 6), kinematic bicycle with slip angle beta:
   x'    = v cos( psi + beta )
   y'    = v sin( psi + beta )
   psi'  = ( v / l_r ) sin( beta )
   v'    = a
   beta  = arctan( ( l_r / ( l_f + l_r ) ) tan( gamma ) )

controls are CLIPPED to the allowed ranges (|a| <= 8 m/s^2, |gamma| <= gamma_max);
equations (5)-(6) are fully differentiable, so training is end-to-end on the same
position losses as the unconstrained model, with no control labels.
```

**Guarantee.** Hard *kinematic* feasibility (no sideways motion, bounded acceleration/steering).
**No** environmental/map feasibility — nothing stops the trajectory leaving the road.

**Cost.** `H` extra forward-Euler steps, negligible; no solve.

**Maps onto us.** Our `control` path *is* this paper, with a point-mass aircraft model and RK4
instead of a bicycle and forward Euler, plus an actuator lag we add and they do not. Two things to
lift: (a) they clip rather than squash, and report no laziness problem — a useful data point
against our tanh; (b) their experiments include the honest negative finding that at short horizons
the unconstrained model can match or beat the kinematic one (the paper reports simple polynomial
baselines "on par with the best-performing DKM"), which is the same shape as our result and worth
citing rather than hiding. It also makes the paper's limitation *our* opening: DKM gets kinematic
feasibility for free but has no way to express the map constraint — which is exactly the corridor
problem, and is what the runway-frame output solves.

---

### 18. Trajectron++ — Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data

- Salzmann, T., Ivanovic, B., Chakravarty, P., Pavone, M. ECCV 2020. arXiv:2001.03093.
  File: `papers/trajectronpp_2001.03093.pdf`.

**Mechanism.** A CVAE-based multi-agent forecaster whose decoder outputs a distribution over
*control actions* (e.g. acceleration and steering rate) rather than positions; the agent's system
dynamics are then integrated to produce trajectories. For linear dynamics (single integrator,
used for pedestrians) the integration of a Gaussian is exactly a Gaussian, so the output
distribution stays closed-form; for the nonlinear dynamically-extended unicycle (vehicles) they
use an approximate propagation scheme, linearising about the current state and control. The
result: every sample is dynamically feasible, and the model is a drop-in for planners.

**Core formulas.**

```
decoder outputs the parameters of a bivariate Gaussian over CONTROLS u^(t)
(e.g. acceleration, steering rate); trajectories are obtained by integrating the
agent's dynamics with those controls.

single integrator (linear):  u^(t) = p'^(t)  =>  the integrated position distribution
   is exactly Gaussian (linear-Gaussian propagation of mean and covariance).

dynamically-extended unicycle (nonlinear): mean and covariance propagated by
   linearising the dynamics about the current state and control.
   (full mean/covariance equations are in the paper's appendix)
```

**Guarantee.** Hard dynamic feasibility of samples (no sideways-moving cars); the *distribution*
for nonlinear models is approximate because of the linearisation. No map/corridor guarantee.

**Cost.** Negligible; an integration, plus a Jacobian for the nonlinear case.

**Maps onto us.** Same family as #17 and as our `control` path. The distinctive transferable idea
is the **closed-form uncertainty propagation**: if we ever want a *probabilistic* corridor
statement ("P(inside the corridor) >= 1 - eps") rather than a deterministic one, Trajectron++ is
the template for pushing a control-space Gaussian through the dynamics to get a state-space
covariance we can compare against the corridor half-width. That would also give a principled
answer to the gating problem — a soft gate becomes a mixture weight rather than a hard switch.
`[UNVERIFIED]`: I read the main text, not the appendix's full covariance equations.

---

### 19. PRIME — Learning to Predict Vehicle Trajectories with Model-based Planning

- Song, H., Luan, D., Ding, W., Wang, M. Y., Chen, Q. (HKUST). CoRL 2021. arXiv:2103.04027.
  File: `papers/prime_2103.04027.pdf`.

**Mechanism.** Split the problem so that the *network never regresses a trajectory*. A model-based
generator searches reachable paths on the HD map (DFS over the lane graph), then samples a finite
set of candidate trajectories along each path with a classical Frenét sampling planner (Werling's
method) under explicit kinematic and environmental constraints `C`. A learning-based evaluator
then only *scores* that feasible set and selects. Because the candidate set is feasible by
construction, so is every possible prediction — and the paper reports this also improves
robustness under imperfect tracking, since the geometry does not have to be inferred from noisy
history.

**Core formulas.**

```
generator:  G : ( s0_tar, M, C )  ->  ( P, T )
   P = { P_j }_{j=1..l}          reachable paths from HD map M by DFS
   T = union_j { T_{j,k} }_{k=1..n_j}    feasible trajectories sampled along P_j
                                          under explicit constraints C;  n = sum_j n_j
   trajectory sampling in the Frenet frame of P_j:  x(s(t), d(t)) = r(s) + d(t) n_r(s)
   (i.e. Werling 2010, cited as the generator)

evaluator:  E : ( P, T, S )  ->  ( T_tar, {p_k} ),   T_tar subset of T
   attention over l paths, m surrounding agents, n candidate trajectories;
   E SCORES trajectories rather than regressing positions.
```

**Guarantee.** Hard, by set construction: every output is an element of a set that was generated
under the constraints. Coverage is the price — if the true behaviour is outside the sampled set,
it cannot be predicted (the paper's own criticism of CoverNet applies partly to itself).

**Cost.** Generation cost is a path search plus polynomial sampling per agent, done outside the
network; the network is a scorer, which is cheap.

**Maps onto us.** A genuinely different third option beside our two paths: **generate a bank of
corridor-feasible arrival trajectories from the published procedure geometry (segment fits at a
grid of speeds/descent profiles/intercept points) and have the transformer score them.** For
arrivals this is more attractive than for driving, because the procedure geometry is *published*
and low-dimensional — the candidate set is small and genuinely covering, in a way that a lane-graph
DFS is not. Also the cleanest answer to the gating problem in the whole cluster: the gate becomes
a *choice among candidates* (some candidates establish early, some late), so it is learned as a
scoring problem rather than being a threshold we have to guess. Worth writing up as a
pre-registered alternative arm.

---

### 20. CoverNet — Multimodal Behavior Prediction using Trajectory Sets

- Phan-Minh, T., Grigore, E. C., Boulton, F. A., Beijbom, O., Wolff, E. M. CVPR 2020.
  arXiv:1911.10298. File: `papers/covernet_1911.10298.pdf`.

**Mechanism (brief).** Recast multimodal prediction as *classification over a trajectory set*
instead of regression. The set is designed by the user, which is where the feasibility comes from:
a **dynamic** set is generated from the agent's current state so that it contains only physically
reachable trajectories, and a **fixed** set is built by solving a set-cover problem for a target
coverage `epsilon`. Mode collapse is impossible by construction and physically impossible
trajectories are excluded a priori.

**Core formulas.**

```
R(s_t) = set of states reachable from s_t in N steps (physical capability);
approximate it by a finite trajectory set K = { s_{t:t+N} }.
dynamic generator  f_N : s_0 -> K   (state-dependent);  fixed generator ignores s_0.

prediction as classification:
   p( s^k_{t:t+N} | x ) = exp f_k(x) / sum_i exp f_i(x)

fixed-set construction as set cover (Eq. 1):
   argmin_K |K|   s.t.  K subset of K_0,   coverage delta(K, K_0) <= epsilon
   with delta = maximum point-wise Euclidean distance between trajectories;
   |K_0| = 20,000 sampled from training data.
```

**Guarantee.** Hard, in the sense that outputs are drawn from a set the designer certified; the
guarantee is exactly as strong as the set-construction procedure, and coverage is `epsilon`-bounded
by design (and *stated*, which is the right practice).

**Maps onto us.** The lighter-weight cousin of #19 and a good baseline: build the trajectory set
from the published approach geometry, classify. Its explicit `epsilon` coverage statement is a
model for how to report a bounded output space honestly.

---

### 21. Map-Adaptive Goal-Based Trajectory Prediction

- Zhang, L., Su, P.-H., Hoang, J., Haynes, G. C., Marchetti-Bowick, M. (Uber ATG). CoRL 2020.
  arXiv:2009.04450. File: `papers/map_adaptive_goal_2009.04450.pdf`.

**Mechanism (brief).** Generate a variable number of candidate *goal paths* per actor from lane
centrelines at run time, then predict the trajectory **in the path-relative frame** — i.e. regress
`tau_ac = Pi_rho(tau_xy)`, the projection of the trajectory into (along-track, cross-track)
coordinates of the reference path `rho` — rather than in an actor-centric Cartesian frame. Their
stated benefits are exactly the two we care about: generalisation across paths of different
curvature, and a clean decomposition of the spatial (cross-track) and temporal (along-track)
dimensions. Supervision for which mode is "correct" uses a cross-track-deviation test: an actor is
following a goal path if its future stays within a cross-track threshold of it.

**Core formulas.**

```
reference path rho with 2D points rho_xy;  Pi_rho = projection into path coordinates.
OUTPUT REPRESENTATION:   regress  tau_ac = Pi_rho( tau_xy )
   (instead of regressing tau_xy in an actor-centric frame)
K = (N + 1) * M output modes:  N goal-based spatial modes (one per candidate path)
   + 1 goal-free mode, times M temporal modes each;  categorical over the K modes.
target spatial-mode probability from a goal-following test: the future trajectory stays
   within a cross-track deviation threshold of the goal path.
```

**Maps onto us.** Two direct transfers. (a) The output representation is ours, and this is a
citable precedent that path-relative regression *improves* accuracy, not just feasibility.
(b) **Their goal-following test is a gate detector of exactly our shape** — "is this trajectory
within a cross-track band of this reference path" is our "established on final" question with the
corridor half-width as the band. They use it to build *training targets* for a mode classifier,
which is a concrete recipe for supervising our gate rather than guessing it at inference.

---

### 22. PBP — Path-based Trajectory Prediction for Autonomous Driving

- Afshar, S., Deo, N., Bhagat, A., Chakraborty, T., Shao, Y., Buddharaju, B. R., Deshpande, A.,
  Cui, H. (Motional). ICRA 2024. arXiv:2309.03750. File: `papers/pbp_pathbased_2309.03750.pdf`.

**Mechanism (brief).** The mature form of #21. Sample variable-length reference paths from the lane
graph, predict a discrete distribution over them with a path classifier, then decode the trajectory
**in the path-relative Frenét frame** conditioned on the selected path, and transform back to
Cartesian at the end. Their argument against goal-based prediction is that a 2D goal captures
intention but not the route or the speed profile; a whole reference path captures both, so a single
mode per path suffices where goal-based models need `M` modes per path.

**Core formulas.** No new closed form; the mechanism is the composition
`Cartesian -> Frenét(path) -> decode -> Frenét^-1 -> Cartesian`, with the key empirical claim that
"trajectories in the Frenét frame have much lower variance" than in Cartesian, which is why the
regression is easier.

**Maps onto us.** The cleanest modern citation for "predict in path-relative coordinates and
transform back", i.e. for our runway-frame `state` head, and for the *variance-reduction* argument
(not just the feasibility argument) in favour of it. The path-classification stage is also the
template for a **procedure classifier** — which published approach/runway, hence which corridor —
which is a better-posed version of our gate than a binary "established" flag.

---

### 23. Ye, Zhou & Wang — Improving the Generalizability of Trajectory Prediction Models with Frenét-Based Domain Normalization

- arXiv:2305.17965v3 (2023). File: `papers/frenet_domain_norm_2305.17965.pdf`.
- Secondary; included as supporting evidence rather than a mechanism.

**Mechanism (brief).** Normalise all trajectories into a Frenét frame defined by the road geometry
before training, so that the learned distribution is over road-relative motion rather than over
global layouts. Evaluated for *cross-domain generalisation* (train on one dataset/region, test on
another).

**Maps onto us.** This is the cross-airport-generalisation argument for the runway frame, which we
need for the multi-airport claim: if predictions are made in runway-relative coordinates, a model
trained on KRDU is being asked to transfer a *road-relative* distribution to KSJC, not a
geographic one. Directly relevant to the "a per-airport ADE/FDE quoted without its route mix is
not a comparison" problem in our open items. `[UNVERIFIED]`: I read the abstract and method
framing, not the full experimental protocol.

---

## A5. Aviation

### 24. Shi, Xu & Pan — 4-D Flight Trajectory Prediction With Constrained LSTM Network

- Shi, Z., Xu, M., Pan, Q. *IEEE Transactions on Intelligent Transportation Systems*
  22(11):7242-7255, 2021. DOI 10.1109/TITS.2020.3004807.
- **NOT DOWNLOADABLE: IEEE paywall.** Semantic Scholar reports `openAccessPdf` status BRONZE with
  a URL that resolves back to `doi.org/10.1109/tits.2020.3004807` (i.e. IEEE Xplore); no author
  copy, preprint, or repository version found. No arXiv id exists.

**Mechanism — from the published abstract only, NOT from the paper.** `[UNVERIFIED]` everything in
this paragraph. An LSTM trained on multi-station ADS-B, with three *phase-specific* constraints:
**Top of Climb** for the climbing phase, **way-points** for cruise, and **runway direction** for the
descending/approach phase. Data are segmented with DBSCAN and linear least squares; sliding windows
maintain trajectory continuity; the 4-D state is spatial position plus timestamp. Compared against
LSTM, Markov / weighted Markov, SVM and a Kalman filter.

**Core formulas.** `[UNVERIFIED — not obtainable]` The abstract does not state how the constraints
are imposed (loss term? inference-time filter? architectural?). **The single fact we most need
about the closest prior work in our own domain is exactly the fact the abstract does not give.**
Someone with IEEE access should read Section III/IV and record whether "runway direction" is a
penalty, a projection, or a coordinate transform.

**Maps onto us.** Nominally the nearest neighbour of this thesis: a 4-D arrival predictor with a
runway-direction constraint on final. Until the PDF is obtained, it can be cited for *motivation*
(phase-dependent constraints are the accepted framing in ATM) but **not** for a mechanism
comparison. Note the structural similarity of "Top of Climb / way-points / runway direction" to our
gating problem: they too must decide *when* each constraint applies, and they do it by phase
segmentation (DBSCAN), not by a learned gate.

---

### 25. Zhang & Chen — Phased Flight Trajectory Prediction with Deep Learning

- Zhang, K. (Lehigh) & Chen, B. (Embry-Riddle). arXiv:2203.09033 (2022); ACM TIST.
  File: `papers/phased_flight_tp_2203.09033.pdf`.

**Mechanism.** Splits prediction by flight phase: a spatio-temporal-graph LSTM (agent-agent
interactions, factor-graph decomposition) for takeoff/approach where traffic is dense, and a
simpler model en route. The constraint handling is the part relevant to us, and it is honest about
being crude: the model predicts a bivariate Gaussian per step, and the inference loop **rejection-
samples** — draw a next position, test it against the phase's angle/turn-rate limit, accept or
redraw. Constraint parameters are *estimated from the data*, not from regulation: the max climb
angle is the max over all observed takeoff trajectories, and likewise for descent and turn rate.

**Core formulas.**

```
Algorithm 1 (inference):
   fit LSTM -> per-step (mu_t, sigma_t, rho_t);  phase <- phase indicator function
   for t = T_obs+1 .. T_pred:
     while phase in {takeoff, approach}:
        sample (x_{t+1}, y_{t+1}, z_{t+1}) ~ N( mu_t, sigma_t, rho_t )
        if it satisfies limitation alpha (takeoff) or beta (approach):  accept, t <- t+1, break
        else: continue          # redraw

climb/descent angle constraint (Eq. 15):
   theta <= theta_c = max{ theta_c* = arctan( Delta h / Delta d ) }
   Delta h = | z_top - z_o |
   Delta d = 2 r arcsin( sqrt( sin^2(Delta y / 2) + cos(y') sin^2(Delta x / 2) ) )   [great circle]
   Delta x = | x_top - x |,  Delta y = | y_top - y |,  cos(y') = cos(y_top) cos(y)

rate-of-turn constraint (Eq. 16):
   omega <= omega_ROT = max{ omega*_ROT },      omega*_ROT = ( 1091 * tan(theta_b) ) / v_a
   (theta_b = bank angle, v_a = airspeed in knots; 1091 is the standard ROT constant)

IMPORTANT CAVEAT stated by the authors: "some information such as heading may be missing in
the trajectory data so that we can not calculate the constraint parameters. Therefore, the
motion constraints module will not be applied in such a case."
```

**Guarantee.** Soft-ish/inference-only. Rejection sampling enforces the limit on *accepted samples*
but is not differentiable, is not applied during training, has no termination guarantee, and is
silently skipped when the features are missing. Constraints are *empirical maxima over the dataset*,
not physical or regulatory limits, so a single outlier track loosens them.

**Cost.** Unbounded in the worst case (a `while` loop with no cap); zero training cost.

**Maps onto us.** The most useful aviation reference in this cluster precisely because it shows the
default practice we are trying to improve on, including all three of its weaknesses: (i) constraints
at inference only, so the network never learns them — the same "the filter will fix it" posture that
made our train-through-filter arm lazy, arrived at from the other direction; (ii) a **hard phase
gate** computed by a separate indicator function, with the constraint silently disabled when inputs
are missing — this is our gating problem, unsolved, and the paper's own admission is the citation
for why the gate deserves to be modelled rather than thresholded; (iii) data-derived rather than
published limits. Our project already has the published procedure geometry, which is strictly
better than a dataset maximum, and that contrast is a clean contribution statement.

---

## Cross-cutting read-out for our two paths

**What the cluster actually says, stripped of method names.**

1. **Constrain the output map, never the weights.** Every method that works (RAYEN, HardNet,
   gauge map, KKT-hPINN, Beucler) constrains `y` as a function of the network's raw output;
   Márquez-Neila's Lagrangian-on-weights approach is the one that failed to beat a soft penalty.
   Our vetoed hinge and our vetoed train-through-filter are both *not* in the working category.

2. **For an affine, per-timestep, input-dependent box, the answer is closed form and known.**
   HardNet-Aff (Eq. 5) reduces to an elementwise clamp when `A` is a selection matrix; that is
   what our runway-axis tanh approximates. The one substantive difference is Prop. 7: HardNet
   leaves *already-feasible* outputs exactly alone, tanh does not. That is a testable, cheap arm.

3. **Bijections beat filters for trainability.** The mechanism behind our measured "laziness" is
   that a saturating filter is many-to-one: distinct latent controls map to the same executed
   control, the gradient dies, and the network offloads the skill. RAYEN's ray map and Tabas &
   Zhang's gauge map are *bijections* onto the feasible set, so every latent value still has a
   distinct consequence. **The gauge map is the specific recommendation for the `control` path.**

4. **The corridor should be a coordinate, not a constraint.** Werling (Frenét), PBP, map-adaptive
   goal-based, and Beucler's residual outputs all say the same thing from four directions: choose
   coordinates in which the constraint is a bound on one scalar, then bounding it is exact rather
   than approximate. We already do this; the literature says to go further and parameterise the
   lateral offset by *distance* rather than by timestep, which decouples it from the speed profile.

5. **Nobody in this cluster solves the gating problem, and that is the gap.** Every method assumes
   `C(x)` is known from `x`. The three partial answers found: (a) HardNet's Remark 4 — let the
   gate move `b_l/b_u` continuously to `+/-inf`, so a soft gate degrades smoothly instead of
   switching; (b) map-adaptive goal-based — the cross-track-deviation test gives *supervised
   targets* for a gate/mode classifier, so the gate can be learned rather than thresholded;
   (c) PRIME / PBP / CoverNet — make the gate a *choice among procedure-feasible candidates*, so
   it is scored rather than decided. Aviation practice (Zhang & Chen, and apparently Shi et al.)
   uses a hard phase-segmentation gate and disables the constraint when inputs are missing.

6. **What is missing from the literature.** No aircraft-arrival paper found imposes a *published
   procedure* constraint by construction. Shi et al. 2021 is the closest and is paywalled; Zhang &
   Chen 2022 do inference-time rejection sampling against data-derived limits. Searches for
   "aircraft trajectory prediction constrained", "physics-informed aircraft trajectory prediction"
   and "approach trajectory prediction procedure" returned only soft/physics-informed-loss work
   (PINN-style penalties, Neural-ODE flight-dynamics models, Lyapunov regularisers) — nothing with
   a hard output layer. **That is the hole this thesis sits in, and it is worth stating plainly.**
