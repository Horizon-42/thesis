# Cluster C — constrained training, aviation-constrained TP, predict-then-optimize, constrained diffusion, phase gating

Collected 2026-09-07. Scope: literature bearing on (i) making constrained training actually
converge, (ii) aviation trajectory prediction that imposes or conditions on procedures /
performance / intent, (iii) the "predict decisions, let a solver enforce constraints" split,
(iv) constrained generative trajectory models, (v) jointly training a discrete mode/goal
decision with a continuous trajectory.

**Our situation, for the "implication" lines below.** Model predicts the rest of an arrival
from a 120 s lookback; `state` path emits positions per step, `control` path emits 64 bounded
piecewise-constant controls + a duration integrated by a differentiable point-mass rollout.
The constraint to enforce is the *published final-approach segment only* (lateral LPV corridor
+ glidepath window), active once established on final. The join point is an ATC decision the
ego-only input cannot see. Measured here already: (a) hinge penalty + primal-dual multiplier
**diverged** — target violation rate epsilon = 0.05 was unreachable (baseline 0.77) and the
multiplier only grew; fixed-weight penalty traded accuracy for violation rate; (b) a
corridor-bounded tanh output works on the `state` path; (c) an inference-time CBF filter works
on the `control` path, but training through it makes the network lean on it.

Math is plain text. `x_t` = subscript, `x^i` = superscript, `[z]_+` = max(0, z).
Everything marked **[verified]** was read out of the PDF itself; **[unverified]** was not.

Files are in `papers/`. Re-fetch with `bash download_C.sh`.

**Count: 35 papers — 33 PDFs downloaded, 2 recorded by DOI only (paywalled).**
That is above the 16–22 target; the overshoot is entirely papers that were named in the brief
plus the aviation set needed to reach 4–6 *real* procedure-aware ones. Entries for the
marginal ones are deliberately short.

---

## 1. Constrained learning: multiplier dynamics, feasible epsilon, barriers

> **Read this cluster in one line:** our divergence is not a bug, it is the documented
> behaviour of gradient ascent on a dual variable whose constraint is infeasible. Two papers
> below state it exactly — Gallego-Posada §G (dual best response is `+inf` for a violated
> constraint) and Chamon T-IT Assumption 4 / `D^ = inf` when the empirical problem is
> infeasible. The fixes are: pick a reachable level, or *learn* the level (Hounie), or damp
> the dual with P/PI control (Stooke, Sohrabi), or drop the dual entirely (Kervadec, CPO).

---

### 1.1 Chamon & Ribeiro 2020 — Probably Approximately Correct Constrained Learning
NeurIPS 2020. arXiv:2006.05487. → `papers/chamon2020_pac_constrained_learning.pdf`

**Mechanism.** Asks when a constrained statistical learning problem (P-CSL) — minimise
E[l_0] subject to E[l_i] <= c_i — can be solved from samples alone. Shows any PAC-learnable
class is also PAC *constrained* learnable via a constrained ERM rule, but that rule needs a
feasible point of a non-convex program, which is itself hard. The practical result: the
**empirical dual** problem is also a PAC constrained learner, so you can train by solving a
sequence of *unconstrained* problems (minimise the Lagrangian, ascend the multipliers) and
still get generalisation guarantees on the constraints, not just on cost+penalty. **[verified]**

**Core formulas.** Empirical Lagrangian and empirical dual: **[verified]**
```
L^(theta, mu, lambda_j)
    = (1/N_0) sum_{n_0} l_0(f_theta(x_n0), y_n0)
    + sum_{i=1..m} mu_i [ (1/N_i) sum_{n_i} l_i(f_theta(x_ni), y_ni) - c_i ]
    + ( pointwise-constraint terms with duals lambda_{j,n_j} )

D^*  =  max_{mu >= 0, lambda_j >= 0}   min_{theta in R^p}  L^(theta, mu, lambda_j)
```
The guarantee rests on three assumptions; **Assumption 3 is the load-bearing one**: there
exists theta' that is **strictly feasible** for (P-CSL) with constraints `c_i - M*nu`, and for
each sampled dataset a theta'' strictly feasible for the empirical problem. **[verified]**

**Guarantee / cost.** Statistical: a bound on the empirical duality gap giving *near-optimal
and near-feasible* solutions; not a per-sample hard guarantee. Cost is a dual-ascent outer
loop around ordinary training.

**Implication for us.** This is the theorem our diverged run violated. The whole PAC guarantee
is conditioned on a strictly feasible point existing *at the level c you wrote down*. With
baseline violation 0.77 and epsilon = 0.05 there is no such point in our hypothesis class, so
neither the bound nor the convergence claim applies — the dual is simply unbounded. Fixing the
level so that a strictly feasible model plausibly exists is a precondition, not a tuning knob.

---

### 1.2 Chamon, Paternain, Calvo-Fullana, Ribeiro 2023 — Constrained Learning with Non-Convex Losses
IEEE Trans. Information Theory. arXiv:2103.05134, DOI 10.1109/TIT.2022.3187948.
→ `papers/chamon2023_constrained_learning_nonconvex_losses.pdf`

**Mechanism.** Extends 1.1 to non-convex losses (i.e. actual deep nets). Proves the
*functional* constrained learning problem has **zero duality gap** under mild conditions even
non-convex, then bounds the two gaps you pay for practice: the **parameterisation gap** (using
a finite-dimensional f_theta instead of the function class) and the **empirical gap** (samples
instead of distributions). The composite bound (their Theorem 1) is what licenses training in
the empirical dual domain and yields a practical dual-ascent algorithm with near-optimality
and near-feasibility. **[verified]**

**Core formulas / conditions.** The infeasibility statement is explicit: **[verified]**
```
D^*  =  +inf   whenever the empirical problem (P-ECRM) is infeasible,
                i.e. for all theta in Theta there exists i with
                (1/N_i) sum_{n_i} l_i(f_theta(x_ni), y_ni)  >  c_i
```
Assumption 4 (strict feasibility with a margin xi > 0): there exist theta', theta^ in Theta with
```
E_{D_i}[ l_i(f_theta'(x), y) ]                 <=  c_i - M*nu - xi
(1/N_i) sum_{n_i} l_i(f_theta^(x_ni), y_ni)    <=  c_i - xi          for all i = 1..m
```
The final bound depends on sample count *and* on "the difficulty of the learning task both in
terms of the parametrization used and how hard the constraints are" — the sensitivity /
shadow-price reading of the multiplier. **[verified]**

**Guarantee / cost.** Statistical (near-optimal + near-feasible, with explicit gaps); no hard
per-trajectory feasibility. Cost: one dual variable per constraint, one dual step per epoch.

**Implication for us.** `D^* = +inf` on an infeasible empirical problem *is* our observed
multiplier blow-up, stated as a theorem rather than a symptom. It also tells us the diagnosis
is cheap: before training, check whether *any* model in the class reaches epsilon on the
training set. If not, the dual will diverge no matter how the step size is tuned.

---

### 1.3 Hounie, Ribeiro, Chamon 2023 — Resilient Constrained Learning
arXiv:2306.02426. → `papers/hounie2023_resilient_constrained_learning.pdf`

**The single most on-point paper in this cluster for our failure.** It solves exactly the
"my epsilon is unreachable" problem by *learning the relaxation* instead of fixing it.

**Mechanism.** Introduces a relaxation vector u >= 0 that loosens each constraint to
`E[l_i] <= u_i`, and trades the benefit of relaxing (the perturbation function P*(u) drops)
against a **relaxation cost** h(u) that penalises drifting away from the problem you meant.
The equilibrium is where the marginal gain of relaxing equals the marginal cost — and the
marginal gain of relaxing constraint i *is* the multiplier lambda_i. So the relaxation grows
precisely for the constraints whose multipliers are large, i.e. the ones that are too tight.
The three variables (theta, u, lambda) are updated in one loop. **[verified]**

**Core formulas.** Relaxed problem and perturbation function: **[verified]**
```
P*(u) = min_{theta}  E_{(x,y)~D_0}[ l_0(f_theta(x), y) ]
        s.t.         E_{(x,y)~D_i}[ l_i(f_theta(x), y) ]  <=  u_i ,  i = 1..m
```
Resilient equilibrium (Def. 1): u* such that `-grad h(u*) = p_{u*} in d P~*(u*)`, i.e. the
relaxation cost gradient equals minus the subgradient of the perturbation function.

Algorithm 1 (the three-way update; **eta, eta_u, eta_lambda > 0**): **[verified]**
```
# primal: N SGD steps on the Lagrangian
theta_{n+1} = theta_n - eta * grad_theta [ l_0(f(x_n0), y_n0)
                                           + sum_i lambda_i * l_i(f(x_ni), y_ni) ]

# relaxation update  <-- the new part
u^(t)       = [ u^(t-1)  -  eta_u * ( grad h(u^(t-1))  -  lambda^(t-1) ) ]_+

# dual update, now against the LEARNED level u^(t)
lambda_i^(t) = [ lambda_i^(t-1)
                 + eta_lambda * ( (1/N) sum_n l_i(f_{theta^(t-1)}(x_ni), y_ni) - u_i^(t) ) ]_+
```
Typical relaxation cost `h(u) = alpha * ||u||_2^2` (so `grad h(u) = 2*alpha*u`); the paper also
uses `h(u) = u^2/2` in the illustrative figure. Smaller alpha permits larger relaxations.
A **linear** cost `h(u) = sum_i gamma_i u_i` is shown to be equivalent to plain fixed-level
constrained learning — i.e. the quadratic (strictly convex) cost is what makes the level
adaptive at all. **[verified]**

**Guarantee / cost.** Statistical, with a generalisation theorem (their Thm. 1) for the
sample-based version; the paper also bounds where u* can land. Cost is one extra variable
per constraint and one extra gradient step — negligible.

**Implication for us.** This is the principled replacement for "pick epsilon = 0.05 and hope".
Read the update: `u` grows exactly when `lambda > grad h(u)`, i.e. when the dual price of the
constraint exceeds what we said a relaxation is worth. Our diverging lambda would have been
absorbed into a growing u instead of running away, and the converged `u*` would be an
*empirical read-out of the reachable violation rate* on our fleet — the number we currently
lack. Strong candidate to run as the next constrained-training arm, with the corridor
constraint per flight and alpha swept to trace the accuracy-vs-violation frontier.

---

### 1.4 Fioretto, Van Hentenryck, Mak, Tran, Baldo, Lombardi 2020 — Lagrangian Duality for Constrained Deep Learning
arXiv:2001.09394. → `papers/fioretto2020_lagrangian_duality_constrained_dl.pdf`

**Mechanism.** A Lagrangian Dual Framework (LDF) for supervised learning under constraints,
built on the *violation-based* (augmented-Lagrangian style) relaxation rather than the
satisfiability-based one: the penalty uses `max(0, g_i(y^))` so satisfied constraints
contribute nothing. Multipliers are updated once per **epoch** by subgradient dual ascent, and
the primal is trained with the multipliers held fixed within the epoch. Applied to constrained
predictors (OPF, transprecision computing) where constraints couple predictions. **[verified]**

**Core formulas.** Violation-based Lagrangian loss and the epoch-level dual ascent (their
Algorithm 1 line 6): **[verified]**
```
L_lambda(y^_l, y_l, d_l)  =  L(y^_l, y_l)  +  sum_{i=1..m} lambda_i * nu( g_i(y^_l, d_l) )

lambda_i^{k+1}  <-  lambda_i^k  +  s_k * sum_{l=1..n} nu_i( g_i(y^_l, d_l) )     for all i
```
where `nu(.)` returns the violation degree: `nu(g) = max(0, g)` for inequalities and `|h|` for
equalities, contrasted explicitly with the classical relaxation `f(y) + lambda_h h(y) +
lambda_g g(y)` which uses satisfiability degrees (negative slack pushes the multiplier down).
`s_k` is the Lagrangian step size, distinct from the optimizer step size alpha. **[verified]**

**Guarantee / cost.** Soft. No feasibility certificate; the claim is better accuracy/violation
trade-offs than fixed-weight penalties. Cost: one extra hyperparameter (s_k) and an epoch-scale
dual loop.

**Implication for us.** Two concrete details worth copying. First, the **epoch-scale** dual step
(not per-batch) — our violation-rate signal is noisy per batch, and a per-batch dual step
amplifies that noise into the multiplier. Second, note the failure mode this design *cannot*
avoid: because `nu = max(0, g)` is non-negative, the multiplier is **monotonically
non-decreasing** whenever the constraint is ever violated. On an unreachable epsilon that is a
guaranteed ramp — which is precisely the shape we observed. Violation-based relaxation needs
either a reachable level (1.3) or dual restarts (1.9).

---

### 1.5 Stooke, Achiam, Abbeel 2020 — Responsive Safety in RL by PID Lagrangian Methods
ICML 2020. arXiv:2007.03964. → `papers/stooke2020_pid_lagrangian.pdf`

**Mechanism.** The key reframing: **the classical Lagrange multiplier update is integral
control** on the constraint, and ill-tuned integral control is exactly what produces the
90-degree phase lag, oscillation and overshoot seen in safe RL. Casting constrained learning as
a first-order dynamical system with lambda as the control input lets you add **proportional**
(hastens response, damps oscillation) and **derivative** (acts *in anticipation* of violations)
terms. Setting K_P = K_D = 0 recovers the traditional method. The integral term is retained
because it is what removes steady-state violation at convergence. **[verified]**

**Core formulas.** Constrained learning as a dynamical system (their eqs. 18–20): **[verified]**
```
theta_{k+1} = F(theta_k, lambda_k) = f(theta_k) + g(theta_k) * lambda_k
y_k         = J_C(pi_{theta_k})
lambda_k    = h(y_0, ..., y_k, d)
   with  f(theta_k) = theta_k + eta * grad_theta J(pi_{theta_k})
         g(theta_k) = -eta * grad_theta J_C(pi_{theta_k})
```
**Algorithm 2, PID-Controlled Lagrange Multiplier — transcribed exactly:** **[verified]**
```
1: Choose tuning parameters:  K_P, K_I, K_D >= 0
2: Integral:        I <- 0
3: Previous Cost:   J_{C,prev} <- 0
4: repeat at each iteration k
5:     Receive cost J_C
6:     Delta   <-  J_C - d
7:     Partial <-  ( J_C - J_{C,prev} )_+          # projected: acts against INCREASES only
8:     I       <-  ( I + Delta )_+
9:     lambda  <-  ( K_P * Delta  +  K_I * I  +  K_D * Partial )_+
10:    J_{C,prev} <- J_C
11:    return lambda
```
Note `I` is itself projected to be non-negative, and the derivative term is one-sided: it
"acts against increases in cost but does not impede decreases". Separate value and cost-value
approximators are used because lambda may change rapidly. **Objective re-scaling** for step-size
consistency when lambda is large (used for their Lagrangian baselines too): **[verified]**
```
theta*(lambda) = argmax_theta  J - lambda * J_C
               = argmax_theta  (1 / (1 + lambda)) * ( J - lambda * J_C )
```

**Guarantee / cost.** Soft — no feasibility certificate; the claim is dramatically reduced
violation *during* training and hyperparameter robustness. Cost: two scalars (K_P, K_D) and one
stored previous cost.

**Implication for us.** Our diverged run is pure integral control with no damping: `I` only
accumulates while the constraint is violated, and with an unreachable epsilon `Delta > 0`
forever, so `I -> inf`. Two things to take. (1) If we keep a fixed epsilon, a P term makes the
multiplier respond to the *current* violation rather than its running sum, which bounds the
response for a constraint that is merely hard rather than impossible. (2) The `1/(1+lambda)`
re-scaling is worth applying to our rollout loss regardless — with a large multiplier the
effective primal step size on our point-mass rollout is not what the LR schedule says.
Caveat: no P/D gain rescues a genuinely unreachable level; the integral term still ramps.

---

### 1.6 Sohrabi, Ramirez, Zhang, Lacoste-Julien, Gallego-Posada 2024 — On PI Controllers for Updating Lagrange Multipliers in Constrained Optimization (νPI)
ICML 2024. arXiv:2406.04558. → `papers/sohrabi2024_nupi_multiplier_controller.pdf`

The brief asked for "Sohrabi et al. 2024, constrained learning survey". **There is no such
survey** — the actual Sohrabi et al. 2024 is this νPI paper, and it is the direct successor to
Stooke for *supervised* (non-RL) constrained deep learning. It is the update rule Cooper (1.10)
ships. **[verified]**

**Mechanism.** Generalises the PI controller for the multiplier by putting an **exponential
moving average of the error signal in the proportional term** (a plain PI controller is
recovered at nu = 0; a PID with kappa_d = 0). Explains theoretically and empirically why
ordinary momentum on the dual (Polyak/Nesterov) does *not* fix gradient-descent-ascent's
oscillations, and proves νPI strictly generalises those momentum methods. **[verified]**

**Core formulas. Algorithm 1, νPI update — transcribed exactly:** **[verified]**
```
Args: EMA coefficient nu, proportional gain kappa_p, integral gain kappa_i;
      initial conditions xi_0 and theta_0.        (theta here = the MULTIPLIER)
1: Measure current system error  e_t                 # e_t = constraint violation g(x_t)
2: xi_t      <-  nu * xi_{t-1}  +  (1 - nu) * e_t                     for t >= 1
3: theta_{t+1} <- theta_0  +  kappa_p * xi_t  +  kappa_i * sum_{tau=0..t} e_tau
```
Equivalent recursive form (their Lemma 2, eqs. 5–6): **[verified]**
```
theta_1     = theta_0 + kappa_i * e_0 + kappa_p * xi_0
theta_{t+1} = theta_t + kappa_i * e_t + kappa_p * (xi_t - xi_{t-1})        for t >= 1
```
followed by projection to `>= 0` for inequality multipliers. Momentum equivalences (Thm. 1):
`UnifiedMomentum(alpha, beta != 1, gamma)` is νPI with `nu <- beta`, `kappa_i <- alpha/(1-beta)`,
`kappa_p <- -alpha*beta/(1-beta)^2 * [1 - gamma*(1-beta)]`, and
`kappa_p^Nesterov = -alpha*beta^2/(1-beta)^2 <= 0` regardless of beta — i.e. Nesterov can only
ever supply a *non-positive* proportional gain, which is why momentum does not help. **[verified]**

**Behaviour (their §4.3 modes).** νPI increases the multiplier *faster* than gradient ascent
when the violation is large; and when the violation has improved enough relative to the
historical average (`e_t in [0, kappa*xi_{t-1}]`, still infeasible) νPI **decreases** the
multiplier where plain GA would keep increasing it. That is the anti-overshoot mechanism.
The paper is explicit about the trade-off: "An insufficient increase of the multiplier will
cause the constraint to be ignored, while an excessively large value ..." **[verified]**

**Guarantee / cost.** Soft, dynamics-level; no feasibility certificate. Cost: two gains + one
EMA state per constraint. Ships in Cooper, so ~zero implementation cost for us.

**Implication for us.** If we keep a fixed epsilon this is the update to use, not vanilla dual
ascent — and its Mode C behaviour (cut the multiplier when violation is *improving* even
while still infeasible) is exactly the damping our run lacked. Pair it with a reachable epsilon
from 1.3; νPI controls the *dynamics*, it does not make an infeasible level feasible.

---

### 1.7 Kervadec, Dolz, Yuan, Desrosiers, Granger, Ben Ayed 2022 — Constrained Deep Networks: Lagrangian Optimization via Log-Barrier Extensions
EUSIPCO 2022, pp. 962–966. arXiv:1904.04205, DOI 10.23919/EUSIPCO55093.2022.9909927.
→ `papers/kervadec2022_log_barrier_extensions.pdf`

**Mechanism.** Argues that for CNNs with millions of parameters the theoretical advantages of
Lagrangian optimisation over penalties do not materialise, so constrained CNNs are handled with
penalties in practice. Proposes a middle road: **log-barrier extensions**, a sequence of
*unconstrained* losses that approximate Lagrangian optimisation via **implicit** dual variables
— no explicit dual step, no projection, and crucially **no need for an initial feasible
solution** (which standard interior-point log barriers require and which is itself an
intractable problem for a CNN). **[verified]**

**Core formula.** The extension (their eq. 3) — the entire trick is the linear continuation
below the threshold, which makes the domain all of R rather than just the feasible set: **[verified]**
```
psi~_t(z)  =   -(1/t) * log(-z)                        if  z <= -1/t^2
               t*z - (1/t)*log(1/t^2) + 1/t            otherwise
```
Total loss (their eq. 2) is `sum over constraints of psi~_t( f_i(s_theta^n) )` added to the task
loss, with `t -> +inf` making psi~_t a smooth approximation of the hard indicator H. **[verified]**

**Guarantee / cost.** Yields an **upper bound on the duality gap**, generalising the standard
log-barrier result, hence sub-optimality certificates for feasible solutions in the convex
case; explicitly *not* guaranteed sub-optimal for non-convex problems. Cost: one schedule on
`t`, no dual variables.

**Implication for us.** The most attractive property here is that it removes the dual variable
entirely — there is nothing to diverge. The `t` schedule is a *warm-up knob with a monotone
interpretation* (start soft, tighten), which maps cleanly onto our join-point problem: keep `t`
small early while the model is still learning where final approach begins, raise it later. The
honest caveat: as `t` grows the barrier on a *violated* constraint grows linearly with slope
`t`, so an unreachable epsilon reappears as an exploding gradient rather than an exploding
multiplier. Same disease, different symptom — the level still has to be reachable.

---

### 1.8 Achiam, Held, Tamar, Abbeel 2017 — Constrained Policy Optimization (CPO)
ICML 2017. arXiv:1705.10528. → `papers/achiam2017_cpo.pdf`

**Mechanism.** A trust-region method for constrained MDPs that guarantees **near-constraint
satisfaction at every iterate**, not just at convergence. The decisive design choice for us:
CPO does **not** carry a stateful multiplier. It linearises objective and constraints, models
the KL trust region as a quadratic, and **solves for fresh dual variables from scratch at each
update**. It contrasts this explicitly with primal-dual (PDO), where duals are stateful and
learned concurrently and only the converged policy is guaranteed feasible. **[verified]**

**Core formulas.** The CPO update (their eq. 10), its convex approximation (11), the dual (12),
the primal solution (13), and the infeasible-recovery step (14): **[verified]**
```
# (11) approximation of the CPO update, with g = grad objective, b_i = grad constraint i,
#      H = Hessian of the KL divergence, c_i = J_{C_i}(pi_k) - d_i
theta_{k+1} = argmax_theta  g^T (theta - theta_k)
              s.t.  c_i + b_i^T (theta - theta_k) <= 0        i = 1..m
                    (1/2) (theta - theta_k)^T H (theta - theta_k)  <=  delta

# (12) dual, with B = [b_1..b_m], c = [c_1..c_m]^T, r = g^T H^-1 B, S = B^T H^-1 B
max_{lambda >= 0, nu >= 0}
    -(1/(2*lambda)) * ( g^T H^-1 g - 2 r^T nu + nu^T S nu )  +  nu^T c  -  lambda*delta/2

# (13) primal solution
theta* = theta_k + (1/lambda*) * H^-1 * ( g - B nu* )

# (14) RECOVERY step when (11) is infeasible: purely decrease the constraint
theta* = theta_k - sqrt( 2*delta / (b^T H^-1 b) ) * H^-1 * b
```
followed by a backtracking line search to enforce the sampled constraints. Contrast, their
eq. (16), the PDO update CPO is arguing against: **[verified]**
```
nu_{k+1} = ( nu_k + alpha_k * ( J_C(pi_k) - d ) )_+
```
Worst-case violation bound (Prop. 2): `J_{C_i}(pi_{k+1}) <= d_i + sqrt(2*delta)*gamma*eps /
(1-gamma)^2`. **[verified]**

**Guarantee / cost.** Near-hard per-iterate (a bounded worst-case violation, plus a line search
that enforces the sampled surrogate constraints), and an explicit **recovery direction for the
infeasible case**. Cost: conjugate-gradient Hessian-vector products per update; expensive.

**Implication for us.** Two transferable ideas even though we are not doing RL. (1) *Stateless
duals.* Recomputing the multiplier from the current violation each step cannot diverge by
construction — this is the structural version of Stooke's proportional term, and it is the
cleanest answer to "the multiplier only grew". (2) *An explicit infeasible branch.* CPO does
not pretend infeasibility cannot happen; when the constraint set and trust region do not
intersect it switches to a step that purely reduces violation. Our training loop had no such
branch — it kept trying to trade off against a constraint it could never meet.

---

### 1.9 Gallego-Posada, Ramirez, Erraqabi, Bengio, Lacoste-Julien 2022 — Controlled Sparsity via Constrained Optimization
NeurIPS 2022. arXiv:2208.04425. → `papers/gallegoposada2022_controlled_sparsity.pdf`

**Contains the cleanest statement anywhere of why our multiplier diverged.**

**Mechanism.** Replaces penalty-tuning for model sparsity with an explicit constraint
`g_const(phi_g) <= epsilon_g` per gate group, solved by GDA on the Lagrangian. Observes that
accumulated violations keep affecting the dynamics *even after a constraint becomes satisfied*
(over-regularisation, pushing the primal into the interior of the feasible set), and proposes
**dual restarts**: zero the multiplier the moment the constraint is satisfied rather than
waiting for negative gradients to walk it down. **[verified]**

**Core formulas.** Lagrangian and standard GDA (their eqs. 4–5): **[verified]**
```
L(theta~, phi, lambda_co) = f_obj(theta~, phi)
                          + sum_{g=1..G} lambda_co^g * ( g_const(phi_g) - epsilon_g )

lambda^_{t+1} = lambda_co^t + eta_dual * ( g_const(phi_g^t) - epsilon_g )_{g=1..G}
```
**Dual restart (their eq. 6) — transcribed exactly:** **[verified]**
```
lambda_co^{t+1}[g]  =   max( 0, lambda_co^t[g] + eta_dual * ( g_const(phi_g^t) - epsilon_g ) )
                            if  g_const(phi_g^t) > epsilon_g
                        0
                            otherwise
```
**And the theoretical characterisation (their Appendix G) — the sentence that explains our
run.** The dual player's best response to a fixed primal is a linear program with a trivial
solution: **[verified]**
```
lambda_co^BR(theta~, phi) = argmax_{lambda_co >= 0}  f_obj + lambda_co * ( g_const(phi) - eps )

  constraint satisfied  ( g_const - eps < 0 )  ->  lambda_co^BR = 0
  satisfied with equality                      ->  lambda_co^BR = R^+
  constraint VIOLATED   ( g_const - eps > 0 )  ->  lambda_co^BR = +inf
```
The paper notes dual restarts implement the best response for the *satisfied* case, and that
"the same reasoning cannot be applied to the case of violated constraints: stability and
overflow issues render a choice of `inf` for a Lagrange multiplier to be impractical for a
numerical implementation." **[verified]**

**Guarantee / cost.** Soft (GDA on a non-convex-concave problem, no convergence guarantee);
the empirical claim is reliably hitting pre-set sparsity targets without accuracy loss. Cost:
a one-line change to the dual update.

**Implication for us.** Read the last box against our setup: with epsilon = 0.05 unreachable,
our corridor constraint was violated at *every* step, so the dual player's best response was
`+inf` at every step and gradient ascent was faithfully walking toward it. Our divergence was
not instability — it was correct optimisation of an unbounded dual. Dual restarts do not help
here (they only fire when the constraint is satisfied, which never happened); they *would*
matter once we adopt a reachable level, to stop over-regularising the flights that already
comply. Also worth noting for the νPI comparison in 1.6: νPI is reported to recover feasible
solutions with minimal overshoot, whereas dual restarts avoid overshoot but land "slightly
infeasible".

---

### 1.10 Gallego-Posada, Ramirez, Hashemizadeh, Lacoste-Julien 2025 — Cooper: A Library for Constrained Optimization in Deep Learning
arXiv:2504.01212. → `papers/gallegoposada2025_cooper_library.pdf`

**Mechanism.** PyTorch library implementing Lagrangian-based first-order schemes for
`min f(x) s.t. g(x) <= 0, h(x) = 0`, designed for mini-batch gradient estimates. User writes a
`ConstrainedMinimizationProblem` holding `Constraint` objects each owning a `Multiplier`;
`compute_cmp_state` returns objective + violations; a `CooperOptimizer` wraps primal and dual
optimizers and `roll()` does zero_grad → build Lagrangian → backward → primal & dual steps. **[verified]**

**Core formulas.** Lagrangian (eq. 2) and simultaneous GDA (eqs. 3a–3c): **[verified]**
```
min_x max_{lambda >= 0, mu}   L(x, lambda, mu)  =  f(x) + lambda^T g(x) + mu^T h(x)

x_{t+1}      <- PrimalOptimizerStep( x_t,      grad_x L(x_t, lambda_t, mu_t) )
lambda_{t+1} <- DualOptimizerStep(   lambda_t, g(x_t) )_+
mu_{t+1}     <- DualOptimizerStep(   mu_t,     h(x_t) )
```
Also implements: Augmented Lagrangian, Quadratic Penalty, the **proxy-Lagrangian** of Cotter
et al. 2019 (for **non-differentiable constraints**), alternating GDA, extragradient, and
**νPI (Sohrabi et al. 2024) "for improving the multiplier dynamics"**. **[verified]**

**Guarantee / cost.** None of its own — it is tooling. Explicit caveat in the paper: "We
recommend the use of Cooper unless specialized algorithms are available for a given
application." **[verified]**

**Implication for us.** Practical: rather than hand-rolling another primal-dual loop, this gives
νPI, dual restarts, augmented Lagrangian and proxy-Lagrangian behind one interface, so the
"which multiplier rule" question becomes an ablation instead of an implementation project. The
**proxy-Lagrangian** is the specifically interesting one: our real quantity of interest is a
*violation rate* (a 0/1 count per flight), which is non-differentiable; today we surrogate it
with a hinge. Proxy-Lagrangian lets the constraint be stated on the true rate while the primal
is driven by the differentiable surrogate — which is the right way to state "epsilon = 0.05 of
flights may leave the corridor".

---

### 1.11 Sangalli, Erdil, Hoetker, Donati, Konukoglu 2021 — Constrained Optimization to Train Neural Networks on Critical and Under-Represented Classes
NeurIPS 2021. arXiv:2102.12894. → `papers/sangalli2021_constrained_opt_underrepresented.pdf`

Short entry — included as the augmented-Lagrangian representative the brief asked for.

**Mechanism.** Casts imbalanced binary classification as constrained optimisation with a
Mann-Whitney-statistic constraint that forces each critical-class output above all negatives by
a margin delta (satisfying it maximises AUC), and solves it with a textbook **Augmented
Lagrangian method** — penalty term added to the *Lagrangian* rather than to the objective, so
mu need not be driven to infinity as in the pure quadratic penalty method. **[verified]**

**Core formulas (their eqs. 2–4).** **[verified]**
```
L_mu(theta, lambda) = F(theta) + mu * sum_{i=1..m} c_i(theta)^2 + sum_{i=1..m} lambda_i * c_i(theta)

iterate:   max_{lambda^k}  min_{theta in Theta}  L_{mu^k}(theta, lambda^k)
multiplier update:   lambda_i^{k+1} = lambda_i^k + mu * c_i(theta)
penalty schedule:    0 < mu^k <= mu^{k+1},  mu^k -> inf   (pre-selected or generated)

constraint:  sum_{k=1..|n|} max( 0, -( f_theta(x_j^p) - f_theta(x_k^n) ) + delta ) = 0,
             for each critical-class sample j
```

**Guarantee / cost.** Soft; the ALM claim is stability (no need for mu -> inf, no convexity
assumption) rather than feasibility. Cost: one penalty schedule plus multipliers.

**Implication for us.** The augmented (quadratic) term is what distinguishes this from our
diverged run: near an infeasible level the quadratic term dominates and keeps the primal
problem well-posed even while lambda is still moving, so the *primal* stays trainable. Note
the constraint here is an **equality at zero violation** — the analogue of our epsilon = 0, and
they can pose it only because the margin formulation makes it reachable. Same lesson again.

---

## 2. Aviation: constrained / procedure-aware / physics-informed trajectory prediction

> **Read this cluster in one line — and this is the headline finding of the whole collection:**
> aviation TP splits cleanly into *physics-as-generator* (the network emits parameters of BADA
> or a neural ODE, which integrates them — hard energy feasibility, but **en-route/vertical
> only, terminal procedures deliberately excluded**) and *learned-sequence* (transformers /
> GNN-CVAEs on ADS-B — no constraint of any kind). **Not one of the ten papers below enforces a
> lateral corridor or a glidepath window, and not one addresses the establishment / join-point
> problem.** The only hard published-regulation constraint found anywhere is Hodgkin's post-hoc
> rejection of samples violating the UK legal 500 ft/min minimum climb rate — a scalar rate
> bound, not a geometry. This gap is unoccupied and is a positioning claim for the thesis.

---

### 2.1 Shi, Xu, Pan 2021 — 4-D Flight Trajectory Prediction With Constrained LSTM Network
IEEE Trans. Intelligent Transportation Systems 22(11):7242–7255, Nov 2021.
DOI 10.1109/TITS.2020.3004807 (ieeexplore.ieee.org/document/9136843).

**NOT DOWNLOADABLE**: IEEE paywall; probed 2026-09-07 and no preprint exists on arXiv or an
author page. Cited by DOI only.

**[unverified — from abstract and citation metadata only, not read.]** Motivation is that sparse
waypoints and shared airways make flight TP structurally different from road traffic; the
proposal is a "constrained" LSTM for 4-D prediction. **The nature of the constraint (architectural,
loss-term, or airway-geometry conditioning) could not be established without the full text**, so
nothing about its mechanism should be asserted in the thesis on the strength of this entry.
If it turns out to be the closest prior art, it needs an interlibrary/institutional copy.

---

### 2.2 Pang, Zhao, Yan, Liu 2021 — Data-driven trajectory prediction with weather uncertainties: A Bayesian deep learning approach
Transportation Research Part C: Emerging Technologies 130:103326, Sept 2021.
DOI 10.1016/j.trc.2021.103326.

**NOT DOWNLOADABLE**: Elsevier paywall (HTTP 403); probed 2026-09-07, no arXiv preprint under
this or a variant title. Cited by DOI only.

**[unverified — from abstract only, not read.]** Bayesian deep learning for TP that treats
weather as the dominant uncertainty source, using dropout as approximate variational inference
to produce predictive distributions. Relevant to us as the standard citation for *probabilistic*
aviation TP, not for constraints — no evidence it imposes procedures or performance limits.

---

### 2.3 Hodgkin, Pepper, Thomas 2026 — Conditioning Aircraft Trajectory Prediction on Meteorological Data with a Physics-Informed Machine Learning Approach
AIAA SciTech 2026 Forum. arXiv:2601.03152. → `papers/hodgkin2026_physics_informed_met_tp.pdf`

**The strongest "physics as hard container" example in the set, and the only one with a hard
regulatory filter.**

**Mechanism.** The network never emits positions. It consumes a 14-element *context* vector
(operator, origin airport, UK intent_code, flight_type, month, FL min/max/range, plus wind and
ISA-temperature statistics from Met Office forecasts) and outputs a **distribution over fPCA
weights** y = [alpha, beta] which reconstruct two continuous functions of altitude: thrust
T_HR(h) and calibrated airspeed V_CAS(h). Those two functions are fed **into BADA**, which
integrates the total-energy equation to produce the trajectory. Energy consistency and monotone
climb are therefore enforced *mechanically* by the physics model, not by a penalty. Published
procedures enter only as coarse categorical conditioning (origin, UK Intention Code, operator) —
explicitly chosen as a proxy for the filed route. A **hard post-hoc rejection filter** discards
any sampled trajectory whose ROC drops below the UK legal minimum 500 ft/min at any radar blip. **[verified]**

**Core formulas.** **[verified]**
```
# (1) BADA total-energy / ROCD -- the integrator
dh/dt = ((T - dT)/T) * [ (T_HR - D) * V_TAS / (m * g0) ] * f(M)

# (2a,2b) reduced-order representation: what the network actually predicts
T_HR^(h)  = mu_THR(h) + sum_{i=1..n_alpha} alpha_i * phi_i(h)
V_CAS^(h) = mu_V(h)   + sum_{j=1..n_beta}  beta_j  * psi_j(h)

# (3) orthonormality
int phi_i(h) phi_j(h) dh = delta_ij ;   int psi_k(h) psi_l(h) dh = delta_kl

# (4-7) GP surrogate over the weights
p(y_i | x) ~ N( mu^(i)(x), sigma^(i)(x, x') )
mu^(i)(x)     = mu_0^(i)(x) + K(x',X)^T K(X,X)^-1 ( Y^(i) - mu_0^(i)(x) )
sigma^2 (i)(x) = K(x',x') - K(x',X)^T K(X,X)^-1 K(x',X)

# (10) scoring rule
CRPS(tau, F(t)) = int_{-inf}^{inf} ( F(t) - 1(t - tau) )^2 dt
```
Deep-ensemble members trained on NLL (stated in prose; no equation given). **[verified]**

**Data + horizon.** UK Mode-S radar, 60 days over May 2019–Apr 2020, London FIR, **above FL150
only — terminal-airspace procedures deliberately excluded**; 10 aircraft types (B738: 30,182
trajectories). **Climb only.** No lookback window — it is generative conditioning, not
seq2seq. GP 22.5% / deep-ensemble 20% mean improvement over an unconditioned Gaussian baseline
across 6 metrics × 10 types. Rejection rates GP 2.8%, DE 3.2%, baseline 6.8%. **[verified]**

**Guarantee.** **Hard** for aircraft-performance/energy physics (BADA is the generator) and
**hard** for the 500 ft/min legal minimum (post-hoc filter). **Conditioning only** for procedures.

**Implication for us.** Structurally the closest published analogue to our `control` path: emit
parameters of a physics integrator rather than positions, and feasibility of the *dynamics* comes
free. Two lessons. (1) The reduced-order (fPCA) output is a smart middle ground — far fewer
degrees of freedom than our 64 controls, and the basis is learned from the fleet. (2) The
rejection filter is the honest precedent for our situation: when a regulation cannot be built
into the generator, Hodgkin **rejects samples** rather than penalising them, and reports the
rejection rate as a headline number. That is a legitimate, publishable way to handle the corridor
if constrained training keeps failing — and the rejection rate *is* the violation rate.

---

### 2.4 Pepper & Thomas 2023 — Learning Generative Models for Climbing Aircraft from Radar Data
J. Aerospace Information Systems 21(6):474–481, 2024, DOI 10.2514/1.I011359. arXiv:2309.14941.
→ `papers/pepper2023_generative_climbing_bada.pdf`

**Mechanism.** Direct predecessor of 2.3, without conditioning. Inverts BADA at every radar blip
to solve for an *effective thrust* T_HR(h), represents the population of fitted thrust functions
by fPCA, and fits one multivariate normal to the fPCA weights. Generation = sample w →
reconstruct T_HR(h) → run BADA. The learned object is a **functional correction to the thrust
inside the physics model**. Adds analytic 95% confidence *bounds* on climb performance, obtained
geometrically from the confidence ellipsoid in weight space at the cost of two extra BADA
evaluations. **[verified]**

**Core formulas.** **[verified]**
```
# (1) BADA total energy
(T_HR - D) * V_TAS = m * ( g0 * dh/dt + V_TAS * dV_TAS/dt )

# (4) fPCA expansion of the effective thrust
T_HR^(h_j) = mu(h_j) + sum_{i=1..n} w_i * phi_i(h_j) ,   j = 1..n_g

# (6) 95% confidence ellipsoid in weight space
(w - mu_w)^T Sigma_w^-1 (w - mu_w)  <=  chi^2_0.95

# (8) analytic best/worst-case weights on the hyperplane a^T w = const, a_i = phi_i(h_k)
w^_l = mu_w - ( sqrt(chi^2_0.95) * c .* n^ ) ;   w^_u = mu_w + ( sqrt(chi^2_0.95) * c .* n^ )
     with c = diag(Sigma_w)^(1/2),  n = a .* c,  n^ = n / ||n||_2
```
**There is no neural network and no loss function in this paper** — fitting is least squares plus
Gaussian MLE. **[verified]**

**Data + horizon.** UK Mode-S radar, 707,236 flights, Jul–Sep 2019. **Climb only, FL150→FL325**,
a band chosen because trajectories there "are less likely to be affected by local operational
procedures". MAE 26.7% lower than nominal BADA on average; 95% bound coverage 97.2% on test. **[verified]**

**Guarantee.** **Hard** physics (BADA integrates), plus an *empirical* 95% performance envelope
with measured coverage. No procedure/route/intent constraint of any kind.

**Implication for us.** The measured-coverage idea is the transferable one: they state a target
(95%), measure achieved coverage (97.2%), and report both. That is the discipline our corridor
work needs — an achieved-vs-target violation rate, rather than a target we could not reach. Note
also that the entire field's convention is to *avoid* the altitude band where procedures bite;
we are deliberately working in the band they exclude.

---

### 2.5 Jarry, Dalmau, Very, Olive 2025 — A Neural ODE Approach to Aircraft Flight Dynamics Modelling (NODE-FDM)
arXiv:2509.23307. → `papers/jarry2025_node_fdm.pdf`

**Mechanism.** Consumes QAR (flight-data-recorder) streams, not ADS-B: state x = {h, d, gamma,
V_TAS, m}; **control u = FMS/autopilot targets** {h_sel, V_sel, Vz_sel, flap, gear, speedbrake} —
which is how intent enters, as a *given* input; context e = {OAT, headwind, crosswind}. Outputs
state derivatives, integrated by an explicit Euler ODE solver (`torchdiffeq`). Four stacked
structured layers: a **purely analytical trajectory layer** (deterministic kinematic conversions),
an angle layer, an engine layer, and a derivative layer that emits **only** dV_TAS and dgamma —
the rest follow from the analytic relations. **[verified]**

**Core formulas.** **[verified]**
```
x(t) = { h(t), d(t), gamma(t), V_TAS(t), m(t) }
V_z   = V_TAS * sin(gamma)                                    # analytic, hard
M     = V_TAS / a ,   a = sqrt( gamma_air * R * T_OAT )       # analytic, hard
V_GS  = V_TAS - V_parallel                                    # analytic, hard
dx/dt = f_theta( x(t), u(t), e(t) )                           # learned

# (6) composite loss -- MSE only, no constraint term, no physics residual penalty
L = sum_{i in C} alpha_i * l_i( y_i^pred, y_i^true ),   C = { x(t), e2(t), e3(t) }
    l_i = MSE ;  alpha_i inversely proportional to each variable's empirical std
```

**Data + horizon.** QAR from nine A320-214 over two years; 0.25 Hz; 5,000/500/500 flights split
**by airframe**. Training sequences are fixed **60 steps = 4 min**; at test time whole flights are
propagated open-loop from the recorded control sequence. Altitude MAE all-phase **61.90 m vs
167.47 m** for a BADA4+TCL pipeline; descent **182.04 vs 560.15 m**; fuel **83.95 kg (1.54%) vs
163.52 kg (3.03%)**. **[verified]**

**Guarantee.** **Hard only for the kinematic identities** and the ODE structure. Everything
performance-related is **soft MSE supervision**, and the paper's own Limitations section states
there are no aerodynamic or performance bounds and that unrealistic behaviour arises
off-distribution. No procedure constraint. **[verified]**

**Implication for us.** Two things. (1) It is the published precedent for our `control` path's
shape — a learned derivative field integrated by a differentiable solver — and it validates the
design choice of learning *only the hard-to-model derivatives* while computing the kinematically
determined ones analytically. Our point-mass rollout could be audited the same way: which of the
64 controls actually need to be learned? (2) Its explicit disclaimer is a warning: an ODE
architecture buys you *consistency*, not *bounds*. Ours has the same property, which is exactly
why the corridor has to be imposed separately.

---

### 2.6 Prutsch, Schinagl, Possegger 2026 — ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace
ICRA 2026. arXiv:2603.16550. → `papers/prutsch2026_ascent_terminal_tp.pdf`

**The closest architectural comparison to our model: ego-only ADS-B, terminal airspace, 120 s
horizon, multi-modal transformer.**

**Mechanism.** Consumes only the ego aircraft's 3D ADS-B history, normalised into a local frame
(subtract last position, rotate by yaw and pitch from the last two points), then re-injects the
discarded global context via a **3D positional embedding** of (x, y, z, sin γ, cos γ, sin θ,
cos θ). Encoder = 2 self-attention blocks + max-pool to a single 128-d vector; decoder = k
**learnable mode queries** with MLP heads. It does **not predict positions**: it predicts
**flight parameters** (speed, yaw, pitch per future step, angles via sin/cos) and integrates them
kinematically, then inverts the normalisation. No performance limits, no procedures, no pattern
geometry — the FAA traffic pattern is left to be learned statistically by the mode queries. **[verified]**

**Core formulas.** **The paper contains no numbered equations** (verified by grepping the full
text). Stated in prose only: **winner-takes-all** training (only the candidate with lowest L2 to
ground truth is optimised), **smooth L1** for regression, **cross-entropy** for the mode-probability
head. Nothing further should be attributed to it. **[verified]**

**Data + horizon.** TrajAir (KBTP, non-towered) and TartanAviation. Three settings, including
**11 s history @1 Hz → 120 s @0.1 Hz, k=5** and 40 s → 120 s, k=20. minADE_5 / minFDE_5 =
**0.35 / 0.58 km** (7Days-avg, 11 s input) vs TrajAirNet 0.78/1.55 and constant velocity
1.86/4.21; at 40 s input with k=20, minADE_20 / minFDE_20 = **0.19 / 0.26 km**. Ablation: the
parameterized (speed/yaw/pitch) output improves minFDE 0.60 → 0.58 at equal minADE; mode queries
beat a CVAE 0.35/0.58 vs 0.44/0.80. **[verified]**

**Guarantee.** **None, hard or soft.** The kinematic output parameterisation is a representation
choice; procedure structure is learned implicitly, i.e. conditioning at best. **[verified]**

**Implication for us.** The direct baseline-and-contrast paper: same phase, same sensor, same
120 s horizon, and a WTA + mode-query decoder — i.e. exactly the gating machinery of cluster 5
already applied to terminal-airspace aviation, but used for generic multimodality rather than for
a constrained regime. It also gives an aviation-specific data point for cluster 5's annealing
argument: it needs k=20 modes for its best numbers, which is the mode-collapse symptom annealed
WTA (5.3) is designed to remove. Their ablation that the kinematic parameterisation helps FDE but
not ADE is worth noting for our `state`-vs-`control` comparison.

---

### 2.7 Patrikar, Moon, Oh, Scherer 2022 — Predicting Like A Pilot: Dataset and Method to Predict Socially-Aware Aircraft Trajectories in Non-Towered Terminal Airspace
ICRA 2022, pp. 2525–2531. arXiv:2109.15158. → `papers/patrikar2022_predicting_like_a_pilot.pdf`

**Mechanism.** Introduces the **TrajAir** dataset and the **TrajAirNet** baseline. TCN encodes
each agent's trajectory; a CNN encodes METAR wind; a **GAT** over aircraft nodes encodes social
context; a **CVAE** decodes multiple futures; an MLP emits **Verlet accelerations** which are
integrated to positions. Coordinates are absolute, in a local frame whose origin is the runway end
and whose x-axis runs along the runway — **that coordinate choice is the entire extent of
procedure awareness**. No BADA, no performance envelope, no pattern geometry, no goal input
(goal-conditioning is listed as future work). **[verified]**

**Core formulas.** **[verified]**
```
h_obs^a  = TCN_obs( x_{1:t_obs}^a )
h_enc^a  = h_obs^a  (+)  CNN( phi_{1:t_obs}^a )
h_gat    = GAT( h_enc^{1:A} )
z ~ Q( z | h_pred^a, h_enc^a (+) h_gat^a )   [train] ;   z ~ N(0, I)   [test]
s_{t_obs : t_obs+t_pred}^a = MLP( h_cvae^a )

# (7) Verlet integration -- the only "dynamics"
x_{t+1} = 2 * x_t  -  x_{t-1}  +  s_t * dt^2

# (8-10) objective: MSE + KL, no constraint term, no physics residual
L_total = L_traj + L_cvae
L_traj  = MSE( x_{t_obs:t_obs+t_pred}^a , x^_{t_obs:t_obs+t_pred}^a )
L_cvae  = D_KL( Q(z | .) || N(0, I) )
```

**Data + horizon.** TrajAir: KBTP, ADS-B + TIS-B, **111 days** (Sep 2020 – Apr 2021) with matched
METAR, filtered to <6000 ft MSL and within 5 km of a runway end, 1 Hz. **t_obs = 11 s, t_pred =
120 s**, best-of-5. ADE/FDE (km) 0.73/1.42 … 0.86/1.72 across the four 7-day splits; constant
velocity 1.79/4.08. **[verified]**

**Guarantee.** **None.** Verlet integration is a soft output parameterisation; runway geometry is
a coordinate frame only.

**Implication for us.** Mainly the dataset/benchmark anchor for terminal-airspace ADS-B TP, and
the source of the 11 s → 120 s convention that 2.6 and 2.10 inherit. Worth citing for the point
that the field's terminal-airspace work is all GA / non-towered single-runway, whereas we are at
five towered US airports with published instrument approaches — a materially different problem in
exactly the respect (published procedure geometry) that matters to us.

---

### 2.8 Guo, Zhang, Yang, Zhang, Yang, Lin 2023 — Integrating spoken instructions into flight trajectory prediction to optimize automation in air traffic control (SIA-FTP)
Preprint submitted to Nature Communications. arXiv:2305.01661.
→ `papers/guo2023_spoken_instructions_tp.pdf`

**The nearest neighbour to our join-point problem in the entire aviation set — and its answer is
to *observe* the ATC decision rather than infer it, which is exactly what our ego-only input
excludes.**

**Mechanism.** Consumes (i) trajectory points [Lon, Lat, Alt, Vx, Vy, Vz] and (ii) the **ATC
spoken instruction**, ASR-transcribed to text. Backbone is FlightBERT++ with a binary-encoded
trajectory representation trained by BCE rather than regression. Three-stage curriculum:
trajectory-only pre-training; BERT MLM pre-training on unlabeled instruction text then fine-tuning
a multi-label **intent-identification head over 16 instruction categories** (6 manoeuvring ones
used: ALT_ADJ, OFFSET, CANOFF, FLYTO, SPD_ADJ, HEAD_ADJ&FLYTO); then concatenate trajectory and
instruction embeddings and fine-tune on the scarce paired data. Intent is injected as
**feature-level conditioning via a learned embedding** — there is no constraint and no mechanism
forcing the prediction to reach the commanded altitude or heading. The paper's own qualitative
analysis documents the baseline climbing to ~10,100 m when the instruction commands 10,400 m;
SIA-FTP fixes this statistically, not by construction. Waypoint positions are stated to be
"learned from historical trajectory samples", not looked up from a procedure database. **[verified]**

**Core formulas.** **[verified]**
```
# (1) vs (2): the whole contribution is the added SI term
P_{t+1:t+n} = F( O_{t-k+1:t} )              # baseline
P_{t+1:t+n} = F( O_{t-k+1:t} ,  SI )        # SIA-FTP

p_t = [ Lon_t, Lat_t, Alt_t, Vx_t, Vy_t, Vz_t ]

H_SI        = { h_w1, ..., h_wl } = BERT(SI)
SI_emb      = SUM( H_SI )
Intent_prob = Sigmoid( MLP( SI_emb ) )
J_mm        = Concat[ Traj_enc , SI_emb ]
Traj_mm     = MLP( J_mm )
```
Objective is **BCE** over the binary-encoded trajectory (stages 1 and 3) and BCE for multi-label
intent (stage 2); the BCE expression is never written out. **No constraint term exists.** **[verified]**

**Data + horizon.** **M2ATS**, a multi-modal dataset from an industrial ATC system in China,
19–27 Feb 2021. En-route/manoeuvring under clearance, **not final approach**. Observation = 9
trajectory points; horizons 1/3/9/15 steps at 20 s = up to 5 min. MDE (NM) **0.15 / 0.29 / 0.77 /
1.22** vs FlightBERT++ 0.16/0.33/1.03/1.55; altitude MAE at horizon 15 **9.75 m** vs 17.37 m;
>20% relative MDE reduction at the long horizons. **[verified]**

**Guarantee.** **Conditioning only** (soft feature fusion). The commanded target is not enforced.

**Implication for us.** This is the paper to cite when justifying the join-point head. It
establishes empirically that (a) the ATC decision is the dominant unmodelled variable in
terminal/manoeuvring TP — supplying it cuts long-horizon error by >20% — and (b) even when you
*hand the model the instruction verbatim*, feature-level conditioning does not make the trajectory
obey it (the 300 m altitude undershoot). Both points argue for our design: infer the decision as
a discrete latent (cluster 5), then *enforce* the consequence geometrically (cluster 4), rather
than hoping a conditioning feature will do the work.

---

### 2.9 Yoon & Lee 2025 — Multi-Agent Inverted Transformer for Flight Trajectory Prediction (MAIFormer)
Accepted, IEEE Trans. Intelligent Transportation Systems. arXiv:2509.21004.
→ `papers/yoon2025_multiagent_itransformer_tp.pdf`

Short entry — included because it is **the iTransformer inverted embedding applied to multi-agent
ADS-B arrivals**, i.e. our exact backbone in our exact domain.

**Mechanism.** Encoder-only transformer with **zero physics and zero procedure input**. Applies
**iTransformer-style inverted embedding** — each variate's whole time series is one token, giving
N·F tokens with **no positional encoding** (agents have no order) — then stacks 3 layers of
*masked multivariate attention* (a mask confines attention to variates of the same aircraft),
*agent attention* (each agent's F variate tokens concatenated into one agent token, attention
agent-to-agent), and an FFN; an MLP decoder emits all future steps non-autoregressively. **[verified]**

**Core formulas.** **[verified]**
```
C_ST^{l-1} = MaskedMultivariateAttention( LN( C^{l-1} ) ) + C^{l-1}
C_SC^{l-1} = AgentAttention( LN( C_A^{l-1} ) ) + C_A^{l-1}
C^l        = FFN( LN( C_SC^{l-1} ) ) + C_SC^{l-1}

# (9) the mask: variates of the same aircraft only
M[m, n] = 1     if floor(m/F) = floor(n/F)
        = -inf  otherwise,     for m, n in {0, ..., N*F - 1}

Y^ = MLP( C^L )
```
Training objective is **plain L2/MSE**; stated in prose, never written out. **No physics residual,
no constraint term.** **[verified]**

**Data + horizon.** FlightRadar24 ADS-B within **70 NM of Incheon (ICN)**, Jan–May 2023,
**arrivals only**, 6 s grid, 509,389 scenes. **Lookback 20 steps = 2 min, horizon 20 steps =
2 min.** MAE at 2 min: lat 0.0037°, lon 0.0047°, **alt 164.18 ft**, vs FlightBERT++
0.0052/0.0059/287.83 and PatchTST 0.0065/0.0074/311.65. **[verified]**

**Guarantee.** **None at all.** The paper's own Conclusion lists conditioning on "meteorological
conditions, flight procedures, and airspace constraints" as future work. **[verified]**

**Implication for us.** Two uses. (1) A same-architecture, same-domain accuracy reference —
including a **PatchTST baseline**, which is one of our two vendored models, on arrival ADS-B.
(2) Its explicit statement that flight procedures and airspace constraints remain future work is
a citable confirmation, from an accepted T-ITS paper in 2025, that the gap we are targeting is
open.

---

### 2.10 Xiang & Chen 2024 — Data-driven Probabilistic Trajectory Learning with High Temporal Resolution in Terminal Airspace
Submitted to AIAA JAIS. arXiv:2409.17359. → `papers/xiang2024_probabilistic_tp_terminal.pdf`

**Mechanism.** Four modules: TCN + CNN feature extraction → GAT fusion → a **CVAE guide
generator** emitting per-step accelerations integrated Verlet-style into a *coarse* 12-point
"trajectory guide" at 10 s spacing → a **Gaussian Mixture Regression** predictor, where a GMM
fitted by EM over the joint (past ⊕ guide, ground-truth future) is conditioned on the guide to
emit the final **1 Hz, 120-point** trajectory. The claimed novelty is that the mixture model
*learns from the network's output* rather than the network emitting mixture parameters, which
defeats error propagation and keeps the GMM in 3 dimensions. Physics is limited to the Verlet
integration inherited from TrajAirNet. **[verified]**

**Core formulas.** **[verified]**
```
# guide generation (Verlet; note the paper's eq. 13 has a typo, printing p_t dt^2
# where the acceleration term s_t dt^2 is meant -- cf. patrikar eq. 7)
p_{t+1}^e = 2 p_t^e - p_{t-1}^e + s_t^e dt^2

# GMR: condition the joint mixture on the guide
mu_{y|x}     = mu_y + Sigma_yx Sigma_xx^-1 ( x - mu_x )
Sigma_{y|x}  = Sigma_yy - Sigma_yx Sigma_xx^-1 Sigma_xy
pi_{y|x_k}   = N_k( x | mu_x_k, Sigma_x_k ) / sum_l N_l( x | mu_x_l, Sigma_x_l )
p( y | x )   = sum_{k=1..K} pi_{y|x_k} * N_k( y | mu_{y|x_k}, Sigma_{y|x_k} )

# objective: MSE + KL on the GUIDE only. No constraint term.
L_total = L_traj + L_cvae ,   L_traj = MSE( T_guide, T^_guide )
```

**Data + horizon.** TrajAir (KBTP), same 4×7-day evaluation splits as 2.7. **11 s past, guide at
10 s spacing, 120 s predicted at 1 Hz (120 points vs TrajAirNet's 12)**, K = 150 Gaussians,
best-of-5. ADE (km) **0.72 / 0.87 / 0.92 / 0.80**; TrajAir10 0.74/0.87/0.89/0.93; and the result
the paper is built around — **TrajAirNet forced to 1 Hz gives 5.57 / 7.33 / 12.61 / 4.68**, i.e.
error propagation destroys it. **[verified]**

**Guarantee.** **None.** Verlet kinematics applies to the *guide* only; the final trajectory is a
GMM conditional subject to no physical or procedural bound. **[verified]**

**Implication for us.** The coarse-guide-then-refine decomposition is worth noting as a third
architecture alongside `state` and `control`: predict a low-rate skeleton, then upsample under a
separate model. Its headline negative result — a 120 s trajectory at 1 Hz produced
autoregressively blows up by an order of magnitude — is a useful caution for our own high-rate
outputs. But like everything else in this cluster it enforces nothing.

---

## 3. Predict-then-optimize / amortized optimization

> **Read this cluster in one line:** if the constraint is hard and the join point is
> unobservable, the cleanest architecture is not "make the network feasible" but "let the
> network predict the *decision* and let a solver own the *constraint*". These papers say what
> the network should be trained to predict in that split, and it is usually **not** the
> ground-truth trajectory.

---

### 3.1 Amos 2023 — Tutorial on Amortized Optimization
Foundations and Trends in Machine Learning. arXiv:2202.00665.
→ `papers/amos2023_amortized_optimization_tutorial.pdf`

**Mechanism.** Systematises "learn to solve a family of optimisation problems parameterised by
a context x". Two orthogonal design axes, and they are exactly the axes of our decision.
(1) **Model**: *fully-amortized* (the network never sees the objective) vs *semi-amortized*
(the model is itself an optimisation process, e.g. K gradient steps, differentiated through).
(2) **Loss**: *regression-based* (imitate a known solution) vs *objective-based* (be good under
the objective itself). **[verified]**

**Core formulas (their eqs. 2.1, 2.9, 2.10, 2.11).** **[verified]**
```
# semi-amortized model: K unrolled steps on the objective
y^_theta^t := y^_theta^{t-1} - alpha * grad_y f( y^_theta^{t-1} ; x ),   t = 1..K

# regression-based loss (behavioural cloning, in control)
L_reg(y^_theta)  :=  E_{x ~ p(x)} || y*(x) - y^_theta(x) ||_2^2

# objective-based loss
L_obj(y^_theta)  :=  E_{x ~ p(x)}  f( y^_theta(x) ; x )

grad_theta L_obj = E_{x~p(x)} [  D_theta[y^_theta(x)]^T  grad_y[ f(y^_theta(x); x) ]  ]
```
The tutorial's own characterisation: regression-based learning "works the best for distilling
known solutions into a faster model ... but can otherwise start failing to work". **[verified]**

**Guarantee / cost.** None per se (a survey); §5.2 discusses generalisation and convergence as
open. Cost: framework.

**Implication for us.** Names the choice we are implicitly making. Our current `control` path
trained on observed tracks is **fully-amortized + regression-based** — it imitates trajectories
without ever seeing the corridor objective. That is the configuration the tutorial says is a
*distillation* tool, and it is why the constraint has to be bolted on afterwards. The
objective-based gradient `D_theta[y^]^T grad_y f` is precisely what our differentiable
point-mass rollout already makes computable, so an objective-based term on the corridor is
available to us without new machinery. And the semi-amortized reading legitimises the CBF
filter: a model that *contains* a solver is a model, not a crutch — provided the loss knows the
solver is there (see 3.3).

---

### 3.2 Sambharya, Hall, Amos, Stellato 2023 — End-to-End Learning to Warm-Start for Real-Time Quadratic Optimization
arXiv:2212.08260. → `papers/sambharya2023_learn_warmstart_qp.pdf`

**Mechanism.** A feedforward net maps problem parameters theta to a warm-start point; that
point is then fed through **k Douglas-Rachford splitting iterations**, and the loss is
back-propagated through those iterations to the network weights — end-to-end. Crucially the
method **does not require solving any optimisation problems offline** to build targets, because
the loss is the solver's own convergence measure rather than a distance to a stored solution. **[verified]**

**Core formulas (their eq. 6 and the risk).** **[verified]**
```
loss = fixed-point residual of the DR operator T_theta:

        l_theta(z)  =  || T_theta(z) - z ||_2

risk    R(h_W)      =  E_{theta ~ D} [  l_theta( T_theta^k( h_W(theta) ) )  ]
empirical risk      =  sum_{i=1..N}    l_{theta_i}( T_{theta_i}^k( h_W(theta_i) ) )
```
where `h_W` is the NN and `T_theta^k` is k solver iterations. **[verified]**

**Guarantee / cost.** Statistical — generalisation bounds from operator theory + Rademacher
complexity that improve simultaneously with the number of training problems *and* the number of
DR iterations. Feasibility comes from the downstream solver, not the net. Empirically **30% to
90% fewer iterations** to a target accuracy. **[verified]**

**Implication for us.** The single most transferable idea in cluster 3: **train the predictor
on the solver's residual, not on the observed track.** If we adopt a predictor-plus-OCP
architecture (network proposes, constrained tracking OCP enforces the corridor), this says the
network should be scored by how much work it saves the OCP, and that such a loss needs no
offline solve library. It also dissolves the "training through the CBF makes the network rely
on it" complaint from a different direction: reliance is fine when the loss is defined *through*
the solver, because then the network is optimising the composite object we actually deploy.

---

### 3.3 Amos, Rodriguez, Sacks, Boots, Kolter 2018 — Differentiable MPC for End-to-end Planning and Control
NeurIPS 2018. arXiv:1810.13400. → `papers/amos2018_differentiable_mpc.pdf`

**Mechanism.** Treats MPC as a differentiable policy class: run a box-constrained iterative LQR
solver forward, then differentiate **analytically through the fixed point** using the KKT
conditions of the convex approximation at that fixed point, rather than unrolling and
backpropagating through every solver iteration. The backward pass is a single additional LQR
solve. Cost and dynamics are then learned end-to-end from a task loss (they recover both from
action-only imitation losses). **[verified]**

**Core mechanism, stated exactly.** "we differentiate through MPC by using the KKT conditions
of the convex approximation at a fixed point of the controller"; "we compute the derivatives of
an iLQR solver with a single LQR step in the backward pass", in contrast to unrolling whose
"time- and space-complexity of the backward pass grows linearly with the forward pass". **[verified]**
(The explicit KKT differentiation formulas were not transcribed here — **[unverified]** beyond
the statements above.)

**Guarantee / cost.** Hard on the *box control constraints* (the solver enforces them by
construction); nothing is guaranteed about state constraints. Cost: one extra LQR solve per
backward pass — cheap relative to unrolling.

**Implication for us.** This is the counter-argument to abandoning "training through the
filter". Our finding that the network learns to lean on the CBF is the expected outcome of
end-to-end training through a constraint-enforcing layer — it is what differentiable MPC is
*for*. The question is whether that is a defect. It is a defect only if we intend to deploy the
network without the filter; if the filter ships, the composite is the policy and the reliance is
free. The concrete borrow is the fixed-point/implicit gradient: our 64-step rollout plus CBF QP
currently backprops through the whole chain, and implicit differentiation at the QP's KKT point
would cut that cost and remove the ill-conditioning that long unrolls introduce.

---

### 3.4 Elmachtoub & Grigas 2022 — Smart "Predict, then Optimize" (SPO)
Management Science. arXiv:1710.08005. → `papers/elmachtoub2022_spo.pdf`

**Mechanism.** For problems where a prediction feeds a downstream optimiser with a *linear*
objective over a known feasible set S, argues you should train against **decision error**, not
prediction error. The true SPO loss is discontinuous and intractable, so they derive a convex
surrogate (SPO+) by duality plus a first-order expansion, and prove it is **statistically
consistent** with the SPO loss under mild conditions (the conditional cost distribution is
continuous and symmetric about its mean). **[verified]**

**Core formulas (their Definitions 1–3).** **[verified]**
```
# Def 1, true SPO loss w.r.t. an optimization oracle w*(.)
l_SPO^{w*}(c^, c)  :=  c^T w*(c^)  -  z*(c)

# Def 2, unambiguous SPO loss
l_SPO(c^, c)       :=  max_{w in W*(c^)}  c^T w  -  z*(c)

# Def 3, the convex surrogate SPO+
l_SPO+(c^, c)      :=  max_{w in S} { c^T w - 2*c^T w }  +  2*c^^T w*(c)  -  z*(c)
                    =  xi_S( c - 2*c^ )  +  2*c^^T w*(c)  -  z*(c)
```
where `c^` is the prediction, `c` the realised cost vector, `z*(c)` the optimal value,
`w*(.)` the optimisation oracle, and `xi_S(c) = max_{w in S} c^T w` the support function of S.
Note `l_SPO(c^, c) = l_SPO+(c^, c) = 0` when `c^ = c`. **[verified]**

**Guarantee / cost.** Statistical consistency (Bayes-risk minimiser of SPO+ is `E[c|x]`, same as
for SPO and for least squares under those conditions); the feasible set is respected by the
downstream oracle, hard, by construction. Cost: each SPO+ gradient needs one call to the
optimisation oracle.

**Implication for us.** The framing transfers even though our downstream objective is not
linear in a predicted cost vector: **if a solver will run anyway, the network should be trained
on the quality of the resulting decision, not on ADE against the observed track.** Two of our
results become legible in that light — a fixed-weight penalty "trading accuracy for violation
rate" is a symptom of scoring the prediction rather than the decision, and the corridor-bounded
tanh works because it moves the feasible set into the model, i.e. it makes S structural. The
caveat is real: SPO+'s consistency theorem needs a linear objective, which our point-mass
rollout is not, so this is a design principle for us and not an importable loss.

---

### 3.5 Guffanti, Gammelli, D'Amico, Pavone 2024 — Transformers for Trajectory Optimization with Application to Spacecraft Rendezvous
IEEE Aerospace Conference 2024. arXiv:2310.13831.
→ `papers/guffanti2024_art_transformer_trajopt_warmstart.pdf`

**Mechanism.** A decoder-style transformer (ART, "Autonomous Rendezvous Transformer") is
trained on a dataset of locally optimal trajectories and then used to generate a **near-optimal
warm start** for a conventional constrained trajectory optimiser, which retains responsibility
for dynamics and safety constraints. Nearest neighbour in spirit to our situation: a learned
sequence model proposes, a solver disposes. **[verified — abstract/premise; detailed equations
unverified]**

**Guarantee / cost.** Hard constraints come from the downstream optimiser; the transformer
carries no guarantee. Claimed benefit is convergence speed/robustness of the optimiser.

**Implication for us.** The closest published precedent for the architecture we would build:
a transformer trained on trajectories whose *output is an initial guess*, not a deliverable.
Directly relevant because our `control` path already emits exactly what an optimiser wants —
a control sequence plus a duration — so it is a warm start in the right coordinates already.

---

### 3.6 Briden, Choi, Yun, Linares, Cauligi 2023 — Constraint-Informed Learning for Warm Starting Trajectory Optimization
arXiv:2312.14336. → `papers/briden2023_constraint_informed_warmstart.pdf`

**Mechanism.** Learns warm starts for constrained trajectory optimisation for spacecraft /
surface-robot autonomy, where the nonlinear solvers are too slow for resource-constrained
flight computers; the "constraint-informed" part is that the learned representation is
conditioned on the problem's constraint structure rather than on the trajectory alone. **[verified
— abstract/premise; detailed equations unverified]**

**Implication for us.** The nearest published statement of "let the learner know *which*
constraints are active, and have it predict that". That maps onto the join-point problem: what
our network most usefully predicts may be **which constraint regime applies from when**, with
the corridor geometry itself supplied analytically. See cluster 5.

---

## 4. Constrained generative (diffusion) trajectory models

> **Read this cluster in one line:** four distinct places to put a constraint in a generative
> sampler — as a **gradient** (Diffuser, MotionDiffuser: soft, cheap, no guarantee), as a
> **projection** (PDM: hard on the constraint set, needs a projection oracle), as a **CBF-QP
> inside the denoiser** (SafeDiffuser: hard-ish, and its time-varying variant is a principled
> constraint warm-up), or as a **warm start to a real solver** (DiffuSolve: hard, via the NLP).

---

### 4.1 Janner, Du, Tenenbaum, Levine 2022 — Planning with Diffusion for Flexible Behavior Synthesis (Diffuser)
ICML 2022 (long talk). arXiv:2205.09991. → `papers/janner2022_diffuser.pdf`

**Mechanism.** Trains a diffusion model over whole trajectories so that *sampling from the
model and planning with it become nearly identical*. States and actions are denoised jointly as
a 2-D array over the horizon (actions are just extra state dimensions), with 1-D temporal
convolutions so the horizon is set by input size, not architecture. Test-time objectives —
rewards, goals, constraints — enter as a factor h(tau) on the density, realised as classifier
guidance on the reverse-process mean. Conditioning (e.g. fixing the current state) is
implemented as inpainting. **[verified]**

**Core formulas (their eqs. 1–3 and Algorithm 1).** **[verified]**
```
perturbed distribution:     p~_theta(tau)  ∝  p_theta(tau) * h(tau)                      (1)

guided reverse transition:  p_theta(tau^{i-1} | tau^i, O_{1:T}) ≈ N( tau^{i-1}; mu + Sigma*g, Sigma )   (3)
    with  g = grad_tau log p(O_{1:T} | tau) |_{tau = mu}
            = sum_{t=0..T} grad_{s_t, a_t} r(s_t, a_t) |_{(s_t,a_t) = mu_t}
            = grad J(mu)

Algorithm 1 (guided planning):
  3:  observe state s; initialize plan tau^N ~ N(0, I)
  4:  for i = N, ..., 1 do
  6:      mu <- mu_theta(tau^i)
  8:      tau^{i-1} ~ N( mu + alpha * Sigma * grad J(mu),  Sigma^i )      # guide
 10:      tau_{s_0}^{i-1} <- s                                            # constrain first state
 11:  execute first action of plan tau_{a_0}^0
```

**Guarantee / cost.** Soft — guidance biases the mean, nothing is certified; line 10 shows that
*hard* conditioning is done by overwriting, not by guidance. Cost: N denoising steps plus a
gradient of the learned J per step.

**Implication for us.** Establishes the base pattern and, more usefully, its limit: everything
Diffuser enforces *hard* it enforces by **direct substitution** (inpainting the known state),
and everything it enforces by gradient it enforces *softly*. Our corridor is a set constraint
over a contiguous sub-interval of the horizon whose start we do not know — neither inpainting
(we do not know the values) nor guidance (no guarantee) covers it, which is precisely why 4.3
and 4.5 matter.

---

### 4.2 Jiang, Cornman, Park, Sapp, Zhou, Anguelov 2023 — MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion
CVPR 2023 (highlight). arXiv:2306.03083. → `papers/jiang2023_motiondiffuser.pdf`

**Mechanism.** Diffusion over the *joint* future of multiple agents, permutation-invariant via a
transformer denoiser over agent tokens, trained with a plain L2 denoising loss and no
trajectory anchors. Trajectories are compressed by **PCA** before diffusion (10 components for
an 80x2 trajectory; 3 components already capture 99.7% of variance), which both speeds sampling
and improves controllability. Constraints enter at inference as guidance from an arbitrary
differentiable cost. **[verified]**

**Core formulas (their eqs. 12–16).** The key move is the **denoiser-output trick**: evaluate
the cost on the *denoised* trajectory `D(S; C, sigma)` rather than the noisy sample S, because
D is near the data manifold at every noise level while S is not. **[verified]**
```
joint score:   grad_S log p(S; C, sigma)  +  grad_S log q(S; C, sigma)                    (12)

constraint gradient score, approximated for an arbitrary differentiable cost L:
               grad_S log q(S; C, sigma)  ≈  lambda * (d/dS) L( D(S; C, sigma) )           (13)

attractor:     L_attract = sum| (D(S;C,sigma) - S_target) ⊙ M_target | / ( sum|M_target| + eps )   (14)
repeller:      A = max( 1 - (1/r) * Delta(D(S;C,sigma)) ⊙ (1 - I),  0 )                    (15)
               L_repell = sum A / ( sum(A > 0) + eps )                                     (16)
```

**Guarantee / cost.** Soft — a weighted score term; lambda trades constraint satisfaction
against sample likelihood, with no feasibility certificate. Cost: one cost gradient through the
denoiser per step.

**Implication for us.** The denoiser-output trick is the transferable engineering detail: our
corridor cost is only meaningful on a physically plausible trajectory, so evaluating it on the
model's *clean* output rather than on an intermediate iterate is the right pattern in any
iterative scheme, diffusion or not. The attractor cost with a **binary mask M_target over
timesteps** is also suggestive for the join point — it is a soft version of "constrain only
these steps", with the mask as the thing to be predicted.

---

### 4.3 Christopher, Baek, Fioretto 2024 — Constrained Synthesis with Projected Diffusion Models (PDM)
NeurIPS 2024. arXiv:2402.03559. → `papers/christopher2024_projected_diffusion.pdf`

**Mechanism.** Recasts reverse diffusion as a **constrained optimisation** problem — minimise
the negative log likelihood along the reverse Markov chain, subject to every iterate lying in
the constraint set C — and solves it by projecting after each Langevin update. Because the
projection is applied at every step (not once at the end), constraint violations decay smoothly
toward zero as t -> 0 and the optimality cost of the late-stage projections is small.
Explicitly distinguishes itself from methods that use diffusion merely to seed a downstream
constrained solver (i.e. from 4.4). **[verified]**

**Core formulas (their eqs. 6–8 and Algorithm 1).** **[verified]**
```
problem:      min  (negative log likelihood along the reverse chain)                      (6a)
              s.t. x_T, ..., x_0  in  C                                                   (6b)

projection:   P_C(x)  =  argmin_{y in C}  || y - x ||_2^2                                 (7)

sampling:     x_t^{i+1} = P_C(  x_t^i  +  gamma_t * grad_{x_t^i} log q(x_t^i | x_0)
                                        +  sqrt(2*gamma_t) * eps  )                       (8)

Algorithm 1 (PDM):
  1  x_T^0 ~ N(0, sigma_T^2 I)
  2  for t = T to 1:
  3      gamma_t <- sigma_t^2 / (2 sigma_T^2)
  4      for i = 1 to M:
  5          eps ~ N(0, I);   g <- s_{theta*}(x_t^{i-1}, t)
  6          x_t^i = P_C( x_t^{i-1} + gamma_t * g + sqrt(2*gamma_t) * eps )
  7      x_{t-1}^0 <- x_t^M
  8  return x_0^0
```

**Guarantee / cost.** **Hard on C** by construction (every returned iterate is a projection onto
C). Convergence to the optimum is guaranteed for **convex** C; for non-convex C the evidence is
empirical, with Theorem 5.2 bounding proximity in terms of the projection cost. Cost: one
projection per inner iteration — the projections can be **warm-started** across steps, which is
what makes it practical for non-convex C. **[verified]**

**Implication for us.** The most directly implementable hard-constraint mechanism in this
cluster, because **our corridor projection is closed-form**. The LPV lateral corridor plus
glidepath window is a box in runway-frame coordinates, so `P_C` is a clip — the same object
our corridor-bounded tanh already computes, but applied as a projection inside an iterative
sampler rather than as an output squashing. The stated caveat matters for the join point: PDM
constrains *every* iterate over the *whole* horizon, whereas our constraint is active only after
establishment; applying it everywhere would forbid the downwind and base legs. So PDM's
projection is the right operator but needs a time mask — which returns us to predicting the
join point (cluster 5).

---

### 4.4 Li, Ding, Dieng, Beeson 2024 — DiffuSolve: Diffusion-based Solver for Non-convex Trajectory Optimization
arXiv:2403.05571. → `papers/li2024_diffusolve.pdf`

**Mechanism.** Two-stage. A diffusion model trained on pre-collected *locally optimal* solutions
samples diverse initial guesses; those warm-start a nonlinear-program solver that fine-tunes
feasibility and optimality (and certifies via KKT). **DiffuSolve+** adds a constraint-violation
loss during *training* so that the samples handed to the NLP violate less to begin with. Tested
on two-car reach-avoid, quadrotor navigation, and cislunar low-thrust transfer (decision
dimensions 81, 241, 64). **[verified]**

**Core formula (DiffuSolve+, their eq. 7).** Violation is evaluated on the **directly predicted
clean sample** `x^_0` (avoiding backprop through multi-step sampling), clipped into the valid
range so the violation function is well defined: **[verified]**
```
L_vio = E_{(x_0,y)~XxY, k in [K], eps ~ N(0,I)}
            [ (1/k) * V( clip( x^_0(x_k, eps_theta), x_0^min, x_0^max ),  y ) ]
```

**Guarantee / cost.** Hard, but supplied by the **NLP solver**, not by the generative model;
the diffusion side is a soft violation penalty. Reported 2x–11x compute improvement. **[verified]**

**Implication for us.** The reference design for "learned proposal + constrained solver", and
the closest to our `control` path: it generates a *control-space* decision vector of comparable
dimension (64!) and hands it to an NLP. Two lessons. (1) The `1/k` weighting and the clip in
eq. 7 are the practical details that make a violation loss trainable inside a generative model
— worth copying if we add a corridor term to any sampler. (2) It is explicit that even with a
violation loss the generative model does *not* own feasibility; the solver does. That is the
architecture our CBF result already points at.

---

### 4.5 Xiao, Wang, Gan, Rus 2023 — SafeDiffuser: Safe Planning with Diffusion Probabilistic Models
arXiv:2306.00148. → `papers/xiao2023_safediffuser.pdf`

**The paper that speaks most directly to our CBF finding, and the one with the best answer to
the warm-up question.**

**Mechanism.** Makes the *denoising process itself* a control system — `tau_dot^j = u^j`, with
`u^j` kept close to the model's own step `(tau^j - tau^{j+1})/Delta tau` so performance is
preserved — and then imposes a **control barrier function** condition on `u^j` via a QP at each
denoising step. Because a sample starts as Gaussian noise (and so starts *infeasible*), ordinary
forward invariance is not enough; they define **finite-time diffusion invariance**, satisfied if
the specification holds from some denoising step onward. Three variants address the "local trap"
problem where an early-locked constraint strands the sample. **[verified]**

**Core formulas.** Robust-safe diffuser (their eq. 8 / Thm. 2): **[verified]**
```
diffusion dynamics:   tau_dot^j = lim_{Delta tau -> 0} (tau^j - tau^{j+1}) / Delta tau ;
                      controllable form  tau_dot^j = u^j            (6), (7)

CBF condition:  h(u_k^j | x_k^j)  :=  ( d b(x_k^j) / d x_k^j ) * u_k^j  +  alpha( b(x_k^j) )  >= 0
                for all k in {0..H},  j in {0..N-1},  alpha an extended class-K function
```
**Definition 4 (Finite-time Diffusion Invariance):** if there exists i in {0..N} such that
`b(x_k^j) >= 0` for all k in {0..H} and all j <= i, the procedure is finite-time diffusion
invariant. **[verified]**

Relaxed-safe diffuser (eq. 9) — a relaxation variable with a schedule that drives it to a hard
constraint: **[verified]**
```
h(u_k^j, r_k^j | x_k^j) := ( d b(x_k^j)/d x_k^j ) * u_k^j + alpha( b(x_k^j) ) - w_k(j) * r_k^j >= 0
    r_k^j a relaxation variable;  w_k(j) >= 0 decreasing to 0 as j -> 0.
    When w_k(j) -> 0 the condition becomes a HARD constraint.
    Optionally run N_a extra steps with j in {-N_a, ..., N-1}; output tau^0 := tau^{-N_a}.
```
**Time-varying-safe diffuser (eqs. 11–12) — the constraint itself is annealed:** **[verified]**
```
modified specification:   b(x_k^j) - gamma_k(j) >= 0,     k in {0..H}, j in {0..N}
    gamma_k continuously differentiable, with  gamma_k(N) <= b(x_k^N)  and  gamma_k(0) = 0

enforced by:  h(u_k^j | x_k^j, gamma_k(j))
              := ( d b(x_k^j)/d x_k^j ) * u_k^j - gamma_dot_k(j) + alpha( b(x_k^j) - gamma_k(j) ) >= 0
```

**Guarantee / cost.** Hard "with almost probability 1" at the end of the diffusion procedure
(Thms. 2–4), conditional on the QP being feasible. Cost: one QP per denoising step per state.

**Implication for us.** Two things, both important. (1) It is the published version of our
own result — a CBF filter over a generative trajectory model works — and it treats the filter as
part of the model rather than as a crutch, which reframes "the network relies on it" as the
intended design. (2) **The time-varying constraint `gamma_k(j)` is exactly the constraint warm-up
we should be running.** `gamma_k(N) <= b(x_k^N)` means the specification starts loose enough to
be satisfiable at the noisy/early stage and `gamma_k(0) = 0` means it is fully tight at the end.
Transposed from denoising step to *training* step, this is a schedule from a reachable violation
level to the target one — a principled alternative to a fixed unreachable epsilon, and cheaper
to implement than the resilient formulation (1.3) because gamma is a schedule, not a learned
variable. It is the same shape as Kervadec's `t` schedule (1.7), stated on the constraint level
instead of on the barrier sharpness.

---

## 5. Gating / phase: training a discrete mode decision jointly with a continuous trajectory

> **Read this cluster in one line:** the join point is a discrete latent, and this literature's
> answer is uniformly "predict a *set* of candidate modes, score them by classification, regress
> the continuous trajectory conditioned on the chosen one, and train both with a
> winner-takes-all assignment — but anneal the assignment or it collapses."

---

### 5.1 Zhao, Gao, Lan, Sun, Sapp, Varadarajan, Shen, Shen, Chai, Schmid, Li, Anguelov 2020 — TNT: Target-driveN Trajectory Prediction
CoRL 2020. arXiv:2008.08294. → `papers/zhao2020_tnt.pdf`

**Mechanism.** Three stages, trained end-to-end, each with an interpretable output. (a) **Target
prediction**: discretise the plausible endpoint space into N candidates *sampled from the map*
(points on lane centerlines — the domain prior), and predict a categorical distribution over
them plus a continuous offset. (b) **Target-conditioned motion estimation**: regress a
trajectory per target, with **teacher forcing** on the ground-truth target during training.
(c) **Scoring and selection**: score whole trajectories with a maximum-entropy model and greedily
suppress near-duplicates. Over-sample candidates (N = 1000) and keep the top M = 50. **[verified]**

**Core formulas (their eqs. 2–7).** **[verified]**
```
# (2) discrete-continuous factorization of the target distribution
p(tau^n | x) = pi(tau^n | x) * N( dx^n | nu_x^n(x) ) * N( dy^n | nu_y^n(x) )
    with  pi(tau^n | x) = exp f(tau^n, x) / sum_{tau'} exp f(tau', x)

# (3) stage-1 loss: classification of the candidate + Huber regression of its offset
L_S1 = L_cls(pi, u)  +  L_offset(nu_x, nu_y, dx^u, dy^u)
    L_cls = cross entropy;  u = the target CLOSEST TO THE GROUND TRUTH location

# (4) stage-2 loss: per-step Huber regression, conditioned on the target
L_S2 = sum_{t=1..T} L_reg( s^_t, s_t )

# (5)-(6) stage-3: max-entropy scoring over the M trajectories + cross entropy to a soft label
phi(s_F | x) = exp( g(s_F, x) ) / sum_{m=1..M} exp( g(s_F^m, x) )
L_S3 = L_CE( phi(s_F | x),  psi(s_F) )
    psi(s_F) = exp( -D(s, s_GT)/alpha ) / sum_{s'} exp( -D(s', s_GT)/alpha )
    D(s_i, s_j) = max( ||s_1^i - s_1^j||_2^2, ..., ||s_t^i - s_t^j||_2^2 )

# (7) total
L = lambda_1 * L_S1 + lambda_2 * L_S2 + lambda_3 * L_S3
```

**Guarantee / cost.** None — soft, purely statistical. The map-derived candidate set is the only
structural prior. Cost: M forward passes of a small MLP decoder.

**Implication for us.** The template for the join point, and the analogy is unusually tight.
TNT's candidates are *sampled from the published map* ("vehicles never depart far away from
lanes"); our candidates would be sampled from the **published final-approach geometry** —
positions along the LPV corridor centerline where a flight can plausibly become established.
Then: classify which join point (cross-entropy against the *closest-to-ground-truth* candidate,
which is a supervision signal we can compute from observed tracks even though ATC's decision is
unobservable), regress the offset, and regress the trajectory conditioned on it — and, exactly
as in eq. (3)-(4), use **teacher forcing on the ground-truth join point during training** so the
trajectory head is not learning against a moving target. The corridor constraint then applies
only after the predicted join point, which is what makes a hard projection (4.3) or a CBF
(4.5) applicable at all.

---

### 5.2 Gu, Sun, Zhao 2021 — DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets
ICCV 2021. arXiv:2108.09640. → `papers/gu2021_densetnt.pdf`

**Mechanism.** Removes TNT's two heuristics — the sparse hand-defined anchors and the rule-based
NMS selection. Predicts probabilities over **dense** goal candidates (a heatmap over road
locations), then a learned **goal set predictor** outputs the final set end-to-end. The training
problem is that goal-set prediction is multi-label but only one future is observed; they solve
it with an **offline optimisation-based model** that finds an optimal goal set from the predicted
goal distribution and uses it as **multi-future pseudo-labels** to supervise the online model. **[verified]**

**Guarantee / cost.** Soft. 1st on Argoverse and the 2021 Waymo Open Dataset Motion Prediction
Challenge. Cost: an offline optimisation pass to manufacture labels. (Exact loss equations not
transcribed — **[unverified]**.)

**Implication for us.** The pseudo-label device is the interesting part for a problem where the
label is genuinely unobservable. We cannot observe ATC's join decision, but we *can* run an
offline optimisation over each observed track to extract the join point that best explains it,
and supervise the online (ego-only, 120 s lookback) model against that. That converts an
unobservable decision into a supervised target — the same trick, applied to a different latent.

---

### 5.3 Xu, Letzelter, Chen, Zablocki, Cord 2024 — Annealed Winner-Takes-All for Motion Forecasting
ICRA 2025. arXiv:2409.11172. → `papers/xu2025_annealed_wta.pdf`

**Mechanism.** Diagnoses the standard multi-hypothesis training recipe: WTA updates only the
best head, which is initialisation-sensitive and **mode-collapses** with few hypotheses, forcing
practitioners to train with many hypotheses (e.g. 64) plus a cumbersome NMS/clustering
post-selection. Replaces the hard argmin assignment with a **softmin over hypotheses at
temperature T(t)**, annealed downward so that training starts as a soft weighted average over
all heads (effectively one mode) and converges to exact WTA. Reduces training queries from 64 to
6 and removes post-selection. **[verified]**

**Core formulas (their eqs. 1–7).** **[verified]**
```
# standard WTA
k*      = argmin_k  l( f_theta^k(x), y )                                       (1)
L(theta) = l( f_theta^{k*}(x), y )                                             (2)
l( f_theta^k(x), y ) = (1/L) * sum_{j=1..L} ( f_theta^k(x)_j - y_j )^2         (3)   # ADE

# annealed WTA: softmin assignment (stop-gradient applied to q_t)
q_t( f_theta^k | x, y ) = (1 / Z_{x,y}) * exp( - l( f_theta^k(x), y ) / T(t) )  (4)
Z_{x,y}                 = sum_{s=1..K} exp( - l( f_theta^s(x), y ) / T(t) )    (5)

L_t(theta) = sum_{k=1..K}  q_t( f^k, x, y ) * l( f_theta^k(x), y )             (6)

T(t) = T_0 * rho^t                                                             (7)   # exp. schedule
```
Limits: `lim_{t->0} q_t = 1/K` (all hypotheses contribute equally; hypotheses converge to the
conditional mean, effective number of modes = 1) and `lim_{t->inf} q_t = 1[k in argmin_s l]`
(exact WTA). Related variants, for comparison: RWTA uses
`(1 - K*eps/(K-1)) * 1_WTA(k) + eps/(K-1)` with eps = 0.05 and no schedule; EWTA updates the
top-n heads with n on a decreasing schedule. **[verified]**

**Guarantee / cost.** Soft; the justification for the Boltzmann `q_t` is Proposition 2 of the
annealed-MCL paper it builds on (optimal way to constrain the soft assignment to a given
entropy) — **[unverified]**, cited not reproduced. Cost: two hyperparameters (T_0, rho); *saves*
compute by cutting 64 queries to 6.

**Implication for us.** If we make the join point a set of K discrete hypotheses (5.1), this is
how to train it without collapse — and the annealing story is the same one that recurs in
SafeDiffuser's gamma_k(j) (4.5) and Kervadec's t (1.7): **start soft, tighten on a schedule.**
Three appearances of the same principle in three unrelated literatures is the strongest
methodological signal in this whole collection, and it is the thing our diverged run most
conspicuously lacked — we asked for epsilon = 0.05 from step zero.

---

## Cross-cutting conclusions

0. **The aviation gap is real and verified.** Across ten aviation TP papers (§2), covering
   BADA-as-generator, neural-ODE, transformer, GNN-CVAE and ATC-voice-conditioned models, **none
   enforces a lateral corridor or a glidepath window, and none addresses the establishment /
   join-point decision.** The two families are disjoint: *physics-as-generator* (2.3, 2.4, 2.5)
   gets hard energy feasibility but works **en-route / vertical only and explicitly excludes the
   altitude band where procedures bite** — Pepper restricts to FL150–FL325 precisely "where
   trajectories are less likely to be affected by local operational procedures"; and
   *learned-sequence* (2.6–2.10) works in terminal airspace but enforces nothing at all. The only
   hard published-regulation constraint found anywhere is Hodgkin's post-hoc rejection of samples
   below the UK legal 500 ft/min climb minimum (2.3) — a scalar rate bound, not a geometry.
   Yoon & Lee (2.9), accepted at T-ITS in 2025, list "flight procedures and airspace constraints"
   as future work. This is a positioning claim the thesis can make on the record.

1. **Our divergence is explained, twice, in print.** Gallego-Posada §G: the dual player's best
   response to a *violated* constraint is `lambda = +inf`. Chamon T-IT: `D* = +inf` when the
   empirical problem is infeasible. Gradient ascent on lambda was working correctly; the level
   was wrong. Nothing about the optimiser needs fixing first — the level does.

2. **Three ways out of an unreachable epsilon, in increasing order of effort.**
   (a) *Anneal the level* — SafeDiffuser's `gamma_k(j)` (4.5) transposed to training steps, or
   Kervadec's `t` schedule (1.7). Cheapest; a schedule, not a new variable.
   (b) *Learn the level* — Hounie's resilient constrained learning (1.3), whose `u*` is a direct
   measurement of the reachable violation rate on our fleet. Best scientific payoff, because it
   answers a question we currently cannot answer.
   (c) *Drop the level* — predict-then-optimize (cluster 3), where feasibility is the solver's
   job and epsilon never appears.

3. **If we keep a fixed epsilon, do not use vanilla dual ascent.** νPI (1.6) is the current
   state of the art and ships in Cooper (1.10); Stooke's PID (1.5) is its ancestor. Both damp
   the integral term that ran away on us. CPO (1.8) offers the structural alternative —
   stateless duals recomputed each step — plus an explicit infeasible-recovery branch we did
   not have.

4. **The join point should be predicted, not assumed.** TNT (5.1) gives the recipe — candidates
   sampled from the published geometry, classification against the closest-to-ground-truth
   candidate, teacher forcing during training, trajectory regressed conditionally — and
   annealed WTA (5.3) gives the loss that keeps the hypothesis set from collapsing. DenseTNT
   (5.2) adds the trick for the unobservable label: manufacture it offline by optimisation, then
   supervise the ego-only model against it. This is the prerequisite for every hard-constraint
   mechanism in cluster 4, all of which need to know *when* the constraint switches on.

5. **Our CBF result is a feature, and the literature agrees.** SafeDiffuser (4.5) and
   differentiable MPC (3.3) both treat a constraint-enforcing solver as part of the policy.
   "The network relies on the filter" is a defect only if we ship without the filter. Our
   corridor projection is closed-form (a clip in runway-frame coordinates), which puts PDM
   (4.3) within easy reach — project inside the rollout rather than squash at the output —
   with the one caveat that the projection must be masked to the post-join segment.

6. **Anneal everything.** The same schedule principle appears independently in 1.7 (barrier
   sharpness), 4.5 (constraint level), and 5.3 (assignment temperature). Our failed run applied
   the target constraint at full strength from step zero.

7. **Train against the decision, not the track.** Sambharya (3.2) — score the predictor by the
   solver's fixed-point residual, which needs no offline solve library; SPO (3.4) — decision
   error rather than prediction error; Amos (3.1) — our current setup is
   fully-amortized + regression-based, i.e. the configuration the tutorial classifies as
   *distillation*, which is why the constraint has to be bolted on afterwards.

---

## Searches that came up empty (recorded so they are not repeated)

- **"Sohrabi et al. 2024, constrained learning: a survey"** — no such survey exists on arXiv.
  The actual Sohrabi et al. 2024 is the νPI paper (1.6), which is the more useful result
  anyway. No general "constrained learning survey" was found under any title queried.
- **arXiv full-text search for aviation trajectory prediction that imposes the *published
  approach procedure* as a constraint** (queries: `"arrival" AND "trajectory prediction" AND
  "approach procedure"`, `"trajectory prediction" AND "final approach"`) returned **zero
  results**. The ten papers in §2 condition on weather, aircraft performance (BADA), spoken ATC
  instructions or airport context — none imposes a published final-approach corridor as a hard
  or soft constraint. Confirmed by reading the eight downloadable ones; see conclusion 0.
- **No paper found that predicts the approach *establishment / join point* as a discrete latent.**
  Guo et al. (2.8) is the closest, and it *observes* the ATC decision from radio transcripts
  rather than inferring it — the one input an ego-only model cannot have.
- Shi, Xu & Pan 2021 and Pang et al. 2021 are paywalled with no preprint (§2.1, §2.2). Both were
  probed directly on 2026-09-07 (IEEE Xplore, ScienceDirect) and searched for author-page and
  arXiv copies; none exist. Their entries are marked **[unverified]** throughout and nothing
  about their mechanisms should be asserted without obtaining the full texts.
