# Cluster B — safety filters in the training loop, differentiable / learned CBFs

Scope: what the literature says about **training a policy with a safety filter in the loop
without the policy becoming dependent on the filter**, plus the differentiable-CBF,
discrete-time-CBF, learned-certificate and safe-imitation families.

Our setting for reference: a network emits N=64 piecewise-constant bounded controls
(thrust fraction, bank, load factor); a differentiable RK4 point-mass rollout with a
first-order actuator lag (tau=2 s) integrates them; a per-segment hook acts as a discrete-time
CBF filter on the bank command, with h = k·hw(d) − |xt| turned into an allowed sin(heading
error) interval and then a bank interval; the filter is **gated** — active only once the
rollout's own state says "established on final". Measured here: as an inference-time layer on
a filter-free network it is a clean net gain; **trained through**, it makes the network lazy
(15–24 % of gated steps clipped, bank skill below baseline, vectored flights worse).

Provenance: all formulas below were read out of the PDF in `papers/` unless the line is
explicitly marked `[UNVERIFIED]`. Metadata (authors, dates, journal refs) came from the arXiv
API. Page/DOI details I could not confirm from the paper itself are marked.
Downloads: `download_B.sh` (never overwrites; `papers/` is shared with another collector).

Index by mechanism (the short answer is in §7):

| Family | Papers | Does it answer "how not to be lazy"? |
|---|---|---|
| Action projection layer + loss on **both** actions | OptLayer, Dalal | Yes — OptLayer CPC is the trick |
| Compensator absorbed into the policy | Cheng 2019 | Yes — supervised distillation of the filter |
| Intervention penalty in the objective | Krasowski, Pizarro Bejarano, CBF-RL | Yes — with measured α-sensitivity |
| Theory of why filtering need not cost performance | Oh/Fisac 2025, Hsu survey | Yes — but only for a *permissive* filter |
| Shield + reward bookkeeping | Alshiekh | Yes — states the trade-off first (2018) |
| Predictive safety filters | Wabersich, Tearle | Filter design, not anti-laziness |
| Differentiable CBF **inside** the net | BarrierNet, SafeDiffuser | Different answer: make the filter trainable |
| Discrete-time CBF condition | Agrawal, Zeng | Our h-condition's provenance |
| Safe imitation learning | Cosner, Yin, Geiger (FAGIL) | Yes — and FAGIL argues *for* train-time filtering |
| Residual / bounded correction | Silver, Johannink | Architecture that inverts the dependence |
| Learned certificates | Dawson survey, Fisac 2019 | Yes — loss-term alternative to a hard filter |

---

## 1. Safety layers / action projection

### 1.1 Dalal, Dvijotham, Vecerik, Hester, Paduraru, Tassa (2018) — *Safe Exploration in Continuous Action Spaces*

- arXiv:1801.08757 (26 Jan 2018). No journal ref on arXiv.
- Local: `papers/dalal2018_safe_exploration_1801.08757.pdf`

**Mechanism.** A "safety layer" is appended to a deterministic policy network and projects the
policy's action onto the set that keeps a set of learned per-step safety signals below their
limits. The key modelling choice is that each safety signal is assumed *linear in the action*
over one step: c_i(s,a) ≈ c̄_i(s) + g(s;w_i)ᵀa, with g a small network trained offline on
random-policy transition data (once, before RL, as a pre-training step). Under the assumption
that at most one constraint is active, the projection QP has a closed-form solution — three
lines of code, no solver, no extra hyper-parameters. The layer is a *fixed* map applied on top
of DDPG; the policy is trained by DDPG through/around it.

**Core formulas.**

```
safety-signal model      c_i(s,a) ≈ c̄_i(s) + g(s;w_i)ᵀ a                                  (1)

model fitting            argmin_w  Σ over (s,a,s') of [ c̄_i(s') − (c̄_i(s) + g(s;w_i)ᵀ a) ]² (2)

projection QP            a* = argmin_a  ½‖a − μ_θ(s)‖²
                              s.t.  c̄_i(s) + g(s;w_i)ᵀ a ≤ C_i   for all i in [K]         (4)

closed form              λ_i* = [ ( g(s;w_i)ᵀ μ_θ(s) + c̄_i(s) − C_i )
                                  / ( g(s;w_i)ᵀ g(s;w_i) ) ]_+                              (5)
                         a*   = μ_θ(s) − λ_{i*}* · g(s;w_{i*}),   i* = argmax_i λ_i*        (6)
```

The paper also gives the penalty alternative it rejects:
`argmin_a ½‖a − μ_θ(s)‖² + Σ_i λ_i [c_i(s,a) − C_i]_+` with λ_i as hyper-parameters (eq. 9) —
an *approximate* solution to the same projection, which is why they prefer the exact
closed form.

**Guarantee / cost.** Soft: per-step constraint satisfaction is only as good as the *learned*
linear model g, so no formal invariance; they report DDPG never violating constraints during
training in their domains. Cost is one extra matrix–vector product per step (negligible) plus
an offline pre-training pass.

**For us.** This is our hook's ancestor: our barrier condition, turned into an allowed
sin(heading-error) interval and then into a bank interval via the lag, is exactly Dalal's
"one active constraint → closed-form scalar projection" pattern, except our g is analytic
rather than learned. Dalal says nothing about laziness — the paper trains DDPG *through* the
layer with the standard tuple and never measures unfiltered performance, so it is a precedent
for the architecture and not for the training question. For gating: Dalal's λ_i* ≥ 0 form
already *is* a soft gate (the layer is the identity whenever the constraint is slack), which is
a cleaner gate than a state-predicate switch because it is continuous in the state.

### 1.2 Pham, De Magistris, Tachibana (2018) — *OptLayer: Practical Constrained Optimization for Deep RL in the Real World*

- arXiv:1709.07643v2 (22 Sep 2017, rev 23 Feb 2018). ICRA 2018.
- Local: `papers/pham2018_optlayer_1709.07643.pdf`
- **This is the key paper for the laziness question in cluster B.**

**Mechanism.** OptLayer is a differentiable QP layer (OptNet-style, solved by a fixed 10-iteration
primal–dual interior-point method for batching) placed after the policy network. It takes the
raw prediction ã and returns the closest action a* satisfying the affine constraints. The
contribution that matters here is not the layer but §V: OptLayer *also* returns the amount by
which the raw prediction violated the constraints, and this quantity is fed back as a **reward
penalty on the un-projected action**, while the projected action is credited with the true
reward. The paper compares four training strategies and this is the one that wins.

**Core formulas.**

```
QP                     min_x ½ xᵀPx + qᵀx   s.t.  Gx ≤ h,  Ax = b                       (6–8)
action objective       min_x ½‖x − ã_i‖²   →  P = I,  q = −ã_i                          (9)
                       a* = the QP solution, i.e. the closest constraint-satisfying action

violation costs        c_eq = ‖A ã − b‖                                                  (27)
                       c_in = ‖ max(G ã − h, 0) ‖                                        (28)
                       c    = c_eq + c_in           (rows of A,G normalised to unit norm)

penalised reward       r̃_i = r_i − c_i

training strategies (BuildTraj tuples are (s_i, action, reward, value)):
  UP  Unconstrained Predictions        —  execute ã, learn on (s, ã,  r )
  CP  Constrained, learn Predictions   —  execute a*, learn on (s, ã,  r )
  CC  Constrained, learn Corrections   —  execute a*, learn on (s, a*, r )
  CPC Constrained, learn Predictions
      AND Corrections                  —  execute a*, then update the network TWICE:
                                          first on (s, ã,  r̃ = r − c)   [penalised raw action]
                                          then  on (s, a*, r )           [credited safe action]
```

Their own gloss: "CP amounts to considering OptLayer separately from the network and as part of
the environment. In the CC strategy, we try to learn corrected actions directly … With CPC, we
first associate raw predictions with discounted rewards before associating corrected actions
with better rewards." Reported result: CPC reaches a given average reward in far fewer
episodes than UP/CP/CC while never colliding.

**Guarantee / cost.** Hard on the *modelled* affine constraints (exact QP projection, subject
to feasibility). Cost: one small QP per step forward, plus the OptNet backward pass, plus a
*second* network update per batch for CPC.

**For us.** CPC is the single most transferable idea in this cluster. Our current arm updates
the network only through the post-filter rollout — that is CC, the strategy OptLayer found
weakest for learning the prediction. The CPC analogue for a regression/rollout loss is: keep
the trajectory loss on the *filtered* rollout (the thing that is deployed), and add a second
term that penalises the *pre-filter* bank command by the amount the filter had to move it —
i.e. `L = L_traj(rollout(u_filtered)) + λ · Σ_gated ‖u_raw − u_filtered‖²`, with the second term's
gradient flowing only into the raw head. That directly attacks laziness: the network is
rewarded for the filtered trajectory but *charged* for needing the filter. Note that OptLayer
normalises the rows of G, h before computing c — for us that means normalising the barrier
row so the penalty is in comparable units across d (the corridor half-width hw(d) shrinks with
d, so an un-normalised violation cost would be dominated by the far-field).

### 1.3 Cheng, Orosz, Murray, Burdick (2019) — *End-to-End Safe RL through Barrier Functions for Safety-Critical Continuous Control Tasks*

- arXiv:1903.08792 (21 Mar 2019). AAAI 2019.
- Local: `papers/cheng2019_e2e_safe_rl_cbf_1903.08792.pdf`

**Mechanism.** Two architectures. (a) The naive one: u_k = u^RL_θk + u^CBF_k(s, u^RL_θk), where the
CBF-QP computes the minimal additive correction; here the CBF only compensates and "does not
guide exploration" — this is precisely our deployed hook, and the paper calls it *suboptimal*.
(b) The one they propose: at each policy iteration the RL policy is updated **around the
previously deployed (already-corrected) controller**, so the accumulated CBF corrections become
part of what the RL policy is expected to produce. Because storing k−1 past networks is
impractical, the accumulated compensation is **distilled into a feedforward MLP u^bar_φ by
supervised regression** on the previous iteration's rollouts. Safety is unaffected by the
approximation because the final QP still filters whatever the two learned terms produce.

**Core formulas.**

```
discrete barrier condition used   h(s_{t+1}) ≥ (1 − η) h(s_t) − ε          (11–12)
                                  (with h^ε(s) = h(s) + ε/η, so h^ε(s_{t+1}) ≥ (1−η) h^ε(s_t))

naive compensating controller     u_k(s) = u^RL_θk(s) + u^CBF_k(s, u^RL_θk)             (13)

CBF-QP (GP-uncertain dynamics, actuator limits):
      (a_t, ε) = argmin_{a_t, ε}  ‖a_t‖² + K·ε
      s.t.  pᵀ f(s_t) + pᵀ g(s_t)[ u^RL_θk(s_t) + a_t ] + pᵀ μ_d(s_t)
                 − k_δ |p|ᵀ σ_d(s_t) + q  ≥ (1 − η) h(s_t) − ε
            a_low^i ≤ a_t^i + u^RL(i)_θk(s_t) ≤ a_high^i,  i = 1..M      (14)

barrier-GUIDED controller (the anti-laziness one):
      u_k(s) = u^RL_θk(s) + Σ_{j=0}^{k−1} u^CBF_j(s, u^RL_θ0,…,u^RL_θ(j−1))
                          + u^CBF_k(s, u^RL_θk + Σ_{j<k} u^CBF_j)                        (15)
      QP (16) is (14) with u^RL_θk(s_t) replaced by u^RL_θk(s_t) + Σ_{j<k} u^CBF_j(s_t)

practical form via distillation:
      u^bar_φk(s) ≈ Σ_{j=0}^{k−1} u^CBF_j(s, ·)        [MLP fitted by supervised regression
                                                        to the previous iteration's corrections]
      u_k(s) = u^RL_θk(s) + u^bar_φk(s) + u^CBF_k(s, u^RL_θk + u^bar_φk)
```

Theorem 2 additionally gives a TRPO performance bound
J(π_k^prop) ≥ J(π_{k−1}) − 2λγ/(1−γ)² · δ_π for the proposed (pre-final-QP) controller.

**Guarantee / cost.** Probabilistic-hard: forward invariance of C with probability 1−δ if the
QP is feasible with ε_max = 0; graceful degradation to the ε_max-inflated set otherwise. Cost:
one QP per step + one extra MLP + a supervised fit per policy iteration.

**For us.** This is the *cleanest* structural answer to laziness in a rollout setting: don't
merely penalise the correction, **make the network's own output absorb it**. The transfer is
direct because our filter is deterministic and differentiable — we can run one epoch in which
the target for the pre-filter bank head is the *filtered* bank the hook produced on the previous
epoch's rollouts (a self-distillation / DAgger step), so at convergence the raw head already
outputs the corridor-respecting bank and the hook clips ~0 %. Note their h-condition
`h(s_{t+1}) ≥ (1−η)h(s_t) − ε` is *exactly* the DCBF form our hook uses, with η ↔ α·Δt; and the
ε slack with cost K·ε is the principled version of our soft-tanh saturation.

---

## 2. Provably safe RL, shields, and "the agent exploits the shield"

### 2.1 Krasowski, Thumm, Müller, Schäfer, Wang, Althoff (2023) — *Provably Safe Reinforcement Learning: Conceptual Analysis, Survey, and Benchmarking*

- arXiv:2205.06750v3 (13 May 2022, rev 18 Nov 2023). **TMLR 2023**.
  (The task named it "Theoretical Framework and Comparison"; the published title is the above.)
- Local: `papers/krasowski2023_provably_safe_rl_2205.06750.pdf`

**Mechanism.** Categorises every provably safe RL method by *how it changes the action*:
**action replacement** (unsafe action → some safe action from a replacement strategy ψ(s), e.g.
a failsafe controller or a uniform sample from the safe set), **action projection** (unsafe
action → closest point of the safe set; a QP), **action masking** (agent only ever samples from
the safe set A_φ). Orthogonal to that, it defines four **learning tuples** — what you actually
put in the replay buffer / policy-gradient batch — and benchmarks all combinations on an
inverted pendulum and a 2D quadrotor with five RL algorithms × 10 seeds.

**Core formulas.**

```
naive              (s, a,   s', r(s, a_φ))                 learn on the RAW action, true reward
adaption penalty   (s, a,   s', r*(s, a, a_φ))
                   r*(s,a,a_φ) = r(s,a_φ) + r_penalty(s,a,a_φ)   when an unsafe action was chosen
                   for action projection, r_penalty may include a term ∝ dist(a, a_φ)
safe action        (s, a_φ, s', r(s, a_φ))                 learn on the EXECUTED (safe) action
both               adaption penalty AND safe action tuples used together
```

**Reported results (the part we need).**
- "For both action replacement and projection, the **adaption penalty tuple leads to the highest
  performance and lowest safety intervention rate**, even outperforming the average over the
  baselines."
- "Generally, we report that a **lower intervention rate often coincides with a higher reward**."
- Action replacement beats projection and masking on average, and "relies significantly less on
  the safety mechanism than projection and masking".
- "In action projection, the naive tuple performs significantly worse than in action
  replacement. The safe action and both tuples seem to be only beneficial when using action
  projection and decrease performance when using action replacement."
- "In our experiments, a **simple constant reward penalty** already improved the performance.
  Other environments may require more careful reward tuning, or the adaption penalty tuple
  could fail altogether." The numeric value of that constant is not stated in the body text
  I extracted — treat the magnitude as `[UNVERIFIED]`.
- Their **intervention rate** metric = share of RL steps per episode in which the safety
  function altered the action (for masking: the safe-set volume ratio vs at equilibrium).

**Guarantee / cost.** Hard, by construction of the safe set (set-based reachability in their
benchmark); the survey confirms all provably safe methods were safe in training. Cost: for
projection, "the main implementation challenge is to guarantee that the optimization problem is
always feasible" — they hit infeasibility from small numerical errors and had to fall back on
the previous solution or a failsafe controller.

**For us.** Two directly usable things. (1) Our 15–24 % clip rate **is** their intervention
rate, and their headline empirical claim is that the adaption-penalty tuple simultaneously
raises reward and *lowers* intervention rate — i.e. the metric we are worried about is exactly
the one a penalty fixes. (2) Their taxonomy says our hook is *action projection*, the class they
found weakest with the naive tuple and most sensitive to the learning tuple. Their advice would
be either to add the penalty or to switch to **action replacement** — for us that would be:
when the gate is on and the raw bank is outside the allowed interval, replace it with a
*fixed failsafe* bank (e.g. the exact corridor-tracking bank) rather than the nearest feasible
one; replacement in their benchmark had a lower intervention rate than projection.

### 2.2 Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu (2018) — *Safe Reinforcement Learning via Shielding*

- arXiv:1708.08611v2 (29 Aug 2017). AAAI 2018 `[UNVERIFIED — venue not on the arXiv record]`.
- Local: `papers/alshiekh2018_shielding_1708.08611.pdf`

**Mechanism.** A shield is a *reactive system* synthesised from a temporal-logic safety
specification φ_s and an abstraction of the MDP, placed either **preemptively** (the shield hands
the learner the set of safe actions each step; Σ_O = 2^A) or **post-posed** (the learner picks,
the shield overrides if unsafe). Correctness is by construction from the automaton, not by a
model of the dynamics. The paper is the origin of the reward-bookkeeping dilemma that our
problem is an instance of.

**Core statement (verbatim structure, §"what reward for the unsafe action a¹_t").**

```
Option 1: assign a punishment r'_{t+1} < 0 to the unsafe action a¹_t.
   "The agent … learns that selecting a¹_t at state s_t is unsafe, without ever violating φ_s.
    However, there is no guarantee that unsafe actions are not part of the final policy.
    Therefore, the shield has to remain active even after the learning phase."

Option 2: assign the real reward r_{t+1} to a¹_t.
   "Therefore, picking unsafe actions can likely be part of an optimal policy by the agent.
    Since an unsafe action is always mapped to a safe one, this does not pose a problem and
    the agent never has to learn to avoid unsafe actions.
    Consequently, the shield is (again) needed during the learning and execution phases."
```

**Guarantee / cost.** Hard w.r.t. the specification and the abstraction; cost is the automaton
synthesis (offline, exponential in the specification) plus an O(1) table lookup online.

**For us.** Option 2 *is* our lazy network, written down in 2018 — the mechanism is not a
pathology of our filter, it is the expected outcome of crediting the raw command with the
filtered outcome. And Alshiekh's honest conclusion is that **neither option removes the need to
keep the shield at deployment**; the punishment only makes the policy less reliant, not
independent. Our current framing ("training through the filter should let us drop it") is
therefore fighting a known-hard problem; the literature's position is: keep the filter at
inference, and use the penalty to *reduce intervention*, not to eliminate it.

### 2.3 Hsu, Hu, Fisac (2023) — *The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems*

- arXiv:2309.05837 (11 Sep 2023). Accepted, **Annual Review of Control, Robotics, and
  Autonomous Systems**.
- Local: `papers/hsu2023_safety_filter_unified_2309.05837.pdf`

**Mechanism.** Decomposes *every* safety filter into three parts: a **safety monitor** Δ (is the
proposed control safe?), an **intervention scheme** (switch to the fallback, or optimise within
the safe set), and a **fallback safety policy** π^ℓ. Monitors are *value-based* (evaluate a
function: HJ value function, CBF) or *rollout-based* (simulate: model predictive shielding,
forward-reachable sets, tube MPC). Their claim is that any provably correct filter can be
analysed via a "Universal Safety Filter Theorem" (§4), and every filter can be seen as relying
on some fallback policy.

**Core formulas.**

```
CBF (zeroing)     sup_{u∈U}  ∇h(x)ᵀ f^c(x,u)  ≥  −α(h(x)),   for all x ∈ Ω = {x : h(x) ≥ 0}   (12)

CBF-QP filter     φ(x, π^task(x)) = argmin_{u∈U} ½ ‖u − π^task(x)‖²                          (13a)
                  s.t.  ∇h(x) f̄(x) + ∇h(x) ḡ(x) u + α(h(x)) ≥ 0                            (13b)
```

**§3.5.3 "Filter-aware task policies" — the section that names our problem.** Verbatim: "some
efforts have investigated endowing task policies with safety filter awareness. These approaches
relax the safety–performance separation paradigm, allowing the robot's task-driven decisions to
explicitly account for the safety filter's behavior; the goal is to avoid unnecessarily
triggering interventions that may negatively impact performance." The works it cites for this
(useful leads, **not read here**, so `[UNVERIFIED]` beyond the citation):
- **Akametalu et al.** — integrates the HJ safety value function V into the RL *reward signal*,
  "leading to a reduced frequency of safety filter interventions, which was shown to benefit
  the learning process";
- **Leung et al.** — filter-aware planning for safety-critical human–robot interaction in
  driving with an HJ filter; the MPC planner "gently swerves" and the trajectory is
  "safe and chatter-free";
- **Hu et al.** — SHARP, shielding-aware robust planning, generalised to a broader filter class,
  with follow-up work combining filter awareness with active uncertainty reduction.

**Guarantee / cost.** Survey; the guarantee is whatever the underlying filter provides. It is
explicit that CBF filters do *not* encode the maximal safe set Ω* (unlike the HJ value
function), i.e. a CBF filter is generally **more restrictive than necessary** — which matters
for §2.4 below.

**For us.** (i) The three-part decomposition is a good way to write up our hook: monitor =
gate + h(x) ≥ 0 test, intervention = optimise (clip to the allowed bank interval), fallback =
the corridor-tracking bank at the interval endpoint. (ii) The single most useful pointer in the
cluster is that "reduce filter interventions" is a *named research direction* with three prior
works, and the earliest of them (Akametalu) does it by putting the safety value function
directly into the reward — which for us is the differentiable-loss route of §6.1, not a filter.

### 2.4 Oh, Nguyen, Hu, Fisac (2025) — *Provably Optimal Reinforcement Learning under Safety Filtering*

- arXiv:2510.18082v2 (20 Oct 2025, rev 11 Feb 2026). Accepted, IASEAI 2026.
- Local: `papers/provably_optimal_rl_safety_filtering_2510.18082.pdf`

**Mechanism.** Formalises safe RL as a **safety-critical MDP** (SC-MDP) requiring *categorical*
(not high-probability) avoidance of a failure set F, and pairs it with a **filtered MDP** M_φ in
which the filter is folded into the environment ("bubble-wrapped"; the agent is agnostic to the
filter). Then proves that if the filter is *perfect* (least-restrictive: it intervenes iff the
action would leave the maximal controlled-invariant set Ω*), learning in M_φ is safe, standard
RL convergence carries over, and an ε-optimal policy on M_φ, **executed through the same
filter**, is ε-optimal among all safe policies of the SC-MDP.

**Core formulas.**

```
Perfect (least-restrictive) safety filter φ : S × A → A, for all s ∈ Ω*:
   1.  φ(s, a) ∈ A_safe(s)   for all a ∈ A
   2.  φ(s, a) = a           for all a ∈ A_safe(s)                                  (Def. 2)

Filtered MDP   M_φ := (Ω*, A, P_φ, r_φ, γ)
   P_φ(· | s,a) := P(· | s, φ(s,a)),     r_φ(s,a) := r(s, φ(s,a))                   (Def. 3)

Executed policy (pushforward of the learned policy through the filter)
   π_exec(B | s) := π_φ^ε ({ a ∈ A : φ(s,a) ∈ B } | s)                              (8)

Theorem 1(3):  if V^{π_φ^ε}_{M_φ}(s) ≥ V*_{M_φ}(s) − ε for all s ∈ Ω*,     (7)
               then V^{π_exec}_{M_SC}(s) ≥ V*_{M_SC}(s) − ε for all s ∈ Ω*  (9)
```

Assumption 1 requires Borel spaces, closed nonempty Ω*, **safe initialisation (all training
episodes start in Ω*)**, stationary bounded rewards, and existence of a measurable
time-invariant perfect filter. Validated on Safety Gymnasium with zero training violations and
final performance matching or exceeding unfiltered baselines.

**Guarantee / cost.** Hard and categorical, but *conditional on the filter being perfect*.
Their stated recipe: "train and deploy RL policies with the **most permissive** safety filter
that is available."

**For us — the sharpest diagnostic in this cluster.** The theorem says the safety–performance
separation is complete *only* when the filter is least-restrictive and *only* when the same
filter is deployed. Our filter is neither maximal (a CBF corridor with a hand-picked α and a
hand-picked k is strictly inside the true viability kernel — Hsu §3.2 says exactly this) nor
gated the way the theorem assumes. So the correct reading of our six-arm result is **not** "training
through a filter is bad", it is "**our filter is not permissive enough**". Two concrete
consequences: (a) before adding penalties, try enlarging α (a larger α = a later, softer
intervention) and check whether the clip rate falls without the corridor being violated;
(b) their Assumption 1.2 (safe initialisation) has a direct analogue we may be violating — a
vectored flight that has not yet turned final can enter the gate already outside the corridor,
which is an infeasible start, and infeasible starts are exactly where projection filters
misbehave (Krasowski) and where the theorem's guarantee does not apply. This is the most likely
explanation for *why the vectored flights got worse specifically*.

### 2.5 Brunke, Greeff, Hall, Yuan, Zhou, Panerati, Schoellig (2022) — *Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning*

- arXiv:2108.06266v2 (13 Aug 2021, rev 6 Dec 2021). **Annual Review of Control, Robotics, and
  Autonomous Systems**, 2022.
- Local: `papers/brunke2022_safe_learning_robotics_2108.06266.pdf`

**Mechanism.** Organises the field by a three-level safety scale and by whether learning enters
the model, the policy, or the filter.

**Core formulas.**

```
Safety Level III (hard):     c_k^j(x_k, u_k, w_k) ≤ 0            for all k, j              (3)
Safety Level II  (chance):   Pr( c_k^j(x_k, u_k, w_k) ≤ 0 ) ≥ p^j                          (4)
Safety Level I   (soft):     c_k^j(x_k, u_k, w_k) ≤ ε^j,  plus a penalty l(ε) ≥ 0 in the
                             objective with l(ε)=0 iff ε=0                                 (5)

Certified learning / safety filter (their unified form):
     u_safe,k = argmin_{u_k ∈ U_c}  ‖ u_k − u_learn,k ‖²                                  (16a)
     s.t.  x_{k+1} = f̄_k(x_k,u_k) + f̂_k(x_k,u_k,w_k) ∈ Ω_safe                            (16b)

Safety-layer form they attribute to Dalal and to Pham:
     u_safe,k = argmin_{u_k} ‖u_k − u_learn,k‖₂²    (a differentiable QP; single-step
                                                     state constraints only)
```

**Guarantee / cost.** Survey. The useful distinction it draws for us: safety layers of the
Dalal/OptLayer kind "consider **single time-step** state constraints", whereas trajectory-level
constraints need the methods of their §3.2.3.

**For us.** Our hook is a single-step (per-segment) constraint on a *64-segment rollout* — i.e.
we are using a Level-III single-step device to enforce what is really a trajectory-level
requirement over the gated tail of the approach. Brunke's Level I formulation (slack ε plus a
penalty l(ε) in the objective) is the cheapest alternative and is exactly what Cheng's ε with
cost K·ε and what §6.1's certificate loss do. If we want the corridor to bind over the whole
gated tail rather than segment-by-segment, the survey points at MPC-style formulations, i.e.
the predictive safety filters of §3.

---

## 3. Predictive safety filters, and training *through* them

### 3.1 Wabersich, Zeilinger (2021) — *A predictive safety filter for learning-based control of constrained nonlinear dynamical systems*

- arXiv:1812.05506v4 (13 Dec 2018, rev 17 May 2021). **Automatica** (as stated in the task;
  the arXiv record carries no journal ref, so the exact volume/pages are `[UNVERIFIED]`).
- Local: `papers/wabersich2021_predictive_safety_filter_1812.05506.pdf`

**Mechanism.** Instead of a precomputed invariant set, safety is certified *online* by finding a
**backup plan**: at each step, solve an MPC-like feasibility problem that (a) applies something
as close as possible to the learning input u_L now, and (b) proves a horizon-N input sequence
exists that keeps the state in X and U and lands in a known safe terminal set S^t. If it is
feasible, u_L is certified and applied; if not, the previous step's plan is followed with a
shortened horizon; if that runs out, a terminal safe policy π_S^t takes over. The uncertain
version propagates a nominal mean μ with a probabilistic tube and restricts the plan to the
model-confident region Z_c.

**Core formulas.**

```
Definition 3.1 (certified input): u_L(k̄) is certified safe if π_S(k̄, x(k̄), u_L(k̄)) = u_L(k̄)
and applying u(k)=π_S(k,x(k),u_L(k)) for k ≥ k̄ implies safety for all time.
The filter "aims at the smallest possible modification by, e.g., minimising
‖π_S(k, x(k), u_L(k)) − u_L(k)‖²."

Nominal PSF online problem:
   min_{u_{i|k}}  ‖ u_L − u_{0|k} ‖                                             (5a)
   s.t.  x_{i+1|k} = f(x_{i|k}, u_{i|k}; θ̄)                                     (5b)
         x_{i|k} ∈ X                                                            (5c)
         u_{i|k} ∈ U                                                            (5d)
         (x_{i|k}, u_{i|k}) ∈ Z_c        [stay in the model-confident set]       (5e)
         x_{N|k} ∈ S^t                   [terminal safe set]                    (5f)
         x_{0|k} = x(k)                                                         (5g)

Algorithm 1: if (5) feasible for horizon N → return u*_{0|k,N}, set k̄ := k;
             else if k < N + k̄        → solve (5) for horizon N − (k − k̄);
             else                      → return π_S^t(x(k)).
```

The uncertain version (6) replaces x by the nominal mean μ, X/U by tightened X̄_i/Ū_i, and adds
the probabilistic tube constraint E_{p_S}(μ_{i|k}, v_{i|k}) ⊆ Ē_i^γ.

**Guarantee / cost.** Hard in the nominal case (recursive feasibility from the terminal set);
"safe in probability" at a chosen level in the uncertain case. Cost: an NMPC solve per step.

**For us.** The important design lesson is the **terminal safe set + horizon shortening**
recursive-feasibility machinery. Our hook has no analogue: if the gate turns on when the
aircraft is already outside the corridor there is no feasible bank, and we currently just clamp.
That is the classic projection-infeasibility failure Krasowski warns about. A cheap PSF-flavoured
fix without an NMPC solve: define a terminal condition (aligned with the corridor, within the
glidepath window, at the gate distance) and, when the barrier is infeasible for the current
segment, fall back to a *fixed corridor-capture* bank profile rather than a clamp.

### 3.2 Tearle, Wabersich, Carron, Zeilinger (2021) — *A predictive safety filter for learning-based racing control*

- arXiv:2102.11907 (23 Feb 2021). RA-L / ICRA `[UNVERIFIED — no journal ref on arXiv]`.
- Local: `papers/tearle2021_psf_racing_2102.11907.pdf`

**Mechanism.** The PSF instantiated for a 1:28-scale RC car on a track: the state constraint is
"stay inside the track boundaries", the terminal set is a curvature-parametrised invariant
ellipsoid E(P) verified by an SDP and shrunk until no violating point is found. The filter's
objective is *only* the deviation from the desired input; secondary objectives can be added with
weights chosen so W ≫ R_S "to ensure priority remains on tracking the desired input".

**Core formulas.**

```
   min_{x_{i|k}, u_{i|k}}  J(u_{i|k}, u_d(k))                                (3a)
   s.t.  x_{0|k} = x(k);  x_{i+1|k} = f(x_{i|k}, u_{i|k});
         x_{i|k} ∈ X;  u_{i|k} ∈ U;  x_{N|k} ∈ S_f                          (3b–3f)
   J(u_{i|k}, u_d(k)) = ‖ u_d(k) − u_{0|k} ‖²                                 (4)
   π_S(x(k), u_d(k)) = u*_{0|k}

Terminal-set synthesis:   min_{E,Y} − log det E   s.t. LMIs (18b–18e)
Terminal-set validation:  max over x̄_r, c  s.t.  x̄_rᵀ P x̄_r ≤ 1,
                          x̄_r(k+1,c) = f(x̄_r(k), κ_f(k), c),  c ∈ [c_min, c_max]  (19)
                          → invariant iff the optimal value < 1
```

The terminal-set curvature is taken from the track **a heuristic distance ahead**, chosen as a
function of the current desired torque and the horizon time t_N = N·T_s.

**Guarantee / cost.** Hard (recursive feasibility) subject to the model and the verified terminal
set. Cost: a nonlinear MPC per step at the car's control rate.

**For us.** The most transferable detail is the *lookahead-parameterised terminal set*: the
terminal invariant set is indexed by the track curvature a distance ahead that scales with the
current command. Our corridor half-width hw(d) is linear in distance-to-threshold, so the
"track" narrows monotonically — a terminal set defined at the *threshold* rather than at the
current d is what a PSF would use, and it is stricter than our per-segment condition. Also
worth copying: they use different filter horizons for training (M=1) and evaluation (M=2) in
the sibling paper §3.3 — i.e. a deliberately *weaker* filter during training.

### 3.3 Pizarro Bejarano, Brunke, Schoellig (2025) — *Safety Filtering While Training: Improving the Performance and Sample Efficiency of Reinforcement Learning Agents*

- arXiv:2410.11671v2 (15 Oct 2024, rev 25 Nov 2024). **IEEE RA-L 2025**, DOI
  10.1109/LRA.2024.3512374. Code: github.com/Federico-PizarroBejarano/safe-control-gym.
- Local: `papers/pizarrobejarano2025_filtering_while_training_2410.11671.pdf`
- **This is the paper that measures laziness directly.**

**Mechanism.** Three separable training modifications around a model-predictive safety filter
(MPSF), ablated in every combination on a Crazyflie 2.0 in `safe-control-gym` and on real
hardware. (A) **Filtering training actions** — execute the certified action, buffer the
uncertified one; (B) **Penalising corrections** — subtract a function of the correction size
from the reward; (C) **Safe reset** — reject episode start states from which the filter is
infeasible, using the filter itself as the feasibility test.

**Core formulas.**

```
(A) on-policy buffered quadruple:
        ( x_k , u_uncert,k , f(x_k, u_cert,k) , R(x_k, u_cert,k) )
    (off-policy may instead buffer u_cert,k, treating the filter as an expert)

(B) correction-penalised reward:
        R_α(x_k, u_uncert,k, u_cert,k) = R(x_k, u_applied,k) − α · P(u_uncert,k, u_cert,k)
    "After experimentation, we found that penalizing the magnitude of the correction
        P = ‖ u_uncert,k − u_cert,k ‖₂²
    led to the best results in our experiments."
    Baselines instead use a constraint-violation penalty  r_k^β = r_k − 1_viol · β.

(C) safe reset: sample x_0 ~ S, keep it iff the filter optimisation is feasible from x_0
    (i.e. x_0 is in the H-step robust positively control invariant set).

Chattering metric: δu_k = (u_k − u_{k−1})/δt, Δu = [δu_1,…,δu_{K−1}], report ‖Δu‖_F.
```

**Reported results (Table I, simulation; mean ± std).**

| Metric | Std β=0 | Std β=0.1 | Safe α=0.1 | Safe α=1 | Safe α=10 | Safe α=100 |
|---|---|---|---|---|---|---|
| Return (with filter) | 200.2±17.1 | 210.3±14.4 | 212.6±13.8 | **214.1±14.0** | 210.7±14.0 | 202.4±13.4 |
| **Return when uncertified** | 222.1±13.7 | 222.2±10.9 | **11.2±15.7** | **31.2±44.8** | **210.7±50.0** | 202.3±19.1 |
| Input rate of change | 16.4±17.0 | 4.7±2.9 | 9.4±2.9 | 7.5±3.3 | 3.8±3.0 | 3.0±3.1 |
| Training violations [%] | 82.8±6.6 | 67.0±4.2 | 0.23 | 0.22 | 0.22 | 0.22 |
| Training time / step [ms] | 2.3±0.3 | 2.8±0.4 | 14.6±1.2 | 11.6±1.5 | 13.5±1.0 | 12.6±1.1 |

The **"Return when uncertified" row is the laziness measurement**: with a weak correction
penalty (α = 0.1, 1) the trained policy collapses without the filter (11.2 and 31.2 vs 212.6 and
214.1 with it); at α = 10 the uncertified return recovers to 210.7, i.e. **the dependence is a
monotone function of the penalty weight and is removable by tuning α**. Their own text: "the RL
agent may grow dependent on the safety filter and thus not perform well without the safety
filter. This is not a problem if the safety filter is used during evaluation, as assumed in this
paper, and can be mitigated using the following approach [penalising corrections]." And later:
"the approaches trained with the safety filter have the same or better return with the safety
filter. This highlights the importance of not removing the safety filter, even when penalizing
corrections, as safety guarantees are lost and performance may decrease."

Other findings: combining all three modifications gives the best return, best convergence and
fewest training violations; it reaches 80 % of final return **10× faster** and passes a return
of 200 with 2.5× fewer environment interactions; but training is **5× slower per step**.
Increasing α or β monotonically reduces input rate of change (chattering).

**Guarantee / cost.** Hard while the MPSF is applied (subject to the robust MPC assumptions,
with softened constraints in practice — hence 0.22 % rather than 0 % training violations).
Cost: 5× training-step time with the full recipe.

**For us — the closest match to our experiment in the whole cluster.** (1) It reproduces our
six-arm finding and then *fixes* it: the fix is a correction penalty on the **squared L2
correction magnitude**, and the weight matters more than anything else. Our arms should be
re-run as a sweep in α, not as a binary "with/without hook", and the single number to report per
arm is the pair (metric with hook, metric without hook) — their "return / return when
uncertified". (2) Their safe-reset modification is our infeasible-gate-entry problem: a
vectored flight entering the gate already outside the corridor is exactly an infeasible start
state, and safe reset "significantly improves convergence". We cannot resample real flights, but
we can (a) *not gate on* until the state is corridor-feasible, or (b) down-weight the loss on
segments where the barrier was infeasible at gate entry — this is the most plausible single fix
for "vectored flights got worse". (3) Their filter horizon is deliberately weaker in training
(M=1) than in evaluation (M=2).

### 3.4 Yang, Werner, de Sa, Ames (2026) — *CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions*

- arXiv:2510.14959v6 (16 Oct 2025, rev 22 Jun 2026). Accepted, **ICRA 2026**.
- Local: `papers/cbfrl_2510.14959.pdf`

**Mechanism.** "Dual" scheme: apply a closed-form CBF safety filter to the policy's action
during training **and** add a barrier-inspired reward term, so the constraint is *internalised*
and the policy can be deployed without a runtime filter. Because the CBF-QP has a single linear
constraint it is solved in closed form (no solver in the massively-parallel IsaacLab loop). A
theorem justifies using the *continuous-time* CBF condition on discrete-time rollouts for small
Δt, with an explicit O(μ(Δt)/Δt) residual.

**Core formulas.**

```
Theorem 1 (continuous → discrete):
    h(q_k) ≥ (1 − Δt·α)^k h(q_0) − μ(Δt)/(Δt·α),   for all k ≥ 0,   (1 − Δt·α) ∈ [0,1)   (15)
    with μ(Δt) → 0 faster than Δt, so the standard DTCBF bound h(q_k) ≥ (1−Δt α)^k h(q_0)
    is recovered as Δt → 0.

Training-time filter (single-linear-constraint QP):
    v_k^safe = argmin_{v_k} ½ ‖ v_k − v_k^policy ‖²                                      (18)
    s.t.  ∇h(q_k)ᵀ v_k ≥ −α h(q_k)                                                       (19)

Closed form, with a_k := ∇h(q_k),  b_k := −α h(q_k):
    v_k^safe = v_k^policy                                        if a_kᵀ v_k^policy ≥ b_k
             = v_k^policy + ( (b_k − a_kᵀ v_k^policy) / ‖a_k‖² ) a_k     otherwise       (20–21)

Safety reward:
    r_cbf(q_k, v_k) = min( a_kᵀ v_k^policy − b_k , 0 )                                   (22)
                    + exp( − ‖ v_k^policy − v^safe ‖² / σ² ) − 1                         (23)
    total reward:   r = r_nominal + r_cbf
```

Their gloss: "this reward penalizes actions whenever the safety filter is activated, and also
incentivizes the model to take actions as close to the safe actions as possible **to reduce the
intervention of the filter**. Because the penalty term exp(…) − 1 is strictly lower-bounded by
−1, it provides a smooth learning signal without causing unbounded instability with respect to
the primary task rewards."

Ablation over four variants (Dual, Reward-only, Filter-only, Nominal), 1500 steps × 4096 parallel
envs, plus an evaluation of policies trained with the filter and then **deployed without one**:
"the Filter Only approach performs well only with an active safety filter", whereas Dual reaches
the goal with and without the runtime filter, on a Unitree G1 humanoid climbing stairs and
avoiding obstacles.

**Guarantee / cost.** Hard-ish during training (closed-form CBF filter with the discretisation
residual quantified by Thm 1); **no guarantee at deployment** if the filter is dropped — the
claim there is empirical internalisation. Cost: a couple of dot products per step.

**For us.** The single most directly copyable formula in the cluster. Two properties of their
reward are worth stealing exactly: (i) it is **two-part** — one term linear in the barrier
violation of the *raw* action (min(a·v_raw − b, 0), zero when the raw action was already safe),
one term bounded in [−1, 0] measuring how far the filter had to move it; (ii) the second term is
a **bounded** Gaussian-shaped penalty, so it cannot swamp the trajectory loss — which matters for
us because our trajectory loss is an ADE/FDE in metres and an unbounded bank-correction penalty
would need careful scaling. Also note their filter is exactly our closed form (single linear
constraint → scalar shift along the gradient), which makes the mapping mechanical.

---

## 4. Differentiable / discrete-time CBFs inside the policy

### 4.1 Xiao, Hasani, Li, Rus / Xiao, Wang, Hasani, Chahine, Amini, Li, Rus (2021/2023) — *BarrierNet*

- arXiv:2111.11277 (22 Nov 2021), titled **"BarrierNet: A Safety-Guaranteed Layer for Neural
  Networks"** (authors Wei Xiao, Ramin Hasani, Xiao Li, Daniela Rus). The journal version is
  **"BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control",
  IEEE Transactions on Robotics 39(3):2289–2307, 2023**, with the expanded author list
  Xiao, Wang, Hasani, Chahine, Amini, Li, Rus `[venue/pages taken from the search record, not
  from the PDF — the local PDF is the arXiv preprint]`.
- Local: `papers/xiao2023_barriernet_2111.11277.pdf`

**Mechanism.** A high-order CBF (HOCBF) constraint is normally a *hard, fixed* constraint whose
class-K functions α_i are hand-tuned and typically over-conservative. BarrierNet multiplies each
class-K function by a **positive penalty function p_i(z) that is an output of the upstream
network**, then embeds the resulting HOCBF constraint in a differentiable QP layer (OptNet-style,
KKT-differentiated). The QP's cost matrix H(z|θ_h) and linear term F(z|θ_f) — F reading as a
learned reference control — are also network outputs. Result: safety is guaranteed for any
p_i > 0, but *how conservative* the filter is becomes a learned, state-dependent quantity.

**Core formulas.**

```
HOCBF sequence (standard):    ψ_i(x) := ψ̇_{i−1}(x) + α_i(ψ_{i−1}(x)),  i = 1..m,  ψ_0 = b(x)  (2)
                              C_i := { x : ψ_{i−1}(x) ≥ 0 }                                    (3)
HOCBF condition:  sup_{u∈U} [ L_f^m b(x) + (L_g L_f^{m−1} b(x)) u + O(b(x)) + α_m(ψ_{m−1}(x)) ] ≥ 0  (4)

*** the differentiable relaxation ***
       ψ_i(x,z) := ψ̇_{i−1}(x,z) + p_i(z) · α_i(ψ_{i−1}(x,z)),   i = 1..m                       (6)
       with p_i : R^d → R_{>0} the OUTPUT OF THE PREVIOUS NETWORK LAYER, Lipschitz continuous
       (this is the AdaCBF form, but trainable and with no auxiliary dynamics to design)

BarrierNet neuron (Def. 7):
       u*(t) = argmin_u  ½ uᵀ H(z|θ_h) u + Fᵀ(z|θ_f) u                                          (9)
       s.t.  L_f^m b_j(x) + (L_g L_f^{m−1} b_j(x)) u + O(b_j(x), z|θ_p)
                  + p_m(z|θ_{p_m}) α_m(ψ_{m−1}(x, z|θ_p))  ≥ 0,   for all j ∈ S               (10)
             u_min ≤ u ≤ u_max,   t = kΔt + t_0
       trainable parameters θ = { θ_h, θ_f, θ_p = (θ_{p1},…,θ_{pm}) }

Backward pass (Amos & Kolter KKT differentiation), with λ the duals on the HOCBF constraints:
       ∇_H ℓ = ½ (d_u uᵀ + u d_uᵀ),      ∇_F ℓ = d_u
       ∇_G ℓ = D(λ*)(d_λ uᵀ + λ d_uᵀ),   ∇_h ℓ = −D(λ*) d_λ                                    (11)
       [ d_u ; d_λ ] = [ H  Gᵀ D(λ*) ; G  D(Gu* − h) ]^{-1} [ (∂ℓ/∂u*)ᵀ ; 0 ]                   (13)
       G_j = − L_g L_f^{m−1} b_j(x)
       h_j = L_f^m b_j(x) + O(b_j(x), z) + p_m(z) α_m(ψ_{m−1}(x, z))                            (12)
       and  ∇_{p_i} ℓ = ∇_{h_j} ℓ · ∇_{p_i} h_j
```

Note ∇_G ℓ is *not applied* in a BarrierNet (G is fixed by the HOCBF); only h — through p_i — is
learned. Control bounds are excluded from G, h since they are not trainable.

**Guarantee / cost.** Hard: the constraint (10) is enforced for every p_i > 0, so safety holds
for *any* setting of the learned parameters, including mid-training. Cost: a QP solve plus a KKT
linear solve per forward/backward step.

**For us — the alternative to penalising the filter: make the filter's conservativeness a network
output.** Our hook has two hand-picked numbers (the corridor factor k and the class-K gain α) and
a hand-picked gate. BarrierNet says: keep the barrier h = k·hw(d) − |x_t| exactly as it is, but
let the network emit p(z) > 0 (e.g. softplus of a head, conditioned on the 120 s lookback and the
current d) multiplying α. The network then *cannot* be lazy in the harmful sense — it can only
choose how early the corridor starts pushing, and it is still bound by the constraint. This also
subsumes the gate: p(z) → small means the barrier is nearly inactive, which is a *learned,
differentiable* gate rather than a state predicate with a discontinuity at the boundary, and it
removes the gradient pathology of a hard on/off gate on a differentiable rollout.

### 4.2 Xiao, Wang, Gan, Rus (2023) — *SafeDiffuser: Safe Planning with Diffusion Probabilistic Models*

- arXiv:2306.00148 (31 May 2023). Project page safediffuser.github.io.
- Local: `papers/xiao2023_safediffuser_2306.00148.pdf`

**Mechanism.** Treats the *denoising* direction as a control system (τ̇ = u) and imposes a CBF
condition on it, so that the specification b(x_k) ≥ 0 becomes satisfied within a *finite* number
of denoising steps and stays satisfied thereafter. Three variants: **robust-safe** (hard, but can
trap the sample at a local minimum since safety locks in early in the diffusion), **relaxed-safe**
(adds a relaxation r_k^j with a diffusion-time-decaying weight w_k(j) → 0, plus N_a extra
denoising steps so the hard constraint only bites at the very end), **time-varying-safe**
(replaces the specification by b(x_k^j) − γ_k(j) ≥ 0 with γ_k(N) ≤ b(x_k^N) and γ_k(0) = 0 —
i.e. the constraint is *annealed in* along the diffusion).

**Core formulas.**

```
Definition 4 (finite-time diffusion invariance): there exists i ∈ {0..N} such that
    b(x_k^j) ≥ 0 for all k ∈ {0..H} and all j ≤ i.

Robust-safe:   h(u_k^j | x_k^j) := (db(x_k^j)/dx_k^j) u_k^j + α(b(x_k^j)) ≥ 0                (8)

Relaxed-safe:  h(u_k^j, r_k^j | x_k^j) := (db/dx) u_k^j + α(b(x_k^j)) − w_k(j) r_k^j ≥ 0     (9–10)
               w_k(j) ≥ 0 decays to 0 as j → 0; run N_a extra steps with j ∈ {−N_a,…,N−1};
               output τ^0 := τ^{−N_a}

Time-varying:  b(x_k^j) − γ_k(j) ≥ 0,   γ_k(N) ≤ b(x_k^N),  γ_k(0) = 0                       (11)
               h(u_k^j | x_k^j, γ_k(j)) := (db/dx) u_k^j − γ̇_k(j) + α(b(x_k^j) − γ_k(j)) ≥ 0 (12)

QP at each denoising step (relaxed version):
       u^{j*}, r^{j*} = argmin_{u^j, r^j} ‖ u^j − (τ^j − τ^{j+1})/Δτ ‖² + ‖ r^j ‖²
                        s.t. (10)                                                            (14)
```

**Guarantee / cost.** Hard at the end of the diffusion ("with almost probability 1"). Cost: one
QP per denoising step per horizon point.

**For us.** Not a diffusion model, but the *relaxed / time-varying* idea transfers directly and is
the best-argued cure for the "local trap" pathology, which is our vectored-flight failure in
disguise. Concretely: rather than switching the corridor constraint on hard at the gate, **anneal
it in** — replace `h = k·hw(d) − |x_t|` by `h_γ = k·hw(d) − |x_t| − γ(d)` with γ(d_gate) large
enough that the constraint is slack at gate entry and γ → 0 at the threshold. That makes the gate
continuous, keeps the constraint hard where it matters (short final), removes the infeasible-entry
problem, and is one line in the hook.

### 4.3 Agrawal, Sreenath (2017) — *Discrete Control Barrier Functions for Safety-Critical Control of Discrete Systems with Application to Bipedal Robot Navigation*

- Robotics: Science and Systems XIII, July 2017. DOI 10.15607/RSS.2017.XIII.073.
  Not on arXiv; downloaded from roboticsproceedings.org.
- Local: `papers/agrawal2017_discrete_cbf_rss13_p73.pdf`

**Mechanism.** The first extension of CBFs to nonlinear discrete-time systems x_{k+1} = f(x_k,u_k).
Two formulations, mirroring the two continuous-time ones. (a) A **reciprocal** DCBF B = 1/h,
which yields a condition that is *not affine in u_k* — the resulting optimisation is in general a
nonlinear program, and only a QCQP under extra structure. (b) An **exponential** DCBF, which is
the form everyone now uses.

**Core formulas.**

```
Definition 3 (reciprocal DCBF):  1/α_1(‖x_k‖_{∂S}) ≤ B(x_k) ≤ 1/α_2(‖x_k‖_{∂S})              (12)
                                 ΔB(x_k, u_k) − γ / B(x_k) ≤ 0,   γ > 0                       (13)
      with B = 1/h(x), the condition expands to a rational (non-affine in u) inequality.

Definition 4 (discrete-time EXPONENTIAL CBF) — here B(x_k) plays the role of h(x_k):
      1)  B_0 ≥ 0
      2)  there exists u_k such that
              ΔB(x_k, u_k) + γ B(x_k) ≥ 0,   for all k ∈ Z⁺,   0 < γ ≤ 1
      with ΔB(x_k, u_k) := B(x_{k+1}) − B(x_k)
      ⇒  B_k ≥ (1 − γ)^k B_0   ("hence the name Exponential Control Barrier Function")

Remark 4: unlike the continuous-time case, the CLF and CBF conditions are NOT affine in u_k
and depend on the choice of V_k and B_k.
```

**Guarantee / cost.** Hard forward invariance of the safe set for the discrete system. Cost: an
NLP (or QCQP under linear dynamics + linear h) per step.

**For us.** This is the provenance of the exact inequality our hook enforces. The Remark-4 point
is the one that bites: in discrete time the DCBF condition is generally **not affine in the
command**, which is why our hook has to invert the condition into an interval on
sin(heading error) and then map that through the actuator lag to a bank interval — that mapping
is doing the work an affine constraint would do for free, and it is where any conservativeness
(and any wrong-side clamp) enters. Worth citing when justifying the interval construction.

### 4.4 Zeng, Zhang, Sreenath (2021) — *Safety-Critical Model Predictive Control with Discrete-Time Control Barrier Function*

- arXiv:2007.11718v3 (22 Jul 2020, rev 23 Mar 2021). **American Control Conference (ACC) 2021**
  `[page numbers/DOI UNVERIFIED]`.
- Local: `papers/zeng2021_mpc_dcbf_2007.11718.pdf`

**Mechanism.** Puts the DCBF condition on **every step of an MPC horizon** rather than only the
current step, so the controller anticipates the barrier instead of reacting at its boundary.
Compared against MPC with plain distance constraints (MPC-DC) and against the myopic
one-step DCLF-DCBF QP; feasibility is discussed geometrically as the intersection of the
reachable set R_k and the DCBF-admissible set S_cbf,k along the horizon.

**Core formulas — the canonical DCBF condition.**

```
       Δh(x_k, u_k) ≥ − γ h(x_k),        0 < γ ≤ 1,      Δh(x_k,u_k) := h(x_{k+1}) − h(x_k)   (6)
   ⇔   h(x_{k+1}) ≥ (1 − γ) h(x_k)
       "the lower bound of the control barrier function h(x) decreases exponentially with the
        rate 1 − γ."

   Remark 1: γ could also be a class K function satisfying 0 < γ(h(x)) ≤ h(x); the scalar form
   is kept for notational simplicity.

One-step DCLF-DCBF QP (for contrast):
       min_{(u_k, δ)}  u_kᵀ H(x) u_k + l δ²
       s.t.  ΔV(x_k,u_k) + α V(x_k) ≤ δ                                                        (9b)
             Δh(x_k,u_k) + γ h(x_k) ≥ 0                                                        (9c)
             u_k ∈ U                                                                           (9d)
       safe set C = {x : h(x) ≥ 0} invariant if h(x_0) ≥ 0 and 0 < γ ≤ 1.

MPC-CBF:
       J*_t(x_t) = min_{u_{t:t+N−1|t}}  p(x_{t+N|t}) + Σ_{k=0}^{N−1} q(x_{t+k|t}, u_{t+k|t})   (10a)
       s.t.  x_{t+k+1|t} = f(x_{t+k|t}, u_{t+k|t}),  x_{t+k|t} ∈ X,  u_{t+k|t} ∈ U,
             x_{t|t} = x_t,   x_{t+N|t} ∈ X_f,
             Δh(x_{t+k|t}, u_{t+k|t}) ≥ − γ h(x_{t+k|t}),   k = 0,…,N−1                       (10f)
```

**Guarantee / cost.** Hard when feasible; the paper is explicit that feasibility of (10f) along a
horizon is *not* guaranteed and depends on the intersection of R_k with S_cbf,k — larger γ
enlarges S_cbf,k. Cost: an NMPC per step.

**For us.** Two things. (1) This is the reference to cite for `h(x_{k+1}) − h(x_k) ≥ −γ h(x_k)`
in the write-up, and for reading γ as an *exponential decay rate* 1−γ on the corridor margin.
(2) Because our rollout already predicts 64 segments ahead differentiably, we are in a far better
position than a one-step filter: we can impose the DCBF condition on **every gated segment of the
predicted rollout at once**, as a differentiable penalty (§6.1), which is precisely the
MPC-CBF-vs-one-step-QP distinction. Their feasibility discussion also predicts our infeasible
gate entry: the intersection R_k ∩ S_cbf,k can be empty, and the remedy in their framing is a
larger γ (= later, gentler intervention), which matches the Oh/Fisac permissiveness reading.

---

## 5. Safe imitation learning, and residual policies as bounded corrections

### 5.1 Cosner, Yue, Ames (2022) — *End-to-End Imitation Learning with Safety Guarantees using Control Barrier Functions*

- arXiv:2212.11365 (21 Dec 2022). IEEE CDC 2022.
- Local: `papers/cosner2022_e2e_il_cbf_2212.11365.pdf`

**Mechanism.** Don't filter the learned controller at all — instead make the **expert** robustly
safe, then bound how much safety the *imitation error* can destroy. The expert is a
Tunable Robust Optimization Program (TR-OP) controller with explicit robustness margins
(φ, a, b). The learned network k_θ is trained by plain behavioural cloning on
(observation, expert action) pairs. The result: k_θ is *input-to-state safe* with respect to the
original safe set, and strictly safe with respect to a **shrunken** set C_δ whose shrinkage is an
explicit function of the imitation error M_e, the network's Lipschitz constant L_{k_θ∘c}, and the
expert's robustness margin φ.

**Core formulas.**

```
Behavioural cloning:  min_{k_θ ∈ H} (1/N) Σ_{i=1}^N L( k_θ(c(x_i)), k(x_i) )                   (5)
                      D = { (c(x_i), k(x_i)) }_{i=1}^N

CBF-QP (the ordinary filter, for contrast):
      k_cbf-qp = argmin_u ½‖u − k_nom(x)‖²   s.t.  L_f h(x) + L_g h(x) u ≥ −α(h(x))

TR-OP expert (robust CBF-QP with tunable margins):
      k_T(x) = argmin_{v} ‖v − k_nom(x)‖²
      s.t.  L_f h(x) + L_g h(x) v − φ‖L_g h(x)‖² − a − b‖v‖ ≥ −α(h(x))
      with φ, a, b ∈ R_{≥0} and α extended class-K∞.

CBF-Compliancy (Def. 6): min_{x∈D} ‖x_1 − x‖ ≤ r_1,   ‖k_T(x_2) − k_θ(c(x_2))‖ ≤ M_e,
                         ‖k_θ(c(x_3)) − k_θ(c(x_4))‖ ≤ L_{k_θ∘c} ‖x_3 − x_4‖        (16–18)

Theorem 2: under CBF-compliancy with φ ≥ φ̄, a ≥ ā, b ≥ b̄, and Lipschitz L_f h, L_g h,
‖L_g h‖², α∘h on ∂C ⊕ B_{r_2}, the closed loop ẋ = f(x) + g(x) k_θ(c(x)) is ISSf w.r.t. C and
safe w.r.t.

      C_δ = { x ∈ R^n :  h(x) ≥ α^{-1}( − (1/(2φ)) ( L_{k_θ∘c} r_3 + M_e )² ) }                (19)
```

with ā = r_3 (L_{L_f h} + L_{α∘h} + L_{φ‖L_g h‖²}) and b̄ = r_3 L_{L_g h} (21–22). They note this
is, to their knowledge, "the first result to establish a direct relationship between the safety
of a system and the parameters of the imitation learning problem", and that the exact Lipschitz
constants are impractical to compute so real systems achieve safety with far smaller φ.

**Guarantee / cost.** Hard on the *inflated* set C_δ, and ISSf w.r.t. C — i.e. a bounded, quantified
safety loss rather than a guarantee on C itself. Cost: zero at inference (pure network); the price
is paid by making the expert conservative (φ) and the network smooth and accurate.

**For us — the cleanest "no filter at inference" option, and it fits our architecture better than
any RL paper here.** Our training *is* supervised (trajectory regression), not RL, so Cosner's
setting is the right one. The recipe: (1) generate a filtered-expert bank command offline with a
*margin* (their φ term is a shift of the barrier by φ‖L_g h‖², i.e. exactly "tighten the corridor
by a margin during data generation"), (2) train the network by plain regression to that expert
with no filter in the loop at all, (3) the guarantee you get is on the margin-shrunken corridor,
degraded by (L·r + M_e)²/(2φ). This is architecturally *simpler* than what we did, avoids the
laziness question entirely, and gives a quantitative reason to widen the training corridor by a
margin relative to the deployed one. It also says the two levers on corridor violation are
**imitation error M_e and network Lipschitz constant** — the second is directly controllable
(weight decay / spectral norm on the bank head) and we are not currently controlling it.

### 5.2 Geiger, Straehle (2022) — *Fail-Safe Adversarial Generative Imitation Learning*

- arXiv:2203.01696 (3 Mar 2022, rev 28 Jul 2023). **TMLR**, 11/2022.
- Local: `papers/failsafe_agil_2203.01696.pdf`

**Mechanism.** A generative imitation policy (Gaussian or normalizing flow) emits a "pre-safe"
action â; a **safety layer** maps it into a provably safe set Ã_{s_t} (inferred by adversarial
reachability analysis of a finite set of fallback manoeuvres plus a Lipschitz argument to cover
their neighbourhoods); the layer is a **piecewise diffeomorphism**, which gives the composed
policy a closed-form differentiable density, so the whole thing trains end-to-end under GAIL.
§3.3 then asks — and answers with bounds — exactly our question: is it better to use the safety
layer during training, or only at test time?

**Core formulas.**

```
Closed-form density through a piecewise-diffeomorphism safety layer (Prop. 3):
      p_ā(ā) = Σ_{k : ā ∈ g_k(A_k)}  | det( J_{g_k^{-1}}(ā) ) | · p_â( g_k^{-1}(ā) )           (6)

Lipschitz constant for the safe set (Prop. 1): if the momentary safety cost d is α-Lipschitz and
the dynamics are β-Lipschitz in state and in action, then a ↦ w_t(s,a) is α·max{1, β^T}-Lipschitz.

*** the train-time vs test-time-only result ***
Remark 1 (TRAIN AND TEST safety layer, LINEAR in horizon):
      assume D_TV(ρ^I, ρ^D) ≤ ε  ⇒  | v^I − v^D | ≤ 2 ε T ‖c*‖_∞

Theorem 1 (TEST-TIME-ONLY safety layer, QUADRATIC in horizon):
      lower bound (existence): | v^O − v^D | ≥ ι · min{ ε T², T } · ‖c*‖_∞                     (7)
      upper bound (for all):   | v^O − v^D | ≤ (4ε/ν) T² ‖c*‖_∞                                (8)
      where ν is the minimum mass of ρ^D(s) within its support.
```

Their intuition, verbatim: "only the former method learns how to properly deal (plan) with the
safety layer, while in the latter case, the safety layer may lead to unvisited states at test
time from which we did not learn to recover" — the worked counterexample is a driver that the
safety layer pushes onto the side strip, a state the unsafely-trained imitator never saw and
cannot recover from.

**Guarantee / cost.** Hard (worst-case, via adversarial reachability of the fallback set) at every
step, in training and at test. Cost: reachability analysis per step to infer Ã; the layer is
discontinuous, which "may complicate training" (their own caveat).

**For us — the strongest argument *in favour of* keeping the filter in the training loop, and it
comes from the imitation setting, i.e. ours.** The compounding-error argument is directly
applicable: our filtered rollout visits corridor-boundary states, and a network trained without
the filter has no data there. So the right conclusion from our six arms is *not* "revert to
inference-only filtering" — Geiger's Theorem 1 says that path has a quadratic-in-horizon error,
and our horizon is 64 segments. It is "keep the filter in the loop and add the correction penalty"
(§3.3, §3.4). Their caveat about discontinuous layers complicating training is also worth
recording: our hard clamp is discontinuous in the same way, which is presumably why the soft-tanh
variant exists; SafeDiffuser's annealed γ (§4.2) is the smoother alternative.

### 5.3 Yin, Seiler, Jin, Arcak (2021) — *Imitation Learning with Stability and Safety Guarantees*

- arXiv:2012.09293v2 (16 Dec 2020, rev 7 Apr 2021). IEEE L-CSS / ACC `[venue UNVERIFIED]`.
- Local: `papers/yin2021_il_stability_safety_2012.09293.pdf`

**Mechanism.** No runtime filter at all: the *weights* of the network are constrained. Local
sector quadratic constraints bound the activation functions; combined with a Lyapunov certificate
for the LTI plant this gives an LMI in (network weights, Lyapunov matrix) that certifies both
stability and forward invariance of the state constraint set X. The training problem then
minimises the imitation loss and simultaneously maximises the volume of the certified region of
attraction, subject to the LMI, solved by ADMM (network step in TensorFlow/Adam, SDP step in
CVX/MOSEK). Demonstrated on an inverted pendulum, **aircraft longitudinal dynamics**, and vehicle
lateral dynamics.

**Core formulas.**

```
Plant:   x(k+1) = A_G x(k) + B_G u(k),    safety condition  X = { x : −h ≤ H x ≤ h },  h ≥ 0   (2–3)
Ellipsoid ROA:  E(P) := { x : xᵀ P x ≤ 1 }                                                     (1)

Safe imitation-learning problem:
     min_{N, Q, L}   η_1 · L(N)  −  η_2 · log det(Q_1)                                        (29a)
     s.t.   LMI(Q, L) ≻ 0                                                                     (29b)
            f(N) Q = L                                                                        (29c)
     where N are the network weights, (Q, L) the SDP variables, L(N) the imitation loss
     (mean squared error in their experiments), and η_1, η_2 > 0 trade imitation accuracy
     against the size of the robustness margin.
```

**Guarantee / cost.** Hard — but only for **LTI** plant dynamics and only on the certified ROA;
zero runtime cost (the deployed object is just the network). Cost is at training time: an SDP
whose decision-variable count grows with the number of activations, and "Step 2 will be
computationally expensive if the number of activation functions n_φ is large".

**For us.** Not directly applicable — our rollout is a nonlinear point-mass model with an actuator
lag, not LTI, and n_φ for our network is far past their scale. But it is the right citation for
the third structural option ("constrain the network, not the action"), it is the only paper in
the cluster with an aircraft example, and its objective `η_1 · imitation_loss − η_2 · log det Q_1`
is the same shape as Tearle's terminal-set synthesis (§3.2) — i.e. "fit the data, and separately
maximise how much room the certificate leaves you". That second term is the formal version of
"don't let the safety machinery become more restrictive than it needs to be", which is the
Oh/Fisac permissiveness point again.

### 5.4 Silver, Allen, Tenenbaum, Kaelbling (2018) — *Residual Policy Learning*

- arXiv:1812.06298v2 (15 Dec 2018, rev 3 Jan 2019).
- Local: `papers/silver2018_residual_policy_learning_1812.06298.pdf`

**Mechanism.** Given any initial policy π (hand-designed, an MPC, anything — it need not be
differentiable), learn only an additive residual f_θ on top of it.

**Core formulas.**

```
     π_θ(s) = π(s) + f_θ(s)
     ∇_θ π_θ(s) = ∇_θ f_θ(s)         ⇒ the base policy need not be differentiable

Initialisation: "If the initial policy is perfect, then we would like the residual policy to have
no influence. We therefore endeavor to initialize the residual function so that f_θ(s) = 0 for
all s ∈ S. We do this by initializing the LAST LAYER of the network to be zero."

Critic burn-in: with actor-critic methods, "if we begin with a perfect initial policy and a poor
critic, the policy performance may degrade, since it is trained with reference to the critic. We
therefore propose to train the critic alone for a 'burn in' period while leaving the policy fixed."
```

**Guarantee / cost.** None — the residual is unbounded in the base formulation; the guarantee has
to come from elsewhere (e.g. clipping the residual, or a filter on top). Cost: negligible.

**For us.** Two operational details worth copying regardless of what else we do. (1) **Zero-init
the last layer** of whatever head produces the correction, so training *starts* at the reference
behaviour instead of at noise — if we adopt the Cheng-style absorption (§1.3) or any
corridor-tracking base command, this is what makes the first epochs stable. (2) The burn-in idea
maps onto our schedule: freeze the bank head and let the rest of the network / the loss scale
settle before the filter is allowed to bite, so the network is not being penalised for
corrections it has not had a chance to learn to avoid.

### 5.5 Johannink, Bahl, Nair, Luo, Kumar, Loskyll, Aparicio Ojea, Solowjow, Levine (2019) — *Residual Reinforcement Learning for Robot Control*

- arXiv:1812.03201v2 (7 Dec 2018, rev 18 Dec 2018). ICRA 2019 `[venue UNVERIFIED]`.
- Local: `papers/johannink2019_residual_rl_1812.03201.pdf`

**Mechanism.** Splits the reward into a part a conventional controller can already optimise and a
part it cannot (contacts, external object dynamics), and superposes a hand-engineered controller
with a learned residual. The **buffer bookkeeping is the interesting bit**: the executed action is
the sum, but what is stored and trained on is the *residual*.

**Core formulas.**

```
Reward decomposition:     r_t = f(s_m) + g(s_o)                                                (4)
                          f(s_m) — geometric part, optimisable a priori by a conventional
                                   controller; g(s_o) — the rest

Control action:           u = π_H(s_m) + π_θ(s_m, s_o)                                         (5)

Algorithm 1 (per step):
     u_t  = π_θ(s_t) + N_t              [policy action + exploration noise]
     u'_t = u_t + π_H(s_t)              [action actually EXECUTED]
     s_{t+1} ~ p(· | s_t, u'_t)
     store ( s_t , u_t , s_{t+1} ) in R  ← the RESIDUAL, not the executed action
     optimise θ on samples from R
```

**Guarantee / cost.** None formally; the argument is that "a properly designed feedback control
law for π_H(s_m) is able to provide exponentially stable error dynamics of s_m if the learned
controller π_θ is neglected". Cost: negligible.

**For us.** The buffer rule is the point: **the learning signal is attached to the network's own
output, while the environment sees the composite.** That is OptLayer's CP tuple in an additive
rather than a projective architecture, and it is what our current arm is *not* doing. If we
restructured the bank head as `bank = bank_corridor_tracking(state) + Δ_θ(features)` with Δ_θ
bounded and zero-initialised (Silver), then the filter would essentially never fire, the network
could not be lazy (its output *is* the deviation from the safe reference), and gating becomes a
smooth blend weight on the reference term rather than an on/off switch. This is the lowest-risk
architectural change in the whole cluster.

---

## 6. Learned certificates and the reachability filter

### 6.1 Dawson, Gao, Fan (2023) — *Safe Control with Learned Certificates: A Survey of Neural Lyapunov, Barrier, and Contraction Methods*

- arXiv:2202.11762v2 (23 Feb 2022, rev 20 Dec 2022). **IEEE Transactions on Robotics**.
  Code: github.com/MIT-REALM/neural_clbf.
- Local: `papers/dawson2023_learned_certificates_2202.11762.pdf`

**Mechanism.** Surveys the family in which the *certificate* (Lyapunov / barrier / contraction
metric) is itself a neural network, trained by penalising the violation of its defining
conditions at sampled points, optionally jointly with the controller. Distinguishes four
learning contexts: certificate only; **certified behaviour cloning**; **certificate-regularised
RL**; and self-supervised joint certificate+policy learning.

**Core formulas.**

```
Empirical certificate loss (the general form):
      L_V = Σ_{i ∈ I} (α_i / N) Σ_{j=1}^{N} max( c_i(x_j, V), 0 )                             (19)
      — one term per certificate condition c_i, hinge on its violation at sampled points x_j,
        with hand-tuned penalty weights α_i.
      Caveat they state explicitly: "zero empirical loss does not guarantee that the certificate
      is valid; it provides only statistical evidence of validity."

Certified behaviour cloning:
      L = L_V + L_BC
      "we simply augment the empirical loss in (19) with a behavior cloning term that penalizes
      the difference between π and the expert policy (often simply the mean of an appropriate
      norm)."

Certificate-regularised RL: L_V is used alongside the reward to update the policy; complicated by
the fact that RL assumes no dynamics model, so Lie derivatives must be estimated from trajectories.

Worked CLF example (their inverted pendulum, λ_1=10, λ_2=10³, λ_3=λ_4=1, λ_5=10²):
      L_V = λ_1 V(x_0)
          + (λ_2/N) Σ_i r(x_i)
          + (λ_3/N) Σ_i max( L_f V(x_i) + L_g V(x_i) u_i , 0 )
          + (λ_4/N) Σ_i max( [ V(x_i + Δt·ẋ(x_i,u_i)) − V(x_i) ] / Δt , 0 )
      with [r(x_i), u_i] the solution of the relaxed CLF-QP
          min_{r,u} ‖u‖² + λ_5 r   s.t.  L_f V(x) + L_g V(x) u ≤ −c V(x) + r,  r ≥ 0
```

They also note that maximising the size of the admissible input set K(x) (or of K ∩ U under
actuator limits) "is less well explored", could be added as a normalised term to (19), "but care
should be taken to not allow the neural certificate to overfit to this term at the expense of
violating the certificate conditions."

**Guarantee / cost.** **Soft / statistical** — this is the family's defining weakness, and the
survey is blunt about it; hard guarantees need a separate verification pass (SMT counterexample
loop, NN verification). Cost: sampling + gradient descent, no runtime optimisation.

**For us — the option we have not tried, and the one that most directly avoids laziness.**
Instead of clipping the bank, add the barrier violation along the *predicted rollout* as a
differentiable loss term:

```
L = L_traj  +  (λ / |gated|) Σ_{k ∈ gated}  max( −[ h(x_{k+1}) − (1−γ) h(x_k) ] , 0 )
```

which is exactly Dawson's L_V with our DCBF condition (Zeng, §4.4) as the single c_i, applied at
the rollout states rather than at random samples. Properties: the gradient reaches the network
through the rollout (which we already have), there is no clipping and therefore nothing to be lazy
about, the gate becomes a differentiable weight on the sum rather than a control-flow branch, and
`L = L_V + L_BC` is literally the certified-behaviour-cloning recipe for our regression setting.
The cost is that the guarantee drops from hard to statistical — so the honest architecture is
**loss term in training + filter at inference**, which is also what Alshiekh (§2.2) and
Pizarro Bejarano (§3.3) conclude on independent grounds.

### 6.2 Fisac, Akametalu, Zeilinger, Kaynama, Gillula, Tomlin (2019) — *A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems*

- arXiv:1705.01292v3 (3 May 2017, rev 14 Feb 2018). **IEEE Transactions on Automatic Control**
  (accepted, per the arXiv comment).
- Local: `papers/fisac2019_general_safety_framework_1705.01292.pdf`

**Mechanism.** Hamilton–Jacobi reachability supplies a safety value function V(x) whose zero
superlevel set is the maximal safe set under worst-case disturbance; the framework then defines a
**least-restrictive supervisory law** that lets the learning policy κ_l run freely wherever
V(x) > 0 and imposes the optimal safe action κ*(x) otherwise. The extension is that the
disturbance bound D̂(x) is a Gaussian-process posterior updated online, and a **Bayesian confidence
λ(x)** in the model triggers *early* intervention when confidence decays — so the filter is
conservative exactly where the model is untrustworthy, and permissive elsewhere.

**Core formulas.**

```
Least-restrictive switching law:
        κ(x) =  κ_l(x)   if V(x) > 0
                κ*(x)   otherwise                                                              (8)

With Bayesian model-confidence validation:
        κ(x) =  κ_l(x)   if ( V(x) > 0 ) ∧ ( λ(x) > λ_0 )
                κ*(x)   otherwise                                                             (20)

They note explicitly that instead of imposing κ*(x) "it would have, in principle, been sufficient
to project the desired κ_l(x) onto the set of control inputs that guarantee nonnegative local
evolution of V for all d ∈ D̂(x). However, κ*(x) results in the greatest predicted increase in
value, which is desirable under model uncertainty."

Set invariance: if the safe-control condition holds for all x ∈ Q_α, then { x : V(x) ≥ α } is
invariant under κ*.
```

**Guarantee / cost.** Hard under the assumed disturbance bound; high-probability once the bound is
a GP credible set. Demonstrated on a quadrotor that learns a vertical flight policy by policy
gradient without ever crashing, including under an unmodelled fan disturbance. Cost: offline HJ
reachability (curse of dimensionality) + online GP updates.

**For us.** Three transferable points. (1) The **switching (replacement) vs projection** choice,
argued on the merits: under model uncertainty they deliberately apply the *maximally* safe action
rather than the minimal correction — which is Krasowski's "action replacement beats action
projection" finding from a completely different direction, and is the single alternative hook
design we should try. (2) The **confidence gate** λ(x) > λ_0 is a much better-motivated gate than a
geometric predicate: intervene early where the model is untrustworthy. For us the analogue is
intervening earlier on flights whose lookback looks unlike the training distribution — i.e.
exactly the vectored flights that got worse. (3) V(x) is the least-restrictive certificate, and
Hsu (§2.3) notes V is itself a valid CBF; our hand-designed h is strictly more conservative, which
is the permissiveness deficit Oh/Fisac (§2.4) says costs performance.

---

## 7. Synthesis — what the literature actually answers

**The phenomenon is known, named and expected.** Alshiekh (2018) predicted it from the reward
bookkeeping alone; Krasowski (2023) measures it as *intervention rate*; Pizarro Bejarano (2025)
measures it as *return when uncertified* and shows it is a monotone function of the correction
penalty weight; CBF-RL (2026) shows "Filter Only" works only with an active filter. Nobody
reports that training through a filter is intrinsically harmful — they report that training
through a filter **with naive reward/loss bookkeeping** is.

**Five distinct cures, in increasing order of change to our code.**

1. **Charge the network for the correction** (OptLayer CPC; Krasowski adaption penalty;
   Pizarro Bejarano P = ‖u_uncert − u_cert‖²; CBF-RL's two-term bounded r_cbf). For us:
   `L = L_traj(filtered rollout) + λ·Σ_gated ‖bank_raw − bank_filtered‖²`, gradient of the second
   term into the raw head only, λ swept. **The evidence for this is the strongest and the change
   is the smallest.** Pizarro Bejarano's table says the sweep is essential: too small and the
   dependence stays, too large and the on-filter return degrades.
2. **Make the network absorb the filter** (Cheng's u^bar_φ distillation). For us: a self-DAgger
   epoch in which the raw bank head is regressed onto the previous epoch's *filtered* bank.
3. **Restructure so the correction is the network's output** (Silver / Johannink residual;
   zero-init last layer; buffer the residual, execute the sum). Laziness becomes structurally
   impossible because the reference is the safe command.
4. **Replace the filter by a differentiable loss** (Dawson's certified behaviour cloning,
   `L = L_V + L_BC`, with L_V the DCBF hinge over gated rollout segments). No clipping in
   training; keep the hook only at inference.
5. **Make the filter's conservativeness learned rather than fixed** (BarrierNet's p_i(z)
   multiplying the class-K function). Safety still hard for any p_i > 0; the gate dissolves into
   a smooth learned quantity.

**Three findings that reframe our result rather than fixing it.**

- **Permissiveness, not filtering, is the cost driver.** Oh/Fisac prove there is *no* asymptotic
  performance penalty for a **perfect (least-restrictive)** filter used at train and test.
  Hsu notes a CBF filter never encodes the maximal safe set. So "our accuracy dropped" is evidence
  that our (k, α, gate) triple is over-restrictive — try enlarging α and k before adding
  machinery.
- **Do not plan to drop the filter.** Alshiekh (both options end with "the shield is needed at
  execution"), Pizarro Bejarano ("this highlights the importance of not removing the safety
  filter, even when penalizing corrections") and Oh/Fisac (the guarantee is for the policy
  *executed through* the filter) agree. CBF-RL is the sole dissenter and its filter-free
  deployment claim is empirical, not proved.
- **In an imitation setting, training through the filter is the *better* option, provably.**
  Geiger & Straehle: train-and-test filtering gives imitation error linear in the horizon T;
  test-time-only filtering has a tight *quadratic* T² bound, because the policy never saw the
  states the filter drives it into. Our horizon is 64 segments, so this is not a small constant.
  This is the argument against reverting to inference-only filtering.

**On the gate specifically.** No paper in this cluster uses a hard state-predicate gate on a
differentiable rollout, and three of them offer smoother substitutes: SafeDiffuser's annealed
γ_k(j) (constraint faded in along the horizon), BarrierNet's learned p(z) (conservativeness as a
network output), and Fisac's confidence gate λ(x) > λ_0 (intervene early where the model is
untrustworthy, not where geometry says so). Two independent lines also predict that an
**infeasible gate entry** — the aircraft already outside the corridor when the gate fires — is
where projection filters misbehave: Krasowski (projection infeasibility from numerical error,
requiring a failsafe fallback) and Pizarro Bejarano (safe reset: reject infeasible starts, which
"significantly improves convergence"). Since radar-vectored flights are the ones that enter the
gate off-corridor, this is the most likely mechanism for the vectored-accuracy regression, and
annealing the constraint in (SafeDiffuser §4.2) is the cheapest test of that hypothesis.

**Searches that came up empty / were not pursued.** I found no paper that (a) trains a
*trajectory-prediction* network through a safety filter, (b) reports a filter-induced accuracy
regression on a *held-out subpopulation* (our vectored flights), or (c) treats a gated filter
(active only on part of the horizon) on a differentiable rollout. The closest adjacent works
surfaced by search but **not downloaded or verified** are: Shield-Loco (predictive safety
filtering for locomotion policies, arXiv 2606.07193), "End-to-End Learning of Safe Optimal
Feedback Control in High Dimensions with Control Barrier Function Layers" (arXiv 2607.20674,
three-operator splitting + Jacobian-free backprop to make training *through* a high-dimensional
CBF layer tractable — potentially very relevant to the cost of our rollout), a modular
RL + predictive-safety-filter marine navigation paper (arXiv 2312.01855), and the three
filter-aware works Hsu §3.5.3 cites (Akametalu; Leung et al.; Hu et al., SHARP). All are
`[UNVERIFIED]` — titles and one-line descriptions only.
