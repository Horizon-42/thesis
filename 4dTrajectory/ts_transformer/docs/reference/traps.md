# ts_transformer reference — traps

Full text behind the index lines of `4dTrajectory/ts_transformer/CLAUDE.md`, moved here
VERBATIM on 2026-09-16 when that file had grown to 118 KB (1,202 lines) and was loaded into
every ts session. Only the `### <ID>` headings are new. Every heading has exactly one
index line in CLAUDE.md that ends in its ID; find one with `grep -n '^### C7 ·' docs/reference/*.md`.

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.** Measurements and campaign evidence still
belong in `docs/ENGINEERING_NOTES.md` / the design documents, status in `docs/OPEN_ITEMS.md`.

**2026-09-18:** the main line's 2026-09-14…09-17 additions to that CLAUDE.md (the control
contracts, the two-tier axes and traps) were placed here the same way when this branch was
rebased — same rule, new IDs.

## Traps (one line each; evidence in `docs/ENGINEERING_NOTES.md`)

### T18 · a stored config lacks every field added after it trained

- **A stored config lacks every field added after it trained — read the absence as
  `from_dict` does** (`config.absent_field_defaults`: the default, except REQUIRED fields).
  Reading it as `None` refused every recipe arm's campaign resume the day a recipe pinned
  `control_thrust_parameterization` (2026-09-14). `pipeline.cv_reuse_error` still compares with
  `!=` (`docs/code-health-followups.md` §33).

### T19 · a specific-force rollout's speed drift is not a bug in the law

- **A specific-force rollout's speed drift is not a bug in the law** — the speed integrates
  the command (no drag feedback, design §2.1). Never give that contract a level-trim neutral.

### T20 · the third actuator is a load factor except under `specific-force+path-angle`

- **The third actuator is a load factor under every law EXCEPT `specific-force+path-angle`**, where it is
  the path-angle target in radians and the flown load is re-solved from the state. A term that prices
  what the rollout FLEW asks the contract's law (`law.geodetic_load`, as the heading-rate loss does since
  2026-09-16); reading `EndpointControlRollout.actual_controls[..., 2]` blind under that contract gives
  radians where a load belongs — it cost the heading-rate loss a sign-flipped, 20×-small target before
  the config refused the pairing (design §14.9). The command hooks read it through
  `outputs/constraints/vertical.VerticalChannel` (the law's resolved load; under that contract a bank
  change needs no load re-coordination, `RESOLVES_LOAD_AT_FLOWN_BANK`), so barrier, speed floor and
  trombone are admitted there since 2026-09-16 (two-tier design §10.8).

### T21 · a tracker scored against the rule guidance must fly the guidance's budget

- **A tracker scored against the rule guidance must be flown to the guidance's budget** (`tracker_lockstep`,
  two-tier §10.7): the guidance's rollout runs to `plan_oracle.closing_horizon_s(T)` = `T + max(30 s, 0.1·T)`
  and "established" is a crossing on the final inside it; a `cta=given` rollout ends exactly at its CTA, so
  an on-time tracker ended there reads established by the sign of a rounding error (15/40 on a perfect
  forecast). The lockstep ends a flight only by crossing or at that horizon, with every ask's CTA raised to
  `max(training future floor, one step)`.

### T22 · a straight-in FDE is a TIMING number

- **A straight-in FDE is a TIMING number.** Along-track carries 60–77 % of Σ FDE² and vertical 1–2 %
  (specific-force design §7.5). A law whose rollouts dive ~100 m low to regain speed wins FDE: δ does exactly
  that at the stall boundary.
  - Read any straight-in FDE comparison with its along/cross/vertical split (`geometry/metrics.py` already
    computes the components per grid point).
  - The vertical-load channel is open loop on every law: a load bias δn grows a height error of about
    `½·g·δn·t²`, ~200 m for 0.004 g over 100 s.

### T23 · a checkpoint trained before `c544db0` is not a same-code baseline

- **A state/control checkpoint trained before `c544db0` (2026-09-09) is NOT a same-code baseline for a run
  trained after it**, even with an identical config. Review A-3 moved the terminal supervision weights of
  the flights whose observed track reaches the threshold (1.9 % of KRDU train). It is a data-side contract
  with no config field, so nothing in the name shows it. `B1_point_matched` retrained with the same seed:
  epoch-1 selection ADE −11 %, and `val_loss` is not comparable across it at all. Re-train the twin
  (specific-force design §11.6).

### T1 · every ts number assumes the LANDED runway is known

- **Every ts number assumes the LANDED runway is known** — the threshold anchor (and the plan
  path's CIFP skeleton) is the harvest's final-approach runway, i.e. future information; no
  trajectory path predicts the runway yet (the runway head is standalone; R2 flies its pick through
  the plan experts in `experiments/runway_intent_r2.py`: +55-79 m of mean FDE against the known
  runway, plan §16). Quote ADE/FDE as runway-given. What the label is worth and why the backbone cannot learn it implicitly: the
  2026-09-03 frame ablation and runway-hypothesis docs; the plan to predict it (runway head +
  per-runway experts + multi-runway scheduling): `docs/2026-09-13_runway_intent_plan.zh.md`.

### T2 · a candidate-symmetric runway head carries no per-runway constant

- **A candidate-symmetric runway head must not carry a per-runway constant** — a row column that is
  constant per runway (its training share, an operator's raw share of its landings) lets shared trees
  re-identify the runway and override the recent landings on a configuration the training days lack
  (R1.1: KSJC closure days 53-59 %); a label-derived column must also be built from EARLIER training
  days, never out of fold (a block's out-of-fold share is anti-correlated with its own labels). Plan §15.

### T3 · arrival separation is a distance rule on the approach clock

- **Arrival separation is a DISTANCE rule, and two landings are compared on the APPROACH CLOCK** —
  `inference/runway_schedule.py` is the one definition (FAA JO 7110.65BB Change 3, every value cited to
  its paragraph; the text itself in repo `docs/literature/arrival_separation/`): 3 NM radar or the CWT
  wake minimum at the threshold on one runway and on parallels under 2,500 ft, dependent diagonals
  1.0 / 1.5 NM, independent from 4,300 ft, types -> CWT from JO 7360.1K. A distance becomes a time at the
  airport's measured approach speed, and a threshold time goes onto the approach clock by its
  threshold's along-course position: parallel thresholds are staggered (KRDU 23L/23R 0.67 NM, KSTL 11 vs
  12L 1.99 NM), and comparing the two threshold times passed 5 short pairs as 0 violations (R3 review,
  2026-09-14). Multi-aircraft work reads the same module.

### T4 · a runway/configuration model is judged on operating days

- **A runway / configuration model cannot be judged on the per-flight split** — same-day flights
  share the configuration, and hourly wind / time of day fingerprint it: the per-flight model read
  KSJC's two 30L-closure days at 100 %, the day-blocked one at 55–57 % (B1 97–98 %). The R series
  splits by OPERATING day (`data/runway_context.operational_day`, cut at 09Z — a UTC-date cut splits
  the evening peak); plan §13.

### T5 · the objective must score velocity

- **The objective must score VELOCITY, not just position** — scoring position at 64 endpoints
  alone let 71 % of predicted bank energy collapse into one profile shared by every flight.

### T6 · bank unsupervised lands below a trivial baseline

- **Bank was never supervised, and unsupervised it lands BELOW a trivial baseline** — position is
  derivative order 0, velocity order 1, bank order 2, so no term ever named it. `simple-v3` fixes
  it (skill 0.124 → 0.735) at no accuracy cost.

### T7 · three causes of the bank wiggle are NOT it

- **Three causes of that bank wiggle were tested and are NOT it** — segment count, training
  budget, conditioning capacity. Do not re-litigate without new evidence.

### T8 · the imitation dose curve is not a ramp

- **The imitation dose curve is NOT a ramp**; below ~11.8× position it is a noisy plateau and
  past ~47× it overshoots into smoother-than-reality.

### T9 · loss weights are calibrated, not chosen

- **Loss weights are calibrated, not chosen** — raw velocity and position terms differ by 642× at
  the converged operating point. Over-constraining looks exactly like the blandness trap.

### T10 · `DEFAULT_CV_PATIENCE = 6` is too small

- **`DEFAULT_CV_PATIENCE = 6` is too small here** — both flight models pass through an early ADE
  transient, so patience 6 turns a τ ranking into a stopping artifact.

### T11 · a named recipe cannot be cross-validated as itself

- **A named recipe cannot be cross-validated as itself** (frozen `epochs`/`patience`).

### T12 · the duration head cannot predict below ~125 s

- **The duration head cannot predict below ~125 s** against a true range starting at 21 s —
  flights anchored close to the runway fly a full loop. Unfixed, present in every flight model.

### T13 · the state model's KRDU endpoints sit ~250 m NW

- **The state model's KRDU endpoints sit ~250 m NW of every runway, and it is the model, not the
  data** — a world-fixed translation present from the FIRST predicted step. Read every arm-A
  per-runway cross-track number with it in mind.

### T14 · `random_train_anchor=True` + imitation is a performance cliff

- **`random_train_anchor=True` + the imitation term is a performance cliff** (the per-flight
  inverse would be recomputed per sample per epoch). A note, not a guard — no recipe uses it.

### T15 · diagnostic scripts must go through `batch_contract.model_forward`

- **Diagnostic scripts that call `model(x)` directly cannot run a corridor-bounded checkpoint** —
  go through `batch_contract.model_forward` with the context row.

### T16 · `StateOutputLayer.offset_mask` is a non-persistent buffer

- **`StateOutputLayer.offset_mask` is a non-persistent buffer**; `load_checkpoint` drops the key
  if a checkpoint stored it. A buffer that IS learned scale stays persistent.

### T17 · bare ignore rules match the package's `data/` and `outputs/`

- **Two bare ignore rules match package directories**: the root `.gitignore`'s `data` and
  `4dTrajectory/.gitignore`'s `outputs` (meant for artifact directories) also matched the
  package's `data/` and `outputs/`; both are re-included by name, and the §4.2 `outputs/`
  modules went uncommitted for a day before anyone noticed (2026-09-10). A new group whose
  name is also an artifact directory needs the same negation; `git status --ignored` is the
  check, and a green suite in the worktree proves nothing about the commit.

### W1 · runway intent — status by stage (moved from the "Where to go next" table)

| doing this | read first |
|---|---|
| predicting the landing runway (runway intent), multi-runway scheduling | `docs/2026-09-13_runway_intent_plan.zh.md` (R0 §11: direction solved by context, the parallel SIDE is the problem; R1 §13: a per-runway head wins the side at KRDU / KSMF by 23–32 points and fails on configurations its training days lack; R1.1b §15: the candidate-symmetric `r11_lift` keeps the gain and survives KSJC's closure days; R2 §16: its pick flown end to end costs +55-79 m of FDE against the known runway (B1 +400-470 m), never-lock adopted; R3 §17: the multi-runway scheduler under the FAA minima keeps the runway intent with 0 conflicts but barely binds — its delays do not improve arrival times, ETA error is the size of the minimum — and the flown closure loses time between asks; R4 not triggered). The separation rules themselves: `inference/runway_schedule.py` and repo `docs/literature/arrival_separation/` |

### W2 · the two-tier model — which document to read (moved from the "Where to go next" table)

| doing this | read first |
|---|---|
| building or reading the two-tier model (L1 short-horizon control layer, L2 segment-token plan layer, L3 graph layer) | **`docs/2026-09-17_two_tier_plan_v2.zh.md`** — the current plan (the requirement verbatim, the eight decisions, gates L1 / L2 / E2E, the picks made without the user and why). The 09-16 feasibility doc is SUPERSEDED: its design (a whole-approach tracker under the instruction plan, T1a) is abandoned; only its measurements are citable — §2 (three evaluation protocols that must never be mixed: a 60 s ADE is low for EVERY model), §3 (error against lead time), §10.2 (the harvest holds no 10 min history), §10.4 (T0(b): open-loop chaining loses), §10.6 (T0(c): 238 s of history carries nothing), §11 (the contract refactor). Readouts: `run_ts.py short_horizon_readout`, `tracker_lockstep`, `two_tier_gates`, `lead_time_error`, `chain_sensitivity` |
