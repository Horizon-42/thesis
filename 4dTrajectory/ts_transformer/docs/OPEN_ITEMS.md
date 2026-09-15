# ts_transformer — open items

Status log for the package: what each campaign settled, what it adopted, and what is left.
Moved out of `CLAUDE.md` so it is read when planning work, not injected into every session.
Newest campaigns first, long-standing scope limits last.

---

## Specific-force control parameterisation — MEASURED, NOT ADOPTED; the final descent diagnosed (2026-09-15, branch `specific-force-control`)

Design `docs/2026-09-14_specific_force_control_design.md`; literature
`docs/literature/control_normalization/` (repo root).

**The ask.** The user: "…是不是可以修改aerodynamic 或者增加一个normalize 处理？" The head's thrust channel was
normalised by the actuator (δ = T/T_max). The new axis `control_thrust_parameterization=specific-force` normalises
it by the effect (n_x = (T − D)/W, re-solved per RK4 stage on the lag model).

**Why it may matter** (KRDU `B1_point_matched`, 2 seeds, established stratum):
- The head's δ is class-invariant; the truth's is not.
- The class-dependent n_x bias spans 0.010–0.012 g, against the truth's own 0.0045 g.
- Heavies fly 4.7–4.9 m/s fast relative to the 737 family.

**Status and next steps:**
- **M1** (`8ce4568`) and **M2** (the speed floor, CLI, docs, smoke) are built and reviewed. Defaults are
  bit-identical, and no stored run is renamed or refused.
- **N3 — the measurement:** KRDU `B1_point_matched` as `custom` + specific-force, seeds 1337/2024
  (`docs/experiments/sf_n3_arms.json`). **Ran** 2026-09-15 00:18–02:33 UTC from the `specific-force` worktree
  at `47b4b40`; campaign `4dTrajectory/outputs/KRDU/experiments/sf_n3`.
  - Gates: the per-class n_x bias range falls toward 0.0045 g; the heavy − 737 speed gap closes.
  - Veto: pooled ADE within ~125 m of the base; the straight-in FDE veto.
- **N4 — the alternative-hypothesis control, RAN; REJECTED on both seeds** (design §11.7). Ratios move the δ head's
  class bias only 0.0125 → 0.0119 / 0.0118 → 0.0105 g, where the specific force moves it to 0.0042 / 0.0044: the
  class structure is the parameterisation's. `ratios` is not adopted. Design: (branch `sf-n4`, design §11): `control_condition_features=
  ratios` hands the thrust-fraction head T_max/W and the 1-g stall speed in place of T_max and S (same information,
  same width, same initial weights). If it closes the class bias as far as N3, the conditioning's presentation
  explains the failure; if not, the parameterisation does. Arms `docs/experiments/sf_n4_arms.json`; launch after
  N3's runner exits. Measured on the way: Cd0, k and the stall parameters are constant on KRDU's 26 types — 4 of
  the 8 channels carry nothing.
- **The stored `B1_point_matched` twins do NOT reproduce at the current code** (design §11.6): the same config,
  seed and split, re-trained deterministically, reads epoch-1 val 7.737 against the twin's 8.381, and the data,
  init and environment are the same. So `sf_n4` also re-trains the δ twins (`N4_twin`, `N4_twin_s2024`, first),
  and N3 and N4 are both read against them. **Cause (bisected): `c544db0`**, review A-3's single terminal
  supervision contract. It is an intended, documented fix, but with no config field, so it is invisible in run
  names. Any comparison of a new run against a pre-2026-09-09 state/control artifact carries it.
- **N5:** the pooled five-airport arm, only if N3's mechanism holds on both seeds (no pooled δ twin is stored, so
  it is four arms).
- **N3 final, both seeds against same-code twins (design §7.3):**
  - gate 1 passes (n_x class range 0.0042 / 0.0044 vs 0.0125 / 0.0118 g);
  - gate 2 fails (the heavy − 737 gap grows 1.0 / 1.4 m/s);
  - the straight-in FDE veto trips (+240 / +222 m p50);
  - pooled ADE is −57 / −73 m.

  Not adopted. The same-code δ twins differ by only 30 m of pooled ADE, against the stored pair's 125 m, so the
  control path's seed line needs re-measuring on converged pairs.
- **The FDE veto's cause, re-measured (design §7.4):** not a missing restoring force.
  - Straight-in flights fly below the model's minimum-drag speed, where δ's drag feedback amplifies errors.
  - The two laws match the truth equally to 5 km out.
  - On the last 5 km the n_x run ends 36–54 m HIGH and 4.4–6.4 m/s SLOW with the total energy right. δ ends
    ~30 m low at the right speed.
  - Why the n_x head flies the final descent shallower (the load-factor channel): diagnosed below. A
    longitudinal law change (N6) does not address it.
- **The final-descent split, diagnosed (design §7.5, read-only, the four same-code runs):**
  - The truth flies the final descent at ~0.6 Cl_max (about 1.3 V_s). Rollouts reach the stall boundary only
    by losing 16–21 m/s.
  - SF's error is a tail (32 / 45 % of straight-ins) with the energy right. Its vertical-load command sits
    +0.004 above the truth. Integrated twice open loop, that bias predicts the height error at 5 km (Spearman
    +0.76 / +0.82). Height comes out of speed; the tail ends ~95 m high, ~8 m/s slow, 1.0–1.1 km behind.
  - δ's load bias is 2–6 × larger. Its rollouts reach the stall boundary on 63 / 68 % of flights, and the lift
    cap dives them: they end ~100 m low and ON TIME. That is its straight-in FDE edge. δ's other flights have a
    worse FDE than SF's (714 / 799 vs 667 / 683 m), and δ ends with the larger |vertical| (127 / 139 vs
    77 / 90 m).
  - A straight-in FDE is 60–77 % along-track and 1–2 % vertical (of Σ FDE²), so the N3 veto rewards δ's
    compensating error. Read it with its along/vertical split.
- **N7 (an inference-time glidepath hook) — WITHDRAWN as a model fix** (the user, 2026-09-16, design
  §7.5.6). It computes the final's vertical profile from the published procedure, so the result would be the
  rule's, not the model's. Kept only as a labelled diagnostic or baseline component.
- **N7′ — proposed, not built (the user's call):** a learned vertical target. The head predicts the path angle
  (or height) per segment, and a fixed tracking law with no procedure in it flies the head's own target. The
  height error then grows linearly or not at all, instead of quadratically. Under SF, `Ė = V·n_x`, so the law
  cannot hide an energy cost the way N6's speed loop did.
- **N6 — the speed command: RAN, FAILED — unstable in training** (design §12.9).
  - The first arm diverged from epoch 26: pre-clip control-head gradients 1e7–1e10, from zoom climbs the speed
    loop hides until the T_max clamp binds.
  - Pooled ADE 2335 vs 1325 m against the same-code twin. The second seed was stopped before training.
  - Not adopted. The next design (bounded loop authority / a vertical loop / damped n_x) is the user's call.
  - The original entry: BUILT and reviewed on `sf-n6` (design §12). The head predicts a target airspeed
  relative to the anchor's, flown by an 8 s speed loop through the specific-force clamp: the invariance N3 showed,
  plus the restoring force whose absence is the FDE veto's mechanism. The reviewer measured each law's teacher flown
  open-loop (300 KRDU val flights): ADE 375 m against 2606 (n_x) and 2700 (δ). §12.6's condition is met: N6 is
  queued after `sf_n4` (`docs/experiments/sf_n6_arms.json`).
- **Known difference, not a bug:** the specific-force speed has no drag feedback (design §2.1).

## Runway intent — R3.1 / R3.2 / R3.3 (2026-09-14, `docs/2026-09-13_runway_intent_plan.zh.md` §18)

The user: "先做1, 2, 最后3" — §17.8's uncertainty-aware scheduling, the closure's time delivery, publishing R3.
**R3.1 MEASURED — no pre-registered gate passes at any airport**: scheduling under ETA uncertainty
(`runway_schedule.sample_schedules`, 200 FCFS schedules per airport on the flights' calibrated ETA errors, each
flight's draws its error stratum's quantile grid in a random order) against the same error model without
interaction: all hours −0.4 to +0.3 s of median |dt|, busy hours −1.3 to +0.5 s, the moved flights worse at
3–4 airports; the training days' median ETA error does not carry to validation days (the calibrated baseline is
worse than the raw ETA at 4 of 5 airports); the centred sensitivity agrees. Landing-time error lives in each
flight's own ETA, not between aircraft; R4 stays off. **R3.2 CLOSED AS A DIAGNOSIS, no fix adopted**: the time
closure times an instruction leg through the 8 km `LEG_EXTENSION_M` placeholder past the fix, reads the flight
minutes late (X −85 / −176 s at KSMF / KSTL) and chases, then is early with no lever at the switch to the
closing (+104 s); three tail models smoke-tested on R3's plan removed the chase but none improved delivery at
both airports (what is flown after a fix is the head's later orders) — stopped before tuning on the evaluation
flights; v3 on branch `wip-r32-leg-timing`; code-health §31, §32. Next (the user's call): a closure timed over
the whole remaining route, developed off the evaluation roster. Artifacts `outputs/POOLED/experiments/
runway_intent_r31{,_centered}_20260914/`, `runway_intent_r32_20260914/diagnosis/`.
**R3.3 CODE DONE, RECORDS WRITTEN, PUBLICATION WAITS FOR THE MERGE** (`220138a` + `a6d921a`): split `dayval`, `runway_intent_r3 --write-records` (5195 records over five airports, identical to R3's formal flights; a record's prediction is the SCHEDULED time), publisher (reuse-only, experiment-only, both halves of held-out checked — outer-test hash and the checkpoint's own train/val — `--category-group`), frontend. Publish only after the ff-merge: the worktree's frontend data is the main tree's. KRDU `categories.json`, overwritten by a test at 16:13, was rebuilt and written back at 17:41 with the user's go-ahead (`check-publication`: 152 categories, 0 errors); evidence in `runway_intent_r3_20260914/incident_20260914_krdu_categories/`.

## Runway intent — R3 MEASURED 2026-09-14 (`docs/2026-09-13_runway_intent_plan.zh.md` §17)

**R3** (multi-runway joint assignment and the arrival scheduler, `inference/runway_schedule.py` +
`experiments/runway_intent_r3.py`): R2b's flights (every day_a validation-day flight the retrained expert
flew, 5195 over five airports) scheduled FCFS by ETA over the runways the head allows, `log p - lambda *
delay`, under the FAA IFR minima (JO 7110.65BB Change 3, quoted in repo `docs/literature/arrival_separation/`),
then FLOWN with the scheduled time as the v5.4 assignment. **Pre-registered gates: KSTL and KMSY pass all
four; KRDU, KSJC and KSMF miss only gate 3's busy-hour clause by 0.3-1.2 s of median timing error — a tie.**
Plans have 0 violations and keep the head's runway (agreement unchanged +/-0.2 points); the unscheduled
forecasts have 23-90 conflicts, resolved by delaying 2.4-7.3 % of flights, and on the flights the schedule
moves its time is NOT nearer the truth (KRDU 50 vs 30 s, KSMF 52 vs 32; nearer on 46-66 %) — the ETA error
(median 11-21 s) is the size of the minimum, so deterministic FCFS treats ETA noise as conflict. The
truth keeps these minima on 99.3-100 % of consecutive roster pairs. Flown: 71-88 % of undelayed and
37-81 % of delayed flights land within 10 s of their slot; the early ones lose the time BETWEEN asks (X
~0 at the first ask, +22-29 s at the last), not for want of distance. Unlanded 3-34 %, 53-92 % of them the
expert failing the flight unassigned too (KMSY 02/20). **R4 is not triggered** (busy runway agreement at
most 2.2 points under quiet, busy timing no worse). Next (plan §17.8, the user's call): uncertainty-aware
scheduling on the quantile duration heads; the closure's time delivery between asks (plan-and-guidance
`guidance/timing.close_time`); publication needs a held-out-days category in the publisher (R3's flights
are in no split of the checkpoint that flew them). Artifacts `outputs/POOLED/experiments/runway_intent_r3_20260914/`.

## Runway intent — R2 MEASURED 2026-09-13 (`docs/2026-09-13_runway_intent_plan.zh.md` §16)

**R2** (the runway head's pick flown end to end by the plan experts — R0b's per-candidate mechanism,
`experiments/runway_intent_r2.py`): **R2a**, the per-airport plan experts on flights neither model saw
(1828): choosing the runway with `r11_lift` costs **+55 m** of paired mean FDE against the known runway,
B1 +405 m, the best rule (B3) +317 m — every pre-registered gate passes. **R2b**, the experts retrained on
day_a's training days, every day_a validation-day flight (5061): four of six gates; KSJC and KMSY fail
because the retrained experts cannot fly the minority-direction runways (KSJC 12R median 6 km, KMSY 02
26 km) and always-the-majority then ends nearer the truth — the head picks those flights 99-100 % right;
on the flights the expert flies when told the runway every gate passes and the choice costs **+73 m**
pooled (B1 +466, B3 +427). **R2c**, the belief re-asked every 30 s: 92-99 % beyond 20 km to 99.2-100 %
inside 3 km, flips mostly toward the truth, and never locking (fly toward the current top pick) is best
end to end at four of five airports (+20-51 m vs +32-98 m for locking at the first ask) — adopted.
**R2d**, the runway fan: 28 picker variants of the per-airport experts (each candidate's forecast and the
head's top pick; `plan_guidance_20260910` group, intents under `variants`), `check-publication` clean on
disk — **the running dev servers must be restarted before they load** (they booted before the files
existed and answer them with the SPA fallback). Colouring the fan by p_r is a frontend feature, not done. R3 (the
multi-runway scheduler) is above. Artifacts `outputs/POOLED/experiments/runway_intent_r2{,b,c,d}_20260913/`.

### R1.1 / R1.1b (2026-09-13, `docs/2026-09-13_runway_intent_plan.zh.md` §14–15)

**R1.1 / R1.1b** (the user: "先做 R1.1", before R2): a CANDIDATE-SYMMETRIC runway head — every R1
sample re-read as one row per candidate runway in that runway's own terms (`data.runway_features.
candidate_rows`), scored by ONE set of trees under a softmax across the candidates (a gradient-boosted
conditional logit, `experiments/runway_intent_r11.ListwiseBooster`, numpy histogram Newton trees with
R1's budget; sklearn cannot take the loss). R1's head is retrained on the same samples and matched
R1's artifact bit for bit, so every comparison is paired. R1.1 as pre-registered failed its key gate:
the per-runway constants in its rows (the runway's training share, the operator's share of its
landings) let the shared trees re-identify the runway, and KSJC's closure days stayed at 53-59 %.
R1.1b drops the prior and moves the operator's share to its deviation from the base rate, counted
only over training days BEFORE the sample's day (an out-of-fold share was anti-correlated with its
own block's labels — review). **`r11_lift` passes every §14 gate**: KSJC closure days 96.8 / 97.6 %
(B1 97.4 / 97.8, R1 54.7 / 57.2), the KRDU / KSMF side gain kept (+23.8 / +27.0, +23.9 / +31.4 points),
G5 at all ten cells, §11.4 everywhere; the day-blocked vs per-flight gap on the paired flights fell
from R1's 10.9 points (KSJC) to at most 0.5. Gate (i) is NOT a blind test — the variant was chosen
after reading the closure days; the blind check is the sealed test days. **Next: R2** — `r11_lift` as
the runway head, plan experts retrained on the day split, the end-to-end paired reading. Code
`a2a39d2` / `b10ed3e` / `0c8f107`; artifacts `outputs/POOLED/experiments/runway_intent_r11_20260913/`
(R1.1) and `runway_intent_r11b_20260913/` (R1.1b; `superseded_b10ed3e/` = the out-of-fold encoding).

R1 (one HistGradientBoosting runway head per airport, causal features, anchors on rings around the
airport reference, the DAY-BLOCKED split — operating days cut at 09Z, two partitions of the
non-test days — beside a per-flight control; code `22d4df5` + `e11ac8a`): at KRDU and KSMF the head
beats the best causal rule on the parallel side by 23–32 points pooled (19–30 at the arrival-slice
entry), minority runways 93–95 % against 40–49 %, NLL 4.5–7× lower — the §11.4 gates pass on both
partitions and the veto does not fire. KSTL: +4.3 / −0.01 points, a tie that fails G1 mechanically
on one partition — report KSTL as on par with the rules. **What the day split does**: on ordinary
days nothing (same flights, the day-blocked model against the per-flight one: −0.2 to +1.1 points
at four airports); on a day whose configuration the training days lack, it exposes a failure the
per-flight split hides — KSJC's two 30L-closure days (everything on 30R, both in one partition's
validation) read 55–57 % against B1's 97–98 %, while the per-flight model, trained on those days'
other flights, reads 100 % on the same flights (KSJC paired 97.9 vs 87.0). Cause, measured: the
hourly wind and time-of-day columns fingerprint the usual configuration (without them the closure
days recover to 87–90 %); the remaining 8–10 points are presumably the per-runway configuration
columns. R1.1 / R1.1b followed (above). Artifacts
`outputs/POOLED/experiments/runway_intent_r1_20260913/` (`superseded_db1e701/` is the pre-review
run, void); readout `run_ts.py runway_intent_r1_readout`.

R0 (no training; the plan heads' val splits; outer-test hashes out of every context pool):
the landing DIRECTION is solved by co-temporal landings (B1 96–99 % at all five airports); the
SIDE is the problem — side-given-direction 73–75 % at KRDU, 61–65 % at KSMF, ~90 % at KSTL, 93–95 %
at KSJC (KMSY has no parallels), with the minority runways at 33–49 % (KRDU 05R / 23L, KSMF 17L,
KSJC 30R, KSTL 11 / 12R). The causal rules are flat along the approach (the aircraft's own track
is the unused signal). Choosing by B1 / B3 raises the plan heads' paired FDE mean by +380–650 m at
KRDU / KSTL / KSMF, +20–30 m at KSJC. R1 (above) read the gates R0 pre-registered (§11.4).
Artifacts `outputs/POOLED/experiments/runway_intent_r0_20260913/` (readout `run_ts.py
runway_intent_r0_readout`); code `fc57d80` + `9e3ac49`, R0a re-run on operating days at `22d4df5`.

**Found on the way, not runway choice:** the per-airport plan heads fly rarely-landed runways badly
EVEN WITH the true runway — KMSY 02 (29 val flights) FDE median 23.2 km, KSTL 24 and KSJC 12L (one
flight each) 23.0 / 5.7 km, and KSMF 17R 956 m / 17L 215 m against 30–70 m on most runways. R2b
sharpened it: retrained on fewer days (day_a's training days), the KSJC expert lost 12R too (median
77 m -> 6.0 km on the same 16 flights) and KMSY 02 stays at 20-26 km under both; on such runways a
WRONG runway's forecast can end nearer the truth, which is what failed R2b's KSJC / KMSY gates. Not
investigated; the per-runway FDE tables are in the R0 and R2b readouts.

### The plan (2026-09-13)

The user's point (2026-09-13): which runway an arrival lands on must be predicted, not given —
it is the intent that multi-aircraft interaction turns on. Today every ts path reads the landed
runway (threshold anchor; the plan path's CIFP skeleton), so every published number is
runway-given. Plan: an explicit runway head `p(r | causal context)` over the airport's
candidates × the existing per-runway predictors as conditional experts; multi-runway
assignment in the §9 step 6 scheduler; a joint scene model only if that shows interaction
error. R0 (zero training: `experiments/runway_hypotheses.py` v4 on all five airports, causal
baselines B0–B4, the anytime curve, the per-airport cost of a wrong runway) is the block above.
**D1 decided 2026-09-13**: (a), the day-blocked split, for the R series only; R1.1 before R2
(decided). **Open, the user's:** D2 scope, D3 the candidate set (v5 roster lacks KRDU 32 / KSMF 35R),
D4 standalone classifier vs backbone head, D5 how the thesis states the runway-given premise.

---

## Decisions left open by the 2026-09-09 package review (branch `dev-pkg-review`)

The review's bugs are fixed (its §7 is the ledger; changelog entry of the same date). These
are the items that change a stored artifact's name, reuse or number, or a protocol, and so
were NOT made without the owner:

- **C-3 — DECIDED and DONE 2026-09-09: the three identity-bearing fields name the run.**
  `lr_plateau_patience` / `lr_plateau_factor` (`lr-patience=` / `lr-factor=`) and
  `random_train_anchor_min_future_s` (`anchor-min-future=`) are in `META_FIELDS`; only `device`
  and the never-set backbone knobs stay in `KNOWN_UNNAMED_FIELDS`. Measured over the 219 stored
  `history.json` configs: 146 display names / slugs moved (75 gain a spelled token, 71 only
  their folded `+N more` count and slug hash — the named recipes pin the scheduler pair, so
  recipe runs are untouched), 0 loadability changes, no directory or category key moves. 132
  published frontend categories carry the old label until relabelled — 109 publisher-managed
  (`publish_ts_experiment_trajectories.py --refresh-labels-only`, once per publication root:
  `KRDU/`, `KSJC/`, `POOLED/experiment_predictions`, `POOLED/checkpoint_publications`) and 23
  hand-published `ts_*` keys (`docs/relabel_published_categories.py`); labels only, no CZML,
  records, keys or directories. Owner-run, one pass.
- **C-4, the duration floor under `uniform`.** `control_duration_uniform_floor` is read by the
  `factorized` head only, but its default is 0.8 and every recipe pins 0.0, so 88 stored
  `uniform` runs carry a non-default inert value and a refusal would stop them loading. It
  stays a field on `DurationSpec` after the §4.3 split (2026-09-10): a `Factorized(floor)`
  variant would refuse the `0.0` every recipe pins under `uniform`, i.e. every recipe. A custom
  `uniform` run can still wear `duration-floor=`; retiring the pin is a recipe-version change.
- **C-7, the test-release ledger is bound to the DIRECTORY.** `test_release.json` sits beside
  `checkpoint.pt`; a copy of the checkpoint elsewhere can be frozen and released again. The fix
  is a registry keyed by the checkpoint digest outside the run directory — where it lives and
  how tests isolate it is the decision. No ledger exists on disk today, so it costs nothing
  whenever it is made.
- **C-9, `cv_results.json` misnames the selection metric as a loss.** `mean_/std_val_macro_loss`
  at the candidate level hold the SELECTION value (ADE, metres); the fold rows' `best_val_macro_loss`
  is the loss. Renaming means a schema bump, after which the two stored files (2026-08-16
  `POOLED/ts_patchtst_normalized_time`, 2026-08-18 `KSJC/experiments/cv_tau_bank_20260818`)
  are no longer reusable by `--skip-cv`. The writer says so in a comment; nothing renamed.
- **C-12, the load-factor floor — DECIDED 2026-09-09: the grader stays at 0.5.** The learned
  head's box floor is 0.2 (`outputs/envelope.py`) and `flyability`'s hard floor is 0.5, so a
  control-path segment at n ∈ [0.2, 0.5) is unflyable by construction on the published metric.
  Neither number moves: the grader's floor is the flyability CLAIM every published number was
  read against, and the head's box is the old path's search space (moving it would change every
  control checkpoint's decoded controls). The plan-and-guidance design's guidance layer commands
  the load factor inside `flyability`'s envelope, so the new path cannot inherit the gap. Still
  unmeasured, and only of interest for reading the old arms: how often a trained head emits
  n < 0.5 (`control/training/diagnostics.py` saturation counts would say).
- **A-2's consequence.** Every number `run_ts.py history_ablation` published before 2026-09-09
  was scored against a truth taken `(max L − L)·dt` before its anchor; the runner is unchanged
  and correct now, its stored outputs are not.
- **§5 DONE 2026-09-10** (the review's §5 resolution paragraph has the per-axis outcome and
  the two corrections: the observed clock and scaled-tcv are recipe-pinned and stay
  selectable; corridor-bounded stays the candidate default). **Still to do, in the review's
  §6 order:** §4.3 config split DONE 2026-09-10 (typed views over the flat dataclass, the
  ownership rule; defaults unchanged; 0 names moved); §4.2 output strategies DONE 2026-09-10
  (`outputs/`, one strategy per path, the spine no longer branches; 0 names moved); §4.4 loop /
  predict extraction DONE 2026-09-10 (`prepare_session` / `train_epoch` / `validate_epoch`;
  `parse_predict_options` → `PredictOptions`); §4.5 runners DONE 2026-09-10
  (`experiments/<name>.py` behind `run_ts.py <name>`, `repo_layout.py`, the twelve red pipeline
  fixtures fixed — the suite is green); §4.6 tests DONE 2026-09-10 (`tests/support.py`, the
  5,855-line `test_ts_transformer.py` split into 23 single-topic files); the folder grouping
  DONE 2026-09-10 (`data/`, `geometry/`, `backbone/`, `training/`, `inference/` — pure moves,
  the layout table in `CLAUDE.md`). The review's §6 order is complete; each step was one
  commit with the full suite and the stored-run census as the acceptance test. The "name every field against
  the nearest recipe" grammar change is deferred to after the folder grouping (it moves
  stored names and needs its own relabel pass like C-3's).

## Plan-and-guidance (2026-09-10 → 09-12) — steps 0–5b built and measured; next: step 6, the multi-aircraft scheduler demonstration on the per-airport heads

`docs/2026-09-09_plan_and_guidance_design.md` (v4; §12 carries the numbers), branch
`dev-plan-guidance`. Built: the shared rollout parts moved out of `outputs/control/`
(step 0); the procedure skeleton reader (`flight_scenarios.procedure_final.procedure_skeleton`,
`outputs/plan/skeleton.py`), the eight plan extractors (`outputs/plan/extractors.py`) and
`run_ts.py plan_extractors` (step 1); the guidance layer (`outputs/plan/guidance/{route,
controller}.py`, `outputs/plan/forecast.py`) and `run_ts.py plan_oracle` (step 2; reviewed,
the review's route and controller findings applied and re-measured). Artifacts:
`4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/{step1_extractors,step2_oracle}/`
(the pre-review oracle kept as `step2_oracle.superseded-20260910T2330Z`).

- **The oracle ceiling passes the §7 veto on both strata** (KRDU val 1404, the truth's own plan
  flown): straight-in ADE 204 m / chamfer 34 m / arrival-time MAE 4.8 s against
  native32's 109 m and 25.9 s pooled; vectored ADE 1591 m against 2870 m; 99.7 % fully
  flyable, 99.7 % established at the threshold (the hook stack: 46 %, ~70 %).
- **Nearly by construction laterally, not yet vertically**: after the join 1.4 % of flights
  leave the design corridor (excess p50 59 m) and 10.0 % the glidepath window (excess
  p50 68 m straight-in) — the height law's from-above capture, controller tuning before
  the reference claim is quoted. Vectored flights arrive -26 s early at the median with the
  route laid to the plan's length (the schedule over the vectors; step 4's assigned-time
  stretch absorbs it by construction).
- **Step 2b (2026-09-11, §12.3): K ≤ 4 fly-by fixes close the vectored shape gap** —
  chamfer 636 → 268 m, Fréchet 2166 → 1648 m from the truth's own fixes
  (`PlanLabels.waypoints` + `waypoint_speeds`, `run_ts.py plan_oracle --route waypoints`);
  with each fix's speed on the schedule the vectored arrival-time MAE is 16.1 s
  (27.3 s) and the corridor after the join 19.9 %
  (3.4 %). The 60 s anchor leaves 45.6 % of plans censored against 59.2 % at L−1: the
  earlier anchor for step 3 is a shorter window (`--anchor-s`), not the 20 km
  remaining-path bin (later than L−1 on a vectored track).
- **The vectored point claim is a timing claim with the truth's own three route
  parameters** (chamfer 636 m): `d_join`, `side`, `L_pre` say how long the 35 km of radar
  vectors are, not where they go. §8 risk 1 measured.
- **Step 3 (2026-09-11, `dev-plan-next`, design v5 §12.4): the single-step head is BUILT and
  runs end to end** — `prediction_output=plan` (`outputs/plan/{labels,model,strategy}.py`),
  the next-instruction readout (`step3a_next_readout/`: at the 60 s anchor a vectored
  flight's next fix is 25–39 km / 3–5 min ahead, the median baseline 5 km off; at random
  anchors 13 km / 112 s and 12.5 km), the rolled oracle (`--route next`; within ~50 m of
  vectored ADE of the whole-path waypoints flight on the smoke once a leg ends where the
  aircraft has executed its instruction — 915 against 862), and the first head on 403
  flights rolled through the guidance (vectored ADE 7657 m against the ceiling's 2293 on
  the same 48; straight-in 827 against 554 once "no fix ahead" covers the established
  flights). Full val: the rolled oracle 1492 m of vectored ADE at L−1 against the whole-path
  flight's 1159; the full head (6856 flights) rolled at the 60 s anchor 3781 m against its paired
  ceiling's 1845 (straight-in 635 against 584), its next fix 1.6 km off at random
  anchors against the median baseline's 10.7 km. The fan over the next fix and the
  corner are next. Decided (v5): the single step, not §8's extra waypoints.
- **Step 3(d) (2026-09-11 night, `dev-plan-lockstep`, design §12.5): the rolling is receding-horizon
  and lockstep** — the policy asked every 30 s, the route in force tracked, the group stepped
  together (`plan_oracle --rolling lockstep`, the default): the oracle 1847 m of vectored ADE at
  L−1 against the leg form's 1492 in 69 s for the split (the leg form ~90 min). **The head
  re-asked every 30 s is WORSE**: 5391 m against 3781 asked once per leg (4991 every
  60 s; its lockstep ceiling 2184) — asked on its own flown windows it never trained on, its
  orders jitter and 29 % of vectored flights run to the cap. Next: train the head on rolled
  windows; hold an order unless the change persists; then the fan over the next fix.
- **Step 3(e) (2026-09-11, `dev-plan-rolled`, design v5.2 §12.6): the head trained on ROLLED
  WINDOWS** — `run_ts.py plan_rolled_windows` (the lockstep flight's windows at every step,
  labelled by the truth's queue at that state; `--policy truth` / `model`, `--extend`),
  `train --plan-rolled-windows-path --plan-rolled-share`; the val split's rolled windows read
  every epoch. MEASURED (§12.6): the re-asked head at 60 s, vectored ADE 5391 → 3641 m at
  share 0.75 (the once-per-leg 3781, the ceiling 2184), established 55 → 94 % pooled, capped
  29 → 9 %; shares 0.25/0.5/0.75/1.0 = 3833/3862/3641/3685 (the rolled val loss falls with
  the share, the observed objective does not move); DAgger round 1 +73 m paired, no gain.
  The 3(d) gate is still open (2870 m). Next: hold an order unless the change persists two
  asks; the fan over the next fix; a second seed.
- **Step 3(f) (2026-09-12, `dev-plan-hold`, design v5.3 §12.7): the order HOLD — MEASURED,
  NOT ADOPTED.** `forecast.held_order` adopts a materially different order only once given
  on `hold_asks` consecutive asks (`plan_oracle --hold-asks N [--hold-flips-only]`). On the
  share-0.75 head at 60 s: hold 2 vectored ADE 3641 → 3816 m, established 86.5 → 60.2 %,
  FDE mean 2803 → 5470 (L−1: 3719 → 3938, 84.2 → 55.4 %); flips only 3723 m / 79.7 %. The
  head's fix WALKS between asks (766 m p50, 2.7 km p75), so two asks never agree within 1 km
  and the step-0 fix stays in force (2038 of 3143 held steps a moved fix). Default
  `ORDER_HOLD_ASKS` = 1 (v5.2's behaviour); the axis stays; `run_ts.py plan_oracle_pair`
  pairs two plan-oracle artifacts flight by flight (hold 1 reproduces §12.6 to ≤ 0.07 m —
  the head's float32 CPU forward is not bit-reproducible between runs).
- **Step 3(g) (2026-09-12, `dev-plan-fan`, design §12.8): the FAN over the next fix — the
  MIXTURE OBJECTIVE ADOPTED, the one-step fan NOT.** `plan_fan_components` = 4 makes the
  instruction group a 4-component diagonal-Gaussian mixture trained by its NLL; the top-weight
  component is the point prediction. Two seeds, KRDU val, 60 s anchor, paired within seed
  against the L1 point head of the same recipe: vectored ADE 3641 → 3087 m (1337) and
  3439 → 2837 m (2024), established 86.5 → 94.2 % and 89.5 → 95.2 %, straight-in unchanged;
  the point head's own seed spread ~200 m. The 3(d) gate (2870 m) reached at one seed, missed
  at the other. As a fan (`run_ts.py plan_fan_readout`): minADE_4 ties a 5 km displaced ring
  (2454 vs 2450), the nearest member beats the top-1 less often than the ring's (58 vs 79 %);
  2σ coverage 98.7 % is the alternatives' width. Three of four components carry weight. The
  config default stays 0 (stored point heads keep their layout); a named plan recipe should
  pin 4. Next: step 4 on the K = 4 head; a fan that carries a claim needs samples or a member
  tracked across asks (listed, not planned).
- **Step 4 (2026-09-12, `dev-plan-cta`, design §12.9): the ASSIGNED TIME delivered, the assigned
  JOIN not in this form.** `strategy.Assignment(arrival_time_s, d_join_m)` replaces the head's
  `T_s` / `d_join_m` at every ask; `fly_lockstep` closes the time on the route in force from the
  aircraft's progress point (`guidance/timing.close_time`: the held speed between the stall floor
  and the maximum with the deceleration point moving with it, then the builder's hold / dog-leg
  bracketed over its quanta, X = what neither absorbs, signed). KRDU val, K = 4 head, 60 s
  anchor, the truth's time: dt MAE 35.1 → 10.2 s pooled (straight-in 3.2 s, |dt| p50 1.5 s),
  paired ADE −447 m (lower on 81 %; straight-in 749 → 281, vectored 3087 → 2655), fully flyable
  100 % in every arm, chamfer unchanged; established 97.3 → 95.0 %. −60 s met to the second by the
  median flight of both strata (vectored established 72 %); +60 / +90 s absorbed at the floor by
  the vectored stratum (flown 6–22 s short), not by the straight-in one (X p50 46 / 76 s, no path
  to stretch). The §7 rows: arrival-time MAE and fully-flyable PASS; "assigned time" 2655 against
  1596 and "time + join" 3451 against 1400 FAIL — the join under the head's own fix is an
  inconsistent order (alone 3876). `plan_oracle --assign-time truth --assign-time-offset-s S
  [--assign-join truth]`. Open: the flown shortfall under large delays (the controller's floor
  against the plan's), establishment under an advance, the route assignment as fixes.
- **Step 5 (2026-09-12, `dev-plan-pool`, design §12.10): the pooled five-airport K = 4 head —
  the closure transfers, the pooled head costs the home airport.** `run_ts.py plan_cohort`
  (the development cohort a random-anchor run needs; reproduces the hand-written KRDU one
  exactly) → 21,911 / 4,496 flights over five airports; the truth table 292k samples; the head
  at share 0.75. KRDU val, 60 s anchor, paired against the KRDU-only head: vectored ADE
  3087 → 3678 m (+227 p50, lower on 37 %), established 94 → 84 %; with the truth's time
  2655 → 3684 (+591). The other airports: straight-in dt MAE 3–5 s with the time, fully flyable
  99.6–100 %, vectored 2.7–4.2 km with established 55–85 %. The pooled head is not KRDU's
  delivery; the other four airports get per-airport heads next (5b). `plan_oracle --by-airport`,
  `plan_oracle_pair --common`. The bootstrap head of a pooled run needs one epoch, not 120.
- **Step 5b (2026-09-12, design §12.11): per-airport K = 4 heads for KSJC, KSTL, KSMF, KMSY
  (their rolled windows from the pooled table) against the pooled head on their own flights,
  60 s anchor.** The own head's vectored ADE is lower on four of five airports unassigned
  (pooled head +79 to +227 m paired; KSJC −40 on 207 flights) and on all five with the truth's
  time (+33 to +591); straight-in and the time closure the same under either. Per-airport
  heads are the delivery everywhere (`step3g_fan4_head`, `step5b_<ICAO>_fan4_head`); the pooled
  head is not. **Two seeds (the same evening, §12.11's last block): KSJC and KSTL agree within
  the KRDU line (paired +30 / +22 m); KSMF and KMSY do NOT (+358 / +215 m paired, KSMF's vectored
  mean +1.3 km with the rolled cap 2.8 → 19 %) — on those two ~2.4k-flight cohorts the own-vs-pooled
  margin (+79 / +217) is inside the seed line and the row is undecided; the delivery there stays
  the seed-1337 head. Open: a third seed on KSMF/KMSY, or the pooled head fine-tuned per
  airport.** Next: step 6 (the scheduler demonstration).
- **The oracle's vertical verdict changed (2026-09-11, schema v2)**: the glidepath window
  binds inside the FAF only, the coded floor before it, and the truth's own rows are graded
  beside every flight — §12.2/§12.3's glidepath shares (10 %, 7.7 %) were read from the
  join and flagged the truth's own level segment; re-read them under v2 before quoting.
- Step-1 findings that changed the design (§12.1): 59 % of KRDU val joins before the L−1
  window (three route parameters censored there); straight-in joins are published
  transitions (88 %), vectored joins radar vectors (2 %); `V_final` has no floor at V_ref
  (the gate's window is an airspeed with the headwind); `h_capture` is the LATERAL join's
  height; `V_mid` is the held path's mean speed; the deceleration is a 0.5 m/s² law.

## Current state (2026-09-07) — the latent-intent design supersedes everything below it

The control path was redesigned on 2026-09-07 (`docs/2026-09-07_latent_intent_design.zh.md`,
its §〇 status table is the live one). What that design settled and what is running:

| step | state | number |
|---|---|---|
| L0 width oracle | done | N\* = 32: 96 operating numbers replace 257 (vectored fit 203 m against 962 m of intent) |
| L1 low-dim head | done (`l1_lowdim_20260907`, `docs/2026-09-07_l1_lowdim_results.zh.md`) | native32 1322 m ≈ N=64 baseline 1333; dense / no-teacher 2515 — the trajectory-error loss alone is NOT enough, the imitation teacher stays |
| L2 latent intent | four campaigns run (L2.c β ladder, L2.d warm posterior, L2.e′ free bits, and the posterior probe that explains them); **L2.f is next** | L2.d's warm β=0.01 is the best point estimate so far (1214 m pooled against native32's 1322) and the fallback. The probe found the failure the totals hid: in all three L2.d/e′ arms the posterior MEAN sits on the prior mean (< 0.2 prior σ) and the KL is spent narrowing q, so z is a denoised constant — which is why shuffled ΔADE ≈ 0, the z-oracle bought 227 m and N(0, I) samples beat the trained prior. L2.f puts information back in the mean, one arm each: `latent_beta_warmup_epochs=40` and `latent_aux_duration_weight=1.0`, arms `docs/experiments/l2f_mean_information_arms.json`, code on `dev-l2f` with the epoch-1 diagnostics (`run_ts.py latent_readout --history`) and `run_ts.py latent_probe` |
| L3 CTA conditioning | code done + reviewed (`dev-l2`); the delivery layer for a LATE CTA is three predict-time hooks, L3.c/d run and L3.e built but **not run** | `cta_conditioning=given`, `predict --cta-offset-s`. L3.c: the barrier holds the corridor but the trajectories stall on the outer segment. L3.d: a speed floor alone has nowhere to put the delay (total time fixed by the CTA + path fixed ⇒ mean speed fixed) and trades stall for thrust-over-max, arriving early and flying on. L3.e (`dev-trombone`, 2026-09-08): `outputs/constraints/trombone.py` + `barrier+trombone` / `barrier+speed-floor+trombone`, and `predict --truncate-at-threshold` so the readouts cover the approach and not the flying after it. Arms `docs/experiments/l3e_path_stretch_arms.json` (KRDU val, four predict-only, dry-runs clean); **known cost: ~58 % of the fleet is already aligned with the course at the anchor and cannot be stretched at all** — its unabsorbed delay is reported as `tromboneDelayS` with an engaged share of zero. See §六 L3.c/d/e of `2026-09-07_latent_intent_design.zh.md` |
| L4 scene encoder | **NOT built — gate failed** | scene entity features add nothing (d_join R² 0.37 vs 0.38); the observable lead ETA correlates 0.11 with the lead's true landing |
| L5.a fitted teacher | code done + reviewed (`dev-l5`); its startup blocker fixed (`e8df12f`) | the fit itself is a GPU job and **has not run**: `docs/experiments/l5_fitted_teacher_arms.json` points at `4dTrajectory/outputs/KRDU/experiments/l5_fitted_teacher_20260907/basis_fit.json`, which **does not exist until** `run_ts.py control_basis_oracle --checkpoint .../l1_lowdim_20260907/L1_native32/checkpoint.pt --out <that directory> --splits train,val` has been run (8255 flights, ~2–3 h). Until then both arms die at the dataset build, by design — the config carries the path, the dataset opens it. It used to die EARLIER, at the provenance check, because it fingerprinted the manifests without the pre-split lateral-pass roster (14 435 candidates against the checkpoint's 14 378 eligible); `e8df12f` moved that rule into `data_provenance.checkpoint_data_provenance`, which every replaying runner now calls |
| A/B anytime prediction + calibrated ETA | **A0 / A1 / B0 measured; A0.b code done and reviewed (`dev-a0b`), its two arms NOT trained**; the rest design only | `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §〇 is the live status table (commands §〇.1, B0's numbers §〇.2). Built and measured: `run_ts.py anytime_curve` (A0-fixed over a remaining-path anchor grid, strata fixed at L−1) and `run_ts.py eta_error_readout` (B0, CPU readout) — KRDU val vectored \|Δt\| p80 **65.8–72.5 s** against straight-in **12.0–20.3 s**, so per-stratum ETA calibration is necessary and B's 120 s veto has about one doubling of margin. **Three A0-random arms have now run and all fail the L−1 veto** (2949–2990 m against native32's 1322); §2.4c's diagnosis is two mechanisms — the LR scheduler stepped on a stalled selection metric (9.4e-7 by epoch 60) and a time-uniform anchor draw that, pooled over flights, sits nearer the runway than its own anchor population (draws 34.3 % under 6 km against a population 25.1 %). **A0.b is the fix and is CODE ONLY**: `lr_plateau_metric` + `random_train_anchor_sampling=remaining-path-uniform`, arms `A0b_lr_objective` / `A0b_lr_objective_path_uniform` in `docs/experiments/a0_random_arms.json` — a GPU job that has not run, and until it does the random-anchor line has no established base. Still design: A2 / A3, B1–B3 (quantile duration head + split-conformal + `--cta-from-quantiles`, the first CTA arm that does not read the future), deliverable B4 = 80 % interval width vs remaining path and `s_freeze`; C = conditional diffusion over the 96 operating numbers trained on the L5.a fitted table, guided THROUGH the differentiable rollout (observed prefix / CTA / corridor), a REPLACEMENT for the CVAE if L2.e′ leaves z under ~1 nat, gated first by a retrieval ceiling on the same table (C0) |

**Abolished by that design** (entries below are history): the P1.d closure tracker (its BLOCKER
is not being fixed; the code was DELETED 2026-09-07 by audit T1-9), the K join-anchor decoder,
the `2026-09-07_control_training_review` P0/P1 objective fixes. The closure output stays as a comparison arm only.

---


- **Scene design Phase 0 DONE 2026-09-05 (`scene_phase0_20260905`, KRDU; results
  `docs/2026-09-05_scene_phase0_results.zh.md`, diagnostics
  `docs/phase0_intent_diagnostics.py`).** The TRUTH join point as input: vectored ADE
  2858 → 2356 m, duration error 39 → 22 s; + the truth lead ETA: no increment; + the truth
  remaining time: 2011 m (−30 %), pooled 1005 m, duration error 5 s — but the time-free path
  error never improves (chamfer 942 → 791 → 850 m). Pre-registered gate (< 1.5 km) not met
  and shown to be mis-sized (truth path + naive speed profile 1.3 km; trombone from the truth
  join + truth timing 1.7 km). Open — a decision for the user: (A) stop, (B) revise the design
  to (d_join, T) decisions, (C) fix the output side first (geometric closure); the results
  doc recommends C then B. **P0 geometric readout DONE 2026-09-05** (`geometry/geometric_metrics.py`
  in both readouts; `readout_geometry.*` backfilled in `scene_phase0_20260905` and
  `control_hooks_v2_20260906` KRDU + KSJC): the join arms gain 15–20 % on chamfer / Fréchet /
  arc-ADE, the duration arm gains none of it; the v2 soft barrier is the only hook that also
  improves vectored geometry (KRDU chamfer 942 → 886 on 81 % of flights). The state output's
  saw-tooth polyline (heading reversals at ~50 % of nodes, length ratio ≈ 2) takes it out
  of the arc family (chamfer / Fréchet still read) — a finding about the state export, not
  yet acted on. **Direction C chosen; P1.a / P1.b DONE 2026-09-05** (`closure_geometry.py`,
  `closure_profile.py`, `docs/p1_closure_oracle.py`; artifacts `closure_p1_20260905/`):
  the via-pose Dubins family (F3) fitted to the truth reaches vectored chamfer p50 180 m /
  Fréchet 1179 / truth-timed ADE 510 m (gate passed), with identifiable canonical labels
  on 96 % of fitted flights; on the truth path a K=4 slowness + K=4 height profile
  reaches 110 m. **P1.c DONE 2026-09-06 (`closure_p1c_20260905`, report
  `docs/2026-09-06_closure_p1c_results.zh.md`): both gates pass.** The `closure` output
  from the ego history alone beats simple-v3 on every stratum (pooled ADE 996 vs 1333,
  vectored 2197 vs 2858, straight-in 310 vs 469); with the truth (d_join, T) as inputs it
  reaches vectored ADE 1235 / chamfer 492 where the control head with the same inputs sat
  at 2011 / 847 — the output side WAS the bottleneck; the family's own ceiling is 455 m.
  Open: flyability, read as a delta — closure paths are 22 % fully flyable under the
  clean polar against the control baseline's 0.1 % (its violations are 59k stall
  samples) and the observed 98 %; per sample closure sits at 99.8 % = the observed, its
  few violations per flight being bank jumps at the CSC junctions and thrust jumps at
  the knots. **P1.d DONE 2026-09-06 (option b, report
  `docs/2026-09-06_closure_p1d_tracking_results.zh.md`)**: the drawn reference flown by
  the point-mass rollout under `control/constraints/closure_tracking.py` (RETIRED — code
  DELETED 2026-09-07) costs ≤ 100 m
  of ADE (C_pred +9 m pooled, +25 m vectored) and brings the fully-flyable rate from 22 %
  to 92 % (observed 98 %) — the delivery form is closure + tracking. **Review (opus, same
  day) found the tracker's nearest-node search unguarded: 8 of 1404 via-Dubins flights
  snap to the wrong leg (endpoints 6–19 km off) and four of the five largest "tracking
  gains" are those; the pooled cost is +10.5 m without them. Five SHOULD-FIXes
  (vertical-law sign untested, stall floor without the commanded load factor, ISA
  density on chart height, the height pinning outside the label contract, the gain
  docstring). Unfixed: the user decided to REDO the plan** — the design doc's §〇.1 is
  the snapshot to resume from. The P2 data plane (`trajectory_data_process/scene_index`,
  `flight_scenarios/scene_context`, `ts_transformer/scene/features`) is committed as WIP
  (`045c233`): 13 tests, not reviewed, no explainability measurement, no KRDU index
  built. Not started: any scene encoder.
- **Hard procedure constraints in TRAINING — survey + plan written 2026-09-08, nothing built
  (`docs/2026-09-08_hard_constraints_survey_and_integration_plan.md`; 83 papers annotated with
  their core formulas in repo `docs/literature/procedure_hard_constraints/`).** What the
  literature settles against our evidence: the tanh-bounded state output is HardNet-Aff's
  closed-form clamp (its Prop. 7 clamp-vs-tanh is a cheap arm); the lazy control network is the
  known result of training through a filter with `CC` bookkeeping and is removed by a swept
  correction penalty `α‖u_raw − u_filtered‖²` (Pizarro Bejarano 2025), the OptLayer-CPC form
  (raw rollout scored beside the filtered one), a feasible-entry committed gate, or a bijective
  gauge map onto the barrier's bank interval — and predict-time-only filtering carries a
  quadratic-in-horizon imitation error (Geiger & Straehle 2022), so it is not the end state; the
  C_dual divergence was an unreachable level (dual best response `+∞`), fixed by annealing /
  learning the level (Hounie 2023) with νPI. Plan H0–H6 in §4: H0 gate readout (no GPU) → H1
  learned monotone commitment gate on `state` → H2 control arms on the L1.c base (after L1.c,
  every arm predicted with AND without the hook) → H3 vertical barrier → H4 only if L1.c passes.
  Work lives in the `../thesis-hc` worktree (`dev-hard-constraints`), unmerged.
- **Control command-hook campaigns DONE 2026-09-06 (`control_hooks_20260906` v1 at KRDU,
  `control_hooks_v2_20260906` at KRDU + KSJC; report
  `docs/2026-09-06_control_hooks_results.zh.md`).** Adopted: the v2 soft barrier as a
  predict-time safety layer; not adopted: any hook inside the training loop (six arms, none
  beat its predict-time counterpart), the hard gate. The nominal law's code was archived
  2026-09-07 (`archive/nominal_law_hook_2026_09/`; it was the unadopted hook's only
  remaining consumer once T1-9 deleted the closure tracker). Open: the combined
  lateral-barrier + vertical-nominal hook at predict time — reviving the vertical half is a
  deliberate un-archive, not an import; the baseline ending 157 / 162 m below the glidepath
  — traced 2026-09-07 to the last minute of the rollout (on the final it sits within ±13 m
  of the glidepath; at the truth's landing time it is 540–680 m short and 140 m low, path
  angle −4…−5° vs −3°), NOT to data, coordinates or the fitted tail (2–6 s), and read as
  four objective-design faults — the isotropic 10 km metre-scale position loss (both
  paths) prices a 150 m height error at 2e-4 per endpoint (a mean over 64), the 47×
  open-loop imitation teacher never speaks to the rollout's own drift, the threshold
  anchors present (the 1.25-weight fitted terminal row, `state_endpoint_loss_weight`)
  share the 10 km scale and nothing stops at the ground, and the path loss carries no
  gradient to the time grid (training rescales durations to the truth; the overrun is
  inference-only):
  `docs/2026-09-07_control_training_review.zh.md` (P0: per-channel position scale, the
  vertical-only procedure term, a threshold-plane crossing loss; P1: a closed-loop
  DAgger-style teacher from the guidance laws). **The km-level error is elsewhere**: vectored
  flights carry 76 % / 60 % of pooled ADE and their error is the ATC join decision, which the
  ego-only input cannot see and a single-output head can only average — design for traffic
  context + join-anchor multimodal output (scene encoder, K join-distance anchors as decoder
  queries, WTA training, top-1 stays on the existing record contract, Phase 0 = oracle
  upper bound before any architecture work): `docs/2026-09-07_scene_join_anchor_design.zh.discard.md`; a "committed to the final" gate for the vectored flights the v1 / hard
  barrier hurt (gate opening at d < 8 km or ≥ 16 km; every bin is net positive under v2 soft); a second seed at KSJC (its −66 m FDE gain is the smaller
  effect); PatchTST and the other three airports.
- **Final-approach constraint campaign DONE 2026-09-04/05 (`final_constraint_20260904`, KRDU +
  KSJC, 3 predict-only + 5 trained arms per airport; report
  `docs/2026-09-05_final_constraint_results.zh.md`, readout `docs/compare_constraint_arms.py`).**
  Bounded output adopted as candidate default (see the config entry above); penalty vetoed;
  projection kept as deployment fallback. Not done: making `corridor-bounded` THE default
  (decide together with the state-v3 continuity term, which addresses the start of the path
  the corridor does not), PatchTST, control output.
- **Procedure constraints in the learned model (2026-09-04 design + measurement):**
  measured on every 3rd rostered arrival (`docs/measure_procedure_adherence.py`) that **0.0 %**
  of observed KRDU/KSJC flights pass an off-axis IAF of their runway's RNAV(GPS) procedure,
  that 85–97 % (KRDU) / 38–83 % (KSJC) are established in the k=0.5 LPV cone by the FAF,
  and that once established 87–99 % of samples sit inside the cone and the −60/+120 m
  glidepath window (the ±22 m gate is met over the whole final by only 14–69 %). So the
  only data-consistent procedure constraint is the final segment (corridor + glidepath,
  gated by each flight's own join distance, never `d_faf`); IAF legs / pre-FAF join
  window / fix discs are normative and must not enter a loss. Design + measurements:
  `docs/2026-09-04_procedure_constraints_design.zh.md`; the method survey (penalty,
  bounded reparametrization, projection layers, primal-dual, sampling, two-stage with the
  optimizer) with reading list and the P0–P3 order:
  `docs/2026-09-04_constraint_methods_survey.zh.md`.
- **Index of the 2026-09-03 frame / runway / state-output experiments (four docs, one
  narrative, Chinese): `docs/2026-09-03_runway_frame_experiments_index.zh.md`; the
  runway-assignment reading list is `docs/literature/runway_assignment/README.md`.**
- **Runway-hypothesis expansion DONE 2026-09-03 (`run_ts.py runway_hypotheses`, no training):**
  one threshold-anchored forecast per candidate runway, scored in the true runway's chart.
  The assigned label reproduces the baseline bit-for-bit (the chain check). What the label is
  worth: at KRDU a causal "active configuration" rule (most-used runway among development
  landings in the 30 min before entry) recovers the DIRECTION (majority runways 80–83 %) but
  guesses the majority sibling for the minority runway (05R/23L 29–31 %), costing +19 %
  pooled FDE (+30 % straight-in), i.e. ~500–800 m on those flights; at KSJC the same rule is
  93.8 % right and costs nothing (30L/30R are 230 m apart). An oracle over the real sibling
  gains 79 m of median FDE at KRDU against 32 m for a mirror-image fake sibling at the same
  separation, so about half of a K=2 sibling oracle is picking the luckier forecast, not
  runway knowledge; at KSJC the fake sibling gains MORE than the real one. The forecast's own
  closest approach to its hypothesised runway is useless as a selector (37–45 %). Left/right
  between parallels is the genuine unresolved mode; direction is not. →
  `docs/2026-09-03_runway_hypothesis_expansion.md`
- **The state model's KRDU endpoints sit ~250 m NW of every runway, and it is the model,
  not the data** (`docs/2026-09-03_krdu_nw_endpoint_bias.md`): a world-fixed translation of
  the whole predicted path present from the FIRST predicted step (240–350 m off the
  aircraft's actual position, path then parallel to the truth within 1.3°), on straight-in
  flights (established +204 m lateral miss, vectored +24 m), reproduced by noise-free
  synthetic straight-in histories, both seeds. Sign = KRDU's population-mean lateral drift
  (63 % of anchors SE of the centreline; observed +60 s drift median 0, mean +192 m NW);
  KSJC's drift is SE-ward and shows no bias. The objective cannot see it: 300 m on a
  straight-in is ~9e-4 per point against a ~0.08 pooled loss dominated by vectored
  kilometres, and the state output has no continuity to the anchor and no cross-track
  term. Read every arm-A per-runway cross-track number with this translation in mind.
  The anchor-relative output (state-v2 candidate, same doc set) fixes the start of the
  path and the straight-in stratum but loses ~350 m of vectored FDE at KRDU on both
  seeds — vetoed by its own pre-registered rule; a continuity term on an absolute output
  (keeping the endpoint prior) is the open next candidate.
- **Airport-frame ablation DONE 2026-09-03 (14 runs, KRDU + KSJC, two seeds; keep `enu`).**
  Removing the threshold anchor makes the model average across each parallel pair (KRDU:
  endpoints nearer the sibling 1.5 % → 12–15 %, minority runway pulled ~600 m, its FDE
  +30–45 %); target coordinates as input channels change none of that; the vectored-stratum
  gain (H2) flipped sign on the second seed. Seed floor on this axis: the threshold arm moves
  5–22 m pooled ADE across seeds, the airport arms up to 107 m — read every margin against
  that. Runner `run_ts.py frame_ablation` (state arms, val split, resumable, no CV, no CZML;
  `--experiment-id` runs refuse a dirty worktree at EVERY arm start), readout
  `docs/compare_frame_arms.py`, results
  `docs/2026-09-03_airport_frame_ablation_results.md`. Not done: PatchTST A/B, control
  output, a KSJC cohort with enough 30R/12L flights to test the parallel pair there.
- **The KRDU run is DONE (three generations; current = 2026-07-20 B3)** — artifacts in
  `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` + `ts_pred_*` (B3 transport-consistent channels
  + physical-velocity fit; the previous generation is parked in `outputs/KRDU/_pre_b3_transport/`,
  the first is not reproducible — **quote ONLY current-artifact numbers**), tables in the package
  README.
  Robust across all generations: one-pass `full` beats chained `window` on whole-approach lateral
  error for both models (1.5–1.7× mean); PatchTST leads at short lead while iTransformer leads at
  600 s on the raw-tensor accounting (5438 vs 7384 m, n=893 — direction held in all three
  generations, margins 1.16–1.36×; channel-independence can't represent a turn's east/north
  coupling). NOT stable across retrains: gate-pass counts (0–4 of 152 — only "forecast ≠
  certifiable approach" survives) and the tail-vs-mean story (architecture-dependent).
  Hence: **treat any margin under ~1.5× as provisional** — both a split change and a ≤0.3 % data
  rescale moved effects of that size. The two lead-time accountings (record vs raw-tensor) are
  NOT comparable — the README states both with their n.
  Remaining: only KRDU trained (4 other airports harvested; the per-threshold ENU frame makes
  pooling a real design question, not a bigger `--data` glob).
- **The gate-pass conclusion needs re-deriving, not re-quoting.** The recorded "gate-pass counts
  0–4 of 152" scored a ±3 m vertical window against data offset ~33 m by the datum bug. Accuracy
  metrics (ADE/FDE, deviation vs a reference in the same frame) should be nearly unchanged; the
  gate verdicts were not measuring what they claimed.
- Follow-ups: single-aircraft only (no traffic interaction / ATC intent) and deterministic (no
  multimodality) — both are the survey's named open problems. Flyability is MEASURED but not
  FIXED (nothing projects a prediction back inside the envelope — README routes 2–4), and its
  polar is clean-configuration only, which is why it is read as a delta.

- Open questions carried over from `docs/notes_7_20.md` (2026-07-20, deleted 2026-09-07):
  why 300 steps?
  figure to show fail of patchtst
  
  cross valition 
  accuracy ???
  hard constraints
  which baseline? with or without dynamic model
  multiple different 
  evaluation them with hard constraints and dynamic
  obliation study
  
  3 weeks for baselines 
  
  Diffusion Model??? guide diffusion; classifier free guidence; tricky to optimize.
  Temp output; video generation; low dimension;
  
  moudlize repo; interfeces
  
  improve;
