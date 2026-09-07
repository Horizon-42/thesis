# Hard procedure constraints in a learned trajectory predictor — reading list (2026-09-08)

Why this folder exists: the question was how to put the final-approach procedure constraint
(the LPV lateral corridor + glidepath window, active once a flight is established on final)
into the TRAINING of `4dTrajectory/ts_transformer`'s two output paths — `state` (positions per
row) and `control` (bounded controls flown through a differentiable point-mass rollout) — as a
HARD constraint, after a soft penalty was vetoed on both paths and a CBF safety filter that
works at predict time made the network lazy when trained through. The assessment that reads
these papers against that evidence is
`4dTrajectory/ts_transformer/docs/2026-09-08_hard_constraints_survey_and_integration_plan.md`
(§3 survey with formulas, §4 plan).

The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`); `./download.sh` re-fetches
every one (it runs the three per-cluster scripts, which skip files that already exist). Three
papers have no open copy and are quoted from abstracts only, marked `[UNVERIFIED]` in the notes:
Shi, Xu & Pan 2021 (T-ITS, DOI 10.1109/TITS.2020.3004807), Pang et al. 2021 (TR-C, DOI
10.1016/j.trc.2021.103326), Gao et al. ICRA 2018 (substituted by Zhou et al. RA-L 2019).

## Layout

| file | what |
|---|---|
| `notes_A.md` | cluster A — hard constraints by construction and differentiable projection layers, trajectory-side feasibility (Frenét, Bézier hull, kinematic decoders), two aviation papers. 25 entries, formulas transcribed from the PDFs, cross-cutting read-out at the end |
| `notes_B.md` | cluster B — safety layers, CBF / discrete CBF / BarrierNet, provably safe RL, predictive safety filters, safe imitation, residual policies, learned certificates; §7 synthesis on the lazy-policy problem. 23 entries |
| `notes_C.md` | cluster C — constrained learning (multiplier dynamics, feasible levels), ten aviation TP papers, predict-then-optimize / amortized optimization, constrained diffusion, discrete-mode + regression training. 35 entries, cross-cutting conclusions and the searches that came up empty |
| `download.sh`, `download_{A,B,C}.sh` | re-fetch scripts |
| `papers/` | 80 PDFs |

Every entry in the notes gives: citation + arXiv id / DOI + local filename, the mechanism in a
few sentences, the CORE FORMULAS in plain-text math as the paper defines them, the guarantee
(hard / soft / statistical) and cost, and what it implies for our two paths and the gate.

## The five findings that matter (details in the survey doc)

1. **By-construction is closed form and known for our state path.** HardNet-Aff's parallel
   projection collapses to an elementwise clamp in runway axes (our `corridor-bounded` tanh
   is its smooth cousin; HardNet's Prop. 7 says the clamp leaves in-corridor rows untouched,
   the tanh does not). Gauge map / RAYEN / Πnet handle coupled convex sets; a saturating
   filter is many-to-one and a gauge map is a bijection — a mechanistic account of the lazy
   network, and its structural cure on the control path (`notes_A.md` §1–6, read-out).
2. **The lazy policy is known, measured and curable.** Pizarro Bejarano 2025 measures it
   ("return when uncertified") and removes it with a correction penalty
   `α‖u_uncert − u_cert‖²`, monotone in α; OptLayer's `CPC` (update on the raw action with a
   violation-penalised reward, then on the projected action with the true reward) beats the
   `CC` strategy our six arms used; Krasowski 2023's projection + adaption penalty gives the
   lowest intervention rate; Oh & Fisac 2026 prove no performance penalty for a
   least-restrictive filter used at train AND test, so an accuracy loss reads as
   over-restriction (sweep α, k) or infeasible gate entry; Geiger & Straehle 2022 prove that in
   an imitation setting test-time-only filtering has a quadratic-in-horizon error against
   linear for train-and-test (`notes_B.md` §7).
3. **The diverged primal-dual run was correct optimisation of an unbounded dual.** The dual
   best response to a violated constraint is `+∞` (Gallego-Posada 2022 App. G, Chamon 2023);
   the level was unreachable. Fixes: anneal the level, learn it (Hounie 2023 resilient
   constrained learning — the converged relaxation `u*` IS the reachable violation rate), or
   drop it (solver stage); with a fixed level use νPI (Sohrabi 2024, in Cooper) or PID
   (Stooke 2020), never plain ascent (`notes_C.md` §1, conclusions 1–3).
4. **The gate should be predicted, not thresholded.** TNT / DenseTNT / annealed WTA give the
   recipe (candidates from the published geometry, classification against the closest to
   truth, teacher forcing, annealed assignment); the map-adaptive goal-based predictor builds
   its mode targets from exactly our cross-track test; the aviation default is a hard phase
   indicator silently disabled when inputs are missing (`notes_A.md` §21–22, 25; `notes_C.md` §5).
5. **The aviation gap is verified.** Across twelve aviation trajectory-prediction papers,
   none enforces a lateral corridor or a glidepath window and none predicts the establishment
   point; physics-as-generator work (BADA, neural ODE) is en-route / vertical only and
   excludes the altitude band where procedures bite; the only published-regulation hard
   constraint found is a scalar climb-rate rejection (`notes_C.md` §2, conclusion 0).

## Papers

Keys in brackets are the `[key]` citations used by the survey doc. `A.n` / `B.n` / `C.n` is the
entry number in the notes file.

### A — construction, projection layers, trajectory-side feasibility

| # | paper | file | key |
|---|---|---|---|
| A.1 | Tordesillas, Klemm, How, Hutter — RAYEN: Imposition of Hard Convex Constraints on Neural Networks (arXiv 2307.08336) | `rayen_2307.08336.pdf` | RAYEN |
| A.2 | Min & Azizan — HardNet: Hard-Constrained Neural Networks with Universal Approximation Guarantees (arXiv 2410.10807) | `hardnet_2410.10807.pdf` | HardNet |
| A.3 | Terpin et al. — Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers (ICLR 2026, arXiv 2508.10480) | `pinet_2508.10480.pdf` | Pinet |
| A.4 | Tabas & Zhang — Computationally Efficient Safe RL for Power Systems (gauge map, ACC 2022, arXiv 2110.10333) | `tabas_zhang_gaugemap_2110.10333.pdf` | GaugeMap |
| A.5 | Tabas & Zhang — Safe and Efficient MPC Using Neural Networks: An Interior Point Approach (arXiv 2203.12196) | `tabas_zhang_mpc_interiorpoint_2203.12196.pdf` | |
| A.6 | Liang, Chen, Low — Homeomorphic Projection (ICML 2023 / JMLR 2024) | `homeomorphic_projection_jmlr25.pdf` | |
| A.7 | Donti, Rolnick, Kolter — DC3 (ICLR 2021, arXiv 2104.12225) | `dc3_2104.12225.pdf` | DC3 |
| A.8 | Amos & Kolter — OptNet (ICML 2017, arXiv 1703.00443) | `optnet_1703.00443.pdf` | OptNet |
| A.9 | Agrawal et al. — Differentiable Convex Optimization Layers (NeurIPS 2019, arXiv 1910.12430) | `cvxpylayers_1910.12430.pdf` | cvxpylayers |
| A.10 | Chen et al. — KKT-hPINN (arXiv 2402.07251) | `kkt_hpinn_2402.07251.pdf` | KKT-hPINN |
| A.11 | Beucler et al. — Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems (PRL 2021, arXiv 1909.00912) | `beucler_analytic_constraints_1909.00912.pdf` | Beucler2021 |
| A.12 | Márquez-Neila, Salzmann, Fua — Imposing Hard Constraints on Deep Networks: Promises and Limitations (arXiv 1706.02025) | `marquezneila_hard_constraints_1706.02025.pdf` | |
| A.13 | Balestriero & LeCun — POLICE (arXiv 2211.01340) | `police_2211.01340.pdf` | |
| A.14 | Werling, Ziegler, Kammel, Thrun — Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenét Frame (ICRA 2010) | `werling2010_frenet_icra.pdf` | Werling2010 |
| A.15 | Zhou, Gao, Wang, Liu, Shen — Robust and Efficient Quadrotor Trajectory Generation (B-spline convex hull, RA-L 2019, arXiv 1907.01531) | `fastplanner_bspline_1907.01531.pdf` | Bernstein-corridor |
| A.16 | Zhou, Wang, Ye, Xu, Gao — EGO-Planner (RA-L 2021, arXiv 2008.08835) | `egoplanner_2008.08835.pdf` | EGO-Planner |
| A.17 | Cui et al. — Deep Kinematic Models for Kinematically Feasible Vehicle Trajectory Predictions (ICRA 2020, arXiv 1908.00219) | `cui_deep_kinematic_1908.00219.pdf` | |
| A.18 | Salzmann et al. — Trajectron++ (ECCV 2020, arXiv 2001.03093) | `trajectronpp_2001.03093.pdf` | |
| A.19 | Song et al. — PRIME (CoRL 2021, arXiv 2103.04027) | `prime_2103.04027.pdf` | PRIME |
| A.20 | Phan-Minh et al. — CoverNet (CVPR 2020, arXiv 1911.10298) | `covernet_1911.10298.pdf` | |
| A.21 | Zhang et al. — Map-Adaptive Goal-Based Trajectory Prediction (CoRL 2020, arXiv 2009.04450) | `map_adaptive_goal_2009.04450.pdf` | MapAdaptiveGoal |
| A.22 | Afshar et al. — PBP: Path-based Trajectory Prediction (ICRA 2024, arXiv 2309.03750) | `pbp_pathbased_2309.03750.pdf` | PBP |
| A.23 | Ye, Zhou, Wang — Frenét-Based Domain Normalization for Trajectory Prediction (arXiv 2305.17965) | `frenet_domain_norm_2305.17965.pdf` | |
| A.24 | Shi, Xu, Pan — 4-D Flight Trajectory Prediction With Constrained LSTM Network (T-ITS 2021) | not downloadable (paywall, no preprint) | Shi2021 |
| A.25 | Zhang & Chen — Phased Flight Trajectory Prediction with Deep Learning (ACM TIST, arXiv 2203.09033) | `phased_flight_tp_2203.09033.pdf` | ZhangChen2022 |

### B — safety filters, CBFs, the lazy-policy problem

| # | paper | file | key |
|---|---|---|---|
| B.1.1 | Dalal et al. — Safe Exploration in Continuous Action Spaces (arXiv 1801.08757) | `dalal2018_safe_exploration_1801.08757.pdf` | Dalal2018 |
| B.1.2 | Pham, De Magistris, Tachibana — OptLayer (ICRA 2018, arXiv 1709.07643) | `pham2018_optlayer_1709.07643.pdf` | OptLayer |
| B.1.3 | Cheng, Orosz, Murray, Burdick — End-to-End Safe RL through Barrier Functions (AAAI 2019, arXiv 1903.08792) | `cheng2019_e2e_safe_rl_cbf_1903.08792.pdf` | Cheng2019 |
| B.2.1 | Krasowski et al. — Provably Safe RL: Conceptual Analysis, Survey, and Benchmarking (TMLR 2023, arXiv 2205.06750) | `krasowski2023_provably_safe_rl_2205.06750.pdf` | Krasowski2023 |
| B.2.2 | Alshiekh et al. — Safe RL via Shielding (AAAI 2018, arXiv 1708.08611) | `alshiekh2018_shielding_1708.08611.pdf` | |
| B.2.3 | Hsu, Hu, Fisac — The Safety Filter: A Unified View (arXiv 2309.05837) | `hsu2023_safety_filter_unified_2309.05837.pdf` | Hsu2023 |
| B.2.4 | Oh, Nguyen, Hu, Fisac — Provably Optimal RL under Safety Filtering (IASEAI 2026, arXiv 2510.18082) | `provably_optimal_rl_safety_filtering_2510.18082.pdf` | OhFisac2026 |
| B.2.5 | Brunke et al. — Safe Learning in Robotics (Annual Review 2022, arXiv 2108.06266) | `brunke2022_safe_learning_robotics_2108.06266.pdf` | |
| B.3.1 | Wabersich & Zeilinger — A predictive safety filter for learning-based control (Automatica 2021, arXiv 1812.05506) | `wabersich2021_predictive_safety_filter_1812.05506.pdf` | PredictiveSafetyFilter |
| B.3.2 | Tearle, Wabersich, Carron, Zeilinger — A predictive safety filter for learning-based racing control (arXiv 2102.11907) | `tearle2021_psf_racing_2102.11907.pdf` | |
| B.3.3 | Pizarro Bejarano, Brunke, Schoellig — Safety Filtering While Training (RA-L 2025, arXiv 2410.11671) | `pizarrobejarano2025_filtering_while_training_2410.11671.pdf` | PizarroBejarano2025 |
| B.3.4 | Yang, Werner, de Sa, Ames — CBF-RL (arXiv 2510.14959) | `cbfrl_2510.14959.pdf` | |
| B.4.1 | Xiao et al. — BarrierNet (IEEE T-RO 2023, arXiv 2111.11277) | `xiao2023_barriernet_2111.11277.pdf` | BarrierNet |
| B.4.2 | Xiao, Wang, Gan, Rus — SafeDiffuser (arXiv 2306.00148) | `xiao2023_safediffuser_2306.00148.pdf` | SafeDiffuser |
| B.4.3 | Agrawal & Sreenath — Discrete Control Barrier Functions (RSS 2017) | `agrawal2017_discrete_cbf_rss13_p73.pdf` | DCBF |
| B.4.4 | Zeng, Zhang, Sreenath — Safety-Critical MPC with Discrete-Time CBF (ACC 2021, arXiv 2007.11718) | `zeng2021_mpc_dcbf_2007.11718.pdf` | DCBF |
| B.5.1 | Cosner, Yue, Ames — End-to-End Imitation Learning with Safety Guarantees using CBFs (arXiv 2212.11365) | `cosner2022_e2e_il_cbf_2212.11365.pdf` | |
| B.5.2 | Geiger & Straehle — Fail-Safe Adversarial Generative Imitation Learning (TMLR 2022, arXiv 2203.01696) | `failsafe_agil_2203.01696.pdf` | GeigerStraehle2022 |
| B.5.3 | Yin, Seiler, Jin, Arcak — Imitation Learning with Stability and Safety Guarantees (arXiv 2012.09293) | `yin2021_il_stability_safety_2012.09293.pdf` | |
| B.5.4 | Silver, Allen, Tenenbaum, Kaelbling — Residual Policy Learning (arXiv 1812.06298) | `silver2018_residual_policy_learning_1812.06298.pdf` | |
| B.5.5 | Johannink et al. — Residual RL for Robot Control (ICRA 2019, arXiv 1812.03201) | `johannink2019_residual_rl_1812.03201.pdf` | |
| B.6.1 | Dawson, Gao, Fan — Safe Control with Learned Certificates (T-RO 2023, arXiv 2202.11762) | `dawson2023_learned_certificates_2202.11762.pdf` | |
| B.6.2 | Fisac et al. — A General Safety Framework for Learning-Based Control (TAC 2019, arXiv 1705.01292) | `fisac2019_general_safety_framework_1705.01292.pdf` | |

Also cited in the survey from cluster B's notes: Ames et al. 2019 (CBF theory, [CBF-theory])
and Nguyen & Sreenath 2016 (exponential CBF, [ECBF]) are referenced through BarrierNet /
Zeng et al. and were not downloaded separately.

### C — constrained learning, aviation, predict-then-optimize, constrained generation, gating

| # | paper | file | key |
|---|---|---|---|
| C.1.1 | Chamon & Ribeiro — Probably Approximately Correct Constrained Learning (NeurIPS 2020) | `chamon2020_pac_constrained_learning.pdf` | Chamon2020 |
| C.1.2 | Chamon, Paternain, Calvo-Fullana, Ribeiro — Constrained Learning with Non-Convex Losses (T-IT 2023) | `chamon2023_constrained_learning_nonconvex_losses.pdf` | Chamon2023 |
| C.1.3 | Hounie, Ribeiro, Chamon — Resilient Constrained Learning (NeurIPS 2023, arXiv 2306.02426) | `hounie2023_resilient_constrained_learning.pdf` | Hounie2023 |
| C.1.4 | Fioretto et al. — Lagrangian Duality for Constrained Deep Learning (ECML-PKDD 2020) | `fioretto2020_lagrangian_duality_constrained_dl.pdf` | Fioretto2020 |
| C.1.5 | Stooke, Achiam, Abbeel — Responsive Safety in RL by PID Lagrangian Methods (ICML 2020, arXiv 2007.03964) | `stooke2020_pid_lagrangian.pdf` | Stooke2020 |
| C.1.6 | Sohrabi et al. — On PI Controllers for Updating Lagrange Multipliers (νPI, ICML 2024, arXiv 2406.04558) | `sohrabi2024_nupi_multiplier_controller.pdf` | nuPI |
| C.1.7 | Kervadec et al. — Constrained Deep Networks: Lagrangian Optimization via Log-Barrier Extensions (ICPR 2022) | `kervadec2022_log_barrier_extensions.pdf` | Kervadec2022 |
| C.1.8 | Achiam, Held, Tamar, Abbeel — Constrained Policy Optimization (ICML 2017) | `achiam2017_cpo.pdf` | |
| C.1.9 | Gallego-Posada et al. — Controlled Sparsity via Constrained Optimization (NeurIPS 2022, arXiv 2208.04425) | `gallegoposada2022_controlled_sparsity.pdf` | GallegoPosada2022 |
| C.1.10 | Gallego-Posada et al. — Cooper: A Library for Constrained Optimization in Deep Learning (2025) | `gallegoposada2025_cooper_library.pdf` | |
| C.1.11 | Sangalli et al. — Constrained Optimization to Train Neural Networks on Critical and Under-Represented Classes (NeurIPS 2021) | `sangalli2021_constrained_opt_underrepresented.pdf` | |
| C.2.1 | Shi, Xu, Pan 2021 (see A.24) | not downloadable | Shi2021 |
| C.2.2 | Pang, Zhao, Yan, Liu — Data-driven trajectory prediction with weather uncertainties (TR-C 2021) | not downloadable (paywall, no preprint) | |
| C.2.3 | Hodgkin, Pepper, Thomas — Conditioning Aircraft Trajectory Prediction on Meteorological Data with a Physics-Informed ML Approach (2026) | `hodgkin2026_physics_informed_met_tp.pdf` | Hodgkin2026 |
| C.2.4 | Pepper & Thomas — Learning Generative Models for Climbing Aircraft from Radar Data (2023) | `pepper2023_generative_climbing_bada.pdf` | PepperThomas2023 |
| C.2.5 | Jarry, Dalmau, Very, Olive — A Neural ODE Approach to Aircraft Flight Dynamics Modelling (2025) | `jarry2025_node_fdm.pdf` | NODE-FDM |
| C.2.6 | Prutsch, Schinagl, Possegger — ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace (2026) | `prutsch2026_ascent_terminal_tp.pdf` | ASCENT |
| C.2.7 | Patrikar, Moon, Oh, Scherer — Predicting Like A Pilot (TrajAirNet, 2022) | `patrikar2022_predicting_like_a_pilot.pdf` | Patrikar2022 |
| C.2.8 | Guo et al. — Integrating spoken instructions into flight trajectory prediction (SIA-FTP, arXiv 2305.01661) | `guo2023_spoken_instructions_tp.pdf` | Guo2023 |
| C.2.9 | Yoon & Lee — Multi-Agent Inverted Transformer for Flight Trajectory Prediction (T-ITS 2025) | `yoon2025_multiagent_itransformer_tp.pdf` | MAIFormer |
| C.2.10 | Xiang & Chen — Data-driven Probabilistic Trajectory Learning with High Temporal Resolution in Terminal Airspace (arXiv 2409.17359) | `xiang2024_probabilistic_tp_terminal.pdf` | XiangChen2024 |
| C.3.1 | Amos — Tutorial on Amortized Optimization (FnT ML 2023) | `amos2023_amortized_optimization_tutorial.pdf` | Amos2023 |
| C.3.2 | Sambharya, Hall, Amos, Stellato — End-to-End Learning to Warm-Start for Real-Time Quadratic Optimization (L4DC 2023) | `sambharya2023_learn_warmstart_qp.pdf` | Sambharya2023 |
| C.3.3 | Amos et al. — Differentiable MPC for End-to-end Planning and Control (NeurIPS 2018) | `amos2018_differentiable_mpc.pdf` | DiffMPC |
| C.3.4 | Elmachtoub & Grigas — Smart "Predict, then Optimize" (Management Science 2022) | `elmachtoub2022_spo.pdf` | SPO |
| C.3.5 | Guffanti, Gammelli, D'Amico, Pavone — Transformers for Trajectory Optimization with Application to Spacecraft Rendezvous (2024) | `guffanti2024_art_transformer_trajopt_warmstart.pdf` | Guffanti2024 |
| C.3.6 | Briden et al. — Constraint-Informed Learning for Warm Starting Trajectory Optimization (2023) | `briden2023_constraint_informed_warmstart.pdf` | Briden2023 |
| C.4.1 | Janner, Du, Tenenbaum, Levine — Diffuser (ICML 2022) | `janner2022_diffuser.pdf` | Diffuser |
| C.4.2 | Jiang et al. — MotionDiffuser (CVPR 2023) | `jiang2023_motiondiffuser.pdf` | MotionDiffuser |
| C.4.3 | Christopher, Baek, Fioretto — Constrained Synthesis with Projected Diffusion Models (NeurIPS 2024, arXiv 2402.03559) | `christopher2024_projected_diffusion.pdf` | ProjectedDiffusion |
| C.4.4 | Li, Ding, Dieng, Beeson — DiffuSolve (2024) | `li2024_diffusolve.pdf` | |
| C.4.5 | Xiao, Wang, Gan, Rus — SafeDiffuser (duplicate of B.4.2, cluster C copy) | `xiao2023_safediffuser.pdf` | SafeDiffuser |
| C.5.1 | Zhao et al. — TNT: Target-driveN Trajectory Prediction (CoRL 2020) | `zhao2020_tnt.pdf` | TNT |
| C.5.2 | Gu, Sun, Zhao — DenseTNT (ICCV 2021) | `gu2021_densetnt.pdf` | DenseTNT |
| C.5.3 | Xu, Letzelter, Chen, Zablocki, Cord — Annealed Winner-Takes-All for Motion Forecasting (ICRA 2025, arXiv 2409.11172) | `xu2025_annealed_wta.pdf` | AnnealedWTA |

## Searches that came up empty (so they are not repeated)

- No aviation trajectory-prediction paper that imposes the published final-approach corridor
  as a constraint, hard or soft (arXiv full text: "trajectory prediction" AND "final approach"
  / "approach procedure" → zero); none that predicts the establishment point as a latent.
- No "Sohrabi et al. 2024 constrained learning survey" exists; that reference is the νPI paper.
- No paper that trains a trajectory-PREDICTION network through a safety filter, or reports a
  filter-induced regression on a held-out subpopulation (our vectored flights).
- Not verified (titles only, surfaced by search): Shield-Loco (arXiv 2606.07193), "End-to-End
  Learning of Safe Optimal Feedback Control in High Dimensions with CBF Layers" (arXiv
  2607.20674), a modular RL + predictive-safety-filter marine navigation paper (arXiv
  2312.01855), and the filter-aware works Hsu §3.5.3 cites (Akametalu; Leung et al.; Hu et al.
  SHARP).
