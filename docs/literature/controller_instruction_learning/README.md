# Learning controller instructions from recorded traffic — reading list (2026-09-28)

Why this folder exists. These papers were gathered on **2026-09-28** for a **novelty / publication
assessment of the two-tier model**:

- an **instruction labeller** that reads controller instruction *sentences* (runway, approach
  clearance, heading, altitude, descent angle, speed) back out of ADS-B arrival tracks, by rule,
  with no voice and no recorded clearances;
- a controller **"language model" prior** — a causal Transformer that speaks those words every 2 s;
- a rule-based **physics executor** that flies the words through point-mass dynamics;
- **closed-loop RL post-training** of the prior through the executor (and, planned, a multi-aircraft
  scene prior).

The assessment itself lives in
`4dTrajectory/ts_transformer/docs/two_tier/readouts/2026-09-28_novelty_assessment.zh.md` (written by
the main session). This folder only holds the sources and what they say.

How to read it:
- Every number was read from the fetched PDF with `pdftotext`, not from the abstract. **"p."** is the
  page of the PDF file as fetched (for the UPM and DR-NTU mirrors this equals the printed page).
- **(reading)** marks my inference — a comparison with our design, a judgement of novelty overlap, an
  explanation of a discrepancy. Everything not so marked is the source's own statement or number.
- The questions asked of the priority papers (#1, #2, #4, #6, #8, #18) are answered in a fixed form:
  **(a)** how actions/instructions are obtained, **(b)** time resolution and instruction types,
  **(c)** whether any generative or closed-loop policy is trained on them.

The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`). `./download.sh` re-fetches the
**21** papers new to this repository (run 2026-09-28: all 21 returned; test: two deleted PDFs — one
repository mirror (Pham 2020, DR-NTU), one arXiv (Plan-R1) — came back **byte-identical by md5**).
**Twenty-one** further sources already live in sibling folders and are cited by path (§3) —
including two the brief listed for fetching: **#12** Hodgkin et al. (arXiv 2504.02529, in
`../control_normalization/`) and **#17** Xiang & Chen (arXiv 2409.17359, in `../prediction_horizons/`).
§4 lists what was found but not fetched (#3, #7, and the Helmke 2016 DASC paper).

Corrections to the brief (verified 2026-09-28 against the arXiv abstract pages, Crossref and
Semantic Scholar):
- **#4** arXiv 2205.09539 is by **Bastas & Vouros** (University of Piraeus), not Kravaris et al.
- **#5** arXiv 2206.07403 is by **Vouros, Papadopoulos, Bastas, Cordero & Rodrigez** ("submitted to PAIS 2022").
- **#19** first author of both Idiap papers is **Nigmatulina** (Zuluaga-Gomez is third / second author).
  2202.03725 appeared at **ICASSP 2022**; 2108.12156 says "submitted to Interspeech 2021".
- **#2** authors: **Pérez-Castán, Pérez Navarro, Serrano-Mira, Bárcena Martín, Ortega Cuevas & Pérez
  Sanz** (UPM + ENAIRE), Appl. Sci. 16(12):6200, published 2026-06-19.
- **#3** is **Zhang, Zhang, Tay & Shankar, ITSC 2022** (IEEE 25th ITSC, pp. 1100–1105,
  DOI 10.1109/ITSC55140.2022.9921823). Crossref gives the first author as "Sheng Zhang", Semantic
  Scholar as "Shenmin Zhang". Title has "Aircraft ... with ADS-B **Data**".
- **#7** is Nigam, Choi, Parikh, Li & Tran, *JAIS* 23(4):305–321 (April 2026); a conference version is
  AIAA SciTech 2025, DOI 10.2514/6.2025-1540. No arXiv version found.
- **#20** the paper that learns "which commands are possible in this radar situation" is
  **Kleinert, Helmke, Siol, Ehr, Finke, Oualil & Srinivasamurthy, 7th SESAR Innovation Days 2017**
  (MALORCA) — fetched. The 2016 DASC paper (Helmke, Ohneiser, Mühlhausen & Wies) was not found open.
- **#23** (DataDrivenMPC-TMA) shares two authors (Sheng Zhang, Yicheng Zhang) with #3 — (reading)
  probably the same Singapore group.

---

## At a glance

| piece of ours | closest prior work here | what that work does **not** do (reading) |
|---|---|---|
| **labeller** (instruction sentences from ADS-B by rule) | Pham 2020 (#1: speed/vertical/course action per flight from sector entry→exit points); Zhang 2022 (#3: holding/vectoring per trajectory, CNN on hand labels); Kleinert 2017 (#20: command-possible areas learned from radar + ASR output) | none recovers a **time-stamped instruction sequence** at seconds resolution from tracks alone; #1 gives **one** ternary action per axis per flight, #3 one event label per trajectory |
| **recorded-instruction sources** (to validate a labeller) | ENAIRE ROSETTA/TAD (#2), ENAIRE ATON (#4), NATS NERC clearances (#8/#11), MALORCA voice + radar (#20), TartanAviation audio + ADS-B (#18) | the first four are proprietary; TartanAviation is open (CC BY 4.0) but **untranscribed** and at two **GA regional** fields |
| **prior** (generative controller language model) | Bastas 2022 (#4: VAE-LSTM predicts *when* the ATCO reacts, 3 modes, 5 s); Tolstaya 2019 (#6: IRL cost + A* planner imitates SEA arrivals); LLM controllers (#13, #14) | #4 predicts reaction modes, not an instruction language, and trains no rollout policy; #6 learns a cost, not a policy over instructions; #13/#14 are prompted, not trained on traffic |
| **executor** (words → physics) | Bluebird digital twin (#8: clearances → pilot agent → BADA+PIML); BlueSky in #13/#22 | these execute clearances in a simulator, but the instructions come from agents/rules, not from a model trained on recorded instructions |
| **closed-loop RL** | Vouros 2022 (#5, DGN MARL), Diffusion-AC (#22), Carvell action-stacking (sibling), Plan-R1 (#21: pretrain on expert data → GRPO with rule rewards + KL to the pretrained reference) | aviation RL agents here learn **from scratch** in simulation, with no data-pretrained prior; the pretrain → RL recipe exists in driving (Plan-R1, SMART-R1, RLFTSim) and maritime (ShipTraj-R1), not over controller instructions |
| **multi-aircraft** | #4 neighbours via CPA features; #5 graph attention with edge features; #6 pairwise cylinder cost; Plan-R1 frozen pretrained copy as reactive world model for other agents | — |

---

## 1. Fetched papers (21)

### 1.1 Controller actions / instructions recovered from recorded traffic

**#1 — Pham, Alam & Duong, "An Air Traffic Controller Action Extraction-Prediction Model Using Machine
Learning Approach", *Complexity* 2020, Article ID 1659103 (CC BY).**
`papers/ATCActionExtraction_Pham2020_air_traffic_controller_action_extraction-prediction_model_using_ml.pdf`
- **What.** Predict the *planning (D-side)* controller's actions for a flight from its state at sector
  entry, with Random Forest and XGBoost (p.1, p.10).
- **Data.** Six months of ADS-B (Sep 2016 – Feb 2017) over **Sector 2E**, Singapore ACC, FL120–FL360,
  en route, ~5 min crossing time (p.3); points ≈ **15 s** apart (p.5); 12,141 flights crossed the sector
  in Dec 2016 alone (p.5). Holding flights (< 1.7 %) removed as outliers; tracks up-sampled to 1 s (p.5).
- **(a) How actions are obtained.** **Rule from tracks, nothing recorded.** Only the **entry and exit
  points** of each flight are used (Algorithm 1, p.9): ground-speed rate = 2(v̄ − v₀)/T, vertical speed =
  Δalt/T, Δcourse = bearing(entry→exit) − entry course (pp.6–9). Each is thresholded into {−1, 0, +1}:
  speed ±10 kt over the ~5 min crossing → ±0.017 m/s², vertical ±100 ft → ±20 ft/min, course ±3°
  (pp.9–10). 86 % of speed actions are "speed up" (p.10).
- **(b) Resolution and types.** **One action per axis per flight** (per sector crossing) — no timing.
  Three axes, three classes each; the joint "3-action" label has 27 classes (p.14). Table 9 lists real
  phraseology these would map to ("Reduce speed 250 knots…", "Turn left heading…"; p.16) but the paper
  does not produce them.
- **(c) Generative / closed-loop?** **No.** Classifiers/regressors on entry features; no rollout, no
  simulator. Results (10-fold CV, all data, Table 7 p.16): vertical ≈ 99 %, speed 79.7 % (RF) / 78.5 %
  (XGB), course 81.1 % / **86.5 %**, 3-action 67.2 % / **69.5 %**; the text rounds to 70 % (p.14).
  Regression R²: vertical 0.870, course 0.858, speed 0.677 (XGB, pp.11–12).
- **Relation to ours (reading).** The only paper found that extracts controller actions **from
  surveillance tracks alone**, as we do — but at the granularity of one net change per sector crossing,
  en route, and it trains a per-flight classifier, not a sequence model or a policy. Our labeller emits a
  2 s instruction stream with values (heading, altitude, speed, descent angle, clearances) in the
  terminal area; that granularity gap is the novelty margin on the labeller side.

**#2 — Pérez-Castán, Pérez Navarro, Serrano-Mira, Bárcena Martín, Ortega Cuevas & Pérez Sanz,
"Data-Driven Inference of ATCO Separation Intent Using Flight Plans, Radar Trajectories and Neural
Networks", *Appl. Sci.* 16(12):6200, 2026 (CC BY).**
`papers/SeparationIntent_PerezCastan2026_data-driven_inference_of_atco_separation_intent.pdf`
- **What.** Classify each already-detected ATC event as **tactical** (A1 level, A2 direct, A3 speed) or
  **separation** (B1/B2/B3 for conflict) — i.e. infer *why* an instruction was given (p.4).
- **Data.** ENAIRE, en-route sector **LECM DGU**, FL345–FL660 (p.15); 55 h selected between 15 Mar 2023
  and 3 Feb 2024, **2,077 events** (p.15; p.25 says "2700 events" — an inconsistency in the paper);
  27,076 trajectories incl. surrounding aircraft (p.17).
- **(a) How actions are obtained.** **Recorded, not inferred.** Two sources: (i) **TAD** — tasks identified
  manually by an engineer/ATCO "from synchronized radar displays and associated ATC audio" (p.3), 1.5 h of
  work per hour annotated (p.15); (ii) **ROSETTA** — ENAIRE's tool that annotates events from the
  **Controller Working Position** inputs (handovers, mouse clicks, action type) (pp.4–5). ROSETTA misses
  27 % of TAD events (p.17). The instruction itself is known; only its **intent** is learned.
- **(b) Resolution and types.** Event-level (an event time t_act), six classes. A BADA trajectory
  predictor + conflict detector adds "Situation of Interest" features (predicted crossing < 6 or 10 NM)
  (pp.4, 8).
- **(c) Generative / closed-loop?** **No.** MLP / CNN / TabNet classifiers. MLP and CNN predict every
  event as A2 (88 of 134 test events are A2, p.20). TabNet with ROSETTA's label as input: accuracy
  **0.939 vs ROSETTA 0.901** on DB1 (p.20; "+4.22 %", p.23). B1/B2 confused with A1/A2 (p.20). BADA-based
  conflict features did not help as expected (p.24).
- **Relation to ours (reading).** Confirms that ANSP-side datasets **do** contain the instructions
  (CWP records) — our labeller's task is the one these authors never face. Relevant to the prior only as
  a reminder that the *reason* for an instruction (separation vs routine) is hard even with the
  instruction in hand; our prior does not model intent explicitly.

**#4 — Bastas & Vouros, "Data-driven prediction of Air Traffic Controllers reactions to resolving
conflicts", arXiv 2205.09539 (2022).**
`papers/ATCoReactions_Bastas2022_data-driven_prediction_of_atco_reactions_to_resolving_conflicts.pdf`
- **What.** Predict **when** an ATCO issues a conflict-resolution action along a trajectory, and its
  type — the first stage of a planned Directed-InfoGAIL imitation pipeline (pp.5–7, 10–11).
- **Data.** Spanish SACTA surveillance, radar points ≈ **5 s** apart, interpolated to a 5 s grid (p.12);
  5 origin–destination pairs from 2017 (LEMG–EGKK, LEMG–EHAM, LPPT–LFPO, LSZH–LPPT, LSGG–LPPT), en route
  only: **255 trajectories / 344 resolution actions** (sector case), **668 / 791** (sector-ignorant case)
  (p.20).
- **(a) How actions are obtained.** **Recorded:** an ENAIRE **"ATCO events" dataset (ATON, Automated
  NORVASE Takes)** giving ⟨callsign, origin, destination, timestamp, resolution-action type⟩ (p.12);
  matched to the trajectory point closest in time (pp.12–13). Only the **type** is used (p.12, fn.).
  The *conflicts* that triggered them are **not recorded** and are reconstructed by perturbing course and
  speed with 20 empirical bins each (21 × 21 evolutions) and a CPA test with wide thresholds (15 NM,
  30 min) (pp.9, 14).
- **(b) Resolution and types.** 5 s. Modes C0 (no conflict, no action), C1 (conflict + action), C2
  (conflict, no action); action types A0 none, **A1 speed change, A2 direct-to-waypoint** (pp.15–16).
  Points within **250 s** before each action are relabelled C1 to fight imbalance (p.19).
- **(c) Generative / closed-loop?** **No closed loop.** A VAE whose encoder (2×64 LSTM) predicts the mode
  and whose decoder predicts categorical + continuous actions (Δcourse, Δspeeds, Δt) is trained
  **supervised** (pp.16–17); no rollout, no simulator, no GAIL stage in this paper ("future work",
  pp.11, 37). Weighted F1 ≥ **0.985 ± 0.004** on all modes (sector-ignorant, p.26); action-type F1 only
  A1 **0.588**, A2 **0.656** (p.27); sector-related A2 F1 0.384 (p.30).
- **Relation to ours (reading).** Closest in spirit to the *prior* — a learned model of controller
  behaviour on real radar with recorded actions — but it predicts a reaction **mode** at each point, not a
  sentence of instruction values, and never flies its predictions. Its conflict-reconstruction step is
  the mirror image of our labeller: they have the action and must infer the situation; we have the
  situation (the track) and must infer the action.

**#5 — Vouros, Papadopoulos, Bastas, Cordero & Rodrigez, "Automating the resolution of flight
conflicts: Deep reinforcement learning in service of air traffic controllers", arXiv 2206.07403 (2022).**
`papers/DRLConflicts_Vouros2022_automating_the_resolution_of_flight_conflicts_drl_in_service_of_atcos.pdf`
- **What.** Graph-convolutional MARL (DGN, deep Q) where each flight is an agent; conflicts are edges with
  CPA features (pp.4–6, 8–9).
- **Data / simulator.** Real scenarios from a sector near Barcelona fed by the SACTA platform, radar
  updates **every 30 s** (p.7, p.11); 42 scenarios (30–36 train, 6 test) (p.13).
- **Actions.** **32 discrete actions**: FL ±1, course ±10/±20°, speed ±3.6 m/s, direct to one of the next 4
  waypoints, no action — with durations 30/60/120/180 s (pp.4–5). Hand-written reward (exit-point
  deviation, speed changes, −10 per alert, −5 per loss) (p.10).
- **Result.** 6Seq6 resolves **90 %** of test conflicts vs 66.67 % for 5Seq6 (p.13).
- **Relation to ours (reading).** Closed-loop RL over an instruction-like action set, multi-agent — but
  learned **from scratch against a hand reward**, with no data-pretrained prior; the recorded ATCO actions
  of #4 are not used to initialise or regularise it.

**#6 — Tolstaya, Ribeiro, Kumar & Kapoor, "Inverse Optimal Planning for Air Traffic Control",
IROS 2019 (DOI 10.1109/IROS40897.2019.8968460), arXiv 1903.10525.**
`papers/InverseOptimalPlanning_Tolstaya2019_inverse_optimal_planning_for_air_traffic_control.pdf`
- **What.** Maximum-entropy IRL learns a **cost** over a 4-D grid (routing cost J_a, a per-cell look-up
  table) plus a pairwise-spacing cost J_o (learned cylinder thresholds); an ARA* planner over Dubins-
  airplane motion primitives then plans arrivals (pp.2–4).
- **Data.** Seattle-Tacoma (SEA) arrivals, **11–13 Jan 2016**, FlightAware, ≈ **30 s** between points,
  spline-interpolated (p.5). (The abstract/introduction name the FAA ASDI feed, p.1; §V names FlightAware,
  p.5 — (reading) FlightAware redistributes ASDI.)
- **(a) How actions are obtained.** **Not at all** — no instructions, recorded or inferred. The
  demonstrations are raw tracks; the "controller" is implicit in the learned cost.
- **(b) Resolution and types.** Planner step Δt = **30 s**; controls are bearing rate ∈ {−Δφ, 0, +Δφ} and
  climb ∈ {−Δz, 0, +Δz} at constant 100 m/s (pp.2, 5). Aircraft are planned **sequentially in arrival
  order**, earlier ones treated as moving obstacles (p.4); an example shows five concurrent landings
  (p.6).
- **(c) Generative / closed-loop?** A planner that *generates* trajectories under the learned cost, run
  open-loop (p.1: "feasible trajectories for open-loop control"); no RL, no instruction policy.
- **Relation to ours (reading).** The closest aviation precedent for "learn how controllers route
  arrivals from tracks, then generate new arrivals with physics" — but the learned object is a **cost map**
  and the generator is a search planner, not a model that speaks instructions. It does support the
  premise that terminal arrival structure is learnable from tracks alone.

### 1.2 Speech + surveillance: the voice side of the same instruction

**#20 — Kleinert, Helmke, Siol, Ehr, Finke, Oualil & Srinivasamurthy, "Machine Learning of Controller
Command Prediction Models from Recorded Radar Data and Controller Speech Utterances", 7th SESAR
Innovation Days, Belgrade, Nov 2017 (MALORCA).**
`papers/CommandPrediction_Kleinert2017_ml_of_controller_command_prediction_models_from_radar_data_and_speech.pdf`
- **What.** Learn the **Command Prediction Model** of Assistant-Based Speech Recognition: for each command
  type (DESCEND, REDUCE, INCREASE, CLEARED_ILS, …) and flight type, a map of **1 NM × 1 NM cells** where
  that command occurs, with the values seen there; used to predict the set of commands possible for each
  aircraft now, and to reject ASR hypotheses outside that set (pp.3–4).
- **Data.** Vienna approach Jul–Sep 2016, Prague approach Aug–Nov 2016: controller voice + radar + flight
  plans (p.5); 18.7 h of clean Vienna speech (p.5); 2,150 descend and 450 reduce commands for learning
  (p.7).
- **How commands are obtained.** From **untranscribed voice** via the project's own ASR (command
  recognition 89.0 % Prague / 60.7 % Vienna), filtered by plausibility rules, then placed at the radar
  position of the aircraft at utterance time (pp.4–5).
- **Result.** Command error rate **4.1 % → 0.9 %** (Prague) and **10.9 % → 2.0 %** (Vienna) (p.1; p.7
  gives 0.8 % for Prague); predicted-command set shrinks by ×8 / ×3 at a 25×25 NM window (pp.6–7).
- **Relation to ours (reading).** This is the prior closest to "which instruction is plausible in this
  situation", learned from real ops data — but as a **spatial look-up set**, not a sequence model, and it
  never generates or flies anything. It also shows the ANSP-side alternative to our labeller: voice + ASR
  gives command *labels* directly; we have neither voice nor ASR.

**#19a — Nigmatulina, Braun, Zuluaga-Gomez & Motlicek, "Improving callsign recognition with
air-surveillance data in air-traffic communication", arXiv 2108.12156 (submitted to Interspeech 2021).**
`papers/CallsignSurveillance_Nigmatulina2021_improving_callsign_recognition_with_air-surveillance_data.pdf`
- Boost the callsigns present in surveillance data inside the ASR grammar and/or lattice rescoring:
  **+28.4 %** absolute callsign accuracy, up to 74.2 % relative callsign WER (p.1). Test sets from ATCO2
  (LiveATC recordings) and MALORCA Prague/Vienna (p.3). Acoustic model trained on ~1,200 h (augmented
  from 195 h) + 700 h semi-supervised LiveATC (p.3).

**#19b — Nigmatulina, Zuluaga-Gomez, Prasad, Sarfjoo & Motlicek, "A two-step approach to leverage
contextual data: speech recognition in air-traffic communications", ICASSP 2022, arXiv 2202.03725.**
`papers/TwoStepContextASR_Nigmatulina2022_two-step_approach_to_leverage_contextual_data_asr_in_atc.pdf`
- ASR boosting + NER, then match the extracted callsign against surveillance: up to **53.7 %** absolute /
  60.4 % relative callsign-recognition gain (p.1). Adds a NATS London-approach test set (HAAWAII) (p.3).
- **Relation to ours, both (reading).** Surveillance data is used here to fix **who** was addressed, not
  **what** was said; they sit on the voice side and would be the route to *validating* a track-only
  labeller (align transcribed commands to our labelled words), not a competitor to it.

**#18 — Patrikar, Dantas, Moon, Hamidi, Ghosh, Keetha, Higgins, Chandak, Yoneyama & Scherer,
"TartanAviation: Image, Speech, and ADS-B Trajectory Datasets for Terminal Airspace Operations",
arXiv 2403.03372; *Scientific Data* 12:468 (2025), DOI 10.1038/s41597-025-04775-6.**
`papers/TartanAviation_Patrikar2024_image_speech_and_adsb_trajectory_datasets_for_terminal_airspace.pdf`
- **Airports.** **KAGC** Allegheny County (towered) and **KBTP** Pittsburgh-Butler (non-towered) — small
  regional / GA fields (p.2).
- **Audio.** Bearcat SR30C scanner: KAGC **tower** 121.1 MHz, KBTP **CTAF** 123.05 MHz; recordings
  **triggered by ADS-B** within 10 km (p.4). 670 days raw; after filtering (> 1 s, peak > −20 dB):
  **3,374.8 h** in 41,823 files (KAGC 2,131.9 h, KBTP 1,242.9 h), but only **477.6 h above −20 dB**
  (p.4). Raw ADS-B for the recording periods is provided alongside (p.4).
- **ADS-B.** Stratux 1090/978 MHz receiver at each field; 381 days (KBTP) + 280 days (KAGC) = 661 days;
  processed data cut to 6,000 ft MSL and 5 km, interpolated to **1 s** (p.4).
- **Transcripts?** **None.** Audio ships as WAV + a start/end time file per day (p.5); speech-to-intent is
  named only as a potential use (p.2). The *Scientific Data* version has the same audio numbers and no
  transcripts (checked 2026-09-28).
- **Licence.** Data: **CC BY 4.0** (theairlab.org/tartanaviation, checked 2026-09-28); code: BSD-3-Clause
  (github.com/castacks/TartanAviation); article: CC BY 4.0.
- **Can it validate our labeller? (reading).** Only weakly. It is the one **open** source of controller
  audio time-aligned with ADS-B in a terminal area, so it could test "does a labelled word coincide with a
  spoken instruction". But: tower/CTAF frequencies, not approach/TRACON (no radar vectors to final at
  25 km); 5 km / 6,000 ft processed cut (raw ADS-B wider); GA-dominated traffic; **no transcripts**, so
  an ATC ASR (e.g. the Idiap models of #19) and hand checks would be needed first. For our commercial
  approach arrivals at K-airports it is not a like-for-like validation set.

### 1.3 Project Bluebird (NATS / Alan Turing Institute / Exeter): a digital twin with recorded clearances

**#8 — Pepper, Keane, Hodgkin, Gould, Henderson, Lauritsen, Vlahos, De Ath, Everson, Cannon, Sierra
Castro, Korna, Carvell & Thomas, "A Probabilistic Digital Twin of UK En Route Airspace for Training and
Evaluating AI Agents for Air Traffic Control", AIAA SciTech 2026 (DOI 10.2514/6.2026-1794), arXiv
2601.03113.**
`papers/ProbDigitalTwin_Pepper2026_probabilistic_digital_twin_of_uk_en_route_airspace_for_ai_atc_agents.pdf`
- **What.** A digital twin of the **London Area Control Centre** (32 base sectors, en route) for training
  and human-in-the-loop assessment of AI ATC agents; fast-time up to ×200 (pp.1, 4, 6).
- **(a) How instructions are obtained.** **Recorded:** "controller-issued clearances … are sourced from
  NATS' NERC en route air traffic management system which provides timestamped event-level updates";
  > 20 M flights since 2016 plus a live feed (p.3). Surveillance from ARTAS (fused PSR/SSR/ADS-B/WAM).
- **(b) Resolution and types.** Surveillance ≈ **6 s** (p.3). Action space (Table 1, p.5): direct-to,
  turn by degrees, fly current heading, climb/descend now, descend at top-of-descent to reach a level at a
  fix, CAS change, Mach change, rate of climb/descent, transfer to next sector. A **pilot agent** turns a
  clearance into intent with realistic delays (p.4).
- **Executor.** Replay outside simulated sectors, simulation inside them; aircraft dynamics are **BADA
  with CAS/thrust/drag replaced by a generative model** (functional PCA + Gaussian mixtures on history)
  (pp.5–6). Mean-mode vs BADA MAE on descending B738s: −27 % CAS, −44 % ROCD (p.12); KS 0.158 and
  Wasserstein 31.8 s on time to bottom of descent (p.9). A "hybrid" mode flies **simulated trajectories
  with data-driven clearances** (p.5).
- **(c) Generative / closed-loop?** The twin is the environment; the paper trains **no** controller policy
  on the recorded clearances. Agents built on it are rules-, search-, optimisation- and RL-based (p.14);
  an open-source subset was announced for April 2026 (p.15).
- **Relation to ours (reading).** This is the strongest overlap with our **executor** and our long-run
  goal: recorded clearances + a physics-informed executor + an RL/gym interface. What it lacks is the
  **prior** — nobody in this project trains a generative model of controller clearances from the NERC
  records (they have exactly the data that would make our labeller unnecessary). It is en route; terminal
  (LTCC) is represented "in a simplified form" (p.15).

**#11 — Keane, Pepper, Burr, Hodgkin, Gould, Korna & Thomas, "A framework for assuring the accuracy and
fidelity of an AI-enabled Digital Twin of en route UK airspace", arXiv 2601.03120 (AIAA SciTech 2026).**
`papers/DTAssurance_Keane2026_framework_for_assuring_accuracy_and_fidelity_of_ai-enabled_digital_twin.pdf`
- Goal-structured assurance case for #8. Two facts that bear on instruction labels: a common
  **annotation error** is a "descend when ready" clearance recorded as "descend now", which they correct
  by **measuring the pilot's delay from the radar** and reclassifying (p.10); holding and conditional
  clearances are not supported, and "speed greater/less than" is flown as "speed equals" (p.13). Radar and
  clearances are discretised to 6 s "blips" (p.13).
- **Relation to ours (reading).** Even with recorded clearances, the NATS team reconstructs *timing* from
  the track — the same inference our labeller makes for every word.

**#9 — Kent, De Ath, Layton, Hart, Everson & Carvell, "A Future Capabilities Agent for Tactical Air
Traffic Control", arXiv 2601.04285 (AIAA SciTech 2026, DOI 10.2514/6.2026-1203).**
`papers/FutureCapabilitiesAgent_Kent2026_future_capabilities_agent_for_tactical_atc.pdf`
- "Agent Mallard": **rules-based** forward-planning agent for systemised airspace — lateral control reduced
  to lane choice on PBN routes, a depth-limited backtracking search over an **expert-elicited strategy
  library**, each candidate checked against stochastic digital-twin rollouts (p.1, pp.4–5); it "does not
  generate novel manoeuvre types beyond its programmed library" (p.17). Tested only in simplified scenarios.
- **Relation (reading).** A hand-built controller vocabulary + simulator verification; nothing learned
  from recorded instructions.

**#10 — Carvell, Thomas, Pace, Dorney, De Ath, Everson, Pepper, Keane, Tomlinson & Cannon,
"Human-in-the-Loop Testing of AI Agents for Air Traffic Control with a Regulated Assessment Framework",
arXiv 2601.04288 (AIAA SciTech 2026, DOI 10.2514/6.2026-2558).**
`papers/HITLTesting_Carvell2026_human-in-the-loop_testing_of_ai_agents_for_atc_regulated_assessment_framework.pdf`
- Maps NATS' regulator-certified trainee course into "Machine Basic Training". Inter-rater reliability on
  19 ~30-min scenarios, ≥ 7 instructors each: Spearman ρ 0.59, Kendall's W 0.64 (p.7). Agents: **Hawk**
  (rules from expert interviews) and **Falcon** (evolutionary optimisation) (p.6); Hawk v2 reached
  Satisfactory in every competency except Safety (p.10). Interaction is through the simulator API, not
  phraseology (p.4).
- **Relation (reading).** An evaluation protocol we could cite for "human-like control"; no learned
  controller from data.

### 1.4 Language-model controllers

**#13 — Andriuškevičius & Sun (TU Delft), "Automatic Control With Human-Like Reasoning: Exploring
Language Model Embodied Air Traffic Agents", arXiv 2409.09717 (2024).**
`papers/LLMAirTrafficAgents_Andriuskevicius2024_automatic_control_with_human-like_reasoning_llm_embodied_atc_agents.pdf`
- Off-the-shelf LLMs (Llama3 7B/70B, Mixtral 8x7B, Gemma2 9B, GPT-4o) with **function calls into BlueSky**
  (get aircraft, get conflicts, send command) and a vector-database "experience library" (pp.2–5).
  120 synthetic conflict scenarios with 2–4 aircraft (p.6); best (single agent, GPT-4o + library): only
  **1 of 120** not fully cleared (p.7).
- **Relation (reading).** "Controller as a language model" in the literal LLM sense — prompted, not
  trained on any traffic or instruction data; en-route conflicts, not arrival sequencing.

**#14 — Ghazanfari, Casanova, Kam, Zongo, Wei, Darrell & Bayen, "Air Traffic Control Using Large
Language Models: Prompt Engineering, Architecture, and Evaluation", arXiv 2608.19299 (2026).**
`papers/LLMATC_Ghazanfari2026_air_traffic_control_using_llms_prompt_engineering_architecture_evaluation.pdf`
- Nine LLMs play ATC to a **hand-transcribed** GA "Bay Tour" flight's pilot transmissions (P0), with an
  in-context worked example from a second first-hand recording (49 turns) (p.1, p.3). Lighter prompts win:
  ROUGE-L 0.244 (C1) vs 0.159 (most scripted C5, −35 %) (p.7).
- **Relation (reading).** Text-level dialogue imitation of phraseology; no trajectories executed, no
  training. Different problem from ours.

### 1.5 Generative trajectory models (no instruction layer)

**#15 — Petit, Torun, Brusset, Kam & Bayen, "FlowATC: Aircraft Trajectory Prediction via Flow
Matching", arXiv 2609.16528 (2026).**
`papers/FlowATC_Petit2026_aircraft_trajectory_prediction_via_flow_matching.pdf`
- Block-causal Transformer; future state tokens denoised by conditional flow matching or DDPM given the
  history (p.1). Data: 12 days (10–22 Apr 2026) of ADS-B within 60 NM of the Bay Area, 21.5 M state
  vectors → **1.15 M** windows of 86 points (43 observed / 43 predicted, ≈ 128 s each), native irregular
  sampling (pp.1, 5). CFM beats DDPM by 11–26 % minADE@20 and CVAE by 31–41 % (p.1).
- **Relation (reading).** State-space generation; no instructions, no executor, no RL — our two-tier split
  is exactly what it does not have.

**#16 — Larsen, Ruocco, Spitieris, Murad & Ragosta (SINTEF/NTNU), "Learning to Land Anywhere:
Transferable Generative Models for Aircraft Trajectories", arXiv 2511.04155 (2025).**
`papers/LandAnywhere_Larsen2025_transferable_generative_models_for_aircraft_trajectories.pdf`
- Diffusion / flow-matching / latent variants pretrained on Zürich landings and fine-tuned on Dublin
  (`traffic` library datasets from OpenSky), trajectories resampled to T = 200, conditioned on airport
  (and runway where known) (p.1, p.3). Diffusion reaches baseline level with ~20 % of Dublin data (p.1).
- **Relation (reading).** Cross-airport transfer of a whole-trajectory generator; relevant only to our
  multi-airport training question, not to instructions.

### 1.6 RL post-training of a pretrained trajectory model (non-aviation)

**#21a — Tang, Kan, Shan & Chen, "Plan-R1: Safe and Feasible Trajectory Planning as Language
Modeling", ICLR 2026, arXiv 2505.17659.**
`papers/Plan-R1_Tang2025_safe_and_feasible_trajectory_planning_as_language_modeling.pdf`
- Stage 1: autoregressive **motion-token** predictor pretrained on nuPlan expert data (1 M instances, p.7).
  Stage 2: **GRPO** with rule-based rewards (multiplicative safety indicators — collision, drivable area —
  times a weighted sum of comfort, speed-limit and progress terms, p.6), the pretrained model as fixed
  reference policy with a **KL** term (pp.2, 6). A **frozen copy of the pretrained model
  rolls out the other agents** as a reactive world model (p.2, p.5). Standard GRPO's per-group
  normalisation drowns rare safety violations (after pretraining ~80 % of groups have none, p.2) →
  **VD-GRPO** (centre, fixed scale). Reactive closed-loop score **87.69** (Val14), +4.89 over Diffusion
  Planner (p.8); GRPO adds +3.04 NR-CLS / +5.54 R-CLS over pretraining (p.8).
- **Relation (reading).** The recipe of our post-training (pretrained token prior → rule-rewarded GRPO with
  KL to the prior) already exists in driving; Plan-R1's frozen-copy-for-others trick is directly relevant
  to our multi-aircraft step, and its variance-decoupling finding to any rare-event reward (loss of
  separation).

**#21b — Zhan, Li, Zhang, Lu & Li, "ShipTraj-R1: Reinforcing Ship Trajectory Prediction in Large
Language Models via Group Relative Policy Optimization", PAKDD 2026, arXiv 2603.02939.**
`papers/ShipTraj-R1_Zhan2026_reinforcing_ship_trajectory_prediction_in_llms_via_grpo.pdf`
- Qwen3 fine-tuned with GRPO to output future AIS coordinates as text after a chain of thought about
  conflicting ships; rewards = format + accuracy (1 if mean position within **120 m**, pp.5–6). Two AIS
  datasets, 5 s interpolation, 2,649 and 2,948 trajectories (p.7).
- **Relation (reading).** GRPO on a trajectory *predictor* against a data-closeness reward — open-loop, no
  executor; shows "R1-style" has reached transport trajectories but not controller instructions.

### 1.7 Aviation closed-loop control without learned instructions

**#22 — Li, Liu, Zeng & Jiang (NUAA), "Diffusion-RL Based Air Traffic Conflict Detection and
Resolution Method", arXiv 2509.03550 (2025).**
`papers/DiffusionRL-CDR_Li2025_diffusion-rl_based_air_traffic_conflict_detection_and_resolution.pdf`
- Diffusion policy guided by a value function ("Diffusion-AC") with a density curriculum; factorised
  action space heading {−5°, 0, +5°} × speed {−50, 0, +50} kt × FL {−1, 0, +1} = 27 classes (pp.3–4,
  p.10). High density: **94.1 %** success, ~**59 %** fewer NMACs than the next baseline (p.1).
- **Relation (reading).** Multimodal policy over instruction-like actions trained **in simulation from
  scratch**; no demonstrations, no data prior.

**#23 — Zhang, Long, Huang, Zhang, Zhang & Yin, "A Data-Driven Model Predictive Control Framework for
Multi-Aircraft TMA Routing Under Travel Time Uncertainty", arXiv 2511.19452 (AI4AT @ AAAI 2026).**
`papers/DataDrivenMPC-TMA_Zhang2025_data-driven_mpc_for_multi-aircraft_tma_routing_under_travel_time_uncertainty.pdf`
- MILP routing/scheduling over a 50 NM network around Changi, rolling-horizon MPC closed with a traffic
  simulator, arrivals from Singapore ADS-B; **7×** less computation than one-shot optimisation at peak
  (p.1).
- **Relation (reading).** Terminal-area multi-aircraft closed loop, but the "controller" is an optimiser
  over a route network; data enters only as arrival times.

---

## 2. What the priority papers say, side by side

| # | (a) how instructions are obtained | (b) time resolution · instruction types | (c) generative / closed-loop policy trained on them? |
|---|---|---|---|
| 1 Pham 2020 | **rule from ADS-B tracks**: entry→exit differences, thresholded | one per flight per sector (~5 min); speed / vertical / course ∈ {−1,0,+1} | no — per-flight classifiers (RF, XGB) |
| 2 Pérez-Castán 2026 | **recorded**: manual TAD (radar replay + audio) and ENAIRE ROSETTA (CWP inputs) | event level; tactical vs separation × level / direct / speed | no — intent classifiers (TabNet best) |
| 3 Zhang 2022 (abstract only) | **hand labels on tracks** (500–1000 trajectories annotated) | per trajectory; holding / vectoring events | no — CNN classifier |
| 4 Bastas 2022 | **recorded** ENAIRE ATON events (callsign, time, type); triggering conflicts reconstructed | 5 s; modes C0/C1/C2, actions speed change / direct-to | no — supervised VAE; closed loop (Directed InfoGAIL) is future work |
| 6 Tolstaya 2019 | **none** — raw tracks as demonstrations | planner Δt 30 s; bearing-rate and climb primitives | a planner generates trajectories under an IRL cost; open loop, no policy over instructions |
| 8 Pepper 2026 | **recorded** NATS NERC clearances, timestamped | ~6 s surveillance; direct-to, turn, heading, climb/descend (incl. at-fix), CAS, Mach, ROCD, transfer | the twin *executes* clearances (pilot agent + PIML BADA) and hosts RL/rule agents, but trains no policy on the recorded clearances |
| 18 TartanAviation | **voice** (tower / CTAF audio), untranscribed, ADS-B-triggered | audio continuous; ADS-B 1 s processed | dataset only |

(reading) On this evidence, the combination *instruction sentences recovered from tracks alone* +
*a generative sequence prior over them* + *a physics executor* + *closed-loop RL through that executor*
does not appear in any single paper found. Each piece has a neighbour: track-derived actions (#1, coarse),
recorded-instruction corpora (#2, #4, #8), physics executors driven by clearances (#8), and
pretrain→GRPO in other domains (#21). The assessment document weighs this.

---

## 3. Cited by path (already in sibling folders — not re-downloaded)

| source | path |
|---|---|
| MotionLM (Seff et al. 2023) | `../manoeuvre_tokens/papers/MotionLM_Seff2023_multi-agent_motion_forecasting_as_language_modeling.pdf` |
| Trajeglish (Philion et al.) | `../manoeuvre_tokens/papers/Trajeglish_Philion2023_traffic_modeling_as_next-token_prediction.pdf` |
| SMART (Wu et al. 2024) | `../manoeuvre_tokens/papers/SMART_Wu2024_scalable_multi-agent_real-time_motion_generation_via_next-token_prediction.pdf` |
| TimeVQVAE-ATM (Murad et al. 2025) | `../manoeuvre_tokens/papers/TimeVQVAE-ATM_Murad2025_synthetic_aircraft_trajectory_generation_using_time-based_vq-vae.pdf` |
| LLM-FTP (Luo et al. 2025) | `../manoeuvre_tokens/papers/LLM-FTP_Luo2025_large_language_models_for_single-step_and_multi-step_flight_trajectory_prediction.pdf` |
| CAT-K (Zhang et al. 2025) | `../trajectory_as_language/papers/CATK_Zhang2025_closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.pdf` |
| R1Sim (Wang et al. 2026) | `../trajectory_as_language/papers/R1Sim_Wang2026_learning_rollout_from_sampling_an_r1-style_tokenized_traffic_simulation_model.pdf` |
| SMART-R1 (Pei et al.) | `../multi_agent_interaction/papers/SMART-R1_Pei2025_advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.pdf` |
| RLFTSim (Ahmadi et al. 2026) | `../multi_agent_interaction/papers/RLFTSim_Ahmadi2026_realistic_and_controllable_multi-agent_traffic_simulation_via_rl_fine-tuning.pdf` |
| RL fine-tuning for driving (Peng et al. 2024) | `../multi_agent_interaction/papers/RLFT_Peng2024_improving_agent_behaviors_with_rl_fine-tuning_for_autonomous_driving.pdf` |
| TrafficSim (Suo et al. 2021) | `../multi_agent_interaction/papers/TrafficSim_Suo2021_learning_to_simulate_realistic_multi-agent_behaviors.pdf` |
| WOSAC (Montali et al. 2023) | `../multi_agent_interaction/papers/WOSAC_Montali2023_the_waymo_open_sim_agents_challenge.pdf` |
| MALTP (Kim et al.) | `../multi_agent_interaction/papers/MALTP_Kim2025_probabilistic_multi-agent_aircraft_landing_time_prediction.pdf` |
| D2MAV-A (Brittain, Yang & Wei 2020) | `../multi_agent_interaction/papers/D2MAV-A_Brittain2020_deep_multi-agent_rl_approach_to_autonomous_separation_assurance.pdf` |
| Relative-state transformer MARL (Groot et al. 2022) | `../multi_agent_interaction/papers/RelStateTransformer_Groot2022_relative_state_transformer_models_for_marl_in_air_traffic_control.pdf` |
| Online action-stacking for ATC RL (Carvell et al. 2026) | `../multi_agent_interaction/papers/ActionStacking_Carvell2026_online_action-stacking_improves_rl_performance_for_air_traffic_control.pdf` |
| MAIFormer (Yoon & Lee 2026) | `../prediction_horizons/papers/MAIFormer_YoonLee2026_multi_agent_inverted_transformer_for_flight_trajectory_prediction.pdf` |
| **#17** Xiang & Chen 2024, arXiv 2409.17359 | `../prediction_horizons/papers/XiangChen2024_data_driven_probabilistic_trajectory_learning_high_temporal_resolution_terminal_airspace.pdf` |
| TrajAirNet (Patrikar et al.) | `../hierarchical_prediction/papers/TrajAirNet_Patrikar2021_predicting_like_a_pilot_dataset_and_method_to_predict_socially-aware_aircraft_trajectories.pdf` |
| FlightBERT + spoken instructions (Guo et al. 2023) | `../hierarchical_prediction/papers/SpokenInstructions_Guo2023_flightbert_with_spoken_instructions_for_flight_trajectory_prediction.pdf` |
| **#12** Hodgkin, Pepper & Thomas 2025, arXiv 2504.02529 | `../control_normalization/papers/DescentPIML_Hodgkin2025_probabilistic_simulation_of_aircraft_descent_physics-informed_ML.pdf` |

(`../procedure_hard_constraints/papers/` also holds duplicate copies of TrajAirNet, Guo spoken
instructions, Xiang & Chen and MAIFormer under short names.) #12 is the descent half of the
physics-informed generator inside the Bluebird twin (#8, p.6); #17 is the conditional generative
terminal-airspace model already discussed in `../prediction_horizons/README.md`.

---

## 4. Found but not fetched

| # | source | why not | what the abstract says |
|---|---|---|---|
| 3 | Zhang, Zhang, Tay & Shankar, "Learning-based Aircraft Trajectory Analysis Tool for Holding and Vectoring Identification with ADS-B Data", IEEE ITSC 2022, pp. 1100–1105, DOI 10.1109/ITSC55140.2022.9921823 | IEEE paywall; no open copy (ResearchGate 403, Unpaywall none) | Small CNN; trajectory and SID/STAR drawn as separate channels of a 2-D array; identifies **holding and vectoring events**; > 95 % train / > 90 % test accuracy; needs **500–1,000 hand-labelled trajectories** ("hours by a single annotator"); > 100 trajectories/s on one laptop CPU; meant as a TMA-efficiency / ATCO-workload analyser. → (a) human labels on tracks, (b) event per trajectory (holding / vectoring), not timed instructions, (c) no generative or closed-loop model. (Abstract via Semantic Scholar record of the DOI.) |
| 7 | Nigam, Choi, Parikh, Li & Tran, "Survey of Inverse Reinforcement Learning in Aviation and Future Outlooks", *J. Aerospace Information Systems* 23(4):305–321, 2026, DOI 10.2514/1.I011635; conference version AIAA SciTech 2025, DOI 10.2514/6.2025-1540 | AIAA paywall; no arXiv or repository copy found | IRL learns a reward from expert demonstrations; the authors "identify a significant, quantifiable gap in applications of IRL in aviation relative to other domains", review foundational methods and aviation applications, discuss causes of the gap and future uses (abstract of the SciTech version, Illinois Experts page). |
| — | Helmke, Ohneiser, Mühlhausen & Wies, "Reducing Controller Workload with Automatic Speech Recognition", IEEE/AIAA DASC 2016 | IEEE; no open copy found; the command-prediction content the brief asked for is covered by Kleinert 2017 (§1.2), which cites it | ABSR with an arrival-manager-driven hypotheses generator in approach control (as summarised in Kleinert 2017, pp.1–2). |
