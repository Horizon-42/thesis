# Control normalisation across platforms — reading list for ts_transformer (2026-09-14)

Why this folder exists: the `control` path of `4dTrajectory/ts_transformer` predicts N piecewise-constant
controls `(δ = T/T_max,static, φ, n = L/W)` that a differentiable RK4 point-mass rollout integrates
(`V̇ = (T − D)/m − g·sinγ`, `ψ̇ = g·n·sinφ/(V·cosγ)`, `γ̇ = g·(n·cosφ − cosγ)/V`, `D = ½ρV²S·(CD0 + k·CL²)`,
`CL = n·m·g/(½ρV²S)`). The fleet runs from a 7 t business jet (C550) to 251 t widebodies, mass is a per-type
constant (landing mass, not observed), and the same δ gives a type-dependent acceleration
(`∂V̇/∂δ = T_max/m`, measured 3.0–4.1 m s⁻² per unit δ over the cohort's 28 types, §3.2; drag per weight
depends on the CLmax bucket). The network sees the raw physical parameters
(mass, T_max, S, CLmax, CD0, k …) as a small conditioning vector. The question: should the control output be
re-parameterised (mass-free, weight-normalised or acceleration-like, then mapped to each aircraft's physical
controls), and what do published implementations do, in any field? Collected 2026-09-14; §3 is the answer,
with the measurements (`measurements/`) it rests on.

The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`); `./download.sh` re-fetches all 31
(tested 2026-09-14: four deleted files came back byte-identical). §4 lists what could not be fetched.
Conventions: formulas in backticks are quoted as the source writes them (symbols transliterated to Unicode);
"checked at" says where the claim was read; **(reading)** marks my inference, not the source's claim.

## 1. Sources

### 1.1 Aviation / ATM

| file / id | what it is | output space · normalisation | bounds per platform | measured evidence | checked at |
|---|---|---|---|---|---|
| `BADA-overview_Nuic2010…` — Nuic, Poles, Mouillet, *Int. J. Adapt. Control Signal Process.* 24(10):850–866, 2010, DOI 10.1002/acs.1176 | EUROCONTROL overview of BADA 3 and the (then new) BADA 4 | Total Energy Model `(T − D)v = W ḣ + m v v̇` (Eq. 1) → `ḣ = (T − D)/(mg) · v · ESF` (Eq. 2), `ESF = [1 + (v/g)(dv/dh)]⁻¹` (Eq. 3). BADA 4: `T/δ = W_MTOW · C_T` (Eq. 17), `C_T = f(M, δ_T)` with throttle parameter δ_T (Eq. 19), `D = ½·C_D·δ·p0·κ·M²·S` (Eq. 12), δ = p/p0. "The physical relationships in BADA 4 are obtained in terms of dimensionless variables … allows discovering physical similarity relationships" (§4) | one coefficient set per ICAO type (§3); ratings MTKOF/MCMB/MCRZ/IDLE are separate C_T laws (§5.1.2) | not a learning paper. BADA 3: mean RMS vertical-speed error < 100 fpm over the normal envelope; BADA 4: < 70 fpm over the complete envelope, 25 aircraft (§5.3) | PDF §2.1, §4, §5.1–5.3 |
| (web) pyBADA, EUROCONTROL's open implementation, `src/pyBADA/bada4.py` @ `a03b3ca` (2026-07-09) | current BADA 4 formulas (the BADA 4 manual is licence-only) | `CL = 2·mass·g·nz / (delta·p_0·Agamma·M²·S)` (`CL`, l.751–774 — load factor inside CL); `Thrust = delta · WREF · CT`, `WREF = MREF · g` (l.138–139, l.1784–1813); `D = 0.5·delta·p_0·Agamma·M·M·S·CD` (l.1762–1782); `CW = mass·g / (MTOW·delta·g)` (l.3419–3432); `ROCD = ((temp − deltaTemp)/temp)·(T − D)·v·ESF/(mass·g)` (l.1873–1905) | per type: MREF (XML `PFM`, l.138), MTOW (XML `DLM`, l.390) | — | source read 2026-09-14 |
| `BADA3-UM_EUROCONTROL2009…` — BADA 3.7 User Manual, EEC Tech./Sci. Report 2009-003 | the BADA 3 specification | `(T − D)·V_TAS = m·g·dh/dt + m·V_TAS·dV_TAS/dt` (3.1-1); `dh/dt = [(T − D)·V_TAS/(m·g)]·f{M}` (3.1-4); "any two of the three variables of thrust, speed, or ROCD" are controlled, the third follows (§3.1, p.7–8); `CL = 2·m·g/(ρ·V_TAS²·S·cosφ)` (3.6-1, p.18); descent thrust as a ratio of the type's max climb thrust, `T_des,app = C_Tdes,app × T_max climb` (3.7-11, p.21) | **type-specific** in newtons: `(T_max climb)_ISA` (3.7-1, p.20). **Fleet-wide, mass-free**: `a_l,max(civ)` = 2.0 ft/s², `a_n,max(civ)` = 5.0 ft/s² (§5.2, p.31); bank φ_nom,civ 15° (TO/LD) / 35°, φ_max,civ 25° (TO/LD) / 35° (HOLD) / 45° (§5.3, p.32) | — | PDF p.7–8, 18, 20–21, 31–32 |
| `MassThrustLearning_Alligier2013…` — Alligier, Gianazza, Durand, *TR-C* 36:45–60, 2013, DOI 10.1016/j.trc.2013.08.006 (OATAO author copy) | learn a thrust-setting law from history + estimate an "equivalent mass" on past points, BADA 3.9 point mass | `Thr = c·Thr_maxclimb` (Eq. 12); `Power = (c·Thr_maxclimb − D)V_a` (Eq. 13); `c = f(x\|θ) = Σθ_i (H_p)^i` (Eq. 14). "The adjusted parameters are not meant to be exact, however they are designed so as to improve the energy rate prediction" (abstract); §3.1 is titled "Equivalent mass at a given point" | c is relative to the type's BADA max climb thrust; one law per category = "all aircraft of a same type departing from a same airport" (§2.5) | altitude RMSE at 10 min: −≈40 % vs BADA max climb + reference mass, −≈50 % vs BADA reduced-power climb (Table 3b, §7.3); single type, proof of concept (§8) | Eqs. 12–14, §2.5, §3.1, §7.3, §8 |
| `ClimbMLMass_Alligier2015…` — Alligier, Gianazza, Durand, *IEEE T-ITS* 16(6):3138–3149, 2015, DOI 10.1109/TITS.2015.2437452 (OATAO copy) | ML (GBM etc.) predicts the mass as a response variable, BADA physics for the rollout | output = scalar mass m̂ (least-squares "mass to be predicted", assuming BADA max-climb thrust). **"there is an infinity of couples (mass, thrust_profile) that give exactly the same trajectory. Intuitively, a heavy aircraft with maximum climb thrust is equivalent to a lighter aircraft with reduced thrust"** (§IV-D, p.3142) | one model per type ("Tuning one model per aircraft is no more an issue than for the standard BADA model", p.3141); 9 types | altitude RMSE −≥58 % vs BADA baseline, −≥27 % vs the two mass-estimation methods (speed profile known); −≥29 % / −≥17 % with BADA speed profile, except E145, F100 (abstract; Tables VIII–IX) | §IV-D, abstract |
| Alligier & Gianazza, *TR-C* 96:72–95, 2018, DOI 10.1016/j.trc.2018.08.012 — **not fetched** | large-scale version: OpenSky ADS-B climbs of 2017, 11 most frequent types | ML predicts the unknown point-mass parameters "mass and the speed intent" | per type | altitude RMSE −48 % on a 10 min horizon, speed −25 %; with pre-take-off information only, altitude −25 % (abstract) | abstract only (Semantic Scholar record); HAL copy behind a bot check |
| `MassBayes_Sun2018…` — Sun, Ellerbroek, Hoekstra, *TR-C* 90:59–73, 2018, DOI 10.1016/j.trc.2018.02.022 (TU Delft final version) | per-phase mass observations from ADS-B fused by Bayesian inference | thrust `η·T_max`, "η is an assumed thrust coefficient … describing the actual thrust setting as a percentage of maximum thrust" (§2.3, Eq. 20, p.63). **"Since the mass is tightly linked to the thrust in the total energy model … both higher and lower thrust-mass combinations may satisfy the equation at the same time"** (§6.3, p.72) | `m` between OEW and MTOW; "thrust reduction is no larger than 20 % of the maximum thrust profile" (Eq. 21, p.63) | A320, median initial mass under max vs 30 %-reduced thrust: take-off 83.9 vs 60.7 t, climb 64.4 vs 42.1 t (Table 1, §4.3.2, p.69); Cessna Citation II validation MAE 4.3 % of true mass (§5.2, p.71) | pages as given |
| `OpenAP_Sun2020…` — Sun, Hoekstra, Ellerbroek, *Aerospace* 7(8):104, 2020, DOI 10.3390/aerospace7080104 | open aircraft performance model | 4-DOF point mass `dV/dt = (T − D)/m − g sinγ` (Eq. 12), `dψ/dt = g tanφ/(V cosγ)` (Eq. 11); "The net thrust is expressed as the product throttle setting (δT) and the maximum thrust", `T = δT f_thr(h, V, VS)` (Eq. 14); generalised turbofan law `T/T0 = A − 0.377(1+λ)/√((1+0.82λ)G0)·ZM + (0.23 + 0.19√λ)XM²` (Eq. 17), T0 = max static thrust, λ = bypass ratio, A, Z, X, G0 from p/p0 and λ (Eqs. 18–21) | per engine: T0, λ, cruise thrust (Table 1, §5.2) | model paper, no cross-type learning evidence | §5.1–5.2 (p.6–7 of 24) |
| `ClimbGenerative_Pepper2024…` — Pepper, Thomas, *J. Aerosp. Inf. Syst.* 21(6):474–481, 2024, DOI 10.2514/1.I011359 (arXiv 2309.14941) | hybrid generative climb model: a learned functional correction to BADA thrust | §2.1 is titled **"Non-identifiability of BADA parameters for climbing aircraft"**: "four parameters are unknown: ROCD, m, V_TAS, and THR … nominal values are used for V_TAS and m, allowing an effective thrust term, THR̂, to be fitted … representing a combination of the thrust of the aircraft and corrections to misspecification of other BADA parameters". Output: fPCA weights of `THR̂(h)` in kN, per type | per type (26 types); 95 % confidence-ellipsoid bounds on the thrust profile (§2.2) | 707,236 flights (UK, 2019); arrival-time MAE at FL250/325 −26.7 % vs BADA with nominal parameters (Table 1, §4.1); 95 % bounds cover 97.2 % of test data; fitted THR̂ "is compensating for misspecification in the nominal BADA speed profiles and/or mass parameter" (§3, p.5) | arXiv v2 p.2–6, 10 |
| `DescentPIML_Hodgkin2025…` (arXiv 2504.02529) + `MetConditioning_Hodgkin2026…` (AIAA SciTech 2026, DOI 10.2514/6.2026-1792, arXiv 2601.03152) — Hodgkin, Pepper (, Thomas) | same group: descent (drag + CAS functions) and climb (thrust + airspeed conditioned on weather) | outputs are functions (fPCA weights) of drag/CAS or thrust/airspeed that "parameterise the BADA equations"; mass fixed at the nominal BADA value (2025, §2.2). Descent: "greater variation within a dataset of descending aircraft than can be accounted for purely by adjusting the aircraft mass parameter in BADA" (Fig. 1, B738, p.3) | "a unique set of basis functions for each aircraft type" (2026, §II); 13 types (2025), 10 types (2026) | 2025: 116,066 trajectories; error in mean time to bottom of descent below BADA's "by a factor of 10" (abstract). 2026: +20 % skill over six metrics vs a context-free probabilistic baseline (abstract) | 2025 p.2–3, abstract; 2026 abstract, §II |

### 1.2 Quadrotors / UAVs

| file / id | what it is | output space · normalisation | bounds per platform | measured evidence | checked at |
|---|---|---|---|---|---|
| `RotorDragFlatness_Faessler2018…` — Faessler, Franchi, Scaramuzza, *IEEE RA-L* 3(2):620–626, 2018, DOI 10.1109/LRA.2017.2776353 | flatness-based tracking with rotor drag; the open reference for the mass-normalised thrust convention (Mellinger & Kumar 2011 is paywalled, §4) | `v̇ = −g z_W + c z_B − R D Rᵀ v` (Eq. 2), "where **c is the mass-normalized collective thrust**, D … the mass-normalized rotor-drag coefficients" — drag is normalised by mass too | — | not cross-platform | §III, p.2 |
| `DeepDroneAcrobatics_Kaufmann2020…` — Kaufmann, Loquercio, Ranftl, Müller, Koltun, Scaramuzza, RSS XVI 2020, DOI 10.15607/RSS.2020.XVI.040 | sensorimotor policy imitating an MPC expert | action `u = [c, ωᵀ]ᵀ` "continuous mass-normalized collective thrust c and bodyrates ω"; body rates tracked by a low-level controller | in the low-level controller | no action-space ablation (the transfer argument is about input abstraction) | §III, p.2 |
| `QuadBenchmark_Kaufmann2022…` — Kaufmann, Bauersfeld, Scaramuzza, ICRA 2022, DOI 10.1109/ICRA46639.2022.9811564 | the one controlled comparison of action spaces for learned flight control | LV `{vx, vy, vz, ωz}`, **CTBR `{c, ωx, ωy, ωz}` with c in m s⁻²** (Table IV: c_max 9.81–33.04 m s⁻²), SRT `{f1, f2, f3, f4}` (§IV); same net, same observations | CTBR: limits live in the low-level body-rate controller; SRT: in the actuator model. Training randomises mass ±30 %, inertia ±30 % (Table II) | **model mismatch** (BEM simulator + 20 ms delay, Table VI): mean tracking error CTBR 0.6 / 2.6 / 5.6 / 14.9 cm vs SRT 11.3 / 17.6 / crash / crash (Hover / RandC / RaceA / RaceC). **Mass-randomised training**: SRT "train[s] slower, and converge[s] to a final performance substantially lower" (Fig. 3). Nominal model: SRT ≈ CTBR (Table V) | Tables II, IV–VI, Fig. 3 (p.4–6) |
| `Swift_Kaufmann2023…` — Kaufmann, Bauersfeld, Loquercio, Müller, Koltun, Scaramuzza, *Nature* 620:982–987, 2023, DOI 10.1038/s41586-023-06419-4 | champion-level racing policy | "control actions u_t in the form of mass-normalized collective thrust and body rates", chosen because it "is known to combine high agility with good robustness to simulation-to-reality transfer" [ref 44 = QuadBenchmark] | Betaflight low-level controller maps to motors | single platform; no randomisation of platform dynamics, real-data fine-tuning instead | Methods, PDF p.7–8 |
| `XAdapt_Zhang2024…` — Zhang, Loquercio, Tang, Wang, Malik, Mueller, *IEEE T-RO* 41:3948–3964, 2025, DOI 10.1109/TRO.2025.3577037 (arXiv 2409.12949) | one learned **low-level** controller for quadcopters of very different mass/size/motors | interface from the high level = **mass-normalised total thrust `c_Σ,des` + body rates `ω_des`**; the policy outputs motor speeds, `a_t = π(x_t, ẑ_t)`, with intrinsics `ẑ_t = φ(x_{t−k:t−1}, a_{t−k:t−1})` estimated from history (Eqs. 1–4). The expert it imitates maps with the platform's own parameters: `F_des = M [m c_Σ,des; τ_des]`, `a_exp = √(F_des/C_F)` (Eqs. 7–8) | per-vehicle parameters in e_t ∈ R³⁴ (mass, arm length, C_F, C_τ, inertia, drag, max motor speed, mixer); training mass 0.226–0.950 kg, test 0.205–1.841 kg (Table II) | Table IV (sim, random vehicles): success PID-PDn (nominal params) 22 %, L1-PDn (adaptation at high level) 62 %, Geo-A 64 %, PID-INDI-A 67 %, PID-L1 (adaptation at low level) 77 %, **ours 100 % (pos. RMSE 0.148 m)**, expert with true params 100 % (0.061 m). "an adaptive low-level controller tends to perform better with the large model disparity across the platforms" (§IV-B). Real: two vehicles, mass ratio 3.68 (Fig. 1) | Tables II, IV, Eqs. 1–8 (p.3–8) |
| `RAPTOR_Eschmann2025…` — Eschmann, Albani, Loianno, *Science Robotics* 11(114), 2026, DOI 10.1126/scirobotics.aec1481 (arXiv 2509.11481) | 2084-parameter recurrent "foundation policy", end-to-end, distilled from 1000 teachers | output = individual motor commands, **normalised motor effort `ω_mi ∈ [0, 1]`** with the thrust curve scaled per platform, `T = r_t2w · 9.81 · m`, `c_fi = C_fi · T/4` (Supp. Eqs. S9–S12) — i.e. a fraction-of-max action, not mass-normalised; the policy must infer "ratios like thrust-to-weight ratio, torque-to-inertia ratio" in context (p.3) | per-platform thrust curve; thrust-to-weight ≤ 5 during training | 10 real quadrotors, 32 g–2.4 kg, zero-shot (abstract); a linear probe on the hidden state predicts thrust-to-weight with R² = 0.949 (p.9, Fig. 3); an out-of-range T/W ≈ 12 platform tracks worse; capped at T/W 5 it tracks well (Discussion, limitation 3, p.17) | p.3, 9, 17, S3 (PDF p.32) |

### 1.3 Robotics cross-embodiment

| file / id | what it is | output space · normalisation | bounds per platform | measured evidence | checked at |
|---|---|---|---|---|---|
| `OSF_Khatib1987…` — Khatib, *IEEE J. Robotics & Automation* RA-3(1):43–53, 1987, DOI 10.1109/JRA.1987.1087068 (scanned; read as page images) | operational-space control | `Λ(x)ẍ + μ(x, ẋ) + p(x) = F` (Eq. 14), `Γ = Jᵀ(q)F` (Eq. 28); `F = F_m + F_ccg`, **`F_m = Λ̂(x)F*_m`**, `F_ccg = μ̂(x, ẋ) + p̂(x)` (Eqs. 29–30): "With a perfect nonlinear dynamic decoupling, the end-effector becomes equivalent to a **single unit mass** I_m0"; the command is designed in that unit-mass space, `F*_m = I ẍ_d − k_p(x − x_d) − k_v(ẋ − ẋ_d)` (Eq. 31) | limits imposed in the unit-mass command space: `ν = min(1, V_max/√(ẋ_dᵀẋ_d))` (Eq. 33); the plant-specific Λ, μ, p enter only in the mapping | theory | §III–IV, p.46–47 |
| `UP-OSI_Yu2017…` — Yu, Tan, Liu, Turk, RSS XIII 2017, DOI 10.15607/RSS.2017.XIII.048 | universal policy conditioned on physical parameters + online system identification | `u_t = π(x_t, μ)`, μ = model parameters (mass, inertia, friction, CoM), normalised to [−1, 1] (Fig. 2 caption); raw actuator outputs | μ range sampled in training | UP-OSI "comparable and sometimes better performance than UP-true" (true μ); cart-pole with parameters 100 % beyond the training range (Fig. 5d) — figures only | §IV–V, Figs. 2, 4, 5 (p.4–8) |
| `RMA_Kumar2021…` — Kumar, Fu, Pathak, Malik, RSS XVII 2021, DOI 10.15607/RSS.2021.XVII.011 | base policy + adaptation module (A1 quadruped) | `a_t = π(x_t, a_{t−1}, z_t)`, `z_t = μ(e_t)`, e_t ∈ R¹⁷ (payload, CoM, friction, motor strength, Kp/Kd, terrain; Table I); action = desired joint positions → PD torque | payload 0–6 kg train, 0–7 kg test (Table I) | sim success (Table II, p.8): Robust (no parameters) 62.4 %, SysID (predicts e_t explicitly) 56.5 %, RMA without adaptation 52.1 %, **RMA 73.5 %, Expert (true z_t) 76.2 %** | Table I–II, §III |
| `MetaMorph_Gupta2022…` — Gupta, Fan, Ganguli, Fei-Fei, ICLR 2022 (arXiv 2203.11931) | transformer over per-limb tokens for many morphologies | per-limb tokens = proprioception + morphology (shape, density, joint range/axis, actuator gear; App. A, p.14); per-limb action outputs (Eq. 4) | per-robot gear/joint limits enter as tokens | "without access to the morphological information, MetaMorph-NM fails to learn a policy that can control diverse robot morphologies" (§5.2, p.7); 5× more sample-efficient than per-morphology MLPs; zero-shot to 400 variants per dynamics/kinematics property (Fig. 6) | §4.2, §5.2–5.3, App. A |
| `OXE_OpenXEmbodiment2023…` — Open X-Embodiment Collaboration, ICRA 2024, DOI 10.1109/ICRA57147.2024.10611477 (arXiv 2310.08864) | RT-1-X / RT-2-X on 22 embodiments | "coarsely aligned" 7-DoF end-effector action; **"We normalize each dataset's actions prior to discretization. This way, an output of the model can be interpreted (de-normalized) differently depending on the embodiment used"**; frames not aligned, absolute or relative allowed | per-dataset statistics | RT-1-X +50 % mean success vs the original methods (Fig. 4); RT-2-X ≈3× on emergent skills (Table II) — credited to co-training; normalisation not ablated | §IV-A "Data format consolidation" (p.4), §V |
| `Octo_OctoModelTeam2024…` — Octo Model Team (Ghosh, Walke, Pertsch, Black, Mees et al.), RSS XX 2024, DOI 10.15607/RSS.2024.XX.090 (arXiv 2405.12213) | open generalist policy, diffusion action head | mixture restricted to datasets with delta end-effector control (§III-B, p.4); new action spaces get a new head at fine-tuning. Code (`octo@241fb35`, `octo/data/utils/data_utils.py`): default `NORMAL` = `(x − mean)/(std + 1e-8)` per dataset; `BOUNDS` = `clip(2(x − p01)/(p99 − p01 + 1e-8) − 1, −1, 1)` | per-dataset statistics | fine-tuning to joint-position control: Berkeley Pick-Up 60 %, Bimanual 80 %; average over six setups 72 % vs 20 % from scratch (Table I, p.7) | §III-B, Table I; code l.34–38, 243–290 |
| `OpenVLA_Kim2024…` — Kim, Pertsch, Karamcheti et al., CoRL 2024, PMLR 270:2679–2713 (arXiv 2406.09246) | 7B VLA | "we discretize each dimension of the robot actions separately into one of 256 bins … uniformly divide the interval between the 1st and 99th quantile of the actions in the training data" (§3.2, p.5); code (`openvla@c8f03f4`, `modeling_prismatic.py::predict_action`) un-normalises per dataset `unnorm_key`: `0.5·(â + 1)·(q99 − q01) + q01` | per-dataset q01/q99 | +16.5 % absolute over RT-2-X on 29 tasks (abstract); normalisation not ablated | §3.2; code |
| `HPT_Wang2024…` — Wang, Chen, Zhao, He, NeurIPS 2024 (arXiv 2409.20537) | embodiment-specific stems + heads around a shared trunk | head "outputs a normalized action trajectory"; loss = Huber "between the normalized action labels based on dataset statistics and the network's action predictions"; head re-initialised for a new embodiment (§3, p.5–6) | per-embodiment head + statistics | > 20 % fine-tuned gain on unseen tasks (abstract) | §3 |
| `CrossFormer_Doshi2024…` — Doshi, Walke, Mees, Dasari, Levine, CoRL 2024, PMLR 270:496–512 (arXiv 2408.11812) | one transformer for arms, bimanual, wheeled, quadruped, quadcopter | per-embodiment action heads fed by readout tokens; **"does not require manual alignment of the observation or action spaces"**; action normalisation not described in the paper | per-head action dimension/chunk | average success 73 % vs 67 % single-robot vs 51 % best prior (Fig. 5, §4.1); 3× over Yang et al., which aligns navigation and manipulation actions (Fig. 6, §4.2); limitation: "Our results do not yet show significant positive transfer across embodiments" (§5) | p.5–8 |
| `RDT-1B_Liu2024…` — Liu, Wu, Li et al., ICLR 2025 (arXiv 2410.07864) | 1.2B diffusion policy, 46 datasets | 128-dim "Physically Interpretable Unified Action Space": each element filled by physical meaning, the rest padded (§4.2, App. C). **"We roughly align the scales of various datasets by unifying the units of physical quantities (m, rad, m/s, rad/s, etc) rather than strictly normalizing to [−1, 1] or N(0, 1) as in prior work … Rescaling the physical quantities will destroy such shared properties and thus impair the model's ability to transfer across robots"** (App. D, p.22) | physical units, no per-dataset scaling | argued, **not ablated** (ablations cover size, pre-training, diffusion; Table 2) | §4.2, App. C–D |

### 1.4 Autonomous driving / multi-agent trajectory prediction

| file / id | what it is | output space · normalisation | bounds per platform | measured evidence | checked at |
|---|---|---|---|---|---|
| `Trajectron++_Salzmann2020…` — Salzmann, Ivanovic, Chakravarty, Pavone, ECCV 2020, LNCS pp. 683–700, DOI 10.1007/978-3-030-58523-5_40 | CVAE forecaster; the net outputs **control actions** that are integrated through agent-type dynamics | pedestrians: single integrator (velocity); vehicles: dynamically-extended unicycle with `u = (a, ω)`, acceleration and heading rate (App. B, p.19). Chosen over a bicycle model because that "requires estimation of the vehicle's center of mass, wheelbase, and front wheel steer angle" (§4, p.6) — one mass-free control space for every vehicle | none stated | nuScenes vehicles (Table 5a, p.13), without → with dynamics integration: FDE-ML @3 s 1.25 → 1.13 m, @4 s 2.24 → 2.17 m; KDE-NLL @3 s 0.87 → −1.67; boundary violations @3 s 2.8 → 3.2 %; "dynamics integration is the dominant performance-improving module" | §4, App. B, Table 5 |
| `DKM_Cui2020…` — Cui, Nguyen, Chou, Lin, Schneider, Bradley, Djuric, ICRA 2020, DOI 10.1109/ICRA40945.2020.9197560 | kinematic layer between the CNN and the output | predicts **longitudinal acceleration a and steering angle γ**, rolled out through a kinematic bicycle model with per-actor κ_i (l_r, l_f "estimated from the tracked state", max acceleration and steering) (Eqs. 4–6) | "the controls are clipped to be within the allowed ranges"; "Maximum absolute acceleration and steering angle are set to 8 m/s² and 45°, respectively, roughly following characteristics of a midsize sedan" (fn. 1) — one bound in acceleration units | Table I (p.5): DKM position ℓ2 1.34 / 4.21 m (@3 / 6 s), heading 3.38 / 4.92°, **0 % infeasible** vs unconstrained UM 1.34 / 4.25 m, 4.82 / 7.69°, 26.0 % infeasible; UM-velo (unconstrained velocity "controls") 27.3 % | §III-B, Table I |

### 1.5 Biomechanics / locomotion scaling (all paywalled; §4)

| file / id | what it is | output space · normalisation | bounds per platform | measured evidence | checked at |
|---|---|---|---|---|---|
| Hof, "Scaling gait data to body size", *Gait & Posture* 4(3):222–223, 1996, DOI 10.1016/0966-6362(95)01057-2 — **not fetched** | the standard dimensionless scaling of gait data | "To convert to dimensionless units, divide quantity by": time `(l/g)^½`, velocity `(g·l)^½`, acceleration `g`, force `m·g`, moment `m·g·l`, work `m·g·l`, power `m·g^{3/2}·l^{1/2}`; m = body mass, l = leg length | — | in the same posting (Rose et al. data): children age 1 → adult walk 0.63 → 1.45 m/s while the normalised velocity is 0.36 → 0.46, "essentially unchanged above age 3½" | Hof's own posting of the table on the Clinical Gait Analysis FAQ (signed, 14 June 1996); the letter itself not read |
| Alexander & Jayes, *J. Zool.* 201:135–152, 1983, DOI 10.1111/j.1469-7998.1983.tb04266.x — **not fetched** | dynamic similarity hypothesis | mammals "move in a dynamically similar fashion whenever they travel at speeds that give them equal values of … the Froude number" | — | "tenable in many cases" for cursorial quadrupeds (most mammals > 5 kg); "a reasonable approximation" across cursorial / non-cursorial and bipeds (abstract) | abstract (Crossref) |
| Moisio, Sumner, Shott, Hurwitz, *J. Biomech.* 36(4):599–603, 2003, DOI 10.1016/S0021-9290(02)00433-5 — **not fetched** | body mass vs body weight × height normalisation of joint moments | `M/m` vs `M/(m·g·h)` | — | 158 subjects: unnormalised, height or weight explain 7–82 % of variance in 10 peak moments; normalised ≤ 6 %, except two components (13 %, 22 %) (abstract) | PubMed 12600350 |
| Pierrynowski & Galea, *Gait & Posture* 13(3):193–201, 2001, DOI 10.1016/S0966-6362(01)00097-2 — **not fetched** | eight scaling strategies compared | none / ad hoc / dimensionless numbers / similarity-based | — | 10 subjects, 1.33–1.96 m, 42.3–148.8 kg: the dimensionless (and two other) strategies cut global inter-subject variation to 44 % of unscaled (abstract) | PubMed 11323225 |

### 1.6 Scientific ML: dimensionless learning

| file / id | what it is | output space · normalisation | bounds per platform | measured evidence | checked at |
|---|---|---|---|---|---|
| `BuckinghamPi_Bakarji2022…` — Bakarji, Callaham, Brunton, Kutz, *Nat. Comput. Sci.* 2:834–844, 2022, DOI 10.1038/s43588-022-00355-5 (arXiv 2202.04643) | learn the Pi groups that collapse the data | `Π = exp(log(P̃)Φ)` with `DΦ = 0` (Eq. 7); `Φ̌p = argmin ‖Πq − ψ(exp(log(P)Φp))‖² + λ1‖Φp‖1 + λ2‖Φp‖2  s.t. DpΦp = 0` (Eq. 11); BuckiNet puts this in the first layer | — | recovers known groups (rotating hoop, Blasius similarity variable, Rayleigh number) — qualitative | §3–3.1, Eqs. 7, 11 (p.5–6); §4.2–4.4 |
| `DimensionlessLearning_Xie2022…` — Xie, Samaei, Guo, Liu, Gan, *Nat. Commun.* 13:7562, 2022, DOI 10.1038/s41467-022-35084-w | two-level search over dimensionally-invariant bases + regression | inputs replaced by learned dimensionless numbers | — | Rayleigh–Bénard: test R² 0.999 at the classical Ra (Fig. 1f); keyhole: the discovered number R² 0.98 vs a local-optimum group 0.64 (Fig. 2); a parameter list missing thermal diffusivity caps R² < 0.80 (main text citing SI §6.1) | p.3–5 |
| `UnitsEquivariance_Villar2023…` — Villar, Yao, Hogg, Blum-Smith, Dumitrascu, *JMLR* 24(109):1–32, 2023 (arXiv 2204.00887) | exact units equivariance | "we first construct a dimensionless version of its inputs … and then perform inference in the dimensionless space"; outputs re-dimensionalised by products of inputs with the right units (e.g. `m_i L_j \|g\|` for the Hamiltonian) | — | springy double pendulum, HNN state relative error (Table 1, p.13): in-distribution dimensional .0055 vs dimensionless .0061; **all kg-quantities scaled by U(3, 7): .3669 vs .0089**; m, k_s, L from a shifted distribution: .1885 vs .0435 | abstract; Table 1 and text, p.13–15 |
| `ClimateInvariant_Beucler2024…` — Beucler, Gentine, Yuval et al., *Sci. Adv.* 10(6):eadj7250, 2024, DOI 10.1126/sciadv.adj7250 (arXiv 2112.08440) | physically rescaled inputs/outputs so their distributions match across climates | q → relative humidity, T → plume buoyancy, LHF → LHF_Δq (Fig. 1, §3) | — | NN trained cold, tested warm: raw-data MSE rises ≈10× (> 100 W² m⁻⁴); the RH transform alone cuts it 5–10×; all three bring it within ≈25 % of a NN trained on the warm climate (§4.1, Fig. 4, p.10–11) | arXiv v5 §4.1 |

## 2. Patterns across fields

**P1 — Predict a platform-free (mass-normalised or kinematic) command; map it to actuators with the platform's own
inverse model.** Khatib (`F = Λ̂F*_m + μ̂ + p̂`, the command lives in a unit-mass space); quadrotor CTBR
(`c` = thrust/mass in m s⁻²: Faessler, Deep Drone Acrobatics, QuadBenchmark, Swift); XAdapt (the high level
emits `c_Σ,des`, the expert maps `m·c_Σ,des` through the mixer and propeller constant); DKM / Trajectron++
(acceleration + steering or heading rate, no mass anywhere); BADA's own rate law `ROCD = (T − D)V/(mg)·ESF`
and its fleet-wide acceleration and bank limits. *Needs:* a trusted per-platform inverse model at the mapping
step (for quadrotors: m, mixer, C_F; for Khatib: Λ, μ, p). *Costs:* actuator limits become state- and
platform-dependent bounds in the normalised space and must be recomputed per step (DKM and BADA sidestep this
with one fleet-wide kinematic bound). *Evidence:* QuadBenchmark Table VI and Fig. 3 (CTBR survives ±30 % mass
randomisation and model mismatch; per-rotor thrust crashes) — but CTBR also adds an inner body-rate feedback
loop, so that benchmark cannot separate mass normalisation from the inner loop **(reading)**. XAdapt Table IV:
putting the platform-specific adaptation below the normalised interface beats putting it above (77 % vs 62 %),
and the true-parameter mapping is best.

**P2 — Per-dataset statistical action normalisation, de-normalised per embodiment.** OXE (normalise, then
discretise), Octo (`(x − μ)/σ` per dataset), OpenVLA (q01–q99 → 256 bins, un-normalised with `unnorm_key`),
HPT (dataset statistics, per-embodiment heads). *Needs:* per-embodiment statistics only, no physics. *Costs:*
the same normalised value means different physical effects on different embodiments; none of these papers
ablates the normalisation. A thrust fraction `δ = T/T_max` is this pattern with the type's T_max as the range
**(reading)**.

**P3 — One physical unit system instead of statistics.** RDT-1B keeps SI units so that "1 (m)" is the same
length in every dataset, and argues that rescaling destroys cross-robot transfer. *Needs:* a shared physical
quantity per output slot. *Evidence:* argued, not ablated.

**P4 — Keep the raw action; condition the network on physical parameters (explicitly or inferred).** UP-OSI
(`π(x, μ)`), RMA (`z = μ(e)`), MetaMorph (morphology tokens), XAdapt (intrinsics), RAPTOR (in-context).
*Needs:* the parameters, or history to infer them. *Evidence:* RMA Expert 76.2 % vs no-parameter Robust
62.4 %; MetaMorph without morphology fails; RAPTOR's hidden state encodes thrust-to-weight (R² 0.949).
Note what gets identified: RAPTOR says the policy needs "ratios like thrust-to-weight ratio, torque-to-inertia
ratio" — the dimensionless combination, not mass and thrust separately. *Costs:* the network has to learn the
multiplicative parameter × action interaction; out-of-range parameters degrade (RAPTOR T/W 12 vs ≤ 5 trained;
XAdapt sees its model-based adaptive baselines "failing near the boundaries of the adaptation range"). All P4
papers are closed-loop feedback policies that correct errors every step **(reading)**.

**P5 — Dimensionless groups / units equivariance.** Hof, Alexander & Jayes, BADA 4 (`C_T`, `C_L` with n inside,
`C_W`), Bakarji, Xie, Villar, Beucler. *Evidence:* the gain is out-of-distribution: Villar .3669 → .0089 when
masses are rescaled 3–7×, but no gain in distribution (.0055 vs .0061); Beucler ≈10× error blow-up removed;
Moisio 7–82 % → ≤ 6 % size-explained variance; Pierrynowski 44 %. *Needs:* the right groups — Xie: the wrong
group gives R² 0.64 vs 0.98, and a missing variable caps R² < 0.80.

**P6 — One model per platform type.** BADA itself, Alligier 2013/2015/2018, Pepper & Thomas, Hodgkin et al.,
OpenAP. Every learned aviation predictor found here trains per type and absorbs the unknown mass into an
"equivalent mass" (Alligier) or "effective thrust" (Pepper), because mass and thrust are not separately
identifiable from trajectories (Alligier 2015 §IV-D, Sun 2018 §6.3, Pepper §2.1). No paper found trains one
learned control model across aircraft types.

**Closest analogues to "predict a weight-normalised longitudinal specific force + load factor + bank, then map
to thrust per aircraft":** (1) QuadBenchmark / Swift / Deep Drone Acrobatics — `c = T/m` + body rates, mapped to
rotors per platform; (2) XAdapt — the same interface, with measured evidence that platform-specific mapping
belongs below it; (3) Khatib — unit-mass command + `Λ̂F* + μ̂ + p̂`, the exact algebraic form of
`T = m·a + D + m·g·sinγ` **(reading)**; (4) BADA — `(T − D)/(mg)` as the specific excess thrust, `n` inside
`C_L`, and fleet-wide limits in ft/s² and degrees; (5) DKM / Trajectron++ — acceleration-level outputs; DKM clips
to one fleet bound (8 m/s², 45°), Trajectron++ picks the unicycle because per-vehicle parameters are hard to
estimate online. The mass–thrust confounding
(Alligier 2015, Sun 2018, Pepper 2024) is the aviation-side argument that only `(T − D)/m` is identifiable.

**Sources that argue against normalising, or show raw-parameter conditioning is enough:** RAPTOR (a
fraction-of-max motor action, no mass normalisation, zero-shot over 32 g–2.4 kg — in closed loop with
in-context identification); CrossFormer (no manual action-space alignment, 3× over an aligned-action baseline;
confounded by architecture and camera setup, and no positive transfer shown); RMA / UP-OSI / MetaMorph
(conditioning on parameters works — RMA 73.5 % vs 76.2 % with true parameters, UP-OSI approaches UP-true,
MetaMorph matches per-morphology MLPs in 2 of 3 environments — but none compares against a normalised action);
Villar Table 1 Experiment 1 (in distribution the dimensional model is as good: .0055 vs .0061); QuadBenchmark
Table V (nominal model: per-rotor thrust ≈ CTBR). RDT-1B argues against *statistical* per-dataset scaling, in
favour of physical units — not against physical normalisation.

## 3. Assessment for ts_transformer

Written 2026-09-14 against `dev-leg-ctrl` @ `d14e21d`. Decision state: **proposal, nothing built** (§3.7).

### 3.1 The answer

Yes, and the literature points one way: predict the thrust channel as the **specific force along the path,
`n_x = (T − D)/W`**, instead of `δ = T/T_max`. With that change all three controls `(n_x, φ, n)` act
identically on every airframe, and the aircraft enters only through feasibility (thrust and stall limits) and
the exported thrust. But the premise needs a correction first: the package has not predicted newtons since
2026-08-18 (`outputs/envelope.py`). `δ = T/T_max` already normalises by the actuator's capacity, and OpenAP
sizes the engines to the airframe, so the leftover type dependence is tens of percent, not a factor (§3.2).
What is still missing is normalising by the **effect**. The measurable consequence is that the head commands
the same δ on every class while the classes need different ones (§3.3).

### 3.2 Measured: how much one δ varies across the fleet (`measurements/fleet_spread.py`)

Scope: 31,119 OpenAP-direct arrivals in the five v5 arrival manifests, 28 types. Operating point: the model's
own approach, 1.3 × its 1-g stall speed at ~2000 ft ISA, on a 3° glide.

- **Thrust-to-weight at landing mass:** 0.310 (A343) to 0.417 (B77W); flight-weighted p5/p50/p95 is
  0.342 / 0.358 / 0.407.
- **Gain `∂V̇/∂δ = T_max/m`:** 3.04 to 4.09 m s⁻² per unit δ; weighted 3.36 / 3.51 / 3.99.
- **δ that holds speed on the glide:** 0.041 (E145) to 0.086 (A321); weighted 0.059 / 0.067 / 0.084.
- **Why the spread is small:** `aircraft/aero_params.aero_params_for_aircraft` sets only `S` and `Cl_max`.
  `Cd0 = 0.02` and `k = 0.04` are the same on every type. So at a fixed V/V_s the lift coefficient is fixed,
  wing loading cancels, and drag per weight depends only on the CLmax bucket: 0.067 / 0.071 / 0.076 / 0.082
  for CLmax 2.2 / 2.4 / 2.7 / 3.0 (bizjets and E145 / heavies / 737 family, E-jets and CRJ / A320 family).

### 3.3 Measured: the head does not adapt its thrust to the type (`measurements/per_class_readout.py`)

Source: KRDU val, 1404 flights, `4dTrajectory/outputs/KRDU/experiments/b1_quantile_20260907/B1_point_matched{,_s2024}_pred_val`
(control, iTransformer, first-order-lag @ scaled-transport-chart-velocity, simple-v3 + final-time 26, N = 32;
seeds 1337 / 2024 on one split). Stratum: established at the anchor. All values are time-means over the span
the prediction and the track share from the anchor. "Truth-needed δ" is the package's own actual-control
inversion (`outputs/dynamics/inverse.actual_controls`) of the observed track under the same flight model, and
`n_x = V̇/g + sin γ`. The truth columns are over seed 1337's shared span; seed 2024's span moves them by at most
0.001.

| class | n | head δ (1337 / 2024) | truth-needed δ | truth n_x (g) | head − truth n_x (g) | speed bias, m/s ± SE | ADE mean (m) |
|---|---:|---|---:|---:|---|---|---|
| B737 family | 314 | 0.051 / 0.045 | 0.036 | −0.0610 | −0.0056 / −0.0083 | −1.41 ± 0.28 / −1.86 ± 0.26 | 422 / 453 |
| A320 family | 233 | 0.050 / 0.043 | 0.041 | −0.0655 | −0.0043 / −0.0072 | +0.90 ± 0.28 / +0.58 ± 0.30 | 363 / 399 |
| regional (E-jets, CRJ) | 250 | 0.053 / 0.048 | 0.031 | −0.0629 | +0.0010 / −0.0006 | +1.15 ± 0.30 / +1.17 ± 0.32 | 384 / 404 |
| heavy (MTOW > 100 t) | 44 | 0.052 / 0.047 | 0.028 | −0.0644 | +0.0048 / +0.0033 | +3.30 ± 0.74 / +3.04 ± 0.72 | 403 / 432 |

- **The head is nearly type-blind in δ.** Its δ varies by only 0.003 (seed 1337) or 0.005 (seed 2024) across
  classes, while the truth-needed δ varies by 0.013.
- **In the quantity that moves the aircraft, that becomes a class-dependent bias.** The head's n_x bias spans
  0.0104 g (seed 1337) or 0.0116 g (seed 2024). The truth's own n_x varies by only 0.0045 g across classes.
- **The classes that need the least thrust are the ones the head over-drives:** heavy (0.028) and regional
  (0.031).
- **Heavies fly faster than their tracks relative to the 737 family,** by 4.7 m/s (seed 1337) or 4.9 m/s
  (seed 2024), about 6 SE.
- **(reading)** A head equally blind to the type but predicting n_x would carry only the truth's own 0.0045 g
  spread.
- **This is a mechanism measurement, not an accuracy claim.** Class ADEs sit within 363–453 m; the heavies are
  not the worst class; there are 44 heavies at one airport; and the two seeds vary the training, not the flights.
- **Part of the truth-needed spread is the model's own CLmax buckets (§3.2).** That is exactly the part an n_x
  head would no longer have to learn.
- **Aside: the duration head does not see the aircraft at all.** It reads the history only
  (`ControlFeatureModel.duration`), and it predicts heavies 7.1 s / 10.4 s long against about 0 for the fleet.
  The vectored stratum shows the same δ blindness, but its class pattern is weaker, and it has only 13 heavies.

### 3.4 Measured: the conditioning vector carries less than it looks

Of the eight channels in `outputs/conditioning.py`, four are constant on all 28 types: `cd0_0p1` = 0.2,
`induced_k_0p1` = 0.4, `stall_threshold` = 0.9, `stall_k_0p2` = 0.5. Mass, T_max and S correlate pairwise at
r = 0.97–0.99, which means they encode size. CLmax takes four values. The groups that actually separate the
dynamics (T_max/W, the CLmax bucket) are quotients the MLP would have to learn. §3.3 says it did not. This is
pattern P4 (condition on raw parameters) without the closed loop that makes P4 work in RMA, UP-OSI and RAPTOR
**(reading)**.

### 3.5 Recommendation: the specific-force parameterisation (pattern P1)

```
controls  (n_x, φ, n),   n_x = (T − D)/W

V̇ = g·(n_x,r − sin γ)
γ̇ = g·(n_r·cos φ − cos γ)/V
ψ̇ = g·n_r·sin φ/(V·cos γ)

n_r    = today's stall-limited load factor (unchanged)
n_x,r  = n_x saturated to [(T_lo − D)/W, (T_max − D)/W],   T_lo = −0.2·T_max (today's floor)
T      = W·n_x,r + D   → the evaluation record stays in newtons
```

Where thrust is not saturated, drag cancels out of V̇. The airframe (m, T_max, S, CLmax, polar) enters only
through the two saturations and the exported thrust. Keeping today's thrust floor means the set of admissible
trajectories is today's set; only the coordinate the head predicts in changes, so the arm is a clean test of
the parameterisation. This is the quadrotor mass-normalised collective thrust (QuadBenchmark, Swift, XAdapt),
Khatib's unit-mass command `F = Λ̂F* + μ̂ + p̂`, and BADA's `(T − D)/(mg)`. It also matches DKM and
Trajectron++, which emit accelerations with no mass anywhere. The fleet-wide head box would be about
[−0.16, +0.35] at the §3.2 operating point; it is sized in the design.

What it buys:
1. **The output space matches the invariance in the data:** 0.0045 g of class spread in the truth, against
   0.010–0.012 g of class bias from a δ head (§3.3).
2. **It predicts the quantity the tracks identify.** Mass and thrust are not separately identifiable from
   trajectories (Alligier 2015 §IV-D, Sun 2018 §6.3, Pepper 2024 §2.1), but `(T − D)/m` is. Today's δ teacher
   is an inference through an assumed m, T_max and polar. The n_x teacher is pure kinematics: it is the
   `tangential` term `inverse.actual_controls` already computes.
3. **The learned head is insulated from the aero model (reading).** Moving to OpenAP's per-type polars plus
   flap and gear drag (the drag the −0.2 floor stands in for) would move every δ target. Under n_x it moves only
   the saturation bounds and the exported thrust. So this change also makes a later `aerodynamic_model`
   upgrade cheap.

What it costs, all inside `ts_transformer` (the `aerodynamic_model` RHS and the CasADi mirror are unchanged):
- **A new `ControlOutput` axis**, e.g. `control_parameterization ∈ {thrust-fraction, specific-force}`, default
  `thrust-fraction`, so every stored checkpoint loads and keeps its name.
- **The RHS adapter evaluates `T` at every RK4 stage.** A thrust held over a segment is not a held n_x.
- **The soft saturation must be inert where it does not bind.** The `soft_max` `s·ln 2` overshoot trap
  (CHANGELOG, L3.d) applies here too.
- **An inverse registered under the same config key.** This is the "inverse of the configured forward model"
  contract.
- **The first-order-lag actuator then lags n_x instead of T.** That approximates spool lag, and the design
  must say so.
- **The speed-floor hook's thrust inversion becomes `n_x ≥ sin γ + (V_floor − V)/(g·τ_eff)`.** The barrier
  and the trombone are untouched.
- **The imitation weight must be recalibrated,** because the channel's scale (`CONTROL_HALF_WIDTH`) changes.

### 3.6 Alternatives and why they are not first

- **Thrust-to-weight `T/W` as the command.** It removes the ±10 % gain spread but keeps drag per weight, and
  drag per weight (the CLmax buckets) is what separates the classes in §3.3.
- **Better conditioning (P4).** Replace the eight raw channels with the dimensionless groups (T_max/W, CLmax,
  CL at the type's published V_ref) and drop the four constants. This is cheap and orthogonal, so run it as its
  own axis; giving the duration head the vector belongs to the same axis. RAPTOR shows a fraction-of-max action
  can work, but only in closed loop with in-context identification. Our head emits one open-loop 32-segment
  schedule.
- **Per-type heads or models (P6).** This is how every learned aviation predictor above works, but the 28 types
  hold 1 to 5,864 arrivals each, and the tail cannot carry its own head.
- **Per-dataset statistical normalisation (P2).** δ = T/T_max already is this pattern.
- **Instance normalisation.** It is off by package contract.
- **Expected size of the gain (reading).** In distribution, raw and normalised forms tie (Villar Exp. 1,
  QuadBenchmark Table V); the gain shows up out of distribution. So expect the effect on the tail (heavies,
  bizjets) and across fleets, not on pooled ADE. The gate is therefore the §3.3 per-class numbers, and pooled
  ADE only needs to stay within the control path's ~125 m seed line.

### 3.7 Plan

| step | what | status |
|---|---|---|
| N0 | §3.2–3.4 measurements | done 2026-09-14 |
| N1 | design doc under `4dTrajectory/ts_transformer/docs/`: the axis, the RK4-stage adapter, the n_x box, lag semantics, the speed-floor hook, export, tests | not started, needs a go |
| N2 | implement specific-force, with unit tests: forward/inverse round trip, drag cancels out of V̇, saturation inert when not binding, newtons exported | not started |
| N3 | KRDU arms on the `B1_point_matched` recipe, 2 seeds each, thrust-fraction vs specific-force. Gates: the per-class n_x bias range shrinks toward 0.0045 g; the heavy − 737 speed-bias gap (4.7–4.9 m/s) closes; pooled ADE within ~125 m; the straight-in FDE veto; flyability | not started |
| N4 | conditioning as dimensionless groups (§3.6), as its own axis | not started |
| N5 | pooled five-airport arm: 698 heavies (2.2 %) and 173 bizjets in the roster, against 57 heavies in KRDU val | not started |

## 4. Not fetched / secondary

- Alligier & Gianazza 2018, *TR-C* 96:72–95, DOI 10.1016/j.trc.2018.08.012 — HAL preprint
  `https://enac.hal.science/hal-01878615/file/trc2018.pdf` sits behind an Anubis bot check (scripted fetch and
  WebFetch both refused); claims in §1.1 are from the abstract (Semantic Scholar API). Code:
  `github.com/richardalligier/trc2018`.
- Sun, Blom, Ellerbroek, Hoekstra, "Aircraft mass and thrust estimation using recursive Bayesian method",
  ICRAT 2018 — `https://research.tudelft.nl/files/103003225/ICRAT_2018_paper_19_1_.pdf` returns 403 to scripts;
  not read, no claims used.
- Mellinger & Kumar, "Minimum snap trajectory generation and control for quadrotors", ICRA 2011, pp. 2520–2525,
  DOI 10.1109/ICRA.2011.5980409 — paywalled; the mass-normalised thrust convention is quoted from Faessler 2018.
- Gallo, Navarro, Nuic, Iagaru, "Advanced aircraft performance modeling for ATM: BADA 4.0 results", DASC 2006,
  DOI 10.1109/DASC.2006.313660 — paywalled, not read.
- BADA 4 User Manual — licence-only; BADA 4 formulas quoted from Nuic 2010 and the pyBADA source.
- Hof 1996, Alexander & Jayes 1983, Moisio et al. 2003, Pierrynowski & Galea 2001 — paywalled; used via
  abstracts (PubMed E-utilities, Crossref) and Hof's own posting.
- Web sources read, not saved: pyBADA `src/pyBADA/bada4.py` @ `a03b3ca`; Octo `octo/data/utils/data_utils.py`,
  `octo/data/dataset.py` @ `241fb35`; OpenVLA `prismatic/vla/datasets/rlds/utils/data_utils.py`,
  `prismatic/extern/hf/modeling_prismatic.py` @ `c8f03f4`; Clinical Gait Analysis FAQ
  `http://www.clinicalgaitanalysis.com/faq/normalisation.html` (Hof's table).
- Seen, abstracts only, not used: CAR, cross-vehicle kinodynamics via a mobility latent (arXiv 2603.06866);
  VDD, vehicle-dynamics-embedded world model (arXiv 2512.02417); a Neural ODE aircraft dynamics model
  (arXiv 2509.23307).
