# A trajectory as a sentence: what makes a token model *speak* rather than imitate (2026-09-20)

Why this folder exists. A design being written for `4dTrajectory/ts_transformer` treats an arrival
trajectory as a **sentence**: each segment becomes a discrete intent **token**, a causal model runs
over the token sequence, and a learned low-level controller flies each token. The claim behind it
is that LLM training practice transfers — that after next-token pretraining on recorded
trajectories (teacher forcing) the model will *generate*, not merely reproduce. This folder
collects the primary text for the step that claim skips. The question:

> After next-token pretraining on recorded trajectories with teacher forcing, **what makes such a
> model "speak"** — generate trajectories that reach the runway safely, inside the procedure and
> inside the dynamics — **rather than imitate**? Specifically: what is tokenised and at what step;
> what the training objective is (teacher-forced next token / on-policy / RL reward); whether
> evaluation is closed-loop; how big the data is; whether a goal or constraint term exists at
> training or at decoding; and **what the post-training step measurably buys over the pretrained
> model**.

How to read it:
- **Quotes are copied** from the saved PDF text (`pdftotext`), each kept under about 30 words, with
  "…" marking a cut, and located by section / table / figure. Where a PDF's text layer breaks a
  word across a line, the quote reproduces what `pdftotext` emitted.
- **"(reading)"** marks anything the text does not state: calling a model's evaluation closed-loop
  when the paper only describes the rollout, converting steps into seconds, or judging what a
  mechanism is analogous to in our design.
- **Nothing is filled from memory.** §4 *Not verified* lists every paper and every number that
  could not be confirmed, with its arXiv id or DOI.
- The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`). `./download.sh` re-fetches
  the **12** papers new to this repository (run 2026-09-20: all 12 returned, `%PDF-` verified).
  **Eight further sources already live in sibling folders and are NOT duplicated here** — they are
  cited by path in §1 (the rule `multimodal_intent/download.sh` and `prediction_horizons/download.sh`
  use). MotionLM, Trajeglish, SMART, grammar-constrained decoding, scheduled sampling and FTP-LLM
  are in `../manoeuvre_tokens/`; DAgger, FlightBERT++, TNT/DenseTNT/MTR are in
  `../hierarchical_prediction/`; TPP is in `../multimodal_intent/`.

---

## At a glance (only what the table below supports)

| | tokenised driving / traffic models (A, B, C, E) | goal-directed decoding (D) | exposure-bias background (F) | aviation (G) |
|---|---|---|---|---|
| what is a token | a **0.1–0.5 s motion delta or segment** (Trajeglish 0.1 s, MotionLM / SMART / InfGen / SaFeR 0.5 s, GUMP 0.5 s), or a **1 s continuous patch** (BehaviorGPT), or a **quantised (accel, steer) pair** (CtRL-Sim, SaFeR) | nothing is tokenised — both trees branch over **continuous candidate trajectories** | a word | a **Gray-coded bit vector per attribute** (FlightBERT++) or the **LLM's own sub-word tokens of printed digits** (FTP-LLM, FLY-EVAL++) |
| vocabulary | 169 (MotionLM) · 384 (Trajeglish) · 512–2048 (SMART) · 1000 (CtRL-Sim) · 2048 (InfGen) · 2972 (GUMP) · 3969 = 63² (SaFeR) | — (TPP: 2 stages × branching 4; DTPP: 2 stages, ≤ 30 branches) | ~10,000 words (MIXER) | no vocabulary: 78-bit input / 48-bit output (FlightBERT++) |
| training objective | **teacher-forced cross-entropy** everywhere in A; **on-policy CAT-K rollout + cross-entropy** in B; **PPO on a learned human-preference reward** (TrafficRLHF), **return-conditioned offline RL, still cross-entropy** (CtRL-Sim), **GRPO on a safety reward** (R1Sim) | the prediction model is trained separately; DTPP adds a **jointly trained learnable cost** | XENT → REINFORCE, annealed (MIXER); on-policy relabelling (DAgger); coin-flip per token (scheduled sampling) | **BCE over bits** (FlightBERT++); **PEFT next-token** (FTP-LLM); **none — prompting only** (FLY-EVAL++) |
| closed-loop evaluation | **yes** for SMART / Trajeglish / GUMP / BehaviorGPT / InfGen / CAT-K / R1Sim / SaFeR (WOSAC 8 s, InfGen 30 s, nuPlan); **no** for MotionLM (autoregressive rollouts scored against GT on the WOMD prediction challenge) | **yes** — nuPlan closed-loop with reactive and non-reactive agents | n/a (greedy generation vs beam search) | **no row is closed-loop.** FLY-EVAL++'s M3 is a **3-step** structured rollout, the longest in aviation here |
| data scale | WOMD **486,995 training scenarios / 574 driving hours / 7.64 M tracks** (GUMP, BehaviorGPT, R1Sim); SMART "2.2M scenarios (or **1B motion tokens**)"; nuScenes **5.5 h** (TrafficRLHF); nuPlan 1,500 h | nuPlan **100k** training scenarios, 200 test scenarios (DTPP) | IWSLT-14 / MSCOCO / Gigaword | **8,643 trajectories** over 9 days (FlightBERT++); **708 GA flight segments** (FLY-EVAL++) |
| goal / constraint term | **none at training** in A and B; a reward at training in C; a **mask at decoding** in E (SaFeR's top-n trust region + Largest Feasible Region) | the **cost function is the goal**, evaluated over the tree | none | **none at training**; FLY-EVAL++ checks constraints **only at evaluation** |
| measured post-training effect | **CAT-K: +0.0111 RMM over the same 7M model and +0.0088 over a 14× larger one; ego collisions −25.7 %, off-road −33.9 %.** TrafficRLHF: failure rate 0.271 → 0.054. R1Sim: +0.0084 RMM over SMART-tiny but **+0.0001 over CAT-K** on the test set | TPP → DTPP closed-loop score 0.7388 → 0.8964 | MIXER over XENT: +3.21 ROUGE-2, +2.99 BLEU-4, +1.36 BLEU-4 | **none measured** — no aviation row post-trains anything |

Four observations the table supports:
- **The pretraining step is settled and the post-training step is not.** Every tokenised model in
  cluster A is trained by teacher-forced cross-entropy and nothing else, and the two that tried
  noise as a substitute for on-policy data — Trajeglish's noisy tokenisation and SMART's trajectory
  perturbation — **both fail to improve** their own benchmark (CAT-K §4.3, and its Tab. 2 rows 3–8).
- **The one intervention with a large, reproduced effect is closed-loop *supervised* fine-tuning,
  not RL.** CAT-K beats scaling the model 14× (§2.1). Of the three RL papers, one fine-tunes
  diffusion/regression models rather than token models (TrafficRLHF), one is offline RL that is
  still cross-entropy (CtRL-Sim), and the one that does GRPO on top of CAT-K adds **+0.0001 RMM**
  on the test split (R1Sim Tab. I).
- **Longer tokens help, up to a point that is still ~1 s.** BehaviorGPT's only clean sweep of
  token length gives REALISM 0.6783 → 0.7273 → 0.7335 for patches of 1, 5 and 10 steps (0.1 s,
  0.5 s, 1 s), with minADE 2.3752 → 1.5599 → 1.5203 (Tab. 2). **(reading)** nothing in this folder
  measures a token longer than 1 s, so our 20–60 s segment remains unattested in either direction,
  exactly as `../manoeuvre_tokens/README.md` §3 item 2 found.
- **Constraints enter at decoding, not at training.** SaFeR is the only motion model here that
  masks tokens for feasibility, and it does so on a *frozen* realism prior: top-n = 20 trust
  region, then a Largest-Feasible-Region filter. Removing the filter raises the collision rate
  (0.761 → 0.827) and collapses the solution rate (0.865 → 0.527) — i.e. the mask is what keeps
  generation *solvable*, and it is bolted on after pretraining.

---

## 1. The table

Cluster letters are used throughout: **A** tokenised next-token motion models, **B** closed-loop
supervised fine-tuning, **C** RL / RLHF fine-tuning, **D** goal-directed decoding, **E** constrained
decoding, **F** exposure bias, **G** aviation.

### 1.1 Cluster A — tokenised motion models trained by next-token prediction

| # | citation | token / vocab / step | training objective | closed-loop eval? | data scale | goal or constraint term |
|---|---|---|---|---|---|---|
| A1 | Seff, Cera, Chen, Ng, Zhou, Nayakanti, Refaat, Al-Rfou, Sapp, *MotionLM: Multi-Agent Motion Forecasting as Language Modeling*, **ICCV 2023**, arXiv:2309.16534. PDF: `../manoeuvre_tokens/papers/MotionLM_Seff2023_multi-agent_motion_forecasting_as_language_modeling.pdf` | axis-aligned `(Δx, Δy)`, uniform bins + Verlet wrapper; **169 = 13²**; **0.5 s** | teacher-forced cross-entropy; the authors explicitly did **not** need to noise the teacher-forced trajectories | **no** — autoregressive joint rollouts, but scored on the WOMD marginal / interactive **prediction** challenge against GT | WOMD "103k 20-second scenarios", "divided into 1.1M examples" of 9 s (§4.1) | none |
| A2 | Philion, Peng, Fidler, *Trajeglish: Traffic Modeling as Next-Token Prediction*, **ICLR 2024**, arXiv:2312.04535. PDF: `../manoeuvre_tokens/papers/Trajeglish_Philion2023_traffic_modeling_as_next-token_prediction.pdf` | state-to-state `(Δposition, Δheading)` template, k-disks; **384**; **0.1 s** | teacher-forced cross-entropy, **plus noisy tokenisation** (a distance-weighted token perturbation with a DAD recovery target) | **yes** — WOMD Sim Agents, "32 rollouts of length 8" s (§4.2) | WOMD | none |
| A3 | Wu, Feng, Gao, Kan, *SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction*, **NeurIPS 2024**, arXiv:2405.15677. PDF: `../manoeuvre_tokens/papers/SMART_Wu2024_scalable_multi-agent_real-time_motion_generation_via_next-token_prediction.pdf` | 0.5 s trajectory segment, k-disks, per agent class; **512 / 1024 / 1024 / 2048**; **0.5 s** | "SMART is trained to minimize the cross entropy between the distribution of the ground truth token label and the predicted distribution"; plus top-k token perturbation | **yes** — WOMD Sim Agents; also zero-shot (train NuPlan, test WOMD) | "2.2M scenarios (or 1B motion tokens under 0.5 s agent motion tokenization)" | none |
| A4 | Yihan Hu, Siqi Chai, Zhening Yang, Jingyu Qian, Kun Li, Wenxin Shao, Haichao Zhang, Wei Xu, Qiang Liu, *Solving Motion Planning Tasks with a Scalable Generative Model* (**GUMP**), **ECCV 2024**, arXiv:2407.02797 | a **key–value token pair per object**: a control token (ID + class) and a state token quantising `(x, y, θ, w, l, vx, vy)` on a meshgrid `[0.2 m, 0.2 m, π/100, 0.5 m, 0.5 m, 0.25 m/s, 0.25 m/s]`; **vocabulary size 2972**; model runs at **2 Hz**, interpolated to 10 Hz | cross-entropy on key and value tokens + an image-reconstruction L1 + an auxiliary MultiPath-style trajectory loss (Eqs. 10–13) | **yes** — WOSAC rollouts, and nuPlan closed-loop planning; GUMP is additionally used as an **online RL training environment** (SAC) for a separate policy | WOMD v1.2.0 "486,995 train"; "7.64 million unique tracks from 574 driving hours across 1750 km"; nuPlan "1,500 hours" | none inside the token model; the RL engine's reward is outside it |
| A5 | Zikang Zhou, Haibo Hu, Xinhong Chen, Jianping Wang, Nan Guan, Kui Wu, Yung-Hui Li, Yu-Kai Huang, Chun Jason Xue, *BehaviorGPT: Smart Agent Simulation for Autonomous Driving with Next-Patch Prediction*, **NeurIPS 2024**, arXiv:2405.17372 | **not discrete**: a *patch* of `ℓ` consecutive states, best `ℓ = 10` (**1 s** at 10 Hz); the head is a mixture (Laplace position/velocity, von Mises yaw) | **negative log-likelihood** with teacher forcing — but "we do not use the ground-truth agent states when updating the RNN's hidden states, intending to train the model to recover from its mistakes" (§3.4) | **yes** — 80 autoregressive steps per agent, up to 128 agents; **1st place, 2024 WOSAC**, 3M parameters | WOMD "486,995/44,097/44,920 training/validation/testing scenarios" | none |
| A6 | Xiuyu Yang, Shuhan Tan, Philipp Krähenbühl, *Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation* (**InfGen**), **ICCV 2025**, arXiv:2506.17213 | motion tokens by k-disks over 0.5 s spans, `\|V_motion\| = 2048`; plus map (1024), position (1849), heading (120) and **control** (4) tokens interleaved in one sequence; **0.5 s** | "We directly train InfGen with the next token prediction objective end-to-end on real data" (§1) | **yes** — 8 s WOSAC *and* **30 s** long-term rollouts with agents entering and leaving | WOMD | none; the control token switches *task* (motion vs scene generation), not feasibility |

### 1.2 Cluster B — closed-loop supervised fine-tuning of a tokenised model (the central item)

| # | citation | how the rollout is guided | training mixture | closed-loop eval? | measured gain over teacher forcing |
|---|---|---|---|---|---|
| B1 | Zhejun Zhang, Peter Karkus, Maximilian Igl, Wenhao Ding, Yuxiao Chen, Boris Ivanovic, Marco Pavone (NVIDIA Research / Stanford), *Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models* (**CAT-K**), **CVPR 2025**, arXiv:2412.05334v2 (14 Mar 2025); code `github.com/NVlabs/catk`; leaderboard name **SMART-tiny-CLSFT** | **Closest Among Top-K.** At each step, for each agent, take the `K` likeliest tokens under the policy, then pick the one whose resulting state is closest to the GT next state (Eq. 2). The target is the **DAD recovery token** — the token from the *whole* vocabulary that brings the rolled-out agent back to the GT (Eq. 4) | **two stages, no mixing**: BC pre-training to convergence (32 epochs), then pure CAT-32 closed-loop fine-tuning (10 epochs) with the cross-entropy loss of Eq. 5; the map encoder is frozen during fine-tuning | **yes** — WOSAC protocol, 32 rollouts per scenario at 10 Hz, 1 s history, 8 s rollout | **leaderboard (test):** RMM 0.7591 → **0.7702** (same 7M SMART-tiny), vs 0.7614 for the 102M SMART-large; minADE 1.4062 → 1.3068. **Ablation (2 % val):** BC pre-train 0.7581, more BC 0.7590, **CAT-32 0.7616**. **Ego GMM planner:** collision 0.0568 → 0.0422, off-road 0.0053 → 0.0035, minADE₃₂ 1.3537 → 0.6912 |

Vocabulary and horizon for B1 (SMART-tiny's, reused): agent vocabulary **2048** tokens, "each token
representing a 0.5-second trajectory at 10Hz"; map vocabulary 1024; the policy "re-plans at 2Hz".
Inference uses top-K sampling with `K_infer = 48`, temperature 1.0, no top-p.

### 1.3 Cluster C — RL / RLHF fine-tuning of trajectory models

| # | citation | what is fine-tuned | reward | closed-loop eval? | data scale | measured effect |
|---|---|---|---|---|---|---|
| C1 | Yulong Cao, Boris Ivanovic, Chaowei Xiao, Marco Pavone, *Reinforcement Learning with Human Feedback for Realistic Traffic Simulation* (**TrafficRLHF**), arXiv:2309.00709 (1 Sep 2023); **no venue on the PDF** | **not a token model**: CTG (conditional diffusion), BITS (bi-level IL) and TrafficGen. Encoder and decoder both, by "an amended PPO from ColossalAI … with 70 epochs" | a **learned human-preference reward model** (CTG encoder + a 512-512-512-128-32 MLP), trained on pairwise comparisons of 5 CTG-generated scenarios per scene | **no closed-loop benchmark**: 100 held-out nuScenes scenes, failure rate + Wasserstein realism deviation | nuScenes, "totaling 5.5 hours"; **500 scenes × 5 scenarios** labelled; 400 train / 100 val preference pairs | under "no collision" guidance, failure rate **0.271 → 0.054** (the abstract's "up to 80%"), realism deviation 0.569 → 0.376, reward cost 13.21 → 3.7. BITS 0.314 → 0.23, TrafficGen 0.39 → 0.22 |
| C2 | Luke Rowe, Roger Girgis, Anthony Gosselin, Bruno Carrez, Florian Golemo, Felix Heide, Liam Paull, Christopher Pal, *CtRL-Sim: Reactive and Controllable Driving Agents with Offline Reinforcement Learning*, **CoRL 2024**, arXiv:2403.19918 | a multi-agent decision-transformer over `(state, return-to-go, action)` tokens; actions **discretised into 20 acceleration × 50 steering = 1000 tokens**, each return-to-go component into 350 bins; 10 Hz | **offline RL by return conditioning**, not policy gradient: factored rewards (goal, vehicle-vehicle collision, vehicle-road-edge) computed in a physics-enhanced Nocturne; control at inference by **exponential tilting** `p(G\|s) exp(κG)`, κ ∈ [−25, 25] | **yes** — 1 s history, 8 s rollout, up to 8 agents, 1000 test scenes | WOMD replayed through Nocturne + Box2D; the long-tail fine-tune uses **3500 CAT scenarios, 30 minutes on 1 A100** | training is still `L = L_action + L_return-to-go + α L_state` (cross-entropy + L2). Positive tilting (κ = 10) gives the best collision and offroad rates among the CtRL-Sim variants |
| C3 | Ziyan Wang, Peng Chen, Ding Li, Chiwei Li, Qichao Zhang, Zhongpu Xia, Guizhen Yu, *Learning Rollout from Sampling: An R1-Style Tokenized Traffic Simulation Model* (**R1Sim**), **IEEE RA-L**, accepted March 2026, DOI `10.1109/LRA.2026.3678842`, arXiv:2603.24989 | **GRPO on top of a pretrained tokenised policy** — SMART-tiny and CAT-K are both used as the reference policy π_ref; entropy-guided adaptive sampling replaces fixed top-K | `r_{i,t} = r^safe_{i,t} · r^dis_{i,t}` with `r^safe = −1` if a Separating-Axis-Theorem collision else `+1`, and `r^dis = exp(−α\|S_{i,t} − y_t\|)` — i.e. **safety × closeness-to-GT**, a process (per-step) reward | **yes** — WOSAC, 32 rollouts | WOMD 486,995/44,097/44,920; **fine-tuned for only 1 epoch** | test split: SMART-tiny 0.7591 → **0.7675** (+0.0084); **CAT-K 0.7687 → 0.7688 (+0.0001)**. On a high-entropy validation subset the gains are +0.0202 over SMART and +0.0096 over CAT-K |

### 1.4 Cluster D — goal-directed decoding: a tree over a learned behaviour model

| # | citation | tree | what scores a branch | closed-loop eval? | measured effect |
|---|---|---|---|---|---|
| D1 | Yuxiao Chen, Peter Karkus, Boris Ivanovic, Xinshuo Weng, Marco Pavone, *Tree-structured Policy Planning with Learned Behavior Models* (**TPP**), **ICRA 2023**, pp. 7902–7908, arXiv:2301.11902, DOI `10.1109/ICRA48891.2023.10161419`. PDF: `../multimodal_intent/papers/TPP_Chen2023_tree-structured_policy_planning_with_learned_behavior_models.pdf` | an **ego trajectory tree** and a **scenario tree**; 2 stages, branching factor 4 | a hand-specified cost; the tree is solved by **dynamic programming** over a finite-horizon MDP whose states are tree nodes, giving a *policy*, not one trajectory | yes (see D2's re-evaluation) | — (the baseline for D2) |
| D2 | Zhiyu Huang, Peter Karkus, Boris Ivanovic, Yuxiao Chen, Marco Pavone, Chen Lv, *DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving*, **ICRA 2024**, arXiv:2310.05885v2 | `T = 8 s` in `N_l = 2` stages, `M ≤ 30` branches: stage 1 is 0–3 s with 3 reference paths and ≤ 30 sampled target states; the **top 5 nodes are kept** and each expanded with 6 targets for stage 2 (3–8 s) | a **learnable, context-aware cost** trained jointly with the ego-conditioned prediction model (max-entropy IRL flavour), replacing TPP's hand-weighted linear cost | **yes** — nuPlan closed-loop, both non-reactive and reactive agents | CL-NR score **TPP 0.7388 → DTPP 0.8964** (PDM, the leaderboard-tuned rule-based planner, 0.9061); collision 0.10 → 0.025. **Pruning without cost learning collapses to 0.3205** |

Goal/anchor-based one-shot predictors (TNT, DenseTNT, MTR) are **not repeated here** — they are
verified with mechanism notes in `../multimodal_intent/README.md` §1.3 and their PDFs are in
`../hierarchical_prediction/papers/`. None of them decodes a sequence of tokens; they pick a goal
and regress a whole trajectory to it.

### 1.5 Cluster E — constrained decoding as feasibility

| # | citation | what is masked | when | measured effect |
|---|---|---|---|---|
| E1 | Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West, *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*, **EMNLP 2023 (Main)**, arXiv:2305.13971. PDF: `../manoeuvre_tokens/papers/GCD_Geng2023_grammar-constrained_decoding_for_structured_nlp_tasks_without_finetuning.pdf` | the whole vocabulary is **pruned to the set an incremental parser allows**, and the allowed set may be **input-dependent** | decoding only, **no finetuning**, any decoding algorithm | mechanism paper; see `../manoeuvre_tokens/README.md` §2.9 |
| E2 | Jinlong Cui, Fenghua Liang, Guo Yang, Chengcheng Tang, Jianxun Cui, *SaFeR: Safety-Critical Scenario Generation for Autonomous Driving Test via Feasibility-Constrained Token Resampling*, arXiv:2603.04071 (4 Mar 2026); **no venue on the PDF** | motion tokens are `(a, ψ)` with `a ∈ [−5, 5] m/s²`, `ψ ∈ [−1.5, 1.5] rad/s` discretised **into 63 × 63 tokens** at `f = 2 Hz`. Decoding is a **two-stage constrained search**: stage 1 keeps the **top-n = 20** tokens of the frozen realism prior ("trust region"); stage 2 picks, inside that set, the token maximising adversarial criticality **subject to the Largest Feasible Region** `{s \| V_h(s) ≤ 0}`, an HJ-reachability set approximated by offline RL | decoding only — the NTP realism prior is pretrained and left alone | **with the LFR mask**: solution rate **0.865**, collision rate 0.761, velocity JSD 0.161. **Without it**: collision rate rises to 0.827 but solution rate falls to **0.527** — the unconstrained generator produces collisions that are theoretically unavoidable. Data: 1000 WOMD + 1000 nuPlan interaction scenarios, closed-loop with a reactive DiffusionPlanner ego |

### 1.6 Cluster F — exposure bias, the reason teacher-forced models fail in closed loop

| # | citation | mechanism | what it costs / buys |
|---|---|---|---|
| F1 | Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (**DAgger**), **AISTATS 2011**, arXiv:1011.0686. PDF: `../hierarchical_prediction/papers/DAgger_Ross2011_reduction_of_imitation_learning_and_structured_prediction_to_no-regret_online_learning.pdf` | Algorithm 3.1: at iteration `i` roll out `π_i = β_i π* + (1 − β_i) π̂_i`, **label the visited states with the expert's action**, aggregate into `D`, retrain. `β_1 = 1`; the parameter-free version is `β_i = I(i = 1)`, "which often performs best in practice" | the bound it removes: plain supervised imitation "can make as many as `T²ε` mistakes in expectation" (Thm 2.1 / §2), where `T` is the horizon. **Requires an expert to query** — which is exactly what CAT-K and DAD replace with the GT trajectory |
| F2 | Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer, *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*, **NeurIPS 2015**, arXiv:1506.03099. PDF: `../manoeuvre_tokens/papers/ScheduledSampling_Bengio2015_scheduled_sampling_for_sequence_prediction_with_recurrent_neural_networks.pdf` | per **token**, feed the true previous token with probability `ε_i`, else the model's own; anneal `ε_i` from 1 to 0. Flipping the coin once per sequence "was much worse" | the canonical cheap fix. **The phrase "exposure bias" does not occur in this paper** (see `../manoeuvre_tokens/README.md` §4) |
| F3 | Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, Wojciech Zaremba, *Sequence Level Training with Recurrent Neural Networks* (**MIXER**), **ICLR 2016**, arXiv:1511.06732v7 | start from the cross-entropy (XENT) optimum, then anneal: use XENT for the first `T − Δ` steps and REINFORCE for the last `Δ` (`Δ` "typically set to two or three"), then `T − 2Δ`, until the whole sequence is REINFORCE | **the paper that names exposure bias.** Greedy generation, XENT → MIXER: summarisation ROUGE-2 **13.01 → 16.22**, translation BLEU-4 **17.74 → 20.73**, captioning BLEU-4 **27.8 → 29.16**; DAD gives 12.18 / 20.12 / 28.16 |

### 1.7 Cluster G — aviation

| # | citation | what is the token | decoding | closed loop / goal term | data scale |
|---|---|---|---|---|---|
| G1 | Dongyue Guo, Zheng Zhang, Zhen Yan, Jianwei Zhang, Yi Lin, *FlightBERT++: A Non-autoregressive Multi-Horizon Flight Trajectory Prediction Framework*, **AAAI** (year unverified, §4). PDF: `../hierarchical_prediction/papers/FlightBERTpp_Guo2023_non-autoregressive_multi-horizon_flight_trajectory_prediction_framework.pdf` | **not a vocabulary**: each attribute is a **Gray-code bit vector** (lon 18 bits, lat 16, alt 11, velocities 11; the *differential* outputs are 8/8/8/9/9/6 bits), so the input is a 78-dim and the output a 48-dim binary vector | **explicitly non-autoregressive**: a horizon-aware context generator emits all 15 future points at once, precisely to avoid "recursive inference" | **no closed loop, no goal term, no constraint term.** The loss is **binary cross-entropy** over bits — a multi-binary-classification task | 8,643 trajectories, 9 days at 20 s intervals; 3-minute history → 5-minute horizon |
| G2 | Kaiwei Luo, Jiliu Zhou, *Large Language Models for Single-Step and Multi-Step Flight Trajectory Prediction* (**FTP-LLM**), arXiv:2501.17459; **no venue on the PDF**. PDF: `../manoeuvre_tokens/papers/LLM-FTP_Luo2025_large_language_models_for_single-step_and_multi-step_flight_trajectory_prediction.pdf` | the **LLM's own BPE sub-tokens of printed digits** ("103.25" → "103", ".", "25") — no codebook, no action space | PEFT fine-tuning of eight open LLMs; prompts of 16 one-minute waypoints | **no** — MAE/RMSE at 1, 4 or 8 minutes; inference latency flagged as the blocker | ADS-B at CAN/PEK/PVG, aggregated to 1-minute steps |
| G3 | Yalun Wu, Junfeng Fang, Jiawei Wang, Haotian Liu, Qijun Yang, Minghan Yang, Hongcheng Guo, Zhoujun Li, Boyang Wang, *FLY-EVAL++: An Evidence-Driven Evaluation Protocol for Safety-Constrained Flight Prediction with Large Language Models*, **COLM 2026**, arXiv:2609.04021 | the LLM's text tokens of a structured flight-state record | **prompting only — nothing is trained.** Three tasks: S1 single step (708 samples), M1 three-state history → next second (504), **M3 three-state history → a three-step future emitted as one structured output** (206) | **constraints are checked at evaluation, never at training**: a deterministic verifier for protocol compliance, physical feasibility and phase-dependent safety corridors (pitch/bank limits, stall margin, stabilised approach) | PilotBench: "708 real-world general-aviation flight segments" across nine FAA-aligned phases; **66 LLMs screened, 21 fully evaluated** |
| G4 | Thaweerath Phisannupawong, Joshua Julian Damanik, Han-Lim Choi, *LLM4Delay: Flight Delay Prediction via Cross-Modality Adaptation of Large Language Models and Aircraft Trajectory Representation*, arXiv:2510.23636v4 (24 Oct 2025); **no venue on the PDF** | **no motion tokens.** A frozen pretrained trajectory encoder produces one **instance-level embedding per trajectory**, projected into the frozen LLM's token space alongside textual aeronautical data | a **regression head** on the final hidden state — the output is a **scalar delay**, not a trajectory | none | OpenSky ADS-B in the Incheon TMA, resampled at 5 s; 12 monthly datasets for 2022 |

**Aviation verdict, stated plainly: no aviation paper in this folder goes beyond imitation.** No
aviation row post-trains a pretrained predictor, none rolls a trajectory out under its own
predictions during *training*, none carries a goal or constraint term in the loss, and the longest
autoregressive aviation rollout that is scored anywhere here is **three steps** (FLY-EVAL++ M3).
The only aviation paper that evaluates constraint satisfaction at all does so on a model that was
never trained for it.

---

## 2. The quotes, per paper, with their location

### 2.1 B1 — CAT-K (Zhang et al., CVPR 2025): how the rollout is guided, and what it buys

**The problem, in the paper's own framing** (§1): tokenised policies "are typically trained through
open-loop behavior cloning, and thus suffer from covariate shift when executed in closed-loop
during simulation" (abstract).

**The rollout rule** (§4.1): CAT-K "deterministically rolls out the policy by selecting, at each
time step and for each agent, the one action among the top-K likeliest according to `π_θ` that
brings the agent closest to the GT next state." Formally (Eq. 2):

```
c_t^i     = argmin_{c ∈ {ξ₁,…,ξ_K}}  d( f(s_t^i, x_c), ŝ_{t+1}^i )
{ξ₁,…,ξ_K} = topK_{c ∈ {1,…,|V|}} [ π(c_t^i | h_t, M) ]
```

`d(·,·)` is "the average Euclidean distance between the four pairs of corners of two bounding
boxes" (§5.1). The rolled-out state `s_{t+1}^i = f(s_t^i, x_{c_t^i})` is fed back as the policy's
input at the next step (Eq. 3) — **this is the on-policy part**.

**The training target** (§4.2, Eq. 4) is DAD's recovery action, searched over the **whole**
vocabulary, not the top-K:

```
ĉ_t^i = argmin_{c ∈ {1,…,|V|}}  d( f(s_t^i, x_c), ŝ_{t+1}^i )
```

and the loss is plain cross-entropy on those targets (Eq. 5).

**The two-stage schedule** (§4.2): "Since CAT-K rollout is effective only when the top-K rollouts
of the policy cover the GT mode, we adopt a two-stage training procedure … First, we obtain a
reasonably well-trained policy through BC pre-training, then fine-tune it using CAT-K rollouts."
Concretely (§5.1): "We run BC pre-training for 32 epochs, each taking 1.7 hour, then continue with
closed-loop supervised fine-tuning with CAT-32 for 10 epochs, each taking 2.6 hours"; "the map
encoder is frozen to save GPU memory". 8× A100 80GB, total batch size 80.

**What `K` means** (§4.3): "CAT-K with `K = |V|` is equivalent to noise-free BC … for `K = 1`,
CAT-K is equivalent to deterministically rolling out the policy by always choosing the most likely
token." So `K` "trades off following the policy (for `K = 1`) vs. following the GT (for
`K = |V|`)". Fig. 5 shows the rollout-to-GT ADE falling towards the **quantisation error** of the
vocabulary as `K` grows and as fine-tuning epochs accumulate.

**Why noise is not enough** (§4.3): "Trajeglish's noisy tokenization does not yield a significant
improvement (see Fig. 9 in [24])" and SMART's perturbation "does not improve performance on WOSAC
itself (see Tab. 4 in [35])". The diagnosis: those strategies "completely ignore what state
distribution would be induced by the learned policy, hence likely oversampling irrelevant states
and undersampling states the learned policy would actually encounter."

**The numbers.** Leaderboard (test split, Tab. 1):

| model | params | RMM ↑ | minADE ↓ |
|---|---|---|---|
| SMART-tiny fine-tuned w. CAT-K (ours) | 7 M | **0.7702** | **1.3068** |
| SMART-large | 102 M | 0.7614 | 1.3728 |
| KiGRAS | 0.7 M | 0.7597 | 1.4384 |
| SMART-tiny | 7 M | 0.7591 | 1.4062 |
| BehaviorGPT | 3 M | 0.7473 | 1.4147 |
| GUMP | 523 M | 0.7431 | 1.6041 |

§5.3.1 adds the control that matters: a large hyper-parameter grid search "allowed us to push the
performance of the BC baseline to 0.7671 RMM on the test split … but still falling significantly
short of the performance of our CAT-K fine-tuning method (0.7702 RMM). … our method improves the
performance more (+0.0031) than scaling up the model size by a factor of 14 to 102M parameters
(+0.0023)."

Ablation on the 2 % validation split (Tab. 2) — same base model, different fine-tuning:

| fine-tuning | RMM ↑ | minADE ↓ |
|---|---|---|
| BC pre-training (no fine-tune) | 0.7581 | 1.3152 |
| more BC fine-tuning | 0.7590 | 1.3039 |
| Trajeglish's noisy tokenisation (K = 5) | 0.7562 | 1.3459 |
| SMART's trajectory perturbation (K = 5) | 0.7556 | 1.3177 |
| Top-5 sampling rollout | 0.6478 | 1.8802 |
| Top-5 + distance filter | 0.6860 | 1.7627 |
| Deterministic (max-prob) rollout | 0.6361 | 1.8695 |
| CAT-5 / CAT-16 / **CAT-32** / CAT-40 / CAT-64 | 0.7423 / 0.7604 / **0.7616** / 0.7617 / 0.7602 | 1.4677 / 1.3372 / 1.3105 / 1.2998 / 1.3028 |

Two readings of that table, both **(reading)**: (i) the closed-loop *rollout* alone is
catastrophic — plain top-5 rollout training costs 0.11 RMM — so it is the **GT-anchored target**,
not the on-policy state distribution, that keeps the supervision valid; (ii) `K` is flat from 16 to
64 and only fails at 5, so the hyper-parameter is not delicate, which is the paper's own claim.

Ego-motion planning with a **GMM** policy (Tab. 3, 2 % val split, 5 fine-tuning epochs) — the
demonstration that the method is not tied to tokens:

| method | collision ↓ | off-road ↓ | RMM ↑ | ADE ↓ | minADE₃₂ ↓ |
|---|---|---|---|---|---|
| BC pre-training | 0.0568 | 0.0053 | 0.8108 | 1.3623 | 1.3537 |
| BC fine-tuning | 0.0599 | 0.0058 | 0.8105 | 1.3520 | 1.3509 |
| Deterministic rollout | 0.0433 | 0.0138 | 0.8081 | **1.1799** | 0.7962 |
| **CAT-3** | **0.0422** | **0.0035** | **0.8169** | 1.3096 | **0.6912** |

Note the trade the authors call out: deterministic-rollout fine-tuning wins ADE by **mode
averaging** — "this reduces ADE, [but] it negatively impacts other metrics" (§5.3.3), and indeed
its off-road rate is 2.6× worse than BC's.

**Conclusion, quoted** (§6): "our results show that closed-loop supervised fine-tuning is a
promising area of future research for policies trained in open-loop, such as the widely used NTP
policies."

### 2.2 C3 — R1Sim (RA-L 2026): RL *after* CAT-K adds almost nothing

- The setup (§V-A): WOMD "containing 486,995/44,097/44,920 training/validation/testing scenarios";
  SMART and CAT-K are "the pretrained" policies; "we fine-tune … R1Sim for only 1 epoch".
- The reward (Eqs. 7–9): `r_{i,t} = r^safe_{i,t} · r^dis_{i,t}` with
  `r^safe = −1 if SAT(C_{i,t}) else 1` and `r^dis = exp(−α|S_{i,t} − y_t|)`. So the "realism" term
  is **distance to the ground truth** — the same anchor CAT-K uses, re-expressed as a reward.
- The gains (Tab. I, test split): SMART-tiny 0.7591 → **0.7675**; **CAT-K 0.7687 → 0.7688**.
- The gains where the authors look for them (Tab. II, validation split by scenario entropy):
  "in the high-entropy scenes, our method achieves improvements of 0.0202 and 0.0096 RMM" over
  SMART (IL) and CAT-K (SFT) respectively.
- Their framing of what SFT lacks (§I): existing methods "often employ winner-takes-all approaches
  that force generated states to match expert demonstrations. However … over-reliance on
  potentially suboptimal ground truth may perpetuate unsafe behaviors."

**(reading)** the +0.0001 test-split gain over CAT-K is the single most decision-relevant number in
this folder: with a pretrained token policy already fine-tuned closed-loop against the truth, a
full GRPO stage bought nothing measurable on the aggregate benchmark, and its visible gains are
confined to a high-entropy subset the authors selected.

### 2.3 C1 — TrafficRLHF (Cao et al. 2023): RLHF, but not on a token model

- The backbone claim (§1): "we employ a state-of-the-art autoregressive backbone model, CTG [5],
  with a preferred roll-out length to optimize the simulation of the near future." **(reading)**
  the same paper describes CTG as "a model predicated upon the principles of conditional diffusion"
  (§4.1) — so "autoregressive" here does not mean a discrete token model, and **this paper is not
  an RLHF fine-tune of a tokenised policy**.
- The preference data (§4.2): "From the nuScenes dataset, we selected a subset of 500 scenes from
  the training split … Each scene was processed through the CTG model to generate five unique
  scenarios", with labellers "identifying the most realistic scenario or indicating if all
  scenarios lacked realism"; 400 training and 100 validation preference units (§4.3).
- The fine-tuning (§C of the appendix): "All the fine-tuning process employs an amended PPO from
  ColossalAI [30] as our fine-tuning strategy with 70 epochs", with `α` (the KL/penalty weight)
  0.1 for CTG, 0.5 for BITS, 0.3 for TrafficGen.
- The effect (Tab. 1, "no collision" guidance): failure rate **0.271 → 0.054**, realism deviation
  0.569 → 0.376. Tab. 2: BITS 0.314 → 0.23, TrafficGen 0.39 → 0.22; reward cost 13.21 → 3.7,
  12.4 → 9.5, 15.2 → 11.3.
- A caution the paper states itself (§4.4): "the realism metrics deteriorate when guidance is
  related to the goal positions/speeds, possibly because the guidance during the diffusion process
  in CTG overpowers the loss provided by the RM penalty losses."

### 2.4 C2 — CtRL-Sim (CoRL 2024): reward at training, but the loss is still cross-entropy

- Tokenisation (Appendix, *CtRL-Sim Training and Inference Details*): "we discretize the
  acceleration and steering into 20 and 50 uniformly quantized bins, respectively, yielding **1000
  action tokens**. For the return-to-gos, we discretize each return-to-go component … into 350
  uniformly quantized bins." Context `H = 32` timesteps, `N = 24` agents.
- The sequence (§2.2): `x = ⟨…, (s_t^i, s_G^i), (G_t^{1,i}, …, G_t^{C,i}), a_t^i, …⟩` — state, then
  factored returns-to-go, then action, per agent per step.
- The objective (§2.2 *Training*): "We train the return-to-go and action headers with the standard
  cross-entropy loss function and the future state sequence header with an L2 regression loss …
  `L = L_action + L_return-to-go + α L_state`." **(reading)** this is teacher-forced supervised
  learning over an offline dataset; the "RL" is entirely in what is conditioned on.
- Control at inference (§2.1): sample returns from the **tilted** distribution
  `G'_t ∼ p_θ(G_t | s_t, s_G) exp(κ G_t)`, "where κ represents" the inverse temperature; "negative
  exponential tilting yields behaviours that are worse than the average behaviours learned from the
  dataset, while positive exponential tilting yields better-than-average behaviours" (§3.2).
- Evaluation (§3.1): "we use 1 second of history and simulate an 8 second future rollout … We
  evaluate on 1000 random test scenes."
- The long-tail fine-tune (§3.2): "CtRL-Sim FT was finetuned in only 30 minutes on 1 NVIDIA
  A100-Large GPU and only 3500 CAT scenarios."

### 2.5 A4 / A5 / A6 — the 2024–2025 successors

**GUMP** (Hu et al., ECCV 2024). Tokenisation (§*Tokenization*): "each object is composed of two
tokens with distinct functionalities, akin to a 'key-value pair': a control token and a state
token"; the state token quantises `(x, y, θ)` plus `(v_x, v_y, w, l)`. Hyper-parameters (Appendix
D): **vocabulary size 2972**, block size 2048, meshgrid
`[0.2 m, 0.2 m, π/100, 0.5 m, 0.5 m, 0.25 m/s, 0.25 m/s]`. Rate (Appendix C): "our model operates in
2Hz, and we interpolate the results to 10Hz for evaluation … we use 1 second of history along with
the current frame as conditions and predict the information for the next 8 seconds." Loss
(Appendix D.3, Eqs. 10–13): reconstruction L1 + cross-entropy over key and value tokens + a
MultiPath-style `L_traj = α L_cls + β L_minADE`. Data (Appendix C): WOMD v1.2.0 "486,995 train";
"7.64 million unique tracks from 574 driving hours across 1750 km urban roadways"; nuPlan
"1,500 hours". GUMP's distinctive claim for this folder is the **online-training module**: the
world model is used as a closed-loop environment in which a separate SAC policy is trained — the
token model itself is never RL-fine-tuned.

**BehaviorGPT** (Zhou et al., NeurIPS 2024). The motivation is exactly our question (§1): "For a
next-token prediction model embedding tokens at 10 Hz, a low training loss can be achieved by
simply copying and pasting the current token as the next one without performing any long-range
interaction reasoning." The fix is the **Next-Patch Prediction Paradigm**: predict a patch of `ℓ`
steps. Crucially the patch is **not quantised** — the head estimates "the position and velocity
components as Laplace distributions and the yaw angle as a von Mises" mixture, trained by NLL
(§3.4). Teacher forcing is used with one deliberate exception: "we utilize teacher forcing to
parallelize the modeling of next-patch prediction and ease the learning difficulty, but we do not
use the ground-truth agent states when updating the RNN's hidden states, intending to train the
model to recover from its mistakes made in next-state prediction" (§3.4). The token-length sweep
(Tab. 2, validation split) is the only one in this folder:

| patch size | replan freq | minADE ↓ | REALISM ↑ | COLLISION ↑ | OFFROAD ↑ |
|---|---|---|---|---|---|
| 1 | 10 Hz | 2.3752 | 0.6783 | 0.9002 | 0.8432 |
| 5 | 2 Hz | 1.5599 | 0.7273 | 0.9181 | 0.9077 |
| 10 | 1 Hz | **1.5203** | **0.7335** | **0.9358** | **0.9132** |

and the paper's own conclusion (§4.3): "using a larger patch indeed helps long-term reasoning, but
a moderate replan frequency is important for temporal stability, which may be neglected by prior
works." The optimal patch is "10, corresponding to 1 second" (§4.2), with **3M parameters** and
first place in the 2024 WOSAC.

**InfGen** (Yang, Tan & Krähenbühl, ICCV 2025). Tokenisation (§4.1): motion tokens follow SMART's
k-disks recipe over "a fixed time span of 0.5 seconds"; the sequence **interleaves** motion, pose,
map and **control** tokens, where the control token "determines which task to execute next"
(motion simulation or scene generation). Vocabulary sizes (Appendix): motion 2048, map 1024,
position 1849, heading 120, control 4. Training (§1): "We directly train InfGen with the next token
prediction objective end-to-end on real data." Results: short-term WOSAC composite **CatK 0.7603 vs
InfGen 0.7514** (Tab. 1) but 30-second composite **SMART-7M 0.6519, CatK 0.6584, InfGen 0.6606**
(Tab. 2). **(reading)** the interesting part for us is not the margin but the fact that the
30-second composite of every model is ~0.09 below its 8-second composite — long-horizon closed-loop
realism degrades for all of them, including the closed-loop-fine-tuned one.

### 2.6 D2 — DTPP (ICRA 2024): the tree is over trajectories, and the cost is learned

- The tree (§IV-B): "The entire planning horizon spans `T = 8` seconds in `N_l = 2` stages with
  maximum `M = 30` branches. In the first stage (short-term, `t¹₀ = 0s`, `t¹_F = 3s`), we consider
  three reference paths and allow for a maximum of 30 sampled target states. After obtaining the
  prediction results, we retain only the top 5 nodes and proceed to expand each node with 6 target
  states for the second stage".
- Why a learned cost (§IV-C): "cost learning is crucial to ensure accurate evaluation of actions
  and effective heuristics for guiding tree expansion"; and the negative control — "Node pruning can
  make performance much worse when cost learning is not included" (0.3205 vs 0.7388, Tab. II).
- Data (§IV-A): "we extract a total of 100k scenarios from the validation subset … and each
  scenario has a future time horizon of 8 seconds"; testing on 200 scenarios of 15 s each.
- Closed-loop scores (Tab. I): TPP 0.7388 / 0.7699 (non-reactive / reactive), **DTPP 0.8964 /
  0.8978**, PDM 0.9061 / 0.9150, IDM 0.6396 / 0.6168, imitation baselines ≈ 0.64–0.66.

**(reading)** DTPP's branches are continuous candidate trajectories generated from reference paths
and sampled target states — **not tokens of a learned vocabulary**. No paper in this folder builds a
tree or MCTS over motion tokens; the tree literature and the token literature have not met.

### 2.7 E2 — SaFeR (2026): the only motion model here that masks tokens for feasibility

- Tokenisation (§III *Motion Tokenization*): "we adopt acceleration `a` and yaw rate `ψ` as motion
  tokens. We discretize `a ∈ [−5, 5] m/s²` and `ψ ∈ [−1.5, 1.5] rad/s` into 63 × 63 [tokens]";
  hyper-parameter table gives `f = 2 Hz` and `n = 20`.
- Stage 1 (§III-D): "To ensure the adversarial vehicle behaves like a human driver, we limit the
  search space to the high-probability manifold of the realism prior model", i.e.
  `W_top−n = {w ∈ V | w ∈ top-n(P_θ(w | s_<t, M))}`. "This truncation guarantees that the selected
  token is inherently realistic."
- Stage 2 (§III-D): inside `W_top−n`, choose the token minimising an adversarial loss that switches
  on the sign of `V_h`, the Hamilton-Jacobi feasibility value — the Largest Feasible Region is
  `{s | V_h(s) ≤ 0}`, "approximating the LFR via offline reinforcement learning" (abstract).
- The ablation (Tab. IV, WOMD): "removing the LFR constraint (w/o LFR) yields the highest Collision
  Rate (0.827) but causes a severe drop in the Solution Rate (0.527)"; the full model reaches
  "the highest SR (0.865) and superior kinematic fidelity (lowest VJ of 0.161 and AJ of 0.499)
  while maintaining a robust adversarial criticality (CR of 0.761)."
- Evaluation (§IV): a dual-stage protocol — log-replay ego to measure collision rate, then a
  reactive DiffusionPlanner ego to measure the **solution rate**, on 1000 WOMD + 1000 nuPlan
  interaction scenarios.

**(reading)** SaFeR is the motion-model analogue of grammar-constrained decoding (E1): the
constraint is a mask over the vocabulary applied to a frozen prior at decode time, with no
finetuning — and its measured contribution is not accuracy but **solvability**.

### 2.8 F — the exposure-bias line, in the papers' own words

- **MIXER** names it (§1): "This process is very brittle because the model was trained on a
  different distribution of inputs … As a result the errors made along the way will quickly
  accumulate. We refer to this discrepancy as **exposure bias**."
- **MIXER's schedule** (§3.2.2): "we start from the optimal policy and then slowly deviate from it
  to let the model explore and make use of its own predictions"; XENT for the first `T − Δ` steps
  and REINFORCE for the last `Δ`, "In our experiments `Δ` is typically set to two or three", then
  `T − 2Δ`, "until only REINFORCE is used to train the whole sequence."
- **MIXER's lineage, stated by the authors** (§3.2.2): "The MIXER algorithm borrows ideas both from
  DAGGER (Ross et al., 2011) and DAD (Venkatraman et al., 2015; Bengio et al., 2015)". **(reading)**
  that is the same pair CAT-K builds on — CAT-K is DAD with a policy-aware rollout, and MIXER is
  DAD/DAgger with a sequence-level reward.
- **DAgger's bound** (§2): plain supervised imitation "can make as many as `T²ε` mistakes in
  expectation", while DAgger's guarantee scales "nearly linearly with the effective horizon".
  Its cost is the expert: Algorithm 3.1 needs `D_i = {(s, π*(s))}` — expert labels at
  *policy-visited* states. **(reading)** CAT-K's contribution is precisely to manufacture that
  label without an expert, by asking which token returns the agent to the recorded trajectory.

### 2.9 G1 / G3 — the aviation readings this folder needed

**FlightBERT++**, read for *this* folder's question (not the horizon question that
`../prediction_horizons/` §2.6 asked of it):

- What the "token" is (§IV *Experimental settings*): "we use 18 and 16 bits to encode the real
  values (decimals) of longitude and latitude into GC representation for the inputs … the input of
  the proposed framework is a 78-dimensional vector while the output is a 48-dimensional vector."
  There is **no vocabulary and no codebook** — each output bit is an independent binary decision.
- The objective (§III-G): "the BCE loss function is employed to optimize the network parameters",
  the task being framed as multi-binary classification (MBC).
- Decoding (§III): "the proposed framework can generate multihorizon predictions directly
  (non-autoregressive) rather than perform recursive inference." The stated reason is error
  accumulation — i.e. FlightBERT++ **avoids** the autoregressive setting rather than fixing it.
- Goal/constraint: **none**. Grep of the fetched text finds no constraint, feasibility or goal term
  in the loss; the entire objective is Eq. (9)–(10)'s BCE.
- Data (§IV-A): "a total of 8643 flight trajectories"; 9 days at 20 s; train on days 1–7,
  validate on day 8, test on day 9.

**FLY-EVAL++** (COLM 2026), the one 2026 aviation paper that measures what we care about:

- The tasks (§3.3): "S1 (single-step prediction, 708 samples)"; "M1 (history-conditioned one-step
  prediction, 504 samples): Given a short history window of three consecutive flight states, the
  model predicts the next-second structured flight state"; "M3 (history-conditioned multi-step
  rollout, 206 samples) … the model predicts a coherent three-step future trajectory as a single
  structured output."
- The finding (abstract): "safety compliance is the most discriminative dimension of model
  behavior: models with comparable predictive performance differ by more than 28 points in safety
  score, and we observe recurrent failures including safety violations under physically plausible
  predictions and instability in multi-step rollouts."
- Where the discrimination lives (§5.2): "M1 is near-saturated (std = 0.53, all models >94.76%) …
  M3 restores discrimination (std = 1.23) because models must generate a coherent multi-step
  trajectory; D4 Safety ranges from 60.2% to 75.7%". And the orthogonality: "Kimi-K2 and
  DeepSeek-V3.1 rank high on D3 Physics (≥97%) but lowest on D4 Safety (60.2%), whereas Claude-4.5
  shows the opposite pattern (Physics 89.8%, Safety 75.7%)."
- The horizon probe (§5.4): "rolling h-step prediction (h ∈ {1, …, 5} s) reveals degradation beyond
  t+4s".

**(reading)** FLY-EVAL++ measures, on aviation data, exactly the failure our design is worried
about — a rollout that stays *physically* plausible while leaving the *operational* corridor — and
it finds the two scores are only weakly correlated (Protocol–Safety ρ = 0.23, Physics–Safety
ρ = 0.31 on S1). It does nothing to fix it: there is no training stage in the paper at all.

---

## 3. What this says for the two-tier plan

Our design, as it stands in `4dTrajectory/ts_transformer/docs/`: the **executor** is trained
one-shot on truth histories and iterated autoregressively **only at evaluation**; the **prior** is
teacher-forced on truth code sequences; only the prior has a planned closed-loop round; the codes
are **K16 FSQ over 20–60 s segments**. Mapping each mechanism onto that, with nothing asserted
beyond §1–§2:

1. **The executor is the configuration every paper in cluster B says fails, and it is the one
   configuration none of them ships.** CAT-K's whole premise — "trained through open-loop behavior
   cloning, and thus suffer from covariate shift when executed in closed-loop" — describes our
   executor exactly: trained on truth histories, iterated on its own output at evaluation. The
   measured size of that gap in the closest analogue: deterministic self-rollout of a BC-trained
   token policy scores **0.6361 RMM against 0.7581 for the same model measured open-loop-trained**
   (CAT-K Tab. 2, rows 1 and 12), and the fix recovers to 0.7616. **(reading)** our two-tier
   evaluation ADE is being read off a model in the "deterministic rollout" row's regime.
2. **The cheapest correct fix for the executor is CAT-K, not RL, and our setting has the one
   ingredient it needs.** CAT-K needs three things: a pretrained policy, a deterministic forward
   dynamics `f(s, a)` queryable during training, and a ground-truth trajectory. We have all three
   — the executor's rollout is a differentiable point-mass integration and the arrival manifest is
   the GT. The adaptation is mechanical: replace "the token among the top-K likeliest that lands
   closest to `ŝ_{t+1}`" with "the **control** among the executor's own K most likely that lands
   closest to the next observed state", and supervise with the whole-vocabulary recovery target.
   **(reading)** for a continuous-output executor the GMM experiment (Tab. 3) is the relevant
   precedent, not the NTP one: CAT-3 on a 16-mode GMM cut collisions 25.7 % and off-road 33.9 %
   while **halving** minADE₃₂ (1.3537 → 0.6912).
3. **Do not expect a second, RL round to add anything on top.** R1Sim ran GRPO with a
   safety × closeness reward on a CAT-K-fine-tuned policy for a full epoch and moved the test-split
   RMM by **+0.0001**. Its reward's "realism" term is `exp(−α|S − y|)` — distance to the truth,
   i.e. the same signal CAT-K already uses, expressed more expensively. **(reading)** if we ever
   want an RL stage, the case for it has to rest on a term the truth distance cannot express (a
   separation constraint, a CTA), not on realism.
4. **The prior's planned closed-loop round is the well-attested half, and its design question is
   `K`, not whether.** CAT-K's `K` sweep is flat from 16 to 64 and only breaks at 5 (Tab. 2). Our
   prior's vocabulary is **16 codes**, so `K = |V|` — the whole vocabulary — is `16`, which in
   CAT-K's terms is *noise-free BC*, not a closed loop. **(reading)** a K16 vocabulary is too small
   for CAT-K's `K` to have any room: the interesting range `5 < K < |V|` is `K ∈ {2, …, 15}`, and
   CAT-5 on a 2048-token vocabulary was the one setting that **lost** 0.016 RMM. This is a concrete
   reason to sweep `K` against vocabulary size before assuming the prior's closed-loop round
   transfers.
5. **Our 20–60 s token is still off the end of every measured curve, but the curve points our
   way.** BehaviorGPT is the only clean sweep: 0.1 s → 0.5 s → 1 s improves REALISM monotonically
   (0.6783 → 0.7273 → 0.7335) and minADE by 36 %. Nothing here measures beyond 1 s. **(reading)**
   the finding that survives is BehaviorGPT's caveat, not its headline: "a moderate replan frequency
   is important for temporal stability" — a 60 s token with a 60 s replan interval is 60× coarser
   than the coarsest replan any of these papers found acceptable, which argues for decoupling the
   token length from the re-plan interval (as our 20 s re-plan already does).
6. **Feasibility belongs at decoding, on a frozen prior, and its payoff is solvability not
   accuracy.** SaFeR is the template: keep the learned prior's top-n as a "trust region" so the
   output stays in-distribution, then filter that set by an externally computed feasible set. Its
   ablation is the argument for putting the procedure/separation constraint *there* rather than in
   the loss: without the filter the generator produced *more* collisions but its scenarios became
   unsolvable (SR 0.865 → 0.527). **(reading)** the direct analogue is masking the prior's 16 codes
   against the LPV corridor / glidepath window at decode time — which is also what
   `../manoeuvre_tokens/README.md` §3 item 6 concluded from grammar-constrained decoding, now with a motion-model
   precedent and a measured cost of omission.
7. **Nothing in aviation does any of this, which is where the contribution sits.** G1–G4: one model
   avoids autoregression entirely (FlightBERT++), one tokenises printed digits (FTP-LLM), one
   predicts a scalar (LLM4Delay), and one only *evaluates* constraint satisfaction, on models that
   were never trained for it (FLY-EVAL++). **(reading)** a closed-loop supervised fine-tuning stage
   on an arrival executor, judged to the threshold, would be the first of its kind in this
   literature — and the honest framing of any comparison is that the published aviation numbers are
   open-loop scores on a different task (the same gap `../prediction_horizons/` §3.4 recorded).

---

## 4. Not verified

Everything below is either unreachable primary text, a claim the fetched text does not state, or a
citation that could not be pinned down. Nothing from these items is used in §1–§3.

1. **FlightBERT (the original).** No arXiv record exists: `ti:"FlightBERT"` on the arXiv API
   returns **zero hits** (checked 2026-09-20). The binary-encoding representation is described
   second-hand inside FlightBERT++ (§II, §III-B), which is what §1 G1 and §2.9 quote. The venue and
   year of the original remain **unverified**, as already recorded in
   `../prediction_horizons/README.md` §4 item 4.
2. **The AAAI year of FlightBERT++.** Unchanged from `../prediction_horizons/` §4: the arXiv record
   carries no `journal_ref`, its comment refers to "the AAAI version", and FlightPatchNet cites it
   as "[Guo et al., 2024]". Cited here without a year.
3. **TrafficRLHF's venue.** arXiv:2309.00709 has **no `journal_ref`, no DOI and a bare
   "9 pages, 4 figures" comment**. The brief's "ICRA 2024" could not be confirmed from the record
   or the PDF, so §1 C1 cites the arXiv record only.
4. **SaFeR's and R1Sim's peer-review status.** SaFeR (arXiv:2603.04071) has no venue, no DOI and no
   comment on its arXiv record. R1Sim carries DOI `10.1109/LRA.2026.3678842` and a
   "Manuscript received … Accepted: March 5, 2026" line on page 1 (RA-L), which is what §1 C3
   cites; the DOI itself was **not resolved** in this session.
5. **LLM4Delay's venue.** arXiv:2510.23636v4 carries no `journal_ref`, no DOI and no comment.
6. **WOMD's training-split size as used by CAT-K.** CAT-K states the **validation split (44,097
   scenarios)** and **test split (44,920)** only (Appendix). The 486,995 training figure quoted in
   §1 comes from GUMP, BehaviorGPT and R1Sim, which name it explicitly; CAT-K's own training-set
   size is **not stated in the fetched text**.
7. **SMART's Tab. 4 and Trajeglish's Fig. 9.** CAT-K's claim that both papers' noising fails to
   improve their own benchmark is quoted from **CAT-K §4.3**, which points at those figures. The
   Trajeglish text fetched here says the opposite in direction if not in size — "noising the tokens
   during training improves rollout performance slightly in the full control setting" (§4.2, Fig. 9
   discussion). **The two statements are not reconciled here**, and §1 A2 reports both. SMART's
   Tab. 4 was not re-read.
8. **MotionLM's closed-loop status.** §1 A1 marks it "no" because the evaluation sections describe
   the WOMD marginal and interactive **prediction** challenges scored by mAP/minADE against GT, with
   rollout aggregation by k-means. The paper does perform autoregressive rollouts; whether a WOSAC
   submission exists elsewhere was **not checked**.
9. **CtRL-Sim's token time step.** The 10 Hz figure is read from the Nocturne appendix ("tracks its
   9 second trajectory from the Waymo Open Motion Dataset at 10 Hz"); the decoder's own step is not
   separately stated, so "10 Hz" for the action token is **(reading)**.
10. **MCTS over motion tokens.** Searched and **not found**. arXiv queries run 2026-09-20:
    `abs:"motion tokens" AND abs:"mask" AND abs:"infeasible"` (0 hits);
    `abs:"constrained decoding" AND (abs:"trajectory" OR abs:"motion")`;
    `abs:"kinematically feasible" AND abs:"token"`;
    `abs:"next-token" AND abs:"traffic simulation" AND abs:"reinforcement learning"`;
    `abs:"closed-loop" AND abs:"next token prediction" AND abs:"driving"`;
    `abs:"tokenized" AND abs:"motion" AND abs:"fine-tuning"`. No paper building a tree search or
    MCTS **over a learned motion-token vocabulary** came back. Absence of evidence only.
11. **Two candidates found but not fetched or read**, listed so they are not re-discovered:
    *On Learning Closed-Loop Probabilistic Multi-Agent Simulator*, arXiv:2508.00384 (IROS 2025);
    *Constrained Decoding for Safe Robot Navigation Foundation Models*, arXiv:2509.01728. Neither
    is cited in §1–§3.
12. **No paper in §1 was read in full.** Only the tokenisation / objective / experimental-setup /
    results-table sections were extracted. Claims about what a paper does *not* do (no goal term,
    no closed loop) are therefore about those sections plus a grep of the whole extracted text, and
    are stated as such.

---

## 5. Layout

| file | what |
|---|---|
| `README.md` | this index |
| `download.sh` | re-fetches the 12 PDFs new to this repository; lists the 8 cited by path in sibling folders |
| `papers/GUMP_Hu2024_….pdf` | A4, ECCV 2024, arXiv:2407.02797 |
| `papers/BehaviorGPT_Zhou2024_….pdf` | A5, NeurIPS 2024, arXiv:2405.17372 |
| `papers/InfGen_Yang2025_….pdf` | A6, ICCV 2025, arXiv:2506.17213 |
| `papers/CATK_Zhang2025_closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.pdf` | **B1, the central item**, CVPR 2025, arXiv:2412.05334 |
| `papers/TrafficRLHF_Cao2023_….pdf` | C1, arXiv:2309.00709 |
| `papers/CtRLSim_Rowe2024_….pdf` | C2, CoRL 2024, arXiv:2403.19918 |
| `papers/R1Sim_Wang2026_….pdf` | C3, IEEE RA-L 2026, arXiv:2603.24989 |
| `papers/DTPP_Huang2024_….pdf` | D2, ICRA 2024, arXiv:2310.05885 |
| `papers/SaFeR_Cui2026_….pdf` | E2, arXiv:2603.04071 |
| `papers/MIXER_Ranzato2016_sequence_level_training_with_recurrent_neural_networks.pdf` | F3, ICLR 2016, arXiv:1511.06732 |
| `papers/FLYEVALpp_Wu2026_….pdf` | G3, COLM 2026, arXiv:2609.04021 |
| `papers/LLM4Delay_Phisannupawong2025_….pdf` | G4, arXiv:2510.23636 |

Cited by path, not duplicated: `../manoeuvre_tokens/papers/` (MotionLM A1, Trajeglish A2, SMART A3,
GCD E1, scheduled sampling F2, FTP-LLM G2); `../hierarchical_prediction/papers/` (DAgger F1,
FlightBERT++ G1, and TNT / DenseTNT / MTR referenced in §1.4);
`../multimodal_intent/papers/` (TPP D1).
