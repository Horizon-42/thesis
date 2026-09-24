# Multi-aircraft interaction for the scene prior — reading list (2026-09-24)

Why this folder exists. The thesis models arrival control as a language: every **2 s** a causal
Transformer **prior** says six instruction words to each arriving aircraft (runway, approach
clearance, heading, altitude, descent angle, speed), and a rule-based **executor** flies those words
through point-mass dynamics. The prior is single-aircraft today; the plan is a **scene prior** over
the arrivals present at each step:

- **architecture** — factorised attention: causal attention over time within each aircraft, plus
  attention across the aircraft present at each step; relative-geometry **edge features** (same or
  parallel runway, along-track gap to the threshold, relative position, closure rate) enter as
  **attention biases**;
- **pretraining** — joint autoregressive teacher forcing on recorded scenes, all aircraft speaking
  **simultaneously** within a step;
- **post-training** — closed loop: the executor flies every aircraft, with filters or rewards for no
  loss of separation (FAA 7110.65 radar and wake minima, see `../arrival_separation/`), landing, and
  staying close to the data; separation **masks** at decode time.

Measured scene size (brief, 2026-09-24): inside a 25 km ring there are median **1–2**, p90 **2–4**
and at most **5–8** other arrivals when an aircraft enters; most sequencing happens before 25 km.

The questions this folder answers from primary text:

> **(1)** How does agent interaction enter the model — graph message passing, attention with edge
> features or biases, or hand-built neighbour features? Is a separate GNN ever better than
> agent-attention on small fully-connected scenes?
>
> **(2)** Is decoding joint or marginal within one time step, and what does each cost?
>
> **(3)** How is interaction actually *learned* — teacher forcing only, or closed-loop training
> (collision/separation losses, closed-loop SFT, RL)? What did each paper measure about the gain?
>
> **(4)** Aviation: multi-aircraft / terminal-area interaction models, and models of controller
> instructions with traffic context.

How to read it:
- Every number was read from the fetched PDF, not the abstract. **"p."** is the page of the PDF file
  as fetched (for Groot et al., PDF pp. 1–2 are the TU Delft cover pages; for Social LSTM the printed
  CVPR page is PDF page + 960). Quotes are copied from `pdftotext` output, cut with "…".
- **(reading)** marks my inference — a derived percentage, an analogy to our design, a judgement of
  what a result transfers to. Everything not so marked is the source's own statement or number.
- Sibling folders already extract most of the tokenised-motion literature; this README repeats a fact
  from them only where it answers (1)–(4), and then points to the sibling note.

The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`). `./download.sh` re-fetches the
**14** papers new to this repository (run 2026-09-24: all 14 returned; test: two deleted PDFs — one
arXiv, one repository mirror — came back **byte-identical by md5**). **Twenty-one** further sources
already live in sibling folders and are cited by path, not duplicated — including two the brief asked
to fetch (**Trajectron++** is in `../control_normalization/papers/`, **CAT-K** in
`../trajectory_as_language/papers/`). §5 lists what was found but not fetched.

---

## At a glance (only what §2 supports)

| question | what the sources say | strongest single numbers |
|---|---|---|
| (1) how interaction enters | Almost every learned model here from 2021 on uses **attention over agents** (exceptions: TrafficSim's message passing, Jung's pairwise GMM); the differences are *which* agents (all, k-nearest, radius 50 m, distance threshold) and *how the pair geometry enters* (scalar logit bias, embedding added to key **and value**, keys/values rotated into the query frame). Hand-built neighbour features (occupancy grid, N-closest in the ego frame) survive in strong models. **No paper found compares a separate GNN with agent-attention on a small fully-connected scene and finds the GNN better**; on a complete graph GAT *is* additive self-attention without a mask. | MTR++: relative (query-centric) position encoding in Q/K **and V** mAP **0.3754** vs none 0.3513 vs global 0.3463 vs Q/K only 0.3658 (Tab. 10). Groot: relative vs absolute state, intrusions **0.40 vs 4.85** (FNN) and **0.77 vs 15.52** (transformer). AgentFormer: Transformer social 0.26/0.47 vs GCN social 0.28/0.50 (ETH/UCY). |
| (2) joint vs marginal within a step | The next-token traffic models (MotionLM, SMART, SMART-R1, Peng) decode all agents **in parallel**, conditionally independent given all earlier steps. Trajeglish decodes them **sequentially** in a random order; the gain is measurable but "weak in general" and shrinks with context. Across steps, **interaction frequency** matters: joint > marginal rollouts. | MotionLM: 0.125 Hz → 2 Hz interactive attention, single-replica mAP **0.1558 → 0.1687**; overlap marginal **0.0404** vs joint **0.0292** (Tabs. 3, 5). Trajeglish: last-in-order agents' collision rate "drops significantly" under partial control (p.18). |
| (3) how interaction is learned | Teacher forcing alone often learns **little** interaction (Makansi: neighbours contribute ≈ 0 on ETH-UCY, SDD, nuScenes). The large, reproduced gains come from **closing the loop** against the recorded data; explicit collision terms add a smaller increment, and RL on top of closed-loop SFT adds little. Hand-made safety rewards can *lower* realism. | TrafficSim scenario collision **5.92 % → 1.28 %** from closed-loop training, **0.60 → 0.50 %** from adding the collision loss (Tab. 2). Peng: WOSAC collision likelihood 0.544 → **0.834** with λ_coll = 0, **0.863** with λ = 2 (Tabs. 1–2). R1Sim on CAT-K: collision 0.9707 → 0.9713, RMM +0.0001. |
| (4) aviation | Terminal-area interaction models exist but are **open-loop predictors** (MAIFormer, MALTP, DA-STGCN, TrajAirNet) or a **pairwise statistical generator** (Jung). Controller-action learners with traffic context are **RL agents in simplified en-route simulators** (Brittain, Groot, Carvell). **None trains a multi-aircraft instruction model on recorded data, and none closes the loop on real terminal data.** | MAIFormer: single-agent FlightBERT++ better at 6 s (alt MAE 25.4 vs 29.2 ft), multi-agent better at 2 min (287.8 vs 164.2 ft). Jung: pairwise model loss of separation **482 vs 781** per 1,000 three-aircraft sets. |

---

## 1. Sources

### 1.1 Fetched here (`papers/`, 14 PDFs)

All arXiv identifiers were checked against the arXiv API record (title, authors, date) on 2026-09-24;
venues are from the PDF header, the arXiv comment/DOI field, or (marked) a publisher index.

| key / file | citation | cluster |
|---|---|---|
| `SocialLSTM_Alahi2016…` | Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, Silvio Savarese, *Social LSTM: Human Trajectory Prediction in Crowded Spaces*, **CVPR 2016**, pp. 961–971. **No arXiv record**; CVF open-access copy | (1) |
| `GAT_Velickovic2018…` | Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio, *Graph Attention Networks*, **ICLR 2018**, arXiv:1710.10903 | (1) |
| `YouMostlyWalkAlone_Makansi2022…` | Osama Makansi, Julius von Kügelgen, Francesco Locatello, Peter Gehler, Dominik Janzing, Thomas Brox, Bernhard Schölkopf, *You Mostly Walk Alone: Analyzing Feature Attribution in Trajectory Prediction*, **ICLR 2022** (venue from ML Anthology; the arXiv v1 PDF read here says "Under review"), arXiv:2110.05304 | (1)(3) |
| `TrafficSim_Suo2021…` | Simon Suo, Sebastian Regalado, Sergio Casas, Raquel Urtasun, *TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors*, **CVPR 2021**, pp. 10400–10409 (venue from the author page; arXiv PDF has none), arXiv:2101.06557 | (2)(3) |
| `WOSAC_Montali2023…` | Nico Montali, John Lambert, Paul Mougin, Alex Kuefler, Nick Rhinehart, Michelle Li, Cole Gulino, Tristan Emrich, Zoey Yang, Shimon Whiteson, Brandyn White, Dragomir Anguelov, *The Waymo Open Sim Agents Challenge*, **NeurIPS 2023 Datasets & Benchmarks**, arXiv:2305.12032 (v4, V1-leaderboard numbers) | (2)(3) |
| `RLFT_Peng2024…` | Zhenghao Peng, Wenjie Luo, Yiren Lu, Tianyi Shen, Cole Gulino, Ari Seff, Justin Fu, *Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving*, **ECCV 2024** (LNCS, DOI `10.1007/978-3-031-72698-9_10`, from Springer), arXiv:2409.18343 | (2)(3) |
| `SMART-R1_Pei2025…` | Muleilan Pei, Shaoshuai Shi, Shaojie Shen, *Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning* (SMART-R1), **ICLR 2026**, arXiv:2509.23993 | (3) |
| `RLFTSim_Ahmadi2026…` | Ehsan Ahmadi, Hunter Schofield, Behzad Khamidehi, Fazel Arasteh, Jinjun Shan, Lili Mou, Dongfeng Bai, Kasra Rezaee, *RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning*, **CVPR 2026** (Highlight, per arXiv comment), arXiv:2605.19033 | (3) |
| `MALTP_Kim2025…` | Kyungmin Kim, Seokbin Yoon, Keumjin Lee, *Probabilistic Multi-Agent Aircraft Landing Time Prediction*, **AIAA SciTech 2026** (per arXiv comment), arXiv:2512.08281 | (4) |
| `DA-STGCN_Kuang2025…` | Yuheng Kuang, Zhengning Wang, Jianping Zhang, Zhenyu Shi, Yuding Zhang, *DA-STGCN: 4D Trajectory Prediction Based on Spatiotemporal Feature Extraction*, arXiv:2503.04823; **no venue** on the PDF or record | (1)(4) |
| `PairwiseTerminal_Jung2023…` | Soyeon Jung, Amelia Hardy, Mykel J. Kochenderfer, *Inferring Traffic Models in Terminal Airspace from Flight Tracks and Procedures*, arXiv:2303.09981 (v3, 27 May 2025); **no venue** on the PDF | (1)(4) |
| `D2MAV-A_Brittain2020…` | Marc Brittain, Xuxi Yang, Peng Wei, *A Deep Multi-Agent Reinforcement Learning Approach to Autonomous Separation Assurance*, arXiv:2003.08353. A journal article by the same authors, *Autonomous Separation Assurance with Deep Multi-Agent Reinforcement Learning*, J. Aerospace Information Systems 18(12) 890–905 (2021), DOI `10.2514/1.I010973`, exists; that it is the published form of this preprint is **not verified** against the journal text | (1)(3)(4) |
| `RelStateTransformer_Groot2022…` | D. J. Groot, Joost Ellerbroek, J. M. Hoekstra, *Using Relative State Transformer Models for Multi-Agent Reinforcement Learning in Air Traffic Control*, **SESAR Innovation Days 2022**, pp. 1–9. **No arXiv record**; TU Delft repository, final published version | (1)(4) |
| `ActionStacking_Carvell2026…` | Ben Carvell, George De Ath, Eseoghene Benjamin, Richard Everson, *Online Action-Stacking Improves Reinforcement Learning Performance for Air Traffic Control*, arXiv:2601.04287, AIAA conference paper DOI `10.2514/6.2026-2746` (from the arXiv record; the PDF names no venue) | (4) |

### 1.2 Cited by path, not duplicated (21 PDFs in sibling folders)

| source | path (relative to this folder) | used here for |
|---|---|---|
| MotionLM (Seff et al., ICCV 2023, arXiv:2309.16534) | `../manoeuvre_tokens/papers/MotionLM_Seff2023_…pdf` | (2) parallel intra-step decoding, interaction frequency |
| Trajeglish (Philion et al., ICLR 2024, arXiv:2312.04535) | `../manoeuvre_tokens/papers/Trajeglish_Philion2023_…pdf` | (2) sequential intra-step decoding |
| SMART (Wu et al., NeurIPS 2024, arXiv:2405.15677) | `../manoeuvre_tokens/papers/SMART_Wu2024_…pdf` | (1) factorised attention with relative embeddings, 50 m radius |
| Graphormer (Ying et al., NeurIPS 2021, arXiv:2106.05234) | `../manoeuvre_tokens/papers/Graphormer_Ying2021_…pdf` | (1) edge feature as additive logit bias |
| Grammar-constrained decoding (Geng et al., EMNLP 2023, arXiv:2305.13971) | `../manoeuvre_tokens/papers/GCD_Geng2023_…pdf` | (3) constraint = decode-time mask |
| SceneTransformer (Ngiam et al., ICLR 2022, arXiv:2106.08417) | `../hierarchical_prediction/papers/SceneTransformer_Ngiam2021_…pdf` | (1) factorised agent/time attention; (2) joint vs marginal loss |
| AgentFormer (Yuan et al., ICCV 2021, arXiv:2103.14023) | `../hierarchical_prediction/papers/AgentFormer_Yuan2021_…pdf` | (1) GCN vs Transformer social model; distance mask |
| Wayformer (Nayakanti et al., ICRA 2023, arXiv:2207.05844) | `../hierarchical_prediction/papers/Wayformer_Nayakanti2022_…pdf` | (1) interaction as a hand-built modality |
| MTR (Shi et al., NeurIPS 2022, arXiv:2209.13508) | `../hierarchical_prediction/papers/MTR_Shi2022_…pdf` | (1) local attention, dense future prediction |
| MTR++ (Shi et al., TPAMI, arXiv:2306.17770) | `../hierarchical_prediction/papers/MTRpp_Shi2023_…pdf` | (1) relative-position ablation; (2) cross-agent query interaction |
| QCNet (Zhou et al., CVPR 2023) | `../hierarchical_prediction/papers/QCNet_Zhou2023_…pdf` | (1) relative embedding into keys/values; cost |
| TrajAirNet (Patrikar et al., ICRA 2022, arXiv:2109.15158) | `../hierarchical_prediction/papers/TrajAirNet_Patrikar2021_…pdf` | (4) GAT social module |
| DAgger (Ross et al., AISTATS 2011, arXiv:1011.0686) | `../hierarchical_prediction/papers/DAgger_Ross2011_…pdf` | (3) background |
| FlightBERT + spoken instructions (Guo et al., Nat. Commun. 2024, arXiv:2305.01661) | `../hierarchical_prediction/papers/SpokenInstructions_Guo2023_…pdf` | (4) controller instructions as input |
| DESIRE (Lee et al., CVPR 2017) | `../multimodal_intent/papers/DESIRE_Lee2017_…pdf` | (1) log-polar social pooling; interaction hurting on small data |
| Trajectron++ (Salzmann et al., ECCV 2020, arXiv:2001.03093) — **asked for in the brief** | `../control_normalization/papers/Trajectron++_Salzmann2020_…pdf` | (1) typed-edge encoding |
| CAT-K (Zhang et al., CVPR 2025, arXiv:2412.05334) — **asked for in the brief** | `../trajectory_as_language/papers/CATK_Zhang2025_…pdf` | (3) closed-loop SFT |
| R1Sim (Wang et al., IEEE RA-L 2026, arXiv:2603.24989) | `../trajectory_as_language/papers/R1Sim_Wang2026_…pdf` | (3) GRPO with a collision reward on top of CAT-K |
| MAIFormer (Yoon & Lee, IEEE T-ITS 2026, arXiv:2509.21004) | `../prediction_horizons/papers/MAIFormer_YoonLee2026_…pdf` | (4) multi-aircraft attention at Incheon |
| ASCENT (Prutsch et al., ICRA 2026, arXiv:2603.16550) | `../procedure_hard_constraints/papers/prutsch2026_ascent_terminal_tp.pdf` | (4) single-aircraft model beating a GAT social model |
| Phased flight TP (Zhang & Chen, arXiv:2203.09033) | `../procedure_hard_constraints/papers/phased_flight_tp_2203.09033.pdf` | (4) structural-RNN over approaching aircraft |

---

## 2. Per-paper notes

### 2.A How interaction enters the model — question (1)

**Social LSTM — Alahi et al., CVPR 2016** *(fetched)*
- **Mechanism: hand-built spatial pooling of neighbours' hidden states.** One LSTM per person, weights
  shared; at each step the hidden states of neighbours are summed into an `N_o × N_o × D` grid
  centred on the person, `H_t^i(m,n,:) = Σ_{j∈N_i} 1_mn[x_t^j − x_t^i, y_t^j − y_t^i] h_{t−1}^j`
  (Eq. 1, p.4), embedded and fed to the next LSTM step; `N_o = 32`, 8×8 sum pooling (p.5). The
  simpler **O-LSTM** pools only the neighbours' *occupancy* (coordinates), no hidden states (Eq. 5,
  p.4). Observe 3.2 s, predict 4.8 s (p.5).
- **Numbers** (Table 1, p.6), average ADE / FDE (m): vanilla LSTM **0.44 / 0.98**, O-LSTM **0.28 /
  0.64**, Social-LSTM **0.27 / 0.61**. On the sparse ETH scene O-LSTM and Social-LSTM tie (ADE 0.49 vs
  0.50); on crowded UCY Social-LSTM wins (O-LSTM 0.35 vs Social-LSTM 0.27), which the authors tie to density ("UCY … a
  total of 32K non-linearities as opposed to … ETH … only 15K", p.6).
- *(reading)* in a sparse scene the **hand-built occupancy feature captured all of the interaction
  gain**; learned pooling paid only in dense crowds.

**Graph Attention Networks — Veličković et al., ICLR 2018** *(fetched)*
- **Mechanism.** `e_ij = a(W h_i, W h_j)`, `α_ij = softmax_j(e_ij)` with
  `a = LeakyReLU(aᵀ[W h_i ‖ W h_j])` (Eqs. 1–3, p.3), multi-head (p.4). The graph enters **only as a
  mask**: "In its most general formulation, the model allows every node to attend on every other node,
  dropping all structural information. We inject the graph structure into the mechanism by performing
  masked attention" (p.3).
- **No edge features.** "extending the model to incorporate edge features … would allow us to tackle
  a larger variety of problems" is listed as future work (p.9). Cost `O(|V|FF′ + |E|F′)` per head
  (p.5).
- *(reading)* on a **complete** graph — every aircraft in the 25 km ring sees every other — GAT is
  additive self-attention with no mask. A "separate GNN" and "agent attention" differ here only in the
  score function and in whether pair features are available; GAT itself has none.

**Graphormer — Ying et al., NeurIPS 2021** *(cross-reference; full note `../manoeuvre_tokens/README.md` §2.7)*
- Edge features enter "via a bias term to the attention module" (p.5):
  `A_ij = (h_iW_Q)(h_jW_K)ᵀ/√d + b_φ(v_i,v_j) + c_ij`. This is exactly the planned "edge features as
  attention biases". *(reading)* a scalar bias only **re-weights** which aircraft is attended; it puts
  no pair information into the attended value — see Groot and MTR++ below for why that matters.

**Trajectron++ — Salzmann et al., ECCV 2020** *(cross-reference, `../control_normalization/`)*
- **Mechanism: typed graph, distance-gated edges, aggregated per type.** An edge `A_i → A_j` exists if
  `‖p_i − p_j‖₂ ≤ d_{S_j}`, a per-class perception range (p.5). Neighbours of one class are
  **element-wise summed**, passed through an 8-unit LSTM shared per edge type, and the edge types are
  combined by additive attention (p.6).
- Makansi et al. (below) show this per-type aggregation contributes ≈ 0 on the common benchmarks and
  that separate per-neighbour edges ("Trajectron++Edge") use neighbours more when the data carry
  interaction. Trajectron++ itself has no "no-edges" ablation (`../hierarchical_prediction/README.md`
  §1.4).

**DESIRE — Lee et al., CVPR 2017** *(cross-reference, `../multimodal_intent/`)*
- **Mechanism:** a Scene Context Fusion unit pools neighbours' hidden states on **log-polar grids with
  average pooling** plus CNN scene features (p.6).
- **Interaction can hurt on small data** (Table 1, p.8; text p.7): at 4 s on KITTI, RNN ED **3.86 m**
  vs RNN ED + scene/interaction **4.29 m** — "the model learns to behave reactive … might be due to
  the small size of the dataset"; on the larger SDD the same addition helps (9.54 → 8.80 px). Full
  DESIRE, scene-only vs scene + interaction, top-10 %: KITTI 2.07 vs 2.06 m, SDD 5.62 vs 5.33 px.

**AgentFormer — Yuan et al., ICCV 2021** *(cross-reference; ablations also in `../hierarchical_prediction/README.md` §1.4)*
- **Mechanism:** one attention over the flattened time × agent sequence, with **two** Q/K projections —
  one for same-agent pairs, one for other-agent pairs — selected by the mask `M_ij = 1(i mod N = j mod N)`
  (Eq. 5, p.4). Rule-based connectivity is a second mask: agents farther apart than `η` at `t = 0` get
  `A_ij = −∞` (p.5); `η = 100` (p.7).
- **The one same-setup GNN-vs-attention comparison found** (Tables 3–4, p.8), same temporal model:
  ETH/UCY ADE/FDE (K = 20) **GCN social + Transformer temporal 0.28 / 0.50 vs Transformer social +
  Transformer temporal 0.26 / 0.47**; nuScenes K = 5: 2.03 / 4.36 vs 1.99 / 4.12. The joint
  socio-temporal model reaches 0.23 / 0.39 and 1.86 / 3.89; replacing agent-aware attention by an
  agent-index encoding gives 0.27 / 0.46.

**SceneTransformer — Ngiam et al., ICLR 2022** *(cross-reference)*
- **Mechanism: axis-factorised attention**, alternating "attention only across time" and "attention
  only across agents" layers; permutation-equivariant over agents; road graph by cross-attention (p.5).
- Replacing each factorised pair by one attention over all agent × time elements "increased the
  computational cost of the model and performed worse": minADE **0.609** factorised vs **0.639**
  non-factorised, Argoverse val (p.8).

**Wayformer — Nayakanti et al., ICRA 2023** *(cross-reference)*
- **Interaction is a hand-built modality:** "For each modeled agent … a fixed number of the closest
  context agents … are considered", with their states "transformed into the frame of reference of our
  ego-agent" (p.2). The paper's ablations are about fusion (early/late) and factorised vs multi-axis
  attention (p.6), not about how neighbours are selected. *(reading)* one of the strongest open-loop
  forecasters of its generation used an N-closest relative-frame feature, not a graph.

**QCNet — Zhou et al., CVPR 2023** *(cross-reference)*
- **Mechanism: relative geometry in the keys and values.** For each pair a 4-D descriptor — distance,
  relative direction, relative orientation, time gap — goes through Fourier features and an MLP to
  `r_{j→i}` (p.4); social attention uses keys/values `[a_j^t ; r_{j→i}^{t→t}]` over neighbours within
  **50 m** (p.4). Factorised social attention costs `O(A²T)`, reduced to `O(A²)` online by caching
  (p.3–4).
- **Cost and gain of the fusion blocks** (Table 3, p.7; densest scene 190 agents, A40): `L_enc = 0`
  minADE₆ **0.76**, 8 ms; `L_enc = 2` **0.73**, 82 ms without reuse, 13 ms with reuse.

**MTR / MTR++ — Shi et al., NeurIPS 2022 / TPAMI** *(cross-reference)*
- MTR: local attention over k-nearest polylines (p.4); a **dense future prediction** head that feeds
  every agent's predicted future back as context is worth **+1.78 % mAP** (p.8).
- **MTR++ Table 10 (p.13) — the cleanest "how to put pair geometry into attention" ablation found.**
  WOMD val, minADE / MR / mAP: no position encoding 0.6886 / 0.1614 / 0.3513; **global** coordinates in
  Q/K/V 0.6913 / 0.1627 / **0.3463** (worse than none); **relative (query-centric) in Q/K and V 0.6490 /
  0.1559 / 0.3754**; relative in Q/K only 0.6523 / 0.1570 / 0.3658; relative in V only 0.6814 / 0.1603 /
  0.3574. The query-centric self-attention (Eq. 14, p.7) adds the relative-position encoding to the
  keys **and** to the values.
- **Cross-agent interaction of the decoder queries** (Table 9, p.12): none 0.3484 mAP; within-agent only
  0.3505; across-agent only 0.3541; both **0.3754**.

**SMART — Wu et al., NeurIPS 2024** *(cross-reference; tokeniser in `../manoeuvre_tokens/README.md` §2.3)*
- Temporal, agent-agent and agent-map attention stacked per block, keys/values carrying QCNet-style
  relative positional embeddings; the agent-agent neighbour set is "determined by a distance threshold
  of 50 meters" (p.5–6).

**You Mostly Walk Alone — Makansi et al., ICLR 2022** *(fetched)*
- **Method:** Shapley-value attribution where removing a neighbour = cutting its edge in the
  interaction graph (Eq. 3, p.5); the **social interaction score** is `E[max_j φ(x_j)]`, the
  contribution of the most influential neighbour (Eq. 4, p.5). A robustness check adds a *random* agent
  from another scene (p.5).
- **Finding** (p.7): on ETH-UCY, SDD and nuScenes "the contribution of the neighbors … is
  insignificant and close to zero, and … the contribution of a random neighbor is almost identical to
  existing neighbors"; "recent methods are unable to exploit information from neighboring agents". On
  SportVU basketball, neighbours matter.
- **Numbers** (Table 1, p.8), minADE / minFDE with and without all neighbours at test time:
  Trajectron++ ETH-UCY 0.30 / 0.51 vs 0.31 / 0.52, nuScenes 0.49 / 0.77 vs 0.49 / 0.77;
  **SportVU 4.86 / 5.31 vs 6.62 / 8.98**. The per-neighbour-edge variant Trajectron++Edge: SportVU
  4.78 / 5.22 vs **7.49 / 11.18**. Social-STGCNN ETH-UCY: 0.45 / 0.76 both ways.
- *(reading)* the drop-all-neighbours test is a zero-cost diagnostic for our scene prior: mask the
  cross-aircraft attention at evaluation and see whether the words change.

### 2.B Joint vs marginal decoding within one step — question (2)

**MotionLM — Seff et al., ICCV 2023** *(cross-reference; tokeniser in `../manoeuvre_tokens/README.md` §2.1)*
- **Parallel within a step.** `p(A_t | A_<t, S) = Π_n p(a_t^n | A_<t, S)` — "conditionally
  independent at time t, given the previous actions and scene context" (Eq. 2, p.4); training mask is
  a "blocked, staircase pattern" (p.5); justified by human reaction time ≥ 500 ms (p.4).
- **Cost:** one self-attention over the flattened `N·T` tokens; "these self-attended sequences grow
  linearly in the number of jointly modeled agents" but are short (32 = 16 steps × 2 agents);
  "Separate passes of factorized agent and time attention are also possible" (p.5).
- **Interaction frequency, not intra-step order, is what they measured** (Table 5, p.15, WOMD
  interactive val, single replica): interactive attention 0.125 Hz (agents see each other only at
  `t = 0`) → 2 Hz: minADE **1.0681 → 1.0345**, mAP **0.1558 → 0.1687**. Prediction overlap (Table 3,
  p.7, validation): marginal **0.0404** vs joint **0.0292** — "a relative 38% higher overlap rate".

**Trajeglish — Philion et al., ICLR 2024** *(cross-reference; `../manoeuvre_tokens/README.md` §2.2)*
- **Sequential within a step**, random agent order in training; the model "is not permutation
  equivariant to agent order" by design (p.5). Intra-timestep interaction is "weak in general" (p.3)
  and "becomes much less important" as context grows (Fig. 10, p.8).
- **Where it pays:** partial control with other agents on replay — "when the autonomous agent is the
  first in the permutation … they reproduce the performance of the model with no intra-timestep
  dependence. When the agent goes last however, the collision rate drops significantly" (p.18). The
  listed causes of intra-step dependence include that "driving logs are recorded at discrete timesteps
  and any interaction in the real world between timesteps gives the appearance of coordinated
  behavior" (p.13).
- *(reading)* the PDF gives no sampling-cost figure; sequential intra-step decoding needs `N` decoder
  passes per step instead of one.

**SceneTransformer** *(cross-reference)* — joint vs marginal is **a switch of the loss**, same network
(p.6). WOMD interactive test, minSADE / minSFDE / SMR at 8 s: marginal-as-joint **2.08 / 5.04 / 0.55**,
joint **1.76 / 4.08 / 0.50** (Table 3, p.8).

**MTR++** *(cross-reference)* — joint prediction by letting agents' intention queries attend to each
other (Table 9 above): +0.57 mAP points alone, +2.49 on top of within-agent interaction.

**TrafficSim — Suo et al., CVPR 2021** *(fetched; training in §2.C)*
- **Joint by a scene latent:** `P(Y^t | X^t) = ∫ P(Y^t | X^t, Z^t) P(Z^t | X^t)` with a deterministic
  decoder, `Z^t = {z_1, …, z_N}` spatially anchored per actor, prior/posterior/decoder built by
  "propagating messages across a fully connected interaction graph with actors as nodes" (p.4). One
  latent sample gives a scene-consistent plan for all actors in one parallel pass (p.4).
- Baselines on ATG4D (Table 1, p.6), scenario collision rate at 12 s: MTP (actors independent)
  **11.00 %**, ESP (social autoregressive) 4.08 %, ILVM (scene latent, open-loop) **2.90 %**,
  TrafficSim **0.50 %**.

**WOSAC — Montali et al., NeurIPS 2023 D&B** *(fetched)*
- **The benchmark's own factorisation rules:** the world model "must be autoregressive for T steps …
  10Hz resampling", and must factorise into an AV policy and an environment simulator (Eq. 1, p.4); a
  "multi-agent" factorisation into per-agent policies is explicitly allowed (p.5, Algorithm 1).
- **What the 2023 field did:** "to the best of our knowledge, TrafficSim is the only closed-loop,
  learned sim agent work to use a joint, scene-centric actor policy" (p.3); "all submissions operated
  in an agent-centric coordinate frame, rather than jointly sampling from a scene representation
  simultaneously" (p.9).
- Scoring: 32 rollouts, 9 component likelihoods (kinematic, interaction — distance to nearest object,
  collisions, time-to-collision — and map), collision and off-road weighted **2×** (p.7); a collision
  is a negative signed box distance at any time (p.20).

**Peng et al., ECCV 2024** *(fetched; training in §2.C)* — a scene-centric MotionLM: "All N motion
tokens at step t can attend to each other and all previous tokens" (p.5), and "we output N tokens
concurrently at each prediction step instead of one token" (p.7) — parallel within the step.

**SMART-R1** *(fetched)* — the policy is written `π(S_t | S_<t, C) = Π_i π(k_t^i | S*_<t, C*)`
(Eq. 2, p.4): parallel within the step, interaction through temporal, map-to-agent and agent-to-agent
attention layers (p.4).

**MALTP — Kim, Yoon & Lee** *(fetched; §2.D)* — the one aviation paper that states the choice: the
decoder "assumes conditional independence among the `y_i`" given the shared multi-agent encoding;
"Although full covariance modeling can better capture explicit inter-agent uncertainty correlations,
it introduces significant computational complexity" (p.5–6).

### 2.C How interaction is learned: teacher forcing vs closed loop — question (3)

**TrafficSim — Suo et al., CVPR 2021** *(fetched)* — **closed-loop training with a collision loss**
- **Training:** "we unroll the policy for closed-loop training and compute the loss `L_t` at each
  simulation step", back-propagating through the differentiable simulation via reparameterised
  samples (p.5). Loss `L = Σ_t λ(t) L_imitation^t + (1 − λ(t)) L_collision^t`, λ(t) annealed "to
  favour supervision from common sense over imitation" through the horizon; posterior samples for
  `t ≤ T_label`, prior samples after (p.5). Collision: each vehicle as 5 circles, pairwise
  `1 − d/(r_i + r_j)` when overlapping (Eqs. 7–8, p.5). Setup: 2 Hz ticks, 3 s history,
  `T_label = 8 s`, 12 s simulated in training and evaluation (p.6).
- **Ablation — the measured split between "closing the loop" and "adding a collision term"** (Table 2,
  p.7), scenario collision rate / traffic-rule violation / minSADE:
  M0 open-loop, 1-step plan **5.92 % / 10.19 % / 0.88 m**; M1 open-loop, 10-step plan 2.32 / 3.43 /
  0.99; M2 closed-loop, 1-step **1.28 / 3.30 / 0.54**; M3 closed-loop, 10-step **0.60 / 3.02 / 0.58**;
  M* + collision loss **0.50 / 2.77 / 0.57**. Authors: "Closed-loop training with back-propagation
  through simulation (M2) is the most important component" (p.8).
- **Constraints at simulation time** (Table 5, p.8): collision 0.50 % → **0.33 %** by rejection
  sampling of colliding plans, → **0.12 %** by gradient optimisation of the scene latent; rule
  violations rise 2.77 → 3.01 / 3.00 %.
- **Re-planning interval** (Table 4, p.8): one inference per 0.5 s → 1 s → 2 s ticks: collision
  0.50 → 0.85 → 0.96 %.
- *(reading)* closed-loop training removed **78 %** of collisions (M0 → M2) and **74 %** (M1 → M3); the
  collision loss removed a further **17 %** (M3 → M*).

**WOSAC** *(fetched)* — what the benchmark saw about closed-loop training
- "the challenge champion, MVTA/MVTE, was the only method to utilize and benefit from closed-loop
  training. Other methods that were trained in open-loop … found operating at slower replan rates
  necessary" (p.8–9). Open-loop Wayformer, identical samples: composite **0.575 at 2 Hz replanning vs
  0.338 at 10 Hz**, "a relative performance drop of 41.2%" (p.16; Table 3, p.9).
- Two warnings: "collision rate can be artificially driven to zero by static policies, and thus
  cannot measure realism" (p.3); collision-minimising post-processing "could be seen as trimming the
  tail of the distribution" because "close calls and collisions do occur in real driving data" (p.9).
- Test composite (Table 3, p.9): constant velocity 0.287, MTR+++ 0.608, MVTE **0.645**, logged oracle
  0.722; collision likelihood MVTE 0.893, oracle 1.000.

**CAT-K — Zhang et al., CVPR 2025** *(cross-reference; full note `../trajectory_as_language/README.md` §2.1)*
- Closed-loop **supervised** fine-tuning: roll out every agent with the top-K token closest to the
  recorded next state, supervise with the recovery token; no collision term anywhere.
- What it does to interaction (Table 4, p.13, WOSAC test): collision likelihood BC-reproduced
  SMART-tiny **0.9653 → 0.9702** fine-tuned; the 14× larger SMART-large 0.9632.

**Peng et al., ECCV 2024** *(fetched)* — **RL fine-tuning of a multi-agent next-token model**
- Pre-train MotionLM (scene-centric, Verlet 13×13 acceleration tokens clipped at 6 m/s², p.6) by
  teacher forcing, then REINFORCE on closed-loop rollouts of all agents with per-agent, per-step reward
  `r_{t,i} = −‖Pos_{t,i} − GT_{t,i}‖₂ − λ Coll_{t,i}` (Eq. 2, p.6), returns normalised over the batch
  (Eqs. 3–5, p.7–8). Policy gradient instead of back-propagation through time "allows us to use
  non-differentiable rewards (such as a boolean collision indicator) and … discrete outputs" (p.4).
  1M steps each stage, lr 5e-6, γ = 0.95 (p.10).
- **Numbers** (Table 1, p.10, WOSAC): 1M model collision likelihood **0.544 → 0.863**, off-road 0.525 →
  0.804, composite 0.490 → 0.597, ADE 6.33 → 2.44 m; 10M model composite 0.549 → 0.608. RL from
  scratch (no pre-training): composite **0.320**, collision 0.239.
- **Collision-weight sweep** (Table 2, p.12, 1M): λ = 0 collision **0.834**, composite 0.590, ADE 2.41;
  λ = 2 **0.863**, 0.597, 2.44; λ = 5 0.844, 0.595, 2.84; λ = 10 0.831, 0.594, 3.06 — "at very high
  values of the collision weight, all metrics tend to degrade … displacement error is a very dense and
  rich reward signal, whereas collision is a more sparse and noisy signal" (p.12–13).
- *(reading)* **91 %** of the collision-likelihood gain (0.544 → 0.834 of 0.544 → 0.863) came with
  λ = 0, i.e. from closed-loop training toward the recorded positions, not from the collision penalty.

**SMART-R1 — Pei, Shi & Shen, ICLR 2026** *(fetched)*
- Pipeline: BC pre-training → CAT-K closed-loop SFT → **Metric-oriented Policy Optimisation** with the
  WOSAC realism meta-metric as reward, advantage `A = r − α` (α = 0.77), per-token KL to the reference
  (β = 0.04) (Eqs. 3–5, p.5) → a second SFT, "SFT–RFT–SFT" (p.6).
- **Numbers.** 2 % val realism (Table 3, p.8): BC 0.7725; + SFT 0.7734; SFT + SFT 0.7730; SFT + RFT
  0.7740; SFT + RFT + SFT **0.7746**. Other RL algorithms on the same SFT model (Table 4, p.8): PPO
  0.7611, DPO 0.7682, GRPO 0.7701 — **all below the SFT model's 0.7734**. Test collision likelihood
  (Table 2, p.7): base 0.9693 → SFT 0.9702 → R1 0.9709; leaderboard realism **0.7858** vs CAT-K
  (CLSFT) 0.7846 (Table 1, p.7). KL too small or too large both hurt (Table 6, p.8).

**RLFTSim — Ahmadi et al., CVPR 2026** *(fetched)*
- REINFORCE with KL to the pre-trained SMART-tiny, reward = a leave-one-out, per-rollout form of the
  realism meta-metric (MLOO, Eq. 2, p.4); the authors argue against an ADE reward: once an agent has
  diverged, "the most realistic next action may not be to return abruptly to a pre-recorded
  ground-truth trajectory" (p.4). 1 epoch, 4 rollouts, lr 3e-6, KL target 0.01 nats (p.7).
- **Reward ablation** (Table 2, p.7, full val, realism): reference 0.7804; minADE reward 0.7801; RMM
  (RLOO) 0.7821; **RMM (MLOO) 0.7830**; collision + off-road + ADE 0.7803; **collision + off-road
  0.7786** — the hand-made safety reward lowered realism. Test (Table 1, p.6): 0.7824 → **0.7867**,
  CAT-K 0.7856.
- Goal conditioning: an indicator added to the **relative positional encoding** of the goal polyline
  beats concatenating goal coordinates to the agent token (miss rate 13.4 vs 15.0 % hard goals,
  Table 3, p.7).

**R1Sim — Wang et al., RA-L 2026** *(cross-reference; `../trajectory_as_language/README.md` §2.2)*
- GRPO with reward `r^safe · r^dis` (collision sign × closeness to GT). Test (Table I, p.5): on
  SMART-tiny collision **0.9601 → 0.9718**, RMM 0.7591 → 0.7675; on CAT-K collision **0.9707 →
  0.9713**, RMM **0.7687 → 0.7688**.

**Grammar-constrained decoding** and **DAgger** *(cross-references)* — the decode-time mask and the
on-policy-labelling argument are summarised in `../manoeuvre_tokens/README.md` §2.9 and
`../trajectory_as_language/README.md` §1.6; not repeated.

**You Mostly Walk Alone** *(§2.A)* belongs here too: teacher-forced interaction modules on common
benchmarks learn **no measurable use of neighbours** (Table 1, p.8).

### 2.D Aviation — question (4)

**MAIFormer — Yoon & Lee, IEEE T-ITS 2026** *(cross-reference; horizons in `../prediction_horizons/README.md` §2.2)*
- **Mechanism:** each aircraft is three variate tokens (lat, lon, alt) embedded over its 2-min window;
  *masked multivariate attention* within each aircraft, then *agent attention* across all aircraft of
  the scene; absolute coordinates, no pair features (p.4–5). Incheon, arrivals within 70 NM, 6 s
  steps (p.4), 509,389 scenes (p.5).
- **The horizon split** (Table I, p.6), altitude MAE (ft): 6 s — FlightBERT++ (single-aircraft)
  **25.4**, MAIFormer 29.2; 2 min — FlightBERT++ 287.8, AgentFormer 219.4, MAIFormer **164.2**. Text:
  single-agent models "slightly outperformed the multi-agent models … at early time steps (e.g.,
  horizons 1 and 5)" because short-term motion is "dominated by the dynamics of individual aircraft"
  (p.6–7).
- Ablation (Table II, p.9): without agent attention (the per-aircraft mask also removed, so variates
  of different aircraft still see each other) altitude MAE at 2 min **244.6 ft** vs 164.2.

**MALTP — Kim, Yoon & Lee, AIAA SciTech 2026** *(fetched)*
- **Mechanism:** MAIFormer's encoder plus a **wake-turbulence-category type embedding** "because the
  required time separations between consecutive arrivals depend on the WTCs" (p.3–4); marginal
  Gaussian landing time per aircraft (p.5). Incheon, Jan–May 2023, arrivals from 70 NM, 6 s, 2-min
  windows (p.6–7).
- **Numbers** (Table 1, p.8): MAE **6.19 s** vs XGBoost 47.34 s, LightGBM 52.23 s; Kendall τ of the
  induced landing order 1.000 vs 0.981. Worked case: the controller gave ACA063 a shortcut; the
  single-aircraft models predicted KAL856 first because its along-route distance was shorter, the
  multi-agent model got the order right (p.8).
- *(reading)* the baselines are single-aircraft tree models, so the gain mixes model class with
  interaction; there is no same-architecture no-interaction ablation.

**DA-STGCN — Kuang et al., arXiv 2025** *(fetched)*
- **Mechanism: GNN with an inverse-distance adjacency** `k = 1/‖v_i − v_j‖` (p.3), re-weighted by
  self-attention, then GAT aggregation (p.4). ADS-B within 50 km of BOS and JFK plus two en-route
  regions, 45,463 records, 10 s (p.5); observe 40 s, predict 60 s (Table 1, p.6).
- **Numbers** (Table 3, p.7), terminal horizontal / vertical ADE: Social-STGCNN (distance adjacency
  only) 0.0096 / 42.54; DA-STGCN 0.0068 / 27.47 (units as printed; horizontal apparently degrees).
  *(reading)* the gain comes from letting **attention** re-weight the hand-set distance graph; there is
  no no-interaction row.

**Jung, Hardy & Kochenderfer, arXiv 2023** *(fetched)*
- **Mechanism: explicit pairwise model.** Arrivals within 25 NM of KJFK (p.5), expressed as deviations
  from the flown procedure; a GMM over *pairs* of successive arrivals
  `τ_pair = [τ⁽¹⁾, δ₁₂, τ⁽²⁾]` with the inter-arrival time `δ₁₂` (Eq. 11, p.12), chained into scenes of
  any size by stitching pairwise covariance blocks (p.13–14). Radar vectoring is named as the stage
  inside 25 NM at KJFK (p.6).
- **Numbers** (Table 1, p.19): three arrivals within 180 s on 13L, 1,000 generated sets — independent
  single-aircraft model **781** losses of separation (< 3 NM) vs pairwise model **482**; Jensen–Shannon
  divergence of speed 0.0650 vs 0.0147, of closest distance 0.0261 vs 0.0307.

**Zhang & Chen, arXiv 2022 (phased flight TP)** *(cross-reference)* — approach-phase aircraft as a
spatio-temporal graph handled by an attention-based structural LSTM. Approach ADE (Table 2, p.14):
vanilla LSTM 2155 m, Social-LSTM 1437 m, theirs 726 m. The ablation removes motion constraints and
attention, not interaction.

**TrajAirNet — Patrikar et al., ICRA 2022** *(cross-reference)* and **ASCENT — Prutsch et al., ICRA 2026** *(cross-reference)*
- TrajAirNet models interaction "using graph attention networks" (p.4); no single-aircraft ablation.
- ASCENT encodes **only the target aircraft's own normalised history plus its pose** (p.4) — "we do
  not use additional information" (p.7) — and beats TrajAirNet on the same TrajAir splits: minADE₅ /
  minFDE₅ **0.35 / 0.58 km vs 0.78 / 1.55 km** (Table I, p.5). *(reading)* on this GA dataset the
  frame normalisation mattered more than the social module.

**Brittain, Yang & Wei, arXiv 2020 (D2MAV-A)** *(fetched)*
- **Controller actions with traffic context, learned by multi-agent RL.** One shared PPO policy per
  aircraft; ownship state plus a variable set of intruders encoded by **Luong attention** from the
  ownship (Eqs. 10–13, p.13). Intruder features are relative and pairwise: distance to ownship,
  distances to the shared intersection (p.10). Speed advisories every 12 s: decelerate / hold /
  accelerate (p.11). Reward −1 below 3 NM, shaped up to 10 NM, small penalty per speed change (p.12).
  En-route sectors in BlueSky.
- The motivation is exactly (1): N-closest inputs need `N` tuned; distance-sorted LSTMs depend on the
  sort; "attention networks have access to all aircraft and are not dependent on a sorting strategy"
  (p.8).
- **Numbers** (Table 2, p.17, goals out of 30, 200 episodes): case C attention **30.0 ± 0.0**, LSTM
  sorted by distance 29.57 ± 1.00, fixed N-closest 28.76 ± 1.56, N-closest by time-to-intersection
  28.52 ± 1.70. Normalised performance flat from 10 to 100 aircraft without retraining (Fig. 5, p.18).

**Groot, Ellerbroek & Hoekstra, SESAR Innovation Days 2022** *(fetched)*
- **The argument for relative values, stated directly:** in dot-product attention "the relation
  between the two aircraft states gets reduced to a single-weight scalar, this allows only the absolute
  state information of the other aircraft to be stored in the hidden state" (p.3). Their fix
  translates and rotates the key/value tokens into each query aircraft's frame, "at the cost of a
  factor (N−1) additional attention computations" (p.3, p.7).
- Setup: 20 aircraft, 5 NM separation, 5 s steps, SAC, heading and speed actions (p.5, p.7).
- **Numbers** (Tables VI–VII, p.8–9), best reward / fewest intrusion steps per agent: **relative FNN on
  the 3 closest −3.65 / 0.40**; relative transformer over all −4.25 / 0.77; absolute FNN −15.10 / 4.85;
  absolute transformer −20.73 / 15.52. Abstract: attention over all 20 "results in similar, but
  slightly lower, performance to handcrafted observation vectors, without requiring manual selection"
  (p.3).

**Carvell et al., AIAA 2026 (online action-stacking)** *(fetched)*
- **Controller instruction vocabulary.** Train PPO on primitive words ({no action, turn ±10°} per
  aircraft; levels likewise) with an action-damping penalty that makes the policy issue bursts; at
  inference, stack a burst into one clearance ("turn right 70 degrees") without advancing time (p.7–9,
  p.11). Two aircraft, **centralised** state (concatenated per-aircraft vectors), 5 NM safety reward
  on projected separation (p.6–9).
- **Numbers:** undamped 113.0 actions per episode, damped 14.5, damped + stacked 7.2 (p.10–12). Two-
  aircraft avoidance, 100 episodes: stacked 5-word policy **1** loss of separation, 81.2 actions;
  37-word policy **9** losses, 126.3 actions (p.14).

**FlightBERT with spoken instructions — Guo et al.** *(cross-reference; `../hierarchical_prediction/README.md` §1.4)*
— the one data-driven aviation model that uses **controller instructions** as an input (single
aircraft, instruction transcribed by ASR); worth −25 % MDE at 3 min. No traffic context.

**Found in the aviation search** (arXiv API, 2026-09-24): `abs:"multi-aircraft" AND abs:"trajectory
prediction"`; `abs:aircraft AND abs:"trajectory prediction" AND abs:attention AND abs:terminal`;
`abs:"terminal airspace" AND abs:interaction`; `abs:"air traffic control" AND abs:"reinforcement
learning" AND abs:attention`; `abs:vectoring AND abs:aircraft AND abs:learning`;
`abs:"controller" AND abs:"instructions" AND abs:aircraft AND abs:learning`; `abs:"arrival sequencing"
AND abs:"reinforcement learning"` (0 hits); plus web search for the Delft relative-state paper. No
paper was found that learns **controller instructions from recorded terminal data with other
aircraft as context**.

---

## 3. What the sources say, question by question

**(1) How interaction enters.**
- The recent models here use **attention across agents**, factorised from attention within each
  agent's own sequence (SceneTransformer, QCNet, SMART, SMART-R1, MAIFormer). Factorised beat un-factorised in the one test
  (0.609 vs 0.639 minADE, SceneTransformer p.8).
- **Pair geometry matters more than the aggregator.** Relative encodings beat none, and global
  (absolute) encodings are worse than none (MTR++ Tab. 10); relative states beat absolute ones by
  12–20× in intrusions (Groot); frame normalisation alone let a single-aircraft model beat a GAT social
  model (ASCENT vs TrajAirNet).
- **Where the pair geometry is injected matters.** MTR++ Tab. 10: relative encoding in keys only
  0.3658 mAP, values only 0.3574, both **0.3754**. Groot: a scalar weight cannot carry relative state
  into the output. Graphormer's bias is keys-side only.
- **Hand-built neighbour features are competitive at small N**: O-LSTM ≈ Social-LSTM on sparse ETH;
  the 3-closest relative FNN beat the all-agent relative transformer (Groot); Wayformer uses N-closest;
  Jung's explicit pairwise model halves losses of separation over independent sampling.
- **GNN vs attention:** the only same-setup comparison (AgentFormer) favours the Transformer social
  model slightly; DA-STGCN's gain over a distance-graph GCN comes from adding attention. No source shows
  a separate GNN beating agent attention on a small, fully connected scene; on a complete graph GAT is
  masked-free additive attention (GAT p.3).

**(2) Joint vs marginal within a step.** Parallel, conditionally independent decoding is the default of
the next-token traffic models (MotionLM, SMART, SMART-R1, Peng), justified by reaction latency and
cheap (one pass per step). Sequential intra-step decoding (Trajeglish) buys a measurable but "weak"
gain that concentrates in short context and partial control, at `N` passes per step *(reading)*. What
is measured to matter is **how often** agents see each other (MotionLM Tab. 5) and whether the
**training loss is joint** (SceneTransformer Tab. 3). The one aviation paper that discusses it
(MALTP) chose marginal outputs over a full covariance for cost.

**(3) How interaction is learned.** Teacher forcing on common benchmarks learns ≈ no neighbour use
(Makansi); interaction features can even hurt on small data (DESIRE KITTI). Closing the loop is the
large, repeated effect: TrafficSim −78 % collisions (open → closed loop), Peng 91 % of the collision
gain with no collision term, CAT-K +0.0049 collision likelihood with no collision term. An explicit
collision term adds a smaller step (TrafficSim −17 %, Peng λ = 2 +0.029 then degrading), and RL after
closed-loop SFT adds little (R1Sim +0.0001 RMM; SMART-R1 +0.0006 realism, PPO/DPO/GRPO below SFT).
Hand-made safety rewards alone can lower realism (RLFTSim 0.7786 < 0.7804). Simulation-time filtering
is a cheap, separate lever (TrafficSim Tab. 5). WOSAC warns that a collision rate can be zeroed by a
static policy and that real data contain close calls.

**(4) Aviation.** Terminal interaction models are open-loop predictors on 1–10 s data at Incheon,
Boston/JFK and a GA field; their measured interaction benefit appears at **minute** horizons, not at
seconds (MAIFormer). Controller-action learners (Brittain, Groot, Carvell) are RL agents in en-route
simulators with 3–5 NM lateral separation, 2 to 100 aircraft, and no recorded controller data. The two
threads have not met: nothing learns instructions from recorded traffic with other aircraft as
context, and nothing closes the loop on real terminal arrivals.

---

## 4. What this means for the scene prior — **(reading, not source)**

Everything in this section is my inference from §2–§3, mapped onto the plan in the introduction.

1. **Use plain agent attention over all aircraft in the ring; do not add a GNN or a neighbour cut-off.**
   With at most 9 aircraft, attention across agents costs ≤ 81 pairs per step; the 50 m radius (QCNet,
   SMART) and distance mask (AgentFormer, Trajectron++) exist for 100+ agent scenes. No source gives a
   reason to prefer a separate GNN on a complete graph (§3 (1)). Keep the factorised time/agent layout
   (SceneTransformer p.8), and the flattened alternative is also affordable at this N if it is simpler
   (MotionLM p.5).
2. **Put the edge features into the values as well as the logits.** The plan says "edge features enter
   as attention biases". A Graphormer-style scalar bias decides *whom* each aircraft listens to but
   cannot tell it *where* that aircraft is relative to itself (Groot p.3); MTR++ Table 10 puts a number
   on the difference (keys-only 0.3658 vs keys + values 0.3754 mAP). Use the pair features (same /
   parallel runway, along-track gap to threshold, relative position in the query aircraft's frame,
   closure rate) both as a per-head logit bias and as an embedding added to the attended value (QCNet,
   MTR++ Eq. 14). Express relative position in the **query aircraft's** frame; do not feed absolute
   coordinates as the interaction signal (MTR++ "global" row is worse than none).
3. **The hand-built pair features will probably carry most of the value — keep a baseline that shows
   it.** In tiny scenes the hand-built relative/occupancy features matched or beat learned aggregation
   (Social LSTM on ETH, Groot, Jung). A cheap control: the same prior with the pair features but no
   learned cross-attention (e.g. features of the leading and trailing aircraft to the same runway only).
4. **Parallel decoding within the 2 s step is the attested default, but the separation mask argues for
   an ordered decode.** With parallel decoding, aircraft *i*'s mask can only be computed against the
   others' *previous-step* words, so two aircraft can each pick words that are safe alone and unsafe
   together; a mask evaluated in a fixed order (e.g. landing sequence, leader first) is well defined
   because each aircraft sees what earlier ones chose. The cost is `N ≤ 9` decoder passes per 2 s step,
   negligible here. Trajeglish also names discrete logging as a cause of apparent intra-step
   coordination, and our step (2 s) is 4–20× coarser than the attested ones (0.1–0.5 s). A controller
   also speaks to one aircraft at a time. A reasonable plan: **pretrain parallel** (as planned), and add
   the ordered intra-step pass only where the mask is applied, measuring the difference.
5. **Expect teacher forcing to learn little interaction, and test for it.** Most sequencing happens
   before 25 km and the median ring holds 1–2 others; Makansi's benchmarks look similar. Run the
   drop-all-other-aircraft test (Makansi Eq. 3: cut the cross-attention at evaluation) on the pretrained
   prior before investing in post-training; if the words do not change, the interaction path has not
   been learned. Report the gain by horizon, because MAIFormer found interaction pays at minutes and
   costs at seconds.
6. **Order the post-training steps as the evidence ranks them.** (a) Closed-loop SFT against the
   recorded words with the executor in the loop (CAT-K; TrafficSim's M2; Peng's λ = 0) — the largest,
   reproduced effect, needing no separation term. (b) Separation as a **decode-time filter/mask**
   (TrafficSim Table 5, grammar-constrained decoding) rather than a loss. (c) RL with a separation /
   landing reward last, with a KL anchor to the reference (SMART-R1, RLFTSim) and a dense data term,
   and a moderate weight (Peng Table 2 degrades above λ = 2). Do not use a separation-only reward: in
   RLFTSim a collision + off-road reward lowered realism.
7. **Calibrate the separation filter on the data, not to zero.** Real arrivals are routinely spaced
   *at* the radar and wake minima, and WOSAC warns that minimising collisions trims a real tail. The
   mask threshold should be the published minimum (`../arrival_separation/`), and the recorded
   scenes' own rate of sub-minimum spacing is the reference the scene prior is judged against — a
   prior that is "safer than the data" has left the data.
8. **Where the contribution sits.** No aviation source learns controller instructions from recorded
   multi-aircraft terminal data, and none closes the loop on real arrivals (§3 (4)). The aviation RL
   agents (Brittain, Groot, Carvell) show the pieces — attention over intruders, relative states, a
   small primitive instruction vocabulary compiled into clearances — but in en-route simulators without
   recorded controllers. A scene prior trained on recorded arrivals and post-trained in closed loop
   with separation filters would be the first of its kind in this literature, and its fair comparators
   are open-loop multi-aircraft predictors (MAIFormer, MALTP), not these RL agents.

---

## 5. Not fetched

| source | status | reason |
|---|---|---|
| **Trajectron++** (Salzmann et al., ECCV 2020, arXiv:2001.03093) | asked for; **not re-downloaded** | already at `../control_normalization/papers/Trajectron++_Salzmann2020_…pdf` (and `../procedure_hard_constraints/papers/trajectronpp_2001.03093.pdf`) |
| **CAT-K** (Zhang et al., CVPR 2025, arXiv:2412.05334) | asked for; **not re-downloaded** | already at `../trajectory_as_language/papers/CATK_Zhang2025_…pdf`, with a full note there |
| B-STAR (Pang et al., KBS 2022), S-STGCNN (JAIS 2023), FPG-SLSTM (EAAI 2025) — multi-aircraft terminal models | not fetched | paywalled; abstracts already recorded in `../hierarchical_prediction/README.md` §4 |
| *Symphony* (Igl et al., ICRA 2022, arXiv:2205.03195) | verified, not fetched | closed-loop sim agents; WOSAC groups it with TrafficSim as trained with "closed-loop adversarial losses" (p.3); outside the 1–3 RL-paper cap |
| *RIFT* (Chen et al., arXiv:2505.03344) | verified, not fetched | group-relative RL fine-tuning in a physics simulator; AV-centric, cap |
| *Human-compatible driving partners* (Cornelisse & Vinitsky, arXiv:2403.19648) | verified, not fetched | data-regularised self-play PPO (KL to a BC policy), trained from scratch rather than fine-tuning a token model; cap |
| *On Learning Closed-Loop Probabilistic Multi-Agent Simulator* (Lu et al., IROS 2025, arXiv:2508.00384) | verified, not fetched | already listed as found-not-read in `../trajectory_as_language/README.md` §4; cap |
| *RoaD* (Garcia-Cobo et al., arXiv:2512.01993) | verified, not fetched | closed-loop SFT of an ego driving policy, not multi-agent |
| *Learning to Explain Air Traffic Situation* (Chai, Yoon & Lee, arXiv:2502.10764) | verified, not fetched | same Incheon multi-agent Transformer as MAIFormer/MALTP, used for attention explanation; adds no new mechanism |
| *Autonomous Air Traffic Controller* (Brittain & Wei, arXiv:1905.01303); *Improving Autonomous Separation Assurance … with Attention Networks* (Brittain, Alvarez & Breeden, AAAI 2024, arXiv:2308.04958) | verified, not fetched | the earlier and later members of the D2MAV line; the 2020 paper is the one with the attention-vs-sorting comparison |
| *Automatic Control With Human-Like Reasoning: … Language Model Embodied Air Traffic Agents* (Andriuškevičius & Sun, arXiv:2409.09717); *A Future Capabilities Agent for Tactical ATC* (Kent et al., DOI 10.2514/6.2026-1203, arXiv:2601.04285) | verified, not fetched | controller agents that are not learned from data (LLM function-calling; rules + search), en-route |

Two verification notes:
- **Makansi et al.** is read from arXiv v1 ("Preprint. Under review."); the ICLR 2022 camera-ready was
  not compared, so Table 1 numbers are the preprint's.
- **Brittain et al. 2020** is cited as the arXiv preprint; the JAIS 2021 article with a different title
  by the same authors is recorded but its identity with the preprint was not checked.

---

## 6. Layout

| file | what |
|---|---|
| `README.md` | this index |
| `download.sh` | re-fetches the 14 PDFs new to this repository (skips existing files); lists the 21 cited by path |
| `papers/SocialLSTM_Alahi2016_…pdf` | CVPR 2016 (CVF open access) |
| `papers/GAT_Velickovic2018_…pdf` | ICLR 2018, arXiv:1710.10903 |
| `papers/YouMostlyWalkAlone_Makansi2022_…pdf` | ICLR 2022, arXiv:2110.05304 |
| `papers/TrafficSim_Suo2021_…pdf` | CVPR 2021, arXiv:2101.06557 |
| `papers/WOSAC_Montali2023_…pdf` | NeurIPS 2023 D&B, arXiv:2305.12032 |
| `papers/RLFT_Peng2024_…pdf` | ECCV 2024, arXiv:2409.18343 |
| `papers/SMART-R1_Pei2025_…pdf` | ICLR 2026, arXiv:2509.23993 |
| `papers/RLFTSim_Ahmadi2026_…pdf` | CVPR 2026, arXiv:2605.19033 |
| `papers/MALTP_Kim2025_…pdf` | AIAA SciTech 2026, arXiv:2512.08281 |
| `papers/DA-STGCN_Kuang2025_…pdf` | arXiv:2503.04823 |
| `papers/PairwiseTerminal_Jung2023_…pdf` | arXiv:2303.09981 |
| `papers/D2MAV-A_Brittain2020_…pdf` | arXiv:2003.08353 |
| `papers/RelStateTransformer_Groot2022_…pdf` | SESAR Innovation Days 2022 (TU Delft repository) |
| `papers/ActionStacking_Carvell2026_…pdf` | arXiv:2601.04287, DOI 10.2514/6.2026-2746 |
