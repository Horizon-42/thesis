# Multiple-hypothesis (multimodal) intent prediction — reading list (2026-09-18)

Why this folder exists. A design under discussion for `4dTrajectory/ts_transformer` would have the
model propose **K candidate intents / goals / trajectories** per step or per segment instead of one,
train them with a **winner-takes-all (min-over-K) loss** — only the hypothesis closest to the
observed track receives the regression gradient — alongside a **mode-probability loss**, and then
**select at inference** by that probability or by a separate scorer. Variants in the literature
anchor the hypotheses on goals or trajectory templates, or structure them as a **tree/beam** over
several stages instead of a flat set. The question this folder answers sources for:

> For each piece of that scheme: **how many hypotheses, what picks the winner during training, what
> picks the answer at inference, and is the prediction one-shot or multi-step?** And: **does any
> aircraft / TMA trajectory-prediction work do explicit K-hypothesis or tree-structured multimodal
> output at all?**

This folder is a **mechanism** reading list, not an assessment. Every number below was read out of
the fetched PDF, not the abstract. Conventions: formulas in backticks are quoted as the source
writes them (symbols transliterated to Unicode); "checked at" says where the claim was read;
**(reading)** marks my inference, not the source's claim; "not stated" means the paper does not say.

The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`); `./download.sh` re-fetches the
**9** papers that are new to this repository (tested 2026-09-18: all 9 came back, `%PDF-` verified).
**Thirteen further sources were already downloaded for sibling folders and are NOT duplicated here**
— they are cited by path in §1.3 and given notes in §2.10–§2.18, the same rule
`manoeuvre_tokens/download.sh` uses for DAgger. §4 lists what could not be fetched.

---

## 1. Sources

### 1.1 Verification of the list as given

Every title, author list, year, venue and identifier below was checked against the arXiv API
abstract record, the publisher/proceedings page, or the PDF itself. Two items needed correction and
one was **not found as described**.

| as given in the brief | status | what the record says |
|---|---|---|
| MultiPath — Chai, Sapp, Bansal, Anguelov, CoRL 2019 | **verified** (title completed) | Yuning Chai, Benjamin Sapp, Mayank Bansal, Dragomir Anguelov, *MultiPath: **Multiple Probabilistic Anchor Trajectory Hypotheses** for Behavior Prediction*, arXiv:1910.05449; arXiv comment: "Appears in CoRL 2019" |
| MultiPath++ — Varadarajan et al., 2022 (ICRA?) | **verified** | arXiv:2111.14973 (posted **2021-11-29**), **ICRA 2022**. First author Balakrishnan Varadarajan; 11 authors |
| TNT — Zhao et al., CoRL 2020 | **verified** | *TNT: Target-driveN Trajectory Prediction*, arXiv:2008.08294, 12 authors (Hang Zhao … Dragomir Anguelov), **CoRL 2020** |
| DenseTNT — Gu, Sun, Zhao, ICCV 2021 | **verified** | Junru Gu, Chen Sun, Hang Zhao, *DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets*, arXiv:2108.09640, arXiv comment "Accepted to ICCV 2021" |
| MTR — Shi, Jiang, Dai, Schiele, NeurIPS 2022; "and MTR++ if it exists" | **verified; MTR++ exists** | MTR = arXiv:2209.13508, comment "Accepted by NeurIPS 2022 as **Oral**". **MTR++** = arXiv:2306.17770, *MTR++: Multi-Agent Motion Prediction with Symmetric Scene Modeling and Guided Intention Querying*, same four authors, **TPAMI 2024** |
| Trajectron++ — Salzmann, Ivanovic, Chakravarty, Pavone, ECCV 2020 | **verified** | arXiv:2001.03093, *Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data*, **ECCV 2020** |
| DESIRE — Lee et al., CVPR 2017 | **verified** (author list longer than "et al." suggests) | Namhoon Lee, Wongun Choi, Paul Vernaza, Christopher B. Choy, **Philip H. S. Torr**, Manmohan Chandraker, *DESIRE: Distant Future Prediction in Dynamic Scenes with Interacting Agents*, arXiv:1704.04394, **CVPR 2017** |
| Multiple Choice Learning — Guzman-Rivera, Batra, Kohli, NeurIPS 2012 | **verified** | Abner Guzman-Rivera, Dhruv Batra, Pushmeet Kohli, *Multiple Choice Learning: Learning to Produce Multiple Structured Outputs*, **NIPS 2012** (Advances in NIPS 25), pp. 1799–1807. The PDF's title page prints the name **without** the accent ("Abner Guzman-Rivera"); the NeurIPS index lists it as "Guzmán-rivera" — cite it as the PDF prints it |
| Rupprecht et al., ICCV 2017 | **verified** | Christian Rupprecht, Iro Laina, Robert DiPietro, Maximilian Baust, Federico Tombari, Nassir Navab, Gregory D. Hager, *Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses*, arXiv:1612.00197, comment "**ICCV 2017**" |
| Evolving WTA — Makansi, Ilg, Çiçek, Brox, CVPR 2019 | **verified** | *Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction*, arXiv:1906.03631, comment "In **CVPR 2019**" |
| mmTransformer — Liu et al., CVPR 2021 | **verified** | Yicheng Liu, Jinghuai Zhang, Liangji Fang, Qinhong Jiang, Bolei Zhou, *Multimodal Motion Prediction with Stacked Transformers*, arXiv:2103.11624, comment "**CVPR2021**" |
| Tree-structured Policy Planning — Chen et al., 2023 ("check exact title/venue") | **verified** | Yuxiao Chen, Peter Karkus, Boris Ivanovic, Xinshuo Weng, Marco Pavone, *Tree-structured Policy Planning with Learned Behavior Models*, arXiv:2301.11902, **ICRA 2023**, pp. 7902–7908, DOI `10.1109/ICRA48891.2023.10161419` |
| Liu & Hansen 2018, aircraft trajectories, deep generative ConvRNN | **verified** | Yulin Liu, Mark Hansen, *Predicting Aircraft Trajectories: A Deep Generative Convolutional Recurrent Neural Networks Approach*, arXiv:1812.11670 (2018-12-31), 24 pp., code `github.com/yulinliu101/DeepTP`; **no venue stated on the PDF** |
| "Pang, Xu, Liu (2018–2020) **Bayesian**/probabilistic aircraft trajectory prediction" | **CORRECTED — the author triple and the method do not go together** | A **Pang, Xu & Liu** paper exists and is open access: Yutian Pang, Nan Xu, Yongming Liu, *Aircraft Trajectory Prediction using LSTM Neural Network with Embedded Convolutional Layer*, **Annual Conference of the PHM Society 11(1), 2019**, DOI `10.36001/phmconf.2019.v11i1.849` — but it is a **deterministic MSE-trained ConvLSTM**, not Bayesian and not multimodal (§2.9). The **Bayesian** work is by a different author set: Pang & Liu, *Probabilistic Aircraft Trajectory Prediction Considering Weather Uncertainties Using Dropout as Bayesian Approximate Variational Inference*, **AIAA SciTech 2020**, AIAA 2020-1413; and Pang, Zhao, Yan & Liu, *Data-driven trajectory prediction with weather uncertainties: A Bayesian deep learning approach*, **Transportation Research Part C 130 (2021) 103326**. Both are paywalled with no preprint — see §4 |

No item on the list turned out to be invented. The only genuine mismatch is the last row.

### 1.2 Fetched here — the mechanism table (9 PDFs)

`K`/`M` below is the paper's own symbol for the number of hypotheses.

#### Cluster A — where the min-over-K loss comes from

| key / file | citation | how many | what picks the winner in training | what selects at inference | one-shot or multi-step | input → output |
|---|---|---|---|---|---|---|
| `MCL_GuzmanRivera2012…` | Abner Guzman-Rivera, Dhruv Batra, Pushmeet Kohli, *Multiple Choice Learning: Learning to Produce Multiple Structured Outputs*, **NIPS 2012**, pp. 1799–1807, https://papers.nips.cc/paper/4549 | `M` (M = 6 in the segmentation figure) | the **oracle / hindsight set-loss** `L(Ŷᵢ) = min_m ℓ(yᵢ, ŷᵢᵐ)`; in the max-margin program a binary flag `ρ_{i,m}` marks the predictor with the smallest hinge loss | **nothing** — the M-tuple is handed to a downstream user/re-ranker; no mode probability is modelled | one-shot (structured output) | image → M segmentations; protein → M side-chain configurations |
| `MHP_Rupprecht2017…` | Rupprecht, Laina, DiPietro, Baust, Tombari, Navab, Hager, *Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses*, **ICCV 2017**, arXiv:1612.00197 | `M` (M-MHP) | a **meta loss** over any base loss: `M(f_θ(xᵢ), yᵢ) = Σ_j δ(yᵢ ∈ Y_j(xᵢ)) L(f_θ^j(xᵢ), yᵢ)`, with `δ` **relaxed to ε = 0.05** so losing heads still get a small gradient | not modelled in the framework (the paper reports oracle/best-hypothesis metrics; a selector is task-specific) | one-shot; shared trunk, M heads | image → M poses / M futures / M segmentations |
| `EWTA_Makansi2019…` | Makansi, Ilg, Çiçek, Brox, *Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction*, **CVPR 2019**, arXiv:1906.03631 | `M` hypotheses → `K` mixture components | **Evolving WTA**: update the **top-k** winners, `k` annealed from `M` down to `1` | stage 2 **fits** a mixture to the hypotheses via soft assignments `γ_k = softmax(z_k)`; the output is a proper mixture density | one-shot, **two-stage network**, single forward pass | image + past → M hypotheses → Laplace/Gaussian mixture |

#### Cluster B — K hypotheses in motion forecasting

| key / file | citation | how many | what picks the winner in training | what selects at inference | one-shot or multi-step | input → output |
|---|---|---|---|---|---|---|
| `MultiPath_Chai2019…` | Chai, Sapp, Bansal, Anguelov, *MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction*, **CoRL 2019**, arXiv:1910.05449 | `K` **anchors** fixed before training (k-means over training trajectories, or uniform enumeration); K = 16 with covariances, K = 64 for means only | **hard assignment to the closest anchor**: `1(k = k̂ₘ)`, where `k̂ₘ` is the anchor with minimum ℓ2 to the ground-truth trajectory — a WTA over a *fixed* hypothesis set | softmax over anchors `π(a_k|x)`; the MAP sample per anchor gives a weighted trajectory set | **one-shot** (whole T-step trajectory per anchor, one forward pass) | raster scene + agent state → K anchor probabilities + per-anchor per-timestep Gaussian offsets |
| `DESIRE_Lee2017…` | Lee, Choi, Vernaza, Choy, Torr, Chandraker, *DESIRE: Distant Future Prediction in Dynamic Scenes with Interacting Agents*, **CVPR 2017**, arXiv:1704.04394 | `K` **CVAE samples** (K is a test-time knob; results are oracle-over-top-K curves) | **not WTA** — the reconstruction loss is the **mean** over samples, `ℓ_Recon = (1/K) Σ_k ‖Yᵢ − Ŷᵢ^(k)‖`, plus a KL term | a learned **IOC scorer**: each sample accumulates per-timestep rewards; the final answer is the sample with the **maximum accumulated future reward** | multi-step in the sense of **iterative refinement** (IT0/IT1/IT4 regression feedback), but each sample is a full 4 s trajectory | past tracks + scene context + other agents → K ranked, refined trajectories |
| `mmTransformer_Liu2021…` | Liu, Zhang, Fang, Jiang, Zhou, *Multimodal Motion Prediction with Stacked Transformers*, **CVPR 2021**, arXiv:2103.11624 | `K` **trajectory proposals** as decoder queries (6 plain; **36** with the region strategy) | **region-based training strategy (RTS)**: the space around the last observation is cut into `M` regions, proposals are split `N = K/M` per region, and **only the proposals of the region containing the GT endpoint** get the Huber regression loss — a spatial WTA, not a per-sample argmin | a confidence head (KL-divergence scoring) plus an auxiliary region classifier; top-6 are emitted | one-shot per proposal | agent history + map + social context → K trajectories + confidences |

#### Cluster C — tree-/branch-structured multi-step prediction

| key / file | citation | how many | what picks the winner in training | what selects at inference | one-shot or multi-step | input → output |
|---|---|---|---|---|---|---|
| `TPP_Chen2023…` | Chen, Karkus, Ivanovic, Weng, Pavone, *Tree-structured Policy Planning with Learned Behavior Models*, **ICRA 2023**, pp. 7902–7908, arXiv:2301.11902 | two trees: an **ego trajectory tree** and a **scenario tree**; **2 stages, branching factor 4** in the experiments | the prediction model is trained separately (three different ones are swapped in); TPP itself has **no WTA** — branch probabilities come from the prediction model | **dynamic programming** over the tree: a finite-horizon MDP whose states are tree nodes, solved backwards for the optimal *policy*, not a single trajectory | **explicitly multi-stage**; each node branches into children per stage | scene + ego candidates → an ego **policy** over the trajectory tree |

#### Cluster D — aviation

| key / file | citation | how many | what picks the winner in training | what selects at inference | one-shot or multi-step | input → output |
|---|---|---|---|---|---|---|
| `DeepTP_LiuHansen2018…` | Yulin Liu, Mark Hansen, *Predicting Aircraft Trajectories: A Deep Generative Convolutional Recurrent Neural Networks Approach*, arXiv:1812.11670 (no venue stated) | **K = 3** Gaussian mixture components per track point; **beam width `N_BS = K² = 9`** | plain **mixture NLL** (no WTA, no hard assignment) | **beam search** over the per-step mixture, keeping the `N_BS` best partial sequences, then **one** output — the trajectory with the highest log-likelihood; Kalman/RTS smoothing after | **multi-step, beam-structured** autoregressive decoding | filed flight plan + wind/temperature/convective-weather cubes → one 4D trajectory (from a beam of 9) |
| `PangXuLiu2019…` | Yutian Pang, Nan Xu, Yongming Liu, *Aircraft Trajectory Prediction using LSTM Neural Network with Embedded Convolutional Layer*, **Annual Conf. of the PHM Society 11(1), 2019**, DOI `10.36001/phmconf.2019.v11i1.849` | **one** | — (MSE) | — | multi-step autoregressive, single hypothesis | flight plan + Echo Top weather cube → one trajectory |

### 1.3 Cross-references — already in this repository, **not duplicated here**

These are on-topic and have notes in §2, but their PDFs live in sibling folders. Run that folder's
`download.sh` if one is missing.

| paper | path | why it is in this topic |
|---|---|---|
| **TNT** (CoRL 2020) | `../hierarchical_prediction/papers/TNT_Zhao2020_target-driven_trajectory_prediction.pdf` | goal/target-anchored hypotheses; the winner is the **target closest to the GT endpoint** |
| **DenseTNT** (ICCV 2021) | `../hierarchical_prediction/papers/DenseTNT_Gu2021_end-to-end_trajectory_prediction_from_dense_goal_sets.pdf` | replaces the goal argmin with a **dense goal heat-map + a goal-*set* predictor trained on offline-optimiser pseudo-labels** |
| **MTR** (NeurIPS 2022) | `../hierarchical_prediction/papers/MTR_Shi2022_motion_transformer_with_global_intention_localization_and_local_movement_refinement.pdf` | 64 intention queries; **hard assignment of the closest intention point**; 6 refinement layers |
| **MTR++** (TPAMI 2024) | `../hierarchical_prediction/papers/MTRpp_Shi2023_multi-agent_motion_prediction_with_symmetric_scene_modeling_and_guided_intention_querying.pdf` | the same with cross-agent intention querying |
| **MultiPath++** (ICRA 2022) | `../hierarchical_prediction/papers/MultiPathPP_Varadarajan2021_efficient_information_fusion_and_trajectory_aggregation_for_behavior_prediction.pdf` | **learned** latent anchors with the same hard-assignment loss; EM-with-hard-assignment ensembling |
| **Trajectron++** (ECCV 2020) | `../control_normalization/papers/Trajectron++_Salzmann2020_dynamically-feasible_trajectory_forecasting_with_heterogeneous_data.pdf` | the **CVAE** alternative: a discrete categorical latent `|Z| = 25`, ELBO, no WTA |
| **CoverNet** (CVPR 2020) | `../procedure_hard_constraints/papers/covernet_1911.10298.pdf` | multimodality as **pure classification over a fixed trajectory set** |
| **Annealed WTA** (Xu et al. 2025) | `../procedure_hard_constraints/papers/xu2025_annealed_wta.pdf` | the current fix for WTA's initialisation sensitivity: a temperature-annealed soft assignment |
| **ASCENT** (ICRA 2026) | `../procedure_hard_constraints/papers/prutsch2026_ascent_terminal_tp.pdf` | **the aviation answer** — k learnable mode queries, WTA regression + cross-entropy on the mode score |
| **TrajAirNet** (ICRA 2022) | `../hierarchical_prediction/papers/TrajAirNet_Patrikar2021_predicting_like_a_pilot_dataset_and_method_to_predict_socially-aware_aircraft_trajectories.pdf` | the aviation CVAE-sampling baseline ASCENT displaces |
| **Wayformer** (ICRA 2023) | `../hierarchical_prediction/papers/Wayformer_Nayakanti2022_motion_forecasting_via_simple_and_efficient_attention_networks.pdf` | the flat control: k learned queries → one mixture, post-hoc k-means aggregation |
| **MotionLM / Trajeglish / SMART** | `../manoeuvre_tokens/papers/` | the **token-sampling** alternative to a hypothesis set — no anchors, no latent variable, no WTA (§2.18) |

### 1.4 The aviation search — what exists

Searches run 2026-09-18 (arXiv API over title/abstract, plus web search):
`abs:"aircraft trajectory" AND abs:multimodal`; `abs:"aircraft trajectory" AND abs:"multi-modal"`;
`abs:"terminal airspace" AND abs:prediction`; `all:"flight trajectory prediction" AND
all:"multi-modal"`; `all:"trajectory prediction" AND all:"winner-takes-all"` (sorted by date);
`abs:aircraft AND abs:"trajectory prediction" AND abs:"mode queries"`; and web searches for
aircraft/TMA trajectory prediction with "multiple hypotheses" / "winner-takes-all" / tree-structured
output, and for the Pang/Liu Bayesian line.

**Such work exists, but it is thin and recent — one paper does the full K-hypothesis + WTA scheme:**

* **ASCENT** (Prutsch, Schinagl & Possegger, **ICRA 2026**, arXiv:2603.16550) — `k` learnable mode
  queries, a probability head, and **"loss computation is done using a winner-takes-all strategy:
  only the candidate trajectory that best fits the ground truth is considered for optimization
  (lowest L2 distance)"**, with cross-entropy on the score (checked at **p.5**). `k = 5` in the
  ablations. §2.17. **This is the only aircraft paper found that states a WTA loss.**
* **DeepTP** (Liu & Hansen 2018) — the only aviation paper found with a **beam-structured** decode
  (beam width 9 over a 3-component mixture), though it collapses to one output (§2.8).
* Everything else in terminal-airspace prediction expresses uncertainty **distributionally, not as a
  hypothesis set**: TrajAirNet (CVAE sampling, ICRA 2022), Barratt, Kochenderfer & Boyd
  (arXiv:1810.09568, a GMM generative model of terminal-area trajectories), Xiang & Chen
  (arXiv:2409.17359, mixture + seq2seq), the Pang/Liu MC-dropout Bayesian line (§4), FlowATC
  (arXiv:2609.16528, flow matching over ADS-B windows, 2026-09-15), MAIFormer (arXiv:2509.21004),
  Kim, Yoon & Lee (arXiv:2512.08281, probabilistic landing-time prediction). None of these proposes
  a fixed K of competing intents that a min-over-K loss then arbitrates.
* **No aircraft/TMA paper was found that structures its multi-step prediction as a tree of
  branching intents** (TPP's scenario tree has no aviation counterpart in these searches). That
  absence is a real result, not a failed search: the two nearest things are DeepTP's beam (one
  output) and ASCENT's flat set of 5 (one shot, no branching).

---

## 2. Per-paper notes

### 2.1 Multiple Choice Learning — Guzman-Rivera, Batra & Kohli, NIPS 2012

**The origin of the min-over-K objective.** The model is a "multiple-output SSVM": a mapping from
`x` to an **M-tuple** `Ŷᵢ = {ŷᵢ¹, …, ŷᵢᴹ}`, with a mean-field-like factorisation `Φ(xᵢ, Y) =
[φ¹(xᵢ, y¹)ᵀ, …, φᴹ(xᵢ, yᴹ)ᵀ]ᵀ`, so that `g` is simply **M independent single-output predictors**
(checked at **p.3**, §3.1). The loss is the **oracle / hindsight set-loss** `L(Ŷᵢ) = min_m ℓ(yᵢ,
ŷᵢᵐ)` — "the set of predictions `Ŷᵢ` only pays a loss for the most accurate prediction contained in
this set" (checked at **p.3–4**, §3.2). The authors name the consequence themselves: the set-loss is
"rather poorly conditioned — if even a single prediction in the ensemble is the ground-truth, the
set-loss is 0, no matter what else is predicted" (**p.4**).

**Optimisation and selection.** The hinge upper bound introduces a binary flag `ρ_{i,m} = 1 if m =
argmin ℏ(w)`, with `Σ_m ρ_{i,m} = 1`, and is minimised by alternating between assigning `ρ` and
re-fitting the predictors — the authors describe the result as "a structured-output version of
**k-means** clustering" (checked at **p.2** and **p.4**). A generalisation replaces the
"pick-one-predictor" constraint with **pick-K**, "where K is a robustness parameter" (checked at
**p.5**). **There is no mode-probability head and no selection rule**: MCL assumes a downstream
oracle (a user, or a re-ranker in a cascade) chooses. Tasks are interactive foreground/background
image segmentation (M = 6 in Fig. 1, **p.6**) and protein side-chain prediction.

*(reading)* everything later in this folder is this loss plus the piece MCL deliberately omits — a
learned way to choose among the M outputs.

### 2.2 MHP — Rupprecht et al., ICCV 2017

**What it adds to MCL: a theory of what the hypotheses converge to, and a relaxation that makes SGD
work.** The framework turns any single-output CNN into an M-head one. Minimising the multiple-output
objective "yields a **Voronoi tessellation** in the output space that is induced by the chosen loss"
(checked at **p.1**, abstract); Theorem 1 states that at the minimum the predictors coincide with the
generators of a **centroidal** Voronoi tessellation, i.e. `f_θ^j` predicts "the conditional mean of
the Voronoi cell it defines" (checked at **p.3**). Training is a **meta loss** over the base loss,
`M(f_θ(xᵢ), yᵢ) = Σ_{j=1..M} δ(yᵢ ∈ Y_j(xᵢ)) L(f_θ^j(xᵢ), yᵢ)` (Eq. 11, checked at **p.4**), which
the authors read as an EM / Lloyd's-method step: E-step assigns the label to a cell, M-step updates
that head.

**The relaxation is the practical contribution.** Hard `δ` starves heads whose initialisation puts
all labels in one cell: "only the k-th generator `f_θ^k(x)` gets updated since `δ(yᵢ ∈ Y_j(xᵢ)) = 0,
∀j ≠ k`" — so `δ` is relaxed with a weight `0 < ε < 1`, and **"in all experiments we set the
association relaxation to ε = 0.05"** (checked at **p.4**). The cost is visible in the outputs: the
ε term "pulls the predictions slightly towards the conditional average", producing faint "ghost-cars"
for non-selected exits (checked at **p.7**). Inference: one forward pass, M hypotheses, **no
probability head** — four tasks (human pose, future prediction, image classification, segmentation),
each reported against its own selection convention.

### 2.3 Evolving WTA — Makansi, Ilg, Çiçek & Brox, CVPR 2019

**The scheduled version of the winner rule, plus a fitting stage that turns hypotheses into a
distribution.** The diagnosis: WTA suffers "inconsistency problems" and MDNs trained directly
"suffer from instabilities in training and mode collapse" (checked at **p.1–2**). **EWTA**: "we
update the **top-k** winners… k weights are 1, while M − k weights are 0. We start with `k = M` and
then decrease `k` until `k = 1`" — each decrement "releases" a hypothesis from an equilibrium so it
can pair with a different ground truth (checked at **p.4**, Fig. 3c). Two hypothesis parameterisations
are tried: a point estimate with Euclidean loss, and `h_k = (µ_k, σ_k)` with an NLL loss (**p.4**).

**Stage 2 is the selection mechanism.** A second network **fits a mixture to the hypotheses** by
predicting soft assignments `γ_k = softmax(z_k)` of each hypothesis to `K` mixture components, giving
mixture weights, means and scales (checked at **p.5**); components are Laplace or Gaussian with
x/y treated as independent (**p.3**). The whole thing is **one forward pass**, one-shot in time
(the future is predicted as a whole, not autoregressively). Evaluation is on the synthetic **Car
Pedestrian Interaction (CPI)** dataset with Earth Mover's Distance against a known ground-truth
distribution, and on the **Stanford Drone Dataset** where only one sample of the true distribution
exists (checked at **p.2**).

### 2.4 MultiPath — Chai, Sapp, Bansal & Anguelov, CoRL 2019

**The anchor form of WTA.** Intent is a distribution over `K` **fixed** anchor trajectories `A =
{a_k}`, `π(a_k|x) = exp f_k(x) / Σ exp f_i(x)`; given an intent, control uncertainty is a Gaussian
per waypoint, `φ(s_t^k | a_k, x) = N(s_t^k | a_t^k + µ_t^k(x), Σ_t^k(x))`, so `µ` is "a
scene-specific **residual** on top of the prior anchor" (checked at **p.3**, §3). The anchors are
obtained **before** training — k-means over training trajectories under a rotation/translation-
invariant distance, or, where k-means gives redundant clusters, "uniformly sampling trajectory
space" (checked at **p.4**).

**The winner rule, stated as a design choice.** The loss is a GMM NLL in which the anchor index is
**hard-assigned**: `1(k = k̂ₘ)`, where "`k̂ᵐ` is the index of the anchor most closely matching the
groundtruth trajectory `ŝᵐ`, measured as ℓ2-norm distance in state-sequence space. This
**hard-assignment of groundtruth anchors** sidesteps the intractability of direct GMM likelihood
fitting, avoids resorting to an expectation-maximization procedure, and gives practitioners control
over the design of the [anchor set]" (checked at **p.4**). Inference is **one forward pass**: a
discrete distribution over anchors plus per-anchor, per-timestep Gaussians; the compact answer is
"the MAP sample from each anchor-intent" (**p.2**).

**The measured argument against a free (anchor-free) min-of-K.** MultiPath's own baseline "Min-of-K"
predicts K trajectories directly with an ℓ2 loss on the closest one: "similar to our method, but
with implicit anchors and **evolving hard-assignment** of anchors to groundtruth as training
progresses. This representation has inherent ambiguity problems and can suffer from mode collapse"
(checked at **p.5**). On the 3-way-intersection toy problem, "Min of K = 5 … is very sensitive to
initial weights, and on 5 trials with 4 learning rates, **collapsed to only 1 or 2 active modes**"
(checked at **p.6**, Fig. 2e). Sizes used: **K = 16** anchors for the `µ, Σ` model, **K = 64** best
for the `µ`-only model (checked at **p.7**); an anchor-count sweep is in App. B.1 and the anchor sets
for K = 1…128 are plotted in Fig. 9 (**p.14**).

### 2.5 DESIRE — Lee, Choi, Vernaza, Choy, Torr & Chandraker, CVPR 2017

**Sample, rank, refine — and note that the training loss is *not* a min-over-K.** A CVAE-based
RNN encoder–decoder generates `K` future samples `Ŷᵢ^(1..K)` by drawing `zᵢ` from the prior and
masking the past encoding, `H_{Xᵢ} ⊙ β(zᵢ)` (checked at **p.4**, §3.1). The two training losses are
the **mean** reconstruction error over the K samples, `ℓ_Recon = (1/K) Σ_k ‖Yᵢ − Ŷᵢ^(k)‖`, and the
KL term (checked at **p.4**) — diversity comes from the latent variable, not from a winner rule.

**Selection is a learned scorer, not a probability head.** The Ranking and Refinement module "assigns
a reward to the prediction samples at each time-step sequentially as IOC frameworks and learns
displacements vector `ΔŶ` to regress the prediction hypotheses… **The final prediction is the sample
with the maximum accumulated future reward**" (checked at **p.4**, Fig. 2 caption). Refinement is
**iterated**: model variants are named `DESIRE-{S,SI}-IT{0,1,4}`, i.e. up to **4** feedback
iterations (checked at **p.7**, Fig. 5 and the results tables). Inputs are past tracks, semantic
scene context and the other agents; outputs are K ranked, refined 4 s trajectories, evaluated as
oracle error over the top-K on **KITTI raw** and the **Stanford Drone Dataset** (checked at **p.1**,
**p.6–7**).

### 2.6 mmTransformer — Liu, Zhang, Fang, Jiang & Zhou, CVPR 2021

**Hypotheses as decoder queries, and a *spatial* winner rule instead of a per-sample argmin.** Each
of the `K` decoder queries is a **trajectory proposal** that "asymptotically aggregate[s] multiple
channels of contextual information from encoders, and make[s] independent predictions" (checked at
**p.2**). Two heads follow: a Trajectory Generator (regression) and a **Trajectory Selector**
producing "the K confidence scores" (checked at **p.4**).

**Region-based training strategy (RTS).** The space around the target's last observed point is
partitioned into `M` non-overlapping regions; the `K` proposals are split evenly, `N = K/M` per
region; "during training, **only the set of proposals assigned to the region where ground truth
locates** will be utilized to optimize the framework", which "enforces individual proposal to focus
on a specific mode, without compromising the latent features learned by other proposals" (checked at
**p.2** and **p.5**). Losses: Huber regression over the selected subset, a KL-divergence confidence
loss, and an auxiliary region-classification loss, combined with learned task weights (Eq. 7,
**p.5**).

**Numbers.** Argoverse, K = 6 outputs. Plain mmTransformer uses **6 proposals**; with RTS it uses
**36 proposals over 6 regions** (6 per region). RTS moves the **miss rate from 17.6 % to 9.2 %**,
while minADE/minFDE drop slightly because "limited by the fixed number (6) of final outputs, we
discard the redundant candidate proposals to retain the diversity (i.e. MR)" (checked at **p.6–7**).
Prediction is one-shot per proposal.

### 2.7 Tree Policy Planning — Chen, Karkus, Ivanovic, Weng & Pavone, ICRA 2023

**The tree here is over *stages*, and it is a planner, not a predictor.** TPP builds two trees: an
**ego trajectory tree** of ego motion candidates and a **scenario tree** of "multi-modal
ego-conditioned environment predictions", then solves the resulting finite-horizon MDP by **dynamic
programming** to get a *policy* — "which node to execute for the next stage given the current ego
and environment [state]" (checked at **p.1–2** and **p.4**). Node `r_i^j` is the j-th node of stage
i; all nodes of a stage carry trajectories of the same duration (**p.2**).

**Requirements it puts on the prediction model** (checked at **p.3**, §III-B): the model must predict
**multiple modes**, be **scene-centric** (modes of the joint distribution, so a branch means a joint
future), be **ego-conditioned**, and be **multi-stage** — "each node branching into multiple children
after each stage". A **causal consistency** condition is defined and proved necessary: two scenario
trees under different ego modes that share the ego trajectory up to stage i must be identical up to
stage i (checked at **p.4**). Branching is capped: "We set an upper bound on the number of children
a node can have, and randomly drop children nodes if the limit is exceeded" (**p.3**).

**Configuration.** Three interchangeable prediction models (a rasterised CNN+CVAE, PredictionNet,
and **AgentFormer**), all modified for multi-stage ego-conditioned prediction; nuScenes; **"All
prediction models use 2 stages and the branching factor is 4"** (checked at **p.5**). Branch
probabilities come from the prediction model, so **TPP contributes no winner rule of its own** — it
contributes the selection machinery (DP over a tree) that a K-hypothesis predictor would feed.

### 2.8 DeepTP — Liu & Hansen 2018 (aviation, beam-structured)

**The one aviation paper in this folder with a search over futures.** Track points are modelled as
**conditional Gaussian mixtures** whose parameters come from an LSTM encoder (over the last-filed
flight plan) and a **mixture-density LSTM decoder**, with convolutional layers over wind, temperature
and convective-weather feature cubes (checked at **p.1**, abstract). **K = 3** mixture components are
used, so "at each timestamp" the decoder emits three weighted Gaussians (checked at **p.17**).

**Selection.** Because the number of sequences grows as `K^T`, "we rank `L_t^i` in a decreasing order
and only keep the largest `N_BS` sequences (a.k.a., **beam search**)"; in the experiments
**`N_BS = K² = 9`**, and the model "only output[s] **one trajectory** with the highest log
likelihood" (checked at **p.16** and **p.18**). Adaptive Kalman filtering and an RTS smoother then
"prune the variance of generated trajectories" (**p.1**). Data: **IAH → BOS, 2013**, FAA TFMS tracks
plus last-filed flight plans, **1,679 flights**, average sequence length 94 (checked at **p.5**,
**p.18**).

*(reading)* mechanically this is the closest published aviation analogue of a tree-structured
multi-hypothesis decode — but the hypotheses are **per-step mixture components of one distribution**,
not competing intents, and the beam is collapsed to a single answer, so nothing downstream ever sees
K futures.

### 2.9 Pang, Xu & Liu 2019 (aviation, single-hypothesis — the corrected entry)

The **Pang–Xu–Liu** paper the brief remembered is not Bayesian. It embeds convolutional layers
**inside** the LSTM recurrence — "we add two convolutional layers and two dense layers before the
computing of forget gate", with `h_x = h_t ⊕ x_t ⊕ x_dense2` (checked at **p.6**) — to fuse **Echo
Top** convective-weather cubes (NASA Sherlock database) with the last on-file flight plan. "The
training loss is defined as the standard **Mean Squared Error** of the predicted tracks and the real
history tracks" (checked at **p.1**). **One trajectory out, no hypotheses, no uncertainty.** Result:
"47.0 % of the deviation between flight tracks are reduced in our trained model… the overall variance
of deviation reduction is 12.3 %" (checked at **p.1** and **p.7**). Open access, CC-BY 3.0 US.

### 2.10 TNT — Zhao et al., CoRL 2020 *(cross-reference, §1.3)*

Three stages trained end-to-end: (a) **target prediction** proposing M targets, (b) target-conditioned
motion estimation, (c) **scoring and selection** (checked at **p.2–3**). The winner rule lives in
stage (a): the loss is `L_cls` (cross-entropy) + `L_offset` (Huber) where "**u is the target closest
to the ground truth location**" (checked at **p.4**) — a min-over-candidates on the *endpoint*, not
the whole trajectory. Candidates are over-sampled, "e.g. **N = 1000**" targets from lane polylines or
a grid (checked at **p.5**), trajectories are completed one-shot per target, and the final K is
chosen by the stage-3 likelihood with NMS-style suppression. Input: VectorNet-style polyline context;
output: K scored trajectories.

### 2.11 DenseTNT — Gu, Sun & Zhao, ICCV 2021 *(cross-reference, §1.3)*

Removes both the sparse-goal argmin and the hand-made selection heuristic. A goal encoder outputs a
**dense probability distribution over road points** (a heat-map), and a **goal-set predictor** emits
the final K goals directly instead of running NMS, which the authors call "a greedy algorithm"
(checked at **p.2–3**). Because no ground-truth *set* exists, the goal-set predictor is trained on
**pseudo-labels from an offline optimisation-based model**: "we devise an offline model to provide
multi-future pseudo-labels for our online model… the optimization algorithm finds an optimal goal set
from the [dense goals]" (checked at **p.1–3**). One-shot trajectory completion per goal;
input/output otherwise as TNT.

### 2.12 MTR and MTR++ — Shi, Jiang, Dai & Schiele, NeurIPS 2022 / TPAMI 2024 *(cross-reference)*

MTR keeps a **fixed set of 64 learnable motion query pairs**, each tied to a k-means **intention
point**, "instead [of] … a large number of queries" (checked at **p.5**, "by default, we utilize 64
motion [query pairs]", **p.7**). The output per decoder layer is a **GMM** over positions, `Z_ij ∈
R^{K×6}` (checked at **p.6**). The winner rule is explicit: "we adopt a **hard-assignment strategy
that selects one closest motion query pair as positive Gaussian component for optimization**, where
the selection is implemented by calculating the distance between each intention point and the
endpoint of GT trajectory" — trained with a Gaussian NLL (checked at **p.6**). Prediction is
**iterated**: 6 stacked decoder layers each re-query the map polylines "whose centers are closest to
the predicted trajectory" (**p.5**), i.e. multi-*pass* refinement of one-shot trajectories, not a
tree over time. MTR++ extends the same scheme with mutually-guided intention querying across agents.

### 2.13 MultiPath++ — Varadarajan et al., ICRA 2022 *(cross-reference, §1.3)*

Keeps MultiPath's output GMM and its winner rule — "We follow the original MultiPath approach and
maximize the likelihood of the groundtruth trajectory under our model's predicted distribution. We
make a **hard-assignment labeling of a 'correct' mixture [component]**" (checked at **p.7**) — but
replaces the static k-means anchors with **learned anchor embeddings** `e_{1:M}`, "trainable model
parameters that are independent of the input", in one-to-one correspondence with the GMM modes and
explicitly compared to DETR's learned queries (checked at **p.8**). Ensembling several such models is
non-trivial precisely because learned anchors do not correspond across models, so the outputs are
merged by "an iterative clustering algorithm, like Expectation-Maximization, but with **hard
assignment**" (checked at **p.9**). One-shot; input is raw continuous agent/road state rather than a
raster.

### 2.14 Trajectron++ — Salzmann, Ivanovic, Chakravarty & Pavone, ECCV 2020 *(cross-reference)*

The **non-WTA** control in this folder. Multimodality comes from "a **discrete Categorical latent
variable** `z ∈ Z` which encodes high-level behavior modes", with **|Z| = 25** (checked at **p.7**);
training maximises an ELBO, and because the latent space is small the expectation "can be computed"
exactly rather than sampled (**p.8**). Selection at inference is a menu rather than a rule: a **"Most
Likely (ML)"** deterministic output where "the high-level latent behavior mode and output trajectory
are the modes" of their distributions, z-mode sampling, and the full distribution (checked at
**p.8**). The decoder emits **control actions** integrated through the agent's dynamics, which is why
this PDF lives in `control_normalization/`.

### 2.15 CoverNet — Phan-Minh, Grigore, Boulton, Beijbom & Wolff, CVPR 2020 *(cross-reference)*

The degenerate case of the scheme: **no regression at all**. Multimodal prediction is framed "as
classification over a trajectory set", which "avoids mode collapse and lets the user design the
trajectory set to meet specific requirements (e.g. **dynamically feasible, coverage guarantees**)"
(checked at **p.1**). The winner rule is the labelling: "cross-entropy with **positive samples
determined by the element in the trajectory set closest to the actual ground truth in minimum average
of point-wise Euclidean distances**" (checked at **p.5**). Inference selects by the class
probabilities; the set is either fixed or generated dynamically from the agent's current state, with
an ε-coverage bound on the set size (**p.1–3**). One-shot.

### 2.16 Annealed WTA — Xu, Letzelter, Chen, Zablocki & Cord, 2025 *(cross-reference, §1.3)*

The current repair for the failure MultiPath measured in 2019. The diagnosis: "motion forecasting
models trained with WTA and a small number of hypotheses (e.g. **6 hyp.**) suffer from **mode
collapse**", and the usual workaround — train with many hypotheses and post-select — is what they
want to remove (checked at **p.1**). The fix replaces the hard indicator with a Boltzmann soft
assignment `q_t ∝ Z⁻¹ exp(−ℓ(f_k(x), y) / T(t))` (checked at **p.2**, table of loss variants): at
high temperature "the predictions-target assignment is very soft… the hypotheses converge toward a
conditional mean, and the effective number of hypotheses is equal to 1"; as `T` decays, `L_t`
"converges to the standard WTA training objective, where `lim_{t→∞} q_t = 1[k ∈ argmin_s ℓ(f_s(x),
y)]`" (checked at **p.3**). Stated result: state-of-the-art forecasters keep their performance with
a **minimal** hypothesis set and **no test-time selection stage**.

### 2.17 ASCENT — Prutsch, Schinagl & Possegger, ICRA 2026 *(cross-reference — the aviation answer)*

**The aviation paper that does the whole scheme.** Problem statement, in the paper's own terms:
"Since future motion can correspond to different maneuvers, e.g., landing or turning, models
typically predict a set of `k` trajectory candidates, resulting in multimodal outputs `F ∈
R^{k×T_f×3}` with associated probability scores `S ∈ R^k`" (checked at **p.2**). The decoder is an
MLP over `k` **learnable mode queries** `Q ∈ R^{k×D}` added to the broadcast agent feature, with a
probability head (checked at **p.4**); positions are not predicted directly but through **flight
parameters** (speed, yaw, pitch) integrated kinematically.

**Loss and selection.** "Loss computation is done using a **winner-takes-all strategy: only the
candidate trajectory that best fits the ground truth is considered for optimization (lowest L2
distance)**. We employ a smooth L1 loss as regression loss and use a standard **cross-entropy loss as
classification loss, i.e., the best-fitting trajectory is assigned the highest probability score**"
(checked at **p.5**). Metrics are `ADE_k`/`FDE_k` **in kilometres**, "reported using the best-fitting
prediction out of the `k` candidates" (**p.5**); ablations use **`k = 5` hypotheses**, an 11 s history
and a **120 s** horizon (checked at **p.6**). Data: **TrajAir** (KBTP, non-towered) and
**TartanAviation** (KBTP + towered KAGC), ADS-B referenced to a static receiver. The decoder ablation
compares a **CVAE-based decoder "similar to the one used in TrajAirNet" against the mode-query
decoder** and finds the mode-query approach better (checked at **p.7**). Prediction is **one-shot**:
k candidates in one pass, no branching, no re-query.

*(reading)* this is a general-aviation, non-towered, 120 s setting — not a commercial TMA final
approach — and the hypotheses are unconditioned mode queries, not runway/procedure intents. The
mechanism transfers; the numbers do not.

### 2.18 MotionLM / Trajeglish / SMART *(cross-reference to `../manoeuvre_tokens/`)*

The token-sampling line is the **alternative to a hypothesis set**, and MotionLM says so directly:
the model "does not require anchors or explicit latent variable optimization" and is "entirely latent
variable and anchor-free, with **multimodality emerging solely as a characteristic of sampling**"
(checked at **p.1–2** of the MotionLM PDF). Where this folder's papers commit to K outputs and
arbitrate them with a loss, MotionLM draws `R` rollouts and recovers "representative modes … via a
simple aggregation utilizing **k-means clustering initialized with non-maximum suppression**"
(checked at **p.3**) — the mode set is produced *after* inference, not defined before training. Same
for Trajeglish and SMART (pure next-token cross-entropy). See `../manoeuvre_tokens/README.md` §2.1–2.3
for the tokenisation details.

---

## 3. What the folder says about the proposed design

Cross-cluster observations, each traceable to §2. Pointers for the design doc, not an assessment of
it.

1. **The winner rule splits cleanly into three families, and they are not interchangeable.** (i)
   **Fixed hypothesis set + closest-member assignment**: MultiPath (anchor with min ℓ2, §2.4),
   CoverNet (closest set element, §2.15), TNT (target closest to the GT endpoint, §2.10), MTR
   (closest intention point to the GT endpoint, §2.12). (ii) **Free hypotheses + argmin over the
   heads**: MCL (§2.1), MHP (§2.2), EWTA (§2.3), ASCENT (§2.17), MultiPath++ (learned anchors but
   still a hard assignment, §2.13). (iii) **No winner rule at all**: CVAE/ELBO (Trajectron++ §2.14,
   DESIRE §2.5) and token sampling (§2.18). *(reading)* family (i) needs a defensible way to build
   the set — in our setting the natural set is runway × procedure, which the other domains do not
   have.
2. **Free min-over-K collapses, and the amount is measured.** MultiPath's own Min-of-K baseline
   "collapsed to only 1 or 2 active modes" on 5 trials × 4 learning rates (§2.4, p.6); Rupprecht
   needs ε = 0.05 to stop heads starving (§2.2); Makansi's whole contribution is the `k: M → 1`
   schedule (§2.3); Xu et al. 2025 still report mode collapse at 6 hypotheses and fix it with an
   annealed temperature (§2.16). **Anyone building family (ii) should start from aWTA or EWTA, not
   from plain WTA** — every paper that used plain WTA reported the same failure.
3. **The mode-probability head is a separate, simpler decision.** Cross-entropy on the winner is the
   near-universal choice (ASCENT §2.17, CoverNet §2.15, TNT stage 1 §2.10); mmTransformer uses a KL
   confidence loss instead (§2.6); MTR and MultiPath fold it into the GMM's mixture weights (§2.12,
   §2.4). Only DESIRE replaces it with a **learned scorer over accumulated reward** (§2.5), and only
   MCL leaves selection to the downstream consumer (§2.1).
4. **Almost everything is one-shot; the two exceptions are different things.** MTR's "iterative"
   refinement is 6 passes over the *same* one-shot trajectory (§2.12); TPP's tree is a genuine
   branching over **stages** (2 stages, branching factor 4, §2.7) but comes from a planner, with the
   branch probabilities supplied by an ordinary multimodal predictor. **No paper here trains a
   per-segment tree of intents end-to-end.** *(reading)* a per-segment K-intent proposal with
   branching is not an incremental variant of any of these — it is TPP's tree structure moved from
   the planner into the predictor.
5. **The aviation gap is specific and worth stating in the thesis.** Exactly one aircraft paper
   (ASCENT, ICRA 2026, §2.17) uses K mode queries with a WTA + cross-entropy loss, and it is
   general-aviation, non-towered, 120 s, k = 5, one-shot. Exactly one (DeepTP 2018, §2.8) searches a
   beam, and it collapses to a single output. Everything else in terminal-airspace prediction is
   distributional (§1.4). **There is no published aircraft/TMA model that proposes K competing
   intents per segment and arbitrates them with a min-over-K loss.**
6. **Two selection mechanisms exist that a probability head does not give you.** DenseTNT's
   **offline optimiser as a pseudo-label generator** for a set-valued output (§2.11) and DESIRE's
   **IOC scorer** (§2.5) both answer "which of these is best" with something other than the model's
   own softmax. *(reading)* for arrivals, a feasibility/regulation score is exactly this kind of
   external scorer, and DenseTNT is the precedent for training a set predictor against an optimiser
   rather than against a label.
7. **Anchors can be learned instead of clustered, and the measured result went the other way from
   the original claim.** MultiPath++: "We find that learning anchors … is more effective than using
   a set of anchors obtained a priori via k-means. This **runs counter to the original findings in
   the MultiPath paper** that anchor-free models suffer from mode collapse" (checked at **p.14** of
   the MultiPath++ PDF; the WOMD numbers for that comparison are tabulated in
   `../hierarchical_prediction/README.md` §1). Both the fixed-set and learned-query routes are
   therefore attested; the loss is the same either way.

---

## 4. Not fetched

| source | issue | status |
|---|---|---|
| **Pang & Liu**, *Probabilistic Aircraft Trajectory Prediction Considering Weather Uncertainties Using Dropout as Bayesian Approximate Variational Inference*, **AIAA SciTech 2020**, AIAA 2020-1413 | AIAA ARC paywall; no preprint found (searched arXiv, Google Scholar, ResearchGate, the ASU repository) | **not fetched**; cited here for the record. Method is **MC-dropout** — a predictive *distribution*, not a hypothesis set |
| **Pang, Zhao, Yan & Liu**, *Data-driven trajectory prediction with weather uncertainties: A Bayesian deep learning approach*, **Transportation Research Part C 130 (2021) 103326** | Elsevier paywall; no open postprint found | **not fetched**; same note as above |
| **Pang, Yao, Hu & Liu**, *A Recurrent Neural Network Approach for Aircraft Trajectory Prediction with Weather Features From Sherlock*, **AIAA Aviation 2019**, AIAA 2019-3413 | AIAA ARC paywall | **not fetched**; the PHM 2019 paper by Pang, **Xu** & Liu (§2.9) is the open member of this group and is in `papers/` |

Everything in §1.2 is in `papers/`. The thirteen sources in §1.3 are deliberately **not** duplicated
here; each is cited by its path in a sibling folder, and `download.sh` lists which folder's script to
run if one is missing.

Three verification notes that are easy to re-discover the hard way:

* **DESIRE is not a WTA method.** Its reconstruction loss is the **mean** over the K samples
  (`ℓ_Recon = (1/K) Σ_k ‖Yᵢ − Ŷᵢ^(k)‖`, §2.5, p.4). It is routinely cited alongside MultiPath and TNT
  as a "multimodal baseline", which is true, but the multimodality is CVAE sampling and the selection
  is the IOC scorer — there is no min-over-K anywhere in it.
* **"Min-of-K" is not MultiPath.** MultiPath is the *anchored* version; the free min-of-K it names
  and beats is reference [20] in that paper (§2.4, p.5). Quoting MultiPath as the source of the
  free-WTA trajectory loss inverts its argument.
* **MCL never predicts a mode probability.** The 2012 formulation deliberately stops at the M-tuple
  and assumes a downstream oracle (§2.1). The probability head that every modern user adds is not
  part of the original loss, and the "+ mode-probability loss" half of the design has its own
  independent lineage (cross-entropy on the winner, first in this folder at MultiPath and CoverNet).
