# Discrete manoeuvre tokens for arrival trajectories — reading list (2026-09-18)

Why this folder exists. A design being written for `4dTrajectory/ts_transformer` proposes to
quantise each **60 s segment** of an arrival trajectory into a **discrete manoeuvre code** (a
learned or clustered codebook), run a causal transformer "prior" as **next-token prediction** over
the code sequence (context tokens: candidate runways, procedure fixes, aircraft type, a predicted
runway token), fly one code at a time with a **learned low-level controller** through a
differentiable point-mass rollout, and later decode **several aircraft jointly** with cross-agent
attention whose edges carry separation constraints (a constraint = **a mask over the vocabulary at
decode time**). The question this folder answers sources for:

> For each piece of that scheme, **what has already been built, how was it parameterised, and what
> did its authors measure?** Specifically: what quantity gets quantised and at what time step; how
> the codebook is learned (or not learned); whether a continuous residual rides on top of the code;
> how several agents are decoded inside one time step; how an edge feature and a hard constraint
> enter a transformer; and what a "manoeuvre" or "flow" is worth in terminal airspace.

This folder is a **mechanism** reading list, not an assessment. Every number below was read out of
the fetched PDF, not the abstract. Conventions: formulas in backticks are quoted as the source
writes them (symbols transliterated to Unicode); "checked at" says where the claim was read;
**(reading)** marks my inference, not the source's claim; "not stated" means the paper does not say.

The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`); `./download.sh` re-fetches all
13 (tested 2026-09-18: three deleted files came back byte-identical by md5). §4 lists what could
not be fetched — this time, nothing. **DAgger** (Ross et al., arXiv 1011.0686) is cited in §3 but
not re-downloaded: it already lives at
`../hierarchical_prediction/papers/DAgger_Ross2011_reduction_of_imitation_learning_and_structured_prediction_to_no-regret_online_learning.pdf`.
MotionLM was **not** in that folder and is fetched here.

---

## 1. Sources

All nine arXiv identifiers supplied in the brief were checked against the arXiv API abstract record
and **every one was correct**; no correction was needed. Venues below are as the arXiv record or
the PDF states them.

### 1.1 Cluster A — motion tokens and next-token prediction over trajectories

| key / file | citation | token: what is quantised | vocab | step | residual? | intra-step agents |
|---|---|---|---|---|---|---|
| `MotionLM_Seff2023…` | Ari Seff, Brian Cera, Dian Chen, Mason Ng, Aurick Zhou, Nigamaa Nayakanti, Khaled S. Refaat, Rami Al-Rfou, Benjamin Sapp, *MotionLM: Multi-Agent Motion Forecasting as Language Modeling*, **ICCV 2023**, arXiv:2309.16534, https://arxiv.org/abs/2309.16534 | axis-aligned `(Δx, Δy)` between consecutive waypoints, agent frame at `t = 0`, **uniform** bins + a Verlet wrapper | **169** = 13² | **0.5 s (2 Hz)** | no | **parallel** (conditionally independent) |
| `Trajeglish_Philion2023…` | Jonah Philion, Xue Bin Peng, Sanja Fidler, *Trajeglish: Traffic Modeling as Next-Token Prediction*, **ICLR 2024**, arXiv:2312.04535, https://arxiv.org/abs/2312.04535 | the **state-to-state transition** `(Δposition, Δheading)` in the frame of the most recent state; template set found by **k-disks** | **384** | **0.1 s (10 Hz)** | no | **sequential** under a random permutation |
| `SMART_Wu2024…` | Wei Wu, Xiaoxin Feng, Ziyan Gao, Yuheng Kan, *SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction*, **NeurIPS 2024**, arXiv:2405.15677, https://arxiv.org/abs/2405.15677 | a **0.5 s trajectory segment**, k-disks over the dataset's segment set; separate vocabulary per agent class | **512 / 1024 / 1024 / 2048** by model size | **0.5 s** | **no** (pure cross-entropy) | per-step attention |

### 1.2 Cluster B — how the codebook is learned

| key / file | citation | what it contributes here |
|---|---|---|
| `VQVAE_vandenOord2017…` | Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu, *Neural Discrete Representation Learning*, **NeurIPS 2017**, arXiv:1711.00937, https://arxiv.org/abs/1711.00937 | the original commitment loss + straight-through estimator; the "collapse" it discusses is **posterior** collapse, not codebook collapse |
| `FSQ_Mentzer2023…` | Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen, *Finite Scalar Quantization: VQ-VAE Made Simple*, **ICLR 2024**, arXiv:2309.15505, https://arxiv.org/abs/2309.15505 | drop the codebook and the auxiliary losses entirely: bound + round each of `d` channels to `L_i` levels; **≈100 % codebook usage by construction** |
| `VQBeT_Lee2024…` | Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, Lerrel Pinto, *Behavior Generation with Latent Actions*, **ICML 2024** (PMLR 235:26991–27008), arXiv:2403.03181, https://arxiv.org/abs/2403.03181 | the one paper here that tokenises an **action chunk** and adds a **continuous offset head** on top of the code |

### 1.3 Cluster C — edges in attention, rollout exposure, constrained decoding

| key / file | citation | what it contributes here |
|---|---|---|
| `Graphormer_Ying2021…` | Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu, *Do Transformers Really Perform Bad for Graph Representation?*, **NeurIPS 2021**, arXiv:2106.05234, https://arxiv.org/abs/2106.05234 | the mechanism for putting a **per-pair edge feature into the attention logit as an additive bias** |
| `ScheduledSampling_Bengio2015…` | Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer, *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*, **NeurIPS 2015**, arXiv:1506.03099, https://arxiv.org/abs/1506.03099 | the canonical fix for train/inference mismatch in an autoregressive rollout |
| `GCD_Geng2023…` | Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West, *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*, **EMNLP 2023 (Main)**, arXiv:2305.13971, https://arxiv.org/abs/2305.13971 | "constraint = mask over the vocabulary at decode time", stated formally and with no finetuning |

### 1.4 Cluster D — what a manoeuvre / flow is in terminal airspace, and aircraft tokens

| key / file | citation | what it contributes here |
|---|---|---|
| `OliveMorio2019…` | Xavier Olive, Jérôme Morio, *Trajectory clustering of air traffic flows around airports*, **Aerospace Science and Technology 84** (2019) 776–781, DOI `10.1016/j.ast.2018.11.031`, ONERA postprint HAL `hal-02350789` (fetched via Wayback, see `download.sh`) | clusters **points, not trajectories**, and chains the clusters into a dependency tree — the closest published thing to a "gate sequence" vocabulary |
| `Corrado2020…` | Samantha J. Corrado, Tejas G. Puranik, Olivia J. Pinon, Dimitri N. Mavris, *Trajectory Clustering within the Terminal Airspace Utilizing a Weighted Distance Function*, **Proceedings 2020, 59(1), 7** (8th OpenSky Symposium, 12–13 Nov 2020), DOI `10.3390/proceedings2020059007`, https://www.mdpi.com/2504-3900/59/1/7 | **exists, open access** (the brief asked to verify); whole-trajectory clustering with a per-point **weighted** Euclidean distance + HDBSCAN |
| `TimeVQVAE-ATM_Murad2025…` | Abdulmajid Murad, Massimiliano Ruocco, *Synthetic Aircraft Trajectory Generation Using Time-Based VQ-VAE*, **ICNS 2025** (25th Integrated Communications, Navigation and Surveillance Conf., Brussels, 8–10 Apr 2025), arXiv:2504.09101, https://arxiv.org/abs/2504.09101 | **the closest aircraft analogue of the proposed scheme**: VQ codebooks over flight-trajectory latents + a transformer prior over the discrete codes |
| `LLM-FTP_Luo2025…` | Kaiwei Luo, Jiliu Zhou, *Large Language Models for Single-Step and Multi-Step Flight Trajectory Prediction*, arXiv:2501.17459 (Sichuan University; no venue stated on the PDF), https://arxiv.org/abs/2501.17459 | the other hit for "flight trajectory next-token" — but its tokens are **BPE sub-tokens of printed digits**, not manoeuvre codes. Kept as the **negative control** |

Searches run for §1.4 additions (arXiv API, 2026-09-18): `"flight trajectory" AND "token"`;
`"aircraft trajectory" AND (tokenization | tokenizer | "discrete tokens" | next-token)`;
`"air traffic" AND ("vector quantization" | VQ-VAE | codebook)`;
`("flight" | "aircraft") AND "maneuver primitives"`; `"ADS-B" AND "tokens"`;
`trajectory AND "discrete tokens" AND (aviation | airspace | flight)`. Only the two papers above
came back with a fetchable, verifiable PDF, so only two were added (the brief's cap was two).

---

## 2. Per-paper notes

### 2.1 MotionLM — Seff et al., ICCV 2023

**Tokenisation.** Continuous waypoints are turned into discrete tokens by "a simple uniform
quantization of axis-aligned deltas between consecutive waypoints of agent trajectories"
(checked at **p.3**, §2 *Discrete sequence modeling in continuous domains*). Each agent's ground
truth is first normalised to that agent's position and heading at `t = 0` of the scenario; a
per-coordinate uniform `(Δx, Δy)` vocabulary is then parameterised by a bin count and min/max delta,
and a **greedy search** picks, step by step, the quantised action that reconstructs the next
waypoint with minimum error (checked at **p.4–5**, §3.2.2 *Quantization*).

**Numbers** (checked at **p.13**, Appendix A *Motion token vocabulary*): step frequency **2 Hz**;
delta interval per step **[−18.0 m, +18.0 m]**; **128 bins**. "At 2 Hz prediction, a maximum delta
magnitude of 18 m covers axis-aligned speeds up to 36 m/s (~80 mph), > 99 % of the WOMD dataset."
A **Verlet wrapper** — "a zero action indicates that the same delta index should be used as the
previous step" (checked at **p.5**) — cuts this to **13 bins per coordinate**, so the final
vocabulary is the Cartesian product, **13² = 169 discrete motion tokens** (checked at **p.5** and
**p.13**). An 8-second future is **16 tokens per agent**; the 2-agent interactive split flattens to
a sequence of 32 (checked at **p.13**).

**Multi-agent decoding within one time step: parallel.** The factorisation is
`p_θ(A_t | A_<t, S) = Π_{n=1..N} p_θ(a_t^n | A_<t, S)` — "Eq. (2) represents the fact that we treat
agent actions as conditionally independent at time t, given the previous actions and scene context"
(checked at **p.4**, Eq. 2). Each agent's token therefore conditions on **all agents' tokens from
strictly earlier steps plus the scene encoding**, and on **no same-step token**. The training mask
"exhibits a blocked, staircase pattern, exposing all agents to each other's histories only up to the
preceding step" (checked at **p.5**); Fig. 8's caption: "The agents may attend to each other's
previous motion tokens … but no future tokens" (checked at **p.13**). The justification for
tolerating a 0.5 s reaction gap is human latency: "non-impaired human drivers generally require at
least 500 ms to release the accelerator in response to a vehicle braking ahead" (checked at **p.4**).
Ablating the interactive-attention frequency from 2 Hz down to 0.125 Hz (agents see each other only
at `t = 0`) raises the prediction overlap rate by a **relative 38 %** (checked at **p.7**, Table 3
discussion).

**Two remarks worth carrying into the design.** (i) Training is plain teacher forcing and the
authors explicitly did **not** need to add noise to the teacher-forced trajectories (checked at
**p.4**). (ii) Their stated reason for discretising at all: "we suspect that discrete motion tokens
also naturally hide some precision from the model, possibly mitigating compounding error effects
that could arise from imperfect continuous value prediction" (checked at **p.4**).

### 2.2 Trajeglish — Philion, Peng & Fidler, ICLR 2024

**Tokeniser (k-disks).** What is quantised is a **state-to-state transition**: "We define `V = {s_i}`
to be a set of template actions, each of which represents a change in position and heading in the
coordinate frame of the most recent state" (checked at **p.4**, §3.1 *Method*). The tokeniser and
renderer are `f(s₀, s) = argmin_i d_{l,w}(s_i, local(s₀, s))` and `r(s₀, a_i) = global(s₀, s_i)`,
where `d_{l,w}` is the **mean L2 distance between the ordered corners of the two bounding boxes**
(checked at **p.5**, Eqs. 3–4). Tokenisation is applied **iteratively along the trajectory**, i.e.
each step is re-anchored on the already-tokenised state, so tokeniser error is not accumulated
(checked at **p.5**). The template set itself is found by **k-disks**: collect many observed state
transitions, sample one, discard every transition within `ε` metres of it, repeat `|V|` times
(checked at **p.5**, Alg. 1) — data-driven but **not** learned end-to-end; it beats k-means at equal
`|V|` across all three agent classes (checked at **p.5**, Fig. 6).

**Numbers.** "…enables us to tokenize the Waymo Open Dataset (WOMD) at an expected **discretization
error of 1 cm** using a small **vocabulary size of 384**" (checked at **p.1**, contributions;
`|V| = 384` restated in the training config, **p.16**). The time step is WOMD's native **10 Hz**, so
**0.1 s per token** (checked at **p.7**, "seconds at 10hz per scenario"; the context axis of Fig. 10
is labelled in 0.1/0.2/0.3 s, **p.8**).

**Intra-timestep ordering: yes, an agent conditions on same-step tokens.** "…we seek a model from
which we can sample an agent's next state conditional on all states sampled in previous timesteps
**as well as any states already sampled at the current timestep**" (checked at **p.3**). The model
is deliberately **not** permutation-equivariant over agents: "the agent order encodes the order in
which agents select actions within a timestep; the ability of our model to predict actions should
improve when the already-chosen actions of other agents are provided" (checked at **p.5**). A
**random agent order** is drawn during training (checked at **p.16**). Fig. 10 quantifies the value:
NLL falls monotonically with how many agents chose before you in the permutation, and "as the
context length increases, intra-timestep interaction becomes much less important to take into
account" (checked at **p.8**). Their own summary of the size of the effect: "While intra-timestep
interaction between agents is weak in general, explicitly modeling this interaction provides a
window into understanding cases when it is important" (checked at **p.3**).

### 2.3 SMART — Wu et al., NeurIPS 2024

**Tokeniser.** "…we segment the continuous trajectories of all agents in the dataset into trajectory
sets by fixed time intervals **t = 0.5 s**. Then, we cluster the trajectory sets using the **k-disks**
algorithm… the sampled trajectories serve as our final agent motion token vocabulary `V_a`"
(checked at **p.4**, §3.1). At each 0.5 s interval the closest token to the ground truth is selected
by search, anchored on the previously matched token (checked at **p.4**, Fig. 1(a) caption). The map
is tokenised too: polylines cut into segments "each within 5 meters in length" against a road-token
vocabulary (checked at **p.4**, Fig. 1(c) caption).

**Vocabulary size.** Per agent category, and scaled with the model: **512 / 1024 / 1024 / 2048** for
the 1.0M / 7.2M / 26.9M / 101.0M-parameter models; road-token vocabulary fixed at 1024
(checked at **p.14**, Table 5). Scale: "2.2M scenarios (or **1B motion tokens** under 0.5 s agent
motion tokenization)", with `log(L) = −0.157 log(X) + 1.52` (checked at **p.8**).

**Residual / continuous refinement: no.** SMART is pure classification — "SMART is trained to
minimize the cross entropy between the distribution of the ground truth token label and the
predicted distribution" (checked at **p.6**), with "a three-layer MLP" prediction head per decoder
layer (checked at **p.14**). No offset, no regression branch anywhere in the model. The authors list
the tokenizer as future work: "we… maintain a relatively simple design for the discrete token
vocabulary" and the "time granularity of agent motion tokens and the size of the token vocabulary"
are unswept hyper-parameters (checked at **p.10**, *Limitations*).

**What it uses instead of a residual.** A scheduled-sampling-flavoured augmentation: "we perturb the
currently matched token by selecting one from the **top-k tokens closest** to the ground truth token
in the vocabulary. Then, in the next time step, we match the motion token based on the perturbed
vehicle state" (checked at **p.4**). Ablation row M4 (adding noised tokenisation) improves the
interaction metrics (checked at **p.9**). *(reading)* this is the paper's answer to compounding
error in a token rollout, and it is cheaper than a residual head.

### 2.4 VQ-VAE — van den Oord, Vinyals & Kavukcuoglu, NeurIPS 2017

**Commitment loss, as the paper writes it** (checked at **p.4**, Eq. 3):

```
L = log p(x | z_q(x)) + ‖sg[z_e(x)] − e‖₂² + β‖z_e(x) − sg[e]‖₂²
```

with `sg` the stop-gradient operator. Division of labour, quoted: "The decoder optimises the first
loss term only, the encoder optimises the first and the last loss terms, and the embeddings are
optimised by the middle loss term." The **stated reason** for the third term: "since the volume of
the embedding space is dimensionless, it can grow arbitrarily if the embeddings `e_i` do not train as
fast as the encoder parameters. To make sure the encoder commits to an embedding and its output does
not grow, we add a commitment loss" (checked at **p.4**). `β = 0.25` in all experiments and "the
resulting algorithm [is] quite robust to β, as the results did not vary for values of β ranging from
0.1 to 2.0" (checked at **p.4**). An EMA alternative to the middle term is given in Appendix A.1
(checked at **p.11**).

**Codebook collapse: not discussed.** The paper's "collapse" is **posterior** collapse — latents
being ignored when paired with a powerful autoregressive decoder — which VQ is claimed to *avoid*:
"Using the VQ method allows the model to circumvent issues of 'posterior collapse'" (checked at
**p.1**, abstract), listed as a contribution ("does not suffer from 'posterior collapse' and has no
variance issues", checked at **p.2**) and demonstrated on DM-LAB with a PixelCNN decoder
("This setup typically breaks VAEs… Our model however does not suffer from this, and the latents are
meaningfully used", checked at **p.6**). **Dead codes, unused codewords and codebook utilisation
appear nowhere in the paper** (grep over the full text: zero hits for "dead code", "codebook
collapse", "unused", "utilisation/utilization"). The problem is named retrospectively by FSQ — see
§2.5.

### 2.5 FSQ — Mentzer et al., ICLR 2024

**Mechanism.** The encoder projects to `d` channels (typically `d < 10`, against `d ≥ 512` for VQ);
each channel is bounded via `z_i ↦ ⌊L/2⌋ tanh(z_i)` and **rounded to an integer**, giving one of
`|C| = Π_i L_i` hypercube corners; gradients pass by straight-through, exactly as in VQ-VAE
(checked at **p.2**, §1 and Fig. 1). There is **no codebook, no commitment loss, no auxiliary loss
at all** (checked at **p.2**, goals i–iii).

**Levels per dimension.** The hyper-parameters are `d` and `L = [L₁, …, L_d]`. The heuristic, quoted:
"**Use `L_i ≥ 5 ∀i`**" (checked at **p.4**, §3.2). Recommended sets (checked at **p.4**, Table 1):

| target codebook size | 2⁸ | 2¹⁰ | 2¹² | 2¹⁴ | 2¹⁶ |
|---|---|---|---|---|---|
| proposed `L` | `[8, 6, 5]` | `[8, 5, 5, 5]` | `[7, 5, 5, 5, 5]` | `[8, 8, 8, 6, 5]` | `[8, 8, 8, 5, 5, 5]` |

**Claim about utilisation vs VQ.** The problem is stated as VQ's: "the well-documented problem of
underutilized codebooks…: as the size of `C` is increased, many codewords will be unused", with
reinitialisation and random restarts cited as the usual patches (checked at **p.1**, §1; the
"random restarts" of Dhariwal et al. described at **p.3**). The claim: "The codebook usage is very
high for FSQ (**≈100 % for most models**), without relying on any auxiliary losses" (checked at
**p.2**, contribution 2). Measured: on MaskGIT ImageNet-256, **VQ 81 % usage vs FSQ 100 %** at
essentially equal sampling FID (4.509 vs 4.534) (checked at **p.6**, Table); on UViM COCO Panoptic
both reach 100 % (checked at **p.8**). The scaling statement: "FSQ gets better Sampling FID and
higher codebook usage for codebook size exceeding 2¹⁰, **while the metrics start deteriorating for
VQ**" (checked at **p.5**, Fig. 3 caption). Note the VQ baseline there is already helped by
MaskGIT's auxiliary **entropy** loss "that aims to increase the entropy of the codebook (to increase
utilization)" (checked at **p.5**) — so the 81 % is not an unassisted VQ number.

### 2.6 VQ-BeT — Lee et al., ICML 2024

**Action representation: residual VQ code(s) + a continuous offset.** The quantised object is an
action **or an action chunk** `a_{t:t+n}`, `n > 1`, encoded by `φ` and quantised by **Residual VQ**:
`N_q` cascaded codebooks where each layer quantises the previous layer's residual and
`z_q(x) = Σ_{i=1..N_q} z_q^i` (checked at **p.3**, §2.3 and §3.2). Layer 1's code is the **primary
code**, the rest are **secondary codes**: "the primary codes in Residual VQ performs coarse
clustering over a large range within the dataset, while the secondary codes handle fine-grained
actions" (checked at **p.3**). In every experiment `N_q := 2` and `λ_commit := 1` (checked at
**p.3**). The RVQ loss is `L_RVQ = L_Recon + ‖SG[φ(a)] − e‖₂² + λ_commit‖φ(a) − SG[e]‖₂²` with an L1
reconstruction term, and the embeddings are updated by moving averages rather than gradients
(checked at **p.3**, Eqs. 2–3).

**What the offset head does.** A MinGPT predicts the codes with a **focal** loss, weighted between
primary and secondary: `L_code = L_focal(ζ_code^{i=1}(o_t)) + β L_focal(ζ_code^{i>1}(o_t))`
(checked at **p.4**, Eq. 4). The decoded, quantised action is
`⌊a_{t:t+n}⌋ = ψ(Σ_{j,i} e_j^i · 1[ζ_code^i = j])` (checked at **p.4**, Eq. 5). Then, quoted: "We
adopt additional **offset head** `ζ_offset` **to maintain full fidelity, adjusting the centers of
discretized actions based on observations**", trained as
`L_offset = ‖a_{t:t+n} − (⌊a_{t:t+n}⌋ + ζ_offset(o_t))‖₁`, and the total objective is
`L_VQ-BeT = L_code + L_offset` (checked at **p.4**, Eqs. 6–7). So the emitted action is
**code-decoded chunk + a continuous residual regressed from the current observation**; the code
carries the mode, the offset carries the precision. *(reading)* this is the exact shape a "manoeuvre
code + per-segment control refinement" head would take, and it is the only paper in this folder that
has one.

### 2.7 Graphormer — Ying et al., NeurIPS 2021

**The three encodings.**

1. **Centrality encoding** (§3.1.1, checked at **p.3–4**): a learnable vector per node **degree**
   (in-degree and out-degree separately for directed graphs) is **added to the node features in the
   input layer**, so "the softmax attention can catch the node importance signal in the queries and
   the keys".
2. **Spatial encoding** (§3.1.2, checked at **p.4**): `φ(v_i, v_j)` is the **shortest-path distance**
   (a special value **−1** if disconnected); each feasible value indexes a **learnable scalar**,
   **shared across all layers**, added to the attention logit —
   `A_ij = (h_i W_Q)(h_j W_K)ᵀ / √d + b_{φ(v_i,v_j)}` (Eq. 6).
3. **Edge encoding** (§3.1.3, checked at **p.5**): **yes — edge features enter as an additive bias on
   the attention logits**, and the paper says so in exactly those words: "The proposed edge encoding
   incorporates edge features via a **bias term to the attention module**." Concretely (Eq. 7, **p.5**):

   ```
   A_ij = (h_i W_Q)(h_j W_K)ᵀ / √d + b_{φ(v_i,v_j)} + c_ij ,   c_ij = (1/N) Σ_{n=1..N} x_{e_n} (w_n^E)ᵀ
   ```

   where `SP_ij = (e₁, …, e_N)` is (one of) the shortest paths from `v_i` to `v_j`, `x_{e_n}` the
   n-th edge's feature and `w_n^E ∈ R^{d_E}` a learnable per-hop embedding. The paper contrasts this
   with the two prior practices — adding edge features to the incident nodes, or aggregating them
   with node features — which "only propagate the edge information to its associated nodes, which
   may not be an effective way to leverage edge information in representation of the whole graph"
   (checked at **p.5**).

One implementation detail that matters if a "global" token is added: the `[VNode]` is connected to
every node, so its shortest-path distance is 1 everywhere; Graphormer **resets** `b_{φ([VNode],v_j)}`
and `b_{φ(v_i,[VNode])}` to a distinct learnable scalar to distinguish virtual from physical
connections (checked at **p.5**).

### 2.8 Scheduled sampling — Bengio et al., NeurIPS 2015

**Mechanism, one sentence.** At training time, for **every token** of every mini-batch, flip a coin:
feed the **true** previous token `y_{t−1}` with probability `ε_i`, or the model's **own** previous
output `ŷ_{t−1}` (sampled from `P(y_{t−1}|h_{t−1})`, or its argmax) with probability `1 − ε_i`, and
anneal `ε_i` from 1 (pure teacher forcing, "the model is trained exactly as before") to 0 ("trained
in the same setting as inference") over training (checked at **p.3**, §2.4). Three decay schedules
are offered — linear, exponential, and inverse sigmoid `ε_i = k/(k + exp(i/k))` (checked at
**p.4**). The coin must be flipped **per token**: "We also tried to flip the coin once per sequence,
but the results were much worse, most probably because consecutive errors are amplified during the
first rounds of training" (checked at **p.3**, footnote 2). The gradient is **not** back-propagated
through the sampling decision in this paper (checked at **p.4**).

**Compounding-error motivation, quoted.** Abstract: "At inference, the unknown previous token is then
replaced by a token generated by the model itself. **This discrepancy between training and inference
can yield errors that can accumulate quickly along the generated sequence**" (checked at **p.1**).
Sharper, in §1: "The main problem is that **mistakes made early in the sequence generation process
are fed as input to the model and can be quickly amplified** because the model might be in **a part
of the state space it has never seen at training time**" (checked at **p.2**).

**Caution on terminology.** The term **"exposure bias" does not appear in this paper** (grep: 0
hits); that name comes from Ranzato et al. 2015 (MIXER). Cite this paper for the *mechanism* and the
*accumulation* argument, not for the phrase. The formal `T²ε` / `uTε` statements of the same failure
are in `../hierarchical_prediction/` (Ross & Bagnell 2010 Thm 2.1; DAgger Thm 2.2; DaD Thm 1).

### 2.9 Grammar-constrained decoding — Geng et al., EMNLP 2023

**Mechanism, one sentence.** "To enforce the formal grammar, we intervene during decoding by
**pruning the probability distribution to include only the subset of tokens that are allowed by the
formal grammar**. The subset of allowed tokens is returned by an **incremental parser**, which takes
the partially generated sequence and the formal grammar as inputs and returns the set of next
allowed tokens" (checked at **p.4**, §2.2). The parser plays the role of a **completion engine**
(Grammatical Framework's incremental parser is the one used), so "the LM-generated sequences will be
guaranteed to be valid" (checked at **p.2**, **p.4**).

Two properties that matter for a constraint-as-mask design: it is **decoding-algorithm agnostic** —
"compatible with any decoding algorithm, including greedy decoding, beam search, top-k sampling" —
and **model-agnostic**: applicable "to any autoregressive language model, provided that we have
access to the distribution over the vocabulary at each decoding step", with **no finetuning**
(checked at **p.4**). The paper's own extension is **input-dependent grammars**, where the valid set
is a function of the input rather than fixed (checked at **p.1**, abstract) — *(reading)* the direct
analogue of "the admissible manoeuvre codes at this decode step depend on the other aircraft's
current state".

### 2.10 Olive & Morio 2019 — trajectory clustering of air traffic flows around airports

**What is clustered: significant points, not trajectories.** The paper's whole argument is that a
whole-trajectory metric fails on converging TMA flows — Hausdorff "does not apply well on converging
air traffic flows", and resampling to an n-vector with a Euclidean distance is "problematic as an
aircraft being put on holding stacks before landing may see this part of the trajectory trimmed out
after resampling" (checked at **p.1–2**). Instead: "The proposed algorithm computes a clustering on
**subsets of significant points** of trajectories while keeping a **dependency tree** of their
temporal chaining; then associates trajectories to **root-to-leaf paths** in the dependency tree
based on the clusters they cross" (checked at **p.2**).

**Distance.** Plain **Euclidean on 2-D positions**, inside **DBSCAN** (density radius `ε`, minimum
sample count `n`), with a **kernel density estimate** fitted on each cluster's core elements at
bandwidth `h = ε` so membership is a density, not a yes/no (checked at **p.3**, §3 and Alg. 1). The
recursion starts from the last airborne positions, forms clusters, removes the points already inside
a cluster, steps one gate backwards in time, and repeats until a cluster falls below a size threshold
(checked at **p.3**, Alg. 1 and Fig. 4). A trajectory is then scored against each root-to-leaf gate
pattern by a matrix `m_t`; a zero score for every pattern flags it as an outlier (checked at **p.4**,
Eq. 5).

**Data and number of clusters.** ADS-B at Toulouse–Blagnac, **5–30 December 2016**, reduced by
Douglas-Peucker from ~5 GB to under 200 kB for **1,991 trajectories**; "most trajectories now consist
of **5 to 7 points**, but can reach 20" (checked at **p.2**). The top-level split is the two landing
directions, QFU 14 (143°) and QFU 32 (323°) (checked at **p.2**; Fig. 5's tree labels QFU-14 nodes
with Latin and QFU-32 nodes with Greek letters, **p.4**). **The number of clusters is not stated
numerically** — Fig. 7 (**p.5**) plots one representative flight per cluster at a threshold of 5
flights per path. What *is* quantified: "Our clustering of trajectories yields **about 30 % of
outliers**" (checked at **p.5**), and the dense northern QFU-14 flow "splits into several clusters
showing how aircraft are scheduled according to a pattern of **linear hold**: aircraft are scheduled
by timing their turn into final approach so as to land **no less than two minutes apart**" (checked
at **p.5**). Conclusion for a procedure-token design: "these two emerging clusters show how **STAR
procedures published by eAIP are not sufficient to describe traffic flows around airports**"
(checked at **p.5**).

### 2.11 Corrado, Puranik, Pinon & Mavris 2020 — weighted distance for terminal clustering

**Exists and is open access** — `Proceedings 2020, 59, 7; doi:10.3390/proceedings2020059007`,
presented at the 8th OpenSky Symposium, 12–13 Nov 2020, published 1 Dec 2020 (checked at **p.1**).

**What is clustered: whole arrival trajectories.** OpenSky state vectors within **20 NM of KSFO and
below 25,000 ft** for all of 2019, via the `traffic` library, fused with ASOS weather; split by
callsign into segments; arrivals kept, departures discarded; touchdown identified at the first point
below 100 kt groundspeed (checked at **p.4**). Each segment is **resampled onto 50 uniform intervals
of cumulative ground-track distance** from touchdown, projected to UTM, and stacked into a feature
vector — **latitude/longitude only, no altitude** (checked at **p.4–5**). Final dataset **178,890
arrival flight segments**, 187 airlines, 110 aircraft types, split into **237 daily** datasets
(~500 trajectories each) and **30 weekly** datasets (~3,500 each) (checked at **p.5**).

**Distance.** The **weighted Euclidean distance** (WED), a per-point reweighting of the standard ED
(checked at **p.3–4**):

```
D_WED^{i,j} = sqrt( Σ_{k=1..n} w_k [ (x_k^i − x_k^j)² + (y_k^i − y_k^j)² ] )
```

Motivation: under the plain ED, "the distances computed between trajectory points closest to the
airport will be relatively small for all trajectories, regardless of the air traffic flow", which
skews the clustering (checked at **p.3**). Four weighting schemes shaped like normal/beta pdfs and
cdfs are tried (checked at **p.5**, Fig. 1). Clustering is **HDBSCAN**, minimum cluster size = **2 %
of the dataset** (≈10 trajectories/day, ≈70/week), with the smoothing parameter tuned per dataset so
that outliers land between **10 % and 15 %** — chosen to remove outlier bias from the comparison
(checked at **p.5**).

**Number of clusters for approach flows.** For the illustrative single-day dataset (~500 arrivals),
"Both the ED and Weighting 1 clusterings identify **five distinct air traffic flows**" (checked at
**p.6**). No cluster count is given for the weekly datasets. Result: weightings 1 and 3 beat the ED on
most daily and weekly datasets at negligible extra computation, and "if trajectory points **closer to
the border of the terminal airspace, but not necessarily at the border**, are weighted highest, then
a more accurate clustering is computed" (checked at **p.8**, Conclusions); centre-heavy symmetric
weightings (2 and 4) are worse (checked at **p.6–7**).

### 2.12 TimeVQVAE for aircraft trajectories — Murad & Ruocco, ICNS 2025

The closest published aircraft analogue of the proposed architecture. Trajectories are taken to the
time-frequency domain by **STFT**, split into a **low-frequency** and a **high-frequency** branch;
each branch has its **own convolutional encoder and its own VQ codebook**, so the discrete latent
space is two-band; gradients pass the quantiser by **straight-through estimator**; a **bidirectional
MaskGIT transformer prior** is then trained **per band** over the discrete codes, and a UNet
"fidelity enhancer" refines the reconstruction back in the time domain (checked at **p.3**, Fig. 1
and §III; STE at **p.4**). Conditional generation on a class `c` (route / flight profile) is
supported: `τ' ~ G(z, c)` (checked at **p.3**).

Data: **OpenSky Network** as the primary source plus the **EUROCONTROL R&D archive** for metadata,
filtered to **specific airport pairs**, resampled at uniform time intervals, features
lat/lon/alt/time-since-departure (checked at **p.3**); route examples EHAM→LIMC (**p.5**) and
ESSA→LFPG (**p.9**). Evaluation is FID/IS plus a marginal-distribution-difference metric plus a
domain-specific **"flyability" assessment: generated trajectories are replayed in the open-source
BlueSky simulator**, which produces "the closest 'flyable' paths based on its physics model", giving
a spectrum rather than a binary (checked at **p.5–6**). Conclusion: TimeVQVAE beats a temporal
convolutional VAE and "most generated trajectories maintain operational feasibility, although
occasional outliers underscore the potential need for additional domain-specific constraints"
(checked at **p.1**, abstract).

Three limits for our purposes, all *(reading)*: it is **generation, not prediction** (no next-step
forecast benchmark and no ADE/FDE); the codes carry **no manoeuvre semantics** (they are STFT-band
latents, not interpretable manoeuvres); and there is **no cross-agent term and no controller** —
feasibility is checked *post hoc* by a simulator rather than enforced by a rollout.

### 2.13 LLM-FTP — Luo & Zhou 2025 (the negative control)

Recorded because it is the other hit for "flight trajectory next-token", and because it is easy to
mistake for a manoeuvre-token paper. Its "tokens" are the **LLM's own sub-word tokens of the printed
numbers**: `T_{1:t} = {p₁, …, p_t} = {w₁, …, w_n}`, and "the longitude value '103.25' is split into
three distinct tokens: '103', '.', and '25' using the LLaMA-3.1 tokenizer" (checked at **p.3**,
Eq. 5). There is no codebook, no quantisation of a manoeuvre, and no discrete action space — just
prompt-formatted ADS-B waypoints and PEFT fine-tuning of eight open LLMs (checked at **p.3–4**).
Data: ADS-B at **PEK and PVG**; sliding windows with the intra-window time interval "strictly
constrained to 1 minute"; window size **17** for single-step prediction (checked at **p.4**).
LLaMA-3.1-8B is best, credited partly to its number tokenisation ("treats numerical values (e.g.
'123') as a single token"), and **inference latency is flagged as the blocker for real-time use**
(checked at **p.1** abstract and **p.6–7**).

---

## 3. What the folder says about the proposed design

Cross-cluster observations, each traceable to §2. These are pointers for the design doc, not an
assessment of it.

1. **No motion-token paper here learns its codebook end-to-end.** MotionLM uses **uniform bins**
   plus a Verlet wrapper (§2.1); Trajeglish and SMART both use **k-disks**, a greedy
   data-driven template selection, not a trained quantiser (§2.2, §2.3). The learned-codebook line
   — VQ-VAE, FSQ, VQ-BeT — comes from images and robot actions, not from trajectory forecasting.
   *(reading)* a clustered codebook is the attested choice; a learned one is the novelty, and FSQ is
   the cheapest way to get one without the commitment loss and the dead-code patches.
2. **Every attested token step is 0.1–0.5 s.** Trajeglish 0.1 s, MotionLM 0.5 s, SMART 0.5 s. A
   **60 s** segment token is **two to three orders of magnitude longer** than anything measured here,
   and SMART explicitly lists "the time granularity of agent motion tokens" as unswept (§2.3, p.10).
   *(reading)* the 60 s choice is unsupported by this literature in either direction and should be a
   searched hyper-parameter, with the vocabulary size swept alongside it (SMART scales `|V|` with
   model size; Trajeglish shows `|V| = 384` already reaches 1 cm at 0.1 s).
3. **The tokeniser is re-anchored, not open-loop.** Trajeglish tokenises "iteratively along the
   trajectory" in the frame of the *already-tokenised* state (§2.2), MotionLM greedily picks the
   action reconstructing the *next* waypoint (§2.1), SMART matches the next token from the
   *perturbed* state (§2.3). All three make the tokeniser's error non-accumulating by construction.
4. **A residual offset on top of the code exists in exactly one paper**: VQ-BeT's `ζ_offset`, an L1
   regression from the observation added to the decoded chunk (§2.6, Eq. 6). SMART and MotionLM and
   Trajeglish are pure classification. *(reading)* a "code + learned controller" design is closer to
   VQ-BeT than to any of the motion-token papers, and VQ-BeT is also the only one that tokenises a
   **chunk** rather than a single step.
5. **Both intra-step multi-agent schemes are attested, and the choice is measurable.** MotionLM
   decodes agents **in parallel**, conditionally independent within a step (§2.1, Eq. 2), and
   defends it with a 500 ms human-reaction argument. Trajeglish decodes them **sequentially** under
   a random permutation and shows with NLL that later-in-permutation agents predict better, with the
   gain shrinking as history grows (§2.2, Fig. 10). Trajeglish's own summary is that intra-timestep
   interaction is "weak in general".
6. **The two mechanisms the design needs for constraints are separable.** A **hard** constraint is a
   mask over the vocabulary at decode time (Geng, §2.9) and needs no finetuning and no particular
   decoding algorithm. A **soft, pairwise** signal — a separation margin on an edge — is an additive
   bias on the attention logit (Graphormer Eq. 7, §2.7). *(reading)* the design's "edges carry
   separation constraints" is Graphormer's `c_ij`; the design's "constraint = mask" is Geng's
   completion engine; they are different layers and can be built independently.
7. **The terminal-airspace flow literature sizes a route vocabulary, not a manoeuvre vocabulary.**
   Five flows per day at KSFO (§2.11) and two landing directions plus a handful of alignment
   patterns at Toulouse (§2.10). Both papers also warn about the same thing from opposite
   directions: Olive & Morio, that published STARs do **not** describe the observed flows (§2.10);
   Corrado et al., that the *distance* — where along the approach you weight the comparison —
   changes which flows you get at all (§2.11).
8. **The aircraft VQ precedent stops short of the proposal in three specific ways** (§2.12):
   generation not prediction, band latents not manoeuvre semantics, and simulator-checked
   feasibility rather than a differentiable rollout. Those three gaps are where the contribution
   would sit.

---

## 4. Not fetched

**Nothing.** Every source listed in §1 is in `papers/`. Two needed a mirror and one is deliberately
not duplicated:

| source | issue | resolution |
|---|---|---|
| **Olive & Morio 2019** (Aerospace Sci. Tech. 84, 776–781) | Elsevier paywalled; the ONERA postprint at `hal.science/hal-02350789` now sits behind an **Anubis proof-of-work bot wall** — every scripted request gets HTTP **200** with an HTML challenge page (`"Making sure you're not a bot!"`), never the PDF, so the `%PDF-` magic check in `get()` is what catches it (a plain `curl -f` would not) | **Wayback snapshot of the same HAL file** under the old `hal.archives-ouvertes.fr` host (`/web/2020id_/`, raw bytes). Verified against the Elsevier galley header `JID:AESCTE AID:4857` and the author line "Xavier Olive, Jérôme Morio — ONERA". A second Wayback URL under the new host returns exactly 1,048,576 bytes (a truncated 1 MiB capture) — **do not** "fix" `download.sh` to point at it |
| **Corrado et al. 2020** (Proceedings 59(1), 7) | MDPI returns HTTP **403** to scripts for `/2504-3900/59/1/7/pdf` | **Semantic Scholar PDF mirror**, verified against the `Proceedings 2020, 59, 7; doi:10.3390/proceedings2020059007` footer on p.1 and the full author/affiliation block |
| **DAgger** — Ross, Gordon & Bagnell, arXiv 1011.0686 | already downloaded for a sibling folder | **not re-downloaded**; cited in §2.8 by its path in `../hierarchical_prediction/papers/`. Run that folder's `download.sh` if it is missing |

Two verification notes, both negative results that are easy to re-discover the hard way:

* **VQ-VAE does not discuss codebook collapse.** Grep over the full text returns zero hits for
  "dead code", "codebook collapse", "unused", "utilisation"/"utilization" and "index collapse". Its
  only "collapse" is **posterior** collapse (§2.4). The underutilisation problem is stated, with
  citations to Łańcucki et al. 2020, Takida et al. 2022, Dhariwal et al. 2020 and Huh et al. 2023,
  **by FSQ** (§2.5, p.1) — cite FSQ, or those four, not van den Oord et al.
* **Scheduled sampling never says "exposure bias"** (zero hits). It says the training/inference
  discrepancy "can yield errors that can accumulate quickly along the generated sequence" (§2.8).
  The name comes from Ranzato et al. 2015.
