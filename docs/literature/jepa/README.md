# JEPA — reading list, notes, and the assessment for ts_transformer (2026-09-06)

Why this folder exists: the question was whether a Joint-Embedding Predictive Architecture
(JEPA) can be applied to `4dTrajectory/ts_transformer` and whether it would make the
trajectory that follows a predicted set of operating parameters (the `control` output:
N segments of bounded controls + a duration, integrated by the point-mass RK4 rollout) more
accurate. The answer is at the bottom (§4) and was written against the state of the
latent-intent design on 2026-09-06 (`4dTrajectory/ts_transformer/docs/2026-09-07_latent_intent_design.zh.md`,
§〇 status table and L2.c / L2.d results).

The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`); `./download.sh`
re-fetches every one from arXiv. LeCun's 2022 position paper is on OpenReview behind a
browser check and has to be saved by hand.

## 1. Papers in `papers/`

| file | what it is | why it matters here |
|---|---|---|
| `I-JEPA_Assran2023…` (arXiv 2301.08243) | The reference JEPA: context encoder → predictor → predicts the **embedding** of masked target blocks produced by an EMA target encoder; L2 in embedding space; no augmentations. | Defines the family: loss in representation space, stop-gradient + EMA against collapse. |
| `V-JEPA2_Assran2025…` (arXiv 2506.09985) | Video JEPA at 1M hours, then **V-JEPA 2-AC**: a frozen encoder, a 300M block-causal predictor trained with teacher-forcing + 2-step rollout L1 losses in latent space on 62 h of robot video, and planning by CEM on the L1 distance to a goal embedding. | The "latent world model for planning" form of JEPA. Reported limitation: autoregressive latent rollouts drift with horizon. |
| `JointEmbeddingsGoTemporal_Ennadir2025…` (arXiv 2509.25449, TS-JEPA) | Patch tokens, >70 % masking, light transformer encoder/predictor, embedding-space prediction; robust to confounders; SOTA-or-near on UCR-style classification and long-term forecasting. | The literal "JEPA on a time series" recipe: a self-supervised pretraining of a PatchTST-shaped backbone. |
| `Phys-JEPA_2026…` (arXiv 2606.16076) | Latent split into physical + residual parts; a projector maps the physical part to measured physical variables; static and transition consistency losses; SIGReg on the residual part; decoder to observations for forecasting. | The closest "physics in a JEPA latent" paper. Gains are 0.6–5.5 % MSE, **single seed**, no PatchTST / iTransformer comparison, and the authors call it initial validation. |
| `SkyJEPA_2026…` (arXiv 2606.23444) | TCN encoders + GRU latent predictor conditioned on actions, trained on 1.5M simulated quadrotor transitions; a **physics-inspired prober** decodes the latent into residual corrections of the known kinematics; MPPI control on the decoded state. | The paper where the known dynamics meet a learned latent: the win (4.5× position RMSE) comes from the prober built on the **known** kinematics, not from the latent. |
| `Auto-JEPA_2026…` (arXiv 2607.29031) | "Continuous intent": a **frozen trajectory autoencoder** encodes the future 8-waypoint ego trajectory into 8 × 1024 tokens; a predictor from V-JEPA 2 visual tokens + history + route regresses those tokens (smooth-L1 + cosine + InfoNCE); the trajectory is then **retrieved** from a 110k memory by latent similarity, scored and gated. NAVSIM v2 89.1 EPDMS. | The nearest analogue to our "latent intent → operating parameters" design. Note the two things it does NOT do: no KL / no sampling (so no posterior collapse), and no parametric decoder (retrieval instead). |
| `PLDM_Sobal2025…` (arXiv 2502.14819) | JEPA-style latent dynamics learned from reward-free offline trajectories; planning in latent; more data-efficient and generalises to unseen layouts better than model-free RL; trajectory stitching. | The evidence that latent planning helps when the dynamics are **unknown** and must be learned from data. |
| LeCun 2022, *A Path Towards Autonomous Machine Intelligence* (OpenReview, not fetched) | The position paper: predict in representation space, a latent z for what the context cannot know about the future with its information content **regularised**, hierarchical JEPA, planning as cost minimisation over latent futures. | Source of the term. Its "z must be information-limited" is the same fight our CVAE's KL / free-bits is having. |

Secondary sources read online, not saved: T-JEPA (trajectory similarity, SIGSPATIAL 2024),
MTS-JEPA (anomaly prediction, 2026), the awesome-jepa list, and the Turing Post JEPA
explainer (used for the LeCun summary).

## 2. What JEPA is, reduced to the parts that matter for us

1. **The loss is in embedding space.** A target encoder (EMA / stop-gradient, or frozen)
   turns the future into `s_y`; a predictor turns the context into `ŝ_y`; the loss is
   `‖ŝ_y − sg(s_y)‖`. The encoder is free to discard whatever the predictor cannot predict.
2. **Collapse is handled by asymmetry or a variance regulariser**, not by a KL: EMA target
   (I-JEPA, V-JEPA), VICReg / SIGReg (Phys-JEPA), or InfoNCE (Auto-JEPA).
3. **The predictor is deterministic.** The multimodal future is handled either by the
   representation itself throwing the unpredictable part away, or, in the position paper,
   by an extra latent z that must be information-limited. None of the applied papers
   sample z at inference; Auto-JEPA gets its candidates from retrieval.
4. **As a world model**, JEPA learns `z_{t+1} = P(z_t, a_t)` and plans in that space. The
   value is measured where the dynamics are unknown (robot video, offline navigation,
   sim-randomised quadrotors).

## 3. Where our design already stands (so the comparison is against the real thing)

- The `control` path IS a "latent + transformer + physics decoder": a CVAE with an 8-dim z
  (`control/latent.py`), `q(z | future)` on the normalised target rows, a K-component prior
  from the fused context, a decoder = bounded control head + duration head, and the
  differentiable RK4 point-mass rollout turning the 96 numbers into the trajectory. The
  reconstruction loss is in trajectory space with dense supervision plus the
  inverse-dynamics imitation teacher on the controls.
- Results to date (KRDU, native32 base = 1322 m pooled ADE): cold-start and β ≥ 0.1 arms
  collapse (KL under the free-bits floor, β inert there); warm posterior β = 0.01 keeps z
  alive at 0.17 nat and is the best point estimate so far, 1214 m pooled (vectored
  2870 → 2643), gate (3) and the veto pass, gates (1) and (2) fail; the z-oracle reaches
  vectored 2416 m against the 1235 m gate, i.e. the posterior carries about one fifth of
  the intent. L2.e′ (free bits as an information budget, 0.5 / 1.0 nat/dim) is queued.
- The measured limit is information, not capacity: the vectoring decision (join distance
  and timing, ~962 m of ADE) is not in the ego history, and the observable traffic scene
  does not predict it either (L4 gate failed: d_join R² 0.37 vs 0.38; lead ETA correlates
  0.11 with the lead's true landing). Backbone capacity, segment count and conditioning
  capacity were each tested and are not the bottleneck.
- There is no self-supervised pretraining anywhere in the package; every backbone is
  trained end-to-end on the supervised objective, on one airport (8,255 train flights).

## 4. Assessment: can JEPA be applied, and would it make the post-dynamics trajectory more accurate?

**Short answer.** JEPA can be applied in two narrow places, and neither is expected to
raise top-1 accuracy of the rolled-out trajectory beyond seed noise. What it can do is fix
the mechanism that is failing in L2 (the posterior's information), and give a distribution
readout without a KL. The gain it cannot give is information the input does not contain.

### 4.1 Why it will not move top-1 ADE

- Top-1 ADE is an L2 metric in trajectory space. A regressor trained with the trajectory
  loss we already use is the conditional-mean estimator for exactly that metric, given the
  same inputs. Moving the loss into an embedding space changes what the point estimate
  averages over, not how much it knows. The residual 962 m is the intent the input does not
  contain, and no target encoder can recover it.
- JEPA's encoder discards what the predictor cannot predict. Here the unpredictable part
  IS the signal (join distance, timing), not a nuisance. An embedding that drops it and is
  then decoded through the control head reproduces the "bland shared profile" trap the
  notes already document (71 % of bank energy collapsing into one profile).
- The published forecasting gains are small and fragile: Phys-JEPA 0.6–5.5 % MSE, single
  seed, no PatchTST / iTransformer baseline; TS-JEPA near-SOTA on benchmarks with large,
  clean context. Our seed floor is 5–22 m pooled ADE and a margin under 1.5× is
  provisional by package rule, so a few-percent effect is unreadable here.
- The one place a paper puts a known physics model next to a JEPA latent (SkyJEPA), the
  gain is attributed to the prober built on the known kinematics. We already have the
  strong form of that: the exact rollout is the decoder. A learned latent dynamics
  (V-JEPA 2-AC / PLDM style) would replace a known, flyable-by-construction model with an
  approximation that drifts with horizon, which the V-JEPA 2 authors list as a limitation.
  This use is rejected.

### 4.2 The two places it fits

**(a) Replace the CVAE's z with a JEPA-style regressed intent embedding.** This is the
Auto-JEPA recipe transposed: a target encoder on the future (EMA of a small encoder, or a
frozen encoder pretrained to reconstruct the future through the control head + rollout),
a predictor from the fused context that regresses that embedding with smooth-L1 + cosine,
a variance regulariser (VICReg / SIGReg) on the target embedding instead of a KL, and the
decoder trained teacher-forced on the target embedding. What this changes mechanically:

- Posterior collapse cannot happen the way L2.c / L2.d showed it. The target embedding is
  kept non-degenerate by the variance term at every step, so the decoder always has a
  z with signal in it; the "0.17 nat, prior narrower than N(0, I)" failure is structurally
  excluded, and the tuning axis (β, free bits, init std) disappears.
- The z-oracle gate (3) becomes the target-embedding teacher-forced arm, and it is
  readable from epoch 0.
- Multimodality has to be re-added: the predictor is deterministic. Either a mixture
  head on the embedding (back to something CVAE-shaped) or Auto-JEPA's retrieval, for
  which we have a ready memory: the L5.a fitted teacher table of per-flight control
  schedules. Nearest-K schedules by embedding distance → K rollouts → minADE_K, with no
  sampling and no collapse. Bounded by memory coverage (same airport, anchor and N).

Expected outcome on the design's gates: (1) passes by construction; (2) is genuinely
open, retrieval is the more likely way to pass it; (3) is where the information limit
shows; top-1 stays at the 1214 m level within seed noise. Pre-registration rule 5 of the
design already states that L2 does not promise top-1 improvement.

**(b) TS-JEPA pretraining of the backbone.** Mask-and-predict-in-latent pretraining of
the iTransformer / PatchTST encoder on every track the harvest holds (42,650 arrivals over
five airports, and the non-arrival tracks the harvest keeps but the models never see), then
fine-tune the control head on one airport. The plausible benefits are data efficiency
and cross-airport transfer, not accuracy on KRDU where the capacity ablations already
say the encoder is not the bottleneck. Worth an arm only if pooled or multi-airport
training becomes the deliverable; on the KRDU single-airport question it is expected to
land inside the seed floor.

### 4.3 What would have to be true for JEPA to help accuracy

Only one thing: an input that carries the vectoring intent. Phase 0 measured that the
truth join point + truth remaining time takes vectored ADE from 2858 to 2011 m. If a
scene / traffic-sequence input exists that predicts d_join better than R² 0.38, a JEPA
objective over the traffic sequence (predict the embedding of the next landings from the
current scene) is a reasonable way to learn its representation. Until such an input is
found, every architecture, JEPA included, is bounded at the same ceiling.

### 4.4 Recommendation

1. Let L2.e′ (free bits as budget) finish; it is the cheapest test of whether the CVAE can
   carry the intent at all.
2. If L2.e′ still leaves z under ~1 nat, build (a) as **L2.f**: JEPA-regressed intent
   embedding with VICReg on the target, retrieval over the L5.a fitted table for the
   K-mode readout. Same base, same gates, same veto; the target-embedding teacher-forced
   arm replaces `--z-from-posterior`.
3. Do not build a learned latent dynamics; the physics rollout is the point of the path.
4. Keep (b) as an option for the pooled-airports question, not for the KRDU result.
