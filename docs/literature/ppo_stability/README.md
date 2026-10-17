# PPO update stability — sources and verified claims (2026-10-17)

Scope: the stability of a PPO-style pass: the ratio clip, the dual clip, gradient-norm clipping and
group-relative advantages. Read for stage D's D219, 2026-10-17. Each entry gives the source, where to get it,
and the claims checked against its primary text, with a verbatim quote and the exact place.

Files. The PDFs in `papers/` are not tracked in git (root `.gitignore` rule `*.pdf`). `./download.sh`
fetches them again from arXiv and AAAI; the arXiv versions are pinned to the version that was read. Web pages
and source files (entries 6, 8 and 9) are not fetched; their entries give the URL, the tag and commit, the line and the
quote, as read on 2026-10-17. The PDF text layer loses ε, ≫ and ≪, so those symbols in the quotes were copied
from the rendered page.

Verdicts. VERIFIED: the source says it. WRONG: the source says something else (the correct value is given).
NOT FOUND: the source does not say it. IMPLIED: it follows from the source's equation or figure, but the text
does not say it.

## 1. Schulman et al. 2017 — Proximal Policy Optimization

- Citation: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. *Proximal Policy
  Optimization Algorithms*. arXiv:1707.06347v2 [cs.LG], 28 Aug 2017.
- URL: https://arxiv.org/abs/1707.06347
- File: `papers/Schulman_2017_Proximal_Policy_Optimization_arXiv1707.06347v2.pdf` (12 p.)

Claims:

- **The clipped surrogate is eq. (7). VERIFIED.** §3 "Clipped Surrogate Objective", p. 3:
  "L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1 − ε, 1 + ε)Â_t)]   (7) where epsilon is a hyperparameter, say,
  ε = 0.2." The ratio is defined on the same page: "r_t(θ) = π_θ(a_t | s_t) / π_θold(a_t | s_t), so r(θ_old) = 1".
- **ε = 0.2 scored best in their comparison. VERIFIED.** §6.1 "Comparison of Surrogate Objectives", Table 1, p. 6:
  "Clipping, ε = 0.1  0.76 / Clipping, ε = 0.2  0.82 / Clipping, ε = 0.3  0.70". 0.82 is the highest of the 11
  settings in the table (no clipping or penalty −0.39; adaptive KL 0.68–0.74; fixed KL 0.62–0.72). The score is
  "Average normalized scores (over 21 runs of the algorithm, on 7 environments)": 7 MuJoCo tasks, 3 seeds each,
  one million timesteps. §6.2, p. 6: "For PPO, we used the hyperparameters from the previous section, with ε = 0.2."
- **The same batch is used for several epochs of minibatch updates. VERIFIED.** Abstract, p. 1: "we propose a novel
  objective function that enables multiple epochs of minibatch updates." §5 "Algorithm", p. 5: "we construct the
  surrogate loss on these NT timesteps of data, and optimize it with minibatch SGD (or usually for better
  performance, Adam [KB14]), for K epochs." Algorithm 1, p. 5: "Optimize surrogate L wrt θ, with K epochs and
  minibatch size M ≤ NT". The values used are on p. 10: Table 3 (MuJoCo) 10 epochs, minibatch 64; Table 4
  (Roboschool) 15 epochs, minibatch 4096; Table 5 (Atari) 3 epochs, minibatch 32 × 8.
- **With a negative advantage the clip bounds the term only for r < 1 − ε; for r > 1 + ε the term r·A is unbounded.
  IMPLIED, not said in words.** The paper never says "unbounded". §3, p. 3, says: "With this scheme, we only ignore
  the change in probability ratio when it would make the objective improve, and we include it when it makes the
  objective worse." and "note that the probability ratio r is clipped at 1 − ε or 1 + ε depending on whether the
  advantage is positive or negative." Figure 1, right panel ("A < 0"), p. 3, draws the term flat for r < 1 − ε and
  as a straight line that falls with no lower limit for all r > 1 − ε. Caption: "Plots showing one term (i.e., a
  single timestep) of the surrogate function L^CLIP as a function of the probability ratio r, for positive
  advantages (left) and negative advantages (right)." The words "unbounded variance" for this case are in
  Ye et al. 2020 (entry 2).

## 2. Ye et al. 2020 — dual-clip PPO

- Citation: Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng
  Wu, Qingwei Guo, Qiaobo Chen, Yinyuting Yin, Hao Zhang, Tengfei Shi, Liang Wang, Qiang Fu, Wei Yang, Lanxiao
  Huang. *Mastering Complex Control in MOBA Games with Deep Reinforcement Learning*. AAAI-20, Proceedings of the
  AAAI Conference on Artificial Intelligence 34(04), pp. 6672–6679, 2020. arXiv:1912.09729v3 [cs.AI], 15 Dec 2020.
- URL: https://arxiv.org/abs/1912.09729 and https://ojs.aaai.org/index.php/AAAI/article/view/6144
- Files: `papers/Ye_2020_Mastering_Complex_Control_in_MOBA_Games_arXiv1912.09729v3.pdf` (8 p.) and
  `papers/Ye_2020_Mastering_Complex_Control_in_MOBA_Games_AAAI34.pdf` (8 p.). The quoted passages below are word
  for word the same in both; the pages are the same (AAAI page numbers in brackets).

Claims:

- **The dual-clip objective and its number. VERIFIED: eq. (5).** Section "Algorithm Design", paragraph
  "Dual-clip PPO", p. 5 [6676]: "When Â_t < 0, the new objective of our dual-clipped PPO is:
  Ê_t[max(min(r_t(θ)Â_t, clip(r_t(θ), 1 − ε, 1 + ε)Â_t), cÂ_t)]   (5) where c > 1 is a constant indicating the
  lower bound." The standard PPO objective is restated there as eq. (4), p. 4 [6675]. Figure 3, p. 5 [6676]:
  "(a) Standard PPO (clip with ε); (b) Our Proposed Dual-clip PPO ( clip with ε and c when Â_t < 0)".
- **The motivation: a large ratio with a negative advantage gives a large, unbounded variance and a large policy
  deviation. VERIFIED.** Same paragraph, p. 4 [6675]: "For example, when π_θ(a_t^(i)|s_t) ≫ π_θold(a_t^(i)|s_t),
  the ratio r_t(θ) is a huge number. When Â_t < 0, such a large ratio r_t(θ) will introduce a big and unbounded
  variance since r_t(θ)Â_t ≪ 0. As a result, even using the objective of PPO, the new policy deviates
  significantly from the old policy, which makes it very difficult to insure the policy convergence." The setting
  they name, same paragraph: "in large-scale off-policy training environments like our framework, the trajectories
  are sampled from various sources of policies, which may differ considerably from the current policy π_θ."
- **The value of c is 3. VERIFIED.** Section "Experiments", "System Setup", p. 5 [6676]: "In the dual-clipped PPO,
  the two clipping hyperparameters ε and c are set as 0.2 and 3, respectively."

## 3. Pascanu, Mikolov, Bengio 2013 — gradient-norm clipping

- Citation: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio. *On the difficulty of training Recurrent Neural
  Networks*. ICML 2013 (PMLR 28). arXiv:1211.5063v2 [cs.LG], 16 Feb 2013.
- URL: https://arxiv.org/abs/1211.5063
- File: `papers/Pascanu_2013_Difficulty_of_Training_RNNs_arXiv1211.5063v2.pdf` (12 p.)

Claims:

- **The norm-clipping algorithm is Algorithm 1. VERIFIED.** §3.2 "Scaling down the gradients", p. 6: "one simple
  mechanism to deal with a sudden increase in the norm of the gradients is to rescale them whenever they go over a
  threshold (see algorithm 1)." Caption: "Algorithm 1 Pseudo-code for norm clipping the gradients whenever they
  explode".
- **The rescaling g ← threshold · g / ‖g‖ when ‖g‖ exceeds the threshold. VERIFIED, with one detail: the test is
  ≥, not >.** Algorithm 1, p. 6: "ĝ ← ∂E/∂θ; if ‖ĝ‖ ≥ threshold then ĝ ← (threshold / ‖ĝ‖) ĝ end if".
- **What they say about choosing the threshold. VERIFIED.** §3.2, p. 6: "it does however introduce an additional
  hyper-parameter, namely the threshold. One good heuristic for setting this threshold is to look at statistics on
  the average norm over a sufficiently large number of updates. In our experiments we have noticed that for a given
  task and model size, training is not very sensitive to this hyperparameter and the algorithm behaves well even
  for rather small thresholds." The thresholds they used (appendix, pp. 11–12): "a cutoff threshold of 6 on the
  norm of the gradients" (synthetic tasks) and "a cut-off threshold of 45 (though we take the sum of the cost over
  the sequence length)" (language modelling).

## 4. Engstrom et al. 2020 — Implementation Matters (PPO and TRPO)

- Citation: Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph,
  Aleksander Madry. *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*. ICLR 2020.
  arXiv:2005.12729v1 [cs.LG], 25 May 2020.
- URL: https://arxiv.org/abs/2005.12729 ; authors' code: https://github.com/MadryLab/implementation-matters
- File: `papers/Engstrom_2020_Implementation_Matters_PPO_TRPO_arXiv2005.12729v1.pdf` (14 p.)

Claims:

- **Global gradient clipping is one of the listed code-level optimizations. VERIFIED: item 9 of 9.** §3
  "Attributing Success in Proximal Policy Optimization", p. 3: "9. Global Gradient Clipping: After computing the
  gradient with respect to the policy and the value networks, the implementation clips the gradients such the
  “global ℓ2 norm” (i.e. the norm of the concatenated gradients of all parameters) does not exceed 0.5."
- **What they found about its effect. NOT FOUND: they did not measure it.** The ablation covers the first four
  items only. §3, p. 3: "we perform a full ablation study on the four optimizations mentioned above³", footnote 3:
  "Due to restrictions on computational resources, we could only perform a full ablation on the first four of the
  identified optimizations." Figure 1 caption, p. 4: "An ablation study on the first four optimizations described
  in Section 3 (value clipping, reward scaling, network initialization, and learning rate annealing)."
- **Related fact: their own PPO runs used no gradient clipping.** App. A.1, Tables 4–6 (Walker2d, Humanoid, Hopper),
  pp. 10–11, row "Gradient clipping (ℓ2 norm)": PPO −1, TRPO −1, PPO-M −1; PPO-NoClip 0.1 / 0.5 / 4; TRPO+ 1 / 0.5 / 1.
  The paper does not define −1. In the authors' code, −1 skips the clipping:
  `MadryLab/implementation-matters` commit 5ee6ecb1, `src/policy_gradients/steps.py` lines 296 and 305:
  "if params.CLIP_GRAD_NORM != -1:".

## 5. Andrychowicz et al. 2021 — What Matters in On-Policy RL

- Citation: Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphaël Marinier,
  Léonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, Sylvain Gelly, Olivier Bachem. *What
  Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study*. ICLR 2021. arXiv:2006.05990v1
  [cs.LG], 10 Jun 2020. Section and page numbers below are those of arXiv v1.
- URL: https://arxiv.org/abs/2006.05990
- File: `papers/Andrychowicz_2021_What_Matters_On-Policy_RL_arXiv2006.05990v1.pdf` (48 p.)

Claims:

- **The finding and the recommendation on gradient clipping. VERIFIED.** §3.3 "Normalization and clipping (based on
  the results in Appendix F)", p. 6: "Finally, gradient clipping (C68) provides a small performance boost with the
  exact clipping threshold making little difference (Fig. 34)." and "Recommendation. [...] Gradient clipping
  might slightly help but is of secondary importance." They recommend no particular threshold.
- **The values tested. VERIFIED.** App. F.1, p. 28: "Gradient clipping (C68): {0.5, 1.0, 2.0, 5.0, None}", sampled
  uniformly with the other choices of that experiment, 2000 configurations per environment, 5 MuJoCo environments,
  3 seeds each. Figure 34, p. 29: "Analysis of choice Gradient clipping (C68): 95th percentile of performance
  scores conditioned on choice (left) and distribution of choices in top 5% of configurations (right)."
- **The default threshold. 0.5.** App. C, Table 2 "Default settings used in experiments", p. 17: "C68 Gradient
  clipping 0.5". The definition, App. B.9, p. 16: "We rescale the gradient before feeding it to the optimizer so that
  its L2 norm does not exceed the desired threshold."
- **Related: the PPO ratio clip ε.** §3.1, p. 4: "Recommendation. Use the PPO policy loss. Start with the clipping
  threshold set to 0.25 but also try lower and higher values if possible."

## 6. Huang et al. 2022 — The 37 Implementation Details of PPO (web page)

- Citation: Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, Weixun Wang. *The 37
  Implementation Details of Proximal Policy Optimization*. ICLR Blog Track, 25 Mar 2022.
- URL: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ (read 2026-10-17; no PDF)
- Reference implementation it describes: `openai/baselines` commit ea25b9e8b234e6ee1bca43083f8f3cf974143998.

Claims:

- **Global gradient clipping with max norm 0.5 in the reference implementation. VERIFIED: core detail 11.** Section
  "13 core implementation details", item 11, "Global Gradient Clipping (ppo2/model.py#L102-L108)": "For each update
  iteration in an epoch, PPO rescales the gradients of the policy and value network so that the “global l2 norm”
  (i.e., the norm of the concatenated gradients of all parameters) does not exceed 0.5." Next line: "Andrychowicz,
  et al. (2021) find global gradient clipping to offer a small performance boost (decision C68, figure 34)."
- **The value in the reference code. VERIFIED.** `openai/baselines` at ea25b9e, `baselines/ppo2/ppo2.py` line 22
  (default argument of `learn`): "vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,"; `baselines/ppo2/model.py`
  lines 105–107: "if max_grad_norm is not None: # Clip the gradients (normalize) grads, _grad_norm =
  tf.clip_by_global_norm(grads, max_grad_norm)".

## 7. Shao et al. 2024 — DeepSeekMath (GRPO)

- Citation: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang,
  Y.K. Li, Y. Wu, Daya Guo. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*.
  arXiv:2402.03300v3 [cs.CL], 27 Apr 2024.
- URL: https://arxiv.org/abs/2402.03300
- File: `papers/Shao_2024_DeepSeekMath_GRPO_arXiv2402.03300v3.pdf` (30 p.)

Claims:

- **The objective is PPO-clipped with a KL term to the reference model. VERIFIED: eq. (3); the KL estimator is
  eq. (4).** §4.1.1 "From PPO to GRPO", p. 13: "J_GRPO(θ) = E[q ∼ P(Q), {o_i}_{i=1}^G ∼ π_θold(O|q)]
  (1/G) Σ_i (1/|o_i|) Σ_t { min[ (π_θ(o_i,t|q, o_i,<t) / π_θold(o_i,t|q, o_i,<t)) Â_i,t, clip(…, 1 − ε, 1 + ε) Â_i,t ]
  − β D_KL[π_θ || π_ref] }   (3)". Same page: "instead of adding KL penalty in the reward, GRPO regularizes by
  directly adding the KL divergence between the trained policy and the reference policy to the loss". p. 14:
  "D_KL[π_θ||π_ref] = π_ref(o_i,t|q,o_i,<t)/π_θ(o_i,t|q,o_i,<t) − log π_ref(o_i,t|q,o_i,<t)/π_θ(o_i,t|q,o_i,<t) − 1,   (4)
  which is guaranteed to be positive." Algorithm 1 (p. 14) cites "the GRPO objective (Equation 21)", which is the
  gradient form in App. A.1.6, pp. 29–30.
- **The advantage is the group-normalised reward (r − mean)/std. VERIFIED; the formula has no equation number.**
  §4.1.2 "Outcome Supervision RL with GRPO", p. 14: "these rewards are normalized by subtracting the group average
  and dividing by the group standard deviation. Outcome supervision provides the normalized reward at the end of
  each output o_i and sets the advantages Â_i,t of all tokens in the output as the normalized reward, i.e.,
  Â_i,t = r̃_i = (r_i − mean(r)) / std(r)". With process supervision (§4.1.3, p. 14) the per-step rewards are
  normalised the same way and the advantage of a token is "the sum of the normalized rewards from the following
  steps".
- **Their settings (related).** §4.2, p. 15: "The KL coefficient is 0.04. For each question, we sample 64 outputs.
  The max length is set to 1024, and the training batch size is 1024. The policy model only has a single update
  following each exploration stage."

## 8. verl — actor configuration (source and docs)

- Source: `verl-project/verl` (https://github.com/volcengine/verl redirects there), tag v0.9.1, commit
  1876b06d0a3e4e71e06230be10af14492ca8a75b (2026-09-20), read 2026-10-17. Docs:
  https://verl.readthedocs.io/en/latest/algo/ppo.html and
  https://verl.readthedocs.io/en/latest/examples/config.html (the same text as `docs/` at the tag).

Claims:

- **`clip_ratio_c` (dual clip) defaults to 3.0. VERIFIED.** `verl/trainer/config/actor/actor.yaml` lines 82–83:
  "# Constant C in Dual-clip PPO; clips when advantage < 0 and ratio > C" / "clip_ratio_c: 3.0".
  `verl/workers/config/actor.py` line 163: "clip_ratio_c: float = 3.0" (the class docstring, line 122, calls it
  "Clipping ratio for critic loss"; the yaml, the docs and the loss code all use it as the dual-clip constant).
  `docs/algo/ppo.md` lines 75–81, section "Dual-clip PPO": "`actor_rollout_ref.actor.clip_ratio_c`: lower bound of
  the value for Dual-clip PPO, defaults to 3.0". The loss, `verl/trainer/ppo/core_algos.py` lines 1323–1324:
  "clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729."
  … "\"clip_ratio_c\", 3.0"; line 1331 asserts "clip_ratio_c > 1.0"; lines 1273–1279 apply it only where
  "advantages < 0". The ratio clip ε defaults to 0.2 (`actor.yaml` line 36, "clip_ratio: 0.2").
- **The actor's `grad_clip` defaults to 1.0. VERIFIED.** `verl/trainer/config/actor/dp_actor.yaml` lines 28–29:
  "# Gradient clipping for actor updates, specific to the strategy." / "grad_clip: 1.0".
  `verl/workers/config/actor.py` line 308 (`FSDPActorConfig`): "grad_clip: float = 1.0". `docs/examples/config.rst`
  line 127 "grad_clip: 1.0" and lines 292–293: "`actor_rollout_ref.actor.grad_clip`: Gradient clipping for actor
  updates". At this tag the FSDP engine clips with the optimizer field `clip_grad`, also 1.0:
  `verl/workers/config/optimizer.py` line 53 "clip_grad: float = 1.0" (line 60: "`grad_clip` is deprecated, use
  `clip_grad` instead." for the optimizer-level field), `verl/trainer/config/optim/fsdp.yaml` line 30
  "clip_grad: 1.0", applied in `verl/workers/engine/fsdp/transformer_impl.py` lines 806–811.

## 9. TRL `GRPOConfig` and transformers `TrainingArguments` — `max_grad_norm`

- Source: `huggingface/transformers` tag v5.20.1, commit 0ea877911a11499053df677befd4b35b172ff502 (2026-10-16);
  `huggingface/trl` tag v1.16.0, commit 1b5c75ed94b7387f42532adb090c935166d4878d (2026-10-15); read 2026-10-17.
- URLs: https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/training_args.py and
  https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_config.py

Claims:

- **transformers `TrainingArguments.max_grad_norm` defaults to 1.0. VERIFIED.** `src/transformers/training_args.py`
  lines 860–862: "max_grad_norm: float = field( default=1.0, metadata={\"help\": \"Maximum gradient norm for gradient
  clipping. Set to 0 to disable.\"} )". Docstring, lines 270–274: "Maximum gradient norm for gradient clipping.
  Applied after backward pass, before optimizer step. Prevents gradient explosion by scaling down gradients when
  their global norm exceeds this threshold. Set to 0 to disable clipping. Typical values: 1.0 (standard), 0.5 (more
  conservative), 5.0 (less aggressive)."
- **TRL `GRPOConfig` keeps that default, 1.0. VERIFIED.** `GRPOConfig` subclasses `_BaseConfig`
  (`trl/trainer/grpo_config.py` line 23), which subclasses `transformers.TrainingArguments`
  (`trl/trainer/base_config.py` line 21). Neither file sets `max_grad_norm`. The class note lists the defaults it
  changes, and `max_grad_norm` is not among them (`grpo_config.py` lines 408–412): "These parameters have default
  values different from [`~transformers.TrainingArguments`]: `logging_steps` [...] `gradient_checkpointing` [...]
  `bf16` [...] `learning_rate`: Defaults to `1e-6` instead of `5e-5`."
- **Related GRPOConfig defaults at this tag.** `epsilon` 0.2 (docstring line 178: "Epsilon value for clipping.");
  `num_iterations` 1 (line 690, "Number of iterations per batch (denoted as μ in the algorithm)."); `beta` 0.0
  (lines 681–684, "KL coefficient. If `0.0` (default), the reference model is not loaded"); `delta` None (lines 180–183:
  "Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` (default), standard GRPO
  clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in the
  INTELLECT-2 tech report").
