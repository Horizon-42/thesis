# Credit assignment without a learned value network — sources and verified claims (2026-10-17)

Scope: credit assignment to words or segments without a learned value network: Monte Carlo values from extra
rollouts, counterfactual one-action changes, expected (all-action) policy gradients, potential-based shaping.
Read for stage D, 2026-10-17. Each entry gives the source, where to get it, and the claims checked against its
primary text, with a verbatim quote and the exact place.

Files. The PDFs in `papers/` are not tracked in git (root `.gitignore` rule `*.pdf`). `./download.sh` fetches them
again from arXiv, AAAI and Stuart Russell's Berkeley page; the arXiv versions are pinned to the version that was
read. The PDF text layer loses γ, Φ, Σ and the minus sign in Ng et al. 1999 and part of the notation in the other
papers, so those symbols in the quotes were copied from the rendered page. Page numbers are those of the PDF named
in the entry; printed proceedings page numbers are given in brackets where the PDF carries them.

Verdicts. VERIFIED: the source says it. WRONG: the source says something else (the correct text is given).
NOT FOUND: the source does not say it. IMPLIED: it follows from the source's equations, but the text does not say it.

## 1. Kazemnejad et al. 2025 — VinePPO

- Citation: Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron
  Courville, Nicolas Le Roux. *VinePPO: Refining Credit Assignment in RL Training of LLMs*. ICML 2025, Proceedings
  of the 42nd International Conference on Machine Learning, PMLR 267:29557–29590. arXiv:2410.01679v2 [cs.LG],
  3 Jun 2025. The title changed between versions: arXiv v1 (2 Oct 2024) is *VinePPO: Unlocking RL Potential For
  LLM Reasoning Through Refined Credit Assignment*. Read: v2.
- URL: https://arxiv.org/abs/2410.01679 and https://proceedings.mlr.press/v267/kazemnejad25a.html
- File: `papers/Kazemnejad_2025_VinePPO_arXiv2410.01679v2.pdf` (34 p.)

Claims:

- **The value of an intermediate state is the mean return of K rollouts sampled from it. VERIFIED: eq. (5).** §4
  "Accurate Credit Assignment with VinePPO", p. 5: "for each state st in a training trajectory τ, we utilize the
  resetting property and re-feed the partial context corresponding to st to the current policy to sample K
  auxiliary rollouts η1, . . . , ηK ∼ πθ(· | st). The empirical mean of returns across these rollouts serves as
  the value estimate: V̂MC(st) = (1/K) Σ_{k=1}^{K} R(ηk).   (5)". The rollouts train nothing, same page:
  "Critically, ηk's are used exclusively for value estimation and do not contribute directly to policy gradient
  updates as we lack CA on them."
- **The advantage of a step is formed from two such values. VERIFIED: eq. (6).** §4, p. 5: "Once the value
  V̂MC(st) is computed, we estimate the advantages of each action using Equation 3: ÂMC(st, at) = r(st, at) +
  γV̂MC(st+1) − V̂MC(st).   (6) For any K ≥ 1, the policy gradient computed using the advantage estimator ÂMC is an
  unbiased estimate of the gradient of expected return gpg." Eq. (3), §3, p. 4: "A(st, at) = Q(st, at) − V(st) =
  rt + γV(st+1) − V(st)." In their setting the intermediate reward is zero and γ = 1 (§3, p. 4: "the expected
  undiscounted (γ = 1) finite-horizon return" and "the reward rt is set to zero for all intermediate actions
  at's"), so
  for every step but the last the advantage is V̂MC(st+1) − V̂MC(st): IMPLIED.
- **The unit is a reasoning step, not a token. VERIFIED.** §4, p. 5: "To enhance the efficiency of ÂMC, we group
  states within a reasoning step and compute a single advantage, which is assigned to all tokens in that step".
  How steps are cut, App. B, p. 14: "For the MATH dataset, we begin by splitting solutions based on clear natural
  boundaries such as newline characters or punctuation marks (e.g., periods or commas). [...] if any resulting step
  exceeds 100 characters, we further try to divide it [...]. For the GSM8K dataset, we take a simpler approach,
  segmenting the reasoning steps by newlines alone".
- **The number of rollouts: K = 9 by default; K ∈ {1, 3, 9} in the ablation. VERIFIED.** §5 "Experimental Setup",
  paragraph "Training Details and Hyperparameters", p. 6: "For V̂MC in VinePPO, we conduct a full ablation study on
  K in Section 6.1, with K = 9 used as the default setting unless otherwise specified." The ablation is in §6.4
  "Effect of K" (not §6.1 as the sentence says), pp. 7–8: "We assess the impact of K, the number of MC samples, by
  running an ablation on RhoMath 1.1B, varying K from 1 to 3 and 9. As shown in Figure 7, VinePPO improves with
  higher K since more MC samples reduce the variance of ÂMC. While high variance of MC estimation could
  theoretically hinder training, our results show that even small K values work well in this setting." Figure 7
  caption, p. 7: "Increasing K consistently improves accuracy." The "ground truth" values of their analysis use 256
  samples, §7, p. 8: "by running 256 MC samples and averaging the returns".

## 2. Ng, Harada, Russell 1999 — potential-based reward shaping

- Citation: Andrew Y. Ng, Daishi Harada, Stuart Russell. *Policy invariance under reward transformations: Theory
  and application to reward shaping*. Proceedings of the Sixteenth International Conference on Machine Learning
  (ICML 1999), Morgan Kaufmann, pp. 278–287. The PDF carries no page numbers; the page range is from
  https://mlanthology.org/icml/1999/ng1999icml-policy/. No arXiv copy.
- URL: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
- File: `papers/Ng_1999_Policy_Invariance_Reward_Shaping_ICML.pdf` (10 p.)

Claims:

- **The shaping term is F(s, a, s') = γΦ(s') − Φ(s). VERIFIED: eq. (2), inside Theorem 1.** §3 "Main results",
  p. 4: "We say F is a potential-based shaping function if there exists a real-valued function Φ : S ↦ ℝ such that
  for all s ∈ S − {s0}, a ∈ A, s' ∈ S, F(s, a, s') = γΦ(s') − Φ(s),   (2) (where S − {s0} = S if γ < 1)." (s0 is
  the absorbing state of the undiscounted case.)
- **The theorem that it leaves the optimal policy unchanged is Theorem 1. VERIFIED.** p. 4, sufficiency part:
  "(Sufficiency) If F is a potential-based shaping function, then every optimal policy in M' will also be an
  optimal policy in M (and vice versa)." M' = (S, A, T, γ, R + F), M = (S, A, T, γ, R).
- **The condition is necessary and sufficient. VERIFIED, with the sense of "necessary" stated.** Theorem 1, p. 4:
  "Then, that F is a potential-based shaping function is a necessary and sufficient condition for it to guarantee
  consistency with the optimal policy [...]" and "(Necessity) If F is not a potential-based shaping function (e.g.
  no such Φ exists satisfying Equation (2)), then there exist (proper) transition functions T and a reward function
  R : S × A ↦ ℝ, such that no optimal policy in M' is optimal in M." The necessity holds over all T and R, not for
  one given MDP; p. 5: "(If we do have intimate knowledge of T, R, then the necessity condition does not say much,
  and it is possible that we might be able to use other shaping functions.)" For infinite state spaces Φ must be
  bounded, p. 4: "we really should demand that Φ be bounded, so that the shaping rewards F are also bounded".
- **Related: how the values change.** Corollary 2, p. 5 (with Φ(s0) = 0 if γ = 1): "Q*_M'(s, a) = Q*_M(s, a) −
  Φ(s),   (3)   V*_M'(s) = V*_M(s) − Φ(s).   (4)". Remark 1, p. 5, extends this to every policy without proof:
  "Although we have not proved it here, the identities in Corollary 2 actually hold for arbitrary policies π, not
  just the optimal policy". The proof of necessity is in Appendix A.

## 3. Ciosek & Whiteson 2018 — Expected Policy Gradients

- Citation: Kamil Ciosek, Shimon Whiteson. *Expected Policy Gradients*. AAAI-18, Proceedings of the AAAI Conference
  on Artificial Intelligence 32(1), pp. 2868–2875, doi:10.1609/aaai.v32i1.11607. arXiv:1706.05374v6 [stat.ML],
  13 Apr 2018. The citation given (Ciosek & Whiteson 2018, AAAI 2018) is correct. Extended version: Kamil Ciosek,
  Shimon Whiteson. *Expected Policy Gradients for Reinforcement Learning*. Journal of Machine Learning Research 21
  (2020) 1–51; arXiv:1801.03326v2, 2 May 2020.
- URL: https://arxiv.org/abs/1706.05374 , https://ojs.aaai.org/index.php/AAAI/article/view/11607 ,
  https://jmlr.org/papers/v21/18-012.html
- Files: `papers/Ciosek_2018_Expected_Policy_Gradients_AAAI32.pdf` (8 p.; the pages below, printed page in
  brackets), `papers/Ciosek_2018_Expected_Policy_Gradients_arXiv1706.05374v6.pdf` (12 p., same text with the
  supplement), `papers/Ciosek_2020_Expected_Policy_Gradients_for_RL_JMLR_arXiv1801.03326v2.pdf` (51 p.).

Claims:

- **The estimator takes the expectation over all actions of a state under π(a|s) instead of the one sampled
  action. VERIFIED; the AAAI paper writes it as an integral, and the sum form for discrete actions is written out
  only in the JMLR version.** AAAI, Introduction, p. 1 [2868]: "the main idea is to integrate across the action
  selected by the stochastic policy when estimating the gradient, instead of relying only on the action selected
  during the sampled trajectory." "Main Algorithm", eq. (9), p. 2 [2869], and eq. (10), p. 3 [2870]: "∇̂J =
  Σ_{t=0}^{T} γ^t Î^Q̂_π(st),   (10) where Î^Q̂_π(s) is some approximation to I^Q̂_π(s) = ∫_a dπ(a | s)∇ log π(a |
  s)(Q̂(a, s) + b(s))." p. 3 [2870]: "SPG as given in (2) performs this quadrature using a simple one-sample Monte
  Carlo method. However, relying on such a method is unnecessary. In fact, the actions used to interact with the
  environment need not be used at all in the evaluation of Î^Q_π(s)". The integrand uses a critic Q̂ at every
  action. The discrete sum, JMLR §5.5 "Probability Distributions over Discrete Actions and Softmax Policies",
  p. 22: "In this case, the integral in I^Q_π(s) becomes a sum, and the policy gradient update takes the form
  I^Q_π(s) = Σ_a π(a | s)∇ log π(a | s)Q(a, s) = Σ_a ∇π(a | s)Q(a, s)." The AAAI paper names discrete actions only
  for a concurrent method, p. 3 [2870]: "The idea behind EPG was also independently and concurrently developed as
  Mean Actor Critic (Asadi et al. 2017), though only for discrete actions".
- **What they say about its variance. VERIFIED: strictly lower than the one-sample estimator (AAAI Lemma 3 = JMLR
  Lemma 4), for the part of the variance that comes from sampling the action.** AAAI "Variance Analysis", p. 6
  [2873]: "We now prove that for any policy, the EPG estimator of (10) has lower variance than the SPG estimator of
  (2). Lemma 3. If for all s ∈ S, the random variable ∇ log π(a | s)Q̂(s, a) where a ∼ π(·|s) has nonzero
  variance, then V_τ[Σ_t γ^t ∇ log π(at|st)(Q̂(st, at) + b(st))] > V_τ[Σ_t γ^t I^Q̂_π(st)]." and "Since we know that
  both the estimators of (2) and (10) are unbiased, the estimator with lower variance has lower MSE." p. 3 [2870]:
  "it completely eliminates variance in the inner integral, as opposed to just reducing it." The JMLR version
  names the variance it does not remove, §3.2 "Variance Analysis", pp. 8–9: "Second, the learned critic value Q̂
  itself can be inaccurate [...] We now prove that EPG completely eliminates the first kind of variance."
- **Related: no gain with discrete actions in their one test. VERIFIED.** JMLR §6.5 "EPG for Discrete Action
  Spaces", pp. 30–31 (Atari Pong, A2C with a critic that also learns Q̂): "there is no measurable benefit from
  introducing a sum over actions to the policy gradient estimate. In discrete domains like Pong, the learned critic
  is often inaccurate, which can be a much greater problem than the variance of the policy estimator." JMLR §7.7,
  p. 37, on priority: "For discrete actions, it was first explicitly introduced in an unrefereed draft by Sutton et
  al. (2000b) and independently developed by Bahdanau et al. (2016)."

## 4. Foerster et al. 2018 — Counterfactual Multi-Agent Policy Gradients (COMA)

- Citation: Jakob N. Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, Shimon Whiteson.
  *Counterfactual Multi-Agent Policy Gradients*. AAAI-18, Proceedings of the AAAI Conference on Artificial
  Intelligence 32(1), pp. 2974–2982, doi:10.1609/aaai.v32i1.11794. arXiv:1705.08926v2 [cs.AI], 14 Dec 2017 (a v3
  of 11 Dec 2024 exists; not read). The citation given is correct.
- URL: https://arxiv.org/abs/1705.08926 and https://ojs.aaai.org/index.php/AAAI/article/view/11794
- Files: `papers/Foerster_2018_COMA_AAAI32.pdf` (9 p.; the pages below, printed page in brackets) and
  `papers/Foerster_2018_COMA_arXiv1705.08926v2.pdf` (10 p.; the same equation (4) on p. 4).

Claims:

- **The counterfactual advantage A^a(s, u) = Q(s, u) − Σ_{u'^a} π^a(u'^a | τ^a) Q(s, (u^{−a}, u'^a)): only agent
  a's action is changed, the others' actions are fixed. VERIFIED: eq. (4).** Section "Counterfactual Multi-Agent
  Policy Gradients", p. 4 [2977]: "For each agent a we can then compute an advantage function that compares the
  Q-value for the current action u^a to a counterfactual baseline that marginalises out u^a, while keeping the
  other agents' actions u^{−a} fixed: A^a(s, u) = Q(s, u) − Σ_{u'^a} π^a(u'^a|τ^a)Q(s, (u^{−a}, u'^a)).   (4)
  Hence, A^a(s, u^a) computes a separate baseline for each agent that uses the centralised critic to reason about
  counterfactuals in which only a's action changes, learned directly from agents' experiences instead of relying
  on extra simulations, a reward model, or a user-designed default action." The sum is weighted by agent a's own
  policy π^a given its own history τ^a; Q is a learned centralised critic.
- **Related: the difference rewards it replaces, and how the sum is computed.** p. 4 [2977]: "each agent learns
  from a shaped reward D^a = r(s, u) − r(s, (u^{−a}, c^a)) that compares the global reward to the reward received
  when the action of agent a is replaced with a default action c^a" and "they typically require access to a
  simulator in order to estimate r(s, (u^{−a}, c^a))". The critic takes u^{−a} as input and "outputs a Q-value for
  each of agent a's actions [...] Consequently, the counterfactual advantage can be calculated efficiently by a
  single forward pass of the actor and critic, for each agent." The baseline adds no bias: "the counterfactual
  baseline's expected contribution to the gradient, as with other policy gradient baselines, is zero."

## 5. Guo et al. 2025 — Segment Policy Optimization (SPO)

- Citation: Yiran Guo, Lijie Xu, Jie Liu, Dan Ye, Shuang Qiu. *Segment Policy Optimization: Effective
  Segment-Level Credit Assignment in RL for Large Language Models*. NeurIPS 2025 (39th Conference on Neural
  Information Processing Systems). arXiv:2505.23564v2 [cs.LG], 21 Oct 2025. The citation given (Guo et al. 2025,
  arXiv 2505.23564) is correct; the full title is above. Code: https://github.com/AIFrameResearch/SPO
- URL: https://arxiv.org/abs/2505.23564
- File: `papers/Guo_2025_Segment_Policy_Optimization_arXiv2505.23564v2.pdf` (26 p.)

Claims:

- **A segment's advantage is the difference of the Monte Carlo values at its two boundaries. VERIFIED: eq. (2)
  and §4 (2).** §3 "Segment Policy Optimization", pp. 4–5 (the equation on p. 4): "A^seg_k := V(s_{t_{k+1}}) − V(s_{t_k}) = V([x, seg1, . . .
  , segk]) − V([x, seg1, . . . , segk−1]),   (2) which measures the incremental improvement in expected reward
  resulting from generating the tokens in segk." and, p. 5, "we propose to bypass the critic entirely and instead adopt
  Monte Carlo (MC) estimation". §4 "SPO-Chain for Short CoT", (2) "Chain-based Segment Advantage Estimation",
  p. 6: "at each segment boundary state s_{t_k} = [x, y_{<t_k}], we independently sample N trajectories from the
  policy πθ [...] and then estimate the value of such a state by averaging the returns from these sampled
  trajectories [...]. The estimated advantage Â^seg_k of each segment k is computed by taking the difference
  between the state values at consecutive segment boundaries". N = 9, §6, p. 7: "For SPO-chain, we set the number
  of MC samples to N = 9 and partitioned the model output at intervals of every 5 cutpoints"; p. 5: "We find that
  the accuracy obtained with a small number of samples (N = 4 or N = 9) is sufficient".
- **The tree form: sibling segments, advantage relative to the siblings. VERIFIED.** §5 "SPO-Tree for Long CoT",
  pp. 6–7: the value of a node is the reward at a leaf and otherwise the mean of its children's values; "The
  estimated advantage Â(n) for node n, relative to its siblings, is computed either in unnormalized or normalized
  form as: Â(n) = V̂(n) − mean_{n'∈Sib(n)} V̂(n') or [V̂(n) − mean_{n'∈Sib(n)} V̂(n')] / std_{n'∈Sib(n)} V̂(n')."
  Unlike the chain form, the rollouts are also trained on: "each node (segment) in the tree can be used as a
  training example".
- **How segments are cut. VERIFIED: by a fixed token count, or at an equal number of low-probability tokens
  ("cutpoints") per segment.** §3, p. 4: "(a) Fixed Token Count Partition: A straightforward strategy that divides
  the sequence into segments of a predetermined fixed number of tokens. (b) Adaptive Cutpoint-based Partition: An
  advanced strategy (see Section 4) that defines segments by accumulating a fixed number of low probability tokens
  (i.e., tokens whose probabilities πθ(yt|st) are less than a threshold ρ)." §4 (1), p. 5: "we first identify the
  cutpoints, defined as positions where token probabilities drop below a pre-defined threshold ρ [...] which
  corresponds to partitioning the trajectory such that each segment contains the same number of cutpoints". The
  values, App. H, p. 21: SPO-chain "Cutpoint interval 5", "K (MC samples) 9", "Prob mask threshold 0.9"; SPO-tree
  "M (Tokens per level) 30" (GSM8K) or 600 (MATH), "Branch factors [6,6,6]". The advantage is then put only on the
  low-probability tokens, eq. (3), p. 6: "Mt is the token probability mask whose value is 1 if πθold(at|st) < ρ and
  0 otherwise".
- **Related: VinePPO as a special case.** §2 "Background", p. 3: "Notably, VinePPO can be viewed as a special case within our
  more general SPO framework."

## 6. Hou et al. 2025 — TreeRL

- Citation: Zhenyu Hou, Ziniu Hu, Yujiang Li, Rui Lu, Jie Tang, Yuxiao Dong. *TreeRL: LLM Reinforcement Learning
  with On-Policy Tree Search*. Proceedings of the 63rd Annual Meeting of the Association for Computational
  Linguistics (ACL 2025, Volume 1: Long Papers), pp. 12355–12369, doi:10.18653/v1/2025.acl-long.604.
  arXiv:2506.11902v1 [cs.LG], 13 Jun 2025. Code: https://github.com/THUDM/TreeRL
- URL: https://arxiv.org/abs/2506.11902 and https://aclanthology.org/2025.acl-long.604/
- File: `papers/Hou_2025_TreeRL_arXiv2506.11902v1.pdf` (15 p.; pages below are the arXiv pages)
- Why this paper and not TreeRPO (Yang et al., arXiv:2506.05183v2, read for the choice, not kept): TreeRL is
  peer-reviewed (ACL 2025) and states the step advantage as numbered differences of shared-prefix values, against
  the root and against the parent node (eqs. (2)–(4)). TreeRPO (v2 marked "Under Review") forms GRPO-style groups
  of sibling steps, normalised by its §3.3 eq. (4) with σ = µ(1 − µ); the sibling-relative form is already in SPO-tree
  (entry 5).

Claims:

- **The value of a step is the share of correct leaves below it in the tree. VERIFIED.** §3.2.1 "Process
  Supervision from Tree Search", p. 4: "for a step sn (corresponding to a node in the tree), let L(sn) denote the
  set of all leaf nodes that are descendants of node sn (including node sn itself if it is a leaf). The value
  V(sn) of node sn is computed as the ratio of correct leaf nodes among its descendants".
- **The step advantage comes from these values: global against the root, local against the parent, summed.
  VERIFIED: eqs. (2), (3), (4).** p. 4: "GA(sn) = V(sn) − V(root)   (2)", "LA(sn) = V(sn) − V(p(sn))   (3) where
  V(p(sn)) denotes the value of the node p(sn). LA(sn) > 0 indicates that step sn is more likely to lead to the
  correct result than its parent node, suggesting that this step should be encouraged", "R(sn) = GA(sn) + LA(sn)
  (4)". Shared steps are down-weighted, §3.2.2, p. 5: "As all non-leaf steps appear in multiple sequences [...] The
  reward for each non-leaf step is modified by dividing the square root of the number of leaf nodes in its
  subtree: R(sn) = R(sn)/√|L(sn)|." Algorithm 1, p. 5, gives the same with the labels "Re-weight Factor", "Global
  Advantage", "Local Advantage".
- **Related: where the tree branches, and its size.** §3.1 "Entropy-Guided Tree Search", p. 3: "the top-N tokens
  with the highest entropy values are selected across the whole tree Ti", with "H(yt) = − log πθ(yt | x, y<t)"
  (the negative log-probability of the sampled token, which the paper calls entropy). A step is therefore the
  text between two branch points. §4.1 "Setup", p. 7: "we used sampling parameters (M, N, L, T) = (6, 2, 1, 2),
  generating 30 responses per prompt" (M trees, N branch points per iteration, L iterations, T branches each).
