# Feasibility report: PIPO in the stage C post-training

**Date:** 2026-10-08. **Source paper:** Wang et al., *Policy Improvement Reinforcement Learning*,
arXiv:2604.00860v5 (`docs/post_training/PIPO.pdf`). **Scope:** the post-training of the two-tier model (stage C,
`4dTrajectory/ts_transformer/docs/two_tier/design/post_training.md`), methods `branch` and `value`.

**Language.** This report uses the ASD-STE100 rules for most of the text (short sentences, active voice, one meaning
for each word). Each key term has its Chinese name one time, at its first use, and again in the glossary (§9).
Paths are relative to `4dTrajectory/ts_transformer/` when they start with `post/`, `experiments/` or `docs/two_tier/`.

**Labels.** "The paper" is the PIPO paper. "Measured" is a number from a project readout, with its file. "Claude's
reading" is an inference or a design choice by Claude that the user did not decide.

---

## 1 Summary

| Question | Answer |
|---|---|
| Can PIPO go into the stage C post-training? | **Yes, technically.** The loss already has the parts that PIPO needs: a clipped-ratio surrogate, a group-relative advantage and a pass against a frozen speaker. |
| Can it go in without a large change to the architecture? | **Yes.** One new module in `post/`, one optional argument in `post.loss.passes`, three new campaign settings with defaults that keep the present behaviour, and about 30 lines in `run_campaign`. No schema of a checkpoint changes. |
| Will it lift the landing rate above the present plateau (约 85 %)? | **Probably not by itself** (Claude's reading, §5). The present evidence says that the step size is not the bottleneck. PIPO acts mostly on the step size. |
| What is the recommendation? | Do **step 0 first** (§7): an offline check on the records of C10 and `post_lr3`, at zero training cost. Run a PIPO campaign only if step 0 shows that the feedback signal agrees in sign with the measured improvement. |

---

## 2 What PIPO does

### 2.1 The problem that the paper names

Usual RL post-training (强化学习后训练) is **open-loop** (开环). Each update uses the rewards and the advantages
(优势) of the current batch. No step checks if the update made the policy (策略) better. Finite samples and noisy
rewards can make a local signal point in a bad direction, and nothing corrects it.

### 2.2 The two phases

PIPO (策略改进策略优化, Policy Improvement Policy Optimization) closes the loop. It has two phases in each iteration t:

1. **Exploration (探索阶段).** The base algorithm (基础算法: PPO, GRPO, …) does its usual update on batch B_t.
2. **Verification (验证阶段).** At the next iteration, the new policy samples a fresh batch. Its mean reward µ_{t+1}
   tells if the last update helped. PIPO then does one more clipped update on the **previous** batch B_t. This
   update reinforces (强化) the previous direction when the policy improved, and suppresses (抑制) it when the policy
   became worse.

### 2.3 The equations

- **Performance estimate (性能估计)**, from the fresh batch of policy θ_t:
  µ_t = mean reward of B_t (paper Eq. 7).
- **Historical anchor (历史锚点)**, a sliding window (滑动窗口) of the last K iterations:
  µ_his = mean(µ_{t−1} … µ_{t−K}), σ_his = their sample standard deviation (Eq. 8).
- **Standardized improvement feedback (标准化改进反馈)**: ξ_t = (µ_t − µ_his) / σ_his (Eq. 9).
- **Rectification function (整流函数)**: ϕ_λ(x) = x when x ≥ 0, and λ·x when x < 0. The paper uses λ = 0.1.
- **Policy-improvement reward (策略改进奖励, PI reward)** for each unit i of the previous batch:
  r̂_i = a_{t−1,i} · ϕ_λ(ξ_t) (Eq. 10). For GRPO, the attribution (归因分数) is the group advantage, normalized
  inside the group: a_i = G · A_i / Σ_j |A_j|.
- **PI update (策略改进更新)**: one clipped-ratio step on B_{t−1} with r̂ in place of the advantage. The ratio
  (重要性比) is against the policy that **spoke** B_{t−1} (Eq. 11).

### 2.4 The paper's settings and results

- K = 8, λ = 0.1, α_PI = α_std = 1e-6, about 250 steps of batch 128 × G = 8 (paper Table 5).
- Gains of +0.9 to +2.8 points of mean Pass@1 on math, +0.7 to +1.6 on code, +0.8 to +2.1 on tool use (Tables 1–2).
- Wall-clock cost (时间开销): +40 % to +49 % for critic-free methods (GRPO, GSPO, DAPO); +1.6 % for PPO (Fig. 5).
- Ablations: λ = 1 is too strong; λ = 0 (reinforce only) still helps; K = 32 is too stale (Tables 4, 7).

### 2.5 The limits of the paper's evidence (Claude's reading)

- The main tables are one run for each cell. The seed study (Appendix B.3) is a figure, with no numbers.
- Some benchmarks are small. AIME25 has 30 problems: 18.5 % → 22.2 % is one problem.
- The theory gives local alignment only. It needs the **sign-consistency assumption (符号一致性假设)**
  ϕ_λ(ξ_t)·ΔJ_t ≥ 0 (Assumption C.2): the noisy feedback must have the same sign as the true improvement.
  The paper does not measure how often this is true.
- The group-relative motivation (Theorem 6.2) is about the division by σ_q in GRPO. Our advantage does **not**
  divide by the group's standard deviation (§3.2). Thus this part of the motivation does not apply to us.

---

## 3 The present post-training (the facts that PIPO must fit)

### 3.1 The round (轮)

A campaign (训练活动) is a sequence of rounds (`experiments/post_train.py`, `run_campaign`). A round r:

1. Draws windows (窗口) of the train days (real, A, D, B; for example 4 × 1,000).
2. Speaks each window one time with the round's start model M_r (the **first pass**, 首遍). A window with reward 1
   gives no sample.
3. Speaks the other windows again at each branch point (分支点), K = 8 continuations (续句). A branch group
   (分支组) is the first sentence plus its continuations. The groups go to `round_r/groups_k.pt`.
4. Trains `epochs` passes over the groups. Every pass's ratio is against `PassStart(M_r)`.
5. Reads the selection readout (选择读数): 1,000 fixed select windows, fixed random numbers.
6. Writes `round.json`, then `checkpoint.pt`, then **deletes the groups files**.

### 3.2 The advantage and the loss

- Advantage of a sentence = its reward − the mean of its group (`post/branches.py`, `Group.advantages`). No division
  by the group's standard deviation. A group with equal rewards gives no sample.
- The advantage applies to the rows after the branch point (with `segment_only`, up to the next branch point).
- Loss of an update (`post/loss.py`): clipped surrogate (ε = 0.2) + 0.04 × pull to the base (KL) + 1.0 × teacher-forced
  data term. One update is in pieces, **one branch group a piece**.
- Method `value` (D171, decided 2026-10-07): one sentence for each window, GAE advantage for each row, a value
  network V. Its code is not on `dev-two-tier` yet.

### 3.3 The measured state (2026-10-07, `docs/two_tier/readouts/2026-10-07_stage_c_experiments.zh.md`)

| Campaign | What changed | Landing rate on select, by round |
|---|---|---|
| C10 (`post_train_20261006`) | branch, K = 8, 14 rounds | 80.2 % → 85.1 % (round 6) → 85.7 % (round 8); flat after round 6 |
| C17 (`post_landed_20261007`) | learn the best landed sentence | 84.2–85.1 % (4 rounds) |
| `post_branch16_20261007` | K = 16 | 84.4–85.8 % (4 rounds) |
| `post_lr3_20261007` | 3 × learning rate, clip-norm 1.0 | 84.4–85.8 % (6 rounds); loss of separation 13.0 % in round 5 |

The noise of one readout is about ±1.2 points. The ceiling readout C15 says that 99.3 % of windows land at least one
time in 32 tries. Thus the model **can** say a landing sentence. It gives that sentence too low a probability.

The readout of `post_lr3` (Claude's reading in that file): larger steps did not help. The remaining explanation is
the credit assignment (信用分配) — one reward at the end of a sentence, spread over many words.

---

## 4 How PIPO maps to this project

### 4.1 The iteration is a round

PIPO needs a **fresh batch from the updated policy** to measure µ. In our loop, new samples come only at the start of
a round. The updates inside a round use the same samples. Thus **one PIPO iteration = one round**. An update is not an
iteration.

This gives a big difference in scale:

| | Paper | This project |
|---|---|---|
| Iterations in a run | about 250 | 4–14 rounds |
| Window K of the anchor | 8 | 8 would start PIPO at round 8, after most campaigns stop |
| Samples for µ | 128 prompts × 8 | 2,800–4,000 first sentences |
| Typical gain for each iteration | rising curve | about 0 on the plateau |

### 4.2 Which µ to use

| Option | What it measures | Problem |
|---|---|---|
| **A. Mean reward of the round's first pass (recommended)** | M_r on the round's own train windows. It is the paper's µ_t exactly. | The windows change from round to round. Sampling noise ≈ √(0.85·0.15/4000) ≈ 0.56 points (Claude's arithmetic, for 4,000 windows and a reward that is near 0/1). |
| B. Selection readout | The model after the pass, on fixed select windows | **Do not use.** The select days choose a round (D7). A training signal from them uses the select days for training. |

Option A needs no new speaking. The first pass already flies every window and gives its reward.

### 4.3 The match between the loss parts

| PIPO part | Our part | Change |
|---|---|---|
| Group attribution a_i = G·A_i / Σ\|A_j\| | `Group.advantages()` (A_i = r_i − mean) | Scale each piece by one number. A piece is one group, so the factor is a scalar of the piece. |
| Clipped surrogate J_PI | `post.loss.update_step` / `_surrogate_words` | None. Reuse it. |
| Ratio against the policy that spoke B_{t−1} | `PassStart` | Make a `PassStart` from the checkpoint of round r − 2 (it **is** M_{r−1}). No new file. |
| B_{t−1} | `round_{r−1}/groups_*.pt` | Keep these files one round longer. |
| PPO case (attribution = GAE advantage) | Method `value` (D171) | The same PI step applies with no group normalization. V does not move in the PI step. |

---

## 5 Expected effect (Claude's reading)

### 5.1 What the PI step is in our loop

When ξ > 0, the PI step is one more clipped pass over the previous round's groups, in the same direction, with a
multiplier ϕ(ξ). It is close to "one more epoch, one round late". When ξ < 0, it is a small step (× λ) against the
previous direction.

### 5.2 Why the gain is probably small on the present plateau

1. **The step size is not the bottleneck.** `post_lr3` made each round's step about 4 times larger (clipped words
   0.3 % → 1.3 %, KL 0.036 → 0.051). The landing rate did not move. The "reinforce" half of PIPO only makes
   steps larger in a direction that the data already chose.
2. **On a plateau, ξ is mostly noise.** When the true gain per round is about 0, the sign of ξ is close to a coin flip.
   Then the PI step adds a random ± push. The sign-consistency assumption (§2.5) is false for about half the rounds.
3. **σ_his from few rounds is not stable.** With K = 3 or 4, σ_his comes from 3 or 4 numbers. ξ then has heavy tails
   (a t distribution with K − 1 degrees of freedom). One small σ_his gives a very large multiplier. The paper has
   no clamp on ξ.
4. **The boundary argument does not apply.** Our advantage has no σ division (§2.5, §3.2).

### 5.3 Where PIPO could help

1. **Stability (稳定性).** The "suppress" half can damp a bad round, for example `post_lr3` round 5 (loss of
   separation 13.0 %). This is a guard, not a lift.
2. **The early, rising part of a campaign.** In C10 rounds 0–6, the gain was about 0.8 points a round, against a µ
   noise of about 0.56 points. The signal-to-noise ratio (信噪比) is then near 1.5. But PIPO starts only after K
   rounds, so it misses most of this part.
3. **The value method with more rounds.** Method `value` speaks one sentence a window. Its rounds cost less, so a
   campaign can have more rounds. More rounds make the anchor window useful.

---

## 6 Design with low intrusion

### 6.1 Principles

- PIPO is **off by default**. With the default settings, a round is the same as today: the same calls, the same
  random numbers, the same files. A test checks this.
- No checkpoint format, identity or `round.json` field changes. PIPO adds a new `pipo` block in `round.json` only
  when it is on.
- The new settings follow the user's standing permission (root `CLAUDE.md`, 2026-10-07): a new campaign setting has a
  default that is the old behaviour; a record without the field reads as that default; no old record is edited.
- `post/` imports no `autopilot/` and no `experiments/` (the architecture test).

### 6.2 New settings (`Settings` in `experiments/post_train.py`)

| Setting | Default | Meaning |
|---|---|---|
| `pipo_window` | `None` | K of the anchor. `None` = PIPO off. |
| `pipo_lambda` | `0.1` | λ of ϕ_λ. |
| `pipo_lr_scale` | `1.0` | α_PI / α_std. The paper uses 1. |

Add the three names to `SETTINGS_ADDED`. Refuse them by name with method `landed` (that method has no advantage).
`__post_init__` checks `pipo_window ≥ 2` (σ_his needs two values) and `0 ≤ pipo_lambda ≤ 1`.

### 6.3 New module `post/improvement.py` (about 60 lines, torch only for the scaling)

```python
def feedback(history: Sequence[float], current: float, window: int) -> float | None:
    """ξ of PIPO Eq. 9: (current − mean of the last `window`) / their sample std; None while history < window."""

def rectify(xi: float, lam: float) -> float:
    """ϕ_λ of PIPO Eq. 10."""

def scaled(piece: Samples, factor: float) -> Samples:
    """One branch group with its advantage × factor × G / Σ|A_j| (the group attribution of PIPO §4.1)."""
```

If σ_his is 0 (equal µ in the window), `feedback` raises an error by name. It does not return a fallback.

### 6.4 One optional argument in `post/loss.py`

```python
def passes(model, base, optimizer, each, step=update_step, clip_norm=None,
           start: PassStart | None = None) -> list[list[LossParts]]:
    start = PassStart(model) if start is None else start
```

The default keeps the present behaviour. The PI step calls `passes` with the `PassStart` of M_{r−1}. The base pass
of a round with PIPO calls it with the `PassStart` of M_r, made **before** the PI step, because M_r spoke B_r.

### 6.5 The changes in `run_campaign` (about 30 lines)

The order in round r, with PIPO on:

1. Speak (unchanged). Compute µ_r = mean reward of the first pass. Write it in the round's record.
2. Make `PassStart(M_r)` (the speaker of B_r).
3. If `feedback(µ_{r−K} … µ_{r−1}, µ_r, K)` is not `None` and round r − 1's groups exist:
   - Load M_{r−1} (the checkpoint of round r − 2, or the campaign start for r − 1 = 0) as a `PassStart`.
   - Run one pass over `round_{r−1}/groups_*.pt`, each piece through `scaled(piece, ϕ_λ(ξ_r))`, with the learning
     rate × `pipo_lr_scale`, the same data term pairs rule, the same `clip_norm`.
4. Base pass over B_r with `start = PassStart(M_r)` (step 2).
5. Readout, `round.json` (with a `pipo` block: µ_r, the window, ξ_r, ϕ, the PI pass means), checkpoint.
6. Delete `round_{r−1}/groups_*.pt`. Keep `round_r/groups_*.pt` for round r + 1.

**Resume.** µ history comes from the earlier `round.json` files. A round is done when its checkpoint exists (as
today). The groups of round r − 1 are deleted only after round r's checkpoint, so a crash in round r leaves them
for the rerun.

**Question for the user — the PI step's loss.** The paper's J_PI is the surrogate only. Our `update_step` also adds
the pull to the base and the data term. Option 1 (Claude's suggestion): reuse `update_step` as it is, so the PI step
keeps the two guards and needs no new loss code. Option 2: a surrogate-only step, as the paper.

### 6.6 Cost

| Item | Cost |
|---|---|
| Training | One more pass over the previous round's groups. About the cost of one base pass (`epochs = 2` level). |
| Speaking | None (µ comes from the first pass). |
| Disk | One more round of groups files (the size is in `round.json` `groups_bytes`). |
| Memory | One more frozen copy of the prior (about 4.9 M parameters) during the PI pass. |
| Code | About 150–200 lines with tests (Claude's estimate). |

### 6.7 Tests

1. PIPO off: a smoke round gives the same groups, the same pass means and the same checkpoint as the code before.
2. `feedback`: `None` before K values; the sign and scale on known inputs; an error when σ_his = 0.
3. `scaled`: a group's attributions sum in absolute value to G × factor.
4. The PI pass's ratio is 1 at its first update when the model equals M_{r−1}.
5. Resume: a round killed after the PI pass reruns with the same ξ and the same groups.

---

## 7 Recommended steps

### Step 0 — offline check of the signal (no training; recommended first)

Use the records that exist on the compute box (C10, `post_lr3`, `post_branch16`):

1. For each round r, compute µ_r, the mean reward of the first pass. The speaking record counts the first-pass
   ends; confirm that it gives the mean reward, or compute it from the ends.
2. Compute ξ_r for K = 3, 4, 8.
3. Compare the sign of ξ_r with the sign of the select readout change that it should predict: the readout of round
   r − 1 (it reads M_r) against the mean readout of the K rounds before.
4. Report the share of rounds with the same sign, and the size of ξ.

**Decision rule (to be set by the user before the check).** Claude's suggestion: if the signs agree in fewer than
about 70 % of the rounds, the feedback is noise at our scale; stop here and do not build PIPO.

### Step 1 — one campaign, only if step 0 passes

- `post_train --method branch` from C10 round 8, K = 8 continuations, the base learning rates, `pipo_window` from
  step 0, λ = 0.1, `pipo_lr_scale` = 1, 8 rounds, a new seed, select seed 1337.
- PIPO starts after `pipo_window` rounds. **Question for the user:** may the anchor start from the µ of the source
  campaign's last rounds (the same model line), or must each campaign build its own (cleaner, but loses rounds)?
- Stop rule as the other stage C campaigns: stop if the last round is not above the start + 1.2 points.
- Write the campaign's intent in `docs/experiments/intents.json` before publication (L27).

### Step 2 — later

If method `value` (D171) runs with many short rounds, PIPO fits it better (§5.3 item 3). Do the same step 0 on its
records first.

---

## 8 Risks and open questions

| # | Risk or question | Who decides |
|---|---|---|
| R1 | ξ is noise on the plateau; the PI step adds a random push | Step 0 measures it |
| R2 | σ_his from few rounds gives a large multiplier. A clamp on ξ would be an approximation, so it must be an explicit setting with a notice, not a silent cap | User |
| R3 | The window kinds and difficulty change from round to round, so µ_r also moves when the policy does not | Step 0 shows the size |
| R4 | PI loss: surrogate only (paper), or `update_step` with the pull and the data term (§6.5) | User |
| R5 | The anchor of a campaign that starts from another campaign's round (§7 step 1) | User |
| R6 | The priority against candidate 5 (finer credit, P55) and method `value` (D171), which attack the bottleneck that the readouts name | User |

---

## 9 Glossary (术语表)

| English | 中文 |
|---|---|
| Policy Improvement Reinforcement Learning (PIRL) | 策略改进强化学习 |
| Policy Improvement Policy Optimization (PIPO) | 策略改进策略优化 |
| open-loop / closed-loop optimization | 开环 / 闭环优化 |
| exploration phase / verification phase | 探索阶段 / 验证阶段 |
| policy improvement | 策略改进 |
| historical anchor, sliding window | 历史锚点，滑动窗口 |
| standardized improvement feedback ξ | 标准化改进反馈 |
| rectification function ϕ_λ | 整流函数 |
| rectification coefficient λ | 整流系数 |
| policy-improvement reward (PI reward) | 策略改进奖励 |
| local attribution score | 局部归因分数 |
| retrospective update | 回溯更新 |
| sign-consistency assumption | 符号一致性假设 |
| group-relative advantage | 组相对优势 |
| importance ratio, clipped surrogate | 重要性比，裁剪代理目标 |
| credit assignment | 信用分配 |
| branch group, branch point, continuation | 分支组，分支点，续句 |
| round, campaign | 轮，训练活动 |
| first pass | 首遍（第一遍说话） |
| selection readout | 选择读数 |
| pull to the base (KL) | 拉回 base 的 KL 项 |
| signal-to-noise ratio | 信噪比 |
| wall-clock cost | 时间开销 |
