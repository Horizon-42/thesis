# Passages cut from live documents, verbatim

Cut on 2026-09-23 when the vocabulary was archived: each block is the removed text exactly, under
the file it came from and where in it. What replaced it (a shorter sentence, a pointer, or
nothing) is in the same commit's diff.

## `4dTrajectory/ts_transformer/CLAUDE.md` — the prediction paths, the `manoeuvre` bullet

- **`manoeuvre` — the SECOND LAYER's line, being rewritten.** The **intent-code** layer (a learned
  FSQ code per segment, the executor conditioned on it, a causal prior over codes) is **ARCHIVED
  2026-09-20**: `archive/manoeuvre_codes_2026_09/` (README there; tokenizer, sequences, prior,
  readout, `plan_token.py`, gates T/X/P/E/S copied to `gates_manoeuvre.py`, the `manoeuvre_*`
  runners and `two_tier_b_queue`). Why: plan v3 §10's audit — both layers trained on truth and
  only ever evaluated closed-loop, and the truth codes were indexed by time, not by where the
  executor was — so stage B was rewritten (2026-09-20, `docs/2026-09-18_two_tier_plan_v3.zh.md`
  §5.2 / §6.2) around an INSTRUCTION vocabulary with closed-loop post-training of both layers.
  Numbers: `docs/2026-09-18_manoeuvre_token_results.zh.md` §9–§11 and the campaign trees
  `4dTrajectory/outputs/KRDU/experiments/{two_tier_v3_b_20260919,manoeuvre_tok_20260918}` +
  `outputs/codebooks/` — read them through §10 item 1 (open-loop-trained executors). `plan_conditioning`
  is `off` or `instruction` (the words in force over the segment, `outputs/control/instruction_token.py`);
  `manoeuvre-code` is refused at load by name (`PLAN_CONDITIONINGS_RETIRED`).
  **Live**: the closed loop — `lockstep.py` (protocols `none` / `truth-instruction`, payload schema
  `ts-manoeuvre-lockstep-v5`), `gates.py` (grid + relative), `segments.py` (the start frame),
  `instructions.py` (the vocabulary + labeller, a leaf), `instruction_sequences.py` /
  `instruction_prior.py` (the sentence prior), `context.py` (the type vocabulary), `failure_modes.py`, runners
  `manoeuvre_lockstep`, `executor_*`, `two_tier_grid_queue` (R8) — i.e. two-tier v3 stage A
  (P10, D30 / D31 / C28 / R7 / R9 are the archived layer's records).

## `4dTrajectory/ts_transformer/CLAUDE.md` — Contracts, the C29 / C30 lines


- Current `course_frame` coordinates: cross-track is right-positive; relative track angle is left-positive; height zero includes threshold crossing height (C29).
- Instruction vocabulary design uses signed turns from the issuing model `psi` and heights above airport MSL elevation; these semantics are not yet implemented (C30).

## `4dTrajectory/ts_transformer/CLAUDE.md` — Current defaults, the `plan_conditioning` row

| `plan_conditioning` | `off` | `instruction` (2026-09-20, plan v3 §5.2.1): the executor is handed the instruction words IN FORCE at the Δ/τ positions of its segment (bin centres, `Vocabulary.conditioning`) from the artefact `instruction_vocabulary` names — the truth's words in training / `predict`, by flown position in the closed loop; the checkpoint stores the vocabulary's sha and refuses an artefact whose sha moved. `manoeuvre-code` ARCHIVED 2026-09-20 (D30 / D31 are its record) and `truth-next` / `waypoints` 2026-09-18 — all three refused at load by name, each pointing at its own archive (`PLAN_CONDITIONINGS_RETIRED`) |

## `4dTrajectory/ts_transformer/CLAUDE.md` — Current defaults, the vocabulary bullets (H5–H12)

- **The instruction vocabulary in force is `2b8bf25c2a36` / `segment-v14`** (2026-09-21): runway
  `AIRPORT:ident` (idents collide across airports; 22 classes on the pooled cohort, of the 23 the
  manifests hold), heading 72 directions at 5° **plus one POSITION word** (track the centreline —
  the only word that is not a velocity target, H11), vertical =
  **flight path angle**, six modes, descent POSITIVE, speed 16 fitted ground-speed centres,
  duration 2 s/151, terminal 3. Tolerances and runway classes are deliberately OUTSIDE the sha.
  Absolute targets are TILED from the data's own 2 s rows — plateaus could not see a ramp, and
  half of every speed profile had no word responsible for it; the vertical, a RATE, by DP
  piecewise-linear fitting of height against horizontal distance (H8, H11). The vocabulary's own
  specification in English: `docs/instruction_vocabulary.en.md`.
- **The instruction prior's first pooled readings** (2026-09-21, two seeds, `two_tier_v3_bprime_20260921`):
  it beats "repeat the last word" on every kind (total NLL 5.80 vs bigram 7.44), but read on the
  positions where the truth CHANGES — which is what an instruction is — recall is 0.55 heading /
  0.44 vertical / **0.17 speed**, and the joint top-8 covers only 20.5 % of truth tuples. **The
  runway row is a COPY, not a prediction** (it is also a context token; plan §3.2 has not been
  done). Seed line: ±0.003 nats, ±0.003 top-1. → `docs/2026-09-21-instruction_prior_results.zh.md`
- **What the vocabulary can SAY, per signal** (2026-09-21, pooled val): heading settled p50 0.41° /
  p90 2.35° and inside its band 80.4 % of the time; vertical — the 5-segment fit is RMS 11 m but
  quantising to the six modes costs p50 35 m / p90 150 m of height; **speed settled p90 18 m/s,
  inside its band only 48.5 %**, because only 51.3 % of an approach is a speed PLATEAU and 61 % of
  it is decelerating — a ramp read as holds, which is the reading rule and not the class count.
  The prior's 0.17 recall on speed changes and H9's 2,035 → 907 m are the same fact (H10).
- **A SENTENCE MUST BE ABLE TO LAND — `run_ts.py instruction_replay` is the gate** and it flies
  what the artefact SAYS, never a re-reading (the user's rule, 2026-09-21). It caught three things
  a vocabulary readout cannot: an absolute heading word cannot say which way round to turn (9.9 %
  of heading changes are exactly a half circle, and guessing mirrors the whole track), it cannot
  hold a LINE (the failures reach the threshold aligned but 2,464 m to the side, where the real
  tracks are 13 m), and the LANDING was in no word at all (the terminal word sat on the last
  change, a median 136 s early). 36.7 % → **99.6 %** landed. A word's resolution is NOT the gap:
  un-quantising the speed word buys 18 %, the observed speed 55 % (H9).
- **If the vertical word were an ALTITUDE, how it would have to be divided** (2026-09-21, KRDU
  only, a design check): under the tiling reader the bin width is the only lever (tolerance is
  free, instructions = bin crossings, error × instructions ≈ 0.3 × the descent); the grading has to
  be GEOMETRIC and the merge takes it unchanged by tiling `ln(1 + h/h0)`; the only altitudes that
  are instructions are the level-offs and they sit on the 1000 ft **MSL** grid, which
  height-above-threshold destroys — plus a +97 ft, season-walking, per-day offset no bin absorbs.
  Best division measured: geometric under 300 m + the 1000 ft MSL ladder above, 17 words and 9.7
  instructions a flight. **Never flown through the replay gate** (H12).
- **Its three superseded measurements, kept because they are what decided it**: the 1000 ft
  altitude bin was indistinguishable from random rounding and 56 % of its "instructions" were the
  threshold crossing (H5); the POSITION angle to the threshold is a criterion, not a word (p50
  +0.04° off the published glidepath, 88.5 % inside ±0.5° — a word the model would always emit),
  while the FLIGHT PATH angle, which is the word, is broad (p50 2.52°, 9.4 % level) — read the
  published glidepath PER RUNWAY, KRDU 32 is 3.50° (H6); and an absolute target's error is bounded
  by half a bin while a rate's compounds, which is what the altitude word was for (H7).

## `4dTrajectory/ts_transformer/CLAUDE.md` — Layout, the `manoeuvre/` rule

**`manoeuvre/`**: `segments` is the one LEAF
  (data plane, `config`, `io_utils`, torch only — an allow-list) and **nothing under `outputs/`
  imports `manoeuvre` any more**; only the runners do. The two-leaf rule and its reverse edge (the
  executor holding the tokenizer as a submodule) were ARCHIVED 2026-09-20 with that layer — L29 is
  their record, and a new edge from `outputs/` amends `MANOEUVRE_LEAVES` and the rule together.

## `4dTrajectory/ts_transformer/CLAUDE.md` — Where to go next, the second-layer row

| building or reading the **second layer** (what the executor is told each segment, a causal prior over it, later a multi-aircraft graph with separation masks) | **`docs/2026-09-18_two_tier_plan_v3.zh.md`** — the OVERVIEW (intent, outline, the metric and readout protocol §3, the framework §4, the 2026-09-20 audit §10) and an index to the two stage documents it was split into on 2026-09-20: **`…_v3_A.zh.md`** (stage A, the no-token executor's (L, Δ) grid, run) and **`…_v3_B.zh.md`** (stage B: an INSTRUCTION vocabulary of SIX word kinds — runway, heading, altitude, speed, intercept, duration — laid out as an EVENT SEQUENCE rather than an even grid (D52/D70/D71, 2026-09-20), closed-loop post-training of both layers, goal-directed decoding; B0′/B0′′ done, no arm trained). The intent-CODE version

## `4dTrajectory/ts_transformer/docs/reference/layout.md` — L29, the live-now sentence

**Live now**: `manoeuvre.segments` is the ONE leaf (the data plane, `config`, `io_utils` and torch — `test_the_manoeuvre_leaves_import_no_layer_above_the_data_plane`), and **nothing under `outputs/` reaches `manoeuvre` at all** — only the runners under `experiments/` do (`test_only_the_runners_reach_the_manoeuvre_package`). The reverse edge the two-leaf rule existed for (the executor holding the tokenizer as a submodule) went with the archive; a new edge from `outputs/` is a layering decision that amends `MANOEUVRE_LEAVES` and that rule together.

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_plan_v3.zh.md` — 审核表 §5 行

| §5 | 实验阶段与门（阶段 A 已跑完，结论与索引在 5.1；阶段 B 2026-09-20 按指令词表重写在 §5.2；C / D 待 B 结果） | 待过 |

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_plan_v3.zh.md` — 审核表 §6 行

| §6 | 决定项（阶段 A 的 D1–D11、D28–D29；阶段 B 的 D49–D60 在 §6.2，作废的 D31–D38 / D43 / D44 / D48 留编号） | 待过 |

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_plan_v3.zh.md` — 审核表 §7 行

| §7 | 里程碑与交付（阶段 A 已到 M-A2；阶段 B 在 §7.2） | 待过 |

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_plan_v3.zh.md` — 审核表 §8 行

| §8 | 队列与成本（阶段 A；阶段 B 在 §8.2） | 待过 |

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_plan_v3.zh.md` — §4 阶段 B 小节

### 阶段 B：指令词表的第二层（2026-09-20 重写；详见 §5.2、§6.2、§7.2、§8.2、§9.2）

- 第二层 = 一串**管制指令词**（绝对目标：航向、高度、速度、切入航道；每 τ 秒一个位置），由自建词表从航迹里读出来；执行器改吃"当前生效的指令目标"；两层都先在真值上预训练、再在自己飞出的历史上闭环再训；部署时先验给出 K 个候选指令，各交给执行器飞一段，按到达跑道入口的目标打分，选一条（"会说话"，不是模仿）。
- 判定仍是相对门：同一执行器家族、同一 cohort、同一闭环读数下，带指令的读法对无 token 基线的 established（全部与雷达引导组）与雷达 ADE。先用真值指令读上限，再读先验 + 目标导向解码的部署形态。
- 2026-09-19 的意图码版本（K16 码本 + top-1 先验）作废：它的执行器从未在自己飞出的历史上训过、码是压缩的段、解码无目标，读数留作证据（结果文档 §9–§11，计划 §10 审计）。

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_plan_v3.zh.md` — §5 的文档索引

**阶段 A 与阶段 B 各自一份文档**（2026-09-20 拆开，原文太长）。节号沿用本文的编号，所以别处引用 §5.2.1、D51
这类记号不受影响。

| 节 | 在哪 |
|---|---|
| §5.1 阶段 A、§6.1、§7.1、§8.1、§9.1 | **`2026-09-18_two_tier_plan_v3_A.zh.md`** |
| §5.2 阶段 B、§6.2、§7.2、§8.2、§9.2 | **`2026-09-18_two_tier_plan_v3_B.zh.md`** |
| §1–§4（意图、大纲、指标与读数协议、实验框架）、§10 审计 | 本文 |
| 两个阶段的全部读数 | `2026-09-18_two_tier_v3_results.zh.md`（A 在 §1–§8，B 在 §9–§14） |
| 词表本身的说明文（词是什么、怎么读、两层怎么用） | `2026-09-20_instruction_vocabulary.zh.md` |

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_v3_results.zh.md` — §12–§15（阶段 B′ 的全部读数）

## 12. 阶段 B′ · B0′：指令词表与人工核（2026-09-20；计划 §5.2.1、D51 / D53；campaign `two_tier_v3_bprime_20260920`）

词表本身的设计说明（词是什么、怎么读、两层怎么用）独立成篇：`2026-09-20_instruction_vocabulary.zh.md`。本节只放读数。

读什么：B cohort（网格的 L60_D60 那份，记录 ≥ 120 s：train 6853 / val 1404 架）的每条航迹，用
`run_ts.py instruction_vocabulary` 读成指令词，τ = 10 s。每一版的产物都留着：`vocabulary_tau10/`（当前版）与
`_superseded/vocabulary_tau10_v*/`（六份，共 **515 MB**，只供对照，不再引用它们的数；磁盘只剩 5.6 GB，队列启动要 3 GB——要不要删，等用户一句话）。人工核 = 每轮抽 train 航班（直线进近 / 雷达引导各一半，按
`approach_difficulty` 分层抽，seed 1337），三个 opus agent 各看一份图，记"对 / 漏读 / 读错"；这不是用户亲手核的数，
用户看图时可以推翻。

### 12.1 档位（D51）：先按起点档位读一遍，再按读出来的分布定

| 版 | 读法 | 高度 / 速度上限 | sha（前 12 位） | 用来做什么 |
|---|---|---|---|---|
| v0 | 变化率阈值（转弯 \|航向率\| > 1°/s ≥ 6 s；下降 < −1.5 m/s ≥ 10 s；减速 < −0.15 m/s² ≥ 10 s） | 8000 ft / 230 kt（计划的起点） | 6648f8f7919b | 只看目标值的分布 |
| v1 | 同上 | 10 000 ft / 320 kt | a9b131916810 | 第一轮人工核 |
| v2 | 改用 level（±4° / ±30 m / ±2.5 m/s 保持 ≥ 20 s） | 同上 | 551ab0556914 | 第二轮人工核 |
| v4 | level + 漂移限制 + 切入按指令 + 最小变化量 | 同上 | cf275801e0ad | 第三轮人工核 |
| v6 | v4 + 切入只认"之前不在航道上"的航向指令 + 漂移按拟合斜率 + 指令发出于离开**当前那条指令**的 level | 同上 | 7e903e67d8a9 | 第四轮（只核 v4 → v6 读法变了的 209 页）；**作废**：发得早 100–200 s |
| v8 | v6 退回发出时刻（离开**紧挨着的前一个** level）+ 最小变化量只压抖动（守住 40 s 的 level 自成一词）+ 速度下限 100 kt | 10 000 ft / **100–320 kt（23 词）** | f8eb7968a250 | 第五轮（新抽 150 页）；**已被 v9 取代** |
| v9 | v8 + 守住 40 s 的 level 还必须和当前那条指令差出一个容差（修第五轮抓到的格子边界拆词） | 同上 | 1586172a6ec6 | **词表定稿候选** |

v0 读出来的目标值（train）：高度目标 p50 1746 ft、p95 7757 ft，起始那条 p95 7992 ft（进圈处 8000 ft MSL 一级高度）；
地速目标 p50 192 kt、p95 290 kt，起始那条 p50 228 kt、p95 307 kt。起点上限把 6.3 % 的高度词（1021 / 16 153）和
27.8 % 的速度词（10 516 / 37 857）压在最边上那一格（clamp），所以档位定为：

| 类 | 档位 | 词数 | 定的理由 |
|---|---|---|---|
| 航向 | 相对最后进近航道 10° 一格，转一圈 | 36 | 目标离格子中心 p95 4.3°（半格 5°），36 格全用到；航道方向 40 %、下风边（180°）13 % |
| 高度 | 跑道入口以上 1000 ft 一格，0 … 10 000 ft（顶格中心 = 上限） | 11 | 10 000 ft 之上只剩 0.6 %；clamp 1021 → 3 个词。第 0 格（降到入口）50 %、3000 ft 21 %、2000 ft 12 % |
| 地速 | 10 kt 一格，100 … 320 kt | 23 | 320 kt 之上 0.1 %；下限先取 120 kt，但 105 kt 的最后进近被压到 120 kt 那一格、差了一整格（第四轮抓到），改 100 kt 后 clamp 151 → 21 个词；各格 5–12 %，没有空格 |
| 切入 | ≤ 30° / 30–45° / > 45° | 3 | 见 12.5：> 45° 那一格占一半以上，不能像计划那样只留两格 |

共 73 个词（加"保持"，也就是这个位置上四个目标都没换）。速度是地速（ADS-B 没有空速，风在里面），文档里不当空速用。

### 12.2 五轮人工核

| 轮 | 读法 | 对 | 直线进近 | 雷达引导 | 漏读 | 读错 | 主要毛病 |
|---|---|---|---|---|---|---|---|
| 1 | v1 变化率阈值 | 271 / 300（0.903） | 136 / 150 | 135 / 150 | 10 | 19 | 缓慢的减速 / 下降过了阈值才算开始，指令晚发 35–65 s；一条长斜坡被切成几个滚动的目标；慢于 1°/s 的转向（−20° → 0°，30 s）读不到 |
| 2 | v2 改用 level | 238 / 300（0.793） | 117 / 150 | 121 / 150 | 0 | 62 | 切入标在对准之后的小摆动上（一个 agent 的 11 个读错全是它）；≈ 3 m/s 的缓降 20 s 内穿过 ±30 m 带就被当成 level，直线下降被切成 305 / 610 m 的假目标（另一个 agent 的 19 个读错全是它）；缓慢减速的速度词在减速快结束时才发；8 kt 的摆动跨格就成一个词 |
| 3 | v4 | 290 / 300（0.967） | 146 / 150 | 144 / 150 | 1 | 9 | 一个 share 的 6 个读错全是速度词晚发 30–60 s（中间有一层被压掉了，发出时刻按那一层算的）；已在航道上的记录在 t = 0 处画了切入（代码复核发现的 blocker，v5 改）；20 s 边缘的 level 忽有忽无 |
| 4 | v6（只核 209 页变了读法的） | 177 / 209（0.847） | 85 / 76 | 92 / 133 | 0 | 32 | 全是速度词**发得太早**：中间那层若是飞机真守住的，v6 把发出时刻算到更早的一层去了，橙线比真正的减速早 100–200 s |
| 5 | v8（新抽 150 页，每层 75） | **136 / 150（0.907）** | 68 / 75 | 68 / 75 | 0 | 14 | 读错 8 条地速、6 条航向，高度和切入一条没错；其中 6 页是同一个 bug——没动的一段中位数压在格子边界上被拆成两条（v9 修掉） |

第 1 轮刚过 0.9，但毛病是系统性的（变化率阈值本身的问题），所以换了读法；第 2 轮低于 0.9 的三种毛病都在 v4 里改了；
第 3 轮过线；第 3 轮剩下的"晚发"我改成了 v6——**改错了**，第 4 轮把它抓了出来（0.847）。第 4 轮之后不再靠人工核调规则，
先量（12.3），量完才定 v8，第 5 轮重新抽样核一遍，又抓出格子边界那个 bug，定为 v9。

**这几轮的数不能直接比**：三个 agent 用的是同一份说明，但"早发或晚发多少算错"各人取的阈值不同（第 4 轮三份分别判了
10、21、1 个读错，同一份代码）。第 5 轮的说明把这个阈值写成了数字（20 s，两个句子位置），三份之间才可比。

### 12.3 两项不看图的量测（第 4 轮之后；定下 v8 的就是它们）

两项都在整个 train split（6853 架）上算，对每条指令（除记录开头那条）：

**(a) 指令发出的时刻**：`早` = 发出之后信号还在**上一条词的目标值**容差内待了多久；`晚` = 发出之前信号已经离开那个值
多久。只有超过 τ = 10 s 才动得了句子（词是每 10 s 采一次的）。另加一项按读图人的眼睛定义的：`画成机动、其实没动的
最长一段` = 一条词画出来的区间（issued → settled）里，信号停在某个值上不动的最长一段——读图的人说"这条词发错了"，
说的就是它。表里的"相邻指令对" = 每架每类的指令数 − 1。

| 版本 | 通道 | 相邻指令对 | 早 p95 | 早 > τ | 晚 p95 | 晚 > τ | 没动的最长一段 p50 | p95 | > 2τ |
|---|---|---|---|---|---|---|---|---|---|
| v4 | 航向 | 10 874 | 4 s | 0.00 | 0 s | 0.00 | 8 s | 22 s | 0.08 |
| v4 | 高度 | 7 265 | 6 s | 0.01 | 0 s | 0.01 | 18 s | 26 s | 0.19 |
| v4 | 地速 | 15 806 | 14 s | 0.09 | 44 s | **0.17** | 18 s | 34 s | 0.38 |
| v6 | 地速 | 14 830 | 28 s | 0.15 | 0 s | 0.00 | 26 s | **98 s** | **0.64** |
| v8 | 地速 | 16 923 | 18 s | 0.12 | 26 s | 0.08 | 20 s | 38 s | 0.43 |
| v9 | 航向 | 10 501 | 4 s | 0.00 | 0 s | 0.00 | 8 s | 26 s | 0.12 |
| v9 | 高度 | 6 970 | 4 s | 0.00 | 0 s | 0.01 | 18 s | 28 s | 0.20 |
| v9 | 地速 | 16 389 | 16 s | 0.10 | 26 s | **0.08** | 20 s | 38 s | 0.44 |

读法：**航向和高度在每一版都是准的**（早/晚 p95 都不到一个句子位置），所有误差都在地速上。v6 的"没动的最长一段"
p95 98 s、64 % 的地速词超过 2τ——这正是第 4 轮读图人说的"橙线画在飞机停着的那段上"，退回去是对的。
（v8 / v9 的"早"和"没动的最长一段"比 v4 略高，是因为多出来的是**小台阶**词：一个 4 m/s 的台阶比这项量测自己的带宽
2 × 2.5 m/s 还窄，整段都会被算成"没动"——这一项对小台阶没有分辨力。）

**(b) 被最小变化量压掉的是什么**：v8 之前 7803 条地速机动被压掉；量了其中 7598 条各自守住的时长，p25 / p50 / p75 =
32 / 42 / 58 s，56 % 守住 ≥ 40 s，其中 **4032 条所在的格子和当时那条指令的格子不一样**——平均每架航班有 0.6 条真实的
速度台阶，飞机守了 40 秒以上，句子里却一个字没说。这才是"早发/晚发"的根，而不是发出时刻那条规则：中间少了一条词，
前后两条词的时刻怎么算都不对。所以 v8 把最小变化量的含义改成"只压抖动"：守住 ≥ 40 s 的一层自成一词。

实际多出来的词比 4032 少：地速词 22 659（v4）→ 23 776（v8，每架 +0.16 条）→ 23 242（v9，每架 +0.085 条）；被压掉的
地速小台阶 7803 → 2086（v8）→ 2609（v9，多出来的 523 条就是 v9 修掉的格子边界拆词）；晚发 17 % → 8 %。

**这一改的代价**（也量了）：地速在两个格子之间来回（甲 → 乙 → 甲 这样连着三条词）——

| 版本 | 来回的词 | 占地速词 | 涉及的航班 |
|---|---|---|---|
| v4 | 147 | 0.6 % | 139（2.0 %） |
| v8 | 301 | 1.3 % | 266（3.9 %） |
| v9 | 237 | 1.0 % | 213（3.1 %） |

地速本来就带风，守住 40 s 又回去的确会有；v9 把其中假的那三分之一去掉了。航向 43 条（0.2 %）、高度 7 条（0.05 %），
三版都没动。结论：用 3.1 % 的航班多一次来回，换回原本一个字没说的速度台阶，值。

### 12.4 定稿的读法（D53，v9）

- 目标 = 一个 level：信号（航向 6 s 平滑、高度与地速 10 s 平滑）在容差内（航向 ±4°、高度 ±30 m、地速 ±2.5 m/s，都不到
  半格）保持 ≥ 20 s（两个 τ 位置），且这 20 s 窗口上的拟合斜率把信号带走的量不超过容差的一半——也就是"还算稳住了"的
  最大漂移率：高度 0.75 m/s、航向 0.1°/s、地速 0.0625 m/s²（随 plateau_min_s 变长而收紧；缓慢下降 / 减速、下降途中的
  一个小停顿，都不算 level）。
- 两个 level 之间是机动，不管快慢：目标 = 到达的那个 level 的中位数；指令发出的时刻 = 信号离开**紧挨着它前面那个
  level**（离那个值超过容差的一半）的那一行——飞机守住过哪一层，指令就算在它离开那一层的时候发出的；settled = 新
  level 开始。
- 记录一开头就在机动中：t = 0 的词就是那个机动的目标；机动一直到记录末尾：目标 = 末尾值；末尾 20 s 内才离开最后一个
  level 的那一小段读不出来（记作 absorbed "short tail"）。
- 不成词的机动都记下来（`Reading.absorbed`，人工核的图上画灰带）：到达的 level 还是原来那个词（same word）、变化不到
  最小量且不满足下面这条（small change）、短的尾巴。最小变化量（航向 5°、高度 500 ft、地速 10 kt）压的是**抖动**——
  8 kt 的地速摆动是风、是格子边上的晃动；**飞机守住 ≥ 40 s（HOLD_MIN_S，四个句子位置）、并且和当前那条指令的目标差出
  至少一个容差的一层，不管变化多小都自成一词**，因为那是飞出来的一层。守住 40 s 这一条是 v8 的核心改动（理由见
  12.3）；"还要差出一个容差"是 v9 加的，用来修第 5 轮抓到的格子边界拆词（一段没动的速度，中位数正好压在两格边界上，
  被拆成两条指令；50 页里 6 页，其中一架 74.5 m/s 稳了 109 s 却画成"减到 77"再"减到 72"）。
- 切入 = 最后一条**换了词的**航向指令，它之前飞机不在航道上（离开的那个 level 上至少有一行不满足
  `approach_difficulty` 的 established 规则）、之后在航道上；角度 = 它前一条航向词的目标（记录一开头就在切入转弯里、
  第一行不在航道上的，用第一行的航向）。从第一行起就在航道上的航班没有切入词；在航道上做的航向修正不算切入；不到 5°
  的切入被当小变化压掉，也没有切入词——这些份额在统计里写明。
- 读法的版本号（`reading_rule = plateau-v9`）在词表 spec 里、进 sha：同样的档位、不同的读法，是两个词表。

v9 在 B cohort 上（train，sha 1586172a6ec6）：一共 59 316 条指令（航向 17 354 / 高度 13 823 / 地速 23 242 /
切入 4 897），每架 p50 7 条、p95 15 条；分开看每架 p50 航向 2（p95 5）、高度 2（4）、地速 3（7）、切入 1（1）；
274 230 个位置里 31 559 个（11.5 %）换了词；71.5 % 的航班有切入词；28.7 % 从记录第一行起就在航道上；
absorbed（同词 / 小变化 / 短尾）：航向 447 / 14 / 2、高度 76 / 8 / 0、地速 3693 / 2609 / 1415；clamp：高度 3、地速 21；
没 clamp 的目标离格子中心 p95：4.3° / 128 m / 2.4 m/s；36 / 11 / 23 / 3 个词全部用到。

### 12.5 切入角的分布与 D59 的门槛

v9 读出的切入角（前一条航向词的目标对航道的夹角）：≤ 30° 24.2 %、30–45° 20.6 %、**> 45° 55.2 %**（val 25.8 / 20.4 /
53.8 %；v4 是 30.6 / 20.5 / 48.9 %——v5 起在航道上的修正不再算切入，切入更多落在基边或下风边那一转上）。> 45° 那一格
是基边（±90°）或下风边直接转上航道的航班，KRDU 雷达引导的常见做法。计划 D59 写的"切入角 > 45° 在打分前排除"会把一半
以上真值航班的切入方式排除掉；门槛怎么定（按程序要求的 ≤ 30° 只放行四分之一，还是按数据放行 > 45°）是用户的决定，
B3′ 之前定。

### 12.6 判定 M-B0′

第 3 轮 0.967 过线（v4）；v6 改错了（第 4 轮 0.847），按 12.3 的量测定为 v8；第 5 轮新抽 150 页重核 **136 / 150 =
0.907**，过了计划的 0.9 门槛，同时抓出格子边界拆词，修为 v9（改动只减词、不增词，量测见 12.3）。
**词表定稿为 v9（sha 1586172a6ec6，`plateau-v9`）**。**四词的 v9 产物已按用户指示删除（2026-09-20）**，五词的还没建；随它一起没的还有 300 页 B0′ 人工核底图，那些在 v10 的读法下不可再生。本节是 B0′ 的记录，数留作对照。执行器与先验都按它训。v9 之后没有再做人工核——它相对 v8 只删掉
被误拆的 523 条地速词，第 5 轮那 6 页的毛病按定义不再出现。

## 13. 阶段 B′ · B0′′：换跑道与复飞的量测（2026-09-20；计划 §5.2.1，决定 D62 / D63 / D64）

为什么量：跑道成了第五个词（D62），于是两件事要有数才能定——跑道词能不能中途改（D64）、复飞词进不进词表（D63）。
harvest 的产物答不了：每条航迹只指派**一条**跑道，outcome 也没有复飞这一类，中断的那次进近还在 25 km 到达切片之外。
所以读的是**完整航迹**，按 manifest 点名，五个机场全部已落地航迹。代码 `run_ts.py approach_events`，产物
`two_tier_v3_bprime_20260920/approach_events/<机场>/`。

### 13.1 判据错了三次，前三轮读数全部作废

这一节留着，因为每一次都是判据的问题、不是数据的问题，而且每一次都是产物自己的某一列把它暴露出来的：

1. **判据太松**：用"横向最近的那条跑道"，于是五个机场里四个的切换点都在 0.11–0.18 km，那是跑道口交叉区；KSMF 报出
   26.4 %、切换点 20.1 km，是另一回事——飞机还在引导中、离每条中心线都远，"最近的那条"在平行跑道的中分线上来回跳。
2. **沿航迹的符号反了**：`final_approach.frame.RunwayFrame` 的 along 轴指向**着陆方向**，所以进近中的飞机
   `along_m < 0`，剩余距离是 `-along_m`。我写成 `along_m > 0` 当"在入口之前"，实际选到的是飞过入口之后的行，以及
   **同一条跑道反向那一端**（它的 along 是很大的正数）。第一、二轮的数因此全部无效。**已写成测试钉住。**
3. **没走读时的高度野点修复**：本仓库的规矩是修复在**读时**做、`tracks/` 永不改写，而我直接读了原始 store。一个
   3.9 km 处 9125 m 的野点把一次普通进近判成了复飞。五个机场读时共修复 561 个高度。**已写成测试钉住。**

定稿判据，用的都是仓库已有的门槛：

- **远处读"在往哪条跑道飞"**：距入口 3–30 km、入口以上 ≤ 3000 m，飞机必须**确实在某条中心线上**（横向 ≤ 500 m，即
  `approach_difficulty` 的 established 半宽）**并且航迹沿着那条航道**（±30°，同一条规则的第三个条件，第一轮漏了）；
  另一条跑道必须连续当选 ≥ 2 km。
- **最后进近读"实际对上了哪条公布航道"**：距入口 0–3 km，半宽取**公布的航道宽度** ±350 ft = 107 m
  （AIM 1-1-18 d 4：最后进近航道"总宽在跑道入口处通常是 700 英尺"），同样要求航迹沿着航道。
- **复飞**：距入口 ≤ 5 km 且入口以上 ≤ 300 m，随后爬升 ≥ 600 m，再落地。

### 13.2 读数（第四轮，定稿）

| 机场 | 已落地 | 远处换跑道 | 占比 | 其中平行 | 切换 p50 | 最后进近对上自己 | 对到别条 | 没稳定在任何航道 | 复飞 | 读时修复高度 |
|---|---|---|---|---|---|---|---|---|---|---|
| KRDU | 16 056 | 701 | 4.37 % | 692 | 17.72 km | 16 037 | 0 | 19 | 26 | 168 |
| KSJC | 11 157 | 1 033 | 9.26 % | 977 | 16.51 km | 11 153 | 0 | 4 | 5 | 182 |
| KSTL | 8 769 | 765 | 8.72 % | 537 | 15.30 km | 8 768 | **1** | 0 | 11 | 168 |
| KSMF | 4 490 | 43 | 0.96 % | 43 | 16.93 km | 4 490 | 0 | 0 | 0 | 29 |
| KMSY | 4 150 | 0 | 0 % | 0 | — | 4 144 | 0 | 6 | 0 | 14 |
| **合计** | **44 622** | **2 542** | **5.70 %** | **2 249** | | **44 592** | **1** | **29** | **42** | **561** |

**平行跑道间距，与公布的航道宽度对照**（runner 顺带算的）：

| 机场 | 平行间距 | 我们的 ±500 m 走廊 | 公布航道 ±107 m | AIM 5-4-19 的 side-step（≤ 1200 ft = 366 m） |
|---|---|---|---|---|
| KSJC | 213.3 m | 重叠，分不开 | 刚好相接 | **有资格** |
| KSTL | 392.0 m | 重叠，分不开 | 分得开 | 无 |
| KRDU | 1067.9 m | 分得开 | 分得开 | 无 |
| KSMF | 1826.4 m | 分得开 | 分得开 | 无 |

AIM 1-1-18 d 4 同一段还给出：LNAV / LNAV+VNAV 的横向完好性限是 0.3 NM = 556 m，**LPV 是 40 m**；FAF 前 2 NM 之前
是 ±1 NM 线性，之后转成类似 ILS 的角度收敛，靠近入口再转回线性、总宽收到 700 ft。KSJC 的 213.3 m 几乎正好是一个
航道总宽。

### 13.3 四条结论

1. **最后进近阶段，跑道是完全分得开的。** 44 622 架里 44 592 架对上了自己那条跑道的公布航道，**只有 1 架**对到了别条
   （KSTL），29 架没有稳定在任何航道上。这回答了"分不开怎么落地"：分不开的是**我们那个 500 m**，不是几何——公布的
   航道在入口处只有 ±107 m，LPV 的完好性限只有 40 m，比我们的判据窄 5 到 14 倍。
2. **晚改跑道（side-step）在这份数据里基本不存在**：1 / 44 622。唯一有资格做 side-step 的 KSJC 是 0 架。
3. **远处那 5.70 % 不能读成"管制员改了跑道"。** 它只说明飞机在引导过程中先沿着一条中心线走了 ≥ 2 km、后来落在另一条。
   ±500 m 的走廊在 KSJC（间距 213 m）和 KSTL（392 m）根本分不开平行跑道，所以那两个机场的 9.26 % 和 8.72 % 尤其不可
   引用。**这一栏只能当作重新指定次数的上界**，真值在 1（晚改）和 2 542（上界）之间，这份数据分辨不了。
4. **复飞是真的，42 架，0.094 %。** 逐架核过剖面：最低点都在距入口 0.2–0.6 km、高 23–48 m，随后爬到 1029–1188 m，
   签名无误（其中一架过入口 15 次，是训练飞行的连续起降）。0.094 % 与公开的复飞率（约千分之一到千分之三）同量级，
   而且是**下界**——改飞别的机场的航班根本不在"已落地"里。折算到 B cohort 的 train（6853 架）：约 **6 架**。

### 13.4 落到决定上

- **D62 跑道词：确认保留。** 每架航班都有一个跑道词，覆盖率 100 %，可学。
- **D64 跑道词可以改：机制保留，不设门槛，并写明这份数据分辨不了它。** FAA 原文写明允许改（§5.2.1），所以句子必须能
  表达；但晚改只有 1 / 44 622，而远处那一栏只是上界。**先验不可能学会它，也不能拿它当门槛。**
- **D63 复飞词：不进词表**（2026-09-20，§13.5 量完之后翻转的结论）。复飞在机队里是真的，但**在建模用的 cohort 里
  永远不可能触发**，见下节。一个永远不触发的词比没有这个词更糟（根 CLAUDE.md）。
- 五个机场合起来复飞也只有 42 架，**合并机场不解决稀少问题**。

### 13.5 复飞词为什么最终没有进词表

词表是在 B cohort 上读的，而 cohort 的序列是**25 km 到达切片**，不是完整航迹。把五个机场产物里记下的复飞航班与 cohort
名册求交，KRDU 有 8 架在 cohort 里；逐架读它们在 cohort 里的序列：

| 航班 | 切片时长 | 切片起点距离 | 最低点 | 最低点之后的爬升 |
|---|---|---|---|---|
| AAL2381_05L | 846 s | 23.6 km | 0.3 km / 25 m | **+0 m** |
| DAL2600_23R | 276 s | 23.1 km | 0.4 km / 25 m | **+0 m** |
| FFT2730_05L | 872 s | 23.7 km | 0.2 km / 20 m | **+0 m** |
| FFT3057_05R | 282 s | 23.2 km | 0.3 km / 23 m | **+0 m** |
| JBU8528_23R | 288 s | 23.2 km | 0.3 km / 20 m | **+0 m** |
| RPA4493_05L | 1186 s | −14.5 km | 0.9 km / 48 m | **+0 m** |
| RPA4596_05R | 1562 s | 22.7 km | 1.0 km / 54 m | **+0 m** |
| RPA4804_23R | 932 s | 23.1 km | 0.6 km / 28 m | **+0 m** |

八架全是 +0 m：最低点就是落地，之后什么都没有。**切片里只有最后那次成功的进近**，中断的那次和随后的盘旋都在切片之外
（切片按最后一次进入 25 km 窗口切）。所以复飞词在这个 cohort 上**一次也不会触发**，它进词表只会是一个恒为 0 的字段，
让先验多一个永远猜得对的头，读数变好看而没有内容。

**要让复飞可建模，得改上游的到达窗口切法**，那会改掉每一个 ts 数据集划分（根 CLAUDE.md 的 Open Items 里列为危险项），
是另一个量级的决定，不在 B0′′ 的范围里。**这一条作为 cohort 的一条限制写明：本阶段的句子表达不了"这次不落地"。**

## 14. 阶段 B′ · B0′′ 的产物：五词词表（2026-09-20）

`run_ts.py instruction_vocabulary` 在 B cohort 上重建，**sha `a5450c0fcd90`，读法 `plateau-v10`**，
产物 `two_tier_v3_bprime_20260920/vocabulary_tau10/`。四词的 v9 产物已按用户指示删除。

**四类几何词一个数都没动**（这是设计要求的：加跑道不改读法）：train 274 230 个位置、31 559 个换词，
absorbed 航向 447/14/2、高度 76/8/0、地速 3693/2609/1415，clamp 高度 3 / 地速 21，未 clamp 的目标离格子中心
p95 4.3° / 128.1 m / 2.4 m/s——与 §12.4 的 v9 逐项相同。

**跑道词读出 4 类，不是 6 类**：05L / 05R / 23L / 23R。KRDU 有 6 条跑道，但到达清单
`harvest-arrivals-v5-takeoff-excluded` 只含这 4 条（14 435 架：05L 3076 / 05R 1491 / 23L 3230 / 23R 6638），
14（13 次落地）与 32（1604 次）在上游就被排除。这不是标注器的问题，是清单的覆盖面。两条后果见计划 D69：
本阶段的句子**说不出 32**；而把清单重建成 v6 会加回 KRDU 32，那会改变跑道词的**类别集合**，于是
`runway_sha256` 变、此前训的每一个先验作废——D62 之前重建只动数据集划分，现在它动身份。

**五个机场的跑道词类别数，量自各自的到达清单**（不是机场的跑道表——两者只有 KSJC / KSTL / KMSY 一致）：

| 机场 | 到达清单的跑道 | 机场的跑道 | 差在哪 |
|---|---|---|---|
| KRDU | 4（05L / 05R / 23L / 23R） | 6 | 少 14、32 |
| KSJC | 4 | 4 | — |
| KSTL | 8 | 8 | — |
| KSMF | **3**（17L / 17R / 35L） | 4 | 少 35R |
| KMSY | 4 | 4 | — |

KSMF 少的那条 35R，正是根 CLAUDE.md 说 v6 重建会加回来的另一条（KRDU 32 + KSMF 35R，共 +1876 架）。

val 侧一致：4 类全部用到，clamp 高度 1 / 地速 1，几何词的分布与 train 同。

### 14.1 跑道词的核对：两半都过，而且是精确的

跑道词不适合看图核。它是每架一个常数，图上看不出东西；要核的是两件事，而且两件都能穷举，不用抽样。

**一、和名册一致（穷举 8257 架，train + val）**，读 `sentences_*.json` 的每一行：

| 检查 | 不合格 |
|---|---|
| 读出的跑道 ≠ 到达清单的跑道 | **0** |
| 跑道那一列不是常数 | **0** |
| 常数 ≠ 名册跑道在词表里的下标 | **0** |
| 名册里找不到这架 | **0** |

各跑道架次 05L 2206 / 05R 533 / 23L 951 / 23R 4567。

**二、和航迹几何一致（不看名册，穷举同样 8257 架）**：用 §13 那套公布航道读法（AIM 1-1-18 d 4 的 ±350 ft），
问"这架飞机在最后 3 km 实际对准了哪条跑道的公布航道"：

| 结果 | 架次 |
|---|---|
| 对上自己被命名的那条 | **8257** |
| 对到别条 | **0** |
| 没稳定在任何航道上 | **0** |

`run_ts.py approach_events --cohort <development_cohort.json>` 加了 `--cohort` 就是为这一步：词只为这些飞机读，
机队的比率说明不了它们。（同一次读数里复飞 14 架，那是**完整航迹**上的；cohort 的 25 km 切片里仍然一次都不触发，
与 §13.5 一致，不矛盾。）

**判定 M-B0′′：跑道词通过。** 两半都是 0 不合格、穷举而非抽样，比四类几何词那 136/150 的抽样口径更强。
**B0′′ 到此完成**：量测（§13）→ 只加跑道词（D62 / D63 / D64）→ 重做词表产物（§14）→ 跑道词核对（本节）。

## 15. 阶段 B′ · 词表重建为六类词的事件序列（2026-09-20）

`run_ts.py instruction_vocabulary` 在 B cohort 上重建，**sha `c7a4f4239f52`，读法 `plateau-v11`**，产物
`two_tier_v3_bprime_20260920/vocabulary_tau10/`（53 MB）。§14 那份五词的 `a5450c0fcd90` 已按规矩删除。

### 15.1 这一版改了三件事（D52 / D70 / D71 / D72 / D73）

**句子从等距网格改成事件序列**：一行一个"有东西变了"的时刻，不再是每 τ = 10 s 一行。网格不丢指令（同类两条
最近隔 18 s，10 s 的格装不下两条），丢的是**时刻**——它把每条指令往后挪 0–8 s、均值 4 s、**永远偏晚**，
按进场 270 kt 算平均多飞 557 m、最多 1112 m，3°/s 转弯平均多转 12°；单向偏置会在闭环里累积，还会被读成模型误差。

**加时长词**（2 秒一格，151 类，300 s 封顶）和**结束词**（继续 / 落地 / 复飞）。结束词取代了先验原来那个
单独的二值 `landed` 头——"这句话在哪结束"是一个问题，不该有两套机制。

**去掉切入词**：4897 条**100 % 与某条航向词同时刻发出**，在事件流里从不产生自己的事件，78 % 可从航向词推出；
按 D50 它本来也不是管制员说出口的目标。随之 `intercept_angle_bins_deg` 与 `established` 的两个数退出 spec
（它们不再决定任何一个词）。

### 15.2 读数

| | train | val |
|---|---|---|
| 航班 | 6853 | 1404 |
| **事件** | **40 270** | 8 158 |
| 每架事件数 p50 | **4** | 4 |
| 事件间隔 p50 / p95 | **44 s / 128 s** | 44 s / 128 s |
| 撞到 300 s 天花板的间隔 | **0** | 0 |
| 记录一开始就在航道上 | 0.29 | 0.30 |

**词的使用**：航向 36/36、高度 11/11、地速 23/23、跑道 4/4 全部用到；**时长 131/151**（最大用到第 143 格 =
286 s，正是实测最大间隔，所以 300 s 的天花板没有截断任何东西）；**结束 2/3**——继续 33 417、落地 6853
（= 航班数，每架恰好一个）、**复飞 0**，与 §13.5 一致。clamp：高度 3、地速 21（train）。

每架指令数 p50：航向 2、高度 2、地速 3、跑道 1；事件数 4。

### 15.3 与网格版的对照

| | 网格（v10，已删） | 事件序列（v11） |
|---|---|---|
| train 的 token 数 | 274 230 | **40 270** |
| 其中与上一行完全相同 | 235 818（88.2 %） | **0**（按定义） |
| 每条指令的时刻误差 | 后挪 0–8 s，均值 4 s，单向 | **0**（就是标注器读出的那一行） |
| 词类 | 5（含切入） | 6（去切入，加时长、结束） |

**一个读数缺陷，修在发布之前**：第一次六词产物印出 `duration 0/151, terminal 0/3`，像是两类新词从未使用。
原因是它们是 `sentence()` 算出来的**列**、不是 `Instruction` 对象，而统计去扫指令表。测试抓不到，是读产物
自己的 summary 时看出来的；若不看，这个数会写成"两个新词从未使用"进结论。已修（提交 53e9c7e6），产物按修好的
读数重出。

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_v3_stage_a_notes.md` — 现状的标题

## 现状（2026-09-20 晚更新；接手的 agent 先读这一节和 §7，再读计划 v3 的 §5.2 / §6.2 / §7.2 / §8.2 / §9.2）

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_v3_stage_a_notes.md` — 现状：2026-09-20 晚 / 白天两段

**2026-09-20 晚**：阶段 B′（指令词表版）的代码在分支 `dev-instruction-vocab`（worktree `.claude/worktrees/instruction-vocab`，基于主树 79f871f = 意图码层归档之后），提交：87fff1cb（标注器）、7e711f75（平台反读）、f043f90e（执行器吃指令 + 指令先验）、426aaee5（门 B1、队列、臂、intents）、f3eed83f / cb7b1712 / 8f60e73c / 9b6dc3a2（反读规则 v4 / v5 / v6 / v8）、608bb06c、46493288（文档）；全套 ts 测试 1263 通过；**未合并到主树，未训练任何臂**。主树上 79f871f 是意图码层的归档提交（agent 做的，我在主树上提交）。
- B0′ 反读与人工核：反读规则 v1 → v8、五轮人工核（结果文档 §12）：率阈值 0.903 → 平台 v2 0.793 → v4 **0.967**（过线）→ v6 **0.847**（我按第 3 轮的报告改发词时刻，改错了）→ v8（第 5 轮新抽 150 页，进行中）。
  第 4 轮之后不再靠看图调规则：先量（§12.5 两项量测，整个 train split）——航向与高度每一版都准（早/晚 p95 < 一个句子位置），误差全在地速；被最小变化量压掉的 7598 条地速台阶里 4032 条飞机守了 40 s 以上且档位不同（每架 0.6 条真实台阶句子里没说）。v8 因此把最小变化量改成"只压瞬态，守住 40 s 的一层自成一词"，并把地速下限降到 100 kt（105 kt 的最后进近曾被夹到 120 kt 那档）。
  档位定了（D51：73 词）。四词的 v9 产物（sha 1586172a6ec6，`plateau-v9`）已按用户指示删除；B0′′ 加跑道词后重建（计划 §5.2.1）。
  **人工核的数不能跨轮直接比**：三个 agent 对"早/晚多少算错"取的阈值不同（第 4 轮同一份代码判出 10 / 21 / 1 个误读）；第 5 轮的说明把阈值写成 20 s，三份才可比。
- 开发项：B′-dev1/2/3/6/7/8 完成（计划 §5.2.6 表有提交号），B′-dev4（执行器闭环再训）与 B′-dev5（目标导向解码）未做——两者都要先在计划里写清（dev4 的替换窗口要带真值目标，dev5 的掩码门槛 D59 待用户）。
- 等用户的：签阶段 B′ 的 §5.2 / §6.2（M-B1′′ 之前）；D59 的切入角掩码门槛（反读出 49 % 的切入 > 45°，结果文档 §12.4）；说"跑"之后才启动 `two_tier_bprime_queue`（基线两 seed 约 8 min，两臂训练约 30 min，真值指令读数 + 门 B1，过门再训先验）；分支要不要合并。
- 队列的启动清单在 §7.4；主树没有在跑的实验；GPU 空闲。

**2026-09-20 白天**：意图码版本的阶段 B 作废（读数留在结果文档 §9–§11）；计划 v3 的 §5.2 / §6.2 / §7.2 / §8.2 / §9.2 已按指令词表重写（提交 1fe0660）：词 = 绝对目标的管制指令，执行器吃生效指令，两层 CAT-K 式闭环再训（D49、D55），K 候选滚出按目标打分（D56），程序作解码掩码（D59），RL 本阶段不做（D58），多机层在阶段 C 之后（D60）；文献 `docs/literature/trajectory_as_language/`。§7 是指令词表版的开发记录；§7.5 是意图码版本（作废）的记录。

## `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_v3_stage_a_notes.md` — §7.1–§7.4（阶段 B′ 的开发记录）

## 7. 阶段 B′ 开发（指令词表版，2026-09-20；分支 `dev-instruction-vocab`）

### 7.1 开发项与状态

| # | 开发 | 状态 | 提交 |
|---|---|---|---|
| B′-dev1 | `manoeuvre/instructions.py`（词表 + 标注器）、`run_ts.py instruction_vocabulary`（产物、句子、统计、300 页人工核图，`--set 字段=值` 改阈值再读） | 完成；反读规则经三轮人工核改到 plateau-v4（结果文档 §12） | 87fff1cb、7e711f75、426aaee5、f3eed83f |
| B′-dev2 | `plan_conditioning = instruction`：`outputs/control/instruction_token.py`（Δ/τ 个位置的四目标按档中心）、config 字段 `instruction_vocabulary`、头里开一次词表并记 sha、checkpoint 存该 sha、加载核对；lockstep 协议 `truth-instruction`（整条记录上读，按执行器所在位置取真值行，每轮记真值时刻与距离；payload v5） | 完成（review 的 blocker：闭环曾在裁过的 cohort 上读指令——改为调用方在裁之前读 `InstructionFeed`） | f043f90e |
| B′-dev3 | `manoeuvre/instruction_sequences.py`、`instruction_prior.py`、`run_ts.py instruction_prior` | 完成（review 的 blocker：落地标签按序列长度——改为 `ends_at_landing`） | f043f90e |
| B′-dev4 | 执行器闭环再训（D49） | 未做；先写计划：`WindowContext.override` 的替换窗口只换历史、目标为零，要扩成带同一绝对时间真值下一段的目标；分支 = 真值指令 ± 相邻档 | — |
| B′-dev5 | 目标导向解码（D56）：lockstep 协议 `prior-decode` | 未做；先写计划；掩码门槛 D59 待用户 | — |
| B′-dev6 | 门 B1 = D57；相对门记候选词表 | 完成 | 426aaee5 |
| B′-dev7 | `two_tier_bprime_queue`（基线 → 训两 seed → 真值指令读数 → 失败方式 → 门 B1 → 先验） | 完成，真实臂文件 dry-run 通过 | 426aaee5 |
| B′-dev8 | `two_tier_v3_bprime_arms.json`（I20 × 2 seed）、intents `two_tier_v3_bprime_20260920` | 完成 | 426aaee5 |

### 7.2 代码事实

- 词表产物是带 sha 的文件（`instruction_vocabulary.json`：档位、容差、最小变化量、反读规则版本、established 规则的两个数、cohort 身份、每词计数）；执行器与先验的 checkpoint 记它的 sha；改任何一项都是另一个词表。`Vocabulary.words = {heading 36, altitude 11, speed 21, intercept 3}`；切入词 −1 = 还没切入。
- 执行器读词的一处来源：`Reading.words_at(t)`（句子里 t 之前最近位置的四个词）→ `Vocabulary.conditioning`（cos/sin 航向、高度与地速占上限的比例、切入分级标志）→ `instruction_token.instruction_context(reading, start_s, config, vocabulary)`；训练与 predict 传锚点时刻，闭环传离飞机最近的真值行的时刻（`nearest_truth_time_s`，整条记录的行都算）。
- lockstep 的协议与执行器要对上：无 token 执行器只能 `none`，指令执行器只能 `truth-instruction` 且要传 `feed`（`InstructionFeed.read(vocabulary, 未裁的 series)`）；`--first-prediction-row` / `--anchor-remaining-km` 裁 cohort 在读 feed 之后。
- `config.control_recipe` 带词表路径（两份词表是两次 run）；`predictability_report` 拒绝指令 checkpoint；`test_architecture` 允许 `outputs/` 引 manoeuvre 的叶模块（`segments`、`instructions`）。
- 归档的意图码 token 的五个 config 字段（`manoeuvre_tokenizer` 等）作为"没人读"的退役字段被 `from_dict` 丢弃（`PLAN_TOKEN_RETIRED_FIELDS`）——阶段 A 的 checkpoint 带着它们（`plan_conditioning = off` 下从未生效），归档后曾一度加载不了；有 canary 测试。这是我做的决定，用户可推翻。
- 先验：输入 = 四个词嵌入 + 状态 token（位置相对跑道入口）+ 位置；上下文 = 机型、跑道；输出 = 四个 softmax + 落地；读数的分母都写明（top-k 对有下一词的位置，翻转率对真值不换的位置，漏改率对真值换的位置，落地写基础率）；`_joint_ranks` 分块算联合 top-K。
- 队列：产物 `<campaign>/baseline/L60_D20_s<seed>/L-1/`、`lockstep/<臂>/truth-instruction/`、`failure_modes/`、`gate/b1_I20/`、`priors/<臂>/`；seed 线来自网格 `gate/after_L120_D120/grid_gate.json`；PID 文件 `two_tier_bprime_queue.pid`。
- 三轮人工核由 opus agent 看图（每轮 3 × 100 页），verdict 文件在会话 scratchpad（`hand_check_part*.csv`、`hand_check_r2_part*.csv`、`hand_check_r3_part*.csv`）；不是用户亲手核的数。

### 7.3 待用户定 / 待办

- 看 §5.2 的设计和 §6.2 的 D49–D60 并认可（在这之前不训任何模型，M-B1′′）；D59 的切入角门槛。分支已合并（dev-two-tier-feasibility）。
- B′-dev4 / dev5 先在计划里写清再建（替换窗口的目标；候选来源与打分项）。
- 第 3 轮人工核结果进结果文档 §12.2 / §12.5；≥ 0.9 则词表定稿。

### 7.4 阶段 B′ 队列的启动清单（用户说"跑"之后）

1. worktree `git status --short` 为空（或合并后主树为空）；`nvidia-smi` 没有别的训练；campaign 目录下没有 `two_tier_bprime_queue.pid`；空余 ≥ 3 GB。
2. dry-run（已通过）：`conda run -n aeroviz --no-capture-output python run_ts.py two_tier_bprime_queue --arms 4dTrajectory/ts_transformer/docs/experiments/two_tier_v3_bprime_arms.json --campaign 4dTrajectory/outputs/KRDU/experiments/two_tier_v3_bprime_20260920 --airport KRDU --dry-run` → `CELL baselines`（4 步）+ `CELL I20`（8 步，先验两步标 `[needs gate b1]`）。
3. 启动（从含代码的 worktree 或合并后的主树；`nohup setsid … > $C/two_tier_bprime_queue.log 2>&1 &`），Monitor 盯 `CELL .* complete|STOP:|queue done`，`skip .*gate b1` 表示门没过、先验没训。
4. 预计：基线两份约 3 min + 失败方式；两臂训练约 2 × 15 min；真值指令读数约 2 min / 臂；先验约 5 min / seed。

## `docs/code-health-followups.md` — the two failing tests' judgement

names the missing field outright. Neither blocks the B′ queue (nothing in the instruction path
writes an evaluation record), which is why they are recorded here rather than fixed inside the
vocabulary change. Whoever owns

## `docs/code-health-followups.md` — the instruction-prior entry (to the end of the file)

## The instruction prior re-derives its sentences instead of reading the artefact's (2026-09-21)

`experiments/instruction_prior.py:136` calls `read_instructions` on every rebuilt series, so the
sentences it trains on are re-derived rather than read from the `sentences_<split>.json` beside
the vocabulary it names. The replay gate deliberately does the opposite (`Reading.from_dict`,
added the same day) for a stated reason: a consumer that re-derives its own input cannot see the
artefact drift from the code that wrote it.

**Verified, and currently harmless**: the reading is deterministic given the series and the
vocabulary, the spec's sha is checked on load, and `reading_rule` is inside that sha — so a
vocabulary read under another rule is refused by name. What is NOT covered is the artefact's
sentence BYTES: a prior and a published sample could be trained and drawn from sentences that
were never compared to each other.

**Judgement, not a bug.** The prior needs the series anyway (the state tokens come from
`supervision_times/values`), so it cannot avoid rebuilding them; the change would be to take the
words from the file and keep only the states from the series, plus a check that every rostered
flight is in the file. Worth doing when the prior next changes, not on its own — it would
invalidate no checkpoint, since the words are identical today.

## `docs/open-items.md` — the arrival-roster decision, as it read with the runway word

- **到达清单不重建，两层 B′ 接受当前的跑道覆盖**（用户 2026-09-20）。
  `harvest-arrivals-v5-takeoff-excluded` 的跑道覆盖就是指令词表里跑道词的类别集合：KRDU 4 条（缺 14、32）、
  KSMF 3 条（缺 35R）、KSJC 4 / KSTL 8 / KMSY 4,并集 23 条。跑道词的类名带机场前缀(2026-09-21),而**类别集合是队列的、不是机场的**:并训用的 pooled 队列里没有一架 KSTL 06,所以它是 22 类。重建成 v6 会加回 KRDU 32 与 KSMF 35R（+1876 架），而自 D62
  起那不只是改数据集划分——它改变跑道词的**类别集合**，`runway_sha256` 变，此前训的每个先验作废。
  **用户决定不重建、不动训练数据**，所以本阶段一切关于「模型能说出去哪条跑道」的结论都带着这个范围，
  必须在结论里写明。

## `trajectory_data_process/docs/06-harvest-reference.md` — published minima, the altitude-word sentence

  200.0–423.4 ft** (highest: KSTL 12L; lowest: KSJC 30L/30R at 199.9 ft). That whole range sits
  inside altitude word 0 of the instruction vocabulary, which spans the threshold ±500 ft — so no
  decision altitude in this fleet can be told from the altitude WORD, and a go-around's timing has
  to be judged on the real height. That is the measurement behind the two-tier plan's D75.

