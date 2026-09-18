# 意图 token 方案：说明与开发计划（2026-09-18，晚间修订）

**一句话**：L1 直接自回归整条进近又慢又差，因为它同时背着几百秒的信用分配、雷达引导段的多模态和闭环误差累积。
分层把后两件拿走：L2 学一个**小的离散意图空间**并在其中做选择（先验），L1 只做"给定这一段的意图，飞出这一段"——短、单峰、好训。
段的意图码由一个分词器给出，它的解码器就是 L1 本身：码里只装"当前状态决定不了的那点决策"，不多不少。

本文取代 `2026-09-17_two_tier_plan_v2.zh.md`（v2）的设计与开发计划；v2 的测量仍可引用（§1）。**本文只放说明、计划、门与决定状态；
campaign 的读数进 `2026-09-18_manoeuvre_token_results.zh.md`（每个 campaign 一节，最新在前），状态表只写一行结论和指向。**
本文的上午版把分词器写成"航迹压缩 + k-means"，与分层的初衷不符，晚间按讨论重写了 §0、§2–§4 与 §6.1（用户 2026-09-18："L2 层要学习的应该是意图或者一个可用的解空间"）。

---

## 状态表（压缩 context 后从这里继续）

| 阶段 | 状态 | 产物 / 指向 |
|---|---|---|
| P0 文档 + 文献 + 仓库整理 | **完成 2026-09-18**：文献 `docs/literature/manoeuvre_tokens/`（13 篇，引用全部核实）；代码归档 §5.1 已执行并提交（**9dbb492**：ts 套件 1177 全过，退役字段在默认值时由 `from_dict` 丢弃、否则报名字拒绝，289 个存档 checkpoint 逐个量过，原来能加载的一个没变）；磁盘 §5.4 已执行；38 个 two_tier 类目已撤下、索引已重建、validator 165 类目 0 错 | 本文 §5；**等用户审核本文后才开始 P1** |
| P1 分词器 + 执行器（联合） | **P1.1–P1.3 完成 2026-09-18**（`manoeuvre/segments.py` 81ca7b1、`manoeuvre/tokenizer.py` 31b9c24、执行器接线 + review 修正 + 码本导出 runner 见下一提交）：ts 套件 1218 全过；opus review 的 2 BLOCKER / 3 MAJOR / 6 MINOR 全部修掉（航向 wrap、码本未记录尺度、越界码、常量重复、冻结模式、身份进 sha）。**P1.4 campaign `manoeuvre_tok_20260918` 已启动 2026-09-18 06:13**（c391265；5 段长 × 14 臂 = 70 臂，队列顺序 60→30→90→20→120 s，每臂约 11 min，链 `<campaign>/queue_p14.sh` PID 2671532；读数 `run_ts.py manoeuvre_readout` 每段训完由 opus 队列代理跑并回报；结果进 results 文件 P1.4 节）。**60 s 段早读 08:00（8/14 臂）：门 T 全 FAIL**——K32 / K128 / 指令词表两种子配对增益 p50 +2…+9 m（线 30 m；增益在雷达引导层、重尾，均值 +35…+45 m），最大码份额 47–79 %（指令词表自己 79 %：25 % 线在这个 cohort 上无法达到）；T(ii) 过（K32 s1337 先验 0.564 对 bigram 1.511 nats/码）；K16 / K64 / K256 训练中，其余段长照跑；事后改门的三个问题在 §6.3 第 14 条 | §4.1 P1.1–P1.4，门 T（§3.3）；§6.3 我替你选的 |
| P2 先验（单机，跑道已知）+ 离散对连续的对照 | **P2.1 代码完成 2026-09-18**（`manoeuvre/sequences.py`、`context.py`、`prior.py`，runner `run_ts.py manoeuvre_prior`：离散与连续同一骨架一个开关，bigram 基线随 metadata 写出；在冒烟 codebook 上端到端跑通）。P2.2 读数与 P2.3 campaign 等 P1.4 的码本（过门 T 的 K）。先验用执行器自己的航班划分（§6.3 第 10 条），运行日划分留给 P4。**诊断链 `<campaign>/p23_S60_K32/` 2026-09-18 08:08 启动**（未过门 T 的 K32，两种子：码本导出 → 离散 / 连续先验 → 先验读数 → atlas → lockstep C / A-truth / A / A-连续 → 门 E / P / X；§6.3 第 13 条）；**完成 08:31**：T(ii) 两种子 PASS（0.564 / 0.526 对 bigram 1.511 / 1.404 nats/码，top-1 0.79 / 0.80，翻转率 0.09 / 0.07）；**门 P 开环 PASS**（雷达 120 / 180 s 位移比 B61 好 110–160 m）；**门 P 离散对连续 FAIL**（连续先验的雷达 ADE 好 272 / 141 m，建立平手）——在没过门 T 的 K 上读的，不在这里定（§6.3 第 15 条） | 门 P |
| P3 lockstep：真值码、端到端、闭环再训 | **P3.1 代码完成 2026-09-18**（`manoeuvre/lockstep.py` 协议 C / A / A-truth，一轮 = 段长，飞出段经码本回灌，e_track 与 e_plan 逐问；`manoeuvre/gates.py` 门 X / P / E / S 拒绝单种子；runner `run_ts.py manoeuvre_lockstep`，在冒烟产物上 C 与 A 端到端跑通）。两轮 opus review 的 3 BLOCKER / 4 MAJOR 已修（changelog 2026-09-18）。P3.3 的**先验侧**已写：`manoeuvre_lockstep --protocol C --split train` 把飞出段经码本回灌并写 `flown_codes` / `flown_states`，`manoeuvre_prior --rolled <该 json> --rolled-share 0.75` 以真值码为时间索引标签混训（`sequences.rolled_sequence`）；**执行器侧**（在飞出腿的窗口上再训）未写——需要 `WindowContext.override` 返回时间索引的真值目标，等门 E 第一读决定要不要做。门 X 的规则制导基线取 §3.1 的 step3d 数（0.956）。**P3.2 第一读（`p23_S60_K32`，08:31）：门 X FAIL**（可飞 0.999 过；建立 0.634 / 0.619 对线 0.860；雷达 ADE 1742 / 1910 对 1500）、**门 E FAIL**（雷达 ADE 2531 / 2637 过进展线，建立 0.422 / 0.481 比 v2 低 17–22 点：先验落地头在飞出历史上早 ~30 s + "落地即停"）；不是过门 S 的候选。**P3.3 先验侧 `p33_S60_K32` 完成 08:49：门 E 仍 FAIL**（回灌先验：建立 0.462 / 0.476 对平的 0.422 / 0.481，雷达 ADE 差 66 / 104 m；落地提前量减半但停点更远）——§6.1 的 DAgger 先例重演，标签与落地时刻查过不是原因，瓶颈是执行器在码下的穿越（协议 C 也只建立 0.63）；执行器侧不写 | 门 X、S、E |
| P4 跑道 token + 程序上下文 | 未开始 | 门 R |
| P5 多机图层 | 未开始 | 门 G（先做 oracle 上限） |
| P6 发布 + 索引 | 未开始 | — |
| 分支 | `dev-two-tier-feasibility`（用户：直接在主树提交，不另开 worktree）；runs worktree `.claude/worktrees/two-tier-runs`（detached，训练前移到已提交的 commit） | — |
| 磁盘 | 19 GB 空余（2026-09-18 删除后） | — |
| 决定 | §6.2 六项已定（2026-09-18）：归档 + checkpoint 归档、删除、撤下、段长消融、审核本文后再开始 P1、**分词器以 L1 为解码器、码只装决策** | — |

---

## 目录

0. [一页摘要](#0-一页摘要)
1. [为什么换方案：证据](#1-为什么换方案证据)
2. [方案说明](#2-方案说明)
3. [评估协议与门（预注册）](#3-评估协议与门预注册)
4. [开发计划](#4-开发计划)
5. [仓库整理](#5-仓库整理)
6. [风险与待决](#6-风险与待决)
7. [引用与索引](#7-引用与索引)

---

## 0. 一页摘要

**分层的初衷。** 现有控制路径让一个网络一次吐出整条进近的控制序列（native32：32 段控制 + 时长，池化 ADE 1322 m，一个臂训数小时），
两层 v2 让 L1 每 60 s 再问一次（闭环里雷达引导 ADE 3044 m，建立 0.64，v2 §12.1）。两者失败的原因相同：L1 同时背着三件事——
几百秒的信用分配、雷达引导段"左基线还是右基线"的多模态、闭环里自己误差的累积。分层的目的是把后两件从 L1 拿走，
L1 只剩一个短、单峰、可监督的问题：**给定这一段的意图，飞出这一段。** v2 证明了 L1 能做好这件事（真值 token 下 ADE[0,60] 81–90 m），
也证明了它没被真正解耦（接口是航路点，两层各自对着真值训，闭环从未见过）。

**方案。** 三个部件，一个接口：

1. **意图空间**（§2.4）：一个分词器把每一段（`segment_s`，20–120 s 五个候选，消融定）映到 K 个码之一。编码器看真值段和当前状态，
   **解码器就是 L1 加动力学 rollout**：码好不好，由"L1 拿着这个码能不能飞出这一段"来定义。码因此只装当前状态决定不了的那部分——
   转不转、往哪转、减不减速、下不下——K 小（几十到一两百）是刻意的：K 就是"意图级"与"航迹级"之间的旋钮。
2. **先验**（§2.5）：码序列上的因果 Transformer，输入 = 上下文 token（候选跑道、程序 fix、机型）+ 每段 [码 + 状态]，
   输出 = 下一码的类别分布、落地终止符、跑道 token。雷达引导段上分布该平就平，top-1 落在一个模态上，不回归到中间。
3. **执行器 L1**（§2.6）：现有控制路径（`specific-force+path-angle`、`first-order-lag`、`control_horizon_s = segment_s`），条件是码的量化向量 z。
   与分词器联合训练，一段一段地训。

接口只传两样东西：码 id（先验之间、多机之间）和 z（先验 → L1）。飞出的段经分词器回到先验，闭环闭在码空间里。

**它不是什么。** 骨架是 PatchTST：每段一个 token，token 上做注意力。加的只有三样——离散瓶颈（类别分布代替回归）、自回归回灌、码作公共接口。
离散瓶颈是核心假设，门 P 里有它的对照：同骨架、去掉量化器、先验回归下一段，离散版必须赢过它，否则分词是白做（§3.3）。

**训练效率**（§2.9）：分词器与 L1 联合，单段、短 rollout，一个臂十几分钟；先验 teacher forcing，分钟级；闭环再训各一轮。整套比一次整段自回归的 campaign 便宜，
且每一层的失败单独可见。

**门**（§3.3）：T（分词器：L1 的增益与先验的可预测性，**不是重建误差**）、P（先验，含离散对连续）、S（段长）、X（执行器在真值码下）、E（端到端）、R（跑道）、G（多机）。

## 1. 为什么换方案：证据

### 1.1 v2 两层的脱节（数字来自 v2 §10–§12，KRDU val，固定锚点 59/60）

| 读数 | 数字 | 含义 |
|---|---|---|
| L1 带真值 token 的 ADE[0,60]（L1c，三契约两种子） | 81–90 m | 60 s 内跟得住真值计划 |
| L1 在真值航路点下的 lockstep，雷达引导建立（pa） | 0.62 / 0.68（门 0.774） | 即使计划完美，转弯段也建立不了 |
| L2 B61 开环位移 p50，雷达 60 / 120 / 180 / 300 s | 448 / 768 / 1396 / 3449 m | 比 native32（625 / 1906 / 2474 / 3666）和 state 臂（663 / 1071 / 1692 / 3455）都好 |
| L2 B91（回看 182 s）对 B61，雷达 60–120 s / 180 s 以后 | 好三成 / 几乎不动 | 更长的本机历史没有雷达引导的未来 |
| 端到端 pa + B61，雷达 ADE / 建立（两种子） | 3044 / 3074 m，0.640 / 0.649 | 门 2745 / 0.94，FAIL |
| 端到端 pa + B91，闭环 e_plan 逐问（雷达） | k0 372 → k5 4378 m（B61：284 → 1926） | 开环更好的头在闭环里更快变差 |

结构性原因（v2 §2 的设计决定）：接口是解码后的航路点，两层各自在真值上训练，周期不一致（L2 段 30 s，L1 时域 60 s 执行 30 s），
队列不一致（L1 openap 机型 6848 / 1401，L2 全机型 10102 / 2104），锚点差一个采样。v2 这么选是为了 S1、S2 并行开发、门 L1 / L2 各判各的；
代价是闭环耦合没有任何一层见过。

### 1.2 plan 路径留下的五条经验（`2026-09-09_plan_and_guidance_design.md` §12.5–§12.8）

plan 路径是"网络给下一个 fix，规则制导飞"的两层。它和本方案是同一族（高层给下一步、低层带动力学执行、周期性再问），
它的测量直接预告本方案会遇到什么：

1. **rolled windows 不是可选项。** 只在观测窗口上训的头每 30 s 再问时雷达引导 ADE 5391 m；用真值策略的 rolled windows（share 0.75）训后 3641 m，脱轨从 29 % 降到 7–12 %（§12.6）。
2. **DAgger 第一轮没帮上**（+73 m 配对）：把头自己的状态配真值标签追加进去，头更常报出多余的 fix。码的版本要先把"该停"写进标签（§12.6 第 (3) 条）。
3. **稳定性只能来自模型本身。** 头给的 fix 相邻两问中位数走 766 m；加滞回让雷达引导建立从 86.5 % 掉到 60.2 %（§12.7）。本方案的对应量是相邻询问的码翻转率，第一天就量。
4. **混合头 top-1 赢点头 550–600 m，fan 不比盲环好**（§12.8）。离散化多模态有效；候选码作备选集要过盲环对照才算信息。
5. **规则制导仍是更强的执行器。** 真值指令下雷达引导建立 0.879、直线 0.998（v2 门 L1 的基线 `plan_guidance_20260910/step3d_lockstep_l1`）。学习的执行器第一个要还的债是转弯段的建立。

### 1.3 信息边界

部署时的输入只有三样（§2.2）：本机最近的观测历史、跑道（现在是真值）、机型。雷达引导段的未来不在这三样里。
本方案不假装用更长的本机历史猜出它：先验在这些段上输出高熵分布，跑道 token 把"跑道已知"从假设变成预测（§2.5），
多机图层把邻机拿进来（§2.8）。哪一项值得做，由 §3.3 的 oracle 门决定，不由直觉决定。

---

## 2. 方案说明

### 2.1 总览与数据流

```
训练（一段一段）：
   真值段的行 + 当前状态 ──► 编码器 ──► 量化（FSQ）──► 码 c、向量 z ──► L1（60 s 历史 + z）──► rollout 飞出这一段
                                                                                   │
                                                              损失 = 飞出的段 对 真值段 ◄──┘   （解码器就是 L1）

部署 / 端到端，每 `segment_s` 一轮：
   上下文 token（候选跑道 · 程序 fix · 机型）
            │
   历史 ──► 分词器（编码器）──► 历史码序列 + 状态 ──► 先验 ──► 下一码 p(c) · 落地位 · 跑道 token
                                    ▲                              │ top-1 → 查码本得 z
                                    │                              ▼
                             飞出段的码 ◄── 编码器 ◄── L1（60 s 历史 + z → 一段控制 → rollout）
                                                                    │
                                                       飞出的段接到历史，下一轮
```

一轮 = `segment_s`。先验每轮读到的历史是真实飞出的（部署）或真值的（训练、开环读数）。执行器每轮只吃一个 z。
多机时（§2.8）先验的一轮是所有在场飞机各出一个码，按到达顺序串行提交。

### 2.2 部署输入契约

| 输入 | 内容 | 谁用 | 来源 |
|---|---|---|---|
| 本机观测历史 | 位置 (lon, lat, HAE) + 时间戳，重采样到 2 s，速度取 chart 导数（`data/channels.py` C1）；执行器取最近 30 行（60 s），先验取整数段（首问最少 1 段） | 两层 | 实时 ADS-B；须复现 harvest 的重采样、高度离群修复与 HAE→MSL 转换（`flight_scenarios/CLAUDE.md`） |
| 机场 | 各跑道端的入口位置与航向、程序 fix 集合（`aeroviz-4d/public/data/airports/<ICAO>/procedure-details`、`runway.geojson`） | 先验的上下文 token | 静态数据 |
| 跑道 | P2–P3：已知（与现有每个 ts 数字同一假设，T1）；**P4 起：先验预测**，仅在 ATC 已分配时作为上下文覆盖 | 执行器的 chart 原点、先验 | ATC / 先验 |
| 机型 | 质量、最大推力、翼面积、极曲线（`outputs/conditioning.py`，8 维） | 执行器（rollout 需要）；先验只用 typecode 嵌入 | icao24 → 机型库 → openap |

没有的输入：风、ATC 指令文本、天气、跑道运行配置（只经"跑道"间接进入）。首问时刻：进入 25 km 切片后 max(60 s, `segment_s`)；
与参照比较时用锚点 60，见 §3.1。

### 2.3 段

- **段长 `segment_s` 是消融轴**（用户 2026-09-18："定死 60 s 没有说服力"），不是常量。整条航迹从锚点向前按 `segment_s` 切（`dt_s = 2`），
  最后一段不足时由先验的落地位处理（§2.5）。执行器的时域 = `segment_s`，控制段数 N₁ = `segment_s / 10`。五个候选与依据：

| `segment_s` | 行数 | 依据 |
|---|---|---|
| 20 s | 11 | 驾驶类 token 模型的 token / 时域比（MotionLM 0.5 s 对 8 s [1]，Trajeglish 0.1 s 对约 9 s [2]）按 300 s 剩余航程折算是 4–20 s；20 s 是仍能装下一次 60° 标准率转弯的最短段（reading） |
| 30 s | 16 | 一次 90° 标准率转弯（3°/s）；v2 的 L2 段长与 plan 路径的询问周期，已有可比读数（v2 §10.6，plan §12.5） |
| 60 s | 31 | v2 的 L1 时域；进近速度下约 4–5 km，一个三边转四边加一段直线（reading） |
| 90 s | 46 | 一次 180° 反向加进出转弯；L2c 每段历史的量级（reading） |
| 120 s | 61 | KRDU 一条完整四边的量级（reading）；每航班只剩 2–3 个 token，检验先验会不会退化成时长头 |

  选择规则预注册在 §3.3 门 S；执行器的历史长度对所有候选固定 60 s（v2 §10.1：30 s 历史已够，60 与 118 s 不再多给）。
- **段起点系**：原点 = 段第一行的位置，x 轴 = 第一行的地面航迹方向，y 轴左，z 轴上。同一机动在任何跑道、任何机场读出同样的行；
  跑道航向不进段（v2 的 L2 特征按**落地**跑道旋转，把 T1 的未来信息写进了特征）。
- **段的输入行**：段内每 2 s 一行的六通道（位置三、速度三），起点系；行序号作位置编码。**没有手工描述子**——分词器的输入就是这些行，
  表示由编码器学；时间隐含在固定行距里，速率是相邻行的位移和速度通道。抽样、正交基、残差之类的预处理都不做（上午版讨论过，结论：
  它们只影响聚类眼里"谁和谁像"，对学习式分词器没有意义；编码器看得到当前状态，"减掉照起点速度直飞"这一项也就不需要了）。

### 2.4 意图空间：分词器

**定义。** 分词器 = 编码器 + 量化器，解码器是 L1。

- **编码器** E(段的行, 当前状态) → 连续向量 → **FSQ** 量化 [5]（每维几个档位，乘起来是 K；无码本、无承诺损失、不塌缩）→ 码 c ∈ {1…K} 与其向量 z（FSQ 的档位坐标，几维）。
  编码器看的是**真值段**（它在训练与真值码 lockstep 里读未来，这是分词器的本分，不是预测）和当前状态（L1 看到的那 60 s 历史的末行状态，或历史本身）。
  看到状态是关键：状态决定得了的东西（速度、航向的延续）编码器不必编码，码里只剩决策。
- **解码器 = L1 + rollout**：L1（60 s 历史 + z）飞出这一段，损失 = 飞出的段对真值段（现有控制路径的 [0, Δ] 位置项与速度项）。
  梯度经 FSQ 的直通估计回到编码器。**码的定义因此是"L1 需要被告知的那点信息"**，不多不少；一个只装噪声的码对 L1 没用，损失不会奖励它。
- **K 是旋钮**：意图级（几十）到航迹级（上千）。默认候选 K ∈ {16, 32, 64, 128, 256}，按门 T 选，取增益相近时最小的（越小越像意图）。
- **码本产物**：`4dTrajectory/outputs/codebooks/<name>/`：编码器权重、FSQ 档位、K、`segment_s`、拟合队列的身份（C26）、`sha256`。
  训完冻结；**每个先验与执行器 checkpoint 写入 `codebook_sha256`，加载时不匹配即拒绝**（与 `DURATION_QUANTILES` 绑定 conformal 表同一规矩，C12）。
  码在冻结后是稳定身份：任何一段（真值的、飞出的）都能算出它的码和 z，先验的输入、L1 的条件、多机的接口用的是同一套。
- **码怎么"看"**：码没有独立的解码图。看一个码，就是从若干典型状态出发让 L1 按它飞一段，画出来；量两个码的差，就是从同一状态飞出的两段的距离。
  多机解码时判断某个码会不会违反间隔，也是这样飞一段（K 次单段 rollout，批量，便宜）。

**为什么不是重建式分词器（VQ-VAE / k-means）。** 重建目标让码去编码段里**一切**会变的东西，包括状态决定的部分和噪声；
词表越大越忠实，但越忠实越不是意图。分层要的恰恰是相反的东西。重建误差不作门，最多作健康检查。

**意图空间的第二种来源：手定的指令词表（基线 B）。** 从真值段按规则读出三项——航向变化档（≤±15°、±45°、±90°、反向）、目标高度变化档、目标速度变化档——
交叉即词表（实现为 7 航向档 × 3 × 3 = 63，见 §6.3 第 6 条；上午版写的 45 是把左右并入一档的估算），像 ATC 的口令。同样的接口喂 L1（one-hot 或嵌入代替 z），同样的门 T 判。它可解释、与管制语言对齐；
学出来的码是数据驱动的。两者都做，哪个给 L1 的增益大、对先验更可预测，哪个上。

### 2.5 先验：码序列上的因果 Transformer

**序列**（一架飞机）：

```
[RWY_1 … RWY_r] [FIX_1 … FIX_f] [TYPE]  [c_1, x_1] [c_2, x_2] … [c_t, x_t]  →  p(c_{t+1}) · p(landed) · p(runway)
   候选跑道        程序 fix       机型        历史码 + 该段末的状态 token
```

- **token 嵌入** = 码嵌入(c)（先验自己学的表，K 行）+ 段序号位置 + 状态 token 嵌入(x)。x 是段末的状态（机场中心 ENU 位置、高度、地速、航向；
  `data/coordinate_frames.py` 的 airport-enu 定义），让先验知道"在哪"。
- **上下文 token**：每个候选跑道端一个（入口位置、航向）；每个程序 fix 一个（位置、类型：IAF / IF / FAF / STAR fix，来自 `outputs/guidance/skeleton.py` 的程序读取器，P4 接入）；机型一个。
- **输出头**：(i) 下一码 softmax，K + 1 类（含 `LANDED` 终止符）；(ii) 落地分数（最后一段内的到达时刻，仅当下一码是 `LANDED`）；
  (iii) 跑道 token softmax，每步都输出，标签是落地跑道（P4 起训练，P2–P3 时跑道作为已知上下文）。
- **损失**：交叉熵（下一码）+ BCE / L1（落地位与分数）+ 交叉熵（跑道）。teacher forcing，因果掩码，一次前向监督每个位置。
- **解码**：top-1 是点预测；采样只用于 fan 读数，且必须过盲环对照（§1.2 第 4 条）才能称为信息。
- **稳定性读数**：相邻两问（真值历史，一段之隔）对同一未来段的 top-1 码翻转率，以及两个码经 L1 飞出的段的位移；这是 plan 路径"fix 走 766 m"的对应量，第一次读数就报。
- **离散对连续的对照（门 P 的一部分）**：同一骨架、同一序列、同一回灌，去掉量化器——先验回归下一段的连续编码向量（编码器量化前的输出），
  L1 吃这个向量。这就是带回灌的 PatchTST。离散版在闭环里必须赢过它，否则分词这一步没有价值。
- **模型大小**：序列很短（KRDU 切片时长 p50 约 300 s，20–120 s 一段是 3–15 个 token），d_model 128、4 层、4 头量级；一个种子分钟级训练。
  不复用 iTransformer / PatchTST 主干：它们是通道序列模型，这里是 token 序列模型。

### 2.6 执行器 L1

- **模型**：现有控制路径，`plan_conditioning` 的新取值 `manoeuvre-code`：token = 码的向量 z（FSQ 几维，同一尺度），
  `plan_conditioning_dropout` 强制 0（v2 §10.8：遮蔽率 0.5 让头学会不看 token），`control_horizon_s = segment_s`，`seq_len = 30`（60 s 历史，所有候选段长相同），
  契约 `specific-force+path-angle` + `first-order-lag`，平滑项（航向率、坡度 TV）为 0——即 v2 §12.1 L1c pa 的配方，三契约里唯一关平滑项后仍可飞的（0.994 / 0.997）。
- **训练**：与编码器联合（§2.4），随机锚点 remaining-path-uniform，每个样本 = 一段真值 + 它之前的 60 s 历史；损失只算 [0, `segment_s`]。
  联合训练结束后码本冻结；之后 L1 单独再训（§2.7 第四步）时 z 取真值段经冻结编码器的输出。
- **执行**：一轮飞满 `segment_s`（v2 的 one-shot 变体），不是执行前一半。飞出的段接到历史，经编码器得到实际执行的码，回到先验。
- **规则制导**（`outputs/guidance/`，保留）在本方案里只做两件事：协议 C 里沿真值段飞，给出"执行器的上限"（对应 plan 路径 §12.6 的 2184 m）；
  端到端里当 L1 的 rollout 不可飞时接管这一段（后备）。它不再是与 L1 平行的"第二执行器"：码没有独立于 L1 的解码，规则制导没有可跟的航路。

### 2.7 闭环：lockstep 协议与训练分布

**lockstep**（`manoeuvre/lockstep.py`，一轮 = `segment_s`）：

| 协议 | 先验读的历史 | 执行器吃的 z | 用途 |
|---|---|---|---|
| C（真值码） | — | 真值段经冻结编码器的 z | 门 X：执行器本身；上限（规则制导沿真值段） |
| A（端到端） | 飞出的历史（分词后） | 先验 top-1 的 z | 门 E |
| A-truth（诊断） | 真值历史（时间索引） | 先验 top-1 的 z | 隔离闭环耦合：v2 §12.3 想做而没做的那个变体 |

结束规则：先验输出 `LANDED` 或到达时限；时限 = 首问时先验自己的到达时间 + 余量（沿用 `plan_oracle.closing_horizon_s` 的规则，函数随 archive 复制到 `manoeuvre/lockstep.py`），
没画出到达时取计划跨度上限。建立、可飞、ADE 的读法沿用 v2 §5（`judge_g3` 的判据）。

**第四步：闭环训练**（两轮 DAgger 式，各只做一轮；第二轮只在第一轮移动了门 E 时做）：

1. 先验吃飞出段：用协议 C 在训练集上跑 lockstep（执行器飞真值码），飞出的历史经冻结编码器成码序列，标签是**同一绝对时间**的真值下一码（时间索引，v2 §8 的决定）；
   与真值序列按 share 0.75 混合训练（plan 路径 §12.6 的值）。"该停"在标签里：飞出历史对应的真值段已落地则标签是 `LANDED`。
2. 执行器在自己飞出的历史上再训：飞出的腿切成窗口，z 仍是真值下一段的（时间索引），码本不动。

联合反传（执行器 → rollout → 编码器 → 先验）不做：先验的采样不可微，而且 v2 的教训是先把读数分得开。

### 2.8 多机图层 L3

**数据现实**（KRDU 到达清单，82 天，14,435 次到达，2026-09-18 量）：

| 量 | 值 |
|---|---|
| 与本机 25 km 切片时间重叠的其他到达 | p50 3，p90 5，最多 14 |
| 至少一架邻机 / 至少两架 | 0.93 / 0.79 |
| 进入切片时同跑道前机仍在空中 | 0.63 |
| 同跑道相邻落地间隔 p50 / 小于 180 s 的占比 | 277 s / 0.33 |

场景是几个节点的小图；六成航班的前机在切片里。`tracks/manifest.json` 另有 50,864 条 not_landing 轨迹（离场、过境）可作邻机。

**设计**：

- **场景**：同一机场、切片时间重叠的航班集合；每一轮的节点 = 各机当前码 + 状态 token。切分**按运行日**（`data/runway_context.operational_day`，T4）：
  同一天的邻机会出现在多条样本里，按航班切分会把邻机的未来漏进训练集。
- **边特征**（有向，前机 → 后机）：本机段起点系里的相对位置、相对高度、相对地速与航向、同跑道 / 平行关系（`runway_schedule.parallel_relations`）、
  沿同一五边的到达时钟差、尾流配对类别（`runway_schedule.wake_category`）、间隔余量（当前间隔 − `faa_separation` 的最小值）。
- **模型**：不另起 GNN 栈；边特征做成跨机注意力的偏置与值调制（Graphormer [7] 的边编码），加在同一先验的同一步跨机注意力上。
  一个模型，两种注意力（时间上因果、机间带边）。
- **解码顺序**：同一轮内按同跑道的预计到达顺序串行提交，前机先，后机的注意力与掩码读前机已提交的码（Trajeglish [2] 的同步串行；MotionLM [1] 是并行，
  [2] 自己测得上下文够长时同步交互作用弱——串行是为了掩码有定义，不是为了精度）。
- **硬约束 = 词表掩码**：后机的某个码，从后机当前状态经 L1 飞出的段若对前机已提交的段违反间隔规则，该码在 softmax 前被遮掉（约束解码 [9]）。
  规则只有一处定义：`inference/runway_schedule.py`（引用见 `docs/literature/arrival_separation/`）。掩码带余量 = 量出的 e_track p90。
  **掩码约束的是计划，不是飞出的航迹**；飞出的实际间隔由门 G 读。
- **先量上限**（v2 §7 S6 的前置测量，本方案里最便宜）：不建图，把前机的**真值**码序列作为上下文 token 喂给后机的单机先验，
  读雷达引导层 120 / 180 s 位移的变化。这个数是 L3 的上限和 go / no-go。

### 2.9 训练效率的账

| 训什么 | 视野 | 目标 | 一个臂的量级 | 对照：整段自回归 |
|---|---|---|---|---|
| 分词器 + L1（联合） | 一段（20–120 s）+ 60 s 历史 | 单峰：飞出这一段 | 十几分钟（L1c：6 臂 77 min；编码器很小） | native32 一个臂数小时，池化 ADE 1322 m |
| 先验 | 整条航班的码序列（3–15 个 token） | 类别分布，teacher forcing | 分钟级 | — |
| 闭环再训 | 同上 | 同上 | 各一轮，约 1 GPU h | v2 的 E2E 无闭环训练 |

慢和差的根源（几百秒的信用分配、多模态回归到中间、闭环累积）分别落到：短视野、类别分布、码空间里的回灌。整套 P1–P5 约 5–6 GPU h（含段长消融），
主要成本是代码与 review。

### 2.10 不在范围内

单机确定性点预测之外的 fan 交付（先过盲环）；风；ATC 指令文本；联合反传；跑道以外的机场运行配置；离场轨迹的预测（只作邻机）；重建式的独立解码器。

---

## 3. 评估协议与门（预注册）

### 3.1 参照与基线

| 参照 | 在哪 | 用于 |
|---|---|---|
| native32（控制路径整段头） | `outputs/KRDU/experiments/l1_lowdim_20260907`（checkpoint 可加载，9dbb492 核过） | 门 P、门 E 的整段参照 |
| state 臂 `A_threshold_enu` | `outputs/KRDU/experiments/airport_frame_20260903`（config 缺 `channels` 等必填字段，加载已失败于本次之前；只引用其 v2 §10.6 的数字） | 门 P 的数字参照 |
| B61 / B91 段计划头 | **代码归档、checkpoint 归档**（§5）；数字引自 v2 §10.6、§12.2 | 门 P 的数字参照（不再重跑） |
| L1 无 token 臂 | v2 §10.1 的 `L1_*` no-plan 列（ADE[0,60] 134–136）；P1 重训一个同配方无 token 臂作同口径对照 | 门 T 的"L1 增益" |
| 规则制导 lockstep（真值指令） | `plan_guidance_20260910/step3d_lockstep_l1/plan_oracle.json`（json 保留；2026-09-18 读出：建立 全 0.956 / 直线 0.998 / 雷达 0.879，完全可飞 0.999，1404 val 航班） | 门 X 的建立基线（`manoeuvre_gates --gate X --guidance-established 0.9558`） |
| 规则制导沿真值段 | P3 新测 | 门 X 的上限，门 E 的解释 |
| v2 最好的端到端（pa + B61） | v2 §12.1 的数字 | 门 E 的进展线 |
| `r11_lift` 跑道头 | `2026-09-13_runway_intent_plan.zh.md` §15 | 门 R |

读数锚点：固定锚点 60 为主；锚点 30（部署首问）作旁读。分层：直线 / 雷达引导（`data/anchor_strata.py`）。每格带 n；forecast 结束后保持末行（v2 §4 的读法），只有真值落地才缺席。

### 3.2 各层指标

| 层 | 指标 |
|---|---|
| 分词器 | **L1 增益**：同配方 L1 带真值码对不带 token 的 ADE[0,Δ] 配对 Δp50；**可预测性**：真值码序列上一个二元 Markov 基线的 NLL 与先验的 NLL；码使用份额（最大份额、未用码数）；每个码从典型状态飞出的图 |
| 先验（开环） | top-1 码经 L1 飞出的路径在 60 / 120 / 180 / 300 s 的位移 p50，分层；next-token 准确率与 NLL；落地时间误差；码翻转率；对连续版本的同一套数 |
| 执行器（协议 C） | 逐码 e_track（段末误差 p50）；整段 ADE、完全可飞、建立；与规则制导沿真值段配对 |
| 端到端（协议 A） | ADE 全 / 直 / 雷，FDE，完全可飞，建立，逐问 e_plan（先验码与真值码各经 L1 飞出的段的终点差）与漂移比 |
| 跑道 token | 按运行日划分的准确率、NLL，分左右层 / 方向层（runway intent 计划 §5 的读法） |
| 多机 | 后机雷达 120 / 180 s 位移（oracle 与学习图各一列）；端到端里的间隔违反率（对 `faa_separation`），与观测数据自身的违反率并列 |

### 3.3 门（两种子）

| 门 | 判据 | 依据 |
|---|---|---|
| **T 分词器**（每个 `segment_s` × K × {学习码, 指令词表}） | (i) L1 增益：带真值码的 ADE[0,Δ] 比同配方无 token 臂低 ≥ 30 m（p50，两种子）；(ii) 可预测性：先验（P2 的单机版，跑道已知）在真值码序列上的 NLL 低于二元 Markov 基线；(iii) 最大码份额 ≤ 25 %，未用码 ≤ 10 %。K 的选择：过门的 K 里，增益与最好者相差 ≤ 10 m 的最小 K | 码的价值只能由 L1 和先验定义；重建误差不判。30 m 是 v2 §10.8 判"用了 token"的线 |
| **P 先验**（每个存活候选） | 锚点 60，雷达引导 120 与 180 s top-1 位移 p50 不差于 B61（768 / 1396）+ 125 m；直线不差于 B61（366 / 587）+ 60 m；**离散对连续**：协议 A 的雷达 ADE 与建立，离散版两种子都不差于连续版（差在种子线内算平手，平手取离散：它才有掩码与多机接口）；翻转率只报不判 | 开环不是主张；离散瓶颈是核心假设，必须有对照 |
| **S 段长选择**（两种子，KRDU） | 在过门 T 的候选里，按协议 A（闭环训练之前）读雷达引导 ADE 与建立份额：两种子都最好的候选胜出；两项不一致时以建立份额为准；差在种子线（125 m / 3 点）内算平手，平手取更长的段。只有胜出者进入 P3 的闭环训练与 P4、P5 | 段长是执行器与先验的共同轴，只有端到端读数能判 |
| **X 执行器** | 协议 C：完全可飞 ≥ 0.95；建立 ≥ 0.9 × 规则制导沿真值段的建立；雷达 ADE ≤ 1500、直线 ≤ 250（v2 门 L1 的数） | 执行器不能比规则制导差太多 |
| **E 端到端** | 目标：雷达 ADE ≤ 2745、直线 ≤ 415、完全可飞 ≥ 0.95、建立 ≥ 0.94（v2 G3）。进展门：雷达 ADE 比 3044 / 3074 低 ≥ 125 m 且建立比 0.640 / 0.649 高 ≥ 5 点，两种子 | G3 从未达到过；进展门保证每一轮有可判的结论 |
| **R 跑道 token** | 按运行日划分，KRDU 左右层准确率与 `r11_lift` 同划分同航班配对差 ≤ 2 点 | 跑道不是主张，只需不拖后腿 |
| **G 多机** | 上限：前机真值码作上下文，后机雷达 120 / 180 s 位移 p50 好 ≥ 125 m → go；学习图：拿到上限增益的一半以上；端到端间隔违反率 ≤ 观测数据自身 | 先量上限，再建图 |

### 3.4 种子与读数规则（沿用包内约定）

两种子 1337 / 2024；控制路径的种子线 125 m 池化 ADE；配对读数用同航班的 Δp50 与"臂更好"份额，不读 p 值；
一个机场的 ADE 不带路线混合不算比较；训练臂在 launch 前登记 `docs/experiments/intents.json`，脏树拒绝启动。

---

## 4. 开发计划

每步：写代码 → opus review → 修 → 测试 → 提交；实验、全套测试、review 交后台 opus agent（用户规则）。**读数进 results 文件，本文状态表只写结论。**

### 4.1 阶段与步骤

| 步 | 内容 | 产物 | 门 / 读数 | 估时 |
|---|---|---|---|---|
| **P0** | 本文；文献；仓库整理；磁盘 | 完成（9dbb492） | — | — |
| **P1.1** | `manoeuvre/segments.py`：段切分、段起点系、段的输入行（六通道 + 行序号）、`segment_s` 参数；测试（旋转不变性、飞出段与观测段同一路径） | 模块 + 测试 | — | 0.5 d |
| **P1.2** | `manoeuvre/tokenizer.py`：编码器（小 transformer，读段的行 + 当前状态）+ FSQ；码本产物与 sha；冻结后的 `encode(段, 状态) → (c, z)`；**指令词表基线**的规则提取器（同接口） | 模块 + 测试 | — | 1 d |
| **P1.3** | 执行器侧：`plan_conditioning = manoeuvre-code`（token = z；dropout 强制 0；`codebook_sha256` 进 checkpoint；`control_horizon_s = segment_s`，N₁ = `segment_s / 10`）；**联合训练**：现有训练循环里编码器作为 L1 的一个子模块，梯度经 FSQ 直通回到编码器；训完把编码器与 FSQ 写成码本产物并冻结 | 模块 + 测试 | — | 1.5 d |
| **P1.4** | campaign `manoeuvre_tok_<date>`：KRDU，5 个 `segment_s` × K ∈ {16,32,64,128,256} × 两种子（联合臂）+ 每段长一个无 token 臂 + 指令词表臂；`manoeuvre/readout.py` 的 L1 增益、码使用、码的图；可预测性在 P2.3 补 | ≤ 60 个联合臂（各约 15 min） | **门 T**（i、iii） | ~15 GPU h，可按段长分批 |
| **P2.1** | `manoeuvre/prior.py`（模型、损失、解码）、`manoeuvre/context.py`（跑道、机型 token；fix 留到 P4）、`manoeuvre/sequences.py`（航班 → [码, 状态] 序列，split 按运行日）；**连续版**（回归编码向量）同一文件一个开关；训练 CLI 复用 campaign 守卫 | 模块 + 测试 | — | 1 d |
| **P2.2** | `manoeuvre/readout.py`：开环位移（自己跑 native32；B61 / state 数字引自 v2）、翻转率、二元 Markov 基线 NLL | runner | — | 0.5 d |
| **P2.3** | campaign `manoeuvre_prior_<date>`：每个过 T(i)(iii) 的候选 × 两种子，离散与连续各一 | ≤ 40 个 checkpoint（分钟级 / 个） | **门 T(ii)**、**门 P**（开环部分） | ~2 GPU h |
| **P3.1** | `manoeuvre/lockstep.py`：协议 C / A / A-truth，一轮 = `segment_s`，飞出段经编码器回灌，规则制导沿真值段（上限）与不可飞时接管；`manoeuvre/gates.py`（T / P / S / X / E / R / G，拒绝单种子） | 模块 + 测试 | — | 1.5 d |
| **P3.2** | 协议 C lockstep（每个存活候选，两种子）；协议 A（离散与连续各一）+ A-truth | 读数 | **门 X**、**门 P**（离散对连续）、**门 S**；门 E 第一读（预期不过） | 分钟级 / 次 |
| **P3.3** | 只对胜出段长：闭环训练第四步——先验吃飞出段（share 0.75）、执行器自训各一轮；端到端再读 | 4 个 checkpoint | **门 E** | ~1 GPU h |
| **P4.1** | 跑道 token 作输出（运行日划分）；程序 fix 上下文（`outputs/guidance/skeleton.py` 读取器） | 模块 + 测试 | — | 1 d |
| **P4.2** | campaign：跑道未知的先验两种子；端到端"跑道未知"读数 | 读数 | **门 R**；E 的跑道未知列 | 分钟级 + lockstep |
| **P5.1** | `manoeuvre/scene.py`：从 `tracks/manifest.json` 建场景（含 not_landing 邻机），运行日划分 | 模块 + 测试 | — | 1 d |
| **P5.2** | oracle：前机真值码作上下文 → 后机读数 | 读数 | **门 G 上限**（no-go 则 P5 止于此） | 分钟级 |
| **P5.3** | `manoeuvre/graph.py`（边特征、注意力偏置）、`manoeuvre/decode.py`（到达顺序、间隔掩码：候选码经 L1 单段 rollout）；多机端到端 | 模块 + campaign | **门 G** | 1–2 d |
| **P6** | 发布每个 campaign 的读数到前端 picker（带 intents）；`CLAUDE.md` 索引行；`ENGINEERING_NOTES`；results 文件收口 | — | — | 0.5 d |

P1.4 是最大的一笔 GPU（段长 × K × 种子）；若要压缩，先跑 K ∈ {32, 128} 两档定量级，再补中间档。P2 依赖 P1 的码本；P4、P5 依赖 P3.3 的胜出者。
**P1 之前先由用户审核本文。**

### 4.2 每步的固定动作

- 代码：新文件放 `manoeuvre/`（§5.3），`tests/test_manoeuvre_<topic>.py` 一主题一文件；`tests/test_architecture.py` 加边界
  （`manoeuvre` 可以 import `outputs.control`、`outputs.guidance`、`data`、`inference.runway_schedule`；反向禁止）。
- 实验：臂声明 `docs/experiments/manoeuvre_<campaign>_arms.json` + intents 条目**先提交再 launch**；cohort 用 `run_ts.py plan_cohort`。
- 读数：写进 `2026-09-18_manoeuvre_token_results.zh.md`，本文状态表一行；每个 campaign 结束由 opus 子代理发布到 picker（带 intents）。
- 不删、不覆盖任何产物；readout / gate 输出目录存在即拒绝。

### 4.3 队列与磁盘

- 队列用 detached 链（v2 §13.3 的机制）：`nohup setsid bash <campaign>/queue.sh > queue.log 2>&1 &`，PID 文件，每步产物存在即跳过、失败重试一次、
  磁盘 < 3 GB 停；Monitor 看日志。runs worktree 训练前移到已提交的 commit，`status --porcelain` 必须为空。
- 磁盘：只给门臂写记录（每臂约 2 GB）。

---

## 5. 仓库整理

原则：新方案在干净的基础上开发；被取代的代码归档（`archive/` 在 import path 之外，`tests/test_architecture.py` 断言），
不留双份；被取代的路径的 checkpoint 不再可加载，其已发布的 CZML 类目不受影响（发布物不依赖代码）。

### 5.1 归档清单（已执行 2026-09-18；每个 archive 目录一个 README）

| 去向 | 内容 | 理由 |
|---|---|---|
| `archive/two_tier_v2_2026_09/` | `outputs/segment_plan/`（整个包）；`experiments/segment_plan_readout.py`、`short_horizon_readout.py`、`two_tier_gates.py`、`tracker_lockstep.py`；`outputs/control/plan_token.py` 的旧实现（`plan_token_v2.py`：`waypoints` 与 `truth-next` token）；臂文件 `two_tier_*_arms.json`；其旧测试原样归档在 `tests/`（用户规则：旧测试随模块归档，不改不删） | 被 §2 取代；`plan_token.py` 留下只含 `off` 的接缝 |
| `archive/plan_head_2026_09/` | `outputs/plan/{model,strategy,labels,extractors,forecast,rolled,__init__}.py`；`experiments/plan_oracle.py`、`plan_oracle_pair.py`、`plan_rolled_windows.py`、`plan_fan_readout.py`、`plan_next_readout.py`、`plan_extractors.py`；**runway intent 的 R2 / R3 / R3.1 / R3.2 runner**（它们读 plan 头；R0 / R1 / R1.1 保留，R1.2 在其上）；`docs/phase0_intent_diagnostics.py`。其旧测试原样归档（含 `test_plan_guidance.py`、`test_plan_timing.py`，它们经 plan 头的 forecast 测制导层） | plan 头被先验取代；**`outputs/plan/guidance/` + `skeleton.py` 移到 `outputs/guidance/`**（第二执行器），它从 plan 头借的两个常量随之搬入（`route.ON_COURSE_FIX_M`、`timing.SPEED_MAX_MPS`） |
| `archive/closure_2026_09/` | `outputs/closure/`；`docs/p1_closure_oracle.py`；旧测试原样归档 | 2026-09-09 已冻结，无新用途 |
| config 收缩 | `PREDICTION_OUTPUTS = (state, control)`；三个旧名字进 `PREDICTION_OUTPUTS_RETIRED`（存的 config 带它们时加载即拒绝，并指向 archive），前端镜像改读 `PREDICTION_OUTPUTS_PUBLISHED`（活的 + 退役的，已发布类目仍能命名）；closure / plan / segment-plan 的全部字段、视图、校验、CLI 标志、命名规则删除；`plan_conditioning` 取值暂为 {`off`}，`manoeuvre-code` 在 P3.1 加；`control_horizon_s`、`plan_conditioning_dropout`、`inference/receding.py` 保留 | 一个字段一个活的用途 |

保留不动：`outputs/control/`（执行器）、`outputs/state/`（参照）、`outputs/dynamics`、`constraints`、`envelope`、`conditioning`、
`data/`、`backbone/`、`training/`、`inference/`（`receding.py`、`runway_schedule.py`）、runway intent 的 R 系列 runner 与测试、
`frame_ablation`、`plan_cohort`、`anytime_curve`、`eta_*`、`quantile_fan_readout`、`latent_*`、`lead_time_error`、`chain_sensitivity`、`anchor_floor_index`。
`control_horizon_s` 保留（执行器用）。

### 5.2 保留但降级的东西

- `2026-09-17_two_tier_plan_v2.zh.md`、`2026-09-09_plan_and_guidance_design.md`：头部加"已被本文取代"，正文不动（数字被引用）。
- `docs/experiments/intents.json` 里 two_tier / plan 的条目：保留（已发布类目的记录）。

### 5.3 新包布局

```
ts_transformer/manoeuvre/
  __init__.py
  segments.py     段切分、段起点系、段的输入行（P1.1）
  tokenizer.py    编码器 + FSQ、码本产物与 sha、冻结后的 encode(段, 状态) → (c, z)、指令词表基线（P1.2）
  sequences.py    航班 → [码, 状态] 序列；运行日划分（P2.1）
  context.py      跑道端、程序 fix、机型的上下文 token（P2.1 / P4.1）
  prior.py        因果 Transformer、损失、解码；连续版开关（P2.1）
  lockstep.py     协议 C / A / A-truth，一轮 = segment_s，回灌，规则制导上限与接管（P3.1）
  readout.py      L1 增益、码使用、码的图、开环位移、翻转率、Markov 基线、逐码 e_track、端到端读数（P1.4 / P2.2 / P3）
  gates.py        门 T / P / S / X / E / R / G 的判定（拒绝单种子）（P3.1）
  scene.py        多机场景（P5.1）
  graph.py        边特征、注意力偏置（P5.3）
  decode.py       到达顺序、间隔掩码（P5.3）
experiments/manoeuvre_{tokenizer,prior,lockstep,readout,gates,scene_oracle}.py   run_ts.py 入口
tests/test_manoeuvre_*.py
4dTrajectory/outputs/codebooks/<name>/            编码器权重、FSQ 档位、K、segment_s、队列身份、sha
4dTrajectory/outputs/<ICAO>/experiments/manoeuvre_<campaign>/
```

执行器侧的改动：`outputs/control/plan_token.py`（`manoeuvre-code` token = z）、`config.py`（取值、`codebook_sha256`、`segment_s`），以及训练循环里把编码器挂成 L1 的子模块做联合训练（P1.3）。

### 5.4 磁盘上的旧产物（已执行 2026-09-18）

| 目录（原在 `4dTrajectory/outputs/KRDU/experiments/`） | 原大小 | 处置 |
|---|---|---|
| `two_tier_l1_20260917` | 5.9 GB | 归档（checkpoint + 读数 JSON/TXT），`short_horizon/records` 删 |
| `two_tier_l2_20260917` | 2.5 GB | 归档，`readout/records` 删（B61 数字在 v2 §10.6，checkpoint 已不可加载） |
| `two_tier_t0b_20260916`、`two_tier_t0c_20260916` | 2.0 + 1.2 GB | 归档，`chain_*/records` 与 `*_pred_val` 删 |
| `two_tier_l1c_20260917`、`l1b`、`t1a`、`l2c`、`l2d`、`feasibility` | 0.9 GB 合计 | 归档（无记录树） |
| `plan_guidance_20260910` | 2.8 GB | **留在原处**（规则制导基线 lockstep json、已发布类目；plan 头 checkpoint 不再可加载，未动） |
| `l1_lowdim_20260907`、`airport_frame_20260903` | 0.8 + 1.0 GB | **留**（参照） |

```bash
# 已执行（2026-09-18）：整树移入归档，再只删记录树
A=/home/supercomputing/studys/thesis/4dTrajectory/outputs/archive/KRDU/two_tier_2026_09
mv 4dTrajectory/outputs/KRDU/experiments/two_tier_* $A/
find $A -type d \( -name records -o -name '*_pred_val' \) -prune -exec rm -rf {} +   # 归档树 1.8 GB
```

之后重建了索引（110 个实验目录），并从 `comparison/categories.json` 撤下 38 个 two_tier 类目（L1 24 + L2 8 + T0c 6），删除其 CZML 目录，剩 165 个类目；`check-publication` 见状态表。

---

## 6. 风险与待决

### 6.1 风险与后备

| 风险 | 信号 | 后备 |
|---|---|---|
| L1 学会不看码（v2 §10.8 的重演） | 门 T(i) 的增益 < 30 m | dropout 已强制 0；K 减小（码更粗、更有用）；检查联合训练里编码器的梯度是否到达 |
| 编码器把段全编进码（K 太大，回到航迹压缩） | T(i) 增益大但 T(ii) 先验 NLL 不比 Markov 基线好、翻转率高 | 按 T 的选择规则取小 K；指令词表基线作参照 |
| 先验在雷达引导段是平的 | NLL 高、翻转率高、位移与 B61 持平 | 这是诚实的输出；P5 的上限决定是否值得建图 |
| 离散版输给连续版（分词无价值） | 门 P 的对照 | 回到带回灌的 PatchTST 形态，掩码与多机改在连续空间做（代价：约束只能罚项） |
| 执行器在转弯段仍不建立 | 门 X 的建立远低于规则制导上限 | 逐码看哪些码不行；规则制导接管这些码；门 S 会偏向短段 |
| 闭环第四步没帮上（plan 路径 DAgger 的先例） | 门 E 进展门不过 | 检查标签的时间索引与 `LANDED`；A-truth 诊断分清是耦合还是先验 |
| 联合训练不稳（FSQ 直通 + rollout） | 损失震荡、码使用塌到几个 | 先冻结一个重建式编码器作初始化再联合；FSQ 档位减少 |
| 运行日划分砍掉数据 | 跑道 token 与多机的 val 只剩几十天 | 五机场合并；报 n |
| 旧发布类目失去代码支撑 | picker 类目在，checkpoint 不在 | 发布物是 CZML，不需要代码；intents 记录保留 |

### 6.2 已决定（用户 2026-09-18）

1. **归档清单 §5.1** 照单执行；**checkpoint 归档而不是删除**（`4dTrajectory/outputs/archive/`），以便日后恢复。
2. **删除清单 §5.4** 照单执行（记录树；checkpoint 与小读数文件归档）。
3. picker 里 two_tier 的类目随索引重建**撤下**。
4. 段长不定死：`segment_s` 作消融轴（§2.3、门 S）。
5. **P0 做完后停下，用户审核本文之后再开始 P1。**
6. **分词器以 L1 为解码器，码只装当前状态决定不了的决策；K 小；重建误差不作门**（晚间讨论，用户：L2 层要学习的应该是意图或者一个可用的解空间）。

其余选择（K、pa 契约为唯一主臂、跑道在 P4 才作输出）按"规则允许的由我定"处理，写在本文，不再另问。

### 6.3 我替你选的（2026-09-18，P1.1–P1.3；明天审核，改了就是新臂）

1. **分层的一个反向边。** 执行器把分词器挂成子模块、把真值段行写进 context row，所以 `outputs/control` 必须 import
   `manoeuvre.segments` 与 `manoeuvre.tokenizer`。这两个是**叶子**（只 import data、config、io_utils、torch，白名单，
   `test_architecture` 断言）；其余 manoeuvre 模块 import 控制路径，反向禁止。§4.2 的"反向禁止"按此解释（L29）。
2. **`segment_s` 就是 `control_horizon_s`**，不另设字段：执行器一轮飞满一段，段长与时域是同一个数；码本产物记 `segment_s`。
3. **`plan_conditioning_dropout` 退役为常量 0**（不是"强制 0"的校验）：存 0.5 的只有已归档的 L1/L1b，加载本就被拒。
4. **FSQ 档位的分解**（K → levels）：16 = (4,4)、32 = (8,4)、64 = (4,4,4)、128 = (8,4,4)、256 = (4,4,4,4)——2–4 维、每维 ≥ 3 档
   （L=2 时 bound 的位移是 atanh(1)）；z 的维数就是 L1 条件向量的宽度。编码器尺寸是模块常量（d 64、2 层、4 头），不是实验轴。
5. **编码器的"当前状态"输入 = 锚点行（跑道入口 ENU 图的六通道）**，不是整段历史；尺度是固定常量并写进码本（review MAJOR-3）。
6. **指令词表是 7 × 3 × 3 = 63**，不是 §2.4 写的"约 45"：航向按 §2.4 列出的四档（≤15°、≤45°、≤90°、反向）× 左右 + 直飞，
   航向变化沿行**展开**（unwrap）读，180° 反向不再被 wrap 读成反方向（review BLOCKER-1）。高度、速度按段内平均速率分档
   （±1 m/s、±0.03 m/s²，reading），与段长无关；真实队列上的 63 类直方图在 P1.4 读数里报，不预设覆盖。
7. **N₁ = segment_s / 10 是 campaign 约定**（写在臂文件），不是 config 规则：N₁ 消融不该要改代码。
8. **码用量与分词器梯度进每轮记录**：`control_training_diagnostics.manoeuvre_codes`（K、用了几个、最大份额、熵）与梯度组
   `manoeuvre_tokenizer`（其他 run 恒为 0——多一个 key 的代价接受了）。
9. **码本身份（尺度、队列身份、来源 checkpoint）都在 sha 之内**；空身份拒写；对着码本训的执行器在 `load_checkpoint` 时
   校验目录里的权重仍是 checkpoint 里的那份（C28）。
10. **先验（P2）在执行器自己的航班划分上训练与读数**（同一 development cohort 的 train / val），不用运行日划分：lockstep 要在执行器的
    val 航班上配对，先验不能见过它们；运行日划分（T4）在 P4 跑道 token 时再引入（`sequences.split_by_operational_day` 已写好）。
11. **先验的状态 token = 跑道入口图里的 (e, n, u, 地速, cos ψ, sin ψ)**（P2–P3 跑道已知，图就是跑道的）；P4 跑道未知时改到机场系。
    `LANDED` 不是第 K+1 类而是每个位置一个独立的二元头（离散、连续两版同一骨架），落地分数 = 末段之后剩余的段分数 ∈ [0, 1)。
12. **连续对照的回归目标 = 编码器 bound 后、round 前的坐标**（与 z 同一空间，`Codebook.encode_continuous`），执行器按 `manoeuvre_z` 吃它。
13. **门 T 在 60 s 段 FAIL 后我做的事（2026-09-18 08:08）**：门按预注册判、不改线；campaign 照跑其余段长（30 → 90 → 20 → 120）；
    在未过门的 (60 s, K32) 上启动 P2.3 + P3.2 的全链作**诊断**（`<campaign>/queue_p23.sh 60 32` → `p23_S60_K32/`），因为它回答的问题
    （先验的可预测性与翻转率、执行器闭环下的建立与可飞、端到端对 v2 3044 m 的位置）不管 K 怎么选都要读，而且与训练队列并行不抢多少 GPU。
    它的读数**不是**过门 S 的候选，除非用户改门 T。**门 E 第一读（A · s1337）到了：雷达 ADE 2531 过进展线、建立 0.422 不过**，按计划
    P3.3 是回答——先验侧（回灌训练）08:20 排队、08:49 完成：**门 E 仍 FAIL**（建立 0.462 / 0.476，雷达 ADE 略差；results 文件 P2.3 + P3.2 节）；执行器侧不写（门 E 没动）。 **08:51 又在指令词表臂上跑同一诊断链**（`queue_p23.sh 60 cv` → `p23_S60_cv/`，无连续对照）：一个固定的语义词表在闭环里的建立与 ADE，是学习码本的非学习参照，第 14(a) / 15 条要用。
14. **待用户决定（三个事后改门的问题，我没有替你改）**：
    (a) **T(iii) 的 25 % 最大份额线**：数据自身在固定锚点的主类份额是 79 %（指令词表"直飞 / 下降 / 减速"），随机锚点上 66 %，任何忠实的
    分词器都过不了。建议改成**相对指令词表**：学习码本的最大份额 ≤ 指令词表的、未用份额 ≤ 指令词表的 + 10 点；或只保留未用份额一条。改了是新门，不是新臂。
    (b) **T(i) 的 30 m 线**来自 v2 §10.8 的整条进近（ADE 1300 m 量级）；60 s 段上无 token 臂本身只有 85 m，30 m 是它的 35 %，而增益本来就
    只可能出现在雷达引导层（n 497）。两个方向：按 Δ 与层缩放（如雷达引导层 p50 增益 ≥ 无 token 臂雷达 ADE 的 15 %）——事后设线，要用户定；
    或保持绝对线、接受 60 s 段 FAIL，由 90 / 120 s 段判（段越长历史决定得越少，码的空间越大，但 §2.3 的单峰性变差）。**我的倾向：保持绝对线，
    等 90 / 120 s 的读数再说。**
    (c) **未用码份额 56–84 %**：§6.1 的后备"先冻结一个重建式编码器作初始化再联合；FSQ 档位减少"是设计改动 = 新臂族，不启动；K16 = (4,4) 已是
    队列里最小的档位，它的读数先看。
    (d) **码里没有垂直意图**（诊断链的码图谱，results 文件 P2.3 + P3.2 节）：K32 的两维 FSQ 共线成一把航向扇，所有码下执行器的垂直剖面几乎相同，
    而真值的平飞 / 陡降段正是历史决定不了的决策。原因是联合训练的 ADE 由水平误差主导（T22）。要码装垂直决策，得给垂直误差单独的分量
    （损失里的垂直权重，或段末高度的单独项）——设计改动 = 新臂族，用户定；我不启动。
    (e) **lockstep 的"落地即停"规则**（§2.7：先验说落地就按份额飞 top-1 然后停）把"建立"变成了先验落地时刻与执行器滞后的联合检验：
    协议 A · s1337 有 732 / 1392 架由落地判断结束、全部不建立，其中直线层停点离入口 p50 539 m、时长只差 −3.5 s（results 文件 P2.3 + P3.2 节）。
    可选的改法：落地判断只停止"再问"，执行器沿最后一个码飞到穿越或预算为止，先验的落地时刻另读（`final_time_error_s`）。这是协议改动，
    门 E 的建立线与 v2 的 0.640 基线都要重解释——用户定。P3.3 的回灌先验（已启动，第 13 条）先回答漂移那一半。
15. **离散对连续的第一读不在这里定。** 门 P 的对照在 (60 s, K32) 上是连续版赢（雷达 ADE 好 272 / 141 m，建立平手；results 文件 P2.3 + P3.2 节）。
    §6.1 的后备是"回到带回灌的 PatchTST 形态、掩码与多机改在连续空间做"——那是换主线的决定。我的读法：这个 K 没过门 T，码本塌成一把 8 档航向扇
    （约 13 个不同终点），连续坐标只是把这把扇的档位变细了；差距 140–270 m 正是量化损失的量级。要判"分词无价值"，得在一个装了垂直意图、
    过了 T 的码本上读。**建议顺序：等 60 s 全读（K16 / K64 / K256）与 P3.3 的门 E → 若仍无 K 过 T，先做第 14(d) 条的垂直分量臂族再判离散对连续；
    用户定。**

---

## 7. 引用与索引

### 7.1 论文（`docs/literature/manoeuvre_tokens/`，2026-09-18 建档，13 篇 PDF 全部核实；每篇的事实与页码在该目录 README §2）

| # | 论文 | 在本文的用处 |
|---|---|---|
| [1] | Seff et al., *MotionLM: Multi-Agent Motion Forecasting as Language Modeling*, ICCV 2023, arXiv:2309.16534 | 离散运动 token（Δx, Δy 量化，13² = 169 词、0.5 s）+ 多机联合自回归，同一步各机并行解码（§2.5、§2.8） |
| [2] | Philion, Peng, Fidler, *Trajeglish: Traffic Modeling as Next-Token Prediction*, ICLR 2024, arXiv:2312.04535 | k-disks 分词（384 词、0.1 s，同 \|V\| 下优于 k-means）；同一步内各机**串行**、后者读前者已采样的 token（§2.8 的提交顺序）；它自己的结论：上下文够长时同步交互作用弱 |
| [3] | Wu et al., *SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction*, NeurIPS 2024, arXiv:2405.15677 | 0.5 s 段 + k-disks 词表（512–2048），无残差、纯交叉熵；作者列出 token 时间粒度未扫（§2.3 的段长消融） |
| [4] | van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning*, NeurIPS 2017, arXiv:1711.00937 | VQ-VAE（P1 后备）；承诺损失 β = 0.25 |
| [5] | Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple*, ICLR 2024, arXiv:2309.15505 | FSQ（P1 后备，优先于 VQ）：无码本、无辅助损失，码本利用率约 100 %（VQ 81 %） |
| [6] | Lee et al., *Behavior Generation with Latent Actions*, ICML 2024, arXiv:2403.03181 | 码 + 由观测回归的连续偏移头（§2.4 的残差轴），且是唯一对动作**块**分词的 |
| [7] | Ying et al., *Do Transformers Really Perform Bad for Graph Representation?*, NeurIPS 2021, arXiv:2106.05234 | 边特征作注意力 logit 的加性偏置 c_ij（§2.8） |
| [8] | Bengio et al., *Scheduled Sampling for Sequence Prediction with RNNs*, NeurIPS 2015, arXiv:1506.03099 | 训练 / 推理分布差与误差累积（§2.7 的闭环训练） |
| [9] | Geng et al., *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*, EMNLP 2023, arXiv:2305.13971 | 约束 = 解码时对词表的掩码，与解码算法无关（§2.8） |
| [10] | Olive & Morio, *Trajectory clustering of air traffic flows around airports*, AST 84 (2019), DOI 10.1016/j.ast.2018.11.031 | 进近流的点级 DBSCAN 聚类；"公布的 STAR 不足以描述机场周围的流"（§2.5 的程序上下文只是上下文） |
| [11] | Corrado et al., *Trajectory Clustering within the Terminal Airspace Utilizing a Weighted Distance Function*, Proceedings 2020 59(1):7 | 整条进近航迹的加权欧氏 + HDBSCAN，KSFO 一天约五个流 |
| [12a] | Murad & Ruocco, *Synthetic Aircraft Trajectory Generation Using Time-Based VQ-VAE*, ICNS 2025, arXiv:2504.09101 | 航空领域唯一的 VQ 先例：生成而非预测，频带隐码无机动语义，可飞性事后用 BlueSky 查 |
| [12b] | Luo & Zhou, *Large Language Models for Single-Step and Multi-Step Flight Trajectory Prediction*, arXiv:2501.17459 | 反例：其"token"是数字字符串的 BPE 子词，不是机动码 |
| — | DAgger（Ross et al. 2011）、Ross & Bagnell 2010 | 已在 `docs/literature/hierarchical_prediction/`；§2.7 的闭环训练 |

建档时读出的两条对设计的提醒：(i) 文献里每个 token 的时间步是 0.1–0.5 s，60 s 比它们长两到三个数量级，且 SMART 明说时间粒度未扫——这正是 §2.3 把段长做成消融轴的理由；(ii) 没有一篇运动 token 论文端到端学码本（均匀分箱或 k-disks），学习式码本（VQ / FSQ / VQ-BeT）来自图像与机器人动作——本文的分词器走学习式一路（§2.4），且解码器是执行器本身，这一点没有先例，是本方案要验证的东西。

### 7.2 仓库内文档

| 文档 | 用处 |
|---|---|
| `2026-09-17_two_tier_plan_v2.zh.md` §10.1–§10.8、§12.1–§12.3 | §1.1 的数字；门 E 的进展线；B61 参照 |
| `2026-09-09_plan_and_guidance_design.md` §12.5–§12.8 | §1.2 的五条经验；规则制导；`closing_horizon_s` |
| `2026-09-08_hard_constraints_survey_and_integration_plan.md` §3.5–§3.7 | 离散决策 + 约束生成的方法归类 |
| `2026-09-13_runway_intent_plan.zh.md` §5、§6、§15 | 运行日划分、跑道读法、`r11_lift` 基线 |
| `docs/literature/arrival_separation/README.md` | 间隔规则的出处（掩码用） |
| `docs/literature/hierarchical_prediction/README.md` | 两层结构的文献与反证 |
| `docs/reference/contracts.md`（C1、C12、C26）、`traps.md`（T1、T4） | 通道、conformal 绑定、数据身份、跑道已知、运行日划分 |
| `docs/ENGINEERING_NOTES.md`、`docs/OPEN_ITEMS.md` | 证据与状态 |

### 7.3 术语索引

| 术语 | 定义处 |
|---|---|
| 段、段起点系、段的输入行 | §2.3 |
| 意图空间、码、码本、K、`codebook_sha256`、指令词表基线 | §2.4 |
| 先验、state token、上下文 token、`LANDED`、跑道 token | §2.5 |
| 执行器（= 解码器）、规则制导上限 | §2.6 |
| 协议 C / A / A-truth、第四步 | §2.7 |
| 场景、边特征、间隔掩码、提交顺序、oracle 上限 | §2.8 |
| e_track、e_plan、翻转率 | §3.2 |
| 门 T / P / S / X / E / R / G | §3.3 |
| 进展门 | §3.3（E） |
| 训练效率的账 | §2.9 |
