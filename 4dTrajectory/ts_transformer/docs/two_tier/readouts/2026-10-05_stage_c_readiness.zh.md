# 两层模型 v4 阶段 C：现在能不能与 B 并行开发（2026-10-05）

问题（用户，2026-10-05）：读 `docs/two_tier/design/` 的设计文档和 `dev-two-tier-v4` 上的代码，评估现在 C 阶段能不能开始并行开发。

读了什么：`outline.md`、`post_training.md`（全文），`prior.md`（§0、§1、§3–§12），`vocabulary.md`（§0、§6、§7.2）；代码读的是
`dev-two-tier-v4-prior` `0ea056f5`（已合入 `dev-two-tier-v4` `58fd8a2b`，即 A32、A33、A36 都在内），因为 C 要读的 `prior/` 只在这条分支上。
下面的“已核对”是我读代码确认过的；“我的读法”是推断，没有跑代码验证。

## 1 结论

**现在不能整体并行开工。** 挡住的主要是设计，不是代码：

1. C 的设计还是大纲，没有能照着做的里程碑计划，两个未决项（O6、O9）挡在关键路径上（§2）。
2. 计划和分支规则没有给 C 并行留位置；B 的并行是用户 2026-10-04 的明确决定，C 也需要用户决定（§2）。
3. A 和 B 的公共接口缺 C 必需的几样东西，C 按规则不能自己改 A、B 的代码，要像 A36 那样向它们提需求（§3）。

把设计补齐、接口需求落地后，C 中**不依赖 base 模型**的部分可以与 B5 并行（§4）。

## 2 设计和流程上的阻碍

| # | 阻碍 | 依据 |
|---|---|---|
| 1 | 设计是大纲。文档自己说是 outline，窗口数量等“在 C 阶段定”；§8 是一段话，没有 C0…Cn 的里程碑、每步的测试和开工时间表。B 开工时 prior.md 已经“对它的部分是完整的”（prior §0 Scope），有 §12 的 B0–B8 | `post_training.md` Scope of this document、§2 item 4、§8；`prior.md` §12 |
| 2 | O6 未决：“已稳定在最后进近上”的规则，以及最后进近上的间隔掩码。分离判定（谁在后、谁负责）、速度词掩码都读它 | `post_training.md` §0.2、§3 |
| 3 | O9 未决：闭环里一个航班最多复飞几次。闭环的起点（`start`）要这个数来排时间 | `post_training.md` §0.2；`vocabulary.md` §6 item 5 |
| 4 | 计划是先后顺序：“C 从 B 的 base 模型开始” | `outline.md` §4 item 3；`post_training.md` §0.4 item 1 |
| 5 | 分支规则只有 A、B 两条分支和它们的合并方向，没有 C 的分支、它从哪里拉、谁合进谁 | `outline.md` §5 rule 1 |
| 6 | 没有写“C 不改 A、B 的代码”这条规则（B 有：“Stage B never changes the code of `instructions/` or `autopilot/`”）。按 B 的先例应当同样适用，但要写进去 | `outline.md` §5 rule 1 |

第 5 条，我的读法：C 读 `prior/`，所以 C 的分支应从 `dev-two-tier-v4-prior` 拉，合并方向是 B → C；B7 合回 `dev-two-tier-v4` 之后改为从
`dev-two-tier-v4` 合入。由用户决定。

## 3 公共接口的缺口（已核对代码，`0ea056f5`）

C 只通过 vocabulary §6 和 prior §7 使用 A、B 的代码。下面这些是 C 的设计要用、但接口现在给不了的：

| # | C 要什么 | 现在的代码 | 向谁提 |
|---|---|---|---|
| 1 | 分支训练（D37）在分支点存下闭环的状态，再从它重说 K = 8 次：executor、判定、loop 的状态 | `autopilot/start.py` 的 `Loop`、`autopilot/executor.py` 的 `Executor` 都没有存、复制或恢复状态的功能（`autopilot/` 里没有 copy / snapshot）。对它们做 `deepcopy` 也许能用，但不是接口承诺，没有测试 | A（vocabulary §6 item 5） |
| 2 | 增强窗口 B：起点平移（绕机场转一个角、改高度、改速度） | `start()`（`autopilot/start.py:215`）只从工件的闭环句子起飞，起点状态由 spec 的起点规则从观测行算出（D77）；没有“从给定的、移动过的状态起飞” | A（vocabulary §6 item 5） |
| 3 | traffic attention 模块的输入经过 speaker 进到模型 | 模型留了位置：`Layer.added(x, extra)`、`Prior.add_at_each_layer`、`Prior.extend(rows, past, extra)`（`prior/model.py:144–146`、`:228`、`:300`）；但 `Speaker.speak` 调用 `self.model.extend(row, self.past)`，不传 `extra`（`prior/speaker.py:137`），`observe` 也一样（`:125`） | B（prior §7 item 3） |
| 4 | 分支时复制说话者的状态：缓存、程序掩码的状态（是否已切入、是否曾在入口高度以下）、已说的词（`Heard`）、复飞计数 | `Past` 是原地写入的，注释写着“要分支就自己复制张量”（`prior/model.py:111–117`）；`Speaker` 没有复制或分支的方法 | B（prior §7 item 3） |
| 5 | 每个续说有自己的随机数（由种子、轮次、窗口、分支点、k 决定），用首句的随机数续说要逐位复现首句（D37 item 4 的测试） | `Speaker` 只有一个 `torch.Generator`，整批一次 `torch.multinomial`（`prior/speaker.py:156`）：一架飞机抽到的数取决于同批还有谁，不能按飞机重放。我的读法：要么每架一个随机流，要么由调用方给每架的均匀随机数 | B（prior §7 item 3） |
| 6 | 裁剪比值损失和对 base 的 KL 都在“加了掩码的分布”上算（post_training §2 item 5），训练时要知道说话时每行每列允许哪些词 | `speak` 每列算出 `permitted`（语法 + 程序掩码 + 调用方掩码），只用来抽词，不存；存下的只有被挡掉的概率和（`forbidden`）、程序掩码挡掉的词（`procedure_blocked`）（`prior/speaker.py:143–162`）。程序掩码的状态随飞行变化，训练时不能简单重算 | B（prior §7 item 3） |
| 7 | 带 traffic 输入的 teacher-forced 打分 | `Prior.forward(rows, extra)` 接受 `extra`（`prior/model.py:294`），但 §7 item 4 的损失 `batch_nll(model, rows)` 调 `model(rows)`，不传（`prior/train.py:85–87`）。C 的数据项是单机样本（D36），不需要；但若 C 直接调 `forward`，要确认这算在接口内 | B（确认即可） |
| 8 | 分离判定要知道一架飞机是否“已稳定在最后进近上”，由一行的状态和 R 算出（D31） | `inference/separation.py` 的说明还写着由调用方给“标注器或 executor 的 capture”（`:7–8`）；v4 的 executor 没有这种状态（vocabulary §6 item 5）。这是 C 自己的代码，但要等 O6 | C 自己（O6 之后） |

已核对、**不是**缺口的：

- 着陆上下文（D31）：`prior/landings.py` 的说明写明“在闭环里，由调用方给闭环知道的着陆”，`state_inputs` 接收着陆计数，C 可以把闭环里的着陆传进去。
- 每架飞机自己的起始周期：`Executor` 支持一批里各自从自己的周期起飞（vocabulary §6 item 5）；C 只指挥一架飞机（D29），其他飞机按记录飞，不经过 executor。
- 速度掩码要的“按 executor 的速率预测到前机过跑道头”：只读 vocabulary §6 item 1 的限制值，C 自己算即可。

## 4 哪些可以并行，哪些必须等

**设计补齐、§3 的接口需求合入之后可以并行**（用合成数据或 B3 的冒烟模型，主要用 CPU）：

- 窗口和场景的构建（20 分钟的记录交通，同一 UTC 网格）；
- 闭环里的着陆上下文（D31）；D23 在场景里的逐位测试；
- 边特征和它的一致性检查（§4 identity 1）；
- 分离判定移植到 v4，接上 O6 的规则；速度词掩码；
- D30 的奖励；
- 分支训练的机制：分支点存状态、续说、用首句的随机数逐位复现首句、推理模式和带梯度飞出的状态相同（D37 item 4、6 的测试）；
- traffic attention 模块（输出初始为 0，加上后 prior 的每个输出不变）。

**必须等**：

| 等什么 | 哪些工作 |
|---|---|
| B5 的 base 模型 | §2 item 6：base 在最后进近上说“go-around”的概率；按 B5 选出的配置测一个说话批次（§8 的 profile：B5 第 1 步可能选 B 或 C 配置，大小不同）；所有正式运行 |
| A34 的正式产物（正在建：`v12_20261005` / `v17_20261005`，2026-10-05 09:48Z 起） | 任何真实数据的冒烟：`v11_20261004` 已被 A32 的格式名拒读 |
| Claude 对 A32–A36 的检查（vocabulary §12.2 item 8） | 可能再改 vocabulary §6；C 依赖的接口在此之后才稳定 |

**资源**：B5 的 31 个训练大约占 7 小时 GPU，各折自由生成大约再加 1.5 小时（都是 prior §0.3 的估计）。这期间 C 只能用 CPU，并遵守 outline §5 rule 13：
不在正式构建旁边开 8 个 worker 的全量测试。

## 5 建议的顺序

1. **现在就能并行的，是 C 的设计**（Claude 的工作）：
   - 与用户定 O6、O9，以及窗口数量（真实窗口和 A、B、D 三种增强窗口各多少）；
   - 把 `post_training.md` §8 写成 C0…Cn 的里程碑，每步写测试和开工条件（参照 prior §12、§0.4 的格式）；
   - 在 outline §4、§5 rule 1 里加 C 的并行计划、分支规则和“C 不改 A、B 的代码”；
   - 写出 §3 的接口需求：第 1、2 条给 A，第 3–6 条给 B，附补丁草稿（参照 A36：由所属阶段实现，C 不动它们的代码）。
2. A、B 合入这些接口之后，C 在自己的分支上开工 §4 的“可以并行”部分。
3. B5 出 base、A34 和对 A32–A36 的检查完成之后，接入真实 base，测 go-around 概率和说话批次的耗时，再定正式规模。

## 6 要用户决定的

1. C 是否与 B 并行（outline §4 现在写的是先后顺序）。
2. 若并行：C 的分支从哪里拉、合并方向（§2 第 5 条我的读法）。
3. O6、O9 和窗口数量，作为设计补齐的第一步。
4. §3 的接口需求是否按 A36 的方式交给 A、B 去做。
