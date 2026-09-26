# 两层模型的总体框架

**一句话**：上层是"说话的人"——一个自回归模型（先验），每 2 s 决定对这架飞机下什么指令；下层是"飞的人"——一个只按动力学
执行指令的自动驾驶（执行器）。两层之间只通过**词**交流，词的定义见[指令词表设计](2026-09-23_instruction_vocabulary_design.zh.md)。
本文定整体结构：分几层、每层放在哪个包里、层与层之间交换什么、按什么顺序做、每一步的门。各层的设计在各自的文档里；
此刻做到哪、产物在哪、关键数字，以[阶段记录](2026-09-24_two_tier_stage_notes.zh.md)为准。

---

## 0 状态

| 阶段 | 内容 | 状态 | 设计 |
|---|---|---|---|
| 1 词表 | 词的种类、含义、包络 | **定稿**：读法 `instruction-v3`，规格 `145d6911e75b` | [词表设计](2026-09-23_instruction_vocabulary_design.zh.md) |
| 2 标注器 | 从观测航迹读出句子，量档位，写出句子产物 | **完成**：现行句子产物 `v5_20260926` | 词表设计 §3、§7 |
| 3 执行器 | 只按动力学飞词的自动驾驶，只用词表 | **完成**：现行规格 `executor/v9_20260926` | [执行器设计](executor_design.zh.md) |
| 4 回放门 | 标注的句子交给执行器飞，能否在所指跑道落地 | 训练集**通过**（v9）；验证集最后一次在 v6 上通过 | 执行器设计 §11 |
| 5 先验 | 自回归模型：只用数据训练（teacher forcing），再让它自己说、执行器飞 | **完成**第 1 步（单机）：base 模型 `prior/v3_step1_20260924/full_s1337` | [先验设计](2026-09-24_prior_design.zh.md) |
| 6 后训练 | 在闭环里按落地强化；程序约束（下滑道下沿）作解码屏蔽 | 第一阶段**采用**，第二阶段**不采用** | [后训练设计](2026-09-25_post_training_design.zh.md) |
| 7 多机 | 以场景为单位：前机作上下文、间隔作生成时的筛选和屏蔽 | 没开始 | 先验设计 §3、§9 第 2–5 步 |

每个阶段结束都有一道门（§4），门没过不进下一阶段。

---

## 1 分层与数据流

```text
观测航迹（harvest arrivals，经 ts 数据平面：读时修复、MSL、2 s 重采样、切到入口）
   │
   ▼
[信号]  机场坐标系下的逐步状态：E、N、MSL 高度、航迹、地速、垂直速率、到各候选跑道的相对量
   │
   ▼
[标注器] 按词表规格把信号读成句子：每步六列词 + 每条指令的包络检查
   │
   ▼
句子产物（带规格 sha 与标注器源码指纹）─────────┐
   │                                          │
   ▼                                          ▼
[先验] 读：候选跑道（几何、落地情况）、         [回放门] 执行器按标注的句子飞，能否落地
      每步此刻知道的状态与生效的词
      写：下一步六列词
   │
   ▼ （生成时）
[执行器] 读当前状态和生效的词，按动力学飞一步，给出新状态 ──► 回到先验
   │
   ▼
[屏蔽] 解码时去掉违规的候选词（词表的相容规则；下滑道下沿）；飞出来的航迹事后再查
```

- **词是唯一的接口**。先验不输出控制量；执行器除词表外只读被指跑道公布的入口跨越高度 TCH（执行器设计 §2）。
- **句子产物是唯一的真值来源**。先验训练、回放门、读数都读同一份产物，不各自从航迹重读。
- **包络只有一套实现**（`instructions/envelope.py`）。标注器检查、执行器的判定、前端显示都调用它。
- **先验只看此刻以前的东西**：输入只给说这一步之前管制员能知道的；这一版不给机型（留"每架飞机的静态属性"接口，
  先验设计 §4.5）。

---

## 2 包结构

四个包与现有的 `data/`、`outputs/` 等并列，都在 `4dTrajectory/ts_transformer/` 下。

```text
instructions/                 第二层的"语言"。不 import torch。
  spec.py                     词表规格：六列、网格、类别、容差、标注参数、读法版本、sha
  words.py                    词的编码与解码：类别下标 ↔ 物理目标；"不变"
  grammar.py                  某一步能说哪些词：词表的相容规则（先验解码时作屏蔽）
  airport.py                  机场坐标系；候选跑道；到跑道的相对量
  signals.py                  一架航班 → 机场坐标系下的逐步信号
  piecewise.py                分段直线拟合（高度、速度共用）
  envelope.py                 每类指令的允许范围与包含检查
  labeller/                   records、lateral、vertical、speed、sentence、read（一架航班的总入口）
  measure.py                  档位的量测与拟合
  artefact.py                 产物读写；规格 sha 或标注器源码指纹不符就拒绝
  readout.py、figures.py      标注器的读数；单架航班的句子画在航迹上
  display.py、training_files.py  前端 Training 视图要画的包络，及它读的文件

autopilot/                    执行器：词 → 速率 → 控制量，按控制路径的点质量动力学每 1 s 飞一个周期
  executor.py                 一个周期：状态 → 生效的词 → 速率 → 控制量
  lateral.py / vertical.py / speed.py   三条律
  inverse.py、plant.py、frame.py        速率到控制量的精确反解；一个周期的动力学；状态按词的读法表示
  sentence.py、judge.py       每周期生效的词；执行器飞得对不对
  params.py、derive.py、spec.py         自身参数（只从词表推出）；规格文件（带源码指纹）
  flights.py、replay.py、runway_data.py 回放用的航班输入、批量回放、跑道公布的 TCH

prior/                        先验
  data.py、scene.py           句子产物作训练数据；场景与机场的落地情况（排除测试集运行日）
  model.py、train.py          网络；teacher forcing 训练、按落地强化的训练
  generate.py                 先验说话：按列顺序采样，相容规则作屏蔽（自由生成的闭环就在这里）
  readout.py、landing_reward.py         读数；落地奖励
  procedure.py、augment.py    下滑道下沿屏蔽；扩充起点（后训练第二阶段，未采用，代码保留）

experiments/                  runner（`run_ts.py <name>`）：instruction_* / executor_* / prior_*
```

**依赖方向**（由 `tests/test_architecture.py` 检查）：

- `instructions/` 不 import torch，只依赖数据平面中不含 torch 的模块、`io_utils`、`flight_scenarios`、`geokit` 等；
  包里只有 runner、`autopilot/`、`prior/` 使用它。
- `autopilot/` 依赖 `instructions/` 和控制路径共享的动力学（`outputs/dynamics`、`outputs/envelope`），不依赖任何模型、
  训练或 `prior/`；只有 runner 使用它。
- `prior/` 依赖 `instructions/`、日划分和 `data.runway_context`，**不依赖 `autopilot/`**；只有 runner 使用它。
- 闭环（先验说、执行器飞）不单独成包：由 runner 把 `prior.generate` 和 `autopilot.executor` 接起来
  （`prior_free_generation`、`prior_landing_reward`、`prior_augmented_reward`）。
- 包里任何模块都不 import runner。

---

## 3 产物与契约

一次标注写一个目录，已存在就拒绝，不覆盖：

```text
4dTrajectory/outputs/POOLED/instruction_language/<名字>/
  signals_{train,select,val}.npz        逐步信号（先验的状态输入读它）
  signals.json                          来源：manifest 与资格名单的摘要、划分、配置、被拒航班及原因
  spec.json                             词表规格（含量出的档值）、sha、标注器源码指纹
  measurements.json                     定档用的量测（只在训练集运行日上量）
  candidates.json                       各机场的候选跑道（清单里的 CIFP 跑道几何）
  sentences_{train,select,val}.npz      每步六列词 + 每条指令记录（逐行对应信号的前若干行）
  labels.json                           每架航班的状态（成功 / 拒绝原因）与逐架读数
  readout.json, readout.md              标注器的门的读数
```

执行器规格同样一次写一个目录（`outputs/POOLED/executor/<名字>/`：`spec.json`、`measurements.json`、回放读数），
带执行器源码指纹：`autopilot/` 和它直接 import 的仓库模块一改，现行代码就拒绝旧规格，要重写规格、重跑回放门。

- **规格的 sha** 覆盖词的定义、网格、类别、容差和读法版本；**标注器源码指纹**覆盖读句子的代码。改任何一项就是新版本。
  读产物的一方核对，不符就拒绝，不做兼容。
- **划分按运行日**（`data/day_split_20260924.json`，契约 C32）：一架航班的运行日 = 落地日（UTC − 9 h）；90 天分为测试 14 /
  验证 14 / 内部选择 9 / 训练 53。**测试集运行日一架都不打开**（不读信号、不标注、不进落地统计和场景）。
- **人群**：信号从 ts 数据平面的 `build_series` 得到，与模型训练看到的是同一批航班、同一种预处理。

---

## 4 各阶段的门

| 阶段 | 门 |
|---|---|
| 2 标注器 | 完整性（成功数、拒绝原因）；包络包含率与宽度成对；句子长度；类别使用。用户确认后冻结词表 |
| 3–4 执行器 | 标注的句子交给执行器飞：能否在所指跑道落地；词是否飞在包络内；evaluation 的判定 |
| 5 先验 | 每步负对数似然赢过"重复上一条"和"只看上一个词"两个基线；自由生成能否以落地结束 |
| 6 后训练 | 自由生成的落地率（直线 / 被引导分开报），护栏：teacher forcing 负对数似然、每架词数与标注之比 |
| 7 多机 | 间隔违规数；先量"看不看得到交互"（先验设计 §9 第 2 步） |

---

## 5 通用规则

- 单位只用米、米/秒、度、秒；管制原文的数值在引用处一次性换成米制常数。
- 机场特有的东西（候选跑道、程序几何）只进输入，不进词。
- 包含率与包络宽度一起报。
- 取值、选轮只用训练集和内部选择集；验证集每个阶段定稿后只读一次；每个机场的读数分开报。
