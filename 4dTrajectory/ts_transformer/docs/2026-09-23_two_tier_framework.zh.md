# 两层模型的总体框架（2026-09-23）

**一句话**：上层是"说话的人"——一个自回归模型，每个时间步决定对这架飞机下什么指令；下层是"飞的人"
——一个只按动力学执行指令的自动驾驶。两层之间只通过**词**交流，词的定义见
[指令词表设计](2026-09-23_instruction_vocabulary_design.zh.md)。本文定整体结构：分几层、每层放在哪个
包里、层与层之间交换什么、按什么顺序做。

---

## 0 状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 词表 | 词的种类、含义、包络 | **已定**：读法 `instruction-v2`，规格 `103a6eae6b90`（2026-09-23；转弯按转弯率、落地与 harvest / evaluation 同条件） |
| 2 标注器 | 从观测航迹读出句子，量档位，写出句子产物 | **完成**（产物 `v2_20260924`，读数文档；前端 Training 集合 `instruction_v2`） |
| 3 执行器 | 只按动力学飞词的自动驾驶 | **设计中**（`2026-09-23_executor_design.zh.md`） |
| 4 回放门 | 真值句子交给执行器，能否飞到跑道 | 等执行器 |
| 5 先验 | 自回归模型：预训练（教师强制） | 等回放门通过 |
| 6 后训练 | 闭环再训、目标导向解码 | 等先验 |
| 7 约束 | 程序与管制规则的解码屏蔽；气压修正 | 等后训练 |
| 8 多机 | 前机句子作上下文、间隔屏蔽 | 最后 |

每个阶段结束都有一道门，门没过不进下一阶段。

---

## 1 分层与数据流

```text
观测航迹（harvest arrivals，经 ts 数据平面：读时修复、MSL、2 s 重采样、切到入口）
   │
   ▼
[信号]  机场坐标系下的逐步状态：E、N、MSL 高度、航迹、地速、垂直速率、到着陆跑道的相对量
   │
   ▼
[标注器] 按词表规格把信号读成句子：每步六列词 + 每条指令的包络检查
   │
   ▼
句子产物（带规格 sha）──────────────┐
   │                               │
   ▼                               ▼
[先验] 读：上下文（机型、候选跑道几何）  [回放门] 执行器按真值句子飞，能否落地
      + 每步（状态、生效的词）
      写：下一步六列词
   │
   ▼ （生成时）
[执行器] 读当前状态和生效的词，按动力学飞 Δt，给出新状态 ──► 回到先验
   │
   ▼
[约束] 解码时屏蔽违规候选；事后按程序评估飞出的航迹
```

- **词是唯一的接口**。先验不输出控制量，执行器不读任何程序数据。
- **句子产物是唯一的真值来源**。先验训练、回放门、读数都读同一份产物，不各自从航迹重读。
- **包络只有一套实现**。标注器检查、执行器约束、显示都调用它。

---

## 2 包结构

新的一组放在 `4dTrajectory/ts_transformer/instructions/`，与现有的 `data/`、`outputs/` 等并列。

```text
instructions/                 第二层的"语言"：词表、信号、包络、标注器、产物。不 import torch。
  spec.py                     词表规格：六列、网格、类别、容差、标注参数、读法版本、sha
  words.py                    词的编码与解码：类别下标 ↔ 物理目标；"不变"
  airport.py                  机场坐标系；候选跑道（入口、真航向、高程、长度）；到跑道的相对量
  signals.py                  一架航班 → 机场坐标系下的逐步信号
  piecewise.py                分段直线拟合（高度、速度共用）
  envelope.py                 每类指令的允许范围与包含检查
  labeller/
    records.py                指令记录与拒绝（带原因类别）
    lateral.py                航向词、大转弯拆分、截获与"许可加入"
    vertical.py               高度词与下降角词
    speed.py                  速度词与"未指定"
    sentence.py               指令 → 每步六列；第 0 步；相容规则
    read.py                   一架航班的总入口：落地处截断、平滑、各列读法、拼句，或带原因的拒绝
  measure.py                  档位的量测与拟合（下降角档、容差、走廊）
  artefact.py                 产物读写；规格 sha 不符时拒绝
  readout.py                  读数：完整性、包络包含率与宽度、句子长度、类别使用
  figures.py                  单架航班的句子画在航迹上，供目视核对

autopilot/                    （阶段 3）执行器：航向、高度与下降角、速度、航线跟踪四种模式
prior/                        （阶段 5）先验：数据集、模型（五个分类头 + 跑道指针头）、训练、解码
closed_loop/                  （阶段 5–6）先验与执行器交替：生成、闭环再训、目标导向解码
constraints/                  （阶段 7）程序与管制规则的屏蔽

experiments/                  runner（`run_ts.py <name>`）
  instruction_signals.py      读开发集（train + val），写信号产物
  instruction_spec.py         在 train 信号上量档位，写规格
  instruction_labels.py       按规格标注 train + val，写句子产物与读数
  instruction_figures.py      抽样画出 val 航班的句子，供目视核对
```

**依赖方向**（由 `tests/test_architecture.py` 检查）：

- `instructions/` 只依赖数据平面中不含 torch 的模块（`data.channels`、`data.coordinate_frames`）、
  `io_utils`、`flight_scenarios`、`aerodynamic_model.common`、`geokit`、`numpy`；不依赖 `outputs/`、
  `training/`、`experiments/`、`cli/`，也不 import torch（`tests/test_architecture.py` 检查）。在执行器和
  先验出现之前，包里只有 runner 使用它。
- 以后的 `autopilot/` 依赖 `instructions/`（词、包络）和动力学后端，不依赖 `prior/`。
- `prior/` 依赖 `instructions/`（词、产物），不依赖 `autopilot/`。
- `closed_loop/` 在最上面，同时依赖 `prior/` 与 `autopilot/`。
- runner 只是使用者，包里任何模块都不 import runner。

---

## 3 产物与契约

一次标注写一个目录，已存在就拒绝，不覆盖：

```text
4dTrajectory/outputs/POOLED/instruction_language/<名字>/
  signals_train.npz, signals_val.npz   逐步信号（先验的状态输入就读它）
  signals.json                        来源：manifest 与资格名单的摘要、划分、配置、被拒航班及原因
  spec.json                           词表规格（含量出的档值）与 sha
  measurements.json                   定档用的量测
  candidates.json                     各机场的候选跑道（清单里的 CIFP 跑道几何）
  sentences_train.npz, sentences_val.npz   每步六列词 + 每条指令记录（逐行对应信号的前若干行）
  labels.json                         每架航班的状态（成功 / 拒绝原因）与逐架读数
  readout.json, readout.md            词表的门的四项读数
  figures/                            目视核对的抽样图
```

- **规格的 sha** 覆盖词的定义、网格、类别、容差和读法版本。改任何一项就是新版本、新 sha。
  读产物的一方核对 sha，不符就拒绝，不做兼容。
- **划分**：只读 train 与 val（`data.splits` 的逐航班划分），test 的航迹不打开。
- **人群**：信号从 ts 数据平面的 `build_series` 得到，与模型训练看到的是同一批航班、同一种预处理。

---

## 4 各阶段的门

| 阶段 | 门 |
|---|---|
| 2 标注器 | 完整性（成功数、拒绝原因）；包络包含率与宽度成对；句子长度；类别使用。用户确认后冻结词表 |
| 3–4 执行器 | 真值句子交给执行器飞，能否在所指跑道落地；与观测航迹的误差；可飞性 |
| 5 先验 | 困惑度、前 K 覆盖；自由生成能否以落地结束；赢过"重复上一条"和"只看上一个词" |
| 6 后训练 | 两层接起来飞：落地率、雷达引导组误差，对照无指令基线 |
| 7 约束 | 屏蔽后的可解率与违规率 |
| 8 多机 | 间隔违规数；先量不建图的上限 |

---

## 5 通用规则

- 单位只用米、米/秒、度、秒；管制原文的数值在引用处一次性换成米制常数。
- 机场特有的东西（候选跑道、将来的程序几何）只进输入，不进词。
- 包含率与包络宽度一起报。
- 读开发集时只用 train 与 val；每个机场的读数分开报。
