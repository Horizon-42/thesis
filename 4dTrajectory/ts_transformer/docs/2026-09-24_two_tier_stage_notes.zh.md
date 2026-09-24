# 两层模型：阶段记录（更新于 2026-09-24 23:40 UTC）

**用途**：压缩上下文之前的交接文档。写明此刻每个阶段做到了哪里、产物在哪、关键数字、用户做过的决定、正在进行的事、
接下来按什么顺序做。设计本身在各自的设计文档里，这里只给结论、指路和实现计划；历史看 git 和 `docs/CHANGELOG.md`。

路径都相对 `4dTrajectory/`（`outputs/…`）或 `4dTrajectory/ts_transformer/`（`docs/…`、包名），除非写明是仓库根目录。

---

## 0 状态

| 阶段 | 状态 | 在哪 |
|---|---|---|
| 1 词表 | **定稿**：读法 `instruction-v3`（航向词逐行标注、5° 一档、提前 4 s 说），按运行日划分后在新训练集上重新测量的规格 `145d6911e75b`（原 `0b4ea75be36d`；只差下降角档边界 ≤ 0.02°、最大加速度 1.7 → 1.4 m/s²） | `docs/2026-09-23_instruction_vocabulary_design.zh.md` |
| 2 标注器 | **完成**：现行句子产物 `outputs/POOLED/instruction_language/v4_20260924/`（按运行日划分：训练 44,362 架、内部选择 6,842、验证 10,633 已标注；测试日不打开） | 包 `instructions/` |
| 3 执行器 | **完成，只用词表**（第一、二阶段）：除词表外只读被指跑道公布的入口跨越高度 TCH。已合入 `dev-two-tier` | `docs/2026-09-23_executor_design.zh.md`；包 `autopilot/` |
| 4 回放门 | **新划分的训练集、验证集上每格都过**（`outputs/POOLED/executor/v6_20260924/`，规格 `0d6a68a92c6f`；验证集自己的动力学 9,246 架：落地 99.9 %、词 98.0 %、evaluation 98.6 %，执行器设计 §11） | |
| 5 先验 | 第一版、第二版训练完（第二版选中 V2d，val 每步 0.2542，读数文档 §2）；第三版设计已定；**第三版第 0 步完成**（划分、句子产物 `instruction_language/v4_20260924`、普查，读数文档 §3）；**第 1 步（单机先验）代码已提交（`dev-prior-v3` `47c7b790`），训练队列 23:28 UTC 起在跑**（§3.3） | `docs/2026-09-24_prior_design.zh.md`（现行设计）、`docs/2026-09-24_prior_readouts.zh.md`（每次训练的读数） |
| 前端 | Training 视图读 instruction-v3；五个机场的 `instruction_v3` 集已导出，**要用户重启 vite 才看得到**（§5） | `aeroviz-4d/public/data/airports/<ICAO>/training/instruction_v3/` |
| 6 后训练 / 7 约束 / 8 多机 | 未开始；多机按场景做已写进先验设计（§3） | 先验设计 §6、§9 |

---

## 1 产物

| 产物 | 路径 | 说明 |
|---|---|---|
| 句子产物（当前） | `outputs/POOLED/instruction_language/v4_20260924/` | 信号 `ts-instruction-signals-v3`（带 `entry_time_utc` / `landing_time_utc`，内含划分）、句子 `ts-instruction-sentences-v2`、规格 `145d6911e75b`；按运行日划分（`data/day_split_20260924.json`），代码 `67e0e5c9` |
| 句子产物（上一版） | `outputs/POOLED/instruction_language/v3_20260924/` | 按航班划分、信号 `ts-instruction-signals-v2`；新代码按格式名拒绝读它；第二版先验用它训练 |
| 第三版第 0 步普查 | `outputs/POOLED/prior/v3_census_20260924/census.json` | 读数文档 §3 |
| 执行器正式产物 | `outputs/POOLED/executor/v6_20260924/` | 规格 `0d6a68a92c6f`（与 v5 逐字节相同），`replay-train/`、`replay-val/`（新划分，每格都过门） |
| 执行器上一版产物 | `outputs/POOLED/executor/v5_20260924/` | 同一规格在旧句子产物上，`replay-train/`、`sensitivity-train-400-seed1337/` |
| 旧的执行器产物 | `outputs/POOLED/executor/{v2,v3,v4}_20260924/` | 已被 v5 取代，**留着不删**（用户 2026-09-24）。v2 的验证集回放门是早期的记录 |
| 先验第一版 | `outputs/POOLED/prior/v1_20260924/` | 读数在读数文档 §1 |
| 先验第二版 | `outputs/POOLED/prior/v2_20260924/` | 队列 22:24 UTC 跑完；选中 V2d（`choice.json`），读数文档 §2 |
| 前端 Training 集 | `aeroviz-4d/public/data/airports/<ICAO>/training/instruction_v3/sample.json` | `aeroviz-training-sample-v7`，每机场 40 架验证集航班（直线进近 / 被引导各 20），校验器从磁盘读 0 错误 |

---

## 2 执行器（结论；设计与推导在执行器设计文档）

- **只用词表**（用户的原则：执行器要简洁，要靠大量从数据挖的参数才能做成，就说明不能泛化）。所有转弯在词表的转弯率 0.5–4.7°/s
  与坡度上限 32° 内；词说出即执行，每 2 s 一行只在行首听一次；τ_ψ = 提前量 4 s；p = 32° ÷ 4 s = 8°/s；截获按
  √(0.5 × 4.7) ≈ 1.53°/s 规划；速度变化快慢 = 速度级差 ÷ 速度词最短保持时间 = 0.25 m/s²；落地越过被指跑道公布的 TCH
  （`autopilot/runway_data.py`，FAA CIFP；候选跑道没有 TCH 时标注器第一步就拒绝）；落地时留在生效那条词的管子里下降，直到从
  管子出发、在平飞与最陡一档下沿 3.73° 之间飞也够不到 TCH ± 12.5 m 才离开。固定的设计选择：周期 1 s、τ_γ 2 s、γ̇ 系数 2、
  超时系数 1.5。
- **正式训练集回放**（v5，每机场 400 架，自己的动力学 1,914 架，按航迹说词）：落地 99.9 %，evaluation 配对 98.7 %，词 98.0 %
  （航向 98.0 %、许可 98.9 %、高度 95.7 %、速度 100 %），按原话飞完 82.5 %；**机场 × 进近方式每格都过 95 % 的门**（词最低 KMSY
  被引导 96.5 %，evaluation 最低 KSTL 被引导 97.5 %）。航向词比第一阶段少 0.4 个点：减速 0.25 m/s² 比数据量出的 0.28 慢，飞机
  偏快、落地比观测早（中位数 −9 s），按航迹说词的时钟把航向词说得更密。敏感度：τ_ψ 2–6 s、γ̇ 系数 1–3、p 6–10°/s 都不敏感，
  p 4°/s 时词 96.8 %。
- **还没做**：验证集回放门（等新划分和用户同意）；执行器的 Experiments 类别发布（替代机型组禁用 A320 动力学）。审查留下的两点：
  复飞期间说的词执行器不理；只有第一次截获结束词的判定窗口——只有先验生成的句子才会碰到，自由生成时处理。

---

## 3 先验

### 3.1 第一版、第二版（读数文档 §1、§2）

- **第一版**（按航班划分，句子产物 v2）：验证集每步负对数似然 0.1778，两个基线 0.3444 / 0.3260；第 0 步跑道前 1 为 80.8 %。
- **第二版**（按航班划分，句子产物 `v3_20260924`；代码在分支 `dev-prior-v2`，工作树 `.claude/worktrees/prior-v2`，`feeb7ce3`）：
  比较五种输入变体（V0 用拟合速度，只量"先验靠未来信息的程度"；V1 只给位置；V1d 加方向差；V2 加过去的运动；V2d 两者都加），
  规则在读数文档 §2（跑之前写定）。**队列正在跑**：进程号在 `outputs/POOLED/prior/v2_20260924/queue.pid`（1019668），日志
  `queue.log`；顺序 V1 → V1d → V2 → V2d（种子 1337）→ 领先者种子 2024 → V0 → `prior_select`（选择 + 一次验证集读数）；最后一行
  `QUEUE DONE`，预计 23:00–23:30 UTC。完成前不要动 `.claude/worktrees/prior-v2`（每次训练记录了提交，`prior_select` 拒绝有改动的
  工作树）。V1 已有的读数：内部选择集 5,022 架，每步 0.2752（基线 0.8432 / 0.5598），第 0 步跑道前 1 为 81.4 %（交接时未许可的
  75.7 %，按机场最常用跑道猜 50.5 %；已许可的 98.0 %）。
- 第二版的数不当第三版的基线（划分、第 0 步的条件、结构都不同），但它的选择结果决定第三版给不给"过去的运动"和"方向差"。

### 3.2 第三版设计（用户 2026-09-24 确认；全文在先验设计文档）

- **原则**：输入只给说这一步之前管制员能知道的东西；标注器读出的词只当预测目标；以场景（同一机场、同一时刻的一组飞机）为单位；
  按运行日划分，测试集按整天另封。
- **数据**：句子前 `N_look` = 8 行（16 s）只观察；第一个预测步六列都要说；每条候选跑道的向量含几何、相对位置、以及此前
  `T_cfg` = 30 分钟内这条跑道上的落地架数和距上次落地的时间（取 harvest 航迹清单里已分配跑道的落地，排除测试集运行日）；
  已说的词 = 每列当前生效的词 + 距上次换词的步数 + 生效跑道的向量；每架飞机的静态属性留接口、本版为空（**不加机型**，用户：
  航迹本身已隐含机型信息）；不给钟点和风。
- **模型**：每层 = 沿时间的因果注意力 → 同一步各架飞机之间的注意力（全连接图上的图注意力，不另接图网络框架；边特征既作每个
  头的偏置、也加到被读取的值上，相对位置用本机坐标系）→ 前馈；d 192、4 层、6 头。同一步内各列按 跑道 → 进近 → 航向 → 高度 →
  下降角 → 速度 的顺序输出，后列看前列刚选的值；各架飞机同时说。
- **划分**：所有机场同一批运行日（`data/runway_context.operational_day`，UTC − 9 h），现有 90 天；测试 15 % / 验证 15 % / 训练 70 %
  （训练的天里 1/7 作内部选择集），种子 1337。测试集运行日的全部航班在两层模型这条线上完全不打开；原来按航班抽的测试集留给 ts 的
  其他模型；跨模型比较只用"新测试集运行日里、原来也在测试集"的航班。
- **分步与通过标准**：0 准备（量、重测规格、重建句子产物）→ 1 单机 → 2 M1（在 25 km 范围里看不看得到交互）→ 3 前机一条边 → 4 完整
  场景 → 5 多机闭环与后训练。每步的门见设计 §9。后训练的顺序（文献依据，设计 §7、§11）：闭环监督微调（CAT-K）→ 间隔作生成时的
  筛选与屏蔽（按落地顺序在同一步里算，门槛取规章最小间隔，以数据里本来的低于最小间隔比例作参照）→ 最后才考虑强化学习。

### 3.3 实现计划（下一步就做这个；先写代码和测试，每个里程碑一次 opus 代码审查，改完再提交）

**分支与工作树**：代码在 `.claude/worktrees/prior-v3`（分支 `dev-prior-v3`）里做，它要先建在"合并了第二版代码的 `dev-two-tier`"上：
第二版队列跑完 → 第二版读数写进读数文档 §2（直接提交 `dev-two-tier`）→ `dev-prior-v2` 变基到 `dev-two-tier`（冲突只在先验设计
文档：取 `dev-two-tier` 的版本，它的 §8.6、§8.7 内容已经挪进读数文档）→ 请用户合并（或得到同意后我合）→ 把 `dev-prior-v3` 重建在
合并后的 `dev-two-tier` 上。**设计文档直接提交到 `dev-two-tier`**（用户 2026-09-24）。

**第 0 步：划分、信号、规格、句子产物**（代码已写完并提交：`dev-prior-v3` `67e0e5c9`，两次 opus 审查已改完，ts 全套 1,392 通过；
与下面计划不同的两处：一架航班的运行日取**落地时刻**，不取进入时刻——航迹清单只有落地时刻，同一架航班在两份清单里要落在同一天，
两种取法不同的只有 20 架；按天数取整的分法**只做一次并提交进仓库**（`data/day_split_20260924.json`，14 / 14 / 9 / 53 天），
harvest 的运行日与它不一致时 runner 拒绝。信号里的绝对时间叫 `entry_time_utc`（不是 `start_time_utc`：那是航迹文件第一次收到信号
的时刻，早约 45 s）。**正式链**：脚本 `$CLAUDE_JOB_DIR/tmp/formal/v4_step0.sh`（进程号 `v4_step0.pid`，日志 `v4_step0.log`），
2026-09-24 22:22 UTC 从干净的 `dev-prior-v3` 工作树启动，依次写 `outputs/POOLED/instruction_language/v4_20260924/`（信号 → 规格 →
句子）、`outputs/POOLED/prior/v3_census_20260924/census.json`、`outputs/POOLED/executor/v6_20260924/`（规格、`replay-train`、
`replay-val`）。）

1. 新模块 `data/day_split.py`（不依赖 torch）：输入运行日列表和种子，按 `sha256(f"{seed}:{day}")` 排序，取前 round(0.15 N) 天为测试、
   再 round(0.15 N) 天为验证、其余里 round(剩余 / 7) 天为内部选择集、剩下为训练——按天数取整，不用阈值（90 天时份额才准）。每架航班
   的运行日取它的 `entry_time_utc`（到达清单）。产物里记下划分方式、种子、每一份的日期列表。
2. `experiments/instruction_signals.py` 改用按运行日的划分（替换 `flight_keys_by_split`）：分成 train / select / val 三份写信号，测试集
   运行日的航班不读。`instructions/artefact.py` 的 `SPLITS` 相应改（它不在标注器源码指纹里）。信号的每架航班记下 `start_time_utc`
   （场景要用绝对时间）→ `instructions/signals.py` 的格式名升为 `ts-instruction-signals-v3`（`signals.py` 在标注器指纹里，指纹会变，
   反正整份产物要重建）。所有钉住格式名的地方一起改，不留旧名。
3. 在新训练集（不含内部选择集）上重新测量词表规格（`instruction_spec`）→ 新规格 sha → `instruction_labels` 标注三份 → 新产物目录
   （例如 `outputs/POOLED/instruction_language/v4_<日期>/`）。只在干净的工作树上跑。
4. 执行器：对新产物重新写规格（`executor_spec`，几秒）、训练集回放（几分钟），确认门仍过。前端 `TRAINING_SPEC_SHA256`
   （`aeroviz-4d/src/data/trainingSample.ts:51`）改成新规格，重新导出 Training 集——交给 agent。
5. 新 runner `experiments/prior_scene_census.py`（只读训练集运行日）量设计 §9 第 0 步的各项：第 8 行时已建立在五边上的比例；此前 30 分钟
   有落地的步的比例；各份的天数与航班数、原测试集航班落在新测试集运行日里的架数；每步在场的说话飞机 / 背景飞机架数、有前机的步的
   比例；时间上串成段的长度。结果写进设计 §11 或读数文档。

**第 1 步：单机先验（在第二版 `prior/` 的基础上改）**——**已实现并提交**（`47c7b790`；两次 opus 审查已改完，ts 全套 1,403 通过，
前端 vitest 712 通过）。与下面计划不同或补充的地方：变体是 full / no-context（候选跑道不带落地情况）/ unordered（各列不按顺序），规则
写在读数文档 §4（跑之前写定）；每架航班的落地情况去掉它自己的落地（航迹清单的落地时刻取整到秒，句子最后几行可能在它之后）；
时间注意力改用 `scaled_dot_product_attention`，每一步总能看到自己（审查发现：整行被屏蔽时有的实现返回 NaN，多机场景里会传给在场
的飞机）；"已建立在五边上"用标注器的截获行 ≤ `N_look`；前端叠加层与导出一起升为 `aeroviz-training-prior-v3`（只在预测的步上有值）。
**训练队列**：`outputs/POOLED/prior/v3_step1_20260924/`（`queue.sh`，进程号 `queue.pid`，日志 `queue.log`）：full（1337）→ no-context →
unordered → full（2024）→ `prior_select`（选择 + 一次验证集读数），每个约 40 分钟以内（全训练集每 epoch 约 80 s）。跑完前不要改
`.claude/worktrees/prior-v3`（`prior_select` 拒绝与训练时不同的代码）。

- `prior/data.py`：去掉 `GIVEN_AT_HANDOVER`（第一个预测步六列都预测）；每架航班从第 `N_look` 行起有目标；输入里已说的词整体往后错一步
  （第 t 步看到的是到 t−1 为止生效的词，第一个预测步之前是"尚未说过"）；候选跑道特征加两项落地情况（此前 30 分钟架数、距上次落地
  时间 + 无记录标志）；每架飞机的静态属性宽度 0；保留第二版选中的运动 / 方向差输入。落地情况从 harvest 航迹清单的已分配落地里按
  时间二分查找（可复用 `data/runway_context.py` 的 `ContextLanding` / `build_airport_context` 的取法，排除测试集运行日）。
- `prior/model.py`：加一根"飞机"轴（批 × 飞机 × 步）；每层 = 时间因果注意力 → 飞机之间的注意力（边特征 → 小网络 → 每头一个偏置；第 1 步
  只有一架，边特征为空）→ 前馈；六个头按顺序，头 k 的输入 = 隐状态 + 前面各列这一步所选值的嵌入之和（训练时用真值）；第一个预测步
  屏蔽"不变"。检查点格式名 `ts-prior-checkpoint-v3`。
- `prior/train.py`、`prior/readout.py`：损失只算预测的步；读数按设计 §8——跑道分方向层 / 左右层（`runway_context.direction_groups`）、
  分"第一个预测步时进近词为许可加入"与否，与 B0 / B1 / B3 比。
- 测试（`tests/test_prior.py`）：只用过去——改动 t 之后的位置、t 之后的落地，第 t 步的输出不变；调换候选跑道顺序只调换跑道分数；
  读到测试集运行日的航班就拒绝；按顺序输出的头在 teacher forcing 下与逐列计算一致；第一个预测步没有"不变"。
- runner：`prior_train`（`--variant`）、`prior_select`；比较规则（落地情况有 / 无、各列顺序有 / 无）在跑之前写进读数文档 §3。
- 第 1 步读完后：先验与执行器接起来做单机自由生成（框架门的第三项），再按门走 §9 第 2 步。

---

## 4 代码、分支、工作树（2026-09-24 23:40 UTC）

- `dev-two-tier`（主检出 `/home/supercomputing/studys/thesis`）：只加了文档提交（本条记录所在的提交）。含词表第三版、只用词表的执行器、
  新的前端 Training 视图、先验第三版设计与读数。
- `dev-prior-v2`（`.claude/worktrees/prior-v2`）：第二版代码，已变基到 `dev-two-tier`（`56f5cec6`，3 个提交），ts 全套 1,384 通过。
  它整个包含在 `dev-prior-v3` 里，不用单独合并。
- `dev-prior-v3`（`.claude/worktrees/prior-v3`，`47c7b790`）= `dev-two-tier` 的 `a5dff836` + 第二版 3 个提交 + 第 0 步 `297c3e10` + 测试夹具
  `a9f4caab` + 第 1 步 `47c7b790`。**等用户合并**（队列跑完之后；`dev-two-tier` 在它之后只多了文档提交，合并时先把 `dev-prior-v3` 变基到
  `dev-two-tier` 上，或用合并提交）。数据目录已软链到主检出。
- 变基之前的提交留了标签，产物里记的 git head 仍能找到：`runs/prior-v2-feeb7ce3`（第二版队列的代码）、`runs/prior-v3-step0-67e0e5c9`
  （第 0 步正式链的代码）。
- 其他工作树（`manoeuvre-runs`、`two-tier-pub`、`two-tier-runs`，游离 HEAD）是更早的，不动。

---

## 5 前端

- `instruction_v3` Training 集已导出到五个机场（每机场 40 架，`aeroviz-training-sample-v7`），校验器从磁盘读 0 错误；在另起的 vite 上逐个
  机场打开过，渲染正常、控制台无错误。
- **正在运行的 5173 前端看不到它**：vite 启动后不会发现 `public/data` 下新加的文件，要重启。agent 重启时被权限检查拦下，已告诉用户：
  杀掉 vite 进程（supervisor 只重启前端），或 `./start_aeroviz_fullstack.sh --replace`。后端合并后也要重启（不热更新）。
- 执行器、先验的叠加层目前只画在 `instruction_v2` 上；第三版产物出来后重新导出。

---

## 6 用户的决定与规矩（2026-09-24）

- **执行器只用词表**；p = 32° ÷ 4 s；截获按几何平均 1.53°/s 规划；速度快慢 0.25 m/s²；落地用跑道公布的 TCH 并改飞法；候选跑道在列入之前
  要确保有 TCH。
- **先验训练始终第一优先**，执行器的问题不能拖住它。
- **先验第三版**：输入只给那一刻知道的；测试集按运行日另封（选 c）；多机按场景、分解注意力、交互在闭环后训练里学；§10 的值全部按建议
  定下；机型不加、留接口。
- **设计文档只写最终设计**，历史交给 git；**设计文档直接提交到 `dev-two-tier`**；代码走分支 + 审查。
- 说人话：不用缩写和中英夹杂的造词（例如"在新 train 天上重量"应写"用新训练集的航班重新测量"）。
- 验证集：新划分出来后，用户同意在新验证集上跑（执行器回放门、先验读数）；**底线是测试集运行日的数据在训练阶段一点都不碰**
  （不读信号、不标注、不进落地统计和场景、不参与任何取值）。旧执行器产物 v2–v4 留着不删。
- 写共享 outputs、删产物、重启服务都要用户同意；`兼容`是禁用词（格式一变就换名字，不留旧名）。

---

## 7 下一步（按优先）

1. ~~第二版队列~~：跑完，选中 V2d，读数文档 §2。~~第三版第 0 步~~：完成，读数文档 §3。
2. **第三版第 1 步训练队列**（在跑）：跑完后把读数写进读数文档 §4（三个变体、种子线、选择、验证集读数：每列、跑道分方向 / 左右、
   已建立 / 未建立、与 B0 / B1 / B3 比），提交到 `dev-two-tier`；然后请用户合并 `dev-prior-v3`。
3. 导出与前端的审查意见（第二次审查，只读）改在 `dev-prior-v3` 上——队列跑完之后再改（`prior_select` 要求代码与训练时一致）。
4. 第 1 步读完：先验与执行器接起来做单机自由生成（框架门的第三项），再按门走 §9 第 2 步（M1：交互看不看得到）。
4. ~~文献~~：已提交，要点写进了先验设计 §6、§7、§8、§11。
5. 等用户：重启 vite。（执行器验证集回放门在新划分出来后直接跑，用户已同意。）
6. 之后：§9 第 2–5 步（交互、多机、闭环与后训练）；先验交叉验证；替代机型的动力学（用户在 `aircraft/` 处理）。

---

## 8 怎么跑（仓库根目录，`conda activate aeroviz`，`PYTHONPATH=.:4dTrajectory`）

```bash
I=4dTrajectory/outputs/POOLED/instruction_language/<句子产物>
E=4dTrajectory/outputs/POOLED/executor/<新名字>
python run_ts.py executor_spec --instructions $I --dir $E --word-clock track                  # 干净的工作树，几秒
python run_ts.py executor_replay --instructions $I --executor $E --split train --per-airport 400 --out $E/replay-train
python run_ts.py executor_sensitivity --instructions $I --executor $E --per-airport 400
python run_ts.py instruction_training_export --dir $I --airports-root aeroviz-4d/public/data/airports \
    --airport KMSY --airport KRDU --airport KSJC --airport KSMF --airport KSTL                  # 写 public/data，要用户同意
```

- 长任务用 `nohup setsid <脚本>` 分离运行，脚本里 `echo $$ > pid`，用 Monitor 按进程号盯（30 分钟到期要重新挂）；脚本文件要 `chmod +x`；
  和先验训练同机时限制线程（`OMP_NUM_THREADS=6`）。执行器训练集链（规格 + 回放 + 敏感度）约 4 分钟；先验一个变体约 25–30 分钟。
- ts 全套测试在前台跑（600 s 超时，约 9 分钟）：`python -m pytest 4dTrajectory/ts_transformer/tests -q --import-mode=importlib -p no:cacheprovider`。
