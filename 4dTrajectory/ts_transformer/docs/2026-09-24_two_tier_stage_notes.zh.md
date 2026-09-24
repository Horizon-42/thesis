# 两层模型：阶段记录（2026-09-24）

**用途**：压缩上下文之前的交接文档。写明到这一刻为止每个阶段做到了哪、产物在哪、关键数字、用户做过的决定、
还在进行的事、下一步的优先顺序。细节在各自的设计文档里，这里只给结论和指路。

---

## 0 状态

| 阶段 | 状态 | 在哪 |
|---|---|---|
| 1 词表 | **定稿**：读法 `instruction-v2`，规格 `103a6eae6b90` | `docs/2026-09-23_instruction_vocabulary_design.zh.md` |
| 2 标注器 | **完成**：产物 `v2_20260924`（train 50,214 架、val 10,540 架已标注；标注器源码 `47c6008a89bd`） | `outputs/POOLED/instruction_language/v2_20260924/` |
| 3 执行器 | **完成并合入 `dev-two-tier`**（合并 `dc717045`），三轮 opus 代码审查的发现都已改 | `docs/2026-09-23_executor_design.zh.md`；包 `autopilot/` |
| 4 回放门 | **val 已跑**（规格 `2674ab8c71a9`）：落地 98.6 %、evaluation 配对 96.7 %、词 94.2 %；按格未全过（§2.3） | `outputs/POOLED/executor/v2_20260924/` |
| 5 先验 | **第一次训练完成**（没有交叉验证）：val 每步负对数似然 0.1778，两个基线 0.3444 / 0.3260 | `docs/2026-09-24_prior_design.zh.md`；包 `prior/`；`outputs/POOLED/prior/v1_20260924/` |
| 5 发布到前端 | **进行中**：opus agent，分支 `dev-publish-executor-prior`（§5） | `.claude/worktrees/publish-executor-prior/` |
| 6 后训练 | 未开始 | |
| 7 约束 / 8 多机 | 未开始 | |

路径都相对 `4dTrajectory/`（`outputs/…`）或 `4dTrajectory/ts_transformer/`（`docs/…`、包名）。

---

## 1 产物

| 产物 | 路径 | 内容 | 由谁写 |
|---|---|---|---|
| 句子产物 | `outputs/POOLED/instruction_language/v2_20260924/` | 信号、句子、候选跑道、规格 | `instruction_{signals,spec,labels}` |
| 执行器规格 | `outputs/POOLED/executor/v2_20260924/spec.json`、`measurements.json` | 格式 `ts-executor-spec-v2`，sha `2674ab8c71a9`，在干净的工作树 `1c63eb3e` 上量 | `run_ts.py executor_spec --word-clock track` |
| train 敏感度 | `…/executor/v2_20260924/sensitivity-train-400-seed1337/` | 每个变动一个文件 + `sensitivity.json` | `run_ts.py executor_sensitivity` |
| val 回放门 | `…/executor/v2_20260924/replay-val/` | `replay.json`（`ts-executor-replay-v1`，逐架 + 门表）、`records/<ICAO>/`（控制路径预测记录 + evaluation 报告） | `run_ts.py executor_replay --split val` |
| 先验 | `outputs/POOLED/prior/v1_20260924/` | `checkpoint.pt`（`ts-prior-checkpoint-v1`）、`config.json`、`history.json`、`readout.json` | `run_ts.py prior_train`（`11888357`，干净） |

执行器规格的参数：r_turn 2.15°/s、φ_cap 25°、τ_ψ 4.5 s、p 2.0°/s、τ_γ 2 s、γ̇ 系数 2、a_dec 0.27、a_acc 0.20、a_unspec
0.30 m/s²、落地瞄准 20.8 m、越过窗口 8.4–36.5 m、延迟（航向 / 纵向 / 速度）1 / 0 / 0 s、超时系数 1.5、说词的时钟 `track`。

旧的 `executor/v1_20260924`（旧格式、旧代码量的）已按用户同意删除。

---

## 2 执行器

### 2.1 这一轮定下的做法（设计文档里都有）

- **词在观测飞机听到它的位置说**（`autopilot/sentence.py`：`TrackClock`，往前最多看 60 s、只进不退、每周期最多前进一行；
  过了观测航迹的尽头按执行器自己的节奏往下走）。原因：一条词只给目标、不给快慢，按观测的时刻说，执行器按自己的节奏飞时
  后面的词就落错地方——那时 train 正式敏感度落地只有 82.5 %。另有 `distance`（按路程）和 `time`（原时刻，对照）。
- **执行器自己切入**：许可后按词的航向偏 4.5° 也碰不到航线时，按 30° 切入；偏离这条航向词超过容差就判这条词不合格
  （模式 `intercepting_off_word`）。
- **落地的瞄准**：截获后瞄准"词的高度管子投到入口"与"观测越过高度窗口（train p5–p95）"的交集；没有交集就去窗口近的
  一端（离开管子，模式 `aim_left_tube`，照实判）。试过"留在管子边上直到 100 m"：evaluation 从 96.5 % 掉到 94.1 %，退回。
- **"未指定"速度**在入口前减到本机型公开的进近速度（替代机型的航班不按 A320 的质量换算）。
- **判定**：每条词在执行器听到它的那一行判；从没说出的词算"没到"；同一列同一行的两条词，前一条算"被取代"。

### 2.2 train 读数（每机场 400 架，自己的动力学 1,581 架，按 `track` 说词）

落地 98.2 %、evaluation 配对 96.5 %、词 94.4 %。敏感度（同一规格）：没有一项可取的变动在落地、按原话飞出、词上同时
比规格好（规格 98.3 % / 70.1 % / 94.3 %）——办法 A、B 定的值就够了。

### 2.3 val 回放门（自己的动力学 7,773 架；替代机型 2,593 架只报）

| | 落地 | 词在包络里 | evaluation 配对 |
|---|---|---|---|
| 全部 | 98.6 % | 94.2 % | 96.7 % |
| 直线进近 4,978 | 100 % | 93.0 % | 98.9 % |
| 雷达引导 2,795 | 96.3 % | 95.1 % | 92.8 % |

按机场 × 进近方式 15 格（门 ≥ 95 %）：落地 14 格过（KSMF 雷达引导 90.3 %）；evaluation 12 格过（KSMF / KSJC / KSTL 雷达引导
80.7 / 93.8 / 94.2 %）；词多数格 93–95 %，只 3 格过。按列：高度词 83.7 %（直线）/ 88.9 %（雷达引导），航向约 95.5 %，
进近 99.0 % / 92.8 %，速度 100 %。替代机型组：落地 89.6 %，evaluation 31.5 %。

### 2.4 发现（都记在执行器设计文档里）

1. **词不说转多急**：回放别人的句子时转弯半径不同，150° 的大转弯到五边前就差出几公里（KSMF 17R 最明显）——雷达引导格
   没过的主因。闭环里先验看执行器自己的状态说词，能改航向；也可以考虑把转弯率放进词表（要用户决定）。
2. **高度词与落地窗口冲突**：落地段的管子投到入口时常与观测越过高度窗口对不上，执行器按窗口落地就出了管子——词的门
   没过的主因。
3. **标注器读动作开始晚**（办法 B，按时刻说词时量）：高度 / 角度词晚 4–6 s，速度词晚 8–9 s，航向早约 2 s。延迟不能取负。
4. **没有风**："未指定"之后执行器地速比观测快约 3.4 m/s（中位数），直线进近落地早约 10 s（按时刻说词时量）。
5. **替代机型**：23 % 的航班用 A320 动力学代替，evaluation 只 31.5 %（用户在 `aircraft/` 处理）。

---

## 3 先验

- 数据：句子产物 train（教师强制）、val（早停和读数）；每步的状态、相对生效跑道的量（第 0 步为 0 + 标志位）、生效的词
  和距上次换词的步数、机场与候选跑道；目标是这一步的六列词（0 = 不变）。
- 模型：因果 Transformer，d 192、4 层、6 头，230 万参数；跑道头只在机场的候选上 softmax。
- 训练：AdamW 3e-4，预热 500 步，按长度分桶每批约 16,000 步，val 负对数似然早停（耐心 3）。28 个 epoch，第 25 个最好，
  每个约 58 s（RTX 4060）。
- 读数（val）：每步 0.1778（困惑度 1.19），"重复上一条" 0.3444，"只看上一个词" 0.3260，六列每列都更低。取值前 5 覆盖
  83–100 %；"什么时候换词"最难（换词的步上给"换"的概率 0.44–0.82）；跑道指针前 1 为 80.8 %（平行跑道分不清）。
- 还没做：和执行器接起来自由生成（先验门第三项）、交叉验证、加机型。

---

## 4 代码与提交（都在 `dev-two-tier`）

- 合并 `dc717045`（`dev-executor` → `dev-two-tier`，38 个文件；分支和工作树已删）；读数文档 `0e842e2d`。
- 包：`ts_transformer/autopilot/`（frame、sentence、flights、plant、inverse、params、lateral、vertical、speed、executor、judge、
  replay、measure、derive、observe、spec）；`ts_transformer/prior/`（data、model、train、readout）。
- runner：`executor_spec`、`executor_sensitivity`、`executor_replay`（R12）；`prior_train`（参考文档里还没有 R-ID）。
- 测试：`tests/test_autopilot.py`、`tests/test_prior.py`、`tests/test_architecture.py`（先验只能 import 词表包；执行器的
  import 白名单）；合并后这几个加训练导出共 89 个测试通过。全 ts 测试集最后一次全跑是在当天较早的提交上（1,324 通过），
  之后的改动没有再全跑——下次动 ts 包之前先全跑一次（前台、600 s 超时，约 8 分钟）。
- 执行器源码 sha 覆盖包内除 spec.py 外的全部模块和它们直接 import 的仓库内模块；改任何一个，旧规格就被拒绝，要重新量。

---

## 5 进行中：发布到前端（opus agent）

- 用户要求（2026-09-24）：把执行器的结果和先验训练的结果发布到前端；**写或完善发布脚本**，不要每次让 agent 手动发；
  前端没有载入和显示就一起做。
- agent 在新工作树 `.claude/worktrees/publish-executor-prior`、新分支 `dev-publish-executor-prior`（基于 `dev-two-tier`）上做，
  不合并；完成后报告：分支、提交、能否快进、发布了什么到哪、重跑的命令、测试、前端核对。
- **agent 报告之后要做的**：代码交 opus 审查（只审代码）→ 改 → 汇报用户，合并由用户定（或用户让我合）。

---

## 6 用户的决定与规矩（这一轮）

- 执行器设计 §14 第 1–9 项都已确认（点质量、1 s 周期、按转弯率、公开进近速度、办法 A + B + train 敏感度、平行跑道上限
  留在 `instructions/airport.py` 的镜像）。
- 格式名跟着内容改，每个钉住它的地方一起改，不为兼容留旧名（记在 memory）。
- 目标（stop hook，已满足）：执行器能把现在的词表飞出来之后，完成先验第一次训练，先不交叉验证。
- 同意：写 outputs 的正式回放门、删旧规格目录、把 `dev-executor` 合进 `dev-two-tier`。
- 写入共享 outputs 目录会被权限检查拦下，要用户点头。

---

## 7 下一步（按优先）

1. **收发布 agent 的结果**：审查、核对前端、汇报。
2. **先验 × 执行器自由生成**（先验门第三项）：每 2 s 由先验按执行器当前状态说词，执行器飞；读落地率和与观测的差别。
   需要先写设计（放进先验设计文档的一节）：采样还是取最大、何时停、runner、读数。
3. **词的门**：高度词与落地窗口的冲突——看是词表落地段的读法问题还是执行器的瞄准问题，给用户选项。
4. **转弯率不在词里**：是否把转弯快慢放进词表（词表改动，用户决定）；或只靠闭环修正。
5. 先验交叉验证（尺寸、学习率）；加机型。
6. 替代机型的动力学（用户在 `aircraft/` 处理）。

---

## 8 怎么跑（从仓库根目录，`conda activate aeroviz`，`PYTHONPATH=.:4dTrajectory`）

```bash
I=4dTrajectory/outputs/POOLED/instruction_language/v2_20260924
E=4dTrajectory/outputs/POOLED/executor/<新名字>
python run_ts.py executor_spec --instructions $I --dir $E --word-clock track --workers 6     # 干净的工作树
python run_ts.py executor_sensitivity --instructions $I --executor $E --per-airport 400        # train
python run_ts.py executor_replay --instructions $I --executor $E --split val --out $E/replay-val   # 阶段 4，val
python run_ts.py prior_train --instructions $I --out 4dTrajectory/outputs/POOLED/prior/<新名字> --device cuda
```

长任务用 `nohup setsid` 分离运行、脚本里 `echo $$ > pid`、用 Monitor 看 PID；训练约 27 分钟，val 回放约 7 分钟。
