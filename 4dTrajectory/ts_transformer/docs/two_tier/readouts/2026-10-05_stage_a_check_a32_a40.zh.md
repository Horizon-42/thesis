# A 阶段 A32–A40 复审，2026-10-05

按 `docs/two_tier/review_guide.md` 复审 `dev-two-tier-v4` 头 `ed2530ae`（用户已合入 `dev-two-tier` 的 `a804633e`）：A37（D80、D83、D85、D90）、A38（D97）、A39（D109）、A40（D111），
以及 A35 发布的集合 `closed_loop_v12_20261005`；A32–A36 的结论沿用上一轮复审，只核对它们之后有没有被改坏。
重点：执行器和先验的输入有没有读到不该读的东西；val 是不是只读一次；设计文字与代码是否一致。

**做法。** 读 `58fd8a2b..ed2530ae` 的全部生产代码改动（约 1,100 行，不含测试和文档）；在 A34 产物上做行为检查
（只读、在内存里、只用 train 和 select，没有读 val）；对新代码做 12 处变异，看测试会不会失败；跑目标测试；对已发布的集合抽样跑 `check_live`。
检查脚本在本会话的临时目录（`$CLAUDE_JOB_DIR/tmp`，不保留）。路径相对 `4dTrajectory/ts_transformer/`，行号是 `ed2530ae` 上的。

---

## 1 结论

**没有发现泄漏进执行器或先验输入的通道，A37–A40 的代码按设计实现。** 发现两处 val 读法（第 2 节第 1、2 条），三处设计文字要改（第 3 条起），都已处理或排给 A41。

| 检查 | 方法 | 结果 |
|---|---|---|
| D77：首个预测步之后的观测样本不进起点 | A34 产物，train 400 + select 400 架（种子 20261005，随机首行）：首步之后每一行的 6 个通道（e、n、高度、航迹角、地速、垂直速率）换成别的值；三条正式起点规则；`flights.start_state`、`flights.observed_rows`、`start.moved_signals`（`NO_MOVE` 和一次真实移动：转 23°、抬 120 m、速度 ×1.07） | 起点状态、存储的观测行、移动后的行逐位相同（800/800）；`moved_signals` 一律只留到首步那一行（800/800） |
| D67/D77：说出的闭环起点回到存值 | `closed_loop_start_check`（R53）在合并后的代码上，select 每机场 20 架 × Δ 2/4/8 s | 300 句，0 失败，位置最大差 0 m；开头的三个一致性检查通过（标注器读 336 架同原；执行器批 0 / 0 m、错开批 0 / 0 m、单架 2.1e-8 m、`moved` 2.2e-6 m（它自己的界 1e-4 m）；闭环 258 架 0 m） |
| A35 的集合在后端活着 | `check_live` 抽样：每机场 2 架（种子 777，不是 A 用的 20261005），每个词的活航段 | 962 段，0 个不同（landed 135、segment_end 812、unstable_at_minimums 15） |
| D78/A37：候选的高度偏移来自已发布数据 | 五个机场的 25 个候选：`candidate_hae_minus_msl_m`（已发布跑道数据，按 sha256 校验）对 manifest 的 `runway_targets` | 逐个相等；候选集与 manifest 的跑道集相同（5 个机场，无多无少） |
| D111：A40 的计数 | 我自己重算（`faulty_flights` 与 Δ = 4 s 的闭环句的交集） | train 498 / 40,530，select 177 / 6,199，与设计相同；种类：held position 484 + 172、jump 63 + 42、reversal 5 + 4（一架可以有几种） |
| D85：A34 产物上没有 val 读数 | `readout.json` / `readout.md`、`closed_loop/summary.json`、A34 报告 | `readout` 和 `summary` 里 val 只有数目（"not shown: read once…"）；报告有两处露出 val（第 2 节第 2 条） |
| 产物没被动过 | `SHA256SUMS`：`instruction_language/v12_20261005` 29 个、`executor/v17_20261005` 73,853 个文件；两个目录只读 | 全部通过 |
| 目标测试 | `test_start`、`test_closed_loop`、`test_track_faults`、`test_autopilot`、`test_training_export`、`test_instruction_labeller`、`test_instruction_conformance` | 245 个通过（104 s） |
| 测试会不会失败 | 对新代码做 12 处变异（临时副本，不动任何分支），各跑相关测试文件 | 见第 4 节 |

**没有重做、按 A 的日志接受的三项**：A36 的样本与 A23 的字节相等（A 的 `cmp`）；A32 的新测试在 `cf549e47` 上会失败（复审员跑过）；A35 的集合在浏览器里能打开、能点词（A 的一次性 agent）——我只做了后端一侧的 `check_live`。

其余核对过、没有问题的地方：

- **D90 的守卫在输入处。** `Loop` 没有 `timed_out()`，`Loop.executor` 公开且读到结束周期、飞行记录和气动参数（`test_the_loops_executor_is_public_and_the_loop_has_no_timed_out`）；
  先验一侧的行为测试在 B 阶段的 `tests/test_prior_free_generation.py:452`（首步之后的观测样本和时间上限变了，输入不变）。时间上限只由观测落地时间定，只用来结束一架飞机，不是任何模型的输入。
- **`Executor.take`** 对逐架的量逐一核对过：输入、跑道、坐标图、时间上限、状态、倾斜、三条律的 `PER_FLIGHT`、`done`、`done_cycle`、`halted`、`last_command`、`runway_issued`、`speed.approach_ias_mps`
  和全部记录（状态、命令、想要的、限制、模式、句时间、跑道行）都取了；没有漏项。`Spoken.take`、`Loop.copy` 也一样；`Loop.copy` 拒绝布尔掩码和浮点。
- **`start_moved`**：先对存储的信号做检查，再移动；移动只用首步行之前（含首步行）的行，之后的行在 `moved_signals` 里被截掉，所以不管起点规则怎么变，它都读不到首步之后的样本（上面 800 架的检查）。
  时间上限仍是该航班自己的（用户 2026-10-05）。
- **A39**：后端和前端都从调用方拿划分；阶段 A 的服务（`stage_a_service`）和面板、发布检查给 train 和 select；val 的拒绝在任何检查之前（变异 M6）。没有集合被重导出。
- **D83 的拐点规则**（`ObservedPath._at_vertex`）：我用 4,000 条随机折线（含近乎掉头）对每个分支做了计数。"另一段的最近点不是共用顶点"这一支在 `t == 0` 时常被走到（42,674 次调用里 23,519 次），
  在 `t == 1` 时一次也走不到（530 次）——由"前进匹配"规则保证。这一支在 `t == 1` 一侧是死代码，无害，不要求改。
- **D97 (3)**：A38 的测量（200 架单飞对 2,048 一批：65 架状态差最大 7.9e-10 m，词、修正、结果全同）和用户的界一致；测试按用户的界写（状态在 `STATE_BOUND_M` 内，词、结束、结果逐位相同）。
  这与 post_training §6.4 当时写的"逐位相同"不符，已改（第 3 节第 3 条）。

---

## 2 发现

| # | 类别 | 位置 | 问题 | 大小 | 改正 | 归属 |
|---|---|---|---|---|---|---|
| 1 | 划分（D85） | `experiments/instruction_figures.py:45`；`experiments/executor_turns.py:347` | `instruction_figures`（R10）从 val 日抽航班画页；`executor_turns` 的 `--split` 接受 val。D85 与"val 每个阶段只读一次"都不允许。A37 的名单没包含它们，A 在日志里"报告未改"。A34 的产物上没有跑过 `instruction_figures`（目录里没有 `figures/`） | 没有数字露出；是下次一运行就会读 val 的口子 | 页面改取 select；`--split` 只允许 `READ_SPLITS` | A（A41） |
| 2 | 划分（D85），低 | `readouts/2026-10-05_stage_a_a34_report.zh.md` 末节"A37 的代码在这个产物上查了一遍"（行 246–257） | 报告写了 val 的一句差异、val 那架航班的拐点转角（−178.6°）和 val 的抽样核对结果（"val 的各格……浮点舍入级差别"）。没有性能数字，也没有被用来选任何东西，但 D85 写的是"没有报告、摘要或打印文字展示 val 的读数" | 几行字 | 删去 val 的数字，写"val：不展示（D85）"；以后核对代码改动只用 train 和 select | A（A41） |
| 3 | 设计文字缺值（Claude 的值） | `autopilot/closed_loop.py:129` `REVERSAL_TURN_DEG = 170.0` | 拐点处转角大于 170° 算掉头、取匹配那一段的一侧；这个值和规则的写法不在设计里（D83 只写"法线之和"）。代码注释自己写了"Claude's value" | 5 句 / 约 170,000 句改变；不重建 | 写进 D83 和 §4.9，写明为什么是 170° 而不是 D111 的 120°：低于 170° 的拐点逐位如前，A34 的产物不用重建 | 我（已写） |
| 4 | 设计文字与测量不符 | `post_training.md` §6.4 末条、D94、第 2 节第 4 项的测试句 | 写的是执行器在不同批里"逐位相同，由 D97 保证"；A38 的测量和用户的界是"词、结束、结果相同，状态在 1e-6 m 内"（实测 ≤ 7.9e-10 m）。C6 的测试若按"逐位"写会失败 | C6 还没开始 | 改成用户的界，并写明一个舍入在个别航班上可以移动一个离散步，这样的窗口计数（D94） | 我（已写） |
| 5 | 设计文字缺项 | vocabulary D97 (4)、§6 第 3、5 项的代码列 | D97 (4) 仍写"A38 的读法留给用户"；用户已答（拉伸位置和高度、时间上限不变）；§6 没写 `Move`、`start_moved`、`Loop.copy`、`Executor.take`、`faults` 的名字；A38 的"转弯在机场坐标系里做，起点速度因此动 0.4 % 以内"没记 | — | 已按用户的答案和 A 的读法写进 D97、§6 | 我（已写） |
| 6 | 设计文字过期 | outline §1 表、§4；vocabulary §0.4、§12.1；prior §0.4 | 状态行还写着"A32–A35 next""C 阶段 not started"；§12.1 里 A35、A37–A40 已做完，规格还留在设计里（outline 规则 10 要移到日志）；prior 的 B5 条件里还有"Claude 对 A32–A37 的复审" | — | 已改；规格移到 A 的日志 | 我（已写） |

**只作信息：**

- 一个合并相关的提示：用户合入之后，主检出的服务还在跑旧代码；新代码读不了旧集合 `closed_loop_v11_20261004`（sample v9），集合选择器会在它上面显示面板内错误，直到删掉它（见第 3 节）。
- A38 的"转弯在机场坐标系里做"，起点速度因地面比例动 ×0.99976（10°）、约 0.4 %（90°）：低于起点规则自己的噪声，不修正；已写进 D97 (4)。
- `faulty_flights` 对每个 split 读全部信号（含首步之后的行）来标记；标记只用于选择（先验 D111），不是输入。
- `Loop.step` 的 D80 检查现在对每架飞机每一行做，包括已完成和被停住的；`RowRefused` 的原因名没变。

---

## 3 要用户决定的事

1. **删除被取代的东西**（计划里早有，要用户的话）：`instruction_language/v11_20261004`（2.3 GB）、`executor/v16_20261004`（4.4 GB）、五个机场里已发布的旧集合 `closed_loop_v11_20261004`
   及 `training/index_v4.json` 里它的条目。没有任何代码读它们（我在 B、C、A 三条线上查过）；新代码读不了旧集合。

没有别的要用户决定的事：第 2 节第 1–2 条在 D85 之内，由 A41 做；第 3–6 条是设计文字。

---

## 4 变异抽样

在 `ed2530ae` 的临时副本上（不动任何分支），每次改一处，跑相关测试文件，看会不会失败：

12 处全部被抓住（测试失败）：

| 变异 | 测试文件 | 结果 |
|---|---|---|
| M1 `moved_signals` 不再截到首步行 | `test_start` | 抓住（2 个失败） |
| M2 `Loop.copy` 不取各航班的复飞次数 | `test_start` | 抓住 |
| M3 `Executor.take` 不取 `done_cycle` | `test_start`、`test_autopilot` | 抓住 |
| M4 `Spoken.take` 不取说过的词 | `test_start`、`test_autopilot` | 抓住 |
| M5 `track_faults` 的中位数含这一步自己 | `test_track_faults` | 抓住 |
| M6 后端在检查之前不拒绝别的划分 | `test_autopilot_segment` | 抓住 |
| M7 拐点：掉头阈值 170° 改 178° | `test_closed_loop` | 抓住 |
| M8 `Move` 转向反了 | `test_start` | 抓住（2 个失败） |
| M9 `compass_track` 保留 360.0 | `test_start`、`test_closed_loop` | 抓住 |
| M10 `Loop.step` 不检查已完成航班的行 | `test_start` | 抓住 |
| M11 `Loop.copy` 不取各航班的时间上限 | `test_start` | 抓住 |
| M12 `Executor.take` 不取倾斜角 | `test_start`、`test_autopilot` | 抓住 |

（M6 第一次因我自己中途杀掉上一轮、副本里留着改动而被跳过；还原副本、先确认基线 23 个通过后重跑，被抓住。）
