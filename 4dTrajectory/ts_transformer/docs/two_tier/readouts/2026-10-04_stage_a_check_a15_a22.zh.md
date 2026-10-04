# 两层模型 v4 阶段 A：Claude 对 A15–A22 和正式产物的核对（2026-10-04）

设计文档 `docs/two_tier/design/vocabulary.md` §12.2 列了阶段 B 的正式运行之前 Claude 要核对的五项。A0–A14 的核对见
[`2026-10-04_stage_a_check.zh.md`](2026-10-04_stage_a_check.zh.md)；本报告核对 A15–A22 和 A21 的正式产物，逐项给结果。
正式产物的读数见 [`2026-10-04_stage_a_a21_report.zh.md`](2026-10-04_stage_a_a21_report.zh.md)。

## 1 结论

五项全部通过，没有需要你决定的新问题。

| 项 | 结果 |
|---|---|
| 1 D57、D58、D59、D61、D62 和 `688e945e` 之后的代码对照设计 | 通过（§2） |
| 2 A15–A22 的定向测试在构建所用的提交上重跑；该提交的完整测试 | 通过：350 个定向测试；完整测试 1,547 个 |
| 3 正式产物的内容 | 通过（§3） |
| 4 只写了新目录；旧目录没动；作废目录已删；校验和相符 | 通过（§4） |
| 5 归档搬移之后没有被改动 | 通过（§5） |

## 2 决定对照代码（第 1 项）

正式构建用的是 `dev-two-tier-v4` 的 `9a986c09`。它和 `df84cbf1` 的代码相同，只多了文档合并。D48–D56 已在 `688e945e` 上核对过。

| 决定 | 代码 | 测试 |
|---|---|---|
| D57 只有一种说词时钟：句子按自己的行说 | `autopilot/executor.py` `fly`（句子时间 = 周期 × 周期长度）；`ExecutorParams` 没有时钟参数；距离、航迹两种时钟和只为它们存在的计数都已删除 | `test_an_open_loop_replay_says_each_word_on_its_own_row`；`test_every_way_of_flying_gives_the_same_flown_states`（单机批量、多机批量、单机单条） |
| D58 高度词是机场标高 E 以上的高度 | `instructions/words.py`（`altitude_index`、`altitude_level_m`、`altitude_msl_m`）；标注器 `labeller/read.py` 只在一处减去 E；执行器的纵向律在 T + E 上飞（批量和单机单条）；判定的包络管和闭环里的语法都按 E 以上读 | 两个标高不同的机场读出同一句话（含复飞）；执行器在 T + E 改平；复飞后的跑道词按 E 以上判断；规格测量的平飞高度按 E 以上收集；第 6 条按给它的高度判断 |
| D59 网格用拟合行（就是 D22 本身） | `instruction_spec --grid fitted`；`measure.fit_altitude_grid`（精确动态规划） | 能找回已知网格；不比穷举出的任何网格差 |
| D61 竖直剖面和状态列进公共接口 | `instructions/airport.py` `VerticalPath`，由第一个 runner 读一次、写进 `candidates.json`（`ts-instruction-candidates-v3`）；判定、回放、执行器规格、执行器一致性都从这里读，第一个 runner 之后没有代码再读 CIFP；`instructions/artefact.py` 有 `STATE_COLUMNS`、`ClosedLoopSentence`、`closed_loop_sentences`，`autopilot/closed_loop.py` 不再留副本 | `candidates.json` 能写能读，旧格式按名字拒绝；从 `candidates.json` 和从 harvest 跑道数据做决断高检查结果相同（KRDU）；读回来的每个字段和写进去的相同；架构测试保证执行器的各条律不读竖直剖面 |
| D62 语法的逐列掩码 | `instructions/grammar.py`：规则 1–6 只写一次（`_rules`），`apply` 和 `column_mask` 都由它求值 | 小词表上逐行、D59 词表上随机行，掩码都等于"后面各列任意补全后经 `apply` 判断"的定义；水平角下低于飞机的高度被允许，随后角度列只允许下降档；调用方不允许任何下降档时，该高度不被允许 |

A22 对行为没有改变，有两项证据：

- **`apply` 的答案没变。** 和改动前的旧模块比，答案逐字相同，拒绝原因和细节说明都一样。我比了 60,000 行随机输入，两轮独立审核又比了约 250 万行，其中包括"高度不是数"的情况。
- **正式数据逐位相同。** 重建的 `v9_20261004` 和作废的 `v8_20261004`（D61 之前那次构建）逐个数组比较：信号、句子，以及闭环文件的全部 198 个数组（每行的词、修正标记、每个 2 s 行的状态、误差）逐位相同。不同的只有记录性的字段（代码哈希、写出时间、git 头、执行器目录名），另外 `candidates.json` 多了竖直剖面。

## 3 正式产物（第 3 项）

- **规格**：`spec.json`（sha `f1aad5a30202`，读法 `instruction-v5`）。
  - 下降名义角 1.5 / 2.5 / 3.0 / 4.5°，分档边界 −0.5 / 2.0 / 2.75 / 3.75 / 10°，爬升 1.5°（D56）。
  - 高度网格 60 / 120 / 450 m，断点 1,260、2,700 m（D59）。
  - `measurements.json` 记下了两个选择：`chosen_candidate` 0.25，`chosen_grid` fitted。
- **候选跑道**：`candidates.json` 里全部 25 条候选跑道都有竖直剖面（TCH、下滑角、决断高）。E 分别是：KMSY 1.22 m、KRDU 132.59 m、KSJC 18.9 m、KSMF 8.23 m、KSTL 188.37 m。
- **一致性记录**：三份通过记录，每份对应的代码哈希都和构建所用代码现算出的值相同。
  - 标注器 `conformance/passed-354bac48abb3.json`；
  - 闭环 `closed_loop/conformance/passed-7c808c099726.json`；
  - 执行器 `v14_20261004/conformance/passed-9c540c8f4fa7.json`。
- **回放能飞回存下的状态**：每个 Δ 的闭环回放都对每个 2 s 行核对位置和高度，任何一行超出界限，回放就会报错退出（`executor_replay.closed_loop_columns`）。六个回放全部跑完，说明全部在界内。
- **D50 的规则**：train、select、val 三个划分，在 Δ = 2、4、8 s 下，横向、纵向的"超出容差却没有朝航迹的修正"都是 0。
- **D34 的读数**：每个 Δ 都有。三个划分都有闭环摘要；train（每个机场 400 架）和 select（全部）都有回放；每个回放的读数表都包含五个机场。val 回放等你决定，没有跑。

## 4 写入与校验和（第 4 项）

- **只写了新目录。** 从第一次正式构建开始（13:36Z）到现在，`4dTrajectory/outputs`、`trajectory_data_process/outputs`、`data` 下除 `v9_20261004`、`v14_20261004` 之外，变动的只有 `POOLED/instruction_language` 和 `POOLED/executor` 这两个父目录本身（建新目录、删作废目录）。没有任何文件被改。
- **作废目录已删，都经你同意。**
  - A18 的 `v7_20261004` / `v12_20261004`；
  - D61 之前那次构建的 `v13_20261004` 和 `v8_20261004`。
- **校验和相符，数据只读。** `v9_20261004` 有 31 个文件，`v14_20261004` 有 73,907 个；`SHA256SUMS` 全部核对通过，两个目录都已设为只读。

## 5 归档（第 5 项）

`archive/two_tier_v3_2026_10/` 自 A0（`e57c62d8`）以来的改动只有两类：

- **README 的索引行**：A0 的审核修正 `d2917c06`、A4–A6 的 `330ffbaf`，都是登记新搬进来的文件；
- **整份搬进来的文件**：A0 审核、A4–A6 搬进来的代码和测试，以及 `a0c7c510`、`a534a2d9` 搬进来的 instruction-v3 设计文档。

已经在归档里的文件没有被改过。

## 6 留给后面的

- **设计文字与读数不完全相符：D45 说"闭环里一个词的平均迟到为零"，正式数据是 Δ = 2 / 4 / 8 s 平均 +0.04 / −0.30 / −1.48 s（负数是提前）。**
  - 这不是代码问题。A12 报告（`2026-10-04_stage_a_a12_a13_report.zh.md` §4）已经量过、解释过：转弯里每 2 s 有一个航向词，Δ 大时一行里落进好几个，只说最后一个，所以说出来的词偏早；前后没有别的航向词的单独一个词，平均接近 0。
  - 读数里的单位是秒：`heading_lateness_rows` 返回的是 2 s 行数，runner 乘上 2 s 后才存。
  - 建议把 D45 改成"单独的词平均迟到为零；一行里有几个同列的词时，说的是最后一个，平均偏早"。这是文字上的事，改不改由你定。

- **后端（`aeroviz_backend/autopilot_segment/`）现在导入时就会失败**，因为它引用的 `autopilot/runway_data.py` 已删。它之前就已经跟不上这个分支（执行器的参数和格式都变了），按 outline 的计划在阶段 D 改写；在此之前，`run_all_tests.sh` 里后端那部分测试会在收集阶段报错。
- **你比较 D34 的读数并选 Δ**（D7、D11）；val 的回放等你决定。
