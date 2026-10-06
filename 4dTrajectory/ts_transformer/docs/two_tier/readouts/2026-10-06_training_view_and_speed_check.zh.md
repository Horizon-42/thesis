# Training 视图第二部分与闭环提速的复审，2026-10-06

复审 B 的实现者在 `dev-two-tier-v4` 上的提交（未合并）：Training 视图第二部分（`65314499`、`550fb0c2`、`67ff4ce9`、
`a3cd6ae3`、`d870921f`；outline §6.2、D134–D136）和闭环提速（A44 `3b535db6`、C13 `65a21c57`、B14 `65adf480`、
`model_speed` `e558e444`；outline D138）。按 `docs/two_tier/review_guide.md` 和 outline D131：两个独立审查员（opus），只读，
只在 CPU 上跑单个测试文件（C10 正在用 CPU 和 GPU），不读 val、test 的读数。路径相对 `4dTrajectory/ts_transformer/`。

## 1 结论

**没有 S1。** 没有任何路径能把 val 的数字放进回答，没有新的 val 读取，没有写进 live 数据或模型目录的地方；快模式和参照模式
没有发现结果上的差别。C10（`two-tier-v4-post`，`98a89871`）不含这些提交中的任何一个。

审查员核对过的（行为检查）：

- **结果路由**：A 的标注读数只读 `train`、`select` 两块；B 的 val 自由生成只在 `holds_written_claim` 成立时读，验证读数还要
  `prior_validation` 自己的登记；选择一节只读 `score`、`folds`；训练一节只读 train、select 的损失，不读 prior 的 `config.json`
  （D120 的 val 计数不会出现）；路径解析后必须在 outputs 根之下；只回答指定字段；不写任何东西。意图路由 200 / 404 / 409 / 400 都有测试。
- **共用写入器**：代码和 `serialise` 与各阶段原来的一样。
- **`flown_sentence`**：A 的指令与 `replay_batch` 建的一样，航迹改为不舍入，旧的舍入值用摘要固定在测试里；B、C 的旧块相等。
- **A44**：`Start.moved` 就是原来 `start_moved` 的函数体搬过去，只是拒绝的先后顺序变了；保留的 series 不被改动，按 split 内
  位置区分，`release` 清空；fork 时不带 CUDA 状态或打开的文件。
- **B14**：`PER_AIRCRAFT` 就是原代码（只加了 float64 的拒绝）；`BATCH` 运算相同、顺序相同。审查员自写模糊测试：3,000 架、
  60 行，位置卡在 FAF、入口、锥边，高度卡在各个界限上，含 NaN 和三种复飞状态，0 处不同。`prior_behaviour` 在 B14 前后答案逐字节相同。
  各运行器用的模式：自由生成、`model_speed`（B）、窗口循环用 `BATCH`；`prior_behaviour` 两种都跑；`prior_validation` 和说话器默认用 `PER_AIRCRAFT`。
- **C13**：经 worker 的选择读数用同样的 places 和 `readout_numbers(seed, p)`，权重按 state_dict 逐位传过去，求和按批的顺序。
- **`model_speed`**：只读 select；只写新的 `--out`；B14 的检查在计时开始前跑完。

## 2 发现

| # | 级别 | 位置 | 问题 | 处理 |
|---|---|---|---|---|
| 1 | S2 | `prior/procedure.py`；`prior_free_generation.py:115`、`post_window_loop.py:119`、`model_speed.py:249` 默认 `BATCH` | B14 定下的真实数据检查（B5 的 KRDU 折用 `BATCH` 重说 select 日，句子逐词相同）要等 C10 结束才能跑，而 `BATCH` 已是默认 | 改顺序：C10 结束后第一件事就跑这项检查，在测速、任何自由生成或战役之前 |
| 2 | S2 | `experiments/training_export.py:218-224` | outline §6.2 第 8 条要求导出拒绝「包络超出飞行航迹」，Python 端没有测试（只有前端读取器有） | 加测试：替换 `envelopes` 让它返回 `endRow = rows + 1`，断言 `flown_sentence` 拒绝 |
| 3 | S2 | `tests/test_prior_training_export.py:389-409` | 第 8 条要求导出前后 base 目录不变；测试只快照输出根，没有快照 prior 目录 | 测试里对 prior 目录的字节做前后比较 |
| 4 | S3 | `training_results.py:145`、`:190` | 闭环一节拒绝非 train/select 的 split、验证读数自己的登记，这两处没有测试（导出写出的东西到不了这里） | — |
| 5 | S3 | `training_results.py:227`、`ResultSections.tsx:189` | 速度读数缺 `device`、`torch`、`threads` 时页面显示「undefined threads」 | — |
| 6 | S3 | `post_training_export.py:304` | `--speed` 不检查读数测的是不是同一个战役（Speed 一节会写出它测的模型名） | — |
| 7 | S3 | `trainingSetResults.ts:168` | 读取器不核对回答的 stage、set、airport 是否就是所问的 | — |
| 8 | S3 | `procedure.py` `require_same_masks` | 每进程检查用 `[airport] * count`，不同候选数的填充只有单元测试覆盖 | — |
| 9 | S3 | `post_train.py:324` | O15：fork 出的 worker 的 `Swap` 和 `ru_maxrss` 含父进程的页，可能高估、拒绝其实装得下的 worker 数（只会多拒，不会错放） | — |
| 10 | S3 | `post_train.py` O15 | 没计每个 worker 留着的读取模型（约 8 MB）和一轮里保留的 series（几十 MB） | — |
| 11 | S3 | `tests/test_start.py` | `start_moved` 就是 `Start(...).moved`，相等测试只比了新建和保留的 series；真实 series 的复用由 C13 的测试覆盖 | — |

S3 按 D131 只列一行，不下命令、不再报告。

## 3 B 的 requests 文件（§1 的九条读法、§3 的设计文本）

九条读法都与代码一致（审查员核对了第 1、7、8、9 条涉及的代码）。第 7 条（A44 不保留到达记录，改为保留每个航班的 series，
直到 `release`）和第 8 条（B14 的每进程检查每个机场 0.6–1.7 s，不是「毫秒」）要改设计文本。
## 4 用户的决定（2026-10-06）

九条读法用户逐条看过，全部接受，写成 outline D139；第 9 条（O15 的内存算法）同时修正高估（扣掉与父进程共享的页，
把 worker 留着的读取模型和 series 算进去）。S2 第 2、3 条和第 9 条的修正交给 B（`notes/stage_b.md`）；S2 第 1 条改为顺序要求：
C10 结束后第一件事跑 B14 的真实数据检查，在任何用 `BATCH` 的真实数据运行之前。
