# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-09，Claude 写，用户转发。

```
你 requests 的 54–62 条评估完了，写进 multi_control D181（MC11）和 O20：
- 接受：54（按你建的）、55、56、58、59、60；
- 57 等 56 量完再定：只有续句确实决定峰值时才做；
- 61、62 暂不做（O20）。
  - 61：两种设备混用时，要么一轮不可复现，要么负载不均；57 若能让 GPU 多放 worker，效果相近还更简单。
  - 62：改动大，只有缓存决定峰值时才值得。
- post_profile 的 executor_step 读成 0 s：C8 已经结束、不会再跑，写进 docs/code-health-followups.md 一条，不改代码。
MC6 正在 run-mc6 里跑：不动那个运行工作树，只写代码、只跑改动文件的测试、少开进程（rule 13），不占 GPU。
路径相对 4dTrajectory/ts_transformer/。

一、代码（在 dev-multi-control 上，先把 dev-two-tier 合进来）
1. 55：select_per_airport 从 PROFILED_SETTINGS 去掉。这是小改动，不用审核 agent。
2. 58、59、60，数字都不能变：
   - 58：WindowLoop.samples(split, rows)，只为读的那些行建样本（copies 只要被改动那架的行）；
   - 59：一个 tick 里每行的场景只算一次；
   - 60：Speakers fork 之前在 campaign 进程里 gc.freeze()。
   58、59 改的是 C 的 post_window_loop，60 改的是 post_train，C 的测试都要原样通过。
3. 58–60 的核对：在固定输入上（CPU、一个线程），C 的窗口和 D 的窗口各一组，一轮的分支组、样本和读数和改之前逐位相同。
4. 56：experiments/multi_speed.py。
   - 一批，各部分各自计时：第一遍、第二遍、续句、说话步、执行器（两次同步之间，照 model_speed）；
     每部分开始时重置 GPU 峰值；
   - cProfile 只在 CPU 上用，而且只在计时器解释不了某一部分时用（你补充的第 56 条）；
   - 现在只在 CPU 上用冒烟批测通，正式测量等 GPU 空出来。
5. 审核按项目 CLAUDE.md 的 "Code review"：58–60 和 56 各走一次 agent，范围是 diff；55 是小改动。
   → 显式路径提交 → 日志写一行 → 报告 dev-multi-control 能否快进 dev-two-tier，由用户合并。

二、合并以后（每一步都等用户的话）
1. MC6 换代码：58–60 核对通过并合并后，向用户提议——
   - 在当前这一轮结束时停下；
   - 从合并后的提交建新的运行工作树，续跑 MC6（设置不变，D157）。
   停不停由用户定。
2. 56 的测量：等 MC6 结束或某一轮的边界、GPU 空着时，按用户的话量一批 1200 s。
   结果报给用户；如果峰值来自续句，再提议做 57（新设置 continuation_rows，默认等于旧行为，只用于下一个 campaign），
   核对办法见 D181。

另外，multi_control §0.3 已过时：还写着 MC1 运行中、MC5 未合并、MC6 未开始。按现状改写：
- 完成的里程碑各留一行和提交号；
- MC6 写运行中、起点、运行工作树和单元名，结果指向日志的节号；
- 开头那段说明同样更新。
设计其余部分已由设计者清理（过时的计划、§6.1 的对照表、完成里程碑的规格已删；新增待定项 O21）。
你只能写各文档 §0.3 的状态行、multi_control §0.3、你的日志和 code-health-followups 那一条。
requests_from_d_to_designer.md 整份重写：54–62 条已处理，删掉；新的读法另起。
```
