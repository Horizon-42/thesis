# 阶段 A：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_a_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。

```
A41：val 日只读一次（D85）。
依据：vocabulary.md §12.1 A41、D85；复审记录 readouts/2026-10-05_stage_a_check_a32_a40.zh.md §2 第 1、2 条。

1. 把 dev-two-tier（含 a804633e 之后的文档提交）合进 dev-two-tier-v4，在 dev-two-tier-v4-a41 上做。
2. 只改代码、测试和一份报告：
   - experiments/instruction_figures.py：页面从 select 日取，不再取 val；
   - experiments/executor_turns.py：--split 只允许 READ_SPLITS（train、select）；
   - readouts/2026-10-05_stage_a_a34_report.zh.md 末节（A37 的代码在产物上的检查）：删去 val 的数字
     （val 的一句差异、val 那架航班的拐点转角、val 的抽样核对结果），写"val：不展示（D85）"；
   - 4dTrajectory/ts_transformer/CLAUDE.md 里 instruction_figures 一行、docs/reference/runners.md 里它的条目：写 select；
   - vocabulary.md §0.3 里 A34 一行写着"报告在做"，改成报告已写成（readouts/2026-10-05_stage_a_a34_report.zh.md）。
3. 测试：两个运行器按名字拒绝 val。
4. 审查（只审代码）→ 提交 → 报告提交号。不建任何产物，不读 val。

另：删除被取代的产物（用户 2026-10-05 同意；与 A41 无先后）。
1. 删除前逐个看一眼目标，只删下面这些，不删别的：
   - 4dTrajectory/outputs/POOLED/instruction_language/v11_20261004（2.3 GB）；
   - 4dTrajectory/outputs/POOLED/executor/v16_20261004（4.4 GB）；
   - 五个机场（KMSY、KRDU、KSJC、KSMF、KSTL）training/ 下的 closed_loop_v11_20261004 目录，
     以及各自 index_v4.json 里它的条目（closed_loop_v12_20261005 的条目和目录不动）。
2. 删完后：跑发布检查（aeroviz-4d/scripts/check_publication.ts），确认 v12 集合仍能读；
   在 vocabulary.md §0.3 里记一行（删了什么、哪天、用户同意）。
3. 不重启服务；重启由用户做。
```
