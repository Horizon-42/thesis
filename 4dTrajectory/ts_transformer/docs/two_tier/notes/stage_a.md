# 阶段 A：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_a_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。

```
A39：Training view 能打开的划分改由调用方给出。
依据：vocabulary.md §12.1 A39、§0.4 第 6 项；outline D109。

1. 把 dev-two-tier（9e44a687）合进来。在哪条分支线上做，按 §0.4 来，不打乱 A37/A38。
2. 只改代码和测试：
   - aeroviz-4d/src/data/trainingSample.ts 的 parseFlight（TRAINING_SPLITS）；
   - aeroviz_backend/autopilot_segment/backend.py 的 set_flown。
   允许哪些划分改由调用方给出；A 阶段自己的集合仍只给 train/select。
3. 测试按 A39 的规格写；A23 的集合和夹具必须逐字节不变。
4. 审查 → 提交 → 报告提交号。不导出、不重建任何集合。
```
