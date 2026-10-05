# 阶段 A：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_a_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。上一条（A42 和清理）已做完（`dev-two-tier-v4` 在 `06b8fde1`）。

```
A43：后端启动时不再跑一致性检查（D73 改了；用户 2026-10-05）。
依据：vocabulary.md §12.1 A43、D73（最后几句）、D71；notes/report_to_designer_backend_checks.md（实测和用户的意见）。

1. 把 dev-two-tier（含本条的提交）合进 dev-two-tier-v4，在新分支 dev-two-tier-v4-a43 上做。
2. 代码、测试、文档；不建产物，不重跑任何数据：
   - backend.py 的 executor_for 改用 replay.open_spec（不跑检查；规格与产物、规格 sha 的核对保留），
     不再调 closed_loop.require_conforming_closed_loop，去掉 _failed；
   - fly.py 里 apart_from_stored 旁边写唯一的一处拒绝：活航段离存储状态超过 STATE_BOUND_M（水平或垂直）就按名字拒绝，
     带上两个距离、集合和该怎么办；后端的 fly 和 B 的钩子（aeroviz_backend/autopilot_segment/prior.py，也调
     apart_from_stored）都用它；check_live 自己在开头跑三个检查，像别的运行器一样；
   - warm_up：同一产物的各集合只读一次 signals；请求不等别的集合的预热（预热用各集合自己的锁，请求遇到没开的集合自己开）；
     B 的钩子在用 _lock、_claim、_latest、_require_split、executor_for、set_flown：名字和意思保持不变，要变就写明；
   - 页面（trainingAutopilot.ts、TrainingAutopilotStatus.tsx）：答复超过 2 s，状态里写"后端正在打开这个集合"；
     拒绝作为带原因的错误显示；不加超时；
   - 文档：backend.py 的模块说明、docs/reference/contracts.md 的 C33、ts_transformer/CLAUDE.md 里那一行（"the backend at its start"）。
3. 测试按 A43 的规格写（预热和 fly 都不调三个检查；改了一条律的执行器在第一次请求就被按名字拒绝；
   没预热完的集合的请求不等别的集合；页面的状态消息）。A23 的集合和夹具、B 的钩子的测试照旧要过。
4. 审查（只审代码）→ 提交 → 在日志里记每个集合预热的秒数（改前改后）→ 报告提交号。
```
