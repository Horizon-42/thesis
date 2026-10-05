# 阶段 A：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_a_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。上一条（B 阶段请求的两处小改动）已由 Claude 按用户的话做完，A 只剩收尾。

```
收尾，不写代码。
1. tests/support.py 的 labelled_instruction_artefact 里的 runway_ends_from 已写好：
   dev-two-tier-v4-a42，提交 06b8fde1（单独一个文件，测试 260 个通过，一次独立审查无意见）。
   把 dev-two-tier-v4 快进到 dev-two-tier-v4-a42。
2. 另一处（B 钩子行 backend.py 第 107 行附近的注释）不由 A 改：这一行只在 B 的分支上，A 的线上没有；
   已改派给 B（notes/stage_b.md 第 3 条）。
3. 快进后清理已合并的工作树和分支：dev-two-tier-v4-a41、dev-two-tier-v4-a42（各自的 .claude/worktrees/ 目录），
   以及 a32、a37–a40 里已合并而没删的。先看每个目标是否已合并、工作树是否干净，再删。
4. 在日志里记一行：做了什么、提交号。报告。
```
