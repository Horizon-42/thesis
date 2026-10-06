# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（D132）已做完（`483b81d7`），C8 也已完成（`c396f9be`、`a2c6d578`）。

```
C10 正在跑（D137：O13 按你的提议定了）。选轮次的标准（D7）由用户定，在验证读数之前。

1. 换分支（用户 2026-10-06，outline §5 规则 1）：
   - 三个阶段的主体已合进 dev-two-tier，dev-two-tier-v4-post 已全部在里面；
   - 以后 C 的开发都在 dev-two-tier-v4 上，工作树 .claude/worktrees/two-tier-v4，开工前先和 dev-two-tier 对齐；
   - C10 正在 two-tier-v4-post 工作树里跑：C10 结束之前不要删这个工作树，也不要改它里面的任何文件。
     C10 结束、数据设为只读以后，再删 two-tier-v4-post 工作树和 dev-two-tier-v4-post 分支
     （先确认它没有不在 dev-two-tier 里的提交）；
   - 更新 post_training.md §0.3 表头那一行的分支和工作树。
2. 共用工作树：B 正在里面做统一布局。只改 C 自己的文件，用显式路径暂存，
   每次提交前看 git diff --cached --stat，不提交别人的文件。
3. 不改 outline §6.2 第 9 条列出的文件：前端 Training 视图的文件、aeroviz_backend/http_server.py，
   以及 C 的 Training 导出（experiments/post_training_export.py、post/training_files.py）。
   这些由 B 的实现者改（D133–D135），窗口视图和窗口导出跟着一起改。
4. C10 已从 two-tier-v4-post 启动，按原样跑完，不改它的代码。
5. C13（post_training.md §8、outline D138）：等 A44 进了 dev-two-tier-v4 再在那里做，下一次战役之前完成，C10 不用它。
   内容：选择读数分给说话的 worker 跑；单进程仍保留为一种模式；用 A44 的 Start，在 fork 之前打开；
   战役开始前测一个 worker 的内存，装不下就按名字拒绝（O15）。
不改 instructions/、autopilot/、prior/。
```
