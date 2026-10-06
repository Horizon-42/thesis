# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（D132）已做完（`483b81d7`），C8 也已完成（`c396f9be`、`a2c6d578`）。

```
现在没有写代码的命令：C10 的正式运行要等 O13（你在日志 §25 的提议）和用户定的标准。

1. 换分支（用户 2026-10-06，outline §5 规则 1）：
   - 三个阶段的主体已合进 dev-two-tier，dev-two-tier-v4-post 已全部在里面；
   - 以后 C 的开发都在 dev-two-tier-v4 上，工作树 .claude/worktrees/two-tier-v4，开工前先和 dev-two-tier 对齐；
   - 现在删掉 two-tier-v4-post 工作树和 dev-two-tier-v4-post 分支（先确认它没有不在 dev-two-tier 里的提交）；
   - 更新 post_training.md §0.3 表头那一行的分支和工作树。
2. 共用工作树：B 正在里面做统一布局。只改 C 自己的文件，用显式路径暂存，
   每次提交前看 git diff --cached --stat，不提交别人的文件。
3. 不改 outline §6.2 第 9 条列出的文件：前端 Training 视图的文件、aeroviz_backend/http_server.py，
   以及 C 的 Training 导出（experiments/post_training_export.py、post/training_files.py）。
   这些由 B 的实现者改（D133–D135），窗口视图和窗口导出跟着一起改。
4. C10 的正式运行等用户合并后从主树启动，不在共用工作树里跑。
不改 instructions/、autopilot/、prior/。
```
