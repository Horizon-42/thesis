# 阶段 A：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_a_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。

```
暂无命令。

分支规则（用户 2026-10-06，outline §5 规则 1）：三个阶段的主体已合进 dev-two-tier。
以后各阶段的开发都在 dev-two-tier-v4 上，工作树 .claude/worktrees/two-tier-v4，方便复审时只看一份差异。
- 开工前（以及有新的设计提交时），先把 dev-two-tier-v4 和 dev-two-tier 对齐（能快进就快进，不能就合并 dev-two-tier）。
- 同一工作树里可能同时有别的阶段在干活：只改自己的文件，用显式路径暂存，
  每次提交前看 git diff --cached --stat，不提交别人的文件。
- 正式构建或正式运行，等用户合并后从主树启动，不在共用工作树里跑。
- 由用户把 dev-two-tier-v4 合进 dev-two-tier。

two-tier-v4-a32、two-tier-v4-a37 两个工作树和它们的分支已按用户的话删掉（2026-10-06，Claude 做的）。
```
