# 阶段 A：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_a_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。

```
A44：起点在一个进程里只打开一次。依据：vocabulary.md §12.1 A44、outline D138。

分支：在 dev-two-tier-v4 上做（工作树 .claude/worktrees/two-tier-v4；outline §5 规则 1）。
开工前先把 dev-two-tier-v4 和 dev-two-tier 对齐。同一工作树里 B 也在干活：只改自己的文件，
用显式路径暂存，每次提交前看 git diff --cached --stat，不提交别人的文件。

1. autopilot/start.py 加一个 Start 对象：按「产物、split、Δ、执行器目录」打开一次，持有每次调用都要读的东西：
   - 打开的执行器（replay.open_executor，D73 的检查仍是每个进程一次）；
   - 候选跑道；
   - 这个 split 的 signals（按 signal index）；
   - 用来重建 series 的到达记录；
   - 标注句子的 offsets。
   Start.moved(sentences, moves, …) 只做这些航班自己的那部分；每个航班的 series 只重建一次并留住。
2. start 和 start_moved 保留：内部打开一个 Start 再调用它，一次调用的写法仍是可读的参照。
3. 测试：
   - Start.moved 和 start_moved 给出的 loop、order、observed rows 逐位相同（不移动、移动起点两种，合成产物上）；
   - 第二次调用不读任何文件（数读取次数）；
   - 在 fork 之前打开的 Start，fork 出来的进程能用。
4. 单文件测试 → 独立审查（只审代码）→ 显式路径提交。改了 autopilot/，按 outline §5 规则 2 跑 D73 的检查
   （closed_loop_start_check），日志里给出最大差值。报告提交号；日志写一行，§0.3 状态表更新（设计正文不改）；不合并进 dev-two-tier，由用户合并。
不改 prior/ 和 post 的代码。C10 正在 two-tier-v4-post 里跑，不碰那个工作树。
```
