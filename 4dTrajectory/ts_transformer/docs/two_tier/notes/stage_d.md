# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-06 深夜，Claude 写，用户转发。

```
C10 已于 2026-10-06 22:46 跑完（十轮，只读）。现在的安排：
- 除多机控制外，能合的分支都已合进 dev-two-tier（7e581b88：v4、frontend.md、C14 的设计；780b7bfc：dev-traffic-scenarios），
  从最新提交继续（outline §4 item 9）；
- C10 续训到 14 轮（post_training D157，C14）交给阶段 C 的实现者做，你不做；
- 你先把阶段 B 的收尾做完，然后继续多机控制。
路径相对 4dTrajectory/ts_transformer/。

先读：outline §4 items 3、6、7、9，§5 rule 1、rule 13；post_training D157；outline D158（只读）。

一、阶段 B 的收尾（在 .claude/worktrees/two-tier-v4 上做；你已经开始了，从当前这一步接着做）
dev-two-tier-v4 现在是 dev-two-tier 的祖先。当前这一项跑完、下一项开始之前，在 two-tier-v4 工作树里执行
git merge --ff-only dev-two-tier。ts 的代码没有变化，只多了文档，以及 traffic-scenarios 的前端、后端和优化器代码。
主机和 GPU 上没有别的任务时，按顺序一项一项做：
1. B14 的真实数据检查：用 BATCH 把 B5 的 KRDU 折在 select 日重说一遍，写在 scratch。
   句子必须和它的读数逐词相同；不同就停下报告。
2. C13 在 GPU 上的检查：跑一轮，单进程和 N 个 worker 的结果必须相同。C 的续训要用 C13，必须等这项通过。
3. 对 base 测速（model_speed）：base 是 4dTrajectory/outputs/POOLED/prior/prior_base_20261006，
   结果写进 4dTrajectory/outputs/POOLED/speed/ 下的新目录。
4. 先对 base 跑 sha256sum -c；再重导 A 的 closed_loop_v12_20261005 和 B 的 prior_sets_20261006
   （同名，新格式，B 的带 source.speed）；再跑一遍 sha256sum -c，结果必须一致。旧的索引和集合先留着。
5. 浏览器检查交给一次性子代理。测试栈从 two-tier-v4 工作树起，端口和停止命令写进报告。
结果写进阶段 B 的日志，新开一节，注明由阶段 D 的实现者完成；更新 vocabulary、prior、post_training 的 §0.3 状态行。
第 2–4 项（占 GPU 的、要计时的）一做完，就在你的日志里写一行并报告。C 的续训等这一行才启动。
最后报告 dev-two-tier-v4 能否快进 dev-two-tier，由用户快进。

二、继续多机控制（在 dev-multi-control 上）
1. 先把 dev-two-tier 合进 dev-multi-control。
2. C 的续训运行期间：只写代码，只在合成输入上测试（rule 13：不做任何会拖慢续训的事）。
3. 续训跑完以后：做真实数据上的部分（MC0 的 D73 检查、D149 真实窗口逐位比对、MC1 普查，同上一份命令），
   然后报告 dev-multi-control 能否合并。

三、清理已经合并的工作树和分支（只是 git 操作，什么时候做都行）
- 工作树 merge-a43、training-attitude、stage2-restart，以及它们的分支
  dev-two-tier-merge-a43、dev-training-attitude、dev-stage2-restart；
- 分离的工作树 a25-build、stage2-real400；
- 没有工作树的分支：docs-optimizer-multi-aircraft、dev-frontend-design、dev-multi-control-design；
- 工作树 traffic-scenarios 和分支 dev-traffic-scenarios。
每个工作树按这个顺序删：
- 先用 find <工作树> -maxdepth 4 -type l 列出数据链接，逐个 unlink；
- 再 git worktree remove；
- 最后 git branch -d。不用 -D；-d 拒绝就停下来报告。
以下不动：
- 工作树 two-tier-v4-post（阶段 C 在用）和分支 dev-two-tier-v4-post；
- 分支 dev-kaus-parts、dev-airport-embedding、dev-step9-one-commanded、wip-r32-leg-timing、dev-multi-control、main；
- 工作树 kaus-parts、airport-embedding、step9-one-commanded、step9-run、multi-control；
- ~/.claude/jobs 下的工作树。

每一步的做法：单文件测试 → 独立审查（只审代码，审查者不能是作者）→ 用显式路径提交
（不用 git add -A，提交前看 git diff --cached --stat）→ 日志写一行。
你只能写这些设计文本：各文档 §0.3 的状态行、multi_control §0.3、你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_d_to_designer.md（整份重写），等用户定。
```
