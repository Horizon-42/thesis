# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-06 深夜，Claude 写，用户转发。

```
C10 已于 2026-10-06 22:46 跑完（十轮，只读）。用户的决定：
- C10 续训到 14 轮，作为同一个 campaign，新加的 4 轮用合并后的代码（含 C13），从主检出跑（post_training D157、C14）；
- 除多机控制外，其余分支都合进 dev-two-tier；dev-two-tier-v4 等阶段 B 的收尾做完再合（outline §4 item 9）。
路径相对 4dTrajectory/ts_transformer/。

先读：outline §4 items 3、6、7、9，§5 rule 1；post_training D157、§0.4、§8 C14；outline D158（只读）。

一、暂停多机控制
在 dev-multi-control 上，把已经审查过的工作提交；没审查的留在工作树里，不提交。日志写一行停在哪里。

二、阶段 B 的收尾（在 .claude/worktrees/two-tier-v4 上做）
先等用户把 dev-c10-extension-design 快进进 dev-two-tier-v4（它包含 dev-frontend-design）。
主机和 GPU 上没有别的任务时，按顺序一项一项做：
1. B14 的真实数据检查：用 BATCH 把 B5 的 KRDU 折在 select 日重说一遍，写在 scratch。
   句子必须和它的读数逐词相同；不同就停下报告。
2. C13 在 GPU 上的检查：跑一轮，单进程和 N 个 worker 的结果必须相同。这也是续训用 C13 之前必须通过的检查（D138）。
3. 对 base 测速（model_speed）：base 是 4dTrajectory/outputs/POOLED/prior/prior_base_20261006，
   结果写进 4dTrajectory/outputs/POOLED/speed/ 下的新目录。
4. 先对 base 跑 sha256sum -c；再重导 A 的 closed_loop_v12_20261005 和 B 的 prior_sets_20261006
   （同名，新格式，B 的带 source.speed）；再跑一遍 sha256sum -c，结果必须一致。旧的索引和集合先留着。
5. 浏览器检查交给一次性子代理。测试栈从 two-tier-v4 工作树起，端口和停止命令写进报告。
结果写进阶段 B 的日志，新开一节，注明由阶段 D 的实现者完成；更新 vocabulary、prior、post_training 的 §0.3 状态行。

三、C14 的代码（post_training §8 C14，在 dev-two-tier-v4 上）
1. open_campaign：续跑时只允许轮数变大，其他设置和输入都必须相同。
   campaign.json 记下每次改轮数：时间、旧轮数、新轮数、commit、checks。
2. 记录里的输入路径用 this_checkout（experiments/training_export.py，只用这一份定义，像 model_speed 那样 import）
   映射后再比对；post_validation 也这样读。记录里原来的路径不改，每次续跑在自己那一条里写下实际读的路径。
3. 测试按 C14 写的做。另外在 experiments/ 里查所有读 campaign 记录 "inputs" 的地方，确认都经过 this_checkout。
4. 在 docs/experiments/intents.json 写新 4 轮的 intent，和代码一起提交。必须在启动之前提交，不能在运行中改。
5. 单文件测试 → 独立审查 → 提交。

四、合并前的准备（在 two-tier-v4 工作树里）
1. 把 dev-two-tier 合进 dev-two-tier-v4。到那时 dev-two-tier 应该已经带上用户合并的 dev-traffic-scenarios。
2. 已知只有一处冲突：post_training.md §0.3 的 C10 行。
   - 保留 dev-two-tier 那一行（Done），也保留 v4 的 C13 行；
   - 两行都按实际情况更新，再加一行 C14。
3. 跑测试：
   - 全套 ts 测试：-n 8 --dist worksteal，OMP_NUM_THREADS=1；
   - aeroviz-4d：npx tsc --noEmit 和 npx vitest run；
   - aeroviz_backend/tests。
   test_traffic_jobs.py::test_no_exited_child_is_left_behind_a_finished_job 在负载下会偶发失败，单独重跑能过就算通过，
   但失败要记进日志。
4. 报告 dev-two-tier-v4 能否快进 dev-two-tier，由用户合并。

五、用户快进 dev-two-tier 之后
1. 跑 C14：
   - 在主检出（树必须干净）上跑。先把 4dTrajectory/outputs/POOLED/post/post_train_20261006 改成可写；
   - 命令和 C10 最后一次 resume 一样（见阶段 C 的日志 §26），只把 --rounds 改成 14；
     所有路径都写成仓库相对路径（C10 就是这样启动的；用绝对路径时，执行器检查会拒绝，见日志 §26 的 launch 一行）；
   - worker 数按 C13 的内存规则（O15）定。
   跑完以后：
   - SHA256SUMS 加上新文件，目录改回只读；
   - 每一轮的 selection readout 写进阶段 C 的日志；
   - 报告给用户，由用户按 D7 在 14 轮里选一轮。
   C14 运行期间，不要把任何代码合进 dev-two-tier，因为主检出正在跑它。
2. C14 运行期间：先把 dev-two-tier 合进 dev-multi-control，然后继续多机控制，只写代码、只在合成输入上测试
   （rule 13：不做任何会拖慢 C14 的事）。
3. 清理已经合并的工作树和分支（只是 git 操作，C14 运行期间也可以做）：
   - 工作树 merge-a43、training-attitude、stage2-restart，以及它们的分支
     dev-two-tier-merge-a43、dev-training-attitude、dev-stage2-restart；
   - 分离的工作树 a25-build、stage2-real400；
   - 没有工作树的分支：docs-optimizer-multi-aircraft、dev-frontend-design、dev-multi-control-design、
     dev-c10-extension-design；
   - 用户合并 traffic-scenarios 之后：工作树 traffic-scenarios 和分支 dev-traffic-scenarios；
   - 第三步做完、合并之后：工作树 two-tier-v4-post 和分支 dev-two-tier-v4-post。
   每个工作树按这个顺序删：
   - 先用 find <工作树> -maxdepth 4 -type l 列出数据链接，逐个 unlink；
   - 再 git worktree remove；
   - 最后 git branch -d。不用 -D；-d 拒绝就停下来报告。
   以下不动：
   - 分支 dev-kaus-parts、dev-airport-embedding、dev-step9-one-commanded、wip-r32-leg-timing、dev-multi-control、main；
   - 工作树 kaus-parts、airport-embedding、step9-one-commanded、step9-run、multi-control；
   - ~/.claude/jobs 下的工作树。
4. C14 跑完以后：做多机控制在真实数据上的部分（MC0 的 D73 检查、D149 真实窗口逐位比对、MC1 普查，同上一份命令），
   然后报告 dev-multi-control 能否合并。

每一步的做法：单文件测试 → 独立审查（只审代码，审查者不能是作者）→ 用显式路径提交
（不用 git add -A，提交前看 git diff --cached --stat）→ 日志写一行。
你只能写这些设计文本：各文档 §0.3 的状态行、multi_control §0.3、你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_d_to_designer.md（整份重写），等用户定。
```
