# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。

```
你是阶段 D（多机控制）的实现者，同时接手阶段 B 的收尾（outline §5 rule 1，用户 2026-10-06）。
路径相对 4dTrajectory/ts_transformer/。

优先级：C10（阶段 C 的正式训练）还在跑，跑完前主机不能做大测量，所以：
- C10 运行期间：先做多机控制的编码（下面「一」）。
- C10 结束后：先做阶段 B 的收尾（「二」），它卡着 dev-two-tier-v4 的合并；再做多机控制在真实数据上的检查和普查（「三」）。
版本 2（multi_control D153，MC8 起）现在不做。

先读：
- docs/two_tier/design/outline.md：§4 items 5–7、§5（尤其 rules 1、2、3、4、10、13）、§3 的 D131（复审标准）；
- docs/two_tier/design/multi_control.md 全文；
- vocabulary.md §6 item 5、prior.md §7 items 2、3、7、post_training.md §9：标「To be built」的条目就是你要建的接口；
- docs/two_tier/review_guide.md；
- 阶段 B 的日志 readouts/2026-10-05_stage_b_implementation_log.md §4、§5（在 dev-two-tier-v4 上），
  以及 readouts/2026-10-06_training_view_and_speed_check.zh.md：收尾各步的来由和细节。

分支和工作树：
1. 等用户把 dev-multi-control-design（设计文字）合进 dev-two-tier-v4。
2. 在 .claude/worktrees/two-tier-v4 里把 dev-two-tier 合进 dev-two-tier-v4。dev-two-tier 带着阶段 C 的修复 832555a5
   （一次更新分块算，post.loss.update_step，设计见 post_training §2 item 5），它和 C13 都改了 experiments/post_train.py：
   两边都保留，跑 test_post_* 各单文件，审查后提交。
3. 从 dev-two-tier-v4 建分支 dev-multi-control，工作树 .claude/worktrees/multi-control；忽略的数据目录按 outline §5 rule 1
   用绝对路径链到 live 数据。多机控制只在这条分支上做；阶段 B 的收尾在 dev-two-tier-v4 上做。
4. 多机控制的每个里程碑之前、每次报告之前，把 dev-two-tier-v4 合进 dev-multi-control。
5. 不碰 .claude/worktrees/two-tier-v4-post（C10 正在跑）；不合并进 dev-two-tier，由用户合并。

一、C10 运行期间：多机控制的编码（只写代码，只用合成输入，只跑改动模块的测试，进程少，线程 1）
1. MC0 · 阶段 A（vocabulary §6 item 5）：
   - Start.moved 收每架飞机的 join tick；
   - Loop 给执行器每架的起始周期：(join tick + s) × 一个 Δ 行的周期数，用 Executor(start_cycle=…)；
   - 没开始的飞机不被听到：语法和复飞上限只读已开始的飞机；
   - 每架的句子时间各算各的；Loop.copy 保留 join tick。
   测试：join tick 全为 0 时和今天逐位相同；在 tick j 加入的飞机，用同样的词，飞出的状态和它单独飞时相差不超过 STATE_BOUND_M。
2. MC0 · 阶段 B（prior §7 items 2、3、7）：
   - LoopRows 按每架的 join tick 取行（压缩窗口用挪动后的 entry 时刻）；
   - 说话器的一行里可以同时有未加入、观察段、说话段的飞机；
   - SpeakingLoop.step 每个 tick 推进一次；
   - 循环中可以给指定飞机加一条着陆（LandingIndex 的检查保留），copy 一起带上。
   测试：multi_control §6.2 items 2–3 和 §6.3 的 Tests。
3. MC0 · 静默飞机（multi_control §6.2 item 4）：确认说话器每一列都接受调用方的 mask，并且首个预测步之后语法在每一列都允许
   "unchanged"。在只许 "unchanged" 的 mask 下，飞机按生效的词飞，对数概率为 0。语法不允许就停下来报告。
4. MC0 · 阶段 C（post_training §9 items 1、2、3、7、8、9、11、12 的 To be built）：
   - 每改一处，先把改动前代码在合成窗口上的输出存下来作参照，改完逐位比对（CPU 单线程）；
   - item 7 的 token 附加部分放在阶段 C 的 token 网络旁边，投影从零开始，格式名另起；
     不要改 post-edges-v1 的特征，不要改 post-traffic-attention-v2 的形状，否则 C10 的 checkpoint 读不进来。
5. MC1 · multi/ 里普查要用的部分（multi_control §10）：窗口抽取（锚点、跨度 L、压缩窗口）和普查，及测试。
6. MC1 · runner experiments/multi_windows.py 及测试；不在真实数据上跑。

二、C10 结束后，先做阶段 B 的收尾（在 dev-two-tier-v4 上；主机和 GPU 上没有别的任务时，一项一项来）
1. B14 的真实数据检查（prior §12 B14）：用 BATCH 把 B5 的 KRDU 折在 select 日重说一遍（写在 scratch），
   句子必须和它的读数逐词相同；不同就停下来报告。在这一步之前，不要用 BATCH 跑任何真实数据。
2. C13 在 GPU 上的检查（post_training §8 C13）：跑一轮，单进程和 N 个 worker 的结果必须相同。
3. 对 base 测速（outline §6.2 item 10，model_speed）：base 是 4dTrajectory/outputs/POOLED/prior/prior_base_20261006，
   写进 4dTrajectory/outputs/POOLED/speed/ 下的新目录。
4. 跑 sha256sum -c 校验 base 的 SHA256SUMS；重导 A 的 closed_loop_v12_20261005 和 B 的 prior_sets_20261006
   （同名，新格式，B 的带 source.speed，outline §6.2 item 9）；再跑一遍 sha256sum -c，结果必须一致。旧的索引和集合先留着。
5. 浏览器检查交给一次性子代理（测试栈从 two-tier-v4 工作树起，端口和停止命令写进报告）。
6. 报告 dev-two-tier-v4 能否快进 dev-two-tier，由用户合并。
这些结果写进阶段 B 的日志（新的一节，注明由阶段 D 的实现者完成），并更新 vocabulary、prior、post_training 的 §0.3 状态行。

三、然后做多机控制在真实数据上的部分（在 dev-multi-control 上）
1. MC0 改了 autopilot/：跑 vocabulary D73 的检查（closed_loop_start_check 先跑三项），日志写最大差。
2. D149 的真实窗口检查：正式普查 outputs/POOLED/post/windows_20261006 里每种窗口、每个机场各 10 个（seed 1337）。
   改动前的输出用 MC0 之前的提交在临时工作树里跑；和改动后逐位比对（CPU 单线程）。
3. MC1 的普查：train 和 select 日，L = 0、5、10、20 min，压缩窗口 c_min = 0.6、0.8，写在 scratch；报告给用户选 O16。
4. 报告 dev-multi-control 能否合并进 dev-two-tier-v4，由用户合并。

每一步：单文件测试 → 独立审查（只审代码，审查者不是作者）→ 显式路径提交（不用 git add -A，提交前看
git diff --cached --stat）→ 日志一行。
你写的设计文本只有：multi_control §0.3 的状态表、各文档 §0.3 的状态行、你的日志 readouts/2026-10-06_stage_d_implementation_log.md
（阶段 B 收尾的结果写进阶段 B 的日志）。设计没说到的地方写成读法，放进 docs/two_tier/design/requests_from_d_to_designer.md
（整份重写），等用户定。日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。

规模估计：MC0 阶段 A、B 约 200 行；阶段 C 约 300 行改动，加 token 附加部分约 50 行；multi/ 约 150 行；
MC1 runner 约 200 行；测试约 800 行。阶段 B 的收尾只跑已有的 runner，不写新代码。
```
