# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。

```
你是阶段 D（多机控制）的实现者。设计已定：docs/two_tier/design/multi_control.md（D140–D153）。
现在做版本 1 的 MC0 和 MC1；版本 2（D153，MC8 起）不做。路径相对 4dTrajectory/ts_transformer/。

先读：
- docs/two_tier/design/outline.md：§5（实现规则，尤其 rules 1、2、3、4、10、13）、§3 的 D131（复审标准）；
- docs/two_tier/design/multi_control.md：全文；
- vocabulary.md §6 item 5、prior.md §7 items 2、3、7、post_training.md §9：标「To be built」的条目就是你要建的接口；
- docs/two_tier/review_guide.md。

分支和工作树：
1. 等用户把 dev-multi-control-design 合进 dev-two-tier-v4（设计文字在那条分支上）。
2. 从 dev-two-tier-v4 建分支 dev-multi-control，工作树 .claude/worktrees/multi-control；
   忽略的数据目录按 outline §5 rule 1 用绝对路径链到 live 数据。
3. 每个里程碑之前、每次报告之前，把 dev-two-tier-v4 合进来。
4. 不在 dev-two-tier-v4 上提交（那是阶段 B 的分支）；不碰 .claude/worktrees/two-tier-v4-post（C10 正在跑）；
   不合并进 dev-two-tier，由用户合并。

C10 运行期间：只写代码，只用合成输入，只跑改动模块的测试（进程少，线程 1）。
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
7. 每一步：单文件测试 → 独立审查（只审代码，审查者不是作者）→ 显式路径提交（不用 git add -A，提交前看
   git diff --cached --stat）→ 日志一行。
8. 你只写 multi_control §0.3 的状态表和你的日志 readouts/2026-10-06_stage_d_implementation_log.md；
   设计没说到的地方写成读法，放进 docs/two_tier/design/requests_from_d_to_designer.md（整份重写），等用户定。
   这三个文件提交到 dev-two-tier：用一个临时工作树加快进，只放这三个文件。

C10 结束、并且阶段 B 的日志写明它的测量做完之后（B14 检查、C13 的 GPU 检查、model_speed、重导；outline §5 rule 13）：
1. MC0 改了 autopilot/：跑 vocabulary D73 的检查（closed_loop_start_check 先跑三项），日志写最大差。
2. D149 的真实窗口检查：正式普查 outputs/POOLED/post/windows_20261006 里每种窗口、每个机场各 10 个（seed 1337）。
   改动前的输出用 MC0 之前的提交在临时工作树里跑；和改动后逐位比对（CPU 单线程）。
3. MC1 的普查：train 和 select 日，L = 0、5、10、20 min，压缩窗口 c_min = 0.6、0.8，写在 scratch；报告给用户选 O16。
4. 报告 dev-multi-control 能否合并进 dev-two-tier-v4，由用户合并。

规模估计：MC0 阶段 A、B 约 200 行；阶段 C 约 300 行改动，加 token 附加部分约 50 行；multi/ 约 150 行；
MC1 runner 约 200 行；测试约 800 行。
```
