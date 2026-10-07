# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-07，Claude 写，用户转发。

```
阶段 B 的收尾和清理都做完了。C 的新改动里有两处要带到 D（用户 2026-10-07 定）：
- 起点（multi_control D164，post_training D162、§9 item 12）：D 从 C 选出的那一轮出发时，和 C 的"从某一轮出发"用同一个函数、
  同样的规则：
  - settings 里记 {campaign, round, checkpoint_sha256}；
  - 字节、identity（base、mask、traffic 形状、轮号）不对就按名字拒绝，正式 campaign 不从 smoke campaign 出发；
  - D 的种子必须和源 campaign 不同；
  - 优化器重新开始，pull 项拉向 base。
  现在这件事有两份代码：C16 的 campaign_start（dev-two-tier-v4-post，3c10e08f，等用户合并）和你的 open_round
  （a1ea664f）。合成一个函数（§9 item 12），C 的 campaign_start 和 D 的起点都调用它。
- 正式实验（MC6、MC9）从运行工作树启动（outline D163）：在用户合并后的提交上建一个分离的工作树，不用主检出；
  intent 在发布前写，启动时不检查。
C14（加轮数）、this_checkout 读路径、P47 都在共享的 open_campaign 里，你的分支已经有了，不用另做。
C15 上限读数暂时不用于 D。
路径相对 4dTrajectory/ts_transformer/。

一、现在
继续多机控制。C 的上限读数正在 GPU 上跑（2–3 小时），这段时间只写代码、只在合成输入上测试（rule 13）。

二、用户把 dev-two-tier-v4-post（C15、C16）合进 dev-two-tier 之后
1. 把 dev-two-tier 合进 dev-multi-control。experiments/post_train.py 会有冲突：
   - C15 给 read_batch、Speakers.read、_read、readout_numbers 加了第几次抽样的参数（draw，第 0 次逐位不变）；
     你的 Stage 骨架里的读数照样把 draw 传下去，默认 0；
   - C16 加了 Settings.start、start_checkpoint、campaign_start，删掉了启动时的 intent 检查。
   两边都保留。
2. 按 §9 item 12 把 open_round 和 campaign_start 的检查合成一个函数，两处都调用它；D 的起点按 D164 写。
3. 测试：
   - C 的测试原样通过（C16 的每种拒绝、C15 第 0 次抽样逐位不变）；
   - D 的起点：每种拒绝按名字（字节不对、另一个 base/mask/traffic 形状/轮号、smoke 源、种子相同）；
   - D149 的逐位参照检查照样通过。
4. 单文件测试 → 独立审查（只审代码，审查者不能是作者）→ 显式路径提交 → 日志写一行。

三、上限读数跑完以后
做多机控制在真实数据上的部分（MC0 的 D73 检查、D149 真实窗口逐位比对、MC1 普查），然后报告 dev-multi-control 能否合并。

每一步的做法：单文件测试 → 独立审查 → 用显式路径提交（不用 git add -A，提交前看 git diff --cached --stat）→ 日志写一行。
你只能写这些设计文本：各文档 §0.3 的状态行、multi_control §0.3、你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_d_to_designer.md（整份重写），等用户定。
```
