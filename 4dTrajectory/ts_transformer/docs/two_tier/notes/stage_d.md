# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-07，Claude 写，用户转发。

```
设计更新（multi_control）：
- 你的第 46–48 条写成了 D172，MC5 也补了一句：说话 worker 按 profile 定数量，每种跨度一个批大小，
  profile 里每种跨度的批只说一遍。
- 用户定（2026-10-07，O19）：MC6 照旧用 D142、D143，也就是一次只变一架飞机的分支组。
  阶段 C 新加的价值网络（post_training D171）D 现在不用，原因有三：
  1. 它估计的是一架飞机的奖励 r，不是整个窗口的 W；
  2. 它读的是录制飞机的未来，而 D 里被指挥飞机的未来取决于模型说的话，没有记录可读；
  3. 它只看一架被指挥飞机。
  等 C 的价值训练读完、确实比分支训练好，再由 Claude 设计 D 自己的价值网络。
- C 接下来做 D170：分支间隔 branch_every_s 和只算到下一个分支点 segment_only，两个都是 C 的设置，默认等于旧行为。
  branch_points 和 post.branches.samples 加的参数都有默认值；你的 multi/credit.py 和你的 pass 不用改，D 的测试要原样通过。
  D 的设置以后要不要用它们，由用户定。
- C 从 D171 的轮次出发时 checkpoint 格式不变（V 单独存在 value.pt 里），D164 的起点照常读，从不读 V。
路径相对 4dTrajectory/ts_transformer/。

一、代码（在 dev-multi-control 上；先做这一步，好让用户在 C 改共用代码之前合并你的分支）
1. 把 dev-two-tier 合进来，带进 C18、C19、C20：
   - pass 那一行要保留 C 对 CPU 上 worker 的检查（--speak-device，ac455d2e）；你的分支去掉了它，合并时留下 C 的
     （C 的 requests 第 3 条）。
   - C18 的 workers_fit 现在对每个调用者都减去已经占用的内存；你的 profiled_fit 原来就减 passed["now"]，
     合并后确认没有减两次。
   - C19 已经改了 multi_train.py 和它的两个测试（open_campaign 传 settings_type=MultiSettings），保留。
   - clip_norm、epochs：D 用默认值（不裁剪、一遍）。
2. 改动文件的测试和 test_architecture（C 的 campaign 正在跑：只跑这些，少开进程，rule 13）
   → 独立审查合并结果（只审代码，审查者不能是作者，审查和测试同时开始）→ 显式路径提交
   （不用 git add -A，提交前看 git diff --cached --stat）→ 日志写一行
   → 报告 dev-multi-control 能否快进 dev-two-tier，由用户合并。

二、MC5（主机和 GPU 上没有别的任务时才跑；现在 C 的 post_k16lr3_20261007 在跑，先不跑）
主机空出来以后，用 fffe900d 的代码在正式规模上按每种跨度把 profile 跑完（用 systemd 单元），
然后把 MC6 的建议设置报告给用户。

三、之后
MC6 的设置和起点（C 的哪个 campaign 的哪一轮）由用户定，从运行工作树启动（outline D163）。

你只能写这些设计文本：各文档 §0.3 的状态行、multi_control §0.3、你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_d_to_designer.md（整份重写；第 46–48 条已写进 D172，删掉），等用户定。
```
