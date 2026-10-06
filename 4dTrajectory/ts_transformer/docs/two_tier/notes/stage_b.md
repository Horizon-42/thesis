# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（`b238c79d`）里的 MC0、MC1 撤回：阶段 D 交给它自己的实现者（outline §5 rule 1）。

```
上一条命令里的 MC0、MC1 撤回，不要做：多机控制（阶段 D）交给新的实现者 D，在它自己的分支 dev-multi-control 上做。
依据：outline D131、D139、§5 rules 1、13；prior §12 B14；post_training §2 item 5、§8 C13。

现在：
1. 把 dev-two-tier 合进 dev-two-tier-v4。dev-two-tier 带着阶段 C 的修复 832555a5：一次更新分块算（post.loss.update_step，
   设计见 post_training §2 item 5）。它和 C13 都改了 experiments/post_train.py，两边都保留；跑 test_post_* 各单文件。
2. 用户会把 dev-multi-control-design（多机控制的设计文字，只有文档）合进 dev-two-tier-v4，你不用做别的。
3. 阶段 D 会改 autopilot/、prior/、post/ 里标「To be built」的接口（vocabulary §6 item 5、prior §7 items 2、3、7、
   post_training §9），改在 dev-multi-control 上。你在 dev-two-tier-v4 上不要动这些接口。

C10 结束后（主机和 GPU 上没有别的任务时），按这个顺序：
1. B14 的真实数据检查：用 BATCH 把 B5 的 KRDU 折在 select 日重说一遍（写在 scratch），句子必须逐词相同；
   不同就停下来报告。在这一步之前，不要用 BATCH 跑任何真实数据。
2. C13 在 GPU 上的检查：跑一轮，单进程和 N 个 worker 的结果必须相同。
3. 对 base 测速（model_speed），写进 4dTrajectory/outputs/POOLED/speed/ 下的新目录。
4. 跑 sha256sum -c 校验 base 的 SHA256SUMS；重导 A 和 B 的集合（B 的带 source.speed）；
   再跑一遍 sha256sum -c，结果必须一致。
5. 浏览器检查交给一次性子代理。
6. 报告 dev-two-tier-v4 能否快进，由用户合并。
你的 1–4 做完之前，阶段 D 不在真实数据上跑任何东西（outline §5 rule 13）；做完在日志里写一行，让 D 知道。

C10 在 two-tier-v4-post 工作树里跑：不合并进 dev-two-tier，不碰那个工作树。
```
