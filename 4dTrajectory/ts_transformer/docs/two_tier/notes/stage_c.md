# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-07，Claude 写，用户转发。

```
用户接受了 P49，按你的提议做（post_training D165，里程碑 C17）。起点是 post_train_20261006 的第 8 轮（用户选的）。
其余设置都取你的提议：N = 8、每个窗口留奖励最高的一句落地句子、C10 的学习率和损失权重、6 轮。
设计里补了一条：这是新的训练方法，所以它的 campaign 用自己的 schema 名；checkpoint 仍是 ts-post-checkpoint-v1，
identity 里写明用的是哪种方法，这样以后的起点（D162、阶段 D 的 D164）照样能打开它。
路径相对 4dTrajectory/ts_transformer/。

一、代码（在 dev-two-tier-v4-post 上，先把 dev-two-tier 合进来）
1. 按 post_training §8 C17 写：一个新的 runner，用 §9 item 11 的 campaign 骨架；
   post_validation 和 Training 导出都要能读它的 campaign。
2. 测试按 C17 写的做。
3. 跑改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者）→ 显式路径提交 → 日志写一行。
4. 在 docs/experiments/intents.json 写这个 campaign 的 intent。
5. 报告 dev-two-tier-v4-post 能否快进 dev-two-tier，由用户合并。

二、跑（用户合并之后）
1. 先确认主机和 GPU 上没有别的任务（rule 13）。
2. 在用户合并后的那个提交上建运行工作树（outline D163），数据目录用绝对路径链到 live 数据，在里面跑：
   起点用第 8 轮，种子不能是 1337，其余设置同 C10。
3. 跑完以后：数据设为只读，写 SHA256SUMS；每一轮的 selection readout 写进日志，和 C10 的 14 轮放在一起对比；
   运行工作树先断开数据链接再删掉。
4. 报告给用户，由用户按 D7 选轮。验证读数等用户说了再做。

你只能写这些设计文本：post_training §0.3 的状态行和你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_c_to_designer.md（整份重写，已经定了的条目删掉），等用户定。
日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
