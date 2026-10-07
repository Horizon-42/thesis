# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-07，Claude 写，用户转发。

```
你的 requests 1–7 都已写进设计（post_training D161、D162、D165、D167、D168、D169，§9 items 10、11）。
设计文档以后不记录具体实验信息：某次运行的目录、选了哪一轮、量出来的数字都写在 intents.json 和你的日志里，
设计只写方法和用户定的规则。§0.3 的状态行只写里程碑做到哪里、提交号，结果指向日志的节号。
路径相对 4dTrajectory/ts_transformer/。

现在 K = 16 的 campaign 正在 run-post-branch16 运行工作树里跑：这段时间只写代码，只跑改动文件的测试，少开进程
（rule 13），不动那个运行工作树。

一、代码（在 dev-two-tier-v4-post 上，先把 dev-two-tier 合进来；每项单独提交）
1. C19（D168，梯度范数裁剪）：Settings.clip_norm，默认 None（不裁剪，即旧行为）；update_step 和 landed_step
   在优化器走一步之前裁剪；round.json 记被裁剪的更新所占比例。测试按 §8 C19。要在调大学习率的那批实验之前做好。
2. C18（D167，续跑复用第一次的内存测量）：第一次启动把 O15 的测量写进 campaign.json；续跑只检查空闲内存；
   只有记录里没有测量、或者要的 worker 超过测量允许时才重新测。测试按 §8 C18。
3. C20（D169，每轮训练多遍）：Settings.epochs，默认 1；在 K = 16 那个 campaign 跑完之后再做。测试按 §8 C20。
新设置都按用户的长期许可：默认值等于旧代码的行为，没有这个字段的旧记录按默认值读，不改任何已有记录。
每项：改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者）→ 显式路径提交 → 日志写一行。
做完报告 dev-two-tier-v4-post 能否快进 dev-two-tier，由用户合并。

二、实验
K = 16 跑完以后，下一个 campaign 由用户定（你日志里的候选）。每个 campaign 都从用户合并后的提交建运行工作树（outline D163），
主机和 GPU 上没有别的任务时跑；intent 在发布前写进 intents.json；跑完数据设为只读并写 SHA256SUMS，结果写进日志。

你只能写这些设计文本：post_training §0.3 的状态行和你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_c_to_designer.md（整份重写，已经写进设计的 1–7 条删掉），等用户定。
日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
