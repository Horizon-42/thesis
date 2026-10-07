# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-07，Claude 写，用户转发。

```
C14 做完了，用户已经看过你的 requests。用户的决定（post_training D161、D162，outline D163）：
- P48 上限读数按你的方案做：N = 32，模型取起点、第 6 轮、第 8 轮，只用 select 日（D161，里程碑 C15）。
- campaign 可以从另一个 campaign 的某一轮出发，按你的读法做（D162，里程碑 C16）。
  补一条规则：从某一轮出发的 campaign，种子必须和源 campaign 不同，相同就按名字拒绝。
- 正式实验以后都从专用的运行工作树启动，不再用主检出（D163）。intent 在启动时不检查，发布时由发布器检查。
- P45 接受（窗口集合的航迹不取整），已写进 D125。
- 第 4 条（路径字段在三个地方各写一遍）是 S3，已记录，不再处理。第 5 条已由 D163 解决。
路径相对 4dTrajectory/ts_transformer/。

一、代码（在 dev-two-tier-v4-post 上，先把 dev-two-tier 合进来）
1. C15、C16 按 post_training §8 写，包括 D162 的种子规则和它的测试。
2. 去掉启动时对 intent 的检查（你已经在做）。
3. 跑改动文件的测试和 test_architecture → 独立审查 → 显式路径提交 → 日志写一行。
4. 报告 dev-two-tier-v4-post 能否快进 dev-two-tier，由用户合并。

二、跑上限读数（用户合并之后）
1. 先确认主机和 GPU 上没有别的任务（rule 13）。
2. 在合并后的提交上建一个分离的运行工作树：git worktree add --detach .claude/worktrees/run-post-ceiling <提交>；
   数据目录按 outline §5 rule 1 用绝对路径链到 live 数据。
3. 在这个工作树里跑。第 0 次抽样必须复现每一轮 round.json 里的读数，对不上就停下来报告。
4. 跑完以后：数据设为只读，写 SHA256SUMS；结果表写进日志；运行工作树先断开数据链接再删掉。
5. 报告给用户：每个模型、每个机场，前 n 次抽样里至少落地一次的比例（n = 1、2、4、…、32），
   每次都失败的窗口，以及这些窗口在三个模型之间的重叠。不下结论，结论由用户看了再定（D7）。

验证读数（D7 选出的那一轮）等用户说了再做。

你只能写这些设计文本：post_training §0.3 的状态行和你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_c_to_designer.md（整份重写，已经定了的条目删掉），等用户定。
日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
