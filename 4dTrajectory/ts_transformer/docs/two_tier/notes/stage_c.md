# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
C25 已合并，F5 由 fronter 做。新任务 C26（post_training §8 C26、§9 item 8；frontend D177 (15)，用户 2026-10-08 定）：
fronter 的导出现在读 WindowLoop._reading（私有：某个录制飞机读到故障点的那些步，D114）。
给它一个公开的读法；fronter 之后改读它，删掉私有读取。
路径相对 4dTrajectory/ts_transformer/。

一、代码（在 dev-two-tier-v4-post 上，先把 dev-two-tier 合进来）
1. experiments/post_window_loop.py：
   - 一个公开的名字，给出一个窗口里录制飞机读到故障点的步；名字你定，报告里写上。
   - 循环的行为不变。
2. 测试：
   - 在有故障点的窗口上，它给出的和读数按 D114 数的一致；
   - fronter 导出的现有测试照样通过（只读，不改导出）。
3. 改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者，审查和测试同时开始）
   → 显式路径提交 → 日志写一行 → 报告 dev-two-tier-v4-post 能否快进 dev-two-tier，由用户合并。

现在没有 campaign 在跑；下一个实验由用户定。
你只能写 post_training §0.3 的状态行和你的日志；requests 有新读法再写。
日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
