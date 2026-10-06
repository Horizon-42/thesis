# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-06 深夜，Claude 写，用户转发。

```
C10 已经跑完（十轮，只读）。用户决定：把 C10 续训到 14 轮，作为同一个 campaign（post_training D157，里程碑 C14）。
这件事交给你做：写代码，再跑新增的 4 轮。新的 4 轮用合并后的代码跑，包括 C13。
路径相对 4dTrajectory/ts_transformer/。

先读：post_training D157、§0.4、§8 C14；outline §4 items 3、6、9，§5 rule 1、rule 13。

一、分支
1. 先在 .claude/worktrees/two-tier-v4-post 里把 dev-two-tier 合过来：git merge --ff-only dev-two-tier。
   dev-two-tier-v4-post 是 dev-two-tier 的祖先，应该能快进；不能快进就停下来报告。
2. C14 只在 dev-two-tier-v4-post 上做。只改 C14 需要的代码：experiments/post_train.py、experiments/post_validation.py
   和它们的测试。其他代码一律不动（阶段 D 的实现者和 fronter 在别的分支上开发）。

二、C14 的代码（post_training §8 C14）
1. open_campaign：续跑时只允许轮数变大，其他设置和输入都必须相同。
   campaign.json 记下每次改轮数：时间、旧轮数、新轮数、commit、checks。
2. 记录里的输入路径用 this_checkout 映射后再比对（experiments/training_export.py 里的那一份，像 model_speed 那样
   import，不要复制）。post_validation 也这样读。记录里原来的路径不改，每次续跑在自己那一条里写下实际读到的路径。
3. 测试按 C14 写的做。另外在 experiments/ 里查所有读 campaign 记录 "inputs" 的地方，确认都经过 this_checkout。
4. 在 docs/experiments/intents.json 里写新 4 轮的 intent，和代码一起提交。必须在启动之前提交。
5. 跑改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者）→ 用显式路径提交（不用 git add -A，
   提交前看 git diff --cached --stat）→ 日志写一行。
6. 报告 dev-two-tier-v4-post 能否快进 dev-two-tier，由用户合并。

三、启动新的 4 轮（用户合并之后）
1. 先等阶段 D 的实现者报告：C13 的 GPU 检查已经通过，测速和重导这些占 GPU、要计时的步骤都已做完
   （见 D 的日志 readouts/2026-10-06_stage_d_implementation_log.md 和阶段 B 日志里新开的那一节）。
   看一眼主机和 GPU，确认上面没有别的任务（rule 13）。
2. 在主检出上跑，树必须干净：
   - 先把 4dTrajectory/outputs/POOLED/post/post_train_20261006 改成可写；
   - 命令和 C10 最后一次 resume 一样（见日志 §26），只把 --rounds 改成 14；
   - 所有路径都写成仓库相对路径（用绝对路径时执行器检查会拒绝，见日志 §26 的 launch 一行）；
   - worker 数按 C13 的内存规则（O15）定；
   - 启动时就设好结束和出错的通知。
3. 运行期间不改主检出里的任何文件，也不往 dev-two-tier 合代码。
4. 跑完以后：
   - SHA256SUMS 加上新文件，目录改回只读；
   - 每一轮的 selection readout 写进日志；
   - 报告给用户，由用户按 D7 在 14 轮里选一轮。验证读数等用户选完再说。

你只能写这些设计文本：post_training §0.3 的状态行和你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_c_to_designer.md（整份重写），等用户定。
日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
