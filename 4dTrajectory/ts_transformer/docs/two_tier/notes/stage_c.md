# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
1. 你对 D175 的读法全部接受，写进了 D175：
   - "没有生效跑道就用记录跑道"永远不会生效，删掉了；
   - no_landing、字段 4 的判法、字段 5 两次读都给、边界值归上一档，都按你的。
2. 你的 --windows 请求，用户选了"完整通用"，写成 post_training D176、里程碑 C25（§8），前端是 frontend F5：
   (1) 窗口清单：一个通用格式 ts-window-list-v1，C、D 共用，只能列 select 窗口，不把窗口移出 select。
       不只用于丢间隔：任何规则选出来的 select 窗口都能用。
   (2) runner experiments/window_list.py（R67）：按一条规则从逐窗口读数生成清单——在给定的几次读数里，
       任一次（或加 --all 每一次）结局属于 --outcomes。读 D175 的 per_window.jsonl 和 D161 的 draw 0。
   (3) 导出器的 --windows 放进 C、D 共用的导出代码，样本升 v5（cohort 分 drawn / listed 两种）。
       按 frontend §8，导出器、样本格式和前端都归 fronter，所以 (3) 由 fronter 在 F5 里做，不是你。
路径相对 4dTrajectory/ts_transformer/。

一、代码：C25（在 dev-two-tier-v4-post 上，先把 dev-two-tier 合进来）
1. post/window_lists.py：ts-window-list-v1 的写和读。
   - 按名字拒绝：别的 schema、split 不是 select、缺 place 或 identity 的窗口。
   - identity：阶段 C 是被指挥航班、row0_s、kind；阶段 D 是锚定航班、row0_s、span。
   - 头部写：所索引的 selection 窗口（select seed、每机场窗口数，D 加 spans）、一句说明、读过的读数（路径 + sha256）。
2. experiments/window_list.py 和 runners.md R67。
3. 测试按 §8 C25。
4. 用 runner 把手工生成的 outputs/POOLED/post/loss_windows_select_20261008 重新生成一遍
   （三次读数任一次丢间隔），核对窗口完全相同。相同之后删掉手工那份目录（root CLAUDE.md：临时代码生成的产物
   被正式的取代后删除，用户已同意）。新清单只读，写 SHA256SUMS。
5. 改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者，审查和测试同时开始）
   → 显式路径提交（提交前看 git diff --cached --stat）→ 日志写一行
   → 报告 dev-two-tier-v4-post 能否快进 dev-two-tier，由用户合并。合并后 fronter 做 F5。

二、导出不归你：两个按清单导出的集合由 fronter 在 F5 之后导（用你提议的两组：C10 的 start 和 8；P55 的 start 和 5）。

你只能写这些设计文本：post_training §0.3 的状态行和你的日志。requests_from_c_to_designer.md 整份重写：第 1 条已处理，
删掉。日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
