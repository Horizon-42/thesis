# Fronter：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_fronter_implementation_log.md`）。
2026-10-07，Claude 写，用户转发。

```
你的 12 条读法用户都接受了（frontend D160），其中第 11、12 条要改。路径相对 aeroviz-4d/src/。

一、两处小改（在 dev-frontend 上；先把 dev-two-tier 合进来）
1. 第 11 条：统计表的复飞列，列名改成 "go-arounds / sentence"，格子里写每句的复飞次数（不是百分比），下面照写计数。
2. 第 12 条：标签页上结局圆点的颜色，落地改成通过绿 #4ade80（utils/trainingWordColors.ts 已有的
   TRAINING_DECISION_PASS_COLOR），其他结局仍为红色。这样青色只表示 closed loop。
   "other sentences"、"other rounds" 叠加层里的结局颜色也一样改（都走 trainingOutcomeColour 这一处）。
3. 跑改动文件的测试（Vitest）→ 独立审查 → 显式路径提交 → 日志写一行；报告能否合并，由用户合并。
   这两处很小，不用做浏览器检查。

二、之后
- F3 等阶段 D 的 MC0 合进 dev-two-tier 再做；F4 等 MC4 之后。
- 做 F3 时，按 D160 (7) 再核对一次：静默的飞机能否为失去间隔负责，以阶段 D 的 MC0 代码为准。
- C14（C10 续训到 14 轮）运行期间，主机只能做轻活：只跑改动文件的测试，少开进程。

你能写的设计文本只有两样：frontend §0.3 的状态表，和你的日志。设计没说到的地方，写成读法放进
docs/two_tier/design/requests_from_fronter_to_designer.md（整份重写；已经定了的条目删掉），等用户定。
日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
