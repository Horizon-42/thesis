# Fronter：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_fronter_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。

```
你是 fronter：Training 视图的实现者。你负责前端的 Training 页、后端的 Training 路由和实时航段，以及各阶段的 Training 导出。
设计：docs/two_tier/design/frontend.md（D154–D156）。路径相对 4dTrajectory/ts_transformer/，前端路径相对 aeroviz-4d/src/。

优先级：C14（C10 续训到 14 轮）运行期间，主机只能做轻活。
- 最先做 F0（用户 2026-10-07 的三处修正，frontend D159、§8 F0）：把手上正在做的这一步做完、审查、提交，然后马上做 F0，
  做完再回到 F1、F2。用户要求只改这三处、代码改动尽量少、不改后端和格式。
- 先做 F1（光标滑块）；
- 再做 F2（多架被指挥飞机的窗口格式，先只用阶段 C 的一架）；
- F3 等阶段 D 的 MC0 合进 dev-two-tier 之后做；
- F4 等阶段 D 的 MC4 之后做。

先读：
- frontend.md 全文；
- outline.md §4 item 8、§5（尤其 rules 1、2、3、4、5、7、10、13）、§3 的 D131（复审标准）；
- multi_control.md §2.4、§3、§5（阶段 D 的窗口、静默飞机、按对的失去间隔）；
- post_training.md §9 items 1、3、8、13；
- docs/two_tier/review_guide.md；
- 阶段 B 的日志 readouts/2026-10-05_stage_b_implementation_log.md §4（统一布局是怎么建的）。

分支和工作树：
1. frontend.md 已经在 dev-two-tier 上（7e581b88 起），不用再等。
2. 从 dev-two-tier 建分支 dev-frontend，工作树 .claude/worktrees/frontend。
   忽略的数据目录按 outline §5 rule 1 用绝对路径链到 live 数据；不在 live 数据里写任何东西。
3. 每个里程碑之前、每次报告之前，把 dev-two-tier 合进来。不合并进 dev-two-tier，由用户合并。
4. 不在 dev-two-tier-v4 上提交（阶段 D 的实现者在那里做收尾和 C14）。
   阶段 D 重导 A、B 的集合之前，不改阶段 A、B 的导出格式（frontend §8）。

F0 用户的三处修正（frontend D159、§3 items 3、6、11、§8 F0；参考 v3 的设计 aeroviz-4d/docs/36-2026-09-20-training-module.zh.md
§4.4、§4.8、§4.9、§4.11，只参考设计，不搬旧代码）：
- 详情页只留两节：「The set and the experiment」和「The models' statistics」。统计表每种句子一行（closed loop、base 的样本、
  后训练各轮，按标签页顺序，行首放颜色块），两组列：this set（按集合里的句子数）、the formal readout（用结果路由已经给的计数），
  列：landed、go-around、timed out，窗口阶段再加失去间隔和平均奖励；每格百分比，下方写计数，没有计数的写 "—"。
  其余各节（标注、每轮 loss、选择、测速、检查）连同它们的测试删掉；不改后端，也不改结果路由。
- 地面上的线（航向带被判定的行、出界的红线、航迹的地面投影）一律改成虚线，宽度不超过 3 px，不透明度 0.6；空中的航迹保持实线。
- 颜色按句子种类（§3 item 11 的表）：观测航迹近白、closed loop 青色（这两样不变）、base 品红 #d946ef、后训练各轮黄绿 #a3e635、
  阶段 D 的模型树莓红 #b82e7a。只在 utils/trainingWordColors.ts 里定义一张表；三维的飞行航迹和它的地面投影、读回图的飞行线、
  图例、标签页上的色块都用它。名字和数字用正文颜色，色块放在旁边。
- 测试见 §8 F0；浏览器检查 A、B、C 三个阶段，交给一次性子代理。

F1 光标滑块（frontend §6.1，D155）：
- 在 components/training/ 里做一个共用的滑块；各阶段的 session 给它光标和标记。
- 去掉 B 的可点概率条（TrainingPriorSession.tsx 里 AtTheCursor 的 SVG），保留它的读数行。
- 各阶段给自己的标记：
  - A：带修正词的行；
  - B：样本页上的复飞概率线；
  - C：失去间隔；
  - 每个阶段都标首个预测步。
- 测试见 frontend §8 F1。
- 阶段 D 的浏览器检查顺带看到三处小问题（阶段 B 日志 §6 末尾），F1 里一起改：
  - docs/experiments/intents.json 里阶段 A 的 intent 还写着 index_v4.json，改成 index_v5.json；
  - 在阶段 B 选了样本页时，读回图窗口的标题仍写 "closed loop · Δ 4 s"（数据其实是样本的），改成样本的名字；
  - 阶段 B 左栏 "At the cursor" 的标签和文字重叠（滑块换掉概率条时一起解决）。
- frontend §0.3 第一行（The one layout）：C10 之后的那几步已由阶段 D 做完（阶段 B 日志 §6），把状态改成已完成。
- 浏览器检查交给一次性子代理。测试栈从你的工作树起，用自己的端口和 scratch 数据目录，放新格式的冒烟集
  （可以复制 B 当时的 /tmp/claude-1000/-home-supercomputing-studys-thesis/7767b9f9-586d-4d01-9b82-2949db171a8a/scratchpad/public_v4，
  不在了就用现有导出在 scratch 里生成）。报告给出地址和停止命令。

F2 多架被指挥飞机的窗口（frontend §5.7、§8 F2）：
- 窗口样本 v4，以及 C 和 D 共用的索引格式；
- 阶段 C 的导出把窗口写成只有一架的列表；
- 窗口实时航段的请求带上飞机，前后端都换新名字；
- 视图的读取器、session 和场景层改成读列表；C 的视图外观不变，只有一架时不显示飞机条。

F3（等阶段 D 的 MC0 合进 dev-two-tier）：frontend §5.1–§5.6、§8 F3。
F4（等阶段 D 的 MC4）：frontend §8 F4。

每一步：测试（Vitest，加改动文件的 pytest；GPU 检查、测速和 C14 运行期间少开进程）→ 独立审查（只审代码，审查者不是作者）
→ 显式路径提交（不用 git add -A，提交前看 git diff --cached --stat）→ 日志写一行。

你能写的设计文本只有两样：frontend §0.3 的状态表，和你的日志 readouts/2026-10-06_fronter_implementation_log.md。
设计没说到的地方，写成读法放进 docs/two_tier/design/requests_from_fronter_to_designer.md（整份重写），等用户定。
日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。

规模估计：F1 约 150 行，F2 约 200 行，F3 约 300 行，各另加测试。
```
