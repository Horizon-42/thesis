# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（统一布局，D133）已做完（`38297528`、`e0aceffd`，未合并）。

```
统一布局的第二部分：结果页、词的包络、三个阶段导出共用。
依据：outline §6.2（已重写）、D134、D135（用户 2026-10-06）。
用户在 D133 的特许之外再特许：可以改 A 和 C 的 Training 导出及其格式（experiments/training_export.py、
instructions/training_files.py、experiments/post_training_export.py、post/training_files.py），
并用新格式重导 A 和 B 已发布的集合。

1. 在 dev-training-layout 上先合并 dev-two-tier（含 D134、D135 的提交）。
2. 详情页按 §6.2 第 3、4 条重做：
   - 第一节「The set and the experiment」：意图，加一行来源（集合来自哪个读数或战役的路径）；
     去掉「What this view shows」和 Row inspector。
   - A：Labelling（只给 train 和 select）、Closed loop、Flown flights、Vocabulary。
   - B：Free generation、Training、Validation（只在基础模型的 val 集）、The choice。
   - C：Rounds、The checks、The window。
   - 后端 GET /training/results?stage=&airport=&set=：只读集合的 source / model 指向的、4dTrajectory/outputs/ 下的文件；
     只回答第 3 条列的字段，不回整个文件；A 读数里的 val 块绝不进任何回答；文件缺失或在别处，按名字回答。
   - B 左栏的「At the cursor」一行和概率条保留，不再链到详情页。
3. 导出共用（§6.2 第 7 条）：
   - A、B、C 三份 training_files 里相同的索引和集合读写代码合成一份，放在 instructions/training_files.py，
     各阶段只留自己的常量；写出的字节不变（各阶段夹具不变）。
   - 在 A 的导出里写 flown_sentence：一句由执行器飞过的句子的整块（结局、结束周期、过线和 DA、
     到结局的航迹（不舍入）、姿态、包络）。A 的 replay_payload、B 的 fly_again、C 的 sentence_payload 都改用它，
     各阶段只加自己的字段；把它加进 tests/test_architecture.py 的 TRAINING_EXPORT_NAMES。
   - 三个阶段的样本格式换新名字。A、B 已发布的集合写进新的索引文件，与旧索引并列。
4. 前端（§6.2 第 6 条）：
   - 一个解析器读 flown_sentence 的块和包络，三个阶段共用；
   - B、C 的句子画出包络：横向是航向词在地面上的被判行，出界的行画红色；垂直是高度管的墙和两条边线；
     速度带只画在读回窗口里；
   - 每个阶段的 Draw 先放包络的开关，再放本阶段自己的。
5. 测试按 §6.2 第 8 条。
6. 重导（先看资源，§5 规则 13）：
   - 先对 4dTrajectory/outputs/POOLED/prior/prior_base_20261006 跑 sha256sum -c SHA256SUMS，记下结果；
   - 用新代码导出 A 的 closed_loop_v12_20261005 和 B 的 prior_sets_20261006（集合 id 和意图不变），写进新索引；
   - 导出后再跑一遍 sha256sum -c，结果必须一致，否则停下来报告；
   - 旧索引和旧集合先不删：新视图合并并检查之后再删，到时另有命令。
7. 每一步：单文件测试 → 独立审查（只审代码）→ 用显式路径提交。
   全部做完后：
   - 跑全量测试；
   - 起测试栈；
   - 浏览器检查交给一次性子代理，三个阶段都查：标签页能选句子；每个 ⓘ 和集合那一行都能打开详情页；
     结果和意图显示出来；每句飞过的句子都显示航向带和高度管；一个词能实时飞；
   - 报告地址和停止命令。
8. 日志记在 B 的日志 §4。requests 文件重写：§3 第 1–4 条已进设计（outline §6.2、D134、D135）。
   分支不合并，报告能否快进。
```
