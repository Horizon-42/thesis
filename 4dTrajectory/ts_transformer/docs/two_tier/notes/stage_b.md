# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（统一布局，D133）已做完（`38297528`、`e0aceffd`，未合并）。

```
统一布局的第二部分：结果页、词的包络、三个阶段导出共用、模型推理速度。
依据：outline §6.2（已重写）、D134、D135、D136（用户 2026-10-06）。
用户在 D133 的特许之外再特许：可以改 A 和 C 的 Training 导出及其格式（experiments/training_export.py、
instructions/training_files.py、experiments/post_training_export.py、post/training_files.py），
并用新格式重导 A 和 B 已发布的集合。

1. 换分支（用户 2026-10-06，outline §5 规则 1：以后各阶段都在 dev-two-tier-v4 上开发）：
   - 在 .claude/worktrees/two-tier-v4 里把 dev-two-tier-v4 快进到 dev-two-tier（C 已合入；含 D134、D135）；
   - 把 dev-training-layout 合进 dev-two-tier-v4，之后的工作都在这里做；
   - 确认 dev-training-layout 的提交都在 dev-two-tier-v4 里之后，删掉 training-layout 工作树和 dev-training-layout 分支；
     dev-two-tier-v4-prior 已全部在 dev-two-tier 里，也删掉它和 two-tier-v4-prior 工作树；
   - 这个工作树 C 以后也会用：只改 outline §6.2 第 9 条的文件，用显式路径暂存，每次提交前看 git diff --cached --stat，
     不提交别人的文件；
   - 重导（第 6 步）是写 aeroviz-4d/public/data，不是正式训练，可以在这个工作树里跑。
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
5. 模型推理速度（§6.2 第 10 条、D136）：
   - 写运行器 experiments/model_speed.py（带测试，runners.md 加条目）：分开计时先验每一行的一步和执行器的那几步；
     CPU 单线程和 GPU；一次一架和一批 400；每个机场 20 架 select 日的航班（种子 1337）；前 20 行不计时；
     GPU 上每次计时前后同步。给出每行 p50 / p95 / 最大（ms）、每句（s）、每秒行数、一行占 Δ = 4 s 的比例。
     不读 val 和 test。C 的部分（窗口循环、交通注意力）也写好，等 C10 之后由 C 去跑。
   - 主机和 GPU 上没有别的任务时，对 B5 的 base 跑一次，写进新目录
     4dTrajectory/outputs/POOLED/speed/<模型运行名>_<日期>/，不写进 base 的只读目录。
   - B 的导出在集合的 source.speed 里写上这个读数；结果路由去读它；详情页 B 加「Speed」一节，每个数字旁边写明它的设置。
6. 测试按 §6.2 第 8 条。
7. 重导（先看资源，§5 规则 13；在第 5 步的速度读数写好之后）：
   - 先对 4dTrajectory/outputs/POOLED/prior/prior_base_20261006 跑 sha256sum -c SHA256SUMS，记下结果；
   - 用新代码导出 A 的 closed_loop_v12_20261005 和 B 的 prior_sets_20261006（集合 id 和意图不变，B 的集合带上 source.speed），写进新索引；
   - 导出后再跑一遍 sha256sum -c，结果必须一致，否则停下来报告；
   - 旧索引和旧集合先不删：新视图合并并检查之后再删，到时另有命令。
8. 每一步：单文件测试 → 独立审查（只审代码）→ 用显式路径提交。
   全部做完后：
   - 跑全量测试；
   - 起测试栈；
   - 浏览器检查交给一次性子代理，三个阶段都查：标签页能选句子；每个 ⓘ 和集合那一行都能打开详情页；
     结果和意图显示出来；B 的 Speed 一节显示 base 的速度；每句飞过的句子都显示航向带和高度管；一个词能实时飞；
   - 报告地址和停止命令。
9. 日志记在 B 的日志 §4。requests 文件重写：§3 第 1–4 条已进设计（outline §6.2、D134、D135）。
   不合并进 dev-two-tier：报告 dev-two-tier-v4 能否快进，由用户合并。
10. 提速（outline D138）。从 2026-10-06 起三个阶段的开发都由你做（outline §5 规则 1），
    包括 instructions/、autopilot/、post 的代码。等 C10 结束才能做的步骤（第 5 步测 base 的速度、第 7 步重导），
    在等待期间先做这三个里程碑，按这个顺序：
    - A44（vocabulary.md §12.1）：起点在一个进程里只打开一次（Start）。start 和 start_moved 保留为一次调用的写法。
      改了 autopilot/，按 outline §5 规则 2 跑 closed_loop_start_check，日志里给出最大差值。
    - C13（post_training.md §8）：选择读数分给说话的 worker 跑，单进程保留为一种模式；
      在 fork 之前打开 Start；战役开始前检查 N 个 worker 的内存（O15）。
    - B14（prior.md §12）：程序掩码加 batch 模式，现在一架一架算的代码原样保留，作为可读的参照模式；
      两种模式必须给出相同的掩码。
    每个里程碑：单文件测试 → 独立审查（只审代码）→ 显式路径提交 → 日志一行。
    C10 在跑：这些代码都不合并进 dev-two-tier，也不碰 two-tier-v4-post 工作树。
11. C10 是在 .claude/worktrees/two-tier-v4-post 工作树里跑的，不是主树（你的记忆里写成了主树，请改掉）。
    就让它在那里跑完：主树和 dev-two-tier-v4 上的提交都影响不到它。
    只是 CPU 和 GPU 跟它共用，所以只有测速度（第 5 步对 base 的那次运行）要等 C10 结束，
    否则测出来的时间被 C10 拖慢，不准；写代码、跑测试、A44 / C13 / B14 都不用等。
```
