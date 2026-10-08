# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
新加一项小代码（frontend D177 (12)，用户 2026-10-08 定；multi_control MC7 那一段）：
fronter 在 experiments/multi_training_export.py 里复制了一份 census 的逐步判定循环（census_losses，标成 MIRROR，
用测试钉在 multi_train.window_losses_of 上），来写"没有飞机负责"的丢间隔（D145）。
把这个循环做成 multi/census.py 里的一个生成器，读数和导出都用它；fronter 之后删掉那份复制。
路径相对 4dTrajectory/ts_transformer/。

一、代码（在 dev-multi-control 上，先把 dev-two-tier 合进来）
1. multi/census.py：
   - 一个生成器：逐步给出一个窗口里含被指挥飞机的丢间隔，是现在 window_losses_of 和 census_losses 共同的那部分；
     名字你定，报告里写上。
   - multi_train.window_losses_of 改用它；读数结果逐位不变。
2. 不改 multi_training_export.py：那是 fronter 的文件，它在 F5 之后改读你的生成器。
3. 测试：
   - 读数的丢间隔计数在固定输入上和改之前逐位相同；
   - 生成器在合成窗口上给出预期的每一步；
   - 导出现有的 MIRROR 测试照样通过。
4. 改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者，审查和测试同时开始）
   → 显式路径提交 → 日志写一行 → 报告 dev-multi-control 能否快进 dev-two-tier，由用户合并。

二、MC5（照旧）
主机和 GPU 上没有别的任务时，用 fffe900d 起的代码在正式规模上按每种跨度把 profile 跑完（systemd 单元），
然后把 MC6 的建议设置报告给用户。
- 开跑前看主机：现在阶段 C 没有 campaign 在跑；fronter 的 F5 合并后要导出三个集合，和它错开。
- 一里的代码可以和 profile 同时写，只跑改动文件的测试。

三、之后
MC6 的设置和起点（C 的哪个 campaign 的哪一轮）由用户定，从运行工作树启动（outline D163）。

你只能写这些设计文本：各文档 §0.3 的状态行、multi_control §0.3、你的日志。设计没说到的地方写成读法，
放进 docs/two_tier/design/requests_from_d_to_designer.md（整份重写），等用户定。
```
