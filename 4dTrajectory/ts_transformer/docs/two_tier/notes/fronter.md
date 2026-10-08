# Fronter：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_fronter_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
新任务 F5：按清单导出窗口集合（frontend §5.7、§7、§8 F5；post_training D176）。用户 2026-10-08 选了"完整通用"：
一个窗口清单格式 ts-window-list-v1，阶段 C、D 共用，只列 select 窗口；共用导出器按清单导出，每个窗口用它在
selection 读数里的随机数飞，所以每个模型说的正是读数判过的那句话。
清单格式和生成清单的 runner 是阶段 C 的 C25（post/window_lists.py、experiments/window_list.py）；
导出器、样本格式和前端归你（frontend §8）。
路径相对仓库根；前端路径相对 aeroviz-4d/src/。

一、F5 的代码（等 C25 合进 dev-two-tier 以后，在 dev-frontend 上先合入 dev-two-tier）
1. 共用导出（experiments/post_training_export.py 的 add_arguments、export_sets）加 --windows <清单>：
   - 用清单里的窗口代替抽样；和 --per-airport、--seed、--kinds 一起给就按名字拒绝。
   - 每个 StageExport 加 listed：在该 campaign 的 selection 窗口里按 place 找、按 identity 核对；
     以下情况按名字拒绝：清单的阶段、split 或 selection 和 campaign 的不一样；某个窗口对不上。
   - 随机数：阶段 C 用 readout_numbers(select_seed, place)；阶段 D 用它读数里每架飞机的随机数（multi_control D166 (34)），
     按跨度取 selection 窗口。
   - 阶段 D 那一块在 experiments/multi_training_export.py 里，也由你建。
2. 样本 v5（aeroviz-training-window-sample-v5）和索引（index_post_v4.json、index_multi_v2.json）：
   - cohort 两种写法：drawn（v4 的字段，不变）或 listed（清单路径、sha256、那句说明、窗口数、随机数的 select seed）。
   - 写者在 post/training_files.py，读者在 data/trainingWindowSample.ts；v4 按名字拒绝。
3. 详情页第一节说明是抽样还是清单；是清单的，显示清单那句说明。
4. 测试按 §8 F5。夹具由写者写，不手写。
5. 改动文件的测试 → 独立审查（与测试同时开始）→ 显式路径提交 → 日志写一行
   → 报告 dev-frontend 能否快进 dev-two-tier，由用户合并。

二、合并以后的导出（用户已下令，2026-10-08）
在用户合并后的提交上建运行工作树（D163），开跑前看主机：没有别的 campaign、空闲内存、GPU。
1. 重导已发布的集合：删掉 windows_seg60_r5_20261008（它的文件和各机场 v3 索引里的条目），用原来的参数在 v5 里再导一次。
   原来的参数在你日志里：--rounds start 5 --kinds real A D B --per-airport 10，以及当时的 --speed。
2. 两个按清单导出的集合，清单用 C25 由 runner 重新生成的那份（outputs/POOLED/post/ 下，C 会报路径）：
   - post_train_20261006 的 start 和 8；
   - post_seg60_20261007 的 start 和 5。
   集合名用 loss_windows_c10_<日期> 和 loss_windows_seg60_<日期>；--speed 用各自 campaign 对应的速度读数；
   C10 第 8 轮还没有速度读数，就像上次一样先跑 model_speed。
   意图先写进 intents.json（L27）。
3. 重启主 checkout 的服务。
4. 浏览器检查（一次性的 sonnet 子代理，只回文字结论）：
   - 三个集合都能打开；
   - 清单集合的详情页显示清单的说明；
   - 丢间隔的窗口有刻度和红线；
   - 两个模型并排。
5. 日志写一行，解开运行工作树的数据链接再删掉它，报告。

你 requests 里的第 1–15 条还在等用户定，这次不动。
规则照旧：campaign 运行期间只跑改动文件的测试、少开进程；每次提交前独立审查；显式路径，不用 git add -A；
设计没说到的地方写成读法放进 requests_from_fronter_to_designer.md（整份重写），等用户定。
```
