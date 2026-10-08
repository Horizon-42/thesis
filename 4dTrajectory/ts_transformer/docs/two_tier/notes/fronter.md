# Fronter：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_fronter_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
你 requests 的 1–15 条用户全部定了，写成 frontend D177：
- 2–15 按你建的接受。
- 第 1 条：落地的绿换深一点。
- 第 12 条：D 出共用的 census 生成器。
- 第 15 条：C 出公开的故障读数读取（post_training C26）。
现在做 F5（C25 已经在 dev-two-tier 里），顺带把这次的改动一起做。
路径相对仓库根；前端路径相对 aeroviz-4d/src/。

一、F5：按清单导出窗口集合（frontend §5.7、§7、§8 F5；post_training D176）
用户选的是"完整通用"：
- 一个窗口清单格式 ts-window-list-v1，阶段 C、D 共用，只列 select 窗口。
  格式和 runner 已由 C25 建好：post/window_lists.py、experiments/window_list.py。
- 共用导出器按清单导出，每个窗口用它在 selection 读数里的随机数飞，所以每个模型说的正是读数判过的那句话。
步骤（在 dev-frontend 上先合入 dev-two-tier）：
1. --windows <清单>：加在共用导出（experiments/post_training_export.py 的 add_arguments、export_sets）里。
   - 用清单的窗口代替抽样；和 --per-airport、--seed、--kinds 一起给就按名字拒绝。
   - 每个 StageExport 加 listed：在 campaign 的 selection 窗口里按 place 找、按 identity 核对。
     以下情况按名字拒绝：清单的阶段、split 或 selection 和 campaign 的不一样；某个窗口对不上。
   - 随机数：阶段 C 用 readout_numbers(select_seed, place)；阶段 D 用它读数里每架飞机的随机数
     （multi_control D166 (34)），按跨度取 selection 窗口。
   - 阶段 D 那一块在 experiments/multi_training_export.py，也由你建。
2. 样本 v5（aeroviz-training-window-sample-v5）和索引（index_post_v4.json、index_multi_v2.json）：
   - cohort 是 drawn（v4 的字段）或 listed（清单路径、sha256、那句说明、窗口数、随机数的 select seed）。
   - 写者在 post/training_files.py，读者在 data/trainingWindowSample.ts；v4 按名字拒绝。
3. 详情页第一节说明是抽样还是清单；清单的，显示清单那句说明。
4. 测试按 §8 F5，夹具由写者写，不手写。

二、D177 的改动（和 F5 一起做）
1. 落地的绿（D177 (1)）：只改 utils/trainingWordColors.ts 里的一个常量。
   新值要求：在条形图底色 #0f131e 上和黄绿 #a3e635 的 OKLab 色差 ≥ 11.9（调色板的下限），仍然一眼是绿；
   在测试里算一下色差，浏览器里看一眼。设计 §3 item 11 的表写的是这个要求，不写具体值；你选定的值记进日志。
2. 等 C26（C 出公开的故障读数读取）和 D 的 census 生成器（multi/census.py）合进 dev-two-tier 以后：
   - 导出改读它们；
   - 删掉对 WindowLoop._reading 的私有读取，删掉 census_losses 的镜像和钉它的测试。
   哪个先到先做哪个，不必等 F5 的导出跑完。

三、提交
改动文件的测试 → 独立审查（与测试同时开始）→ 显式路径提交 → 日志写一行
→ 报告 dev-frontend 能否快进 dev-two-tier，由用户合并。

四、合并以后的导出（用户已下令，2026-10-08）
在用户合并后的提交上建运行工作树（D163），开跑前看主机：没有别的 campaign、空闲内存、GPU。
1. 重导已发布的集合：删掉 windows_seg60_r5_20261008（它的文件和各机场 v3 索引里的条目），用原来的参数在 v5 里再导一次。
   原来的参数在你日志里：--rounds start 5 --kinds real A D B --per-airport 10，以及当时的 --speed。
2. 两个按清单导出的集合，清单是 4dTrajectory/outputs/POOLED/post/windows_lost_separation_select_20261008（C25 生成，193 个窗口）：
   - post_train_20261006 的 start 和 8，集合名 loss_windows_c10_<日期>；
   - post_seg60_20261007 的 start 和 5，集合名 loss_windows_seg60_<日期>。
   --speed 用各自对应的速度读数；C10 第 8 轮还没有的话，像上次一样先跑 model_speed。
   意图先写进 intents.json（L27）。
3. 重启主 checkout 的服务。
4. 浏览器检查（一次性的 sonnet 子代理，只回文字结论）：
   - 三个集合都能打开；
   - 清单集合的详情页显示清单的说明；
   - 丢间隔的窗口有刻度和红线；
   - 两个模型并排；
   - 新的落地绿和黄绿分得开。
5. 日志写一行，解开运行工作树的数据链接再删掉它，报告。

requests_from_fronter_to_designer.md 整份重写：1–15 条已定，删掉；新的读法另起。
规则照旧：campaign 运行期间只跑改动文件的测试、少开进程；每次提交前独立审查；显式路径，不用 git add -A。
```
