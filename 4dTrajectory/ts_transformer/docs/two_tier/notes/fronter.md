# Fronter：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_fronter_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。先读你日志的 §Handover。

```
路径相对仓库根；前端路径相对 aeroviz-4d/src/。按顺序做，一步报告一次。

一、"start (base)" 标签改正（frontend §4.3，2026-10-08 新加的一句）
D162 以后，campaign 可以从另一个 campaign 的某一轮出发：C10 之后的每个 campaign 都是（P55 从 C10 第 8 轮出发，
32 秒从 P55 第 5 轮出发）。导出的 start 用的是这个起点（post_train.campaign_start），但前端在三处一律写成 base：
components/training/TrainingWindowSession.tsx:122、data/trainingStatistics.ts:125（和 :128 的注释）、
data/trainingWindowSample.ts:564。
1. 在 dev-frontend 上先合入 dev-two-tier。
2. 读者：parseModel 读 model.settings.start（导出已经写了 model.settings = campaign 的 settings）：
   null，或 {campaign, round, checkpoint_sha256}；只读这一个键，必需，null 合法，其他形状按名拒绝。
3. 一个函数给出起点的名字，三处都用它：null → base；否则 campaign 目录名加轮次（例如
   "start (post_train_20261006 r8)"；标签页上短一点也行，tooltip 说全：哪个 campaign 的哪一轮，D162）。
4. experiments/post_training_export.py 第 16 行的 docstring 补上"或它出发的另一个 campaign 的那一轮（D162）"。
5. 测试：两种起点的名字；读者对 start 的三种情况（null、一轮、坏形状）。夹具照旧由写者写；
   如果要一个"从某一轮出发"的夹具，用导出在冒烟 campaign 上写，不手写。
6. 改动文件的测试 → 独立审查（与测试同时开始）→ 显式路径提交 → 日志写一行 → 报告能否快进，由用户合并。

二、发布阶段 C 的第一个正式数据集（用户 2026-10-08 定：P55）
在一合并进 dev-two-tier 之后，从合并后的提交建一个运行工作树（D163；数据目录链接到正式数据），在那里跑：
1. 开跑前看主机：没有别的 campaign（C23 可能已经启动）、空闲内存、GPU。有 campaign 在跑就先问我。
2. 阶段 C 的速度读数（D136；导出需要 --speed，现在只有阶段 B 的）：
   python run_ts.py model_speed --campaign 4dTrajectory/outputs/POOLED/post/post_seg60_20261007 --round 5 \
       --out 4dTrajectory/outputs/POOLED/speed/post_seg60_r5_20261008
3. 意图先写（L27）：intents.json 里这个数据集的一行（post_seg60_20261007 的条目已有，加数据集的 run 行：
   起点 C10 第 8 轮 85.7 % 对第 5 轮 87.7 %，select 窗口，四种窗口）。和日志一起提交到 dev-two-tier。
4. 导出：
   python run_ts.py post_training_export --campaign 4dTrajectory/outputs/POOLED/post/post_seg60_20261007 \
       --rounds start 5 --split select --kinds real A D B --per-airport 10 \
       --set-id windows_seg60_r5_20261008 --speed 4dTrajectory/outputs/POOLED/speed/post_seg60_r5_20261008
5. 重启主 checkout 的服务（./start_aeroviz_fullstack.sh）：它还在答 window segment v1。
6. 浏览器检查（一次性的 sonnet 子代理，只回文字结论）在主服务上（5173 / 8765）：start 标签写的是
   post_train_20261006 r8；r5 和 start 并排；有失去间隔的窗口出现刻度和红线；每种窗口一个活的词；
   详情页的正式读数（上次在冒烟集上没看到的那一项）。
7. 日志写一行（数据集、命令、窗口数、检查结果），解开运行工作树的数据链接再删掉它，报告。

三、F3（frontend §5.1–§5.6、§8 F3）
条件已满足：阶段 D 的 MC0（和 MC4）已在 dev-two-tier（246eca69 起）。二做完后开始，按你日志 §Handover 的
"Things to carry in" 做（D160 (7) 用 MC0 的代码再核对；多架飞机时 WindowCursor 经窗口时钟换算；夹具由阶段 C 的
导出在合成的两架被指挥飞机窗口上写）。F4 的导出和冒烟集在 F3 之后；F4 的正式数据集等 MC6，意图先写。

规则照旧：campaign 运行期间只跑改动文件的测试、少开进程；每次提交前独立审查；显式路径，不用 git add -A；
设计没说到的地方写成读法放进 requests_from_fronter_to_designer.md（整份重写），等用户定。
```
