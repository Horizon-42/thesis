# 飞机模型与姿态：待执行的收尾步骤

这份文档是 Training 模块"飞机用模型、绑定真实姿态"（`36-2026-09-20-training-module.zh.md` §4.12）剩下的步骤，照着做就能收尾。
设计、定下的事、代码说明都在 §4.12 与 `35-viewer-reference.md` AV42（这两处的最新版在分支上，见下）。

## 状态（2026-10-01，合并与 CPU 部分的发布之后）

| 项 | 状态 |
|---|---|
| 代码 | **已合并**（用户 2026-10-01：合并 + 发布 CPU 部分 + 重启）：`dev-two-tier` 快进到 `fc37ba66`；分支 `dev-training-attitude`、工作树 `.claude/worktrees/training-attitude` 还在（见第 3 步清理）。 一次 opus 审查（没有必须改的，4 条应改已改），Vitest 107 个文件 902 条、相关 ts 与后端测试全过；执行器、标注器、场景边源码 sha256 不变（`a0a6e20bdd84` / `55f6f0bcd6ee` / `352a21c2f3b8`） |
| 不用 GPU 的导出 | **做完**（2026-10-01）：`4dTrajectory/outputs/POOLED/training_publish/attitude_20260930/airports`。五个机场各一份：真实句子集合、执行器回放、17 个真实起点的模型句子（base、landing r1–r8、augmented r1–r8），共 95 个文件，逐个与已发布的比过：除姿态、写入时间、格式名、生成者外完全相同（`cmp_all.py`）。脚本、日志在同一目录 |
| 要 GPU 的导出 | **没做**：窗口集合 + 2 个窗口模型句子、3 个增强起点模型句子（每个机场 6 个文件）。等 7.6 正式实验不用 GPU 时再做 |
| 发布、合并、重启 | **CPU 部分已发布、已合并、已重启**（2026-10-01）：五个机场原来的 `training/` 改名为 `training.bak-20260930T231655Z`，导出目录的 `training/` 复制进去；8765 与 5173 由 supervisor 重启（后端 pid 3043444、前端 3043506）。`check-publication --server`：每个机场只有 GPU 部分那 6 个旧格式文件报错（窗口集合 traffic-v1、3 个增强起点 augmented-v2；2 个窗口叠加层挂在窗口集合上）。**在 GPU 部分发布前，Training 里的窗口模式与增强起点读不了**（按名字拒读），其余照常 |

导出目录的结构：五个机场（KMSY、KRDU、KSJC、KSMF、KSTL）各是真目录，里面除 `training/` 外的文件都是指向正式数据的链接，
`training/` 是从正式数据复制来的真目录（导出器往里写）；其他机场整个是链接。**不能往链接的机场导出**，会直接写进正式数据。

目前导出目录里 `check-publication` 每个机场报 4 个错误，正是还没重做的旧格式文件：窗口集合（traffic-v1）和 3 个增强起点叠加层
（augmented-generation-v2）；2 个窗口叠加层挂在窗口集合上，报的是警告。

## 第 1 步：GPU 部分的导出

**先看资源**（规则：不打扰正在跑的实验）：`nvidia-smi`、`free -g`、`ps -eo pid,etime,cmd | grep -E "traffic_reward|run_ts"`。
7.6 正式实验（`m4_window_20260930`，另一个会话管）在跑就等它。

在工作树里跑（`cd /home/supercomputing/studys/thesis/.claude/worktrees/training-attitude`，`conda activate aeroviz`），导出根目录
`E=4dTrajectory/outputs/POOLED/training_publish/attitude_20260930/airports`，机场按字母顺序（抽样的随机数流按这个顺序，已发布的就是这样）。
超过 10 分钟的跑法用 `nohup setsid` 分离运行、脚本里 `echo $$ > pid`、用 Monitor 看（照 `run_cpu_export.sh` 的写法）。

1. **先把旧文件从导出目录拿掉**（只动导出目录，不动 `public/data`）：每个机场 `training/index.json` 删掉集合 `traffic_windows_select`
   的条目并删掉该目录；`training/overlays.json` 删掉 `generation_augstart_*`（3 个）与 `windows_*`（2 个）的条目并删掉它们的目录。
   （`prep_export.py` 是同样的做法，可以照着改。）
2. **3 个增强起点模型句子**（GPU，已发布的是 `--device cuda`，种子都是 1337）：

   ```bash
   P=4dTrajectory/outputs/POOLED; A="--airport KMSY --airport KRDU --airport KSJC --airport KSMF --airport KSTL"
   for spec in "generation_augstart_base_fb81ccc9 v3_step1_20260924/full_s1337" \
               "generation_augstart_landing_r01_53225a69 v3_rl_20260925/grpo_s1337/round_01" \
               "generation_augstart_augmented_r07_6bd12ca3 v3_stage2_clip_20260926/aug_s1337/round_07"; do
     set -- $spec
     python run_ts.py prior_generation_training_export --prior $P/prior/$2 --instructions $P/instruction_language/v5_20260926 \
       --executor $P/executor/v11_20260927 --airports-root $E --set instruction_v3_day_split $A \
       --samples 4 --temperature 1.0 --seed 1337 --augment-seed 1337 --device cuda --overlay-id $1
   done
   ```
3. **窗口集合 + 2 个窗口模型句子**（GPU；第一个导出写集合，第二个核对集合相同）：

   ```bash
   python run_ts.py window_training_export --prior $P/prior/v3_stage2_clip_20260926/aug_s1337/round_07 \
     --instructions $P/instruction_language/v5_20260926 --executor $P/executor/v11_20260927 --airports-root $E $A \
     --windows 20 --samples 4 --temperature 1.0 --seed 1337 --device cuda --workers 4 --overlay-id windows_augmented_r07_6bd12ca3
   python run_ts.py window_training_export --prior $P/prior/m4_passes_20260929/traffic_s1337/round_05 \
     --instructions $P/instruction_language/v5_20260926 --executor $P/executor/v11_20260927 \
     --readout $P/traffic/window_generation_r5_20260930 --airports-root $E $A \
     --windows 20 --samples 4 --temperature 1.0 --seed 1337 --device cuda --workers 4 --overlay-id windows_traffic_r05_f0c4db55
   ```

   以前每个模型约 7 分钟（GPU）。

4. **核对**：把 `cmp_all.py` 的文件清单扩到这 6 个文件（集合 `traffic_windows_select/traffic.json`、3 个 `generation.json`、2 个
   `window_generation.json`），与 `public/data` 里已发布的比，除姿态、写入时间、格式名、生成者外应完全相同。
   - **窗口的两个叠加层可能对不上**：已发布的是 2026-09-30 在未提交的工作树上（`d1243d3b` dirty）导出的，之后窗口循环的代码又改过
     （7.6 的修复，如 `cbfdc5f2`）。如果窗口集合相同、但窗口里的样本不同，说明抽样或飞法代码变了，**不是姿态的问题**：先查是哪次提交
     改了抽出的结果，再问用户是发布新抽的样本（同一个 overlay id 还是新 id），还是别的办法。增强起点的 3 个若不同，也先查原因再问。
5. **用前端读一遍**：`cd aeroviz-4d && npm run check-publication -- --airports-root ../$E`，五个机场都应 0 错误（旧词表的集合是警告，正常）。

## 第 2 步：发布、重启（**每一步先问用户**；合并已做）

1. （合并已于 2026-10-01 做完。以后若代码再改：先把主线合进分支再测。）重跑：
   `npx vitest run`、`npx tsc --noEmit -p .`、ts 的 `test_training_attitude` `test_instruction_training_export` `test_training_overlays`
   `test_prior_generation_training_export` `test_window_training_export` `test_architecture`、后端 `aeroviz_backend/tests`
   （Python 用 `PYTHONPATH=.:4dTrajectory python -m pytest ... --import-mode=importlib`）。确认执行器、标注器、场景边 sha256 仍不变。
2. **发布 GPU 部分**（问用户）：只把这 6 个文件（每个机场的 `traffic_windows_select/`、3 个 `generation_augstart_*/`、2 个 `windows_*/`
   目录）和 `index.json`、`overlays.json` 从导出目录复制到 `public/data/airports/<ICAO>/training/`（先把被替换的留一份备份）。
   注意：导出目录的 `training/` 是 CPU 部分发布时的样子，里面其余文件与正式数据相同，也可以像 CPU 部分那样整个替换（再留一份
   `training.bak-<UTC>`）。
3. （合并已做。）
4. **重启**（问用户）：后端不热加载，要重启 8765（Fly 的新格式 segment v8）；5173 也重启（AV5）。然后
   `npm run check-publication -- --server http://localhost:5173` 五个机场 0 错误；在浏览器里看：单机光标处的飞机模型与读数、转弯处有坡度、
   窗口里按角色的飞机、点一个词 Fly 出来的那架也是模型。
5. 确认没问题后（问用户）删掉 `training.bak-*`（包括 CPU 部分发布时留的 `training.bak-20260930T231655Z`）。

## 第 3 步：清理（用户说清理再做）

- 停测试服务器 5177：`kill $(lsof -t -iTCP:5177 -sTCP:LISTEN)`。
- 工作树里的链接先解掉：`data`、`trajectory_data_process/outputs`、`4dTrajectory/outputs`、`aeroviz-4d/node_modules`，以及
  `aeroviz-4d/public/data/airports`（现在指向一个临时试导出目录，不是正式数据）；再 `git worktree remove .claude/worktrees/training-attitude`、
  `git branch -d dev-training-attitude`（提交都已在 `dev-two-tier` 才删）。
- 导出目录 `training_publish/attitude_20260930/` 发布后可留作记录，或问用户删。

## 要记得的坑

- **导出器往 `--airports-root` 里写**：根目录里是链接的机场会写进正式数据。只往真目录导。
- **机场顺序影响样本**：模型句子的随机数流按命名的机场顺序走，已发布的都是 KMSY、KRDU、KSJC、KSMF、KSTL。单导一个机场，样本就对不上。
- **GPU 与 CPU 抽的样本不同**：增强起点和窗口的已发布版本是 `cuda`，必须用 `--device cuda` 才能复现。
- **机器忙时很慢**：CPU 部分在别的任务占满时一个模型要 20–28 分钟，空闲时约 3 分钟。
- 导出目录里的旧格式文件，`check-publication` 会报错，这是正常的，只要报的都是还没重做的那几个。
