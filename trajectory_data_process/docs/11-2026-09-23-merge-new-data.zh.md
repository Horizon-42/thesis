# 2026-09-23 合并新下载数据（new_data_9_22）并冻结旧数据

## 状态

| 步骤 | 状态 | 说明 |
|---|---|---|
| 1. review（trajectory_data_process、flight_scenarios，对照 evaluation / ts） | 完成 | 修掉的在下面“代码改动”里；其余的记在 `docs/code-health-followups.md` 同日条目 |
| 2. 代码改动（分支 `data-merge-0923`，worktree `.claude/worktrees/data-merge`） | 完成，已复核 | 数据侧 591 个、ts 1212 个测试通过；review 12 条：修 10 条，2 条不改（理由见 TD21、TD23），1 条推迟（followups #16）；复核又修 1 条（二次合并丢审计） |
| 3. 下载 8/21–9/22 的 METAR | 完成 | `data/metar/<ICAO>/asos_2026-08-21_2026-09-22.csv`，795–907 条/机场（git 忽略的数据目录）|
| 3b. KMSY 试跑（scratch 目录，硬链接） | 完成 | 约 2 分钟；tracks 35,740 = 19,581 + 16,159，arrivals 7,696 = 4,147 + 3,549；live 数据 sha 未变 |
| 4. 冻结旧数据 | 完成（16:59） | `outputs/harvest` → `outputs/harvest-v5-20260823` + `FROZEN.json`，已 `chmod -R a-w`；旧 checkpoint（two_tier_v3 L60_D60）按指纹找到冻结代，provenance 校验通过 |
| 5. 合并 | 完成（第 2 次，15:02–15:24 UTC） | 第 1 次：后台 runner 的 PID 记错（`setsid` 自己 fork，记下的父进程立即退出），我误以为它停了，又在前台对 KMSY 并发跑了一次合并；前台那次删掉了后台正在写的 arrivals/ 和部分 approach/ 记录，后台随即在评估阶段报错停止。KMSY 的 tracks/ 完整（35,740 条，文件齐全），其余 4 个机场未动，冻结代只读未受影响。第 2 次：KMSY `--evaluate-only` 重建视图，其余 4 个机场合并，runner 自己写 PID |
| 6. lateral 名单 + 划分核对 | 完成，通过 | 结果见“合并结果” |
| 7. 前端发布（observed CZML / 报告） | 未做，需要另问 | 合并时加 `--no-czml --no-publish` |

## 用户的决定（2026-09-23）

- 旧 checkpoint 要继续能用：旧数据整代冻结（改名，不改内容），合并结果作为新的 `outputs/harvest`。
- 这次一起修：跑道航向精度（改用 CIFP）、进场切片终点（实测跨越点必须在切片内）。
- 下载新数据时间段的 METAR。
- 删掉 arrivals 读取 v5/v6 的兼容分支。用户的理解（已采纳）：数据只是数据，变的只是 manifest；
  读取端只认当前版本（v7），冻结那一代按字节内容（sha256）识别后原样读取，不做转换。
- 同意目录改名和 ts 按数据指纹找数据的跨包改动。

## 数据事实（合并前）

- 旧数据 `outputs/harvest`：着陆时间 2026-05-01 → 2026-07-22（其中约 6/1–6/26 没有数据），
  arrivals v5，共 42,650 条，lateral 通过 42,571 条。
- 新数据 `outputs/new_data_9_22`：2026-08-22 → 2026-09-22，tracks 169,449 条，arrivals v6 共 28,078 条。
- 新旧没有重叠：flight_key 零重复，按 (icao24, 落地时间) 也零重复。中间空一个月（7/22–8/22）。
- train/val/test 划分 = `sha256(1337:机场:flight_key)`：旧航班保持原划分，新航班按同一规则分入。
  按现有 lateral 结果粗估，合并后约 70k 条可用（train 49,248 / val 10,319 / test 10,583），
  修好 KSMF 35R、加上旧数据里的 KRDU 32 后约 72k。

## 代码改动（分支 data-merge-0923）

- `harvest/cifp.py` `read_runway_records`：CIFP 跑道记录（PG），ARINC 424-23 §4.1.10.1；
  §5.36/5.37 Note 5 规定其坐标就是着陆阈值点。用 Path Point 校验解码（≤1 m、≤1 ft，≥75%）。
- `harvest/airports.py`：没有 LPV 的跑道，阈值位置和标高取 CIFP 跑道记录（KSMF 35R 原来偏 39.4 m，
  导致 383/383 横向判 fail）；所有跑道航向 = 两端 CIFP 阈值点连线（原来是 OurAirports 整度数，最多差 0.45°）。
  指纹 schema 升到 v2，所有存储的阈值事件都要重算（合并会重算）。
- `harvest/arrivals.py`：schema v7 `harvest-arrivals-v7-measured-crossing-in-slice`；
  实测跨越的切片延到跨越后那个点（落地点和 flight_key 不变）；`entry_time_utc` 精确到毫秒；
  整个名单建好后才替换旧文件；读取只认 v7 或冻结代的字节。
- `harvest/observed.py`：记录先写到临时目录，全部成功后再替换。
- `harvest/merge.py` + `__main__.py`：保留每个来源的 `source_integrity`；`--jobs` 传到重新分类；
  普通下载拒绝覆盖合并过/重建过的数据目录。
- `harvest/generations.py`（新）：冻结代登记 `FROZEN_GENERATIONS`、`FROZEN.json`、freeze 命令。
- ts `repo_layout.checkpoint_arrival_manifest(s)`：回放按 checkpoint 记录的 sha256 在“当前 + 冻结代”里找
  manifest；7 个回放 runner 改用它；4 处重复的 `HARVEST_ROOT` 改为 import。
- `download_landings.py`：默认值改为 import CLI 的（原来 CIFP 周期抄旧了）。

参考文档：`06-harvest-reference.md` TD21–TD24（及 TD9/TD16/TD17 的更正）；ts `docs/reference/contracts.md` C29。

## 数据步骤（代码 review 通过、提交之后）

```bash
# 3. METAR（已完成；IEM 连续请求会返回 429，逐个机场重试即可）
python -m trajectory_data_process.metar.fetch_iem_asos --airport <ICAO> --start 2026-08-21 --end 2026-09-22
# 4. 冻结：改名 + 标记；新 live 根目录用硬链接复制 tracks/（不占空间）
mv trajectory_data_process/outputs/harvest trajectory_data_process/outputs/harvest-v5-20260823
python -m trajectory_data_process.harvest.generations freeze trajectory_data_process/outputs/harvest-v5-20260823 --reason "..."
mkdir -p trajectory_data_process/outputs/harvest/<ICAO>
cp -al trajectory_data_process/outputs/harvest-v5-20260823/<ICAO>/tracks trajectory_data_process/outputs/harvest/<ICAO>/tracks
chmod -R a-w trajectory_data_process/outputs/harvest-v5-20260823      # 冻结代只读
# 5. 合并（每个机场）
python -m trajectory_data_process.harvest --airport <ICAO> --merge-source trajectory_data_process/outputs/new_data_9_22 --jobs 20 --no-czml --no-publish
# 6. lateral 名单
python -c "from ts_transformer.data.lateral_eligibility import ensure_lateral_pass_roster; ..."
```

## 合并结果（2026-09-23，核对脚本输出）

| 机场 | arrivals（旧 v5 → 新 v7） | eligible（旧 → 新） | 新代 train / val / test |
|---|---:|---:|---|
| KMSY | 4,147 → 7,696 | 4,140 → 7,680 | 5,390 / 1,122 / 1,168 |
| KRDU | 14,435 → 24,202 | 14,378 → 23,920 | 16,723 / 3,575 / 3,622 |
| KSJC | 11,082 → 16,772 | 11,076 → 16,760 | 11,815 / 2,498 / 2,447 |
| KSMF | 4,219 → 9,224 | 4,216 → 9,216 | 6,465 / 1,333 / 1,418 |
| KSTL | 8,767 → 14,680 | 8,761 → 14,671 | 10,300 / 2,107 / 2,264 |
| 合计 | 42,650 → 72,574 | 42,571 → 72,247 | **50,693 / 10,635 / 10,919** |

按时段：旧时段（5/1–7/22）44,284 条（train 31,148 / val 6,455 / test 6,681，比 v5 多出的是 KRDU 32
和 KSMF 35R），新时段（8/22–9/22）27,963 条（train 19,545 / val 4,180 / test 4,238）。划分种子 1337。

- **冻结代完整**：五个机场的 arrivals / tracks manifest sha256 与 `FROZEN.json` 一致；旧 checkpoint
  （two_tier_v3 L60_D60）按指纹解析到 `harvest-v5-20260823`，provenance 校验通过。
- **旧划分保住了**：旧 eligible 航班里 flight_key 变化的为 0；旧 test 航班离开 test 的为 0（五个机场旧 test ⊆ 新 test）；
  旧 eligible 失去资格的为 0。
- **唯一例外：KSMF 35L 的 2 条“一段轨迹两次进近”航班**，跑道几何修正后重新分类选了另一次进近作落地，
  落地时刻和 key 都变了：SWA1495（18:32:18 → 18:40:30，旧 val → 新 train）、SWA1521（07:35:33 → 07:24:44，train → train）。
  test 不受影响；旧 val 里只有 SWA1495 这 1 条移到了新 train。
- **KSMF 35R 恢复**：643/643 横向通过（修正前 383/383 全挂）。
- **KRDU 32 横向 fail 206/2,647（7.8%）**，其他跑道 0–2.1%（最高 KMSY 02 6/289）；阈值点与 CIFP 只差 0.4 m，
  所以不是几何问题，是这条跑道的真实飞法（短跑道、通航多），记在这里，结论里要说明。
- **风数据**：新时段 87–94% 的航班用上了 METAR 风（KRDU 最低 7,101/8,198），之前全是地速代理；旧时段 92–98%。
- **v7 切片**：实测跨越的切片延到跨越后一点——KSMF 4,661/9,224 条，KRDU 586/24,202 条（KRDU 多数航迹在阈值前就没信号，是推算跨越）。
- **现有 development cohort**：32 个里 29 个在新划分下仍然合法；3 个（plan_guidance_20260910 的 step5_pooled、
  step5b_KSMF，runway_intent_r2b 的 KSMF_day_a）各含上面那 1–2 条 KSMF 航班，在新数据上会被拒绝——
  都是旧实验的 cohort，新实验本来就要重建 cohort。

## 还没做 / 需要另问

- 前端发布：observed CZML 和观测报告（`--no-czml --no-publish` 没发布）；`evaluation_report.html` 只有 `run_all_evaluations.py` 生成。
- 用新数据训练前要重建实验 cohort（`plan_cohort` 等），旧 cohort 文件绑定的是旧代。
- `outputs/new_data_9_22`（5.2 GB）已合并进 live，可删，需要你同意。
- 冻结代里的 `.KSJC-reclassify-akrwpor_`（170 MB，8/24 被杀的重新分类留下的）可删，需要你同意。

## 合并后核对（清单）

- 旧的 eligible 航班在新代里 flight_key 不变、所在划分不变；key 变了的逐条列出。
- 旧 test 集 ⊆ 新 test 集（旧 test 航班不会进入新 train）。
- 现有 development cohort 文件在新划分下仍合法。
- KSMF 35R 横向通过率恢复正常；新旧两个时段的速度门都用上了风数据。
- 各机场 arrivals / eligible / train / val / test 数量。
- 冻结代的每个文件 sha256 与 `FROZEN.json` 一致；抽一个旧 checkpoint 做一次回放（只读）。
