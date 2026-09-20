# Training 模块设计（2026-09-20）

前端加一个新任务页，叫 **Training**，放在 Observe 右边。它不训练模型、不算评估，只做一件事：
**把两层模型中间那些看不见的东西画出来** —— 一次进近被读成了哪几句话，这几句话本身意味着什么样的
航迹，以及（等模型训出来以后）模型自己说出来的那句话又意味着什么样的航迹。

这篇是设计文档，写在动手之前。实现按 §7 的步骤走，一步一评审。

---

## §1 状态与索引

| 项 | 值 |
|---|---|
| 文档 | `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`（本文） |
| 状态 | 设计已定（用户 2026-09-20 回答了全部待定项，见 §9）；**T1 已落地、未提交**，下一步 T2。进度看 §7 的表 |
| 分支 | `dev-two-tier-feasibility`（最新提交 `30a45cab`） |
| 它要展示的实验 | 两层计划 v3 阶段 B（`4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_plan_v3.zh.md` §5.2） |
| 词表设计 | 同上 §5.2.1；单独一篇 `4dTrajectory/ts_transformer/docs/2026-09-20_instruction_vocabulary.zh.md` |
| 词表代码 | `4dTrajectory/ts_transformer/manoeuvre/instructions.py`（标注器 + 词表，叶模块） |
| 产物写出代码 | `4dTrajectory/ts_transformer/experiments/instruction_vocabulary.py`（`run_ts.py instruction_vocabulary`） |
| **词表产物** | **有了**（2026-09-20，B0″ 跑完）：`4dTrajectory/outputs/KRDU/experiments/two_tier_v3_bprime_20260920/vocabulary_tau10/`，sha `a5450c0fcd90…`，读法 `plateau-v10`，τ = 10 s。内含 `instruction_vocabulary.json`、`sentences_train.json`、`sentences_val.json`、`summary.json`、`summary.txt`、`hand_check/` 300 页。以上各项我都开文件核过 |
| 跑道词的类别 | **KRDU 只有 4 类：05L / 05R / 23L / 23R**（不是机场的 6 条跑道，见 §4.4 第四条） |
| 前端怎么开发 | T1、T2、T4 用 **mock 数据**（假数据），放测试目录，**不进 `public/`**；T3 起读上面这份真产物。**不从 `_superseded/` 导出任何东西**（用户 2026-09-20 明确否掉） |
| 前端现有任务页 | Observe / Fly / Optimize / Compare（`src/components/WorkbenchTopBar.tsx` 的 `TASK_TABS`），Procedures 是一个独立开关不是任务 |
| 前端惯例 | 全局状态在 `src/context/AppContext.tsx`；二维图自己画 SVG（没有图表库，见 `package.json`）；`public/data` 不进 git |

**这份设计是在"输入文件还不存在"的前提下写的。** 因此 §4 把"文件不在"当成一等状态来设计，
而不是当成错误；界面在没有数据的时候必须说清楚它找的是哪个路径、那个文件由哪条命令生成。

**mock 数据和"界面上的假数据"是两回事，别混了**：mock（假数据）只存在于
`src/**/__tests__/` 下的测试用例里，用来在真产物出来之前把界面做出来并测试；
`public/data/.../training/` 下**永远只放真导出**。用户在开发机上打开 Training，
在 B0″ 跑完之前看到的就是 §4.5 的空状态，不是假航迹。

---

## §2 放在哪，为什么

### 2.1 位置

`WorkbenchTopBar` 的任务条从

```
Observe │ Fly │ Optimize │ Compare ‖ Procedures
```

变成

```
Observe │ Training │ Fly │ Optimize │ Compare ‖ Procedures
```

代码上：`AppContext` 的 `WorkbenchMode` 联合类型加一个 `"training"`，`TASK_TABS` 加一项，
`WorkbenchLeftDock` 多一个分支。**不新增第二套模式系统** —— 仓库里已经因为"两套互相打架的 mode"
做过一次合并，不能倒回去。

### 2.2 为什么挨着 Observe 成立

- **看的是同一批飞机。** Observe 回答"真飞机是怎么飞的"，Training 回答"这条真航迹被读成了哪几句话"。
  两页共用顶栏的机场和跑道，也共用同一个航班身份（`flight_key` = `id_跑道_icao24_落地时刻`，
  不是呼号）。从 Observe 选中一架切到 Training，选中状态应该跟着走。
- **Fly / Optimize / Compare 是"造一条航迹出来"，Training 是"这条航迹被说成了什么"。** 它属于
  观测这一侧，不属于求解那一侧。放在 Observe 右边，读起来就是"同一批数据，下一个问题"。
- **它不抢 Observe 的资源。** `src/data/observedTracks.ts` 写明观测航迹的大文件只在 Observe 里加载
  （它会驱动共享的 Cesium 时钟，在别的任务里加载会把那边的播放顶掉）。Training 读自己的抽样文件，
  文件里自带需要的那几十条航迹，所以**不加载 Observe 的那份 CZML，也就不会抢时钟**。

### 2.3 它不是什么

不是"训练控制台"。不启动训练、不调后端、不写任何磁盘文件。它只读已经写好的 JSON。
真要跑训练，是命令行的事。

---

## §3 屏幕长什么样

三样东西，优先级就是用户给的顺序：

1. **一句话** —— 抽几架飞机，把从它航迹里读出来的指令词显示出来；
2. **词画出来的航迹** —— 只按这几个词、用一套很简单的运动学飞出来的大致航迹，和真航迹并排；
3. **模型说的句子画出来的航迹** —— 等第二层训出来以后（阶段 B3′），同一套画法画第三条线。

### 3.1 主屏

```
┌─ WorkbenchTopBar ─────────────────────────────────────────────────────────────┐
│ AeroViz-4D   机场 [KRDU▾] 跑道 [05L▾]   Observe │Training│ Fly │…│  ⚙层  ▣演示│
└───────────────────────────────────────────────────────────────────────────────┘
┌─ 左坞（Training）──────┐                                  ┌─ 右侧读数 ────────┐
│ 抽样集                 │                                  │ 这一句话           │
│ [vocabulary_tau10 ▾]   │            Cesium 三维            │ 跑道 05L ←另四类  │
│ 读法 plateau-v10       │                                  │   都相对它量      │
│ sha …（见 index.json） │      ━━━ 真实航迹（白）           │ 航向 +90°→+50°→0° │
│ τ = 10 s               │      ━━━ 按词画的（橙）           │ 高度 3000→2000→0  │
│ 40 架（直线20/引导20）  │      ┄┄┄ 模型说的（紫，暂无）      │ 速度 210→180→140  │
│                        │                                  │ 切入 ≤30°（只标记）│
│ 航班                   │                                  ├───────────────────┤
│ ┌────────────────────┐ │           ● 指令发出的位置        │ 这一刻 t=120 s    │
│ │▸DAL123  引导  9 词 │ │                                  │ 离入口 12.4 km    │
│ │ AAL456  直线  5 词 │ │                                  │ 两条线差 310 m    │
│ │ SWA789  引导 12 词 │ │                                  │ 正在执行：        │
│ │ …                  │ │                                  │  航向 +50°（第3条）│
│ └────────────────────┘ │                                  │  高度 2000 ft     │
│                        │                                  │  速度 180 kt      │
│ 显示                   │                                  │                   │
│ ☑真实 ☑词 ☐模型 ☑指令点 │                                  │ [打开读数核对窗口]│
└────────────────────────┘                                  └───────────────────┘
┌─ 句子条（底部，跨整宽；WorkbenchBottomBar 上方）──────────────────────────────┐
│跑道│05L 05L 05L 05L 05L 05L 05L 05L 05L 05L 05L 05L 05L 05L 05L 05L 05L 05L │
│航向│+90 +90 +90 +90│+50 +50 +50│+20│ 0   0   0   0   0   0   0   0   0   0  │
│高度│3000 3000 3000│2000 2000 2000 2000 2000│1000 1000│ 0   0   0   0   0   0 │
│速度│210 210│180 180 180 180 180│160 160 160│140 140 140 140 140 140 140 140 │
│切入│ —   —   —   —   —   —  │≤30 ≤30 ≤30 ≤30 ≤30 ≤30 ≤30 ≤30 ≤30 ≤30 ≤30  │
│    └0───20──40──60──80─100─120─140─160─180─200─220─240─260─280─300─320─340s┘│
│     │ = 这一刻换了词（发出一条指令）   ░ = 读到了但没成词（absorbed）         │
│     ▲ 当前时刻（跟 Cesium 时钟联动，可拖）                                    │
└───────────────────────────────────────────────────────────────────────────────┘
```

句子条是这个模块的主角：一行一类词，横轴是时间，换词的位置画一条竖线。用户一眼就能看到
"这架飞机一共被说了几句话、在哪几个时刻说的"。它和三维画面用同一个时钟，拖句子条就是拖时间。

**句子条要按多大来画**（数来自真产物的 `summary.json`，train 6853 架，我核过）：

| 量 | 值 | 对排版的意思 |
|---|---|---|
| 一架的时长中位数 | 296 s | τ = 10 s ⇒ **一句话典型 30 个位置**，句子条按 30 格设计，能横着放下 |
| 全部位置 | 274 230 | — |
| 其中换了词的 | 31 559（**11.5 %**） | 30 格里大约 **3–4 根竖线**；竖线是稀疏事件，可以画粗、可以标号 |
| 一架的指令条数中位数 | 航向 2、高度 2、速度 3、切入 1、跑道 1（共 9） | 右侧读数一屏放得下整句话，不用滚动 |
| 各类用到的词数 | 航向 36/36、高度 11/11、速度 23/23、切入 3/3、跑道 4/4 | **没有一类有空档**，图例可以把每一类的全部词都列出来 |

时长是中位数，不是上限：长的航班位置会多不少，所以句子条要能横向滚动或压缩，
**不能按 30 格写死**。

### 3.2 读数核对窗口（点右侧按钮打开，走 portal 渲染到 `document.body`）

这是现有 300 页人工核 PNG（`instruction_vocabulary.py` 的 `hand_check_figure`）的可交互版本。
那些 PNG 是 2×2 的四张静态图：平面图 + 三路信号配阶梯线。这里把它们做成联动的、能悬停读数的。

```
┌─ 读数核对 · DAL123 · 跑道 05L · 引导 · 9 条指令 · 3 条 absorbed ──────── [×] ┐
│ 平面图（跑道坐标系，入口在原点）                    ┌── 图例 ─────────────┐ │
│ 横向 km                                             │ ━━ 真实   ━━ 词画的  │ │
│  +8 ┤                                               │ ● 航向 ● 高度       │ │
│  +4 ┤        ╭────────────╮                         │ ● 速度 ● 切入       │ │
│   0 ┤ ━━━━━━━╯            ╰──●━━━━━━━━━━━━━━▶│入口   │ ─── 航道中心线      │ │
│  −4 ┤  ┈┈┈┈┈┈┈┈┈┈╮      ╭┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈   └─────────────────────┘ │
│  −8 ┤            ╰──────╯                                                  │
│     └──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──── │
│          −40    −35    −30    −25    −20    −15    −10     −5      0  km  │
├───────────────────────────────────────────────────────────────────────────┤
│ 航向（相对最后进近航道，°；展开过的，不在 ±180 处跳）                       │
│ +180┤░░░                                                                   │
│  +90┤   ━━━━━━━━━▬▬▬▬▬▬▬╲                        ▬▬ = 这一段生效的词的目标 │
│    0┤                    ╲━━━━▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬  ━━ = 实测（6 s 平滑）    │
│     └────────────────────────────────────────────────────────────────────  │
│ 高度（跑道入口以上，m）                                                     │
│ 1000┤▬▬▬▬▬━━━━━╲                                                          │
│  500┤           ╲━━━▬▬▬▬▬╲                                                │
│    0┤                     ╲━━━━━━━━━━━━━━━━━▬▬▬                           │
│     └────────────────────────────────────────────────────────────────────  │
│ 地速（m/s）                                                                 │
│  110┤▬▬▬━━━╲                                                              │
│   90┤       ╲━━▬▬▬▬░░░░━━╲                                                │
│   70┤                     ╲━━━━━━━▬▬▬▬▬▬▬▬▬▬▬▬▬▬                          │
│     └──┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬───   │
│        0    60   120   180   240   300   360   420   480   540   600  s    │
│                          ▲ 当前时刻（与主屏、三维、句子条同一个）            │
│ ░ = absorbed（读到了这段机动，但按规则没成词；悬停说原因：同词/变化太小/尾巴太短）│
└───────────────────────────────────────────────────────────────────────────┘
```

**比 PNG 强在哪**（这是用户要求"不要闷头写、要好设计"的落点）：

1. **四张图共一个时间游标**，拖任意一张，平面图上的点和三维里的飞机一起动；
2. **悬停出数**：悬停一条 `▬▬` 说"第 4 条航向指令，目标 +50°（词 5），在 t=138 s 发出，
   t=176 s 稳住，离格子中心 2.1°"；悬停一段 `░` 说它为什么没成词；
3. **点击跳转**：点一条指令，主屏三维把镜头移到它发出的位置，时钟跳到那一刻；
4. **两条航迹在同一张平面图上**（真实实线、词画的虚线），差在哪一眼看得见；
5. **按分层过滤航班**（直线进近 / 雷达引导），和实验里的分层用同一套规则
   （`data/approach_difficulty.py`，`STRAIGHT_TORTUOSITY = 1.05`）。

---

## §4 数据从哪来

### 4.1 谁写、写到哪

`aeroviz-4d/public/data` 不进 git（本地产物，由预处理脚本重新生成）。Training 的输入按同样的方式办：
**由一个新的 ts runner 写进去**，路径

```
aeroviz-4d/public/data/airports/<ICAO>/training/
    index.json                       有哪几个抽样集
    <集合 id>/sample.json            这一集的全部内容（航迹 + 句子 + 指令 + 几何航迹）
```

**为什么是 ts runner，不是 `aeroviz-4d/python/` 里的脚本**：那个目录下的 Python **不准 import 建模树**
（`aeroviz-4d/CLAUDE.md` 第一段），而导出必须调标注器（`manoeuvre/instructions.read_instructions`）、
必须通过 checkpoint 重建 cohort 的数据（`experiments/support.rebuild_cohort`）、还要用
`outputs/guidance/` 的常数飞几何航迹 —— 全是建模树。所以放

```
4dTrajectory/ts_transformer/experiments/instruction_sample_export.py
→ python run_ts.py instruction_sample_export \
      --vocabulary <…/vocabulary_tau10/instruction_vocabulary.json> \
      --executor   <…/two_tier_v3_grid_20260918/L60_D20_s1337/checkpoint.pt> \
      --cohort     <…/two_tier_v3_grid_20260918/cohorts/L60_D60/development_cohort.json> \
      --out        aeroviz-4d/public/data/airports/KRDU/training/vocabulary_tau10 \
      --flights 40 --seed 1337 [--split train]
```

checkpoint 在这里只是"进数据的门"（C25：通过它的 data provenance 重建同一批飞机），不读它的权重。
抽样分层与 B0′ 的人工核同一套：一半直线进近、一半雷达引导，种子给定，短了就报错不偷偷补。

### 4.2 `index.json`

```json
{
  "schema": "aeroviz-training-index-v1",
  "writtenUtc": "2026-09-20T12:00:00Z",
  "airport": "KRDU",
  "sets": [
    {
      "id": "vocabulary_tau10",
      "kind": "vocabulary-readback",
      "title": "指令词表 τ=10 s（五类词）",
      "file": "vocabulary_tau10/sample.json",
      "vocabularySha256": "<词表 spec 的 sha256，B0″ 跑完才有>",
      "runwaySha256": "<跑道词类别集合的 sha256，与上面那个各管各的>",
      "readingRule": "plateau-v10",
      "tokenStepS": 10.0,
      "flights": 40,
      "cohort": { "split": "train", "straightIn": 20, "vectored": 20, "seed": 1337,
                  "path": "4dTrajectory/outputs/KRDU/experiments/two_tier_v3_grid_20260918/cohorts/L60_D60/development_cohort.json" },
      "source": { "vocabulary": "…/vocabulary_tau10/instruction_vocabulary.json",
                  "executorSha256": "21b6cbd19a68…" }
    }
  ]
}
```

`kind` 目前只有两个取值：`vocabulary-readback`（面板一、二；词从真航迹读出来）和
`prior-generated`（面板三；词是模型说的）。**新增取值要同时改导出器和前端的校验，改一处就坏。**

### 4.3 `sample.json`

```json
{
  "schema": "aeroviz-training-sample-v1",
  "setId": "vocabulary_tau10",
  "airport": "KRDU",
  "kinds": ["heading", "altitude", "speed", "intercept", "runway"],
  "vocabulary": {
    "sha256": "<词表 spec 的 sha256>", "readingRule": "plateau-v10", "tokenStepS": 10.0,
    "headingBinDeg": 10.0, "altitudeBinM": 304.8, "altitudeMaxM": 3048.0,
    "speedBinMps": 5.1444, "speedMinMps": 51.444, "speedMaxMps": 164.62,
    "interceptAngleBinsDeg": [30.0, 45.0],
    "runwayIdents": ["05L", "05R", "23L", "23R"],
    "runwaySha256": "<跑道类别集合自己的 sha256；不在上面那个里，见 §4.4>",
    "noIntercept": -1
  },
  "geometry": { … 见 §5.4 … },
  "flights": [ { … 见下 … } ]
}
```

一架飞机：

```json
{
  "flightKey": "DAL123_05L_a1b2c3_1699999999",
  "callsign": "DAL123",
  "runway": "05L",
  "stratum": "vectored",
  "durationS": 612.0,
  "establishedFromStart": false,
  "threshold": { "lon": -78.7875, "lat": 35.8776, "altM": 132.0, "courseDeg": 52.0 },
  "observed": {
    "tS": [0, 2, 4, "…"],
    "lon": ["…"], "lat": ["…"], "altM": ["…"],
    "toGoM": ["…"], "crossM": ["…"], "heightM": ["…"],
    "relCourseDeg": ["…"], "courseUnwrappedDeg": ["…"],
    "groundSpeedMps": ["…"], "established": [0, 0, 1, "…"]
  },
  "sentence": { "positionsS": [0, 10, 20, "…"],
                "words": [[9, 3, 11, -1, 0], [9, 3, 11, -1, 0], "…"] },
  "instructions": [
    { "kind": "heading", "word": 9, "target": 88.4, "issuedS": 120.0, "settledS": 164.0, "clamped": false }
  ],
  "absorbed": [
    { "kind": "speed", "startS": 240.0, "endS": 268.0, "word": 11, "change": 3.1, "reason": "small change" }
  ],
  "geometric": {
    "tS": ["…"], "lon": ["…"], "lat": ["…"], "altM": ["…"],
    "toGoM": ["…"], "crossM": ["…"], "heightM": ["…"], "groundSpeedMps": ["…"],
    "endsAtThreshold": true, "endReason": "crossed-threshold",
    "finalGapM": 412.0, "crossTrackP95M": 780.0, "meanGapM": 530.0
  }
}
```

对应关系（**这些是镜像，必须逐条对得上，不许前端另起一套**）：

| JSON 字段 | 来源 |
|---|---|
| `sentence.words` 的五列顺序 | `instructions.INSTRUCTION_KINDS`（航向、高度、速度、切入、跑道） |
| `sentence.positionsS` / `words` | `Reading.positions_s` / `Reading.words` |
| `instructions[]` | `Instruction.to_dict()` 原样，键名转驼峰 |
| `absorbed[].reason` | `ABSORBED_SAME_WORD` / `ABSORBED_SMALL_CHANGE` / `ABSORBED_SHORT_TAIL` 三个之一 |
| `observed.*` | `instructions.course_frame(series)` 的各路 + 经纬高 |
| `vocabulary.*` | `Vocabulary.to_dict()` 的对应字段；`runwayIdents` 来自 `RunwayVocabulary` |
| `-1` | `instructions.NO_INTERCEPT`（只有"切入"这一类允许没有词） |

**高度的基准**：`altM` 是海拔（MSL），和 CZML 出口一致 —— 观测的 ADS-B 是椭球高（HAE），
在 `flight_scenarios` 那一层换算一次，出到 CZML 时再换回去。导出器写的是 MSL，**不在这里做第二次换算**。
`heightM` 是"跑道入口以上"的相对高，词表用的就是它。

**为什么整条航迹进文件而不是复用 Observe 的 CZML**：Observe 的 CZML 是整机场几万架的大文件，
里面没有跑道坐标系下的量（沿航道距离、横向距离、相对航道的航向），也没有平滑后的信号。
Training 要画的三路时间图全是这些量。40 架 × 约 300 行 × 10 个数 ≈ 每个集合 5–10 MB，可接受。
**这是一个刻意的重复：同一条航迹在两个文件里各有一份。** 理由是坐标系不同、用途不同，
而且抽样集是只读快照（它要和某一个词表 sha 绑死）。

### 4.4 词表的四条性质，界面必须体现

这四条是 2026-09-20 当天定下来的，写在这里是因为它们直接决定界面画什么、不画什么。
第四条是**最容易画错的一条**。

**一、词是五类，不是四类。** 跑道、航向、高度、速度、切入。跑道是 2026-09-20 加的第五类
（计划的决定 D62）。为什么必须加：其余四类全是**相对某一条跑道**量出来的角度和高度
（航向相对最后进近航道、高度相对跑道入口），所以一句没有跑道词的话**说不出"去哪条跑道"**；
而且真实管制里跑道本来就是管制员要说出口的一句话。
→ **界面只要显示一句话，就必须同时显示这句话是对哪条跑道说的**，而且要写明另外四类是相对它量的
（§3.1 的右侧读数把跑道排在第一行，§3.2 的窗口标题里也带）。这不是装饰：同一串航向词
配不同的跑道词，指的是完全不同的两条航迹。

**二、跑道词不在词表的 sha 里。** 跑道词的类别集合是**每个机场各一套**，它挂在词表产物的 spec
**旁边**（`runway_idents`），不进 `sha256`；做法和机型词表（`manoeuvre.context.TypeVocabulary`）
一样。为什么这么设计：进了 sha，一份词表就不能通用于五个机场了。
**后果，界面上要认这个后果**：两份 sha 相同的产物，跑道类别集合可以不一样。所以
`instructions.py` 另有一个 `runway_sha256(runway_vocabulary)`（第 350 行），先验的产物里
这两个 sha 是**分开两个字段**存的（`experiments/instruction_prior.py` 第 156–157、166–167 行）。
Training 的 `index.json` / `sample.json` 照抄这个做法：`vocabularySha256` 和 `runwaySha256`
各是一个字段。**只比对其中一个就当成"同一份词表"是错的。**

**注意一个坑**：`runway_sha256` **不是词表产物文件里的一个字段**（`instruction_vocabulary.json`
的顶层只有 `schema / written_utc / spec / sha256 / runway_idents / words / cohort_identity /
counts / source` —— 我开文件核过）。它是一个**函数**，导出器要拿 `runway_idents` 建出
`RunwayVocabulary` 再调它算出来。T3 不要去文件里找这个字段，找不到。

**三、没有"复飞"这个词，而且以后也不会有。** 这一条量过才定的（结果文档 §13.5）：
复飞在机队里是真的 —— 五个机场 44 622 条航迹里有 42 条（0.094 %，和公开的复飞率同量级，
逐架核过剖面）—— 但**词表是在 25 km 到达切片上读的，切片只留最后那一次成功的进近**，
所以复飞在这个 cohort 里一次也不会触发。加了它只会是一个恒为 0 的字段，
按仓库的规矩（**永远不触发的词比没有更糟**）不加（决定 D63，当天由"加"翻转为"不加"）。
→ **界面不要留复飞的位置，也不要画一个永远空着的槽。** 这是这份数据的一条限制：
**本阶段的句子表达不了"这次不落地"**；要改得动上游的到达窗口，那会改掉每一个 ts 数据集划分，
不在这个模块的范围内。这句限制要写在读数核对窗口的说明里，不然看的人会以为是读漏了。

**同一条规矩也适用于这个模块自己**：一个永远不会被设置的字段、一个永远不会亮的提示条，
就不要建。（用户 2026-09-20 否掉"从 `_superseded/` 导一份临时数据"时，`supersededSource`
标志和它配的红条就一起删掉了，不留在设计里。）

**四、跑道词的类别要从产物里读，绝不能从 `runway.geojson` 读。** 这是最容易画错的一条，
因为两边在 KRDU 就对不上：

| | 有哪几条 | 几条 |
|---|---|---|
| 机场的跑道表（`public/data/airports/KRDU/runway.geojson`、`landings/index.json`） | 05L 05R **14** 23L 23R **32** | 6 |
| **词表产物的 `runway_idents`** | 05L 05R 23L 23R | **4** |

差的那两条不是读漏了，是**上游就没有**：这条线全部建在到达清单
`harvest-arrivals-v5-takeoff-excluded` 上，KRDU 那 14 435 架里只有这四条
（05L 3076、05R 1491、23L 3230、23R 6638）；跑道 14 只有 13 次落地、跑道 32 有 1 604 次，
两条都在上游被排除了。

→ **界面上任何一处列出跑道词的地方（选择器、图例、句子条的跑道行），类别集合都读
`sample.json` 的 `vocabulary.runwayIdents`，不读机场的跑道表。** 画成 6 条会告诉看的人
"模型能说 32"，而它说不出来 —— 这是凭空多给了模型两个它没有的词。
**而且要在看得见的地方写一句**：这里的跑道类别是**到达清单覆盖到的跑道**，不是机场的全部跑道。
（放在句子条跑道行的悬停说明里，或读数核对窗口的说明里。）

这一条对别的机场同样成立，数也各不相同，所以**不要把 4 写死**：`runwayIdents` 有几个就是几个。

### 4.5 文件不在的时候界面显示什么

这是设计的重点，因为**今天这个文件就是不存在的**。三种状态各有各的说法，都要说出路径和命令：

```
① index.json 取不到（404 或不是 JSON）
┌──────────────────────────────────────────────────────────────┐
│ KRDU 还没有导出 Training 数据。                              │
│                                                              │
│ 界面找的是：                                                 │
│   public/data/airports/KRDU/training/index.json               │
│                                                              │
│ 它由这条命令生成（需要先有词表产物）：                        │
│   python run_ts.py instruction_sample_export \                │
│       --vocabulary <…/vocabulary_tau10/instruction_vocabulary.json> \
│       --out aeroviz-4d/public/data/airports/KRDU/training/…   │
│                                                              │
│ 注：五类词的词表产物（plateau-v10）正在生成（两层计划 §5.2.5  │
│     的步骤 B0″）。四词的旧产物已于 2026-09-20 删除，          │
│     被取代的那几份不会拿来顶替。                              │
└──────────────────────────────────────────────────────────────┘

② index.json 有，但没有某一类集合
   → 面板三（模型说的句子）显示："这个机场还没有 prior-generated 的集合。
      第二层模型要到阶段 B3′ 才有产物。" 面板一、二照常工作。

③ 某个集合的 sample.json 取不到 / 字段不对
   → 只有这一个集合在下拉里标灰并写出坏在哪个字段，其它集合照常。
```

**③ 是对 `AV6` 的一次有意偏离，要写在代码注释里。** 比较视图的清单校验是
`.every(isComparisonCategory)` —— 一个坏条目清空整个机场的选择器，这在仓库里坑过两次
（`aeroviz-4d/docs/35-viewer-reference.md` AV6）。Training 反过来做：**逐集合校验，坏的标灰**。
理由是它是开发期的视图，"导出到一半"是常态而不是事故；而且它的坏消息必须指向字段名，
不能只说"清单无效"。

另外记住 `AV5`：**开着的 vite 开发服务器看不到新建的目录**（`public/data` 不在 watch 里）。
第一次导出之后要重启前端（杀 `vite` 的 node 进程，不是 `npm run dev` 那层壳）。
这条要写进空状态的提示文字里，否则第一次用的人一定会撞上。

---

## §5 词怎么画成航迹

### 5.1 要解决的问题

一句话只说四个绝对目标：**转到（相对航道）某个角度、降到入口以上某个高度、减到某个地速、
（某个角度）切上航道**。它不说转多快、降多快 —— 那是机型和动力学决定的，是执行器（第一层）的事
（计划的决定 D50）。所以"把词画出来"必须补一套最简单的运动学：**朝目标转、朝目标降、朝目标减**，
转弯速率受坡度限制。

### 5.2 复用了什么

| 量 | 值 | 来自哪里 |
|---|---|---|
| 转弯用的坡度 | 20° | `outputs/guidance/route.py` `ROUTE_BANK_RAD` |
| 转弯半径 | `max(V, 40)² / (g·tan 20°)` | 同上 `route_turn_radius_m()` |
| 重力 | 9.81 | `geometry/flyability.py` `G` |
| 高度误差的收敛时间 | 12 s | `outputs/guidance/controller.py` `HEIGHT_GAIN_S` |
| 下降角上限 | 6° | 同上 `DESCENT_MAX_RAD` |
| 爬升角上限 | 2° | 同上 `CLIMB_MAX_RAD` |
| 加减速上限 | 1.0 m/s² | 同上 `ACCEL_MAX_MPS2` |
| 跑道坐标系 | 沿航道距离 / 横向距离 / 相对航向 | `data/approach_difficulty.course_frame_rows`，经 `instructions.course_frame` |
| 每格的中心值 | — | `Vocabulary.heading_centre_deg / altitude_centre_m / speed_centre_mps` |
| 某一刻生效的词 | — | `Reading.words_at`（取"之前最近的那个位置"，和执行器读词的规则同一处） |

**没有直接复用 `route.build_route`。** 它解的是另一个问题：从当前位姿铺一条转—直—转的路径切到航道上某一点
（Dubins 路径）。这里没有"要切到哪一点"这个输入 —— 句子只给角度目标。所以复用的是它的**常数和半径公式**，
不是它的路径搜索。这一点写明，免得后人以为漏用了。

`route.DECEL_RATE_MPS2 = 0.5` **没有用** —— 那是"路线的速度计划"里减速段的速率，是另一个旋钮；
这里追的是词的速度目标，用的是控制器的加减速上限 1.0 m/s²。两个数不要混为一谈。

### 5.3 一步是怎么算的（dt = 1 s）

状态：跑道坐标系里的位置（沿航道 `d`、横向 `x`）、入口以上高度 `h`、地速 `V`、相对航道的航向 `χ`。

1. 用 `words_at(t)` 取此刻生效的五个词；
2. **航向**：目标 `χ* = heading_centre_deg(词)`。最大转弯速率 `ω_max = g·tan(20°)/V`；
   本步转过的角度 = `sign(χ*−χ) · min(|χ*−χ|, ω_max·dt)`（角度差按 ±180° 绕回）；
3. **高度**：目标 `h* = altitude_centre_m(词)`。要求的航迹角 `γ = atan((h*−h)/(V·12 s))`，
   下降不超过 6°、爬升不超过 2°；`h += V·sin γ·dt`；
4. **速度**：目标 `V* = speed_centre_mps(词)`；`V += clamp(V*−V, ±1.0·dt)`；
5. 水平位移 `V·cos γ·dt` 沿新的 `χ` 推进，得到新的 `d`、`x`；
6. 反算经纬高写进文件（用 `geokit` 的换算，跟 CZML 出口同一处定义）。

### 5.4 六条必须写明的假设

`sample.json` 的 `geometry` 块把它们写进文件，界面上也要能看到 —— **一个不写明的近似比没有还糟**：

```json
"geometry": {
  "method": "instruction-kinematics-v1",
  "dtS": 1.0,
  "bankDeg": 20.0, "heightGainS": 12.0,
  "descentMaxDeg": 6.0, "climbMaxDeg": 2.0, "accelMaxMps2": 1.0,
  "startsAt": "observed-first-row",
  "interceptWordFlown": false,
  "windModelled": false,
  "aircraftTypeModelled": false,
  "stopRule": "threshold-plane-or-sentence-end+tau",
  "constantsFrom": ["outputs/guidance/route.py", "outputs/guidance/controller.py",
                    "geometry/flyability.py"]
}
```

1. **起点取真航迹的第一行**（位置、高度、地速、航向）。句子里没有"从哪儿开始"这一项，词全是相对量。
   **后果要说清楚：两条线的起点按构造就是重合的，所以它们之间的差是"词没说出来的那部分"，
   不是某个模型的误差。** 界面上这句话要写出来，不能让人看成模型评估。
2. **切入词不参与飞行，只画一个标记。** 航向词已经在指挥这次转弯，再让切入词也去操舵就是重复计一次；
   而且当前产物里切入词只带角度（`Instruction.target` = 切入前保持的相对航向），不带位置锚，
   要飞它就得先发明一个锚点。这是**待定项**，见 §9 问题 4。
3. **不建风模型。** 词说的是**地速**（ADS-B 没有空速），风已经在这个数里了 —— 这一条对这里反而有利。
4. **不分机型**：坡度固定 20°，不按机型改。
5. **管制员说话比航迹动早几秒**，这几秒航迹里没有，读不出来（`instructions.py` 的模块说明已写明）。
   所以词画出来的航迹相对真航迹会**系统性地晚一点**开始转，这是读法的性质，不是画法的缺陷。
6. **终止**：`d ≤ 0`（过了入口平面）、或者过了句子最后一个位置再加一个 τ、或者到观测时长 + 120 s 的硬上限。
   哪一条先到就写进 `endReason`，同时写 `finalGapM`（停下来时离入口多远）。**不偷偷延长航迹去凑到入口。**

### 5.5 它是诊断，不是模型的答案

这条是用户的既有规则（记忆条目 *no-procedure-answer-as-model-fix*，以及计划 §5.2.3 的
"规则制导只作基线与诊断，不作模型的修法"）。界面上这条线的图例文字必须是
**"按词用规则画出来的（基线）"**，不能只写"预测"或者把它和模型输出画成同一种样式。
颜色上：真实白、规则橙（实线细）、模型紫（虚线）—— 和比较视图的配色约定不冲突。

---

## §6 v1 不做什么

写清楚，免得后面有人以为忘了：

1. **不能在界面上改一个词再重飞。** 那需要把 §5.3 的运动学搬进浏览器（等于制造第二份定义），
   或者给后端加一个端点。**第二条路是 v2 的方向**（`aeroviz_backend` 加
   `POST /training/fly-sentence`，收一串词返回一条航迹），v1 不做。
2. **不做人工核的判定录入**（一致 / 漏读 / 误读）。那是实验产物，要进带 sha 的 artefact，
   不能进浏览器的本地存储（本地存储只在这台浏览器里，Claude 和别人都读不到）。
3. **不浏览全体**。B cohort 是 train 6853 / val 1404 架；导出是**抽样**，份额和种子写在
   `index.json` 里，界面上明写"这是 40 架的抽样"。不做无声的截断。
4. **面板三（模型说的句子）v1 只有占位**，因为阶段 B3′ 还没跑，没有产物。占位要说清楚在等什么。
5. **不做 τ = 5 s 的对照**（计划的 B4′）。两个 τ 只是两个集合，等有了产物自然就能并排看。
6. **不改 Observe、不改比较视图**。Training 只新增文件，不动既有任务页的行为。
7. **不做多机**（阶段 C 之后的事）。
8. **不做演示模式的特别排版**（`presentationMode`），v1 沿用左坞 + 底栏。

---

## §7 步骤（按这个顺序做，一步一评审）

每一步都能单独落地、单独评审。**T1、T2、T4a、T4b 在没有词表产物的情况下也能做完**（界面上是
空状态，测试里用 mock 数据）；T3、T5、T6 等 B0″ 的五类词产物。

| 步 | 做什么 | 落在哪 | 怎么算做完 | 依赖 |
|---|---|---|---|---|
| **T1 ✅完成（未提交）** | 前端骨架：`WorkbenchMode` 加 `"training"`；`TASK_TABS` 在 Observe 右边加一项；`WorkbenchLeftDock` 加分支；`TrainingPanel` 显示 §4.5 状态 ①（没有数据 + 路径 + 命令 + 重启前端的提醒）。**T1 不发任何请求**：状态 ② ③ 要先有解析好的清单，和 T2 一起来；在 T3 之前"没有导出"在每台机器上都是实话，所以这个阶段它不会说谎 | `context/AppContext.tsx`（联合类型 +1）、`components/WorkbenchTopBar.tsx`（标签 +1）、`WorkbenchLeftDock.tsx`（分支 +1）、新 `components/TrainingPanel.tsx`、`index.css`（一段样式）、新 `__tests__/TrainingPanel.test.tsx` + 另两个测试各加一条 | **已达成**：三个测试文件 17 条全过；`npx tsc --noEmit` 干净；全套 83 文件 569 条全过 | 无 |
| **T2** | 数据契约（纯前端）：类型 + 逐字段校验 + fetch；坏一个集合只坏这一个；**mock 数据（一个手写的最小样本）建在 `__tests__/` 下，供 T4 用** | 新 `src/data/trainingSample.ts` + `__tests__/trainingSample.test.ts` + `__tests__/trainingSample.fixture.ts` | 四个用例：index 缺失、集合缺 `kind`、sample 的 `words` 列数不是 5（要报出实际列数和期望列数）、`runwaySha256` 缺失（跑道 sha 是独立字段，见 §4.4） | T1 |
| **T3** | 导出器（Python）：先只导句子那部分（`observed` + `sentence` + `instructions` + `absorbed`），几何航迹留到 T5。`runwaySha256` 要**自己算**（§4.4 的坑） | 新 `4dTrajectory/ts_transformer/experiments/instruction_sample_export.py` + `tests/test_instruction_sample_export.py`；`run_ts.py` 注册 | 小 fixture 上跑通，断言 schema；**词表产物不在时按名字拒绝**，不做任何回退；**一个测试钉住抽样是人工核那 300 页的前 40**（V13） | **已满足**：产物在 `…/vocabulary_tau10/`（§1） |
| **T4a** | 底部句子条：五行、换词竖线、absorbed 灰底、时间游标接 Cesium 时钟 | 新 `components/TrainingSentenceBar.tsx` | 用 T2 的 fixture 渲染；点第 3 条指令，时钟跳到 `issuedS` | T2 |
| **T4b** | 读数核对窗口：平面图 + 三张时间图，portal 到 `document.body`（`AV7`），悬停出数、点击跳转 | 新 `components/TrainingReadbackWindow.tsx` | 四张图共一个游标；悬停 absorbed 说出原因 | T4a |
| **T5** | 几何航迹（Python）：§5.3 的运动学，写进 `sample.json` 的 `geometric`；常数全部 import，不抄 | 导出器里一个独立模块 + 单元测试 | 三个手算用例：保持航向直飞、转 90° 的半径对得上 `route_turn_radius_m`、从 1000 m 降到 0 的时间对得上 12 s 收敛 + 6° 上限 | T3 |
| **T6** | 面板二：两条航迹进三维（白 / 橙）+ 平面图加第二条线 + 右侧"这一刻差多少" | `TrainingPanel` + 新 hook `useTrainingTrackLayer` | 切换显示开关，两条线能单独开关；差值读数对得上 `geometric.meanGapM` | T5、T4b |
| **T7** | 面板三：导出器加 `kind: "prior-generated"`（模型说的句子 + 同一套运动学画的航迹）；界面用同一套组件画第三条线 | 导出器 + `TrainingPanel` | 三条线同屏；模型那条的图例写明是模型说的词 | 阶段 B3′ 有产物 |
| **T8** | 发布检查：`npm run check-publication` 学会 `training/index.json`（或加一个 `check-training`），把 `AV6` 的教训写进去 | `scripts/check_publication.ts` | 故意写坏一个字段，脚本报出集合 id 和字段名 | T3 |

**优先级**：T1 → T2 → T4a → T4b 先把界面做出来（用 mock 数据，只放测试目录，**不进 `public/`**）；
T3 → T5 → T6 等词表产物；T7、T8 最后。

**为什么把 T4 排在 T3 之前**：界面这一侧的工作量最大，又完全不依赖真产物；而 T3 的输出卡在
B0″ 上。先做界面，产物一到就能接上看。

**T3 不再等任何东西**（2026-09-20 当天 B0″ 就跑完了）：`--vocabulary` 指
`…/two_tier_v3_bprime_20260920/vocabulary_tau10/instruction_vocabulary.json`（sha `a5450c0fcd90…`）。
顺序不变，还是先把界面做到 T4b，再回来做 T3。

---

## §8 决定项（做了什么选择，为什么）

| # | 决定 | 理由 | 谁定的 |
|---|---|---|---|
| V1 | Training 是第五个任务页，放 Observe 右边，共用顶栏的机场/跑道和选中的航班 | 看的是同一批飞机的同一件事的下一个问题；不新建第二套模式系统 | 用户指定位置；其余我定 |
| V2 | Training **不加载** Observe 的观测 CZML，自带抽样文件 | 那份大文件会驱动共享时钟，在别的任务里加载会顶掉播放（`observedTracks.ts` 写明） | 我定 |
| V3 | 输入由**新的 ts runner** 写进 `public/data/airports/<ICAO>/training/` | `aeroviz-4d/python/` 不准 import 建模树，而导出必须调标注器和 cohort 重建 | 仓库规矩 |
| V4 | 抽样文件里**自带航迹**（跑道坐标系 + 经纬高），不复用 CZML | 三张时间图要的量 CZML 里没有；抽样集是绑 sha 的只读快照 | 我定 |
| V5 | 几何航迹在 **Python 里算**，文件里带结果 | 运动学常数只能有一处定义；前端唯一允许的镜像是**生成**出来的 `geoConstants.json` | 仓库规矩（单一来源） |
| V6 | 几何航迹的**起点取真航迹第一行** | 句子不说起点；两条线起点重合是构造使然，界面要写明差值不是模型误差 | 我定 |
| V7 | **切入词只画标记，不参与飞行**，而且**界面上要写出"这是标记，不是飞出来的"** | 航向词已在指挥这次转弯，两个都飞就是把同一次转弯算两遍；当前产物的切入词只带角度、不带位置锚，要飞它得先发明一个锚点 | 我提，用户 2026-09-20 认可并要求界面写明 |
| V8 | 每个集合**单独校验**，坏的标灰，不清空整个列表 | `AV6` 的教训反过来用：开发期"导出到一半"是常态，坏消息必须指向字段名 | 我定 |
| V9 | 几何航迹那条线的图例写**"按词用规则画的（基线）"** | 用户既有规则：规则制导只作基线和诊断，不作模型的修法 | 用户既有规则 |
| V10 | 二维图**自己画 SVG**，不引图表库 | 仓库里已有先例（`ApproachViewPanel` 手画 SVG），不为一个视图加一个依赖 | 我定 |
| V11 | 空状态必须写出**路径 + 生成命令 + 重启前端的提醒** | `AV5`：开着的 vite 看不到新建目录；不提醒的话第一次用一定撞上 | 我定 |
| V12 | **不从 `_superseded/` 导任何数据**；真产物出来之前前端用 mock 数据（只在测试目录），界面上照样是空状态 | 被取代的产物不与真产物并存；而且界面上给用户看假航迹，和在测试里用假数据是两回事 | **用户 2026-09-20**（「不用做，直接等跑完 B0 后用真实数据，前端测试可以用 mock 数据」） |
| V13 | 抽样：**40 架**，直线进近 20 / 雷达引导 20，种子 1337，池子和随机数生成器的建法**逐行照搬** `instruction_vocabulary.py` 的人工核抽样 | 两个视图要看的是**同一批飞机** —— 纸上核过的和屏幕上显示的对得上，才能互相印证。照搬建法之后这是**可验证的**：人工核每层抽 150，我每层抽 20，`rng.permutation(pool)` 的消耗只跟池子大小有关、跟抽几个无关，所以**我这 20 架正好是那 150 架的前 20**（已用 KRDU 的真实池子 3823 / 3021 跑过，两层都成立）。**建法一改这条就断**，所以 T3 要有一个测试钉住它 | 我提，用户 2026-09-20 确认（并点明"同一批"是重点） |
| V14 | 第一个机场：**KRDU** | B cohort 就是 KRDU 的 L60_D60 | 用户 2026-09-20 |
| V15 | 三条线就叫**「真实航迹」「按词画的」「模型说的」** | 说的是它们各自是什么，不用新造词 | 我提，用户 2026-09-20 确认 |
| V16 | **不建复飞相关的任何字段、槽位或提示**；这条限制写进读数核对窗口的说明 | 复飞在 cohort 里一次也不会触发（结果文档 §13.5），永远不触发的词比没有更糟；同一条规矩也杀掉了 `supersededSource` 标志和它的红条 | 仓库规矩 + 决定 D63 |
| V17 | 跑道词的类别**一律从 `sample.json` 的 `vocabulary.runwayIdents` 读**，绝不从 `runway.geojson` / `landings/index.json` 读；并在界面上写明"这是到达清单覆盖到的跑道，不是机场的全部跑道" | KRDU 两边差两条（6 对 4，见 §4.4 第四条）：画成 6 条等于告诉看的人"模型能说 32"，而它说不出来。数量不写死，有几个画几个 | **用户 2026-09-20 更正**（我原稿按机场跑道表写了 6 条，是错的） |
| V18 | **界面上的字用英文**，和这个应用其余部分一致（顶栏是 `Observe` / `Active Airport` / `All runways`）；中文只出现在这份设计文档和代码注释里 | 全前端 `src/**` 只有一个文件带中文，而且是注释（`WorkbenchBottomBar.tsx` 里的"联动"）。新模块跟着现状走，不在一个界面里混两种语言 | 我定（T1 落地时核过） |

---

## §9 待定项的去向（用户 2026-09-20 全部回答）

设计阶段列的五个问题都有答案了，**没有还挡着开工的问题**。答案已经写进 §8 的决定表
（V12–V15）和 §4.4，这里只留一句话的去向，外加真正还没定的东西。

| 当时问的 | 答案 | 落在哪 |
|---|---|---|
| 要不要从 `_superseded/` 导一份临时数据 | **不要。** 等 B0″ 跑完用真数据；前端测试用 mock 数据 | V12；`supersededSource` 标志和红条一并删除（§4.4 末段） |
| 抽样规模与种子 | 就按 40 架 / 20 / 20 / 种子 1337 | V13 |
| 先做哪个机场 | KRDU | V14 |
| 切入词 v1 不飞 | 同意，理由也认（飞两个等于算两遍）；**界面要写明是标记不是飞出来的** | V7 |
| 三条线的叫法 | 就用「真实航迹」「按词画的」「模型说的」 | V15 |

**我不同意的：没有。** 四个判断我都同意，理由和我写的是同一个；其中"两个视图要看同一批飞机"
这一点用户说得比我原来的理由更准 —— 我原来写的是"与人工核同一套分层和种子"，
真正的原因是**纸上核过的那几架和屏幕上显示的那几架必须是同一批，两边才能互相印证**，
已经按这个说法改进 V13（而且现在是可验证的，不只是说法）。

**我原稿写错、当天被更正的一处**（记下来，因为这类错会再犯）：我按机场的跑道表把 KRDU 的
跑道词写成 6 类，实际是 4 类 —— 词表的类别来自**到达清单的覆盖**，不是机场有几条跑道。
教训一般化：**凡是"某某有哪几类"，都要去产物里读，不要去看起来相关的另一个文件里读。**
已写成 §4.4 第四条和 V17。

**真正还没定的（都不挡 T1–T8）**：

1. **切入词要不要带位置锚** —— 这是词表那边的事（计划 §5.2.0 说它"带位置锚"，
   但当前产物里只有角度）。词表改了，这里的 V7 才需要重看。
2. **Training 以后要不要兼作人工核的录入界面** —— 现在明确不做（§6 第 2 条）。
   真要做，判定必须回到带 sha 的产物里，不能留在浏览器本地存储，那会是另一套存储设计。
3. **别的机场的跑道类别数** —— KRDU 是 4。其余四个机场要各自去它们的产物里读，
   现在还没有产物。界面不写死数量，所以这一条不挡任何事。

---

## §10 维护约定

- 本文是这个模块的设计与状态的唯一去处。实现过程中每落地一步，**回来更新 §7 的表**
  （加一列"提交"），以及 §1 的状态行。
- 落地之后产生的**长期事实**（契约、默认值、坑）：一行写进 `aeroviz-4d/CLAUDE.md` 的索引，
  全文写进 `aeroviz-4d/docs/35-viewer-reference.md` 并给一个新的 `AV` 编号 —— 那两处才是
  下一次会话真正看得到的地方，本文不是。
- 日志式的记录（哪天改了什么）写进 `docs/CHANGELOG.md`，不写在这里。
