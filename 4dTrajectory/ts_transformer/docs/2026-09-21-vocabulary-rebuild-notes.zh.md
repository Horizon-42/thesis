# 词表改版与词序列重建 · 交接笔记(2026-09-21)

压缩上下文前的状态快照。设计本身在 `2026-09-21-instruction_vocabulary_plan.zh.md`,
量测方法在 `2026-09-21_vertical_word_method.zh.md`,这份只记**当前进行到哪、代码改了什么、
下一步是什么**。

---

## 1 现在在做什么

**目标:给意图模型(第二层先验)产出词序列。** 执行器是后面的阶段,这次只在它"读词表"这一点上
被动跟着改。

| | 状态 |
|---|---|
| 词表代码 | **改完** |
| 读法(反算词序列) | **改完**,新读法 `segment-v12` |
| 测试 | **1318 项全绿** |
| 旧产物删除 | **已删** 573 MB(见 §5) |
| KRDU 词序列 | **已产出** |
| 五机场词序列 | **在跑**(见 §4) |
| 先验训练 | 未开始 |

### 1.1 怎么接上(压缩之后先看这里)

**分支 `dev-two-tier-feasibility`。这次的词表改动全部未提交**(约 22 个文件在工作区),按规矩要先
review 再 commit。HEAD 上另有别的 session 的提交——**这个 checkout 是多 session 共用的**,提交前务必
`git diff --cached --stat`,只 stage 自己的路径。

**五机场那一轮的重跑命令**(在跑时 PID 1729114,清空后按日志判断):

```
conda run -n aeroviz --no-capture-output python run_ts.py instruction_vocabulary \
  --airports KRDU KSJC KSTL KSMF KMSY \
  --cohort 4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/step5_pooled_cohort/development_cohort.json \
  --out 4dTrajectory/outputs/POOLED/experiments/two_tier_v3_bprime_20260921/vocabulary_five_airports \
  --hand-check 0
```

产物目录存在就会拒绝覆盖,重跑前要先 `rm -rf` 掉。日志在 scratchpad 的 `rebuild5.log`。
**判断有没有成功**:产物目录里要有 `instruction_vocabulary.json` / `sentences_train.json` /
`sentences_val.json` / `summary.json`,四个都在才算写完(前几次都是读完了在写的时候崩的)。

**前端**:读端**已经对齐新词表**(`trainingSample.ts` 里是 `verticalModesDeg` / `speedCentresMps`,
另一个 session 在 `b7922b1b` 做的)。但**发布出去的数据被我删了**(`public/data/airports/KRDU/training/`),
所以前端的 Training 面板现在是空的——新产物出来之后要用
`experiments/instruction_sample_export.py` 重新发布一次。

**旧计划文档** `2026-09-18_two_tier_plan_v3_B.zh.md` 已在头部标记废弃,并写明哪些部分**没有**被取代
(多机接口设想、里程碑与队列成本、逐条决定的沿革)。

---

## 2 新词表(已定,用户逐条拍板)

**sha `fffa8bdf24a0`,读法 `segment-v13`**(v12 的产物已作废,见 §7.2)。

| 词 | 取值 | 类数 |
|---|---|---|
| 跑道 | 类名是 `机场:跑道号`,按机场绑定(不进 sha) | 单 KRDU 4;五机场队列 **22**(清单里有 23) |
| 航向 | 相对最后进近航道,**5°** 一格 | 72 |
| 垂直 | **航迹倾角**:上升 3.0 / 平飞 0 / 下降 1.4 / 2.4 / 3.1 / 4.4 度(**下降为正**) | 6 |
| 速度 | 地速拟合档 44/56/63/68/74/79/86/93/99/107/114/121/129/138/147/157 m/s | 16 |
| 时长 | 2 秒一格,0–300 s | 151 |
| 结束 | 继续 / 落地 / 复飞 | 3 |

**容差**(不进 sha,是解码与读数的参数):垂直平飞档 ±0.1° 绝对,其余 ±7 %;速度 ±3 %。

**用户定下的其余几条**:训练用**五机场合训**;门二读数**训完立刻读**,不等另一版;重建产物**已获许可**。

### 关键陷阱

- **垂直词 0 是上升档,不是平飞。** 平飞是 word 1。任何"默认 0"的夹具都会变成爬升。
- **航向词翻倍**:旧的 18(180°)现在是 36,27(−90°)现在是 54。
- **速度档不均匀**,不能用 `(v - min) / bin` 算,只能取最近的档心。

---

## 3 代码改了哪些(全部已提交前状态,尚未 commit)

**`manoeuvre/instructions.py`** —— 主体。
- 常量:`HEADING_BIN_DEG=5.0`、`VERTICAL_MODES_DEG`、`SPEED_CENTRES_MPS`、三个容差、`VERTICAL_SEGMENTS=5`、`VERTICAL_FIT_POINTS=70`。
- `MANDATORY_KINDS` 的 `altitude` 改名 **`vertical`**(全仓库跟着改)。
- `Vocabulary`:`vertical_bin/centre/tolerance`、`speed_bin/centre/tolerance`、`_nearest`;`from_dict` 要把 JSON 读回的 list **还原成 tuple**(否则不相等、sha 漂)。
- **新读法** `_profile` / `_segment_costs` / `_breakpoints` / `_vertical_instructions`:高度对累计水平距离、动态规划取最优断点、每段斜率取档。
- **航向平台容差 4.0° → 2.0°**(必须严格小于半格 2.5°;量过:±4° 下 18.2 % 的平台跨两个词,±2° 下为 0,而平台反而更多)。

**`manoeuvre/instruction_kinematics.py`** —— 只因为它读词表:**词即 γ**(不再追目标高度),加高度地板。
**`outputs/guidance/controller.py`** —— 爬升限幅 **2° → 4°**(见 §6)。
**下游**:词表 runner 的 summary/plot、先验打印、前端 spec 导出、token 探针。
**runner 新入口 `--airports`** —— 见 §4。

---

## 4 产物与 runner 的新入口

**KRDU(已产出)**
`4dTrajectory/outputs/KRDU/experiments/two_tier_v3_bprime_20260920/vocabulary_segment_v12/`
train 6,853 架 / 50,825 事件(每架中位 6),val 1,404 架 / 10,309 事件;间隔 p50 34 s / p95 108 s,
**0 撞顶**;用到 heading 72/72、vertical 6/6、speed 16/16、duration 121/151、terminal 2/3;
越界 vertical 750、speed 303。

**五机场(在跑)**
`4dTrajectory/outputs/POOLED/experiments/two_tier_v3_bprime_20260921/vocabulary_five_airports/`
cohort = `KRDU/experiments/plan_guidance_20260910/step5_pooled_cohort/development_cohort.json`
(train 21,911 / val 4,496,KRDU 8,261 + KSJC 6,110 + KSTL 5,952 + KSMF 3,198 + KMSY 2,886)。

**为什么要给 runner 加 `--airports`**:它原本只能用 executor checkpoint 当数据之门,而
**覆盖五机场的 checkpoint 全是旧配置**(缺 `reference_velocity_source`,被 `TSConfig` 按名拒绝),
当前配置下没有五机场执行器。而**读词表根本不需要执行器**——句子是从航迹读的。所以新入口直接按五份
到场清单读,并把**五份清单各自的 SHA-256** 记进产物的出处块,出处仍可追。

**25 架(0.09 %)被排除**:`build_series` 本身跳过短于一个窗口的航迹,那是 ts 数据层的契约,不是这个
runner 加的过滤。它们按 flight key 记进产物的 `excluded_short_track`;**超过 1 % 直接拒绝**(那已是
另一个 cohort)。

> 我在这里连错三次,记下来免得重犯:(1) 抄了 `usable_series`,那是**执行器训练窗口**的过滤,和读词表
> 无关;(2) 加分支时漏了 `source` 块里的 `executor` 引用,读完才在写产物时崩;(3) 把"被数据层跳过"
> 写成"名册对不上"并直接拒绝。

---

## 5 删掉的东西

- `KRDU/.../two_tier_v3_bprime_20260920/vocabulary_tau10/`(53 M)
- 同目录 `_superseded/`(515 M)与 `vocabulary_tau10.log`
- `aeroviz-4d/public/data/airports/KRDU/training/`(4.9 M)

**没碰**:同目录 `approach_events/`(44 K,航迹几何量测,不依赖词表)、其余 118 个实验目录。

---

## 6 改代码时查出来的两个真 bug

**(a) 固定段数会造假指令。** 动态规划被要求固定切 5 段,而一条真实只有 4 段的剖面会把多出来的那段
花在 4 秒的碎片上。**修法**:段数是**上限**不是配额,用**自底向上合并到稳定**——只要还有段短于
`plateau_min_s`(20 s),就把最短的并进角度最接近的邻居,重算该邻居的角度;最后再合并相邻同词。
顺序很重要:一个夹在两个同词段中间的碎片,必须先消掉,那两段才能被看成一条指令(否则句子会因为
"事件重复"被拒)。同词碎片**不记 absorbed**(那是拟合自己的账,不是被丢掉的指令)。

**(b) 执行器爬升上限 2° 飞不出复飞档的 3°。** 已改成 4°,并加了**不变量测试**:词表能说出的每个垂直
档,**连同它的容差带**,都必须落在执行器限幅内(3.0 × 1.07 = 3.21°)。副作用:`CLIMB_MAX_RAD` 也被
规则制导用着,所以规则制导的爬升限幅跟着放宽——它是基线/诊断,不是模型的修法。

---

## 7 一条被实测推翻的设计陈述

设计文档原写"上升 3.0 度这一档在训练数据里出现 0 次"。**读出来它被用到了**——要落进这一档爬升得
超过 1.5°,不是噪声。真正 0 次的是**"结束"词的复飞值**(KRDU 读数 terminal 2/3)。文档已订正。

---

## 7.1 跑道词必须带机场前缀(2026-09-21 改)

五机场产物读出来只有 **16** 个跑道类,不是各机场之和。原因:`flight_runway` 返回的是光跑道号,而
**跑道号跨机场重名**——KSJC 与 KSTL 都有 12L/12R/30L/30R,KSTL 与 KMSY 都有 11/29,六个号码盖住十二
条进近航道朝向完全不同的跑道。合在一起,同一个嵌入要同时代表两条跑道,这个词就说不清自己指的是哪一
条。改法:类名写成 `机场:跑道号`(`KSJC:12L`)。**类别集合是队列的,不是机场的**:五机场到达清单里
有 23 条跑道(KRDU 4 / KSJC 4 / KSTL 8 / KSMF 3 / KMSY 4),pooled 队列里没有一架 KSTL 06,所以读出
来 **22 类**。单机场时前缀只是冗余,不改变任何读数。

**不用重跑词序列**:跑道词是每条航迹的一个常量,别的列跟它无关,所以这是一次**纯索引重映射**——
改 `runway_idents`、每条航迹的 `runway` 标签、词矩阵的跑道列、那条 `runway` 指令的 `word`,再按新
索引重算跑道词频与 `words_used`,重渲 `summary.txt`。两个产物都已就地改完,**并且验证过**:各抽 40
条航迹用当前代码重读一遍,整条 `Reading.to_dict()`(不只是跑道列)与文件逐字节一致。改前的备份在会
话 scratchpad 里。

---

## 7.2 v12 产物作废:句子飞不到跑道(2026-09-21)

用户在前端看出重建的几何航迹和原始航迹差很多、有些转弯方向完全相反(SWA3226、FDX1738)。查下来是
**三件事**,前两件是词表的表达能力,第三件是预览模型:

**(a) 航向词说不出往哪边转。** 词是绝对目标方向,"转到 X" 不含左右;`wrap_deg(180)` 恒为 −180,所以
半圈转弯永远往同一边转,中了就整条航迹镜像出去(最多 18 km)。五机场训练集 28,221 次航向变化里
13.4 % 超过 150°、**9.9 % 正好半圈**,17.1 % 的航班至少中一次。改法:读的时候把超过 `turn_split_deg`
(150°)的转弯拆成两句,中间那句**在转弯开始时**发出、飞机真转到中间航向时再交给下一句。方向本来就在
未展开的航向里,只是绝对目标词把它丢了。

**(b) 航向词说不出"保持中线"——这才是降落不了的原因。** 实测 150 条 KRDU 进近,在 10 km 内首次对正
航道的那一刻:真实航迹离中线 **13 m**,能降落的重建 33 m,降落不了的 **2,464 m**——它对正了航道然后
沿平行线一直飞,`to_go ≤ 0` 永远碰不上"在最后进近上"。词 0 的含义是**航道本身**,飞这个词就是切上并
保持中线(`target_course_deg`,切入角上限 30°)。单这一条:36.7 % → 94.7 %。它不给词表加任何东西,
也不改任何一句话——它是"飞这个词"的含义。

**(c) 预览的机动模型比真机快一半。** 平台到平台真实转弯率是 20° 坡度的 0.67 倍(=14°);|dV/dt| 在真
在变速的样本上 p50 0.19 / p90 0.54 / p99 0.99 m/s²,原来的 1.0 是 p99。改成实测值。单这一条 36.7 %
→ 48.7 %。

**误差的大头不是分档粗。** 把速度词换成读出来的**未量化平台值**,gap p95 中位只从 2035 降到 1669 m
(−18 %);换成**逐秒实测地速**降到 907 m(−55 %)。剩下的距离是"一段一个常速"这个模型和没有风,不是
词表分辨率。

**验收做成了 runner**:`run_ts.py instruction_replay`,飞的是产物文件里写的句子(`Reading.from_dict`),
不是重读航迹——自己重算输入的门看不见文件和代码脱节。

## 8 下一步

1. ~~等五机场产物落地~~ **已落地并已订正跑道词**(见 §7.1):`vocabulary_five_airports/` 22 类、
   `vocabulary_segment_v12/` 4 类。
2. **提交**:词表 + 读法 + 执行器 + 限幅 + 下游 + 测试 + runner 新入口 + 文档订正。**改动大,按规矩要
   先 review 再提交。**
3. **先验训练**。好消息:`PriorConfig.words` 是 `word_counts(vocabulary, runway_vocabulary)` **算出来
   的,不是写死的**,所以嵌入表形状和跑道类数会自动跟着新词表走,这一处不用改。
4. 未做:公布的**复飞最小爬升梯度**还没从程序数据里抽出来,所以 3° 这个取值没有对着规章核过。
