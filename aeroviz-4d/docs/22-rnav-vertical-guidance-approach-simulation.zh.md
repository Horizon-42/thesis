# RNAV 垂直导引进近模拟

## 状态

日期：2026-05-04

本文档是 AeroViz-4D 程序可视化的教学性场景说明。它不是运行手册、
飞行训练材料、ATC (Air Traffic Control，空中交通管制) 话术训练，也不
替代飞机手册、运营人程序、航图注记、NOTAM (Notice to Airmen，航行通告)、
天气标准或 ATC 指令。

## 目的

现代 RNAV (Area Navigation，区域导航) / RNP APCH (Required Navigation
Performance Approach，所需导航性能进近) 程序经常在同一张 RNAV (GPS)
航图上发布多条 minima (最低标准)，例如：

- `LPV (Localizer Performance with Vertical Guidance，带垂直导引的航向道性能)`
- `LNAV/VNAV (Lateral Navigation / Vertical Navigation，横向导航/垂直导航)`
- `LP (Localizer Performance，航向道性能)`
- `LNAV (Lateral Navigation，横向导航)`

当飞机、航电、程序、天气、温度限制、导航数据库和运营人批准都支持时，机组
通常更愿意选择带垂直导引的 minima，例如 `LPV` 或 `LNAV/VNAV`。原因是它们
支持更稳定的下降到 DA (Decision Altitude，决断高度)，而不是用纯 `LNAV`
方式下降到 MDA (Minimum Descent Altitude，最低下降高度) 后改平飞或使用
非精密进近下降技术。

ATC 通常放行飞机执行某个 approach procedure (进近程序)。ATC 一般不替机组
选择 minima line (最低标准行)。哪一条 minima 合法、可用、适合当前飞机和运行，
由机组根据航图、航电能力、天气、批准和公司程序决定。

## 角色

### Aircraft (飞机/机组/航电)

这里的 Aircraft 角色包括 flight crew (飞行机组) 和 avionics (航电系统)：

- 从当前 navigation database (导航数据库) 载入 RNAV procedure (RNAV 程序)。
- 确认 runway (跑道)、transition (过渡段)、fixes (定位点)、FAC (Final
  Approach Course，最后进近航迹)、missed approach (复飞程序)、minima、
  altimeter setting (高度表拨正值) 和 temperature restrictions (温度限制)。
- 检查航电 annunciation (方式显示/状态显示) 是否支持 `LPV`、`LNAV/VNAV`、
  `LP`，还是只能支持 `LNAV`。
- 在合适条件下 arm (预位) lateral guidance (横向导引) 和 vertical guidance
  (垂直导引)。
- 飞到 DA/MDA 后，根据目视参考决定 landing (着陆) 或 missed approach (复飞)。

### Approach Control (进近管制)

Approach Control 负责排序、雷达引导、下降高度和进近放行。典型职责包括：

- "Maintain 3000 until established."，保持 3000 ft 直到 established (建立在航迹上)。
- "Cleared RNAV (GPS) runway 05L approach."，放行 RNAV (GPS) RWY 05L 进近。
- 在 final 附近把飞机移交给 Tower (塔台)。

Approach Control 通常不会说：

- "Fly the LPV minima."
- "Use LNAV/VNAV minima."
- "Use LNAV minima."

ATC 给的是 procedure clearance (程序放行)。飞机能用哪条 minima，由飞机资格、
航电状态和机组程序决定。

### Tower (塔台)

Tower 管理 runway use (跑道使用) 和 landing clearance (着陆许可)：

- 确认落地顺序。
- 提供风向风速。
- 发布 landing clearance。
- 如果跑道状态或交通需要，取消或改变许可。

Tower 一般也不选择 `LPV`、`LNAV/VNAV` 或 `LNAV`。Tower 管 runway，不管你
用哪条 minima 作为飞行运行决策。

## 技术区别

### LNAV (Lateral Navigation，横向导航)

在 minima 的语境里，`LNAV` 是 lateral navigation only (只有横向导航)。它的
final OEA (Obstacle Evaluation Area，障碍物评估区) 是横向保护/评估区域。
它能说明障碍物是否落在 lateral area (横向区域) 内，但它本身不是 sloping
vertical clearance surface (倾斜垂直净空面)。

在 AeroViz 中：

- `FINAL_LNAV_OEA` 是 lateral footprint (横向足迹/平面区域)。
- 它不应被显示成 vertical guidance (垂直导引)。
- 它不应被当成 glidepath (下滑路径)。

### LNAV/VNAV (Lateral Navigation / Vertical Navigation，横向导航/垂直导航)

`LNAV/VNAV` 增加了 approved vertical guidance (批准的垂直导引)，常见来源包括
baro-VNAV (Barometric Vertical Navigation，气压垂直导航) 或 WAAS-derived VNAV
(Wide Area Augmentation System derived Vertical Navigation，WAAS 派生垂直导航)，
具体取决于飞机安装和程序。它通常对应 DA (Decision Altitude，决断高度)。

在 AeroViz 中：

- 横向 footprint 可能使用 LNAV OEA 尺寸。
- 垂直 OCS (Obstacle Clearance Surface，障碍物净空面) 或 path-related surface
  必须作为单独对象表达。
- 可视化必须区分 lateral OEA 和 vertical OCS。

### LPV (Localizer Performance with Vertical Guidance，带垂直导引的航向道性能)

`LPV` 使用 SBAS (Satellite-Based Augmentation System，星基增强系统) / WAAS
(Wide Area Augmentation System，广域增强系统) 形式的 lateral and vertical
guidance (横向和垂直导引)。运行体验上类似有 glidepath 的精密进近，可以到 DA，
但它不是 ILS (Instrument Landing System，仪表着陆系统) 的地面 localizer 信号。

在 AeroViz 中：

- 如果目标是研究 certified obstacle surfaces (经认证的障碍物面)，`LPV` 不应被
  简单折叠成 plain LNAV。
- `LPV` final geometry 有自己的 W/X/Y surface 概念，应在有规则和源数据时单独建模。

### LP (Localizer Performance，航向道性能)

`LP` 的 lateral performance (横向性能) 比 LNAV 更精细，但没有 approved vertical
guidance。它通常使用 MDA (Minimum Descent Altitude，最低下降高度)。

在 AeroViz 中：

- `LP` final OEA 不是 `LNAV` final OEA。
- `LP` lateral geometry 按自己的公式和尺寸向 runway threshold (跑道入口) 收缩。

## 场景：同一张 RNAV 航图，不同 minima

下面是一个虚构的、训练风格的叙事。呼号、天气、fix、数值都只是示例。

### 设置

一架 business jet (公务机) 正在进近 Raleigh-Durham。航图是 RNAV (GPS)
RWY 05L。航图发布了：

- `LPV DA`
- `LNAV/VNAV DA`
- `LNAV MDA`

飞机有当前 navigation database，并具备 WAAS-capable avionics (支持 WAAS 的航电)。
天气高于 LPV minima。运营人批准 LPV approaches。

### Cockpit Briefing (驾驶舱简令)

Lin 机长看着 approach page。

"RNAV GPS runway zero-five-left loaded. Final approach fix DUHAM, runway
threshold RW05L, missed approach climbs straight ahead then to the hold."

Chen 副驾驶沿着显示器上的 magenta line (洋红色航迹线) 检查。

"Minima available: LPV, LNAV/VNAV, LNAV. WAAS is available, approach
annunciation currently shows LPV armed. We'll brief LPV. If we lose LPV but keep
LNAV/VNAV, we can continue only if we're still before the required point and the
annunciation and minima remain valid. Otherwise we revert to LNAV or go missed
according to company procedure."

机长点头。

"Set LPV DA. Cross-check altimeter, temperature note, missed approach altitude.
The important part for us: lateral course is the same charted final, but the
vertical decision changes. We don't invent a glidepath for LNAV."

技术旁白：

- 同一条 final course 不等于同一套 vertical minima。
- `LNAV` 可以有 advisory vertical path (建议垂直路径)，但这不等同于 approved
  vertical guidance。
- 对 AeroViz 来说，lateral OEA 和 vertical OCS 必须分开表达。

### Approach Control (进近管制)

进近管制呼叫：

> "AeroViz 452, descend and maintain three thousand. Proceed direct DUHAM."

副驾驶回答：

> "Descend and maintain three thousand, direct DUHAM, AeroViz 452."

驾驶舱内，FMS (Flight Management System，飞行管理系统) 画出转向 final approach
fix 的航迹。机组再次检查 approach mode。

"LPV still armed," Chen 说。

Lin 回答：

"Good. If this were only LNAV, we would treat the vertical path differently.
Here the box is giving approved vertical guidance."

进近管制再次呼叫：

> "AeroViz 452, maintain three thousand until established on the final approach
> course, cleared RNAV GPS runway zero-five-left approach."

Chen 复诵：

> "Maintain three thousand until established, cleared RNAV GPS runway
> zero-five-left approach, AeroViz 452."

注意 ATC 没有说：

- "Cleared LPV."
- "Use LNAV/VNAV."
- "Use LNAV minima."

ATC 放行的是 RNAV approach。机组根据航图、飞机能力、航电 annunciation、天气和
运营人规则选择 LPV minima。

### Intercept (截获 final course)

飞机接近 final course。lateral guidance 先截获。

Chen callout：

"Final approach course alive. LPV still annunciated. Glidepath armed."

Lin 看着 vertical path indicator (垂直路径指示) 下移到中心。

"Established. Continue."

AeroViz 可视化在这里必须避免混淆两个概念：

- lateral protected area (横向保护区) 是围绕 final course 的平面区域。
- vertical guidance path 或 OCS 是独立的垂直对象。

飞机不是在飞彩色 OEA polygon。飞机飞的是 coded path (编码路径) 和 approved
guidance (批准导引)。OEA 是 procedure design 和 obstacle evaluation 的几何对象。

### Final Approach (最后进近)

到 FAF (Final Approach Fix，最后进近定位点)：

"FAF," Chen callout. "Glidepath captured. Landing checklist complete."

Lin 保持稳定下降。autopilot (自动驾驶) 跟随 lateral and vertical guidance。
机组监控 speed (速度)、descent rate (下降率)、course deviation (航迹偏差) 和
vertical deviation (垂直偏差)。

Chen 说：

"This is why we prefer LPV when it is available. We get a managed descent to DA
instead of descending to an MDA and leveling."

Lin 回答：

"Exactly. But the preference is conditional. If the avionics downgrade, we don't
pretend LPV still exists. The annunciation drives what minima we can use."

### Tower (塔台)

Approach 移交 Tower：

> "AeroViz 452, contact tower one two zero point seven."

Chen 回复：

> "Tower one two zero point seven, AeroViz 452."

随后：

> "Raleigh Tower, AeroViz 452, RNAV GPS zero-five-left, five miles final."

Tower 回答：

> "AeroViz 452, Raleigh Tower, wind zero six zero at eight, runway zero-five-left,
> cleared to land."

Chen 复诵：

> "Cleared to land runway zero-five-left, AeroViz 452."

再一次，Tower clear runway (放行跑道)。Tower 没有选择 LPV 或 LNAV/VNAV。
机组继续使用已经选择且航电显示有效的 minima。

### DA (Decision Altitude，决断高度)

接近 DA 时，Chen callout：

"Approaching minimums."

到 DA：

"Minimums."

Lin 看见 runway environment (跑道环境)：

"Landing."

如果没有看到必要目视参考，回答会是：

"Go around."

然后飞机执行 published missed approach (公布复飞程序)，而不是临时继续 final descent。

## 如果 LPV 不可用怎么办？

同一条 approach 可能变成另一种运行方式。

在 FAF 之前，航电 annunciation 从 `LPV` 降级为 `LNAV/VNAV`。

Chen 说：

"Downgrade. LPV no longer available. LNAV/VNAV available."

Lin 回答：

"Check minima. Weather still above LNAV/VNAV DA. We can continue with
LNAV/VNAV if all restrictions are satisfied."

如果航电再次降级到 `LNAV`，机组可能需要：

- 使用 LNAV MDA，前提是合法、已简令、天气满足、公司程序允许。
- 改变 vertical mode 和下降技术。
- 如果降级太晚或公司程序要求，执行 missed approach。

核心点：航图可以发布多条 minima，但当前使用哪条 minima 是 crew and avionics
decision (机组和航电决策)，不是 Tower decision (塔台决策)。

## 对 AeroViz 的可视化含义

这个模拟对 AeroViz 的设计意味着：

1. `FINAL_LNAV_OEA` 必须显示为 lateral-only (仅横向)。
2. 当存在 GPA (Glide Path Angle，下滑路径角) / TCH (Threshold Crossing Height，
   跑道入口通过高度) 数据时，`LNAV/VNAV OCS` 必须显示为独立 vertical surface。
3. `LPV` / `GLS (GBAS Landing System，地基增强系统着陆系统)` 的 W/X/Y surfaces
   应在实现后单独显示。
4. 不能让平面 OEA polygon 暗示它具有 vertical guidance。
5. annotation (标注) 必须说明对象类型：
   - lateral OEA；
   - vertical OCS；
   - display aid；
   - debug estimate；
   - missing source。

## 简短结论

机组通常更愿意使用 `LPV` 或 `LNAV/VNAV`，因为 vertical guidance 支持更稳定地下降
到 DA。ATC 负责 approach clearance 和 landing clearance；ATC 通常不替机组选择
minima。飞机、机组、航电、公布航图、天气和运营人批准共同决定当前能使用
`LPV`、`LNAV/VNAV`、`LP` 还是 `LNAV`。
