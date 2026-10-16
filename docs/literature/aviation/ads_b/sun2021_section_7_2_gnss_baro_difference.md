# Sun (2021), §7.2 — GNSS and barometric altitudes difference / GNSS 高度与气压高度之差

Source / 来源: J. Sun, *The 1090 Megahertz Riddle: A Guide to Decoding Mode S and ADS-B Signals*, 2nd ed., TU Delft OPEN, 2021, §7.2, pp. 71–72 (`papers/Sun2021_1090_Megahertz_Riddle_2nd_ed.pdf`).
Date / 日期: 2026-10-16.

Everything in the two summary parts is from the book. Part 3 is my own reading, marked as such. / 前两部分来自书中,第三部分是我自己的读法,已标明。

## 1. Where the field sits / 这个字段在哪里

| English | 中文 |
|---|---|
| Chapter 7 decodes the **airborne velocity message** (Type Code 19). Besides horizontal and vertical velocity, it carries the navigation uncertainty category for velocity and **the difference between the GNSS height and the barometric altitude** (p. 70). | 第 7 章解码**空中速度报文**(类型码 19)。除水平、垂直速度外,它还带有速度的导航不确定度类别,以及 **GNSS 高度与气压高度之差**(第 70 页)。 |
| §7.2 describes the last eight bits of the ME field. | §7.2 讲的是 ME 字段的最后 8 位。 |

## 2. What §7.2 says / §7.2 的内容

| English | 中文 |
|---|---|
| The last eight bits of the ME field contain the difference between the barometric and GNSS altitudes. | ME 字段的最后 8 位给出气压高度与 GNSS 高度之差。 |
| The first bit is the **sign**; the remaining bits are the value. A sign bit of `1` means the value is **negative**, i.e. the GNSS altitude is **below** the barometric altitude. | 第 1 位是**符号位**,其余位是数值。符号位为 `1` 表示**负值**,即 GNSS 高度**低于**气压高度。 |
| The value is encoded in **25 ft** increments. | 数值以 **25 ft** 为步长编码。 |
| All bits zero or all bits one: the information is **not available**. | 全 0 或全 1:该信息**不可用**。 |

Formula (7.4) / 公式 (7.4):

$$\Delta h = s\,(n-1)\times 25 \quad[\mathrm{ft}]$$

| English | 中文 |
|---|---|
| $s$ is the sign (+1 or −1) and $n$ is the decimal value of the encoded difference. | $s$ 是符号(+1 或 −1),$n$ 是编码值的十进制数。 |
| Example, message A (sub-type 1): sign bit 0, $n=23$, so $\Delta h = 1\times(23-1)\times 25 = 550$ ft (eq. 7.5). | 例,报文 A(子类型 1):符号位 0,$n=23$,所以 $\Delta h = 1\times(23-1)\times 25 = 550$ ft(式 7.5)。 |
| Example, message B (sub-type 3): all bits zero, so the difference is not available. | 例,报文 B(子类型 3):全 0,所以该差值不可用。 |

## 3. What this section does not say (my reading) / 本节没有说的内容(我的读法)

| English | 中文 |
|---|---|
| **The direction of the subtraction is implied, not written.** The book names the field "the difference between the barometric and GNSS altitudes" in one sentence and "between GNSS and barometric" in the example. Only the sign rule is explicit: negative means GNSS is below barometric. Read together, $\Delta h$ is GNSS minus barometric. | **相减方向是隐含的,没有明写。** 书中一处说"气压与 GNSS 之差",例子里又说"GNSS 与气压之差"。只有符号规则是明确的:负值表示 GNSS 低于气压。合起来读,$\Delta h$ 是 GNSS 减气压。 |
| **No reference surface is named.** A search of the whole book finds no "ellipsoid", "HAE", "geoid" or "mean sea level". §5.5.2 says the 12-bit GNSS height of position messages (Type Code 20–22) is in metres, without saying above what. | **没有说明基准面。** 全书检索不到 "ellipsoid"、"HAE"、"geoid"、"mean sea level"。§5.5.2 只说位置报文(类型码 20–22)的 12 位 GNSS 高度单位是米,没有说是相对什么面。 |
| So §7.2 **cannot be cited** for the project's vertical-datum rule (observed ADS-B altitude is ellipsoidal, MSL is lower by about 33 m over the US). It only supports: an ADS-B stream can carry a GNSS-minus-barometric offset, in 25 ft steps. | 因此 §7.2 **不能用来支撑**项目的垂直基准规则(观测 ADS-B 高度是椭球高,美国境内 MSL 约低 33 m)。它只能支撑:ADS-B 流可以携带 GNSS 减气压的偏差,步长 25 ft。 |
| Do not mix it up with the geoid undulation. The field compares GNSS height with **barometric (pressure) altitude**, which changes with the weather; it is not the fixed ellipsoid-to-MSL offset. | 不要与大地水准面差距混淆。该字段比较的是 GNSS 高度和**气压(压力)高度**,后者随天气变化,不是椭球面到 MSL 的固定差值。 |
| OpenSky's history tables may or may not keep this field. I did not check; look at the columns before relying on it. | OpenSky 历史表是否保留这个字段,我没有检查,使用前请先看列名。 |

## Neighbouring text / 相邻内容

§7.1 (p. 71) decodes the vertical rate: bit VrSrc says whether the measurement source is GNSS (0) or barometric (1), and $VS = \pm 64\,(VR-1)$ ft/min, with the sign from bit Svr (0 climb, 1 descent). Not part of §7.2.

§7.1(第 71 页)解码垂直速率:VrSrc 位表示测量来源,0 为 GNSS,1 为气压;$VS = \pm 64\,(VR-1)$ ft/min,符号由 Svr 位给出(0 爬升,1 下降)。不属于 §7.2。
