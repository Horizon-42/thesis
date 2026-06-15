# 基于机型的 \(C_L(\alpha)\) 经验建模方案

本文整理一个适合当前飞行动力学模拟器使用的方案：

```text
飞机注册号 / ICAO24
→ 飞机机型 typecode
→ 机翼几何与性能参数
→ 经验 / 半经验公式
→ 生成 C_L(alpha) 曲线
```

目标不是获得某一架真实飞机的厂商级气动数据库，而是构建一个：

- 可自动化；
- 可追溯；
- 覆盖多类固定翼飞机；
- 适合 3-DOF / point-mass 仿真的升力模型。

---

# 1. 结论

最可行的工程路线是：

\[
\boxed{
\text{OpenSky} \rightarrow \text{OpenAP} \rightarrow \text{DATCOM/Raymer 风格经验公式}
}
\]

具体来说：

1. 使用 OpenSky aircraft metadata 把注册号或 ICAO24 解析成飞机型号，例如 `A320`、`B738`、`E190`。
2. 使用 OpenAP 按 ICAO typecode 获取机翼面积、翼展、平均气动弦、后掠角、质量范围等参数。
3. 使用 DATCOM / Raymer / Roskam 风格的有限翼经验公式估计 \(C_{L_\alpha}\)。
4. 使用飞机类别先验值补充 \(C_{L0}\)、\(C_{Lmax}\)、失速迎角等缺失参数。
5. 输出一条标明来源和置信度的 \(C_L(\alpha)\) 曲线。

推荐在论文中使用的措辞：

> The lift model follows a DATCOM/Raymer-style semi-empirical finite-wing formulation. Aircraft identity is resolved from OpenSky metadata, geometric and performance parameters are obtained from OpenAP where available, and missing aerodynamic constants are assigned from aircraft-class priors.

---

# 2. 为什么不能直接查真实 \(C_L(\alpha)\)

OpenSky 和 OpenAP 都不能直接给出某架飞机的真实 \(C_L(\alpha)\) 曲线。

## 2.1 OpenSky 提供什么

OpenSky aircraft metadata 主要包含：

```text
icao24
registration
manufacturericao
manufacturername
model
typecode
serialnumber
operator
owner
engines
categoryDescription
```

它可以解决：

\[
\text{registration / icao24} \rightarrow \text{typecode}
\]

但它不包含：

```text
CL0
CL_alpha
CLmax
alpha_stall
wing aerodynamic polar
flight dynamics derivatives
```

## 2.2 OpenAP 提供什么

OpenAP 可以按机型提供常用性能参数。例如 A320 包含：

```text
wing.area
wing.span
wing.mac
wing.sweep
drag.cd0
drag.k
drag.e
limits.MTOW
limits.OEW
limits.MLW
```

这些参数足够支持经验公式建模，但 OpenAP 也不直接提供：

\[
\alpha \rightarrow C_L
\]

## 2.3 A320 这类客机为什么没有完整公开气动表

商用客机的全机气动数据库通常属于厂商和认证数据，公开资料中一般不会出现类似 C172 教学数据那样完整的：

```text
CL(alpha, Mach, flap)
CD(alpha, Mach, flap)
Cm(alpha, Mach, elevator)
CLq, Cmq, CL_delta_e, Cm_delta_e
```

因此，对于 A320/B738/E190 这类高频机型，比较稳妥的做法是：

```text
真实机型识别
→ 几何参数公开来源
→ 半经验气动模型
→ 可选的离线校准
```

---

# 3. 推荐建模层级

## 3.1 Level 0: 直接使用类别默认曲线

当只知道飞机大类，但缺少机翼几何时：

\[
C_L = C_{L0,class} + C_{L_\alpha,class}\alpha
\]

例如：

```text
transport_jet:
  CL0_clean ≈ 0.20
  CLalpha ≈ 4.8 - 5.5 per rad
  CLmax_clean ≈ 1.3 - 1.6
```

优点：

- 实现最快；
- 数据缺失时仍然能运行。

缺点：

- 机型差异很粗；
- 不适合严肃比较 A320 与 B738 的气动差异。

## 3.2 Level 1: OpenAP 几何 + 有限翼经验公式

这是推荐的第一版正式方案。

从 OpenAP 获取：

| 参数 | 含义 |
|---|---|
| \(S\) | 机翼参考面积 |
| \(b\) | 翼展 |
| \(\bar{c}\) | 平均气动弦 |
| \(\Lambda\) | 后掠角 |
| \(e\) | Oswald / span efficiency 相关参数 |

然后计算：

\[
AR = \frac{b^2}{S}
\]

再用经验公式估计 \(C_{L_\alpha}\)。

优点：

- 每个机型都有不同几何；
- 对 A320/B738/B77W/C172 等都可以统一处理；
- 足够支撑 3-DOF 飞行动力学仿真。

缺点：

- \(C_{L0}\)、\(C_{Lmax}\)、失速区仍需要类别默认值或校准。

## 3.3 Level 2: DATCOM / Digital DATCOM

如果希望更接近传统工程估算方法，可以使用 USAF Stability and Control DATCOM 或 Digital DATCOM。

Digital DATCOM 可以基于飞机几何和飞行条件输出：

```text
CL
CD
Cm
CL_alpha
Cm_alpha
CY_beta
Cn_beta
Cl_beta
```

它本质上是一套系统化的经验 / 半经验气动估算方法，适合概念设计和早期仿真建模。

优点：

- 权威性强；
- 可覆盖机翼、机身、尾翼、襟翼、控制面、动态导数；
- 论文中好解释。

缺点：

- 输入几何比 OpenAP 多很多；
- 对现代客机复杂构型仍是近似；
- 自动化工作量更大。

## 3.4 Level 3: OpenVSP / AVL / CFD 离线校准

对高频机型，例如：

```text
A320
B738
E190
B77W
C172
```

可以离线建立简化几何，然后扫迎角：

```text
alpha = -6° ... 14°
→ CL, CD, Cm
→ 拟合 CL0, CL_alpha, alpha_stall, CLmax
```

这适合生成本地 `aero_curves.json`，作为 Level 1 经验公式的校准版本。

---

# 4. 核心公式

## 4.1 小迎角线性模型

常规固定翼在小迎角、未失速范围内：

\[
\boxed{
C_L(\alpha) = C_{L0} + C_{L_\alpha}\alpha
}
\]

也可以写成：

\[
\boxed{
C_L(\alpha) = C_{L_\alpha}(\alpha-\alpha_{0L})
}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(C_L\) | 升力系数 |
| \(C_{L0}\) | 零迎角升力系数 |
| \(C_{L_\alpha}\) | 升力曲线斜率，单位通常为 `per rad` |
| \(\alpha\) | 迎角，单位用弧度 |
| \(\alpha_{0L}\) | 零升力迎角 |

注意：

\[
\alpha_{0L} = -\frac{C_{L0}}{C_{L_\alpha}}
\]

## 4.2 DATCOM / Raymer 风格有限翼升力斜率

对亚音速固定翼，常用：

\[
\boxed{
C_{L_\alpha}
=
\frac{2\pi AR}
{2+\sqrt{
4+
\left(\frac{AR\beta}{\eta}\right)^2
\left(1+\frac{\tan^2 \Lambda_{c/2}}{\beta^2}\right)
}}
}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(AR\) | 展弦比，\(AR=b^2/S\) |
| \(b\) | 翼展 |
| \(S\) | 机翼参考面积 |
| \(M\) | 马赫数 |
| \(\beta\) | 亚音速压缩性修正，\(\beta=\sqrt{1-M^2}\) |
| \(\eta\) | 翼型效率因子，常取 \(0.9\sim1.0\) |
| \(\Lambda_{c/2}\) | 半弦线后掠角 |

这个公式的直觉：

- 展弦比越大，\(C_{L_\alpha}\) 越接近二维翼型的 \(2\pi\)；
- 后掠角越大，升力斜率通常越小；
- Mach 数越高，压缩性影响越明显；
- 低展弦比飞机的升力斜率更低。

## 4.3 低速简化版

如果不考虑 Mach 和后掠，可以用：

\[
\boxed{
C_{L_\alpha}
=
\frac{a_0}
{1+\frac{a_0}{\pi e AR}}
}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(a_0\) | 二维翼型升力斜率，常取 \(2\pi\ \text{per rad}\) |
| \(e\) | Oswald / span efficiency，常取 \(0.75\sim0.9\) |
| \(AR\) | 展弦比 |

这条公式更适合：

- C172 这类低速通航飞机；
- 初版仿真；
- 数据缺失时的 fallback。

---

# 5. 失速限制

线性模型不能无限延伸。实际飞机达到失速迎角后，\(C_L\) 不会继续线性增加。

最简单的工程实现是：

\[
\boxed{
C_L = clamp(C_{L0}+C_{L_\alpha}\alpha,\ C_{Lmin},\ C_{Lmax})
}
\]

其中：

| 参数 | 含义 |
|---|---|
| \(C_{Lmin}\) | 最小升力系数 |
| \(C_{Lmax}\) | 最大升力系数 |

更平滑的实现可以用 soft stall 模型：

```python
def cl_with_soft_stall(alpha_rad, cl0, cla, clmax, alpha_stall_rad):
    cl_linear = cl0 + cla * alpha_rad
    if alpha_rad <= alpha_stall_rad:
        return min(cl_linear, clmax)

    excess = alpha_rad - alpha_stall_rad
    return clmax - 0.6 * excess
```

第一版建议先用 `clamp`，因为更稳定，也更容易解释。

---

# 6. 不同飞机类型的默认参数

下面的值不是厂商数据，而是用于经验模型的类别先验。它们应该在输出中标注为：

```text
source = aircraft_class_prior
confidence = low_to_medium
```

| 飞机类型 | \(e\) | \(\eta\) | \(C_{L0,clean}\) | \(C_{Lmax,clean}\) | \(C_{Lmax,landing}\) | 说明 |
|---|---:|---:|---:|---:|---:|---|
| light_ga | 0.80-0.90 | 0.95-1.00 | 0.20-0.40 | 1.4-1.7 | 1.8-2.3 | C172、PA28 等低速通航飞机 |
| turboprop | 0.78-0.88 | 0.95-1.00 | 0.20-0.35 | 1.4-1.7 | 2.0-2.6 | Dash 8、ATR 等 |
| regional_jet | 0.75-0.85 | 0.90-1.00 | 0.15-0.30 | 1.3-1.6 | 2.1-2.6 | E170/E190/CRJ |
| transport_jet | 0.75-0.85 | 0.90-1.00 | 0.10-0.30 | 1.3-1.6 | 2.2-2.8 | A320/B738/B77W/B789 |
| business_jet | 0.70-0.85 | 0.90-1.00 | 0.10-0.25 | 1.3-1.5 | 1.9-2.4 | C550/Gulfstream 等 |
| glider | 0.85-0.95 | 0.95-1.00 | 0.20-0.40 | 1.3-1.7 | 1.3-1.7 | 高展弦比，通常无复杂高升力装置 |
| fighter_delta | 0.50-0.75 | 0.80-0.95 | 0.00-0.15 | 1.0-1.4 | 1.0-1.4 | 高迎角需额外 vortex lift 模型 |

不建议把下面类型强行塞进同一个 \(C_L(\alpha)\) 模型：

| 类型 | 原因 |
|---|---|
| helicopter | 主升力来自旋翼，需要 blade element / momentum theory |
| multicopter | 升力来自多个旋翼推力，不是固定翼升力曲线 |
| eVTOL transition aircraft | 需要分飞行阶段建模 |
| missile / rocket | 更适合 Missile DATCOM 或专门弹体气动模型 |

---

# 7. 参数获取方式

| 参数 | 推荐来源 | 备注 |
|---|---|---|
| `registration` | 输入数据 | 例如 `N123AB` |
| `icao24` | OpenSky / ADS-B | 可直接关联 OpenSky metadata |
| `typecode` | OpenSky aircraft metadata | 例如 `A320`、`B738` |
| \(S\) | OpenAP `wing.area` | 缺失时查公开规格表 |
| \(b\) | OpenAP `wing.span` | 用于计算 \(AR\) |
| \(\bar{c}\) | OpenAP `wing.mac` | 缺失时可近似 \(S/b\) |
| \(\Lambda\) | OpenAP `wing.sweep` | 若只有四分之一弦后掠，需在文档中说明 |
| \(e\) | OpenAP `drag.e` 或类别默认值 | 不是所有机型都有可靠值 |
| \(M\) | 当前飞行状态 | 用 \(M=V/a\) 计算 |
| \(C_{L0}\) | 类别默认值 / 本地校准 | OpenAP 不直接提供 |
| \(C_{Lmax}\) | 类别默认值 / 手册失速速度反推 | 构型相关 |
| \(\alpha_{stall}\) | 类别默认值 / 校准 | clean 常在约 10-15 deg |

---

# 8. 从失速速度反推 \(C_{Lmax}\)

如果可以获得某机型的失速速度 \(V_s\)，可以反推：

\[
\boxed{
C_{Lmax}
=
\frac{2W}{\rho V_s^2 S}
}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(W\) | 重量，\(W=mg\) |
| \(\rho\) | 空气密度 |
| \(V_s\) | 失速速度，必须确认是 CAS、EAS 还是 TAS |
| \(S\) | 参考面积 |

注意：

- 失速速度通常和构型有关，例如 clean、takeoff flap、landing flap；
- 必须知道测试重量；
- 如果使用 EAS，可近似使用海平面标准密度；
- 这比直接猜 \(C_{Lmax}\) 更有依据。

---

# 9. 推荐数据结构

本地可以维护一个 `aero_curves.json`：

```json
{
  "A320": {
    "aircraft_class": "transport_jet",
    "model": "datcom_raymer_finite_wing",
    "config": "clean",
    "cl0": 0.20,
    "cla_per_rad": 5.05,
    "cl_min": -0.80,
    "cl_max": 1.45,
    "alpha_valid_deg": [-6.0, 12.0],
    "geometry_source": "OpenAP",
    "aero_source": "OpenAP geometry + Raymer/DATCOM-style finite-wing formula + class prior",
    "confidence": "medium"
  }
}
```

如果某机型经过 OpenVSP / AVL / DATCOM 离线校准，可以改成：

```json
{
  "A320": {
    "aircraft_class": "transport_jet",
    "model": "calibrated_lookup",
    "config": "clean",
    "alpha_deg": [-6, -4, -2, 0, 2, 4, 6, 8, 10, 12],
    "cl": [-0.33, -0.15, 0.03, 0.20, 0.38, 0.56, 0.74, 0.92, 1.10, 1.28],
    "geometry_source": "OpenAP + simplified geometry",
    "aero_source": "OpenVSP/VSPAERO offline sweep",
    "confidence": "medium_to_high"
  }
}
```

---

# 10. 实现草图

```python
import math


CLASS_PRIORS = {
    "transport_jet": {
        "eta": 0.95,
        "e": 0.80,
        "cl0_clean": 0.20,
        "cl_min_clean": -0.80,
        "clmax_clean": 1.45,
    },
    "light_ga": {
        "eta": 0.98,
        "e": 0.85,
        "cl0_clean": 0.30,
        "cl_min_clean": -0.80,
        "clmax_clean": 1.55,
    },
}


def finite_wing_cl_alpha(
    wing_area_m2: float,
    wing_span_m: float,
    sweep_half_chord_deg: float,
    mach: float,
    eta: float = 0.95,
) -> float:
    """Return CL_alpha in per-radian units."""
    ar = wing_span_m**2 / wing_area_m2
    beta = math.sqrt(max(1.0 - mach**2, 1e-6))
    sweep = math.radians(sweep_half_chord_deg)

    denominator = 2.0 + math.sqrt(
        4.0
        + (ar * beta / eta) ** 2
        * (1.0 + (math.tan(sweep) ** 2) / (beta**2))
    )

    return 2.0 * math.pi * ar / denominator


def cl_alpha_curve(alpha_rad: float, cl0: float, cla: float, cl_min: float, cl_max: float) -> float:
    cl = cl0 + cla * alpha_rad
    return max(cl_min, min(cl, cl_max))
```

完整流程：

```text
1. resolve_aircraft_identity(registration_or_icao24)
   → typecode

2. load_openap_aircraft(typecode)
   → S, b, mac, sweep, e, MTOW, OEW

3. classify_aircraft(typecode, openap_data)
   → transport_jet / light_ga / regional_jet / ...

4. compute_cl_alpha(S, b, sweep, mach, eta)
   → cla_per_rad

5. choose_priors(aircraft_class)
   → cl0, cl_min, clmax

6. build_curve(alpha_grid)
   → [(alpha_deg, cl), ...]
```

---

# 11. A320 示例

OpenAP 中 A320 的典型参数包括：

```text
S ≈ 124 m²
b ≈ 35.8 m
MAC ≈ 4.19 m
sweep ≈ 25 deg
e ≈ 0.799
MTOW ≈ 78000 kg
OEW ≈ 42600 kg
```

计算展弦比：

\[
AR = \frac{35.8^2}{124} \approx 10.3
\]

使用 transport_jet 类别默认值：

```text
CL0_clean = 0.20
CLmax_clean = 1.45
eta = 0.95
```

得到一条近似：

\[
C_L(\alpha)
=
0.20 + C_{L_\alpha}\alpha
\]

这条曲线应标注为：

```text
source = OpenAP geometry + DATCOM/Raymer-style finite-wing formula
confidence = medium
validity = small angle, clean configuration, pre-stall
```

---

# 12. 局限性

这个经验模型不适合解释以下现象：

- 真实失速与分离流；
- 大迎角非线性升力；
- 襟翼/缝翼复杂构型；
- Mach 接近 1 的跨音速激波效应；
- 具体飞机序列号、改装、老化、污染造成的气动差异；
- 飞控系统对姿态和迎角的限制；
- 旋翼机或多旋翼飞行器。

对当前 3-DOF / point-mass 模型来说，它适合回答：

```text
给定机型、速度、高度、迎角，估计升力
```

不适合声称：

```text
这是某架 A320 的真实认证气动数据库
```

---

# 13. 推荐在代码中的置信度标注

每条曲线建议带上：

| 字段 | 示例 |
|---|---|
| `geometry_source` | `OpenAP` |
| `aero_source` | `Raymer/DATCOM-style finite-wing formula` |
| `config` | `clean` |
| `confidence` | `low`, `medium`, `medium_to_high` |
| `valid_alpha_deg` | `[-6, 12]` |
| `valid_mach` | `[0.0, 0.85]` |
| `notes` | `pre-stall linear approximation` |

这样后续即使混用真实数据、经验公式、OpenVSP 校准结果，也能保持来源透明。

---

# 14. References

1. Hoak, D. E., Ellison, D. E., et al. *USAF Stability and Control DATCOM*. Air Force Wright Aeronautical Laboratories, revised 1978. This is the main reference family for semi-empirical stability and control derivative estimation.
2. Williams, J. E., and Vukelich, S. R. *The USAF Stability and Control Digital DATCOM, Volume I: Users Manual*. AFFDL-TR-79-3032, 1979.
3. Williams, J. E., and Vukelich, S. R. *The USAF Stability and Control Digital DATCOM, Volume II: Implementation of DATCOM Methods*. AFFDL-TR-79-3032, 1979.
4. Raymer, Daniel P. *Aircraft Design: A Conceptual Approach*. AIAA Education Series. Used here as the practical conceptual-design reference for finite-wing lift-curve-slope estimation.
5. Roskam, Jan. *Airplane Design*. Roskam Aviation and Engineering Corporation. Useful for component build-up, stability derivatives, and class-based aircraft design estimates.
6. OpenAP Handbook, "Aircraft and engines". <https://openap.dev/aircraft_engine.html>
7. OpenAP API Reference, `prop` module. <https://openap.dev/api/prop.html>
8. OpenSky Network aircraft metadata CSV. <https://opensky-network.org/datasets/metadata/aircraftDatabase.csv>
9. NASA Common Research Model. <https://commonresearchmodel.larc.nasa.gov/>
10. NASA CRM experimental results search. <https://commonresearchmodel.larc.nasa.gov/experiment-results-search/>
11. OpenVSP official documentation and overview. <https://openvsp.org/learn.shtml>
