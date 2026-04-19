# 教程 03 — OCS 几何详解（中文版）

**涉及文件：** `src/utils/ocsGeometry.ts` 与其单元测试 `src/utils/__tests__/ocsGeometry.test.ts`

**读者假设：** 你大致会写 TypeScript，但不一定熟悉航空术语或球面几何。本文档会把每一步都拆到"高中三角函数就能理解"的程度。

---

## 0. 一句话说清楚我们在做什么

> **飞机从 FAF（最终进近定位点）降到跑道入口（threshold）这一段，ICAO 规定了一个"保护空域"的立体包络。我们要算出这个包络的 3D 多边形顶点坐标。**

这个包络就叫 **OCS（Obstacle Clearance Surface，障碍物净空面）**。
它由 **3 个面** 组成：

```
          FAF 端（高）                    threshold 端（低）
   ┌──────────────────────────────┐
   │ 次级左       主区       次级右│
   │ 斜面         水平       斜面   │
   └──────────────────────────────┘
```

- **主区（primary）**：一个平的四边形（梯形），位于进近中线左右各 `primaryHalfWidthM` 米。
- **次级左/右（secondary left/right）**：沿主区左右两侧向外延伸 `secondaryWidthM` 米的"斜坡"；外缘高度比主区低 **1/7 × secondaryWidthM**。

我们输入 2 个 3D 点（FAF 与 threshold）+ 2 个宽度参数，输出 3 个多边形共 12 个顶点坐标。

---

## 0.5 OCS 到底是什么？（更详细的定义）

### 先澄清容易混淆的几个概念

很多人第一次接触 PANS-OPS 会把这几个术语搞混：

| 名词                       | 是什么                                                          | 关系                         |
|----------------------------|-----------------------------------------------------------------|------------------------------|
| **保护空域 / 保护区**      | 俯视图（2D），规定"平面上哪些范围要被保护"                      | 这是一个**平面轮廓**         |
| **OCS（障碍物净空面）**    | 一个**3D 斜面**，规定"这个平面轮廓之下，障碍物不能高于多少米"   | 保护区 + 高度 → OCS          |
| **OCH / OCA**              | 单一数字：该进近的最低决断高度/高                               | 由 OCS 与实际障碍物共同决定  |
| **MCA（最低穿越高度）**    | 在某个 fix 上空飞机至少要达到的高度                             | 设计 OCS 时的输入之一        |

**一句话：OCS 是"如果障碍物穿过这个面，飞机就可能撞上"的那张面。**

### OCS 的正式定义（PANS-OPS Vol II, Part I）

> An Obstacle Clearance Surface (OCS) is a surface associated with a segment of
> an instrument procedure, below which obstacles must not penetrate without
> triggering an increase in the minimum altitude for that segment.

翻译：**OCS 是与仪表程序某个航段关联的"曲面"**——该航段的所有障碍物都必须位于该面**之下**；否则，该航段的最低飞行高度就必须提高，直到所有障碍物都被"盖住"为止。

### 几何三要素

一个最终进近 OCS 完全由三件事决定：

1. **起止两点**：FAF 与跑道入口（threshold）。——决定了**中线位置**与**两端的高度锚点**。
2. **主区宽度**：由导航精度（ILS / RNP / VOR 等）决定。——决定了"绝对保护"的横向范围。
3. **7:1 斜率**：PANS-OPS 硬性规定。——决定了次级区从主区外缘到地面的过渡方式。

### 为什么把"一张面"拆成"主区 + 次级区"？

纯工程考虑：

- **飞机位置误差是有概率分布的**（不是非黑即白）。在中线附近概率最高，越远概率越低。
- 主区 = "位置误差 99.7% 都在这里面"的区域 → 用**全保护**的水平面。
- 次级区 = "万一偏得更远的尾部概率"→ 用**递减保护**的斜面（越靠外，允许的障碍物越高）。
- 7:1 是 ICAO 经过长期实践选出的"容错坡度"。

### 在 4D 轨迹可视化里 OCS 用来做什么？

这也是本项目使用 OCS 的主要动机：

1. **合规性判定**：把机场 20 km 内的 FAA DOF 障碍物和 OCS 做布尔运算，找出**穿透 OCS 的障碍物** → 这些就是"程序设计必须加高 OCH 的元凶"。
2. **可视化验证**：在 3D 场景里把 OCS 画出来，对比历史航迹是否在 OCS 之上飞行（低于 OCS = 可能违反保护高度）。
3. **教学演示**：向非航空背景的读者解释"为什么跑道附近不能盖高楼"。

---

## 1. 航空背景：为什么是"主区 + 次级区"？

ICAO Doc 8168（PANS-OPS）规定：在仪表进近最终阶段，为了保证飞机不会撞到地面障碍物，必须在中线左右划出"保护空域"。

| 区域     | 保障程度         | 几何特征                         |
|----------|------------------|----------------------------------|
| 主区     | **100% 净空**    | 水平矩形/梯形，高度 = 程序规定最低高度 |
| 次级区   | **线性递减至 0** | 7:1 斜面，外缘与地面相交         |

### 1.1 什么叫 "7:1 斜率"？

**水平方向每走 7 米，高度允许下降 1 米。**

```
   主区外缘 (高度 H)                   次级外缘 (高度 H - Δ)
   •─────────── 7·Δ 米 ───────────•
   │                               │
   │                               ↓ Δ 米
   │                               │
```

所以如果次级区宽度 = 70 m，则外缘比主区低 70 / 7 = **10 米**。
代码里写成 `faf.altM - secondaryWidthM / 7`。

### 1.2 为什么 threshold 端不下降？

次级斜面的设计意图是"在跑道入口处正好落到地面高度"。因此：

- **FAF 端**：次级外缘高度 = `faf.altM - secondaryWidthM / 7`（最高）
- **threshold 端**：次级外缘高度 = `threshold.altM`（等于跑道入口标高，不再下降）

换句话说，次级面是一个**倾斜的四边形**：FAF 端比主区低一点，threshold 端和主区平齐。

---

## 2. 数学基础：为什么可以"把地球当平面"？

### 2.1 地球其实是球，但在几十公里尺度上可以近似为平面

地球半径 ≈ 6371 km。FAF 到 threshold 的距离一般只有 5–15 km，这个尺度上：

- 地球曲率带来的误差 < **0.1%**
- 用平面几何近似，绝对位置误差 < 20 m（OCS 宽度是 150 m 量级，完全够用）

### 2.2 "度" 与 "米" 的换算

纬度方向（南北）无论你在地球哪里，1° ≈ **111 320 米**（基本是常数）。

经度方向（东西）就不是常数了 —— 越靠近两极，经线越挤：

```
1° 经度的米数 = 111320 × cos(当前纬度)
```

为什么是 cos？想象一下切橙子：赤道那圈最粗，越往两极越细。在纬度 φ 处，"地球切片"的半径是 `R · cos(φ)`，所以单位经度对应的弧长也乘以 cos(φ)。

**例：**
- 赤道（φ=0°）：cos(0)=1 → 1° 经度 ≈ 111 km
- 北纬 60°：cos(60°)=0.5 → 1° 经度 ≈ 55 km
- KRDU（约北纬 35.9°）：cos(35.9°)≈0.81 → 1° 经度 ≈ 90 km

代码里写成：

```ts
const METRES_PER_DEG_LAT = 111_320;
function metresPerDegLon(latDeg: number): number {
  return METRES_PER_DEG_LAT * Math.cos((latDeg * Math.PI) / 180);
}
```

**容易踩的坑：** `Math.cos` 的参数是**弧度**不是度。所以要先 `× π/180` 把度转弧度。

---

## 3. 函数一：`bearingRad` —— 算两点间的方位角

### 3.1 方位角是什么

**方位角（bearing）** = "从 A 点看 B 点在哪个方向"，以**正北为 0，顺时针**为正。

```
       0 (北)
        ↑
−π/2 ←──┼──→ π/2
 (西)   │   (东)
        ↓
     ±π (南)
```

不是从东轴开始量的角度！航海/航空沿用指南针的习惯。

### 3.2 公式

在平面近似下：

```
Δx = (lonB − lonA) × metresPerDegLon(latA)   ← 东向分量（米）
Δy = (latB − latA) × 111320                   ← 北向分量（米）
bearing = atan2(Δx, Δy)
```

**⚠️ 注意参数顺序：** 这里是 `atan2(东, 北)`，不是 `atan2(y, x)`。

因为我们定义正北为 0°。当 B 在 A 正北时 Δx=0、Δy>0，`atan2(0, >0) = 0` ✓。
当 B 在 A 正东时 Δx>0、Δy=0，`atan2(>0, 0) = π/2` ✓。

### 3.3 代码

```ts
export function bearingRad(
  lonA: number, latA: number,
  lonB: number, latB: number
): number {
  const dx = (lonB - lonA) * metresPerDegLon(latA);
  const dy = (latB - latA) * METRES_PER_DEG_LAT;
  return Math.atan2(dx, dy);
}
```

### 3.4 手算验证

FAF 在 (−119.38, 49.95)，threshold 在 (−119.38, 49.90) —— 经度相同、纬度下降，所以 threshold 在 FAF **正南**。

```
Δx = 0 × metresPerDegLon(49.95) = 0
Δy = (49.90 − 49.95) × 111320 = −5566 m
bearing = atan2(0, −5566) = ±π (正南)
```

`atan2` 返回 `(−π, π]`，所以严格讲是 `π`；但实际上 JS 里 `atan2(0, -1)` 返回 `π`，`atan2(-0, -1)` 返回 `-π`，两者都是正南。单元测试里用 `Math.abs(result) ≈ π` 做断言。

---

## 4. 函数二：`offsetPoint` —— 沿某方向移动 N 米

### 4.1 我们要做什么

给定 (lon, lat, altM) 一点，沿方位角 `β` 移动 `d` 米，得到新点。

### 4.2 先把位移拆成东/北分量

```
东向位移 = d · sin(β)    ← 为什么是 sin？因为 β=π/2（正东）时 sin=1
北向位移 = d · cos(β)    ← 为什么是 cos？因为 β=0（正北）时 cos=1
```

这和普通三角函数中 `x = r cos θ, y = r sin θ` 参数顺序相反 —— 原因还是"方位角从正北起算"。

### 4.3 再把米换算回度

```
Δlon = 东向位移 / metresPerDegLon(lat)
Δlat = 北向位移 / 111320
```

### 4.4 完整公式

```ts
newLon = lon + (d · sin(β)) / metresPerDegLon(lat)
newLat = lat + (d · cos(β)) / 111320
newAlt = altM   // 只做水平移动，高度不变
```

### 4.5 验证一下：向东 1000 米

```
β = π/2,  d = 1000,  lat = 49.95
Δlon = 1000 · sin(π/2) / metresPerDegLon(49.95)
     = 1000 · 1 / (111320 · cos(49.95°))
     = 1000 / (111320 · 0.6432)
     = 1000 / 71591
     ≈ 0.01397°
Δlat = 1000 · cos(π/2) / 111320 = 0   ✓（经度变化，纬度不变）
```

---

## 5. 函数三：`buildFinalApproachOCS` —— 组装 3 个多边形

这个函数把前两个函数组合起来，就能产出完整的 OCS 几何。

### 5.1 算法骨架

```
输入：faf, threshold, primaryHalfWidthM (P), secondaryWidthM (S)

1. 算中线方位角 β = bearingRad(faf → threshold)
2. 左右垂直方向：perpLeft = β − π/2,  perpRight = β + π/2
3. 用 offsetPoint 沿 perpLeft/perpRight 移 P 米 —— 得到主区 4 个角
4. 再沿 perpLeft/perpRight 移 (P + S) 米 —— 得到次级外缘 4 个角
   - FAF 端外缘高度 = faf.altM − S / 7
   - threshold 端外缘高度 = threshold.altM
5. 把角按逆时针顺序装进 3 个数组返回
```

### 5.2 为什么 `β − π/2` 是"左"，`β + π/2` 是"右"？

想象你朝中线方向走（从 FAF 看向 threshold），方位角就是 β。那么：

- 你的左手指向 `β − 90°`（顺时针坐标系里"减 90 度"等于逆时针转 90 度）
- 你的右手指向 `β + 90°`

### 5.3 主区 4 个角

```ts
const perpLeft = bearing - Math.PI / 2;
const perpRight = bearing + Math.PI / 2;

const fafLeft  = offsetPoint(faf.lon, faf.lat, faf.altM, perpLeft,  P);
const fafRight = offsetPoint(faf.lon, faf.lat, faf.altM, perpRight, P);
const thrLeft  = offsetPoint(threshold.lon, threshold.lat, threshold.altM, perpLeft,  P);
const thrRight = offsetPoint(threshold.lon, threshold.lat, threshold.altM, perpRight, P);
```

### 5.4 次级外缘 4 个角（含 7:1 斜率）

```ts
const outerAltAtFaf = faf.altM - secondaryWidthM / 7;   // 只在 FAF 端下降
const outerAltAtThr = threshold.altM;                    // threshold 端不变
const outerOffset   = primaryHalfWidthM + secondaryWidthM;

const secFafLeft  = offsetPoint(faf.lon, faf.lat, outerAltAtFaf, perpLeft,  outerOffset);
const secFafRight = offsetPoint(faf.lon, faf.lat, outerAltAtFaf, perpRight, outerOffset);
const secThrLeft  = offsetPoint(threshold.lon, threshold.lat, outerAltAtThr, perpLeft,  outerOffset);
const secThrRight = offsetPoint(threshold.lon, threshold.lat, outerAltAtThr, perpRight, outerOffset);
```

### 5.5 顶点顺序（重要，否则 Cesium 画出来会有自相交）

我们统一走"逆时针"：

```
primaryPolygon  : [fafLeft, fafRight, thrRight, thrLeft]
secondaryLeft   : [fafLeft, secFafLeft, secThrLeft, thrLeft]       ← 从内到外到内
secondaryRight  : [fafRight, secFafRight, secThrRight, thrRight]
```

**关键特性**：`secondaryLeft[0]` 与 `primaryPolygon[0]` 完全相同（都是 `fafLeft`），`secondaryLeft[3]` 与 `primaryPolygon[3]` 相同（都是 `thrLeft`）。这保证两个多边形的**内边完全贴合**，Cesium 渲染时没有缝隙。右侧同理。

### 5.6 `faf.altM` 和 `threshold.altM` 到底是从哪儿来的？

**答：完全来自 fix 点自身的坐标，不做任何二次加工。**

换句话说，主区/次级区的所有高度都**原封不动地继承**自 `procedures.geojson` 里 LineString 顶点的第 3 个分量（z 值）。数据链路如下：

```
FAA CIFP 原始数据 (ARINC 424 纯文本)
      ↓  preprocess_procedures.py
procedures.geojson 的 LineString 坐标 [lon, lat, altM]
      ↓  useOcsLayer 读取
coords[fafIdx] → { lon, lat, altM }            作为 faf
coords[mapIdx] → { lon, lat, altM }            作为 threshold
      ↓  buildFinalApproachOCS 使用
主区 4 顶点 altM  = faf.altM 或 threshold.altM       （原样）
次级外缘 FAF 端   = faf.altM - secondaryWidthM / 7   （仅此处减 1/7·S）
次级外缘 threshold 端 = threshold.altM               （原样）
```

#### 用项目里的实际数据验证

打开 `public/data/procedures.geojson`，找到 KRDU 的 R05LY 程序：

**FAF（WEPAS）：**
```json
"coordinates": [-78.88295556, 35.80876667, 670.56]
"altitudeFt": 2200,  "geometryAltitudeFt": 2200
```
验算：670.56 m = 2200 ft × 0.3048 ✓

**MAPt（RW05L）：**
```json
"coordinates": [-78.80196389, 35.87445, 243.23]
"altitudeFt": 424,  "geometryAltitudeFt": 798
```
验算：243.23 m = **798** ft × 0.3048 ✓（注意不是 424）

所以 LineString 的 z 值等于 `geometryAltitudeFt × 0.3048` —— 即 CIFP 里的**几何高度**。

#### 代码里对应的这一行

`src/hooks/useOcsLayer.ts`:

```ts
function coordToPoint(coord: [number, number, number]): GeoPoint3D {
  return { lon: coord[0], lat: coord[1], altM: coord[2] ?? 0 };
}
```

直接读第三个分量，**既不查表也不换算**。

#### 重要：CIFP 里有两个高度字段，目前用的是哪个？

| 字段                  | 含义                               | 单位 | 目前是否被用 |
|-----------------------|------------------------------------|------|---------------|
| `altitudeFt`          | 程序规定的**最低穿越高度**（MCA）  | ft   | ❌ 没有被用   |
| `geometryAltitudeFt`  | 预处理脚本写进 LineString 的**几何高度** | ft   | ✅ 当前用的是它 |

两者的差异以 RW05L 最典型：`altitudeFt=424` ft（跑道入口的 MCA，表示飞机允许降到这个高度）vs `geometryAltitudeFt=798` ft（几何上 fix 点被画在这个高度）。

#### 这是一个已知的简化，将来可以改

严格按 PANS-OPS 的工程逻辑，OCS 的主区高度应该锚定在 **MCA**（`altitudeFt`），因为 OCS 的语义是"**低于这个高度不能有障碍物**"。目前用几何高度只是因为它直接在 LineString 里，方便接入。

如果要切换到 MCA，只需要改 `useOcsLayer.ts`：

```ts
// 现在（读 LineString z 值）：
const pair = {
  faf:       coordToPoint(coords[fafIdx]),
  threshold: coordToPoint(coords[mapIdx]),
};

// 改成（读 samples 的 MCA）：
const ft2m = (ft: number | null) => (ft ?? 0) * 0.3048;
const pair = {
  faf: {
    lon: coords[fafIdx][0],
    lat: coords[fafIdx][1],
    altM: ft2m(samples[fafIdx].altitudeFt),
  },
  threshold: {
    lon: coords[mapIdx][0],
    lat: coords[mapIdx][1],
    altM: ft2m(samples[mapIdx].altitudeFt),
  },
};
```

其余不用动 —— `buildFinalApproachOCS` 是纯函数，换了输入就自动产出不同高度的 OCS 多边形。

---

## 6. 图解：一个朝正南的进近

参数：
- FAF = (−119.38, **49.95**, 4500 m)
- threshold = (−119.38, **49.90**, 430 m)
- P = 75 m, S = 75 m

因为两点经度相同、纬度下降，β ≈ **−π（正南）**。
`perpLeft = −π − π/2 = −3π/2`（等价于 π/2，即**正东**）
`perpRight = −π + π/2 = −π/2`（即**正西**）

> 直觉检查：朝南飞行时，飞机左手边是东边、右手边是西边。✓

主区 4 个角：

| 角       | lon               | lat    | altM |
|----------|-------------------|--------|------|
| fafLeft  | −119.38 + 0.00105 | 49.95  | 4500 |
| fafRight | −119.38 − 0.00105 | 49.95  | 4500 |
| thrRight | −119.38 − 0.00105 | 49.90  | 430  |
| thrLeft  | −119.38 + 0.00105 | 49.90  | 430  |

(0.00105° ≈ 75 m / metresPerDegLon(49.95°))

次级左外缘：

| 角         | lon               | lat    | altM                |
|------------|-------------------|--------|---------------------|
| secFafLeft | −119.38 + 0.00210 | 49.95  | 4500 − 75/7 ≈ 4489.3 |
| secThrLeft | −119.38 + 0.00210 | 49.90  | **430**（不下降）    |

---

## 7. 单元测试的思路

### 7.1 为什么先测 `bearingRad`？

它有**精确解析解**：

| 情况        | 输入                                 | 预期    |
|-------------|--------------------------------------|---------|
| B 在 A 正北 | 经度相同、纬度增                     | 0       |
| B 在 A 正东 | 纬度相同、经度增                     | π/2     |
| B 在 A 正西 | 纬度相同、经度减                     | −π/2    |
| B 在 A 正南 | 经度相同、纬度减                     | ±π      |

浮点比较用 `toBeCloseTo(expected, 4)`，永远不用 `toBe`。

### 7.2 `offsetPoint` 测什么？

- 向东 1000 米：Δlon 应该等于 `1000 / metresPerDegLon(lat)`，Δlat=0，altM 不变。
- 向北 1000 米：Δlat ≈ 0.008983°，Δlon=0。
- 距离为 0：输入 = 输出。

### 7.3 `buildFinalApproachOCS` 测什么？

1. **多边形顶点数**：主区 4 个，次级左/右各 4 个。
2. **对称性**：主区两边的 lon 与中线 lon 对称分布。
3. **高度**：主区顶点 = FAF 或 threshold 高度（不动）。
4. **外缘高度**：次级外缘在 FAF 端下降了 `secondaryWidthM/7`；在 threshold 端等于 threshold 高度。
5. **贴合性**：次级多边形的内两个顶点 === 主区对应顶点（用 `toEqual` 做深相等比较）。
6. **外在主之外**：次级外缘的经度 | 距中线 | > 主区外缘的经度 | 距中线 |。

---

## 8. 和 Cesium 的接口（概念上）

`ocsGeometry.ts` 本身不引用 Cesium，返回的是纯 JS 数字数组。`useOcsLayer` 这个 React Hook 负责把几何变成 Cesium 实体：

```ts
viewer.entities.add({
  polygon: {
    hierarchy: new Cesium.PolygonHierarchy(
      polygon.map(p => Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.altM))
    ),
    material: Cesium.Color.RED.withAlpha(0.28),
    perPositionHeight: true,   // ← 关键：让每个顶点用自己的高度
    outline: true,
    outlineColor: Cesium.Color.RED.withAlpha(0.9),
  },
});
```

**`perPositionHeight: true` 非常关键**：没它的话 Cesium 会把整个多边形"拍"到一个平均高度上，7:1 斜率就看不出来了。

---

## 9. 容易踩的坑（速查）

| 坑                                              | 现象                                       | 解法                           |
|-------------------------------------------------|--------------------------------------------|--------------------------------|
| `atan2(dy, dx)` 而不是 `atan2(dx, dy)`          | 方位角变成从东起算，左右颠倒               | 牢记 `atan2(东, 北)`           |
| `Math.cos(latDeg)` 忘了转弧度                   | cos 值不对，偏差一个数量级                 | `Math.cos(lat * Math.PI / 180)`|
| 多边形顶点顺序混乱（顺逆时针混用）              | Cesium 画出来是蝴蝶结形                    | 统一"逆时针"                   |
| `perPositionHeight` 忘了设 true                 | 次级斜面看起来是平的                       | Cesium polygon 里显式开启      |
| `secondaryWidthM / 7` 写成 `/ 1/7`              | 斜率反了                                   | 除法，不是倒数                 |
| 用 `toBe` 比较浮点数                            | 测试偶尔挂                                 | 始终用 `toBeCloseTo`           |

---

## 10. 实现完成后的验证清单

- [x] `bearingRad` 4 个方向测试通过
- [x] `offsetPoint` 东/北/零距离测试通过
- [x] `buildFinalApproachOCS` 返回 3 个各 4 角的多边形
- [x] 7:1 斜率测试通过（外缘高度 ≈ `faf.altM − S/7`）
- [x] 内边与主区贴合（`toEqual` 断言）
- [ ] `npm run dev` 后能在 Cesium 场景中看到红色主区 + 橙色次级面
- [ ] 侧面看能观察到次级面 FAF 端比主区低、threshold 端平齐

前 5 项对应 13 个单元测试，已全部通过。后两项需要跑 `npm run dev` 并在浏览器里目视确认。

---

## 附：术语对照表

| 英文缩写    | 中文             | 含义                                   |
|-------------|------------------|----------------------------------------|
| FAF         | 最后进近定位点   | 最后进近段开始处，飞机在此建立下降轨迹 |
| MAPt        | 复飞点           | 未能目视跑道时开始复飞的点             |
| threshold   | 跑道入口         | 可用于着陆的跑道起始处                 |
| OCS         | 障碍物净空面     | 保证无障碍的立体包络                   |
| PANS-OPS    | 空中航行程序     | ICAO Doc 8168，进近程序设计规范        |
| IAF / IF    | 起始/中间进近点  | 进近序列更早的 fix                     |
| RNP         | 所需导航性能     | 定义横向/纵向的导航精度                |
| NM          | 海里             | 1 NM = 1852 m                          |
