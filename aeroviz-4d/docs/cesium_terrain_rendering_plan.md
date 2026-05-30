# CesiumJS 局部地形 3D 感增强渲染方案

## 1. 目标

本文档用于整理一套在 CesiumJS（Cesium JavaScript）中增强局部 terrain（地形）三维表现的推荐方案。目标包括：

- 让山体、沟谷、坡面具有更明显的明暗层次；
- 在保证视觉美观的同时，避免画面过暗、过曝或发灰；
- 尽量使用 CesiumJS 原生能力，必要时叠加制图型 hillshade（山体阴影渲染）增强层；
- 兼顾性能、可维护性和展示效果。

核心结论：

> 最优方案不是单纯调太阳光，而是采用 **高质量 terrain + 顶点法线 + 固定艺术方向光 + 轻微垂直夸张 + 局部 hillshade 贴图叠加** 的组合方案。

---

## 2. 为什么单纯使用全局太阳光效果不明显

CesiumJS 的默认场景光照主要来自太阳方向。对于局部地形展示来说，默认太阳光经常存在以下问题：

1. **真实太阳角度不一定适合展示**  
   正午太阳高度较高时，坡面明暗差异很弱；傍晚太阳角度较低时，阴影又可能过长、过黑。

2. **terrain 需要顶点法线才能产生明显坡面明暗**  
   如果 terrain 数据没有 vertex normals（顶点法线），或者客户端没有请求法线数据，即使开启了光照，坡面的明暗也可能不明显。

3. **真实光照不等于制图美化**  
   真实光照追求物理合理，而地形展示更需要可读性和视觉层次。hillshade（山体阴影渲染）通常比真实太阳光更适合做地图式地形增强。

---

## 3. 推荐总体架构

推荐采用五层组合：

| 层级 | 技术 | 作用 | 推荐程度 |
|---|---|---|---|
| 第一层 | 高质量 terrain | 提供真实地形几何起伏 | 必须 |
| 第二层 | vertex normals（顶点法线） | 支持坡面光照计算 | 必须 |
| 第三层 | 固定 DirectionalLight（方向光） | 提供稳定、美观的主光方向 | 强烈推荐 |
| 第四层 | vertical exaggeration（垂直夸张） | 强化局部地形 3D（Three-Dimensional，三维）感 | 推荐 |
| 第五层 | hillshade（山体阴影渲染）影像层 | 提升制图质感和地形可读性 | 强烈推荐 |

推荐组合：

```text
真实 terrain 几何
+ 顶点法线光照
+ 固定艺术方向光
+ 轻微垂直夸张
+ 半透明局部 hillshade 增强层
```

---

## 4. 基础设置：开启 terrain 顶点法线和 Globe 光照

### 4.1 使用 Cesium World Terrain

如果使用 Cesium World Terrain，可以这样初始化：

```js
const viewer = new Cesium.Viewer("cesiumContainer", {
  terrain: Cesium.Terrain.fromWorldTerrain({
    requestVertexNormals: true,
  }),
  scene3DOnly: true,
  shadows: false,
});

const scene = viewer.scene;
const globe = scene.globe;

globe.enableLighting = true;
globe.lambertDiffuseMultiplier = 1.45;
globe.vertexShadowDarkness = 0.48;
```

### 4.2 关键参数说明

| 参数 | 作用 | 推荐值 |
|---|---|---:|
| `requestVertexNormals` | 请求 terrain 顶点法线 | `true` |
| `globe.enableLighting` | 开启 Globe（地球表面）光照 | `true` |
| `globe.lambertDiffuseMultiplier` | 增强 Lambert（兰伯特）漫反射强度 | `1.2 ~ 1.8` |
| `globe.vertexShadowDarkness` | 控制顶点阴影暗度 | `0.35 ~ 0.65` |
| `shadows` | 是否启用真实阴影 | 默认建议 `false` |

---

## 5. 使用固定艺术方向光，而不是完全依赖真实太阳

### 5.1 推荐原因

对于局部地形展示，固定方向光通常比真实太阳光更稳定、更美观：

- 画面不会随时间变化而忽明忽暗；
- 可以固定采用经典的西北光方向；
- 更容易控制山脊、沟谷、坡面的明暗关系；
- 适合做展示系统、数字孪生、地理信息可视化大屏。

推荐方向：

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| 光源方位角 | `315°` | 西北方向来光，地形阅读性好 |
| 光源高度角 | `30° ~ 45°` | 三维感明显，同时避免暗部过重 |
| 光照强度 | `1.5 ~ 2.5` | 根据底图亮度调整 |
| 光照颜色 | 暖白色 | 比纯白光更自然 |

### 5.2 固定方向光代码

```js
function createLocalLightDirection(lon, lat, sourceAzimuthDeg = 315, sourceElevationDeg = 35) {
  const center = Cesium.Cartesian3.fromDegrees(lon, lat);
  const enu = Cesium.Transforms.eastNorthUpToFixedFrame(center);

  const east4 = Cesium.Matrix4.getColumn(enu, 0, new Cesium.Cartesian4());
  const north4 = Cesium.Matrix4.getColumn(enu, 1, new Cesium.Cartesian4());
  const up4 = Cesium.Matrix4.getColumn(enu, 2, new Cesium.Cartesian4());

  const east = new Cesium.Cartesian3(east4.x, east4.y, east4.z);
  const north = new Cesium.Cartesian3(north4.x, north4.y, north4.z);
  const up = new Cesium.Cartesian3(up4.x, up4.y, up4.z);

  const az = Cesium.Math.toRadians(sourceAzimuthDeg);
  const el = Cesium.Math.toRadians(sourceElevationDeg);

  // 从地面指向光源的方向：方位角按“北为 0°，顺时针增加”计算。
  const sourceDir = new Cesium.Cartesian3();

  Cesium.Cartesian3.multiplyByScalar(
    east,
    Math.sin(az) * Math.cos(el),
    sourceDir
  );

  const northPart = Cesium.Cartesian3.multiplyByScalar(
    north,
    Math.cos(az) * Math.cos(el),
    new Cesium.Cartesian3()
  );
  Cesium.Cartesian3.add(sourceDir, northPart, sourceDir);

  const upPart = Cesium.Cartesian3.multiplyByScalar(
    up,
    Math.sin(el),
    new Cesium.Cartesian3()
  );
  Cesium.Cartesian3.add(sourceDir, upPart, sourceDir);

  // DirectionalLight.direction 表示光线传播方向，因此需要取反。
  return Cesium.Cartesian3.negate(
    Cesium.Cartesian3.normalize(sourceDir, sourceDir),
    new Cesium.Cartesian3()
  );
}

// 替换为局部地形中心点。
const localLon = 103.85;
const localLat = 31.68;

viewer.scene.light = new Cesium.DirectionalLight({
  direction: createLocalLightDirection(localLon, localLat, 315, 35),
  color: Cesium.Color.fromBytes(255, 244, 225),
  intensity: 2.0,
});
```

---

## 6. 轻微垂直夸张：增强 3D 感

### 6.1 推荐配置

```js
viewer.scene.verticalExaggeration = 1.25;
viewer.scene.verticalExaggerationRelativeHeight = 0.0;
```

### 6.2 参数建议

| 地形类型 | 推荐垂直夸张倍数 |
|---|---:|
| 高山峡谷 | `1.1 ~ 1.3` |
| 丘陵、台地 | `1.3 ~ 1.8` |
| 平原微地貌 | `1.8 ~ 3.0` |
| 科研、测绘、严肃分析 | `1.0` 或明确标注夸张倍数 |

建议从 `1.25` 开始调。  
如果超过 `2.0`，局部地形可能会变得过于“沙盘化”或“模型化”。

---

## 7. 局部 hillshade 叠加：提升美观和可读性

### 7.1 为什么推荐 hillshade

hillshade（山体阴影渲染）是地图制图中常用的地形增强方式。它的优势是：

- 可以稳定地表现山脊、沟谷、坡面；
- 不依赖真实太阳时间；
- 可通过透明度、对比度、亮度等参数精细调节；
- 多方向 hillshade 可以避免单方向阴影导致的过曝和死黑。

在美观展示场景中，hillshade 往往比真实 shadow map（阴影贴图）更可控。

### 7.2 推荐数据处理流程

```text
DEM（Digital Elevation Model，数字高程模型）
→ 生成多方向 hillshade
→ 色彩和对比度调整
→ 输出局部瓦片
→ CesiumJS 作为 imagery layer（影像图层）叠加
```

可选瓦片格式：

- XYZ（Slippy Map XYZ 瓦片）；
- WMTS（Web Map Tile Service，网络地图瓦片服务）；
- 局部静态 PNG（Portable Network Graphics，便携式网络图形）切片。

### 7.3 CesiumJS 中叠加 hillshade 图层

```js
const hillshadeLayer = viewer.imageryLayers.addImageryProvider(
  new Cesium.UrlTemplateImageryProvider({
    url: "/tiles/hillshade/{z}/{x}/{y}.png",
    rectangle: Cesium.Rectangle.fromDegrees(
      minLon,
      minLat,
      maxLon,
      maxLat
    ),
  })
);

hillshadeLayer.alpha = 0.28;
hillshadeLayer.contrast = 1.15;
hillshadeLayer.brightness = 1.03;
hillshadeLayer.gamma = 0.95;
```

### 7.4 hillshade 叠加方式选择

| 方式 | 效果 | 维护成本 | 推荐程度 |
|---|---|---:|---:|
| 服务端预融合底图 + hillshade | 最稳定、最美观 | 中 | ★★★★★ |
| hillshade 做成透明阴影/高光层 | 灵活，适合局部增强 | 中 | ★★★★☆ |
| 灰度 hillshade 直接半透明叠加 | 简单，但容易让画面发灰 | 低 | ★★☆☆☆ |
| 修改 CesiumJS Globe shader（着色器）实现 multiply（正片叠底） | 效果可控，但升级维护成本高 | 高 | ★★★☆☆ |

最推荐：

```text
服务端预融合底图 + 多方向 hillshade
```

如果需要前端灵活控制，则推荐：

```text
透明 hillshade 增强层 + CesiumJS ImageryLayer 参数调节
```

---

## 8. 是否开启真实 terrain 阴影

### 8.1 不建议默认开启

真实 terrain 阴影可以增强遮挡关系，但它通常不适合作为默认美化方案，原因包括：

- 对性能有明显影响；
- 低太阳角时阴影可能过长；
- 山谷和背光面可能过黑；
- 局部浏览时可能产生闪烁或阴影分辨率不足的问题；
- 视觉效果不如 hillshade 稳定可控。

### 8.2 只有在这些场景中建议开启

| 场景 | 是否建议开启真实阴影 |
|---|---|
| 普通地形展示 | 不建议 |
| 局部重点演示 | 可以开启 |
| 日照分析 | 建议开启 |
| 低端设备或移动端 | 不建议 |
| 大屏美观展示 | 优先使用 hillshade |

### 8.3 开启真实阴影的示例

```js
const viewer = new Cesium.Viewer("cesiumContainer", {
  terrain: Cesium.Terrain.fromWorldTerrain({
    requestVertexNormals: true,
  }),
  shadows: true,
});

viewer.scene.globe.enableLighting = true;
viewer.scene.globe.shadows = Cesium.ShadowMode.ENABLED;
```

---

## 9. 推荐完整初始化代码

下面是一套适合局部地形美化展示的基础配置：

```js
const viewer = new Cesium.Viewer("cesiumContainer", {
  terrain: Cesium.Terrain.fromWorldTerrain({
    requestVertexNormals: true,
  }),
  scene3DOnly: true,
  shadows: false,
});

const scene = viewer.scene;
const globe = scene.globe;

// 1. 开启 Globe 光照。
globe.enableLighting = true;

// 2. 增强坡面明暗。
globe.lambertDiffuseMultiplier = 1.45;
globe.vertexShadowDarkness = 0.48;

// 3. 轻微垂直夸张。
scene.verticalExaggeration = 1.25;
scene.verticalExaggerationRelativeHeight = 0.0;

// 4. 设置固定艺术方向光。
scene.light = new Cesium.DirectionalLight({
  direction: createLocalLightDirection(103.85, 31.68, 315, 35),
  color: Cesium.Color.fromBytes(255, 244, 225),
  intensity: 2.0,
});

// 5. 可选：叠加局部 hillshade 图层。
const hillshadeLayer = viewer.imageryLayers.addImageryProvider(
  new Cesium.UrlTemplateImageryProvider({
    url: "/tiles/hillshade/{z}/{x}/{y}.png",
    rectangle: Cesium.Rectangle.fromDegrees(minLon, minLat, maxLon, maxLat),
  })
);

hillshadeLayer.alpha = 0.28;
hillshadeLayer.contrast = 1.15;
hillshadeLayer.brightness = 1.03;
hillshadeLayer.gamma = 0.95;
```

配套方向光函数：

```js
function createLocalLightDirection(lon, lat, sourceAzimuthDeg = 315, sourceElevationDeg = 35) {
  const center = Cesium.Cartesian3.fromDegrees(lon, lat);
  const enu = Cesium.Transforms.eastNorthUpToFixedFrame(center);

  const east4 = Cesium.Matrix4.getColumn(enu, 0, new Cesium.Cartesian4());
  const north4 = Cesium.Matrix4.getColumn(enu, 1, new Cesium.Cartesian4());
  const up4 = Cesium.Matrix4.getColumn(enu, 2, new Cesium.Cartesian4());

  const east = new Cesium.Cartesian3(east4.x, east4.y, east4.z);
  const north = new Cesium.Cartesian3(north4.x, north4.y, north4.z);
  const up = new Cesium.Cartesian3(up4.x, up4.y, up4.z);

  const az = Cesium.Math.toRadians(sourceAzimuthDeg);
  const el = Cesium.Math.toRadians(sourceElevationDeg);

  const sourceDir = new Cesium.Cartesian3();

  Cesium.Cartesian3.multiplyByScalar(
    east,
    Math.sin(az) * Math.cos(el),
    sourceDir
  );

  const northPart = Cesium.Cartesian3.multiplyByScalar(
    north,
    Math.cos(az) * Math.cos(el),
    new Cesium.Cartesian3()
  );
  Cesium.Cartesian3.add(sourceDir, northPart, sourceDir);

  const upPart = Cesium.Cartesian3.multiplyByScalar(
    up,
    Math.sin(el),
    new Cesium.Cartesian3()
  );
  Cesium.Cartesian3.add(sourceDir, upPart, sourceDir);

  return Cesium.Cartesian3.negate(
    Cesium.Cartesian3.normalize(sourceDir, sourceDir),
    new Cesium.Cartesian3()
  );
}
```

---

## 10. 推荐调参顺序

建议按以下顺序调试，不要一开始同时改太多参数：

1. **确认 terrain 数据质量**  
   确保 terrain 分辨率足够高，并支持 vertex normals（顶点法线）。

2. **开启 Globe 光照**

   ```js
   viewer.scene.globe.enableLighting = true;
   ```

3. **增强 Lambert 漫反射**

   ```js
   viewer.scene.globe.lambertDiffuseMultiplier = 1.3;
   ```

4. **调整暗部强度**

   ```js
   viewer.scene.globe.vertexShadowDarkness = 0.45;
   ```

5. **切换到固定方向光**

   ```js
   viewer.scene.light = new Cesium.DirectionalLight({
     direction: createLocalLightDirection(localLon, localLat, 315, 35),
     intensity: 2.0,
   });
   ```

6. **增加轻微垂直夸张**

   ```js
   viewer.scene.verticalExaggeration = 1.25;
   ```

7. **最后叠加 hillshade**

   ```js
   hillshadeLayer.alpha = 0.25;
   ```

---

## 11. 推荐参数起点

| 参数 | 推荐起点 | 可调范围 |
|---|---:|---:|
| `requestVertexNormals` | `true` | 固定建议开启 |
| `globe.enableLighting` | `true` | 固定建议开启 |
| `globe.lambertDiffuseMultiplier` | `1.45` | `1.2 ~ 1.8` |
| `globe.vertexShadowDarkness` | `0.48` | `0.35 ~ 0.65` |
| `scene.verticalExaggeration` | `1.25` | `1.1 ~ 1.8` |
| 方向光方位角 | `315°` | `300° ~ 330°` |
| 方向光高度角 | `35°` | `30° ~ 45°` |
| 方向光强度 | `2.0` | `1.5 ~ 2.5` |
| hillshade 透明度 | `0.28` | `0.18 ~ 0.38` |
| hillshade 对比度 | `1.15` | `1.0 ~ 1.3` |
| hillshade 亮度 | `1.03` | `0.95 ~ 1.1` |
| hillshade gamma（伽马） | `0.95` | `0.85 ~ 1.05` |

---

## 12. 常见问题排查

### 12.1 开启光照后 terrain 仍然没有明暗

优先检查：

```js
terrain: Cesium.Terrain.fromWorldTerrain({
  requestVertexNormals: true,
})
```

如果是自定义 terrain 服务，还要确认服务端数据本身是否包含 vertex normals（顶点法线）。客户端请求法线不代表服务端一定提供法线。

### 12.2 画面太暗

可以依次调整：

```js
globe.vertexShadowDarkness = 0.35;
viewer.scene.light.intensity = 1.8;
hillshadeLayer.alpha = 0.22;
```

### 12.3 画面太平，没有 3D 感

可以依次调整：

```js
globe.lambertDiffuseMultiplier = 1.6;
viewer.scene.verticalExaggeration = 1.4;
hillshadeLayer.contrast = 1.2;
```

### 12.4 hillshade 叠加后底图发灰

解决方式：

- 降低 `hillshadeLayer.alpha`；
- 提高 `hillshadeLayer.contrast`；
- 尽量使用透明阴影/高光层，而不是完整灰度图直接半透明覆盖；
- 更推荐在服务端预融合 hillshade 和底图。

### 12.5 真实阴影效果不稳定

建议关闭真实阴影，改用 hillshade：

```js
const viewer = new Cesium.Viewer("cesiumContainer", {
  shadows: false,
});
```

---

## 13. 最终推荐方案

如果项目目标是“局部地形 3D 感强、美观、稳定、性能可控”，建议最终采用：

```text
CesiumJS terrain：开启 requestVertexNormals
CesiumJS Globe：开启 enableLighting
CesiumJS Light：使用固定 DirectionalLight
CesiumJS Scene：使用 1.2 ~ 1.5 倍垂直夸张
ImageryLayer：叠加局部多方向 hillshade
Shadow map：默认关闭，仅在日照分析或重点展示中开启
```

一句话总结：

> CesiumJS 原生光照负责真实坡面明暗，hillshade 负责美观和地形可读性，垂直夸张负责强化三维感。三者组合，比单纯调太阳光效果更稳定、更好看。
