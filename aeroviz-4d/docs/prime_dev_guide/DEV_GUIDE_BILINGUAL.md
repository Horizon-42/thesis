# 机场4D轨迹与地形数字孪生可视化系统 — 开发文档
# Airport 4D Trajectory & Terrain Digital Twin Visualization — Development Guide

> 格式说明：每个章节先呈现中文版，紧随英文版，以 `<!-- LANG:ZH -->` 和 `<!-- LANG:EN -->` 标记区分，便于脚本自动拆分。
>
> Format note: Each section presents the Chinese version first, immediately followed by the English version, separated by `<!-- LANG:ZH -->` and `<!-- LANG:EN -->` markers for automated splitting.

---

<!-- LANG:ZH -->
## 0. 文档元信息

| 字段 | 内容 |
|------|------|
| 项目代号 | AeroViz-4D |
| 文档版本 | v1.0.0 |
| 最后更新 | 2026-04-01 |
| 目标读者 | 前端开发者、航空算法研究员、AI辅助编程（Vibe-coding）用户 |
| 技术栈 | React + TypeScript + CesiumJS + Python FastAPI |
| 数据协议 | GeoJSON（静态）/ CZML（动态4D轨迹） |

<!-- LANG:EN -->
## 0. Document Metadata

| Field | Content |
|-------|---------|
| Project Code | AeroViz-4D |
| Document Version | v1.0.0 |
| Last Updated | 2026-04-01 |
| Target Audience | Frontend developers, aviation algorithm researchers, AI-assisted (Vibe-coding) users |
| Tech Stack | React + TypeScript + CesiumJS + Python FastAPI |
| Data Protocol | GeoJSON (static) / CZML (dynamic 4D trajectories) |

---

<!-- LANG:ZH -->
## 1. 项目目的与研究背景

### 1.1 为什么需要这个系统

现代航空终端区管理（Terminal Maneuvering Area, TMA）面临三重挑战：

1. **地形复杂性**：山区机场（如加拿大基洛纳 CYLW、尼泊尔卢克拉 VNLK）的进近程序必须精确规避地形，任何路径优化算法都不能脱离真实高程数据进行验证。
2. **4D轨迹时序性**：飞机的空间位置（X/Y/Z）加上精确的到达时间约束（T）构成4D轨迹。传统2D纸质图表无法直观传达多架飞机在时间维度上的间隔协调。
3. **算法结果的可解释性**：混合整数线性规划（MILP）、遗传算法等排序调度模型输出的是抽象数字序列，研究者需要一个可视化沙盘来验证其合理性，并向评审委员会（如答辩评委）直观演示。

本系统的核心价值：**将抽象的算法输出转化为可交互的3D时空演示**，让"这架飞机在13:42:30应该在哪里"这一抽象约束变成屏幕上可拖动时间轴验证的可见轨迹。

### 1.2 学术与工程双重定位

- **学术侧**：支持论文中"地形感知4D轨迹预测"和"进场排序调度"章节的结果可视化与验证。
- **工程侧**：构建可复用的可视化组件库，供后续研究迭代。

<!-- LANG:EN -->
## 1. Project Purpose & Research Background

### 1.1 Why This System Is Needed

Modern Terminal Maneuvering Area (TMA) management faces three compounding challenges:

1. **Terrain Complexity**: Approach procedures at mountainous airports (e.g., Kelowna CYLW in Canada, Lukla VNLK in Nepal) must precisely avoid terrain obstacles. No path-optimization algorithm can be validated without real elevation data as the ground truth.
2. **4D Trajectory Temporality**: An aircraft's position (X/Y/Z) combined with a precise arrival time constraint (T) constitutes a 4D trajectory. Traditional 2D paper charts cannot intuitively convey the temporal spacing coordination among multiple aircraft.
3. **Algorithm Result Interpretability**: Sequencing and scheduling models such as Mixed-Integer Linear Programming (MILP) or genetic algorithms output abstract numeric sequences. Researchers need a visualization sandbox to validate correctness and present findings intuitively to review committees (thesis defense panels, conference reviewers).

The core value of this system: **Transform abstract algorithm output into interactive 3D spatiotemporal demonstrations**, making the abstract constraint "aircraft X should be at position Y at 13:42:30" into a visible trajectory verifiable by dragging a timeline slider.

### 1.2 Dual Academic & Engineering Positioning

- **Academic side**: Supports result visualization and validation for "terrain-aware 4D trajectory prediction" and "arrival sequencing scheduling" chapters in the thesis.
- **Engineering side**: Builds a reusable visualization component library for iterative research.

---

<!-- LANG:ZH -->
## 2. 所需知识背景

### 2.1 航空领域知识

#### 2.1.1 终端区与进近程序基础

| 概念 | 说明 |
|------|------|
| TMA（终端区）| 机场周围半径通常30-100海里、高度FL245以下的受控空域 |
| IAF / IF / FAF | 起始进近定位点 / 中间进近定位点 / 最终进近定位点，构成进近程序的路径节点 |
| RNAV/RNP | 区域导航/所需导航性能，允许飞机在精确导航下沿弯曲路径飞行 |
| 最低扇区高度（MSA）| 距机场导航台一定半径内保证300m以上障碍物净空的最低飞行高度 |
| OCA/H | 障碍物超越高度/高，飞机必须在此高度以上飞越以避开障碍物 |

#### 2.1.2 4D轨迹概念

- **RBT（参考商业轨迹）**：航空公司向ATC承诺的飞行路径，是TBO（轨迹基础运行）的核心概念。
- **CTA（受控到达时间）**：ATC分配给飞机在某定位点上空通过的精确时刻，误差通常在±30秒以内。
- **间隔标准**：雷达间隔最小3海里，非雷达间隔5-10分钟飞行时间；这些约束构成调度算法的核心不等式约束。

#### 2.1.3 障碍物超限面（OCS）

PANS-OPS (ICAO Doc 8168) 定义了进近程序的保护区几何：
- **主保护区**：跑道中心线两侧对称延伸，全程提供完整障碍物清除保障。
- **副保护区**：主保护区外侧，以7:1的水平-垂直比例向外向上倾斜，障碍物清除保障逐渐减弱至零。
- **OCS斜面**：最终进近段的障碍物超限面以特定角度（如2.5%坡度）向跑道入口升起，所有地面/人工障碍物必须在此面以下。

### 2.2 前端技术知识

#### 2.2.1 React + TypeScript 基础

需要掌握的概念：
- `useEffect` / `useRef`：CesiumJS Viewer必须在DOM挂载后初始化，并通过ref持有其引用。
- `useState` + Context API：管理时间轴播放状态、选中飞机、图层可见性等全局UI状态。
- TypeScript接口定义：为CZML数据结构、GeoJSON Feature属性定义强类型，避免运行时错误。

#### 2.2.2 CesiumJS 核心概念

| API层级 | 用途 |
|---------|------|
| `Viewer` | 顶层容器，包含场景、相机、时钟、时间轴等所有子系统 |
| `Scene` / `Globe` | 控制地球渲染、光照、大气效果 |
| `Entity API` | 声明式绘制几何体（Polygon、Polyline、Billboard、Model）；适合中小规模静态数据 |
| `Primitive API` | 命令式高性能绘制；适合大规模动态更新场景 |
| `DataSource` | 批量加载GeoJSON/CZML数据集的容器 |
| `Clock` | 驱动动画时间的系统时钟，与CZML时间戳联动 |
| `SampledPositionProperty` | 存储带时间戳的位置序列，支持插值（线性/拉格朗日/Hermite） |
| `Camera` | 控制视角，支持`flyTo`、`lookAt`、`setView`等操作 |

#### 2.2.3 坐标系统

- **WGS84**：地球椭球坐标系，经纬度+大地高，CesiumJS的基础坐标系。
- **Cartesian3**：CesiumJS内部笛卡尔坐标，`Cesium.Cartesian3.fromDegrees(lon, lat, alt)` 完成转换。
- **ENU（东北天）局部坐标**：用于计算OCS几何体的局部偏移，再转换回WGS84。

### 2.3 Python后端知识

#### 2.3.1 OpenAP 库

OpenAP是一个开源的航空器性能模型库，提供：
- 各机型（B737、A320等）的爬升/巡航/下降性能包络。
- 给定航路点序列时，计算满足性能约束的速度/高度剖面。
- 用于验证4D轨迹的可飞性（flyability）。

#### 2.3.2 CZML 数据格式

CZML（Cesium Language）是一种JSON时间序列格式，专为CesiumJS设计：
```json
[
  { "id": "document", "name": "4D轨迹", "version": "1.0",
    "clock": { "interval": "2026-04-01T08:00:00Z/2026-04-01T09:00:00Z",
               "currentTime": "2026-04-01T08:00:00Z", "multiplier": 60 }},
  { "id": "UAL123",
    "model": { "gltf": "/models/aircraft.glb", "scale": 3.0 },
    "position": {
      "epoch": "2026-04-01T08:00:00Z",
      "cartographicDegrees": [
        0,   -119.38, 49.95, 4500,
        120, -119.42, 49.88, 3800,
        240, -119.45, 49.80, 3200
      ]
    },
    "orientation": { "velocityReference": "#UAL123.position" }
  }
]
```
`cartographicDegrees` 中每4个值为一组：`[秒偏移, 经度, 纬度, 高度(米)]`。

<!-- LANG:EN -->
## 2. Required Knowledge Background

### 2.1 Aviation Domain Knowledge

#### 2.1.1 Terminal Area & Approach Procedure Basics

| Concept | Description |
|---------|-------------|
| TMA (Terminal Maneuvering Area) | Controlled airspace surrounding an airport, typically 30–100 NM radius and below FL245 |
| IAF / IF / FAF | Initial Approach Fix / Intermediate Fix / Final Approach Fix — the waypoint nodes of an approach procedure |
| RNAV/RNP | Area Navigation / Required Navigation Performance — allows aircraft to fly curved paths with precision guidance |
| MSA (Minimum Sector Altitude) | Lowest altitude within a defined radius of the airport navaid that provides 300 m obstacle clearance |
| OCA/H | Obstacle Clearance Altitude/Height — aircraft must fly above this to clear terrain obstacles |

#### 2.1.2 4D Trajectory Concepts

- **RBT (Reference Business Trajectory)**: The flight path an airline commits to with ATC; the core construct of TBO (Trajectory-Based Operations).
- **CTA (Controlled Time of Arrival)**: The precise time slot ATC assigns for an aircraft to cross a given fix, typically within a ±30-second tolerance window.
- **Separation Standards**: Radar separation minimum 3 NM; non-radar 5–10 flight minutes. These constraints form the core inequality constraints of the scheduling algorithm.

#### 2.1.3 Obstacle Clearance Surface (OCS)

PANS-OPS (ICAO Doc 8168) defines the protection area geometry for approach procedures:
- **Primary Protection Area**: Symmetric extension on both sides of the runway centerline; full obstacle clearance is guaranteed throughout.
- **Secondary Protection Area**: Outer flanks of the primary area; slopes outward and upward at a 7:1 horizontal-to-vertical ratio, with obstacle clearance gradually reducing to zero.
- **OCS Slope**: The obstacle clearance surface for the final approach segment rises toward the runway threshold at a defined gradient (e.g., 2.5% slope); all terrain and man-made obstacles must remain below this surface.

### 2.2 Frontend Technical Knowledge

#### 2.2.1 React + TypeScript Fundamentals

Key concepts to master:
- `useEffect` / `useRef`: The CesiumJS Viewer must be initialized after the DOM mounts, and its instance must be held via a ref to avoid stale closure issues.
- `useState` + Context API: Manages global UI state such as timeline playback status, selected aircraft, and layer visibility.
- TypeScript interface definitions: Strongly type CZML data structures and GeoJSON Feature properties to prevent runtime errors.

#### 2.2.2 CesiumJS Core Concepts

| API Layer | Purpose |
|-----------|---------|
| `Viewer` | Top-level container holding scene, camera, clock, timeline, and all subsystems |
| `Scene` / `Globe` | Controls Earth rendering, lighting, and atmospheric effects |
| `Entity API` | Declarative geometry drawing (Polygon, Polyline, Billboard, Model); suited for small-to-medium static datasets |
| `Primitive API` | Imperative high-performance rendering; suited for large-scale dynamic update scenarios |
| `DataSource` | Container for bulk-loading GeoJSON/CZML datasets |
| `Clock` | System clock driving animation time, synchronized with CZML timestamps |
| `SampledPositionProperty` | Stores time-stamped position sequences with interpolation support (linear/Lagrange/Hermite) |
| `Camera` | Controls the viewpoint; supports `flyTo`, `lookAt`, `setView`, and other operations |

#### 2.2.3 Coordinate Systems

- **WGS84**: Earth ellipsoid coordinate system — longitude, latitude, and ellipsoidal height; the foundational coordinate system for CesiumJS.
- **Cartesian3**: CesiumJS internal Cartesian coordinates; use `Cesium.Cartesian3.fromDegrees(lon, lat, alt)` for conversion.
- **ENU (East-North-Up) Local Frame**: Used for computing local offsets of OCS geometry before converting back to WGS84.

### 2.3 Python Backend Knowledge

#### 2.3.1 The OpenAP Library

OpenAP is an open-source aircraft performance model library providing:
- Climb/cruise/descent performance envelopes for common aircraft types (B737, A320, etc.).
- Speed/altitude profile computation satisfying performance constraints for a given waypoint sequence.
- Flyability validation for 4D trajectories.

#### 2.3.2 CZML Data Format

CZML (Cesium Language) is a JSON time-series format designed specifically for CesiumJS:
```json
[
  { "id": "document", "name": "4D Trajectories", "version": "1.0",
    "clock": { "interval": "2026-04-01T08:00:00Z/2026-04-01T09:00:00Z",
               "currentTime": "2026-04-01T08:00:00Z", "multiplier": 60 }},
  { "id": "UAL123",
    "model": { "gltf": "/models/aircraft.glb", "scale": 3.0 },
    "position": {
      "epoch": "2026-04-01T08:00:00Z",
      "cartographicDegrees": [
        0,   -119.38, 49.95, 4500,
        120, -119.42, 49.88, 3800,
        240, -119.45, 49.80, 3200
      ]
    },
    "orientation": { "velocityReference": "#UAL123.position" }
  }
]
```
Each group of 4 values in `cartographicDegrees`: `[secondOffset, longitude, latitude, altitude(m)]`.

---

<!-- LANG:ZH -->
## 3. 系统架构设计

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户浏览器                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              React + TypeScript 前端                  │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ CesiumViewer│  │ControlPanel  │  │FlightTable │  │   │
│  │  │  Component  │  │  Component   │  │ Component  │  │   │
│  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │   │
│  │         └────────────────┼────────────────┘          │   │
│  │                    AppContext                         │   │
│  │              (时钟状态/选中航班/图层开关)               │   │
│  └──────────────────────────────────────────────────────┘   │
│                             │                                │
│               HTTP / 静态文件服务                             │
└─────────────────────────────┼───────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────┴─────┐        ┌─────┴─────┐       ┌─────┴──────┐
    │ GeoJSON  │        │   CZML    │       │ Python     │
    │  静态数据 │        │  轨迹数据  │       │ FastAPI    │
    │ (跑道/   │        │ (4D航班   │       │ (算法后端  │
    │  障碍面) │        │  序列)    │       │  /可选)    │
    └──────────┘        └───────────┘       └────────────┘
```

### 3.2 数据流向

```
OurAirports CSV → Python预处理 → runway.geojson
                                        ↓
ICAO PANS-OPS几何计算 → ocs_surfaces.geojson
                                        ↓
Nav Canada/FAA CIFP → waypoints.geojson
                                        ↓
调度算法(MILP/遗传算法) → trajectories.czml
                                        ↓
                              CesiumJS 3D渲染
```

### 3.3 目录结构

```
aeroviz-4d/
├── public/
│   ├── models/
│   │   └── aircraft.glb          # 3D飞机模型
│   └── data/
│       ├── runway.geojson         # 跑道多边形
│       ├── ocs_surfaces.geojson   # OCS保护面
│       ├── waypoints.geojson      # 航路点
│       └── trajectories.czml     # 4D轨迹(由Python生成)
├── src/
│   ├── components/
│   │   ├── CesiumViewer.tsx       # 主3D视图组件
│   │   ├── ControlPanel.tsx       # 播放控制面板
│   │   ├── FlightTable.tsx        # 航班序列表格
│   │   └── LayerToggle.tsx        # 图层开关组件
│   ├── hooks/
│   │   ├── useCesiumViewer.ts     # Viewer初始化Hook
│   │   ├── useTerrainLoader.ts    # 地形加载Hook
│   │   └── useCzmlLoader.ts       # CZML加载Hook
│   ├── context/
│   │   └── AppContext.tsx          # 全局状态Context
│   ├── types/
│   │   ├── czml.d.ts              # CZML类型定义
│   │   └── geojson-aviation.d.ts  # 航空GeoJSON属性类型
│   ├── utils/
│   │   ├── ocsGeometry.ts         # OCS几何计算工具
│   │   └── czmlBuilder.ts         # CZML构建辅助函数
│   ├── App.tsx
│   └── main.tsx
├── python/
│   ├── generate_czml.py           # CZML生成脚本
│   ├── preprocess_airports.py     # 机场数据预处理
│   └── requirements.txt
├── vite.config.ts
├── tsconfig.json
└── package.json
```

<!-- LANG:EN -->
## 3. System Architecture Design

### 3.1 Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       User Browser                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              React + TypeScript Frontend              │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ CesiumViewer│  │ControlPanel  │  │FlightTable │  │   │
│  │  │  Component  │  │  Component   │  │ Component  │  │   │
│  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │   │
│  │         └────────────────┼────────────────┘          │   │
│  │                    AppContext                         │   │
│  │         (clock state / selected flight / layers)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                             │                                │
│               HTTP / Static file serving                     │
└─────────────────────────────┼───────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────┴─────┐        ┌─────┴─────┐       ┌─────┴──────┐
    │ GeoJSON  │        │   CZML    │       │  Python    │
    │  Static  │        │ Trajectory│       │  FastAPI   │
    │ (runway/ │        │  (4D seq) │       │  (backend  │
    │  OCS)    │        │           │       │  optional) │
    └──────────┘        └───────────┘       └────────────┘
```

### 3.2 Data Flow

```
OurAirports CSV → Python preprocessing → runway.geojson
                                               ↓
ICAO PANS-OPS geometry calculation → ocs_surfaces.geojson
                                               ↓
Nav Canada/FAA CIFP → waypoints.geojson
                                               ↓
Scheduling algorithm (MILP/GA) → trajectories.czml
                                               ↓
                                    CesiumJS 3D rendering
```

### 3.3 Directory Structure

```
aeroviz-4d/
├── public/
│   ├── models/
│   │   └── aircraft.glb          # 3D aircraft model
│   └── data/
│       ├── runway.geojson         # runway polygons
│       ├── ocs_surfaces.geojson   # OCS protection surfaces
│       ├── waypoints.geojson      # approach waypoints
│       └── trajectories.czml     # 4D trajectories (Python-generated)
├── src/
│   ├── components/
│   │   ├── CesiumViewer.tsx       # main 3D view component
│   │   ├── ControlPanel.tsx       # playback control panel
│   │   ├── FlightTable.tsx        # flight sequence table
│   │   └── LayerToggle.tsx        # layer toggle component
│   ├── hooks/
│   │   ├── useCesiumViewer.ts     # Viewer initialization hook
│   │   ├── useTerrainLoader.ts    # terrain loading hook
│   │   └── useCzmlLoader.ts      # CZML loading hook
│   ├── context/
│   │   └── AppContext.tsx         # global state context
│   ├── types/
│   │   ├── czml.d.ts             # CZML type definitions
│   │   └── geojson-aviation.d.ts # aviation GeoJSON property types
│   ├── utils/
│   │   ├── ocsGeometry.ts        # OCS geometry computation utilities
│   │   └── czmlBuilder.ts        # CZML construction helpers
│   ├── App.tsx
│   └── main.tsx
├── python/
│   ├── generate_czml.py          # CZML generation script
│   ├── preprocess_airports.py    # airport data preprocessing
│   └── requirements.txt
├── vite.config.ts
├── tsconfig.json
└── package.json
```

---

<!-- LANG:ZH -->
## 4. 阶段一：基础环境搭建与CesiumJS初始化

### 4.1 目标

建立拥有真实地球曲率、高精度卫星底图、地形渲染、动态光照的3D画布，使后续所有图层有正确的地理坐标基础。

### 4.2 环境安装

```bash
# 初始化Vite + React + TypeScript项目
npm create vite@latest aeroviz-4d -- --template react-ts
cd aeroviz-4d

# 安装CesiumJS及Vite插件
npm install cesium
npm install -D vite-plugin-cesium

# 安装其他依赖
npm install
```

### 4.3 Vite配置

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import cesium from 'vite-plugin-cesium';

export default defineConfig({
  plugins: [react(), cesium()],
});
```

### 4.4 主Viewer组件

```typescript
// src/components/CesiumViewer.tsx
import { useEffect, useRef } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';

// !! 在此填入您的 Cesium Ion Access Token
// 从 https://cesium.com/ion/tokens 免费申请
const CESIUM_ION_TOKEN = 'YOUR_TOKEN_HERE';

// 目标机场坐标（以CYLW基洛纳为例）
const AIRPORT_LON = -119.3775;
const AIRPORT_LAT = 49.9561;
const INITIAL_HEIGHT = 15000; // 初始相机高度，单位：米

export default function CesiumViewerComponent() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);

  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    Cesium.Ion.defaultAccessToken = CESIUM_ION_TOKEN;

    viewerRef.current = new Cesium.Viewer(containerRef.current, {
      // 使用Cesium World Terrain高精度地形
      terrain: Cesium.Terrain.fromWorldTerrain({
        requestVertexNormals: true,   // 开启法线以计算光照/阴影
        requestWaterMask: true,        // 开启水体反射遮罩
      }),
      // 隐藏默认UI控件，保持界面简洁
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      // 保留时间轴和动画控制器（4D轨迹需要）
      animation: true,
      timeline: true,
      // 开启HDR渲染和大气效果
      skyAtmosphere: new Cesium.SkyAtmosphere(),
    });

    // 开启地形光照
    viewerRef.current.scene.globe.enableLighting = true;

    // 设置初始相机视角（倾斜俯瞰机场）
    viewerRef.current.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(
        AIRPORT_LON, AIRPORT_LAT, INITIAL_HEIGHT
      ),
      orientation: {
        heading: Cesium.Math.toRadians(0),   // 朝北
        pitch: Cesium.Math.toRadians(-45),    // 向下倾斜45度
        roll: 0,
      },
    });

    return () => {
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: '100vw', height: '100vh', position: 'absolute', top: 0, left: 0 }}
    />
  );
}
```

### 4.5 AI辅助提示词

> 将以下提示词发给 Cursor / Windsurf / Claude 等AI编程工具：

```
请基于上述 CesiumViewerComponent 组件，完成以下工作：
1. 在 App.tsx 中引入该组件，使其占满全屏。
2. 在页面左上角添加一个半透明的信息面板（HUD），使用绝对定位，
   显示当前相机高度（米）和经纬度，数据来自 viewer.camera。
3. 所有样式使用 CSS Modules（*.module.css），不要用内联样式。
```

<!-- LANG:EN -->
## 4. Phase 1: Environment Setup & CesiumJS Initialization

### 4.1 Objective

Establish a 3D canvas with realistic Earth curvature, high-resolution satellite imagery, terrain rendering, and dynamic lighting — giving all subsequent layers a correct geographic coordinate foundation.

### 4.2 Environment Installation

```bash
# Initialize Vite + React + TypeScript project
npm create vite@latest aeroviz-4d -- --template react-ts
cd aeroviz-4d

# Install CesiumJS and the Vite plugin
npm install cesium
npm install -D vite-plugin-cesium

# Install all dependencies
npm install
```

### 4.3 Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import cesium from 'vite-plugin-cesium';

export default defineConfig({
  plugins: [react(), cesium()],
});
```

### 4.4 Main Viewer Component

```typescript
// src/components/CesiumViewer.tsx
import { useEffect, useRef } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';

// !! Fill in your Cesium Ion Access Token here
// Get one free at https://cesium.com/ion/tokens
const CESIUM_ION_TOKEN = 'YOUR_TOKEN_HERE';

// Target airport coordinates (Kelowna CYLW as example)
const AIRPORT_LON = -119.3775;
const AIRPORT_LAT = 49.9561;
const INITIAL_HEIGHT = 15000; // initial camera height in meters

export default function CesiumViewerComponent() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);

  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    Cesium.Ion.defaultAccessToken = CESIUM_ION_TOKEN;

    viewerRef.current = new Cesium.Viewer(containerRef.current, {
      // Use Cesium World Terrain high-precision elevation
      terrain: Cesium.Terrain.fromWorldTerrain({
        requestVertexNormals: true,   // enable normals for lighting/shadows
        requestWaterMask: true,        // enable water reflection mask
      }),
      // Hide default UI controls for a clean interface
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      // Keep timeline and animation controller (needed for 4D trajectories)
      animation: true,
      timeline: true,
      // Enable HDR rendering and atmospheric effects
      skyAtmosphere: new Cesium.SkyAtmosphere(),
    });

    // Enable terrain lighting
    viewerRef.current.scene.globe.enableLighting = true;

    // Set initial camera view (tilted bird's-eye view over airport)
    viewerRef.current.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(
        AIRPORT_LON, AIRPORT_LAT, INITIAL_HEIGHT
      ),
      orientation: {
        heading: Cesium.Math.toRadians(0),   // facing north
        pitch: Cesium.Math.toRadians(-45),    // tilted 45° downward
        roll: 0,
      },
    });

    return () => {
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: '100vw', height: '100vh', position: 'absolute', top: 0, left: 0 }}
    />
  );
}
```

### 4.5 AI-Assisted Prompt

> Send the following prompt to Cursor / Windsurf / Claude or similar AI coding tools:

```
Based on the CesiumViewerComponent above, complete the following:
1. Import the component in App.tsx so it fills the full screen.
2. Add a semi-transparent HUD panel in the top-left corner using absolute
   positioning, displaying the current camera altitude (meters) and
   lat/lon coordinates read from viewer.camera.
3. All styling must use CSS Modules (*.module.css) — no inline styles.
```

---

<!-- LANG:ZH -->
## 5. 阶段二：地形与机场几何高保真建模

### 5.1 目标

叠加真实跑道几何形状（从OurAirports数据集提取），使其精确贴合地形表面，并添加跑道灯光视觉效果。

### 5.2 数据准备：OurAirports → GeoJSON

```python
# python/preprocess_airports.py
import pandas as pd
import json
import math

def runway_to_polygon(lat1, lon1, lat2, lon2, width_ft=150):
    """
    将跑道中心线两端点转换为多边形（考虑跑道宽度）
    width_ft: 跑道宽度，英尺（典型值：150ft ≈ 45.7m）
    """
    width_m = width_ft * 0.3048
    bearing = math.atan2(lon2 - lon1, lat2 - lat1)
    perp = bearing + math.pi / 2

    # 经纬度偏移（近似，适用于小范围）
    dlat = (width_m / 2) / 111320
    dlon = (width_m / 2) / (111320 * math.cos(math.radians(lat1)))

    # 四个角点
    corners = [
        [lon1 + dlon * math.cos(perp), lat1 + dlat * math.sin(perp)],
        [lon1 - dlon * math.cos(perp), lat1 - dlat * math.sin(perp)],
        [lon2 - dlon * math.cos(perp), lat2 - dlat * math.sin(perp)],
        [lon2 + dlon * math.cos(perp), lat2 + dlat * math.sin(perp)],
        [lon1 + dlon * math.cos(perp), lat1 + dlat * math.sin(perp)],  # 闭合
    ]
    return corners

# 读取OurAirports runways.csv
df = pd.read_csv('runways.csv')
cylw = df[df['airport_ident'] == 'CYLW']

features = []
for _, row in cylw.iterrows():
    coords = runway_to_polygon(
        row['le_latitude_deg'], row['le_longitude_deg'],
        row['he_latitude_deg'], row['he_longitude_deg'],
        width_ft=row.get('width_ft', 150)
    )
    features.append({
        "type": "Feature",
        "properties": {
            "id": row['id'],
            "airport": row['airport_ident'],
            "le_ident": row['le_ident'],
            "he_ident": row['he_ident'],
            "surface": row.get('surface', 'ASP'),
            "length_ft": row.get('length_ft', 0),
        },
        "geometry": { "type": "Polygon", "coordinates": [coords] }
    })

with open('../public/data/runway.geojson', 'w') as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
print(f"已生成 {len(features)} 条跑道多边形")
```

### 5.3 前端加载跑道

```typescript
// src/hooks/useRunwayLayer.ts
import { useEffect } from 'react';
import * as Cesium from 'cesium';

export function useRunwayLayer(viewer: Cesium.Viewer | null) {
  useEffect(() => {
    if (!viewer) return;

    const dataSource = new Cesium.GeoJsonDataSource('runways');
    
    dataSource.load('/data/runway.geojson', {
      clampToGround: true,   // 跑道多边形严格贴合地形
      fill: new Cesium.Color(0.15, 0.15, 0.15, 0.9),   // 深灰色半透明
      stroke: new Cesium.Color(1.0, 0.9, 0.0, 1.0),     // 黄色边框
      strokeWidth: 2,
    }).then((ds) => {
      viewer.dataSources.add(ds);
      // 调整各跑道实体样式
      ds.entities.values.forEach((entity) => {
        if (entity.polygon) {
          entity.polygon.classificationType =
            Cesium.ClassificationType.TERRAIN; // 仅贴地形，不遮挡模型
        }
      });
    });

    return () => {
      viewer.dataSources.removeAll();
    };
  }, [viewer]);
}
```

### 5.4 AI辅助提示词

```
在现有的 useRunwayLayer Hook 基础上，添加以下功能：
1. 当用户点击某条跑道多边形时，弹出一个信息气泡（InfoBox），
   显示跑道标识（如"34L/16R"）、长度（英尺）和道面类型。
2. 被点击的跑道高亮为蓝色，其他跑道恢复深灰色。
3. 高亮状态通过 AppContext 中的 selectedRunway 状态管理。
```

<!-- LANG:EN -->
## 5. Phase 2: High-Fidelity Terrain & Airport Geometry Modeling

### 5.1 Objective

Overlay real runway geometry (extracted from the OurAirports dataset), clamped precisely to the terrain surface, and add runway lighting visual effects.

### 5.2 Data Preparation: OurAirports → GeoJSON

```python
# python/preprocess_airports.py
import pandas as pd
import json
import math

def runway_to_polygon(lat1, lon1, lat2, lon2, width_ft=150):
    """
    Convert runway centerline endpoints to a polygon (accounting for width).
    width_ft: runway width in feet (typical: 150 ft ≈ 45.7 m)
    """
    width_m = width_ft * 0.3048
    bearing = math.atan2(lon2 - lon1, lat2 - lat1)
    perp = bearing + math.pi / 2

    # Lat/lon offsets (approximate, valid for small areas)
    dlat = (width_m / 2) / 111320
    dlon = (width_m / 2) / (111320 * math.cos(math.radians(lat1)))

    # Four corner points
    corners = [
        [lon1 + dlon * math.cos(perp), lat1 + dlat * math.sin(perp)],
        [lon1 - dlon * math.cos(perp), lat1 - dlat * math.sin(perp)],
        [lon2 - dlon * math.cos(perp), lat2 - dlat * math.sin(perp)],
        [lon2 + dlon * math.cos(perp), lat2 + dlat * math.sin(perp)],
        [lon1 + dlon * math.cos(perp), lat1 + dlat * math.sin(perp)],  # close
    ]
    return corners

# Load OurAirports runways.csv
df = pd.read_csv('runways.csv')
cylw = df[df['airport_ident'] == 'CYLW']

features = []
for _, row in cylw.iterrows():
    coords = runway_to_polygon(
        row['le_latitude_deg'], row['le_longitude_deg'],
        row['he_latitude_deg'], row['he_longitude_deg'],
        width_ft=row.get('width_ft', 150)
    )
    features.append({
        "type": "Feature",
        "properties": {
            "id": row['id'],
            "airport": row['airport_ident'],
            "le_ident": row['le_ident'],
            "he_ident": row['he_ident'],
            "surface": row.get('surface', 'ASP'),
            "length_ft": row.get('length_ft', 0),
        },
        "geometry": { "type": "Polygon", "coordinates": [coords] }
    })

with open('../public/data/runway.geojson', 'w') as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
print(f"Generated {len(features)} runway polygons")
```

### 5.3 Frontend Runway Loading

```typescript
// src/hooks/useRunwayLayer.ts
import { useEffect } from 'react';
import * as Cesium from 'cesium';

export function useRunwayLayer(viewer: Cesium.Viewer | null) {
  useEffect(() => {
    if (!viewer) return;

    const dataSource = new Cesium.GeoJsonDataSource('runways');
    
    dataSource.load('/data/runway.geojson', {
      clampToGround: true,   // strictly clamp runway polygon to terrain
      fill: new Cesium.Color(0.15, 0.15, 0.15, 0.9),   // dark grey semi-transparent
      stroke: new Cesium.Color(1.0, 0.9, 0.0, 1.0),     // yellow border
      strokeWidth: 2,
    }).then((ds) => {
      viewer.dataSources.add(ds);
      ds.entities.values.forEach((entity) => {
        if (entity.polygon) {
          entity.polygon.classificationType =
            Cesium.ClassificationType.TERRAIN; // clamp to terrain only, don't occlude models
        }
      });
    });

    return () => {
      viewer.dataSources.removeAll();
    };
  }, [viewer]);
}
```

### 5.4 AI-Assisted Prompt

```
Building on the existing useRunwayLayer hook, add the following:
1. When the user clicks a runway polygon, show an InfoBox popup
   displaying runway identifier (e.g. "34L/16R"), length (feet),
   and surface type.
2. The clicked runway highlights in blue; all others return to dark grey.
3. Highlight state is managed through selectedRunway in AppContext.
```

---

<!-- LANG:ZH -->
## 6. 阶段三：静态空域结构与OCS超障面可视化

### 6.1 目标

将PANS-OPS定义的抽象保护面几何规则转化为悬浮在真实地形上方、可交互的3D半透明固体，直观展示飞机在进近过程中必须保持的安全间隔。

### 6.2 OCS几何计算

```typescript
// src/utils/ocsGeometry.ts
import * as Cesium from 'cesium';

interface OCSParams {
  /** 最终进近定位点（FAF）坐标 */
  fafLon: number;
  fafLat: number;
  /** 跑道入口坐标 */
  thresholdLon: number;
  thresholdLat: number;
  /** FAF高度（米） */
  fafAlt: number;
  /** 跑道入口高度（米） */
  thresholdAlt: number;
  /** 主保护区半宽（米，典型值：75m用于Cat I ILS） */
  primaryHalfWidth: number;
  /** 副保护区额外宽度（米） */
  secondaryWidth: number;
}

/**
 * 生成最终进近段OCS保护面的Cesium实体数组
 * 返回：主保护区（红色半透明）+ 副保护区×2（橙色半透明）
 */
export function buildOCSSurfaces(params: OCSParams): Cesium.Entity[] {
  const {
    fafLon, fafLat, fafAlt,
    thresholdLon, thresholdLat, thresholdAlt,
    primaryHalfWidth, secondaryWidth,
  } = params;

  // 计算跑道方位角
  const dx = thresholdLon - fafLon;
  const dy = thresholdLat - fafLat;
  const bearingRad = Math.atan2(dx, dy);
  const perpRad = bearingRad + Math.PI / 2;

  // 1度纬度 ≈ 111320m，1度经度 ≈ 111320 * cos(lat) m
  const metersPerDegLat = 111320;
  const metersPerDegLon = 111320 * Math.cos(Cesium.Math.toRadians(fafLat));

  function offsetPoint(lon: number, lat: number, bearing: number, distMeters: number) {
    return {
      lon: lon + (distMeters / metersPerDegLon) * Math.sin(bearing),
      lat: lat + (distMeters / metersPerDegLat) * Math.cos(bearing),
    };
  }

  // FAF处主保护区四个角
  const fafLeft = offsetPoint(fafLon, fafLat, perpRad, primaryHalfWidth);
  const fafRight = offsetPoint(fafLon, fafLat, perpRad, -primaryHalfWidth);
  const thrLeft = offsetPoint(thresholdLon, thresholdLat, perpRad, primaryHalfWidth);
  const thrRight = offsetPoint(thresholdLon, thresholdLat, perpRad, -primaryHalfWidth);

  const primaryEntity = new Cesium.Entity({
    name: 'OCS Primary Protection Area',
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(
        Cesium.Cartesian3.fromDegreesArrayHeights([
          fafLeft.lon, fafLeft.lat, fafAlt,
          fafRight.lon, fafRight.lat, fafAlt,
          thrRight.lon, thrRight.lat, thresholdAlt,
          thrLeft.lon, thrLeft.lat, thresholdAlt,
        ])
      ),
      perPositionHeight: true,
      material: Cesium.Color.RED.withAlpha(0.25),
      outline: true,
      outlineColor: Cesium.Color.RED,
    },
  });

  // 副保护区（右侧，7:1斜面）
  const secFafRight = offsetPoint(fafLon, fafLat, perpRad, -(primaryHalfWidth + secondaryWidth));
  const secAltAtFaf = fafAlt - secondaryWidth / 7; // 7:1斜率
  const secThrRight = offsetPoint(thresholdLon, thresholdLat, perpRad, -(primaryHalfWidth + secondaryWidth));

  const secondaryRightEntity = new Cesium.Entity({
    name: 'OCS Secondary Protection Area (Right)',
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(
        Cesium.Cartesian3.fromDegreesArrayHeights([
          fafRight.lon, fafRight.lat, fafAlt,
          secFafRight.lon, secFafRight.lat, secAltAtFaf,
          secThrRight.lon, secThrRight.lat, thresholdAlt,
          thrRight.lon, thrRight.lat, thresholdAlt,
        ])
      ),
      perPositionHeight: true,
      material: Cesium.Color.ORANGE.withAlpha(0.2),
      outline: true,
      outlineColor: Cesium.Color.ORANGE,
    },
  });

  return [primaryEntity, secondaryRightEntity];
}
```

### 6.3 航路点与进近路径绘制

```typescript
// src/hooks/useWaypointLayer.ts
import { useEffect } from 'react';
import * as Cesium from 'cesium';

export function useWaypointLayer(viewer: Cesium.Viewer | null) {
  useEffect(() => {
    if (!viewer) return;

    fetch('/data/waypoints.geojson')
      .then(r => r.json())
      .then(geojson => {
        geojson.features.forEach((f: GeoJSON.Feature) => {
          const [lon, lat, alt] = f.geometry.coordinates as number[];
          const props = f.properties as { name: string; type: string; minAlt?: number };

          // 航路点圆柱体标记
          viewer.entities.add({
            position: Cesium.Cartesian3.fromDegrees(lon, lat, alt),
            cylinder: {
              length: 300,
              topRadius: 150,
              bottomRadius: 150,
              material: props.type === 'FAF'
                ? Cesium.Color.YELLOW.withAlpha(0.8)
                : Cesium.Color.CYAN.withAlpha(0.7),
            },
            label: {
              text: props.name,
              font: '14px monospace',
              fillColor: Cesium.Color.WHITE,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              outlineWidth: 2,
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              pixelOffset: new Cesium.Cartesian2(0, -20),
            },
          });
        });
      });
  }, [viewer]);
}
```

<!-- LANG:EN -->
## 6. Phase 3: Static Airspace Structures & OCS Visualization

### 6.1 Objective

Convert the abstract protection surface geometry rules defined by PANS-OPS into interactive 3D semi-transparent solids floating above real terrain, visually communicating the safety margins aircraft must maintain during approach.

### 6.2 OCS Geometry Calculation

```typescript
// src/utils/ocsGeometry.ts
import * as Cesium from 'cesium';

interface OCSParams {
  /** Final Approach Fix (FAF) coordinates */
  fafLon: number;
  fafLat: number;
  /** Runway threshold coordinates */
  thresholdLon: number;
  thresholdLat: number;
  /** FAF altitude (meters) */
  fafAlt: number;
  /** Runway threshold altitude (meters) */
  thresholdAlt: number;
  /** Primary protection area half-width (meters, typical: 75m for Cat I ILS) */
  primaryHalfWidth: number;
  /** Secondary protection area additional width (meters) */
  secondaryWidth: number;
}

/**
 * Generate an array of Cesium Entities representing the final approach OCS surfaces.
 * Returns: primary area (red semi-transparent) + 2× secondary areas (orange semi-transparent)
 */
export function buildOCSSurfaces(params: OCSParams): Cesium.Entity[] {
  const {
    fafLon, fafLat, fafAlt,
    thresholdLon, thresholdLat, thresholdAlt,
    primaryHalfWidth, secondaryWidth,
  } = params;

  // Compute runway bearing
  const dx = thresholdLon - fafLon;
  const dy = thresholdLat - fafLat;
  const bearingRad = Math.atan2(dx, dy);
  const perpRad = bearingRad + Math.PI / 2;

  // 1° latitude ≈ 111320 m; 1° longitude ≈ 111320 × cos(lat) m
  const metersPerDegLat = 111320;
  const metersPerDegLon = 111320 * Math.cos(Cesium.Math.toRadians(fafLat));

  function offsetPoint(lon: number, lat: number, bearing: number, distMeters: number) {
    return {
      lon: lon + (distMeters / metersPerDegLon) * Math.sin(bearing),
      lat: lat + (distMeters / metersPerDegLat) * Math.cos(bearing),
    };
  }

  // Four corners of the primary protection area at FAF
  const fafLeft = offsetPoint(fafLon, fafLat, perpRad, primaryHalfWidth);
  const fafRight = offsetPoint(fafLon, fafLat, perpRad, -primaryHalfWidth);
  const thrLeft = offsetPoint(thresholdLon, thresholdLat, perpRad, primaryHalfWidth);
  const thrRight = offsetPoint(thresholdLon, thresholdLat, perpRad, -primaryHalfWidth);

  const primaryEntity = new Cesium.Entity({
    name: 'OCS Primary Protection Area',
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(
        Cesium.Cartesian3.fromDegreesArrayHeights([
          fafLeft.lon, fafLeft.lat, fafAlt,
          fafRight.lon, fafRight.lat, fafAlt,
          thrRight.lon, thrRight.lat, thresholdAlt,
          thrLeft.lon, thrLeft.lat, thresholdAlt,
        ])
      ),
      perPositionHeight: true,
      material: Cesium.Color.RED.withAlpha(0.25),
      outline: true,
      outlineColor: Cesium.Color.RED,
    },
  });

  // Secondary protection area (right side, 7:1 slope)
  const secFafRight = offsetPoint(fafLon, fafLat, perpRad, -(primaryHalfWidth + secondaryWidth));
  const secAltAtFaf = fafAlt - secondaryWidth / 7; // 7:1 slope ratio
  const secThrRight = offsetPoint(thresholdLon, thresholdLat, perpRad, -(primaryHalfWidth + secondaryWidth));

  const secondaryRightEntity = new Cesium.Entity({
    name: 'OCS Secondary Protection Area (Right)',
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(
        Cesium.Cartesian3.fromDegreesArrayHeights([
          fafRight.lon, fafRight.lat, fafAlt,
          secFafRight.lon, secFafRight.lat, secAltAtFaf,
          secThrRight.lon, secThrRight.lat, thresholdAlt,
          thrRight.lon, thrRight.lat, thresholdAlt,
        ])
      ),
      perPositionHeight: true,
      material: Cesium.Color.ORANGE.withAlpha(0.2),
      outline: true,
      outlineColor: Cesium.Color.ORANGE,
    },
  });

  return [primaryEntity, secondaryRightEntity];
}
```

### 6.3 Waypoint & Approach Path Rendering

```typescript
// src/hooks/useWaypointLayer.ts
import { useEffect } from 'react';
import * as Cesium from 'cesium';

export function useWaypointLayer(viewer: Cesium.Viewer | null) {
  useEffect(() => {
    if (!viewer) return;

    fetch('/data/waypoints.geojson')
      .then(r => r.json())
      .then(geojson => {
        geojson.features.forEach((f: GeoJSON.Feature) => {
          const [lon, lat, alt] = f.geometry.coordinates as number[];
          const props = f.properties as { name: string; type: string; minAlt?: number };

          // Waypoint cylinder marker
          viewer.entities.add({
            position: Cesium.Cartesian3.fromDegrees(lon, lat, alt),
            cylinder: {
              length: 300,
              topRadius: 150,
              bottomRadius: 150,
              material: props.type === 'FAF'
                ? Cesium.Color.YELLOW.withAlpha(0.8)
                : Cesium.Color.CYAN.withAlpha(0.7),
            },
            label: {
              text: props.name,
              font: '14px monospace',
              fillColor: Cesium.Color.WHITE,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              outlineWidth: 2,
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              pixelOffset: new Cesium.Cartesian2(0, -20),
            },
          });
        });
      });
  }, [viewer]);
}
```

---

<!-- LANG:ZH -->
## 7. 阶段四：4D轨迹时空回放系统

### 7.1 目标

将Python调度算法生成的多架飞机4D轨迹以CZML格式加载入CesiumJS，实现可拖动时间轴的高保真动态回放，支持追踪单架飞机、速度调节、轨迹路径着色。

### 7.2 Python后端：CZML生成器

```python
# python/generate_czml.py
import json
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

def build_czml(
    flights: List[dict],
    start_time: datetime,
    playback_multiplier: int = 60
) -> list:
    """
    参数:
        flights: 航班列表，每项格式：
            {
              "id": "UAL123",
              "callsign": "United 123",
              "type": "B738",
              "waypoints": [
                  (offset_sec, lon, lat, alt_m),  # alt_m: 海拔高度（米）
                  ...
              ]
            }
        start_time: 仿真开始的UTC时间
        playback_multiplier: 时间加速倍率（60=1秒代表1分钟）
    """
    end_time = start_time + timedelta(
        seconds=max(wpt[0] for f in flights for wpt in f['waypoints'])
    )

    document = {
        "id": "document",
        "name": "AeroViz-4D Trajectories",
        "version": "1.0",
        "clock": {
            "interval": f"{start_time.isoformat()}/{end_time.isoformat()}",
            "currentTime": start_time.isoformat(),
            "multiplier": playback_multiplier,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER"
        }
    }

    entities = [document]
    colors = [
        [1.0, 0.5, 0.0],  # 橙色
        [0.0, 0.8, 1.0],  # 青色
        [0.8, 1.0, 0.0],  # 黄绿色
        [1.0, 0.2, 0.8],  # 粉色
    ]

    for i, flight in enumerate(flights):
        color = colors[i % len(colors)]
        epoch_iso = start_time.isoformat()

        # 构建 cartographicDegrees 数组：[秒偏移, 经度, 纬度, 高度, ...]
        cart_degrees = []
        for (offset_sec, lon, lat, alt_m) in flight['waypoints']:
            cart_degrees.extend([offset_sec, lon, lat, alt_m])

        # 构建轨迹路径（尾迹线）
        trail_positions = []
        for (offset_sec, lon, lat, alt_m) in flight['waypoints']:
            trail_positions.extend([offset_sec, lon, lat, alt_m])

        entity = {
            "id": flight['id'],
            "name": flight['callsign'],
            "description": f"<b>{flight['callsign']}</b><br/>机型: {flight['type']}",
            # 3D飞机模型
            "model": {
                "gltf": "/models/aircraft.glb",
                "scale": 3.0,
                "minimumPixelSize": 32,
                "maximumScale": 20000,
                "runAnimations": True
            },
            # 位置时间序列（线性插值）
            "position": {
                "epoch": epoch_iso,
                "cartographicDegrees": cart_degrees,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 3,
                "forwardExtrapolationType": "HOLD"
            },
            # 自动从速度向量计算朝向
            "orientation": {
                "velocityReference": f"#{flight['id']}.position"
            },
            # 轨迹尾迹线
            "path": {
                "show": True,
                "leadTime": 0,
                "trailTime": 300,  # 显示过去300秒轨迹
                "width": 2,
                "material": {
                    "solidColor": {
                        "color": {
                            "rgba": [int(c*255) for c in color] + [200]
                        }
                    }
                }
            },
            # 标签
            "label": {
                "text": flight['callsign'],
                "font": "12px sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "verticalOrigin": "BOTTOM",
                "pixelOffset": {"cartesian2": [0, -30]}
            }
        }
        entities.append(entity)

    return entities


# 示例：生成Mock数据用于前端联调
if __name__ == '__main__':
    start = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)

    mock_flights = [
        {
            "id": "UAL123", "callsign": "United 123", "type": "B738",
            "waypoints": [
                (0,    -119.10, 50.20, 5500),
                (180,  -119.20, 50.10, 4800),
                (360,  -119.30, 50.00, 4000),
                (540,  -119.36, 49.97, 3200),
                (660,  -119.38, 49.96, 2500),
                (780,  -119.385, 49.957, 1800),
                (900,  -119.390, 49.955, 1200),
            ]
        },
        {
            "id": "WJA456", "callsign": "WestJet 456", "type": "B737",
            "waypoints": [
                (0,    -119.05, 50.30, 6000),
                (240,  -119.15, 50.15, 5200),
                (480,  -119.28, 50.02, 4200),
                (720,  -119.35, 49.98, 3300),
                (900,  -119.37, 49.96, 2600),
                (1020, -119.382, 49.958, 1900),
                (1140, -119.390, 49.955, 1200),
            ]
        },
    ]

    czml_data = build_czml(mock_flights, start, playback_multiplier=60)
    
    output_path = '../public/data/trajectories.czml'
    with open(output_path, 'w') as f:
        json.dump(czml_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已生成 {len(mock_flights)} 架航班的CZML轨迹文件")
    print(f"  输出路径: {output_path}")
```

### 7.3 前端CZML加载Hook

```typescript
// src/hooks/useCzmlLoader.ts
import { useEffect, useState } from 'react';
import * as Cesium from 'cesium';

interface CzmlState {
  isLoaded: boolean;
  flightIds: string[];
  error: string | null;
}

export function useCzmlLoader(
  viewer: Cesium.Viewer | null,
  czmlUrl: string
): CzmlState {
  const [state, setState] = useState<CzmlState>({
    isLoaded: false, flightIds: [], error: null,
  });

  useEffect(() => {
    if (!viewer) return;

    let dataSource: Cesium.CzmlDataSource;

    Cesium.CzmlDataSource.load(czmlUrl)
      .then((ds) => {
        dataSource = ds;
        viewer.dataSources.add(ds);

        // 将Viewer时钟与CZML时间区间同步
        const clock = viewer.clock;
        const interval = ds.clock.startTime;
        clock.startTime = ds.clock.startTime.clone();
        clock.stopTime = ds.clock.stopTime.clone();
        clock.currentTime = ds.clock.startTime.clone();
        clock.clockRange = Cesium.ClockRange.LOOP_STOP;
        clock.multiplier = 60; // 60倍速播放
        clock.shouldAnimate = true;

        // 同步时间轴显示范围
        viewer.timeline.zoomTo(clock.startTime, clock.stopTime);

        // 收集所有航班ID
        const ids = ds.entities.values
          .filter(e => e.id !== 'document')
          .map(e => e.id);

        // 默认追踪第一架航班
        if (ids.length > 0) {
          viewer.trackedEntity = ds.entities.getById(ids[0]) ?? undefined;
        }

        setState({ isLoaded: true, flightIds: ids, error: null });
      })
      .catch((err) => {
        setState({ isLoaded: false, flightIds: [], error: err.message });
      });

    return () => {
      if (dataSource) viewer.dataSources.remove(dataSource, true);
    };
  }, [viewer, czmlUrl]);

  return state;
}
```

<!-- LANG:EN -->
## 7. Phase 4: 4D Trajectory Spatiotemporal Playback System

### 7.1 Objective

Load multi-aircraft 4D trajectories generated by the Python scheduling algorithm in CZML format into CesiumJS, enabling high-fidelity dynamic playback with a draggable timeline. Support single-aircraft tracking, speed control, and trajectory path coloring.

### 7.2 Python Backend: CZML Generator

```python
# python/generate_czml.py
import json
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

def build_czml(
    flights: List[dict],
    start_time: datetime,
    playback_multiplier: int = 60
) -> list:
    """
    Args:
        flights: list of flight dicts, each with format:
            {
              "id": "UAL123",
              "callsign": "United 123",
              "type": "B738",
              "waypoints": [
                  (offset_sec, lon, lat, alt_m),
                  ...
              ]
            }
        start_time: simulation start time in UTC
        playback_multiplier: time acceleration (60 = 1 second represents 1 minute)
    """
    end_time = start_time + timedelta(
        seconds=max(wpt[0] for f in flights for wpt in f['waypoints'])
    )

    document = {
        "id": "document",
        "name": "AeroViz-4D Trajectories",
        "version": "1.0",
        "clock": {
            "interval": f"{start_time.isoformat()}/{end_time.isoformat()}",
            "currentTime": start_time.isoformat(),
            "multiplier": playback_multiplier,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER"
        }
    }

    entities = [document]
    colors = [
        [1.0, 0.5, 0.0],  # orange
        [0.0, 0.8, 1.0],  # cyan
        [0.8, 1.0, 0.0],  # yellow-green
        [1.0, 0.2, 0.8],  # pink
    ]

    for i, flight in enumerate(flights):
        color = colors[i % len(colors)]
        epoch_iso = start_time.isoformat()

        # Build cartographicDegrees array: [offsetSec, lon, lat, alt, ...]
        cart_degrees = []
        for (offset_sec, lon, lat, alt_m) in flight['waypoints']:
            cart_degrees.extend([offset_sec, lon, lat, alt_m])

        entity = {
            "id": flight['id'],
            "name": flight['callsign'],
            "description": f"<b>{flight['callsign']}</b><br/>Type: {flight['type']}",
            # 3D aircraft model
            "model": {
                "gltf": "/models/aircraft.glb",
                "scale": 3.0,
                "minimumPixelSize": 32,
                "maximumScale": 20000,
                "runAnimations": True
            },
            # Position time series (Lagrange interpolation)
            "position": {
                "epoch": epoch_iso,
                "cartographicDegrees": cart_degrees,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 3,
                "forwardExtrapolationType": "HOLD"
            },
            # Auto-compute heading from velocity vector
            "orientation": {
                "velocityReference": f"#{flight['id']}.position"
            },
            # Trail path
            "path": {
                "show": True,
                "leadTime": 0,
                "trailTime": 300,  # show last 300 seconds of trail
                "width": 2,
                "material": {
                    "solidColor": {
                        "color": {
                            "rgba": [int(c*255) for c in color] + [200]
                        }
                    }
                }
            },
            # Callsign label
            "label": {
                "text": flight['callsign'],
                "font": "12px sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "verticalOrigin": "BOTTOM",
                "pixelOffset": {"cartesian2": [0, -30]}
            }
        }
        entities.append(entity)

    return entities


# Example: generate mock data for frontend integration testing
if __name__ == '__main__':
    start = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)

    mock_flights = [
        {
            "id": "UAL123", "callsign": "United 123", "type": "B738",
            "waypoints": [
                (0,    -119.10, 50.20, 5500),
                (180,  -119.20, 50.10, 4800),
                (360,  -119.30, 50.00, 4000),
                (540,  -119.36, 49.97, 3200),
                (660,  -119.38, 49.96, 2500),
                (780,  -119.385, 49.957, 1800),
                (900,  -119.390, 49.955, 1200),
            ]
        },
        {
            "id": "WJA456", "callsign": "WestJet 456", "type": "B737",
            "waypoints": [
                (0,    -119.05, 50.30, 6000),
                (240,  -119.15, 50.15, 5200),
                (480,  -119.28, 50.02, 4200),
                (720,  -119.35, 49.98, 3300),
                (900,  -119.37, 49.96, 2600),
                (1020, -119.382, 49.958, 1900),
                (1140, -119.390, 49.955, 1200),
            ]
        },
    ]

    czml_data = build_czml(mock_flights, start, playback_multiplier=60)
    
    output_path = '../public/data/trajectories.czml'
    with open(output_path, 'w') as f:
        json.dump(czml_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated CZML for {len(mock_flights)} flights")
    print(f"  Output: {output_path}")
```

### 7.3 Frontend CZML Loading Hook

```typescript
// src/hooks/useCzmlLoader.ts
import { useEffect, useState } from 'react';
import * as Cesium from 'cesium';

interface CzmlState {
  isLoaded: boolean;
  flightIds: string[];
  error: string | null;
}

export function useCzmlLoader(
  viewer: Cesium.Viewer | null,
  czmlUrl: string
): CzmlState {
  const [state, setState] = useState<CzmlState>({
    isLoaded: false, flightIds: [], error: null,
  });

  useEffect(() => {
    if (!viewer) return;

    let dataSource: Cesium.CzmlDataSource;

    Cesium.CzmlDataSource.load(czmlUrl)
      .then((ds) => {
        dataSource = ds;
        viewer.dataSources.add(ds);

        // Synchronize Viewer clock with CZML time interval
        const clock = viewer.clock;
        clock.startTime = ds.clock.startTime.clone();
        clock.stopTime = ds.clock.stopTime.clone();
        clock.currentTime = ds.clock.startTime.clone();
        clock.clockRange = Cesium.ClockRange.LOOP_STOP;
        clock.multiplier = 60;
        clock.shouldAnimate = true;

        // Sync timeline display range
        viewer.timeline.zoomTo(clock.startTime, clock.stopTime);

        // Collect all flight IDs
        const ids = ds.entities.values
          .filter(e => e.id !== 'document')
          .map(e => e.id);

        // Default: track the first flight
        if (ids.length > 0) {
          viewer.trackedEntity = ds.entities.getById(ids[0]) ?? undefined;
        }

        setState({ isLoaded: true, flightIds: ids, error: null });
      })
      .catch((err) => {
        setState({ isLoaded: false, flightIds: [], error: err.message });
      });

    return () => {
      if (dataSource) viewer.dataSources.remove(dataSource, true);
    };
  }, [viewer, czmlUrl]);

  return state;
}
```

---

<!-- LANG:ZH -->
## 8. 阶段五：控制面板与UI集成

### 8.1 全局状态管理

```typescript
// src/context/AppContext.tsx
import { createContext, useContext, useState, ReactNode } from 'react';
import * as Cesium from 'cesium';

interface AppState {
  viewer: Cesium.Viewer | null;
  setViewer: (v: Cesium.Viewer) => void;
  selectedFlightId: string | null;
  setSelectedFlightId: (id: string | null) => void;
  layers: {
    terrain: boolean;
    runways: boolean;
    waypoints: boolean;
    ocsSurfaces: boolean;
    trajectories: boolean;
  };
  toggleLayer: (key: keyof AppState['layers']) => void;
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [viewer, setViewer] = useState<Cesium.Viewer | null>(null);
  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(60);
  const [layers, setLayers] = useState({
    terrain: true,
    runways: true,
    waypoints: true,
    ocsSurfaces: true,
    trajectories: true,
  });

  const toggleLayer = (key: keyof typeof layers) => {
    setLayers(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <AppContext.Provider value={{
      viewer, setViewer,
      selectedFlightId, setSelectedFlightId,
      layers, toggleLayer,
      playbackSpeed, setPlaybackSpeed,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export const useApp = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
};
```

### 8.2 控制面板组件

```typescript
// src/components/ControlPanel.tsx
import { useApp } from '../context/AppContext';
import * as Cesium from 'cesium';

const SPEED_OPTIONS = [
  { label: '1×', value: 1 },
  { label: '10×', value: 10 },
  { label: '30×', value: 30 },
  { label: '60×', value: 60 },
  { label: '120×', value: 120 },
];

export default function ControlPanel() {
  const { viewer, layers, toggleLayer, playbackSpeed, setPlaybackSpeed } = useApp();

  const handleSpeedChange = (speed: number) => {
    if (!viewer) return;
    viewer.clock.multiplier = speed;
    setPlaybackSpeed(speed);
  };

  const handlePlayPause = () => {
    if (!viewer) return;
    viewer.clock.shouldAnimate = !viewer.clock.shouldAnimate;
  };

  const handleReset = () => {
    if (!viewer) return;
    viewer.clock.currentTime = viewer.clock.startTime.clone();
    viewer.clock.shouldAnimate = false;
  };

  return (
    <div className="control-panel">
      <h3>AeroViz-4D</h3>

      {/* 播放控制 */}
      <section>
        <button onClick={handlePlayPause}>播放/暂停</button>
        <button onClick={handleReset}>重置</button>
        <div className="speed-buttons">
          {SPEED_OPTIONS.map(opt => (
            <button
              key={opt.value}
              className={playbackSpeed === opt.value ? 'active' : ''}
              onClick={() => handleSpeedChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </section>

      {/* 图层开关 */}
      <section>
        <h4>图层</h4>
        {(Object.keys(layers) as Array<keyof typeof layers>).map(key => (
          <label key={key}>
            <input
              type="checkbox"
              checked={layers[key]}
              onChange={() => toggleLayer(key)}
            />
            {key}
          </label>
        ))}
      </section>
    </div>
  );
}
```

<!-- LANG:EN -->
## 8. Phase 5: Control Panel & UI Integration

### 8.1 Global State Management

```typescript
// src/context/AppContext.tsx
import { createContext, useContext, useState, ReactNode } from 'react';
import * as Cesium from 'cesium';

interface AppState {
  viewer: Cesium.Viewer | null;
  setViewer: (v: Cesium.Viewer) => void;
  selectedFlightId: string | null;
  setSelectedFlightId: (id: string | null) => void;
  layers: {
    terrain: boolean;
    runways: boolean;
    waypoints: boolean;
    ocsSurfaces: boolean;
    trajectories: boolean;
  };
  toggleLayer: (key: keyof AppState['layers']) => void;
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [viewer, setViewer] = useState<Cesium.Viewer | null>(null);
  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(60);
  const [layers, setLayers] = useState({
    terrain: true,
    runways: true,
    waypoints: true,
    ocsSurfaces: true,
    trajectories: true,
  });

  const toggleLayer = (key: keyof typeof layers) => {
    setLayers(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <AppContext.Provider value={{
      viewer, setViewer,
      selectedFlightId, setSelectedFlightId,
      layers, toggleLayer,
      playbackSpeed, setPlaybackSpeed,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export const useApp = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
};
```

### 8.2 Control Panel Component

```typescript
// src/components/ControlPanel.tsx
import { useApp } from '../context/AppContext';

const SPEED_OPTIONS = [
  { label: '1×', value: 1 },
  { label: '10×', value: 10 },
  { label: '30×', value: 30 },
  { label: '60×', value: 60 },
  { label: '120×', value: 120 },
];

export default function ControlPanel() {
  const { viewer, layers, toggleLayer, playbackSpeed, setPlaybackSpeed } = useApp();

  const handleSpeedChange = (speed: number) => {
    if (!viewer) return;
    viewer.clock.multiplier = speed;
    setPlaybackSpeed(speed);
  };

  const handlePlayPause = () => {
    if (!viewer) return;
    viewer.clock.shouldAnimate = !viewer.clock.shouldAnimate;
  };

  const handleReset = () => {
    if (!viewer) return;
    viewer.clock.currentTime = viewer.clock.startTime.clone();
    viewer.clock.shouldAnimate = false;
  };

  return (
    <div className="control-panel">
      <h3>AeroViz-4D</h3>

      {/* Playback controls */}
      <section>
        <button onClick={handlePlayPause}>Play / Pause</button>
        <button onClick={handleReset}>Reset</button>
        <div className="speed-buttons">
          {SPEED_OPTIONS.map(opt => (
            <button
              key={opt.value}
              className={playbackSpeed === opt.value ? 'active' : ''}
              onClick={() => handleSpeedChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </section>

      {/* Layer toggles */}
      <section>
        <h4>Layers</h4>
        {(Object.keys(layers) as Array<keyof typeof layers>).map(key => (
          <label key={key}>
            <input
              type="checkbox"
              checked={layers[key]}
              onChange={() => toggleLayer(key)}
            />
            {key}
          </label>
        ))}
      </section>
    </div>
  );
}
```

---

<!-- LANG:ZH -->
## 9. 数据接口规范

### 9.1 runway.geojson 规范

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1,lat1],[lon2,lat2],[lon3,lat3],[lon4,lat4],[lon1,lat1]]]
      },
      "properties": {
        "airport_ident": "CYLW",
        "le_ident": "34",
        "he_ident": "16",
        "length_ft": 8000,
        "width_ft": 150,
        "surface": "ASP",
        "lighted": 1,
        "le_elevation_ft": 1421,
        "he_elevation_ft": 1398
      }
    }
  ]
}
```

### 9.2 waypoints.geojson 规范

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [lon, lat, alt_m]
      },
      "properties": {
        "name": "KEVOL",
        "type": "IAF",
        "min_alt_ft": 9000,
        "procedure": "RNAV(GNSS) Z RWY 34",
        "sequence": 1
      }
    }
  ]
}
```

### 9.3 Python–前端数据契约摘要

| 数据文件 | 生产方 | 消费方 | 更新频率 |
|---------|--------|--------|---------|
| `runway.geojson` | Python预处理脚本（一次性） | React前端 | 部署时 |
| `waypoints.geojson` | Python预处理脚本（一次性） | React前端 | 部署时 |
| `ocs_surfaces.geojson` | Python算法模块 | React前端 | 研究迭代时 |
| `trajectories.czml` | Python调度算法（核心输出） | React前端 | 每次算法运行后 |

<!-- LANG:EN -->
## 9. Data Interface Specifications

### 9.1 runway.geojson Specification

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1,lat1],[lon2,lat2],[lon3,lat3],[lon4,lat4],[lon1,lat1]]]
      },
      "properties": {
        "airport_ident": "CYLW",
        "le_ident": "34",
        "he_ident": "16",
        "length_ft": 8000,
        "width_ft": 150,
        "surface": "ASP",
        "lighted": 1,
        "le_elevation_ft": 1421,
        "he_elevation_ft": 1398
      }
    }
  ]
}
```

### 9.2 waypoints.geojson Specification

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [lon, lat, alt_m]
      },
      "properties": {
        "name": "KEVOL",
        "type": "IAF",
        "min_alt_ft": 9000,
        "procedure": "RNAV(GNSS) Z RWY 34",
        "sequence": 1
      }
    }
  ]
}
```

### 9.3 Python–Frontend Data Contract Summary

| Data File | Producer | Consumer | Update Frequency |
|-----------|----------|----------|-----------------|
| `runway.geojson` | Python preprocessing script (one-time) | React frontend | At deployment |
| `waypoints.geojson` | Python preprocessing script (one-time) | React frontend | At deployment |
| `ocs_surfaces.geojson` | Python algorithm module | React frontend | Each research iteration |
| `trajectories.czml` | Python scheduling algorithm (core output) | React frontend | After each algorithm run |

---

<!-- LANG:ZH -->
## 10. 开发执行路线图

### 10.1 推荐执行顺序

```
Week 1  ────────────────────────────────────────────────────────
  Day 1-2  阶段1: Vite+React+CesiumJS 环境搭建，验证地球渲染正常
  Day 3    阶段2: 加载World Terrain，验证山脉高程显示
  Day 4    阶段2: 处理OurAirports数据，绘制贴地跑道多边形
  Day 5    阶段3: 绘制航路点标记和进近路径连线

Week 2  ────────────────────────────────────────────────────────
  Day 6-7  阶段3: 实现OCS保护面几何计算和3D渲染
  Day 8    阶段4: Python生成Mock CZML，前端加载测试
  Day 9    阶段4: 时间轴联调，时钟同步，追踪视角
  Day 10   阶段5: 控制面板UI，图层开关，播放速度

Week 3+  ────────────────────────────────────────────────────────
  接入真实调度算法输出 → 替换Mock CZML → 论文截图与演示
```

### 10.2 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Viewer报 `AccessToken` 错误 | Ion Token未配置 | 前往 cesium.com/ion/tokens 申请免费Token |
| 跑道多边形悬空 | `clampToGround` 未设置 | 在 `GeoJsonDataSource.load()` 选项中设置 `clampToGround: true` |
| CZML加载后飞机不动 | Clock未与DataSource同步 | 手动将 `viewer.clock.startTime` 设为 CZML 的时间起点 |
| 3D模型不显示 | `.glb` 路径错误或未放入 `public/` | 确认模型路径相对于项目根目录，Vite静态资源需在 `public/` 下 |
| 地形加载缓慢 | 网络延迟 | 开发阶段可临时禁用terrain使用平面地球，联调完成后再开启 |
| OCS面穿透地形 | 高度计算基准不一致 | 确认所有高度值均为MSL（平均海平面）高度（米），非AGL |

### 10.3 后续开发计划：跑道图例（Legend）

**目标**
- 在不增加用户学习成本的前提下，明确区分：
  - `runway_surface`（跑道本体，深灰）
  - `landing_zone`（位移阈值后的可降落区域，淡绿色）

**实现范围（MVP）**
- 在主视图右上角增加固定图例卡片（不参与地图拾取）。
- 图例与实际图层颜色保持一一对应。
- 支持中英文标签（默认跟随界面语言，未接入i18n前可先固定英文）。

**推荐执行顺序**
```
Day 11  UI: 新建 Legend 组件（颜色块 + 文本），挂载到 App 顶层
Day 12  状态联动: 与 layers.runways 开关联动（关闭跑道图层时图例置灰）
Day 13  可用性优化: 移动端适配、对比度检查、截图审阅
```

**验收标准**
- 首次打开页面，用户可在 3 秒内理解两种跑道区域含义。
- 图例颜色与 `useRunwayLayer.ts` 实际渲染颜色一致。
- 1920x1080 与常见移动端宽度下均不遮挡核心控件。

### 10.4 后续开发计划：METAR/QNH 压力高度校正

**目标**
- 在保持原始轨迹可追溯的前提下，引入基于 METAR/QNH 的压力高度改正。
- 降低低空进近阶段 barometric altitude 与机场场高之间的系统偏差。
- 将校正过程参数化并写入元数据，满足研究复现与审计要求。

**实现范围（MVP）**
- 在 OpenSky 轨迹预处理模块增加高度模式：
  - `raw`：不改动原始高度（研究基线）
  - `metar_qnh`：按时间匹配的 METAR QNH 进行压力高度改正
- 按机场与时刻抓取 METAR（至少覆盖：报告时间、QNH、风场可选）。
- 对每条航迹逐点计算修正高度，并保留原始高度字段。
- 在导出 JSON 中写入：数据源、QNH、改正公式版本、质量标记。

**推荐执行顺序**
```
Day 14  数据接入: 建立 METAR 拉取与缓存层（按机场+小时索引）
Day 15  算法实现: 按时间对齐 QNH，完成 barometric altitude -> corrected altitude
Day 16  质量控制: 缺测回退策略、异常QNH阈值检查、元数据输出
Day 17  结果评估: 统计触地点高度误差分布，生成对比图（raw vs corrected）
```

**验收标准**
- `raw` 模式输出与原始 OpenSky 高度一致（无隐式改动）。
- `metar_qnh` 模式下，每条航迹都包含可追溯校正元数据。
- 含落地样本的航迹中，触地点高度中位误差相对场高显著下降。
- 缺少 METAR 时自动回退 `raw`，并在质量标记中注明原因。

<!-- LANG:EN -->
## 10. Development Execution Roadmap

### 10.1 Recommended Execution Order

```
Week 1  ────────────────────────────────────────────────────────
  Day 1-2  Phase 1: Vite+React+CesiumJS setup, verify globe renders
  Day 3    Phase 2: Load World Terrain, verify mountain elevation display
  Day 4    Phase 2: Process OurAirports data, render ground-clamped runway polygons
  Day 5    Phase 3: Draw waypoint markers and approach path polylines

Week 2  ────────────────────────────────────────────────────────
  Day 6-7  Phase 3: Implement OCS geometry calculation and 3D rendering
  Day 8    Phase 4: Generate mock CZML with Python, load in frontend
  Day 9    Phase 4: Timeline sync, clock synchronization, camera tracking
  Day 10   Phase 5: Control panel UI, layer toggles, playback speed

Week 3+  ────────────────────────────────────────────────────────
  Integrate real scheduling algorithm output → replace mock CZML
  → capture thesis screenshots and live demo
```

### 10.2 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|---------|
| Viewer throws `AccessToken` error | Ion token not configured | Get a free token at cesium.com/ion/tokens |
| Runway polygons float above terrain | `clampToGround` not set | Add `clampToGround: true` to `GeoJsonDataSource.load()` options |
| Aircraft don't move after CZML loads | Clock not synced to DataSource | Manually set `viewer.clock.startTime` to CZML start time |
| 3D model not visible | `.glb` path wrong or not in `public/` | Confirm the model path is under `public/`; Vite serves static assets from there |
| Terrain loads slowly | Network latency | Disable terrain temporarily during development; re-enable for final demo |
| OCS surfaces clip through terrain | Inconsistent altitude datum | Confirm all altitudes are MSL (Mean Sea Level) in meters, not AGL |

### 10.3 Follow-up Plan: Runway Legend

**Goal**
- Make the two runway zones immediately understandable with minimal cognitive load:
  - `runway_surface` (dark gray)
  - `landing_zone` (light green)

**MVP Scope**
- Add a fixed legend card at the top-right of the main view (non-pickable overlay).
- Keep legend colors strictly aligned with actual layer rendering.
- Support bilingual labels (follow UI language; use English-only as interim fallback).

**Recommended Execution Order**
```
Day 11  UI: Create a Legend component (color swatches + labels), mount at App root
Day 12  State sync: Link legend state to layers.runways toggle (dim when hidden)
Day 13  Usability polish: mobile layout, contrast checks, screenshot review
```

**Acceptance Criteria**
- First-time users can understand both runway zones within 3 seconds.
- Legend colors match runtime colors in `useRunwayLayer.ts`.
- No overlap with primary controls at 1920x1080 and common mobile widths.

### 10.4 Follow-up Plan: METAR/QNH Pressure Altitude Correction

**Goal**
- Introduce METAR/QNH-based pressure altitude correction while preserving raw trajectory traceability.
- Reduce systematic low-altitude bias between OpenSky barometric altitude and airport field elevation.
- Make correction parameters explicit and auditable for research reproducibility.

**MVP Scope**
- Add altitude handling modes in trajectory preprocessing:
  - `raw`: keep original altitude unchanged (research baseline)
  - `metar_qnh`: apply pressure-altitude correction using time-aligned METAR QNH
- Fetch METAR by airport and timestamp (minimum fields: report time, QNH).
- Compute corrected altitude per waypoint while retaining raw altitude values.
- Export correction metadata: source, QNH, formula version, and quality flags.

**Recommended Execution Order**
```
Day 14  Data layer: implement METAR fetch + cache (airport + hourly index)
Day 15  Core algorithm: align waypoint time with QNH and compute corrected altitude
Day 16  QA rules: fallback behavior, abnormal-QNH thresholds, metadata output
Day 17  Evaluation: touchdown altitude error distribution (raw vs corrected)
```

**Acceptance Criteria**
- `raw` mode remains bitwise-consistent with original OpenSky altitude.
- `metar_qnh` mode outputs per-flight auditable correction metadata.
- For flights with touchdown samples, median touchdown altitude error vs field elevation is significantly reduced.
- When METAR is unavailable, pipeline falls back to `raw` with explicit quality flag.

---

<!-- LANG:ZH -->
## 11. Python依赖与环境配置

```
# python/requirements.txt
openap>=1.3.0          # 航空器性能模型
pandas>=2.0.0          # 数据处理（OurAirports CSV）
numpy>=1.24.0          # 数值计算（坐标变换）
scipy>=1.10.0          # 插值与优化
fastapi>=0.100.0       # 可选：RESTful API服务
uvicorn>=0.23.0        # 可选：FastAPI服务器
pyproj>=3.5.0          # 大地坐标投影变换
shapely>=2.0.0         # 几何计算（保护区多边形）
pulp>=2.7.0            # MILP求解器（调度算法）
```

安装：
```bash
cd python
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

<!-- LANG:EN -->
## 11. Python Dependencies & Environment Setup

```
# python/requirements.txt
openap>=1.3.0          # aircraft performance model
pandas>=2.0.0          # data processing (OurAirports CSV)
numpy>=1.24.0          # numerical computation (coordinate transforms)
scipy>=1.10.0          # interpolation and optimization
fastapi>=0.100.0       # optional: RESTful API server
uvicorn>=0.23.0        # optional: FastAPI ASGI server
pyproj>=3.5.0          # geodetic coordinate projection
shapely>=2.0.0         # geometric computation (protection area polygons)
pulp>=2.7.0            # MILP solver (scheduling algorithm)
```

Installation:
```bash
cd python
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

<!-- LANG:ZH -->
## 12. 参考资料

| 资料 | 链接/来源 |
|------|---------|
| CesiumJS官方文档 | https://cesium.com/learn/cesiumjs/ref-doc/ |
| CZML格式规范 | https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Guide |
| OurAirports数据集 | https://ourairports.com/data/ |
| OpenAP性能库 | https://openap.dev/ |
| ICAO PANS-OPS Doc 8168 | ICAO官方出版物（需购买或机构授权访问） |
| FAA CIFP数据 | https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/ |
| Nav Canada数据 | https://www.navcanada.ca/en/aeronautical-information/ |
| Vite配置文档 | https://vitejs.dev/config/ |

<!-- LANG:EN -->
## 12. References

| Resource | Link / Source |
|----------|--------------|
| CesiumJS Official Docs | https://cesium.com/learn/cesiumjs/ref-doc/ |
| CZML Format Specification | https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Guide |
| OurAirports Dataset | https://ourairports.com/data/ |
| OpenAP Performance Library | https://openap.dev/ |
| ICAO PANS-OPS Doc 8168 | ICAO official publication (purchase or institutional access required) |
| FAA CIFP Data | https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/ |
| Nav Canada AIP | https://www.navcanada.ca/en/aeronautical-information/ |
| Vite Configuration Docs | https://vitejs.dev/config/ |

---

*文档结束 / End of Document*
