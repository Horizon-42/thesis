# 三自由度飞行动力学模型（3-DOF Point-Mass Flight Dynamics）整理与推导

---

# 1. 模型背景

本文整理两种常见的三自由度点质量飞行动力学模型：

1. 图中模型（较完整的空气动力学模型）
2. 简化模型（以 thrust、bank angle、load factor 为控制输入）

二者都属于：

\[
\text{3-DOF Point-Mass Model}
\]

即：

- 飞机被视为一个“质点”；
- 不考虑刚体姿态动力学；
- 只研究飞机轨迹与速度变化；
- 研究位置、速度和飞行方向的变化。

---

# 2. 坐标与变量定义

## 2.1 状态变量（State Variables）

采用状态向量：

\[
\mathbf{x}
=
\begin{bmatrix}
X\\
Y\\
h\\
V\\
\psi\\
\gamma\\
m
\end{bmatrix}
\]

其中：

| 变量 | 含义 |
|---|---|
| \(X,Y\) | 水平位置坐标 |
| \(h\) | 高度（height） |
| \(V\) | 速度大小（speed magnitude） |
| \(\psi\) | 航向角（heading angle） |
| \(\gamma\) | 飞行路径角 / 爬升角（flight path angle） |
| \(m\) | 质量（mass） |

---

## 2.2 控制输入（Control Inputs）

### 图中模型的控制输入

\[
\mathbf{u}
=
\begin{bmatrix}
T\\
\phi\\
\alpha
\end{bmatrix}
\]

其中：

| 变量 | 含义 |
|---|---|
| \(T\) | 推力（thrust） |
| \(\phi\) | 倾侧角 / 滚转角（bank angle / roll angle） |
| \(\alpha\) | 攻角（angle of attack） |

---

### 简化模型（你的模型）的控制输入

\[
\mathbf{u}
=
\begin{bmatrix}
T\\
\mu\\
n
\end{bmatrix}
\]

其中：

| 变量 | 含义 |
|---|---|
| \(T\) | 推力（thrust） |
| \(\mu\) | 倾侧角（bank angle） |
| \(n\) | 载荷因子（load factor） |

载荷因子定义为：

\[
n=\frac{L}{mg}
\]

因此：

\[
L=nmg
\]

其中：

- \(L\)：升力（lift）
- \(g\)：重力加速度（gravitational acceleration）

---

# 3. 图中模型（完整形式）

## 3.1 模型方程

\[
\frac{d}{dt}
\begin{bmatrix}
X\\
Y\\
h\\
V\\
\psi\\
\gamma\\
m
\end{bmatrix}
=
\begin{bmatrix}
V\cos\psi\cos\gamma+\omega_x\\[6pt]
V\sin\psi\cos\gamma+\omega_y\\[6pt]
V\sin\gamma+\omega_z\\[6pt]
\dfrac{T\cos\alpha-D}{m}-g\sin\gamma\\[10pt]
\dfrac{(L+T\sin\alpha)\sin\phi}{mV\cos\gamma}\\[10pt]
\dfrac{(L+T\sin\alpha)\cos\phi-mg\cos\gamma}{mV}\\[10pt]
-\dfrac{c_T}{g}T
\end{bmatrix}
\]

---

## 3.2 额外变量说明

| 变量 | 含义 |
|---|---|
| \(L\) | 升力（lift） |
| \(D\) | 阻力（drag） |
| \(\omega_x,\omega_y,\omega_z\) | 风速分量 |
| \(c_T\) | 燃油消耗相关参数 |
| \(g\) | 重力加速度 |

---

# 4. 简化模型（你的模型）

## 4.1 模型方程

你的模型做了两个关键假设：

1. 推力方向基本沿速度方向：

\[
\alpha \approx 0
\]

因此：

\[
\cos\alpha \approx 1
\]

\[
\sin\alpha \approx 0
\]

2. 用载荷因子直接控制升力：

\[
L=nmg
\]

于是得到简化模型：

\[
\frac{d}{dt}
\begin{bmatrix}
X\\
Y\\
h\\
V\\
\psi\\
\gamma\\
m
\end{bmatrix}
=
\begin{bmatrix}
V\cos\psi\cos\gamma\\[6pt]
V\sin\psi\cos\gamma\\[6pt]
V\sin\gamma\\[6pt]
\dfrac{T-D}{m}-g\sin\gamma\\[10pt]
\dfrac{g}{V\cos\gamma}n\sin\mu\\[10pt]
\dfrac{g}{V}(n\cos\mu-\cos\gamma)\\[10pt]
0
\end{bmatrix}
\]

---

# 5. 两种模型的核心区别

| 项目 | 图中模型 | 简化模型 |
|---|---|---|
| 推力方向 | 可偏离速度方向 | 假设沿速度方向 |
| 升力来源 | 由 \(\alpha\) 间接决定 | 直接由 \(n\) 决定 |
| 推力法向分量 | 保留 \(T\sin\alpha\) | 忽略 |
| 控制输入 | \(T,\phi,\alpha\) | \(T,\mu,n\) |
| 空气动力细节 | 更多 | 更简化 |
| 模型用途 | 飞机性能分析 | 快速轨迹仿真 / 制导 |

---

# 6. 位置方程推导

飞机速度大小为：

\[
V
\]

飞行路径角为：

\[
\gamma
\]

则：

- 水平速度：

\[
V\cos\gamma
\]

- 竖直速度：

\[
V\sin\gamma
\]

水平速度再按航向角 \(\psi\) 分解：

\[
\dot X=V\cos\gamma\cos\psi
\]

\[
\dot Y=V\cos\gamma\sin\psi
\]

高度变化率：

\[
\dot h=V\sin\gamma
\]

---

# 7. 角速度与加速度

## 7.1 为什么“速度 × 角速度 = 加速度”

设：

- 速度大小：

\[
V
\]

- 方向变化角速度：

\[
\omega
\]

则方向变化造成的加速度为：

\[
a=V\omega
\]

原因如下。

在很小时间 \(\Delta t\) 内，方向转过：

\[
\Delta\theta
\]

则速度向量变化量近似为：

\[
\Delta V \approx V\Delta\theta
\]

加速度定义：

\[
a=\frac{\Delta V}{\Delta t}
\]

代入：

\[
a\approx\frac{V\Delta\theta}{\Delta t}
\]

因为：

\[
\dot\theta=\frac{d\theta}{dt}
\]

所以：

\[
a=V\dot\theta
\]

即：

\[
\boxed{a=V\times\text{角速度}}
\]

---

# 8. \(\dot\psi\) 的推导

## 8.1 航向角的物理意义

\[
\psi
\]

表示飞机在水平面中的转向角。

其变化率：

\[
\dot\psi=\frac{d\psi}{dt}
\]

表示飞机水平转向有多快。

---

## 8.2 为什么有 \(V\cos\gamma\)

真正参与水平转弯的是“水平速度”：

\[
V\cos\gamma
\]

因此水平转向加速度：

\[
a_\psi=V\cos\gamma\dot\psi
\]

根据牛顿第二定律：

\[
F=ma
\]

得到：

\[
F_\psi=mV\cos\gamma\dot\psi
\]

---

## 8.3 图中模型的推导

法向总力：

\[
L+T\sin\alpha
\]

其中水平转弯方向的分量：

\[
(L+T\sin\alpha)\sin\phi
\]

因此：

\[
mV\cos\gamma\dot\psi
=
(L+T\sin\alpha)\sin\phi
\]

解得：

\[
\boxed{
\dot\psi
=
\frac{(L+T\sin\alpha)\sin\phi}{mV\cos\gamma}
}
\]

---

## 8.4 简化模型的推导

简化模型中：

\[
L=nmg
\]

并且：

\[
T\sin\alpha\approx0
\]

于是：

\[
mV\cos\gamma\dot\psi
=
nmg\sin\mu
\]

约掉 \(m\)：

\[
\boxed{
\dot\psi
=
\frac{g}{V\cos\gamma}n\sin\mu
}
\]

---

# 9. \(\dot\gamma\) 的推导

## 9.1 飞行路径角的物理意义

\[
\gamma
\]

表示速度方向相对水平面的角度。

其变化率：

\[
\dot\gamma=\frac{d\gamma}{dt}
\]

表示飞机抬头或下压的快慢。

---

## 9.2 为什么：

\[
a_\gamma=V\dot\gamma
\]

因为速度方向在竖直平面内转动。

根据：

\[
a=V\times\text{角速度}
\]

得到：

\[
\boxed{a_\gamma=V\dot\gamma}
\]

对应力：

\[
F_\gamma=mV\dot\gamma
\]

---

## 9.3 图中模型的推导

法向总力：

\[
L+T\sin\alpha
\]

其在竖直平面中的分量：

\[
(L+T\sin\alpha)\cos\phi
\]

重力在该方向的分量：

\[
mg\cos\gamma
\]

于是：

\[
mV\dot\gamma
=
(L+T\sin\alpha)\cos\phi-mg\cos\gamma
\]

解得：

\[
\boxed{
\dot\gamma
=
\frac{(L+T\sin\alpha)\cos\phi-mg\cos\gamma}{mV}
}
\]

---

## 9.4 简化模型的推导

简化模型中：

\[
L=nmg
\]

并且：

\[
T\sin\alpha\approx0
\]

于是：

\[
mV\dot\gamma
=
nmg\cos\mu-mg\cos\gamma
\]

约掉 \(m\)：

\[
\boxed{
\dot\gamma
=
\frac{g}{V}(n\cos\mu-\cos\gamma)
}
\]

---

# 10. \(\dot V\) 的推导

## 10.1 速度大小变化率

\[
\dot V=\frac{dV}{dt}
\]

表示飞机加速或减速。

只需要考虑“沿速度方向”的力。

根据牛顿第二定律：

\[
F_\parallel=m\dot V
\]

---

## 10.2 图中模型的推导

沿速度方向的力包括：

### 推力分量

\[
T\cos\alpha
\]

### 阻力

\[
-D
\]

### 重力分量

\[
-mg\sin\gamma
\]

因此：

\[
m\dot V
=
T\cos\alpha-D-mg\sin\gamma
\]

解得：

\[
\boxed{
\dot V
=
\frac{T\cos\alpha-D}{m}-g\sin\gamma
}
\]

---

## 10.3 简化模型的推导

简化模型假设：

\[
\alpha\approx0
\]

所以：

\[
\cos\alpha\approx1
\]

因此：

\[
m\dot V
=
T-D-mg\sin\gamma
\]

解得：

\[
\boxed{
\dot V
=
\frac{T-D}{m}-g\sin\gamma
}
\]

---

# 11. \(\dot m\) 与燃油消耗推导

在图中模型中，最后一个状态方程为：

\[
\dot m
=-\frac{c_T}{g}T
\]

它描述的是：

> 飞机由于消耗燃油，质量会随时间减小。

---

## 11.1 从重量方程开始

很多航空动力学教材首先写的是重量变化率：

\[
\dot W=-c_TT
\]

其中：

| 变量 | 含义 |
|---|---|
| \(W\) | 重量（weight） |
| \(T\) | 推力（thrust） |
| \(c_T\) | 燃油消耗参数 |

负号表示：

> 飞机持续消耗燃油，因此重量不断减小。

---

## 11.2 为什么推力越大，燃油消耗越快

发动机产生更大的推力时，需要燃烧更多燃油。

因此通常近似认为：

\[
\text{燃油消耗率} \propto T
\]

于是写成：

\[
\dot W=-c_TT
\]

这里：

\[
c_T
\]

相当于“单位推力对应的重量消耗速度”。

---

## 11.3 为什么能得到 \(\dot m\)

因为：

\[
W=mg
\]

其中：

- \(W\)：重量
- \(m\)：质量
- \(g\)：重力加速度

对时间求导：

\[
\dot W=\frac{d}{dt}(mg)
\]

因为：

\[
g
\]

近似是常数，所以：

\[
\dot W=g\dot m
\]

又因为：

\[
\dot W=-c_TT
\]

所以：

\[
g\dot m=-c_TT
\]

两边除以 \(g\)：

\[
\boxed{
\dot m=-\frac{c_T}{g}T
}
\]

这就是图中模型最后一个方程的来源。

---

## 11.4 物理意义

方程：

\[
\dot m=-\frac{c_T}{g}T
\]

表示：

- 推力越大；
- 燃油消耗越快；
- 飞机质量下降越快。

如果：

\[
T=0
\]

则：

\[
\dot m=0
\]

表示不消耗燃油。

---

## 11.5 为什么很多简化模型直接令 \(\dot m=0\)

因为很多轨迹仿真：

- 时间较短；
- 质量变化不明显；
- 更关注轨迹与制导。

因此经常近似：

\[
\boxed{\dot m=0}
\]

这样可以简化模型。

---

# 12. 重量 \(W\) 与质量 \(m\)

二者关系：

\[
W=mg
\]

其中：

| 变量 | 含义 |
|---|---|
| \(m\) | 质量（kg） |
| \(W\) | 重量 / 重力（N） |
| \(g\) | 重力加速度 |

因此：

\[
m=\frac{W}{g}
\]

所以：

\[
\frac{1}{m}=\frac{g}{W}
\]

这就是为什么很多航空公式里会出现：

\[
\frac{g}{W}
\]

它本质上等价于：

\[
\frac{1}{m}
\]

---

# 12. 最终总结

图中模型：

- 更完整；
- 考虑推力方向偏离速度方向；
- 通过攻角 \(\alpha\) 决定升力；
- 更贴近真实空气动力学。

你的模型：

- 更简洁；
- 假设推力沿速度方向；
- 用 \(n\) 直接控制升力；
- 更适合快速轨迹仿真与制导算法。

你的模型本质上是图中模型在以下假设下的简化：

\[
\alpha\approx0
\]

\[
T\sin\alpha\approx0
\]

\[
L=nmg
\]

\[
\phi=\mu
\]

因此得到：

\[
\dot\psi
=
\frac{g}{V\cos\gamma}n\sin\mu
\]

\[
\dot\gamma
=
\frac{g}{V}(n\cos\mu-\cos\gamma)
\]

\[
\dot V
=
\frac{T-D}{m}-g\sin\gamma
\]
