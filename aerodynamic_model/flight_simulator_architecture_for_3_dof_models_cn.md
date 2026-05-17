# 基于公式的飞行模拟器：典型架构与工程实现

# 1. 你现在的问题本质是什么

你现在实际上是在做：

\[
\text{Flight Dynamics Simulation}
\]

即：

> 用微分方程描述飞机运动，然后通过数值积分不断推进时间。

因此飞行模拟器本质上并不是：

- “把公式写进去然后一次算完”；

而是：

- “每一帧重新计算状态变化率，再积分得到下一时刻状态”。

所以它本质是：

\[
\boxed{
\text{状态方程} + \text{数值积分器} + \text{空气动力模型}
}
\]

---

# 2. 一个飞行模拟器最核心的结构

无论是：

- 导弹模拟；
- 飞机模拟；
- 空战模拟；
- 无人机；
- 制导律；

核心结构几乎都一样。

通常是：

```text
控制输入
   ↓
空气动力学计算
   ↓
计算力和力矩
   ↓
状态方程
   ↓
求导（dx/dt）
   ↓
数值积分
   ↓
得到下一时刻状态
   ↓
重复循环
```

---

# 3. 最经典的工程架构（推荐你采用）

对于你现在的三自由度模型，推荐架构如下：

```text
Simulator
│
├── State（状态）
├── Control（控制输入）
├── Atmosphere（大气模型）
├── Aerodynamics（空气动力学）
├── Dynamics（动力学方程）
├── Integrator（积分器）
└── Renderer / Logger（显示与记录）
```

---

# 4. 每一层到底做什么

# 4.1 State（状态）

保存飞机当前状态：

\[
\mathbf{x}
=
[X,Y,h,V,\psi,\gamma,m]
\]

例如：

```python
class State:
    x: float
    y: float
    h: float
    V: float
    psi: float
    gamma: float
    m: float
```

这是：

> 飞机“当前时刻”的状态。

---

# 4.2 Control（控制输入）

你的控制输入：

\[
[T,\mu,n]
\]

例如：

```python
class Control:
    thrust: float
    bank: float
    load_factor: float
```

它表示：

> 飞行员 / 自动驾驶 / 制导系统给出的操纵命令。

---

# 4.3 Atmosphere（大气模型）

用于计算：

\[
\rho(h)
\]

即空气密度。

最简单：

```python
rho = rho0 * exp(-h / H)
```

或者使用：

- ISA（International Standard Atmosphere，国际标准大气）

因为：

- 升力；
- 阻力；
- 马赫数；

都依赖空气密度。

---

# 4.4 Aerodynamics（空气动力学）

这一层是：

> 根据状态计算空气动力。

通常做：

## Step 1

计算：

\[
L=nmg
\]

---

## Step 2

计算：

\[
C_L
=
\frac{2L}{\rho V^2 S}
\]

---

## Step 3

计算：

\[
C_D=C_{D0}+kC_L^2
\]

---

## Step 4

计算：

\[
D=\frac12 \rho V^2 S C_D
\]

---

最终输出：

```python
Lift
Drag
```

这就是空气动力层。

---

# 4.5 Dynamics（动力学方程）

这是：

> 真正的运动微分方程。

输入：

- 当前状态；
- 控制输入；
- 空气动力；

输出：

\[
\dot x
\]

即：

```python
[x_dot, y_dot, ...]
```

例如：

\[
\dot V
=
\frac{T-D}{m}-g\sin\gamma
\]

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

---

# 4.6 Integrator（积分器）

这是整个模拟器最关键的一层。

因为你得到的只是：

\[
\dot x
\]

即：

> “变化率”。

你需要把它积分成：

\[
x(t+\Delta t)
\]

---

## 最简单：Euler（欧拉法）

```python
x_next = x + x_dot * dt
```

例如：

```python
V = V + V_dot * dt
psi = psi + psi_dot * dt
```

这就是最基本的飞行模拟器。

---

## 更推荐：RK4（Runge-Kutta 4）

工程中更常见。

优点：

- 更稳定；
- 精度高；
- 不容易发散。

你以后几乎一定会用 RK4。

---

# 4.7 Renderer / Logger

最后：

- 可视化轨迹；
- 输出 CSV；
- 绘图；
- 连接 Unity / Unreal。

例如：

```text
(X,Y,h)
→ 3D轨迹
```

---

# 5. 一个真正的仿真循环长什么样

这是最核心的东西。

---

# 每一帧：

```python
while running:

    # 1. 读取状态
    state

    # 2. 输入控制
    control

    # 3. 计算空气密度
    rho

    # 4. 计算升力阻力
    L, D

    # 5. 计算状态导数
    x_dot

    # 6. 数值积分
    x_next

    # 7. 更新时间
    t += dt
```

---

# 6. 所以本质上不是“套公式”

很多初学者会以为：

> 飞行模拟器 = 把公式直接写进去。

实际上不是。

真正本质是：

\[
\boxed{
\text{连续动力学系统的数值积分}
}
\]

你写的公式其实只是：

\[
\dot x=f(x,u)
\]

即：

> 状态变化率。

模拟器真正做的是：

\[
\boxed{
x_{k+1}=x_k+f(x_k,u_k)\Delta t
}
\]

---

# 7. 工业界是怎么做的

真正的大型飞行模拟器通常分层：

```text
Control Layer
↓
Flight Dynamics
↓
Aerodynamics
↓
Engine
↓
Environment
↓
Physics Integrator
↓
Rendering
```

例如：

- X-Plane
- DCS
- JSBSim
- FlightGear

都类似。

---

# 8. 有没有现成的库？

有，而且很多。

---

# 8.1 JSBSim（最推荐）

这是最经典的开源飞行动力学库。

特点：

- 开源；
- 专业；
- 支持 3DOF / 6DOF；
- FlightGear 在用；
- 很多研究项目在用。

它本质就是：

```text
空气动力 + 动力学方程 + 数值积分
```

如果你以后做更真实模拟，非常值得研究。

---

# 8.2 FlightGear

开源飞行模拟器。

它底层很多动力学来自 JSBSim。

适合：

- 看真实工程架构；
- 学习仿真系统组织方式。

---

# 8.3 MATLAB / Simulink

很多航空航天研究都在用。

优点：

- 快速建模；
- 很强的控制系统工具；
- 很适合验证公式。

缺点：

- 商业软件；
- 不适合大型实时游戏。

---

# 8.4 Python 科学计算路线

很多研究原型会用：

```python
numpy
scipy
matplotlib
```

积分器：

```python
scipy.integrate.solve_ivp
```

这是非常推荐你的路线。

因为：

- 开发快；
- 易调试；
- 非常适合验证模型。

---

# 9. 对你来说最推荐的路线

你现在最适合：

---

# 第一阶段（非常推荐）

自己从零实现：

```text
状态方程
+
Euler / RK4
+
阻力模型
```

原因：

> 你会真正理解飞行动力学。

这是最重要的。

不要一开始就用复杂库。

---

# 第二阶段

加入：

- ISA 大气；
- 马赫数；
- 更真实阻力；
- 自动驾驶；
- 制导律；
- PID 控制；
- 能量机动。

---

# 第三阶段

再研究：

- JSBSim；
- FlightGear；
- 6DOF；
- 刚体姿态动力学；
- 四元数；
- 完整空气动力数据库。

---

# 10. 你现在最应该掌握的数学

按重要程度排序：

---

# 第一优先级

## 常微分方程（ODE，Ordinary Differential Equation）

因为飞行动力学本质上是：

\[
\dot x=f(x,u)
\]

---

# 第二优先级

## 数值积分

重点：

- Euler
- RK4
- Stability（稳定性）

---

# 第三优先级

## 向量与坐标变换

重点：

- 速度分解；
- 坐标系；
- 投影；
- 欧拉角。

---

# 第四优先级

## 空气动力学

重点：

- Lift
- Drag
- CL/CD
- Drag Polar

---

# 11. 最后：真正的核心思想

飞行模拟器最核心的思想只有一句话：

\[
\boxed{
\text{根据当前状态计算导数，再积分得到下一状态}
}
\]

即：

```text
当前状态
→ 算力
→ 算加速度
→ 算导数
→ 积分
→ 下一状态
```

这就是绝大多数飞行模拟器的本质。

