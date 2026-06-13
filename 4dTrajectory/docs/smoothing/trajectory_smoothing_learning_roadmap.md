# 轨迹重建、平滑、插值与外推：方法选择与初学者学习路线

这份笔记面向刚开始做轨迹数据处理的人。目标不是一次性学完所有理论，而是帮助你知道：

1. 每种方法适合什么场景；
2. 学习顺序应该怎么排；
3. 初学者应该先看哪些书、教程或论文；
4. 如何从简单实现逐步过渡到更可靠的工程方案。

---

## 0. 先给结论：怎么选方法？

| 你的问题 | 优先推荐 | 为什么 |
|---|---|---|
| 只想快速把轨迹变平滑 | 平滑样条、带差分正则化的最小二乘 | 简单、快、容易实现 |
| 观测点有不同可信度 | WLS（Weighted Least Squares，加权最小二乘） | 可以给高精度点更大权重 |
| 有跳点、离群点、误检 | Huber 损失、L1（一范数）损失、鲁棒最小二乘 | 不容易被异常点拉偏 |
| 有明确运动模型，例如匀速、匀加速 | Kalman 平滑器 | 能同时估计位置、速度、加速度 |
| 需要实时在线处理 | Kalman 滤波器 | 每来一个新观测就能更新 |
| 需要离线得到更高质量结果 | RTS（Rauch–Tung–Striebel，Rauch–Tung–Striebel）平滑器 | 利用过去和未来观测，比单向滤波更稳 |
| 需要置信区间或不确定性 | GPR（Gaussian Process Regression，高斯过程回归） | 输出均值，也能输出不确定性 |
| 有速度、加速度、转弯半径等物理约束 | 带约束优化、MHE（Moving Horizon Estimation，移动时域估计） | 能避免物理上不可能的轨迹 |
| 轨迹本来就是分段变化，例如突然转弯 | trend filtering（趋势滤波） | 适合分段线性或分段多项式结构 |
| 数据量很大、想要工程稳定 | Kalman 平滑器或样条 | 通常比 GPR（Gaussian Process Regression，高斯过程回归）更容易扩展 |

一句话建议：

> 如果你是初学者，先学“带差分正则化的最小二乘”和“平滑样条”；然后学鲁棒损失；再学 Kalman 滤波器/平滑器；最后根据需求学 GPR（Gaussian Process Regression，高斯过程回归）或带约束优化。

---

## 1. 方法一：带差分正则化的最小二乘

### 1.1 它解决什么问题？

它要找一条轨迹，让它同时满足：

- 尽量靠近观测点；
- 不要抖动太厉害；
- 不要出现过大的加速度；
- 不要出现过大的 jerk（加加速度）。

典型形式是：

```text
最小化：
观测误差
+ λ1 × 加速度惩罚
+ λ2 × jerk 惩罚
```

其中：

- `λ1` 控制“不想要大加速度”的程度；
- `λ2` 控制“不想要大 jerk”的程度；
- `λ1` 和 `λ2` 越大，轨迹越平滑；
- `λ1` 和 `λ2` 越小，轨迹越贴近观测点。

### 1.2 直观理解

可以把它想象成：

> 用一根有弹性的软尺去贴一堆测量点。  
> 软尺既想靠近点，又不想弯得太突然。

如果观测点有噪声，完全穿过每个点会导致轨迹锯齿状。这个方法允许轨迹稍微偏离观测点，从而换来更平滑的整体形状。

### 1.3 适用场景

适合：

- 轨迹采样时间比较规则；
- 数据量大，需要很快求解；
- 想做离线平滑、插值、短时间外推；
- 暂时没有复杂物理模型。

不太适合：

- 观测点有很多严重离群点；
- 轨迹有真实急转弯，但正则化太强；
- 时间间隔严重不均匀，但差分矩阵没有相应修改；
- 需要长期预测未来轨迹。

### 1.4 初学者学习路线

#### 第一步：学普通最小二乘

先理解：

```text
已知一些点，找一条线或曲线，让误差平方和最小。
```

你需要掌握：

- 矩阵乘法；
- 向量和矩阵范数；
- 最小二乘的直观意义；
- 为什么平方误差容易被异常点影响。

#### 第二步：学正则化

理解这个形式：

```text
最小化：
拟合误差 + λ × 模型复杂度
```

在轨迹问题里，“模型复杂度”可以理解为：

- 加速度太大；
- jerk 太大；
- 曲线弯得太剧烈。

#### 第三步：自己实现一个一维版本

先不要做三维轨迹，只做一维信号 `x(t)`：

```text
minimize sum((x_i - y_i)^2) + λ sum((x_i - 2x_{i+1} + x_{i+2})^2)
```

这个练习非常重要。做完之后，你会真正理解“二阶差分约束为什么会让曲线变平滑”。

#### 第四步：扩展到二维或三维

分别对：

- `x(t)`
- `y(t)`
- `z(t)`

做同样的平滑，或者把它们合在一个矩阵里一起写。

#### 第五步：用交叉验证选参数

不要凭感觉选 `λ1` 和 `λ2`。可以随机拿掉一部分观测点：

1. 用剩下的点拟合轨迹；
2. 在拿掉的点上计算误差；
3. 选择误差最小的 `λ1` 和 `λ2`。

### 1.5 推荐资源

**入门书：**

- Stephen Boyd, Lieven Vandenberghe, *Convex Optimization*  
  重点看 least squares（最小二乘）和 regularization（正则化）相关内容。  
  官方页面：<https://stanford.edu/~boyd/cvxbook/>

**配套实践：**

- 用 `numpy.linalg.solve` 解线性方程；
- 用 `scipy.sparse` 构造稀疏差分矩阵；
- 用 `matplotlib` 画原始点和平滑轨迹。

---

## 2. 方法二：WLS（Weighted Least Squares，加权最小二乘）

### 2.1 它和普通最小二乘有什么区别？

普通最小二乘默认每个观测点一样可信。

WLS（Weighted Least Squares，加权最小二乘）认为：

> 有些观测点更可靠，有些观测点更不可靠。

于是它给不同观测点不同权重：

```text
高可信观测点：权重大
低可信观测点：权重小
```

### 2.2 什么时候用？

适合：

- 多传感器融合；
- 有些传感器误差大，有些传感器误差小；
- 有些时刻测量质量差；
- GPS（Global Positioning System，全球定位系统）信号有时好、有时差；
- 雷达或视觉检测给出了置信度分数。

### 2.3 学习路线

#### 第一步：理解“误差方差”

如果某个测量点噪声方差大，说明它不稳定；如果噪声方差小，说明它更可靠。

常见权重设置是：

```text
权重 ≈ 1 / 方差
```

方差越大，权重越小。

#### 第二步：把原方法的第一项改掉

原来是：

```text
观测误差平方和
```

改成：

```text
加权观测误差平方和
```

也就是让更可靠的点对结果影响更大。

#### 第三步：和普通最小二乘对比

做一个实验：

1. 生成一条真实轨迹；
2. 给一半点加小噪声；
3. 给另一半点加大噪声；
4. 比较普通最小二乘和 WLS（Weighted Least Squares，加权最小二乘）的结果。

### 2.4 推荐资源

**先学：**

- 最小二乘；
- 噪声方差；
- 高斯噪声；
- 最大似然估计的基本直觉。

**实践建议：**

- 从一维信号开始；
- 给每个点设置不同权重；
- 画出权重对平滑结果的影响。

---

## 3. 方法三：鲁棒损失，Huber 损失和 L1（一范数）损失

### 3.1 为什么需要鲁棒损失？

普通平方误差有一个问题：

> 一个特别离谱的点，可能把整条轨迹拉歪。

例如 GPS（Global Positioning System，全球定位系统）突然跳到几百米外，如果继续用平方误差，这个异常点会产生非常大的惩罚，优化器会努力去靠近它。

鲁棒损失的目的就是：

> 对小误差仍然认真处理，但对特别大的误差不要过度相信。

### 3.2 Huber 损失的直观理解

Huber 损失可以理解成：

```text
误差小的时候：像平方误差
误差大的时候：像绝对值误差
```

这样它兼顾了两个优点：

- 小误差附近比较平滑，容易优化；
- 大误差不会产生过分夸张的影响。

### 3.3 L1（一范数）损失的直观理解

L1（一范数）损失就是误差绝对值之和：

```text
sum(|error_i|)
```

它比平方误差更抗异常点，但优化起来通常比平方误差麻烦一些。

### 3.4 什么时候用？

适合：

- GPS（Global Positioning System，全球定位系统）跳点；
- 视觉跟踪漂移；
- 雷达误检；
- 传感器偶尔给出非常离谱的点；
- 不想手工删除异常点。

### 3.5 学习路线

#### 第一步：画三种损失函数

画出：

- 平方损失；
- L1（一范数）损失；
- Huber 损失。

观察大误差区域三者增长速度有什么不同。

#### 第二步：用一维直线拟合做实验

先不要上轨迹。先做最简单直线拟合：

1. 生成一条直线；
2. 加一些正常噪声；
3. 加几个离群点；
4. 分别用平方损失、L1（一范数）损失、Huber 损失拟合。

你会非常直观地看到鲁棒损失的价值。

#### 第三步：把鲁棒损失接到轨迹平滑里

把原方法里的观测误差项从平方误差改成 Huber 损失或 L1（一范数）损失。

平滑项仍然可以保留：

```text
鲁棒观测误差 + 加速度平滑惩罚 + jerk 平滑惩罚
```

#### 第四步：学习 IRLS（Iteratively Reweighted Least Squares，迭代重加权最小二乘）

很多鲁棒拟合可以通过 IRLS（Iteratively Reweighted Least Squares，迭代重加权最小二乘）实现。直观上就是：

1. 先做一次普通拟合；
2. 看哪些点误差很大；
3. 给误差大的点降低权重；
4. 再拟合；
5. 重复直到稳定。

### 3.6 推荐资源

**论文：**

- Peter J. Huber, “Robust Estimation of a Location Parameter”, 1964  
  这是 Huber 损失相关思想的经典来源。  
  页面：<https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full>

**实践文档：**

- SciPy（Scientific Python，科学计算 Python 库）的 `least_squares` 支持 `soft_l1` 和 `huber` 等鲁棒损失。  
  页面：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>

---

## 4. 方法四：Kalman 滤波器与 Kalman 平滑器

### 4.1 核心思想

Kalman 方法不是只看“轨迹形状”，而是显式建立运动状态。

例如状态可以写成：

```text
位置 + 速度
```

或者：

```text
位置 + 速度 + 加速度
```

然后假设运动大致遵循某个模型，例如：

```text
下一时刻位置 ≈ 当前的位置 + 当前速度 × 时间间隔
```

### 4.2 滤波和平滑有什么区别？

Kalman 滤波器：

```text
只用当前和过去的观测
```

适合实时系统。

Kalman 平滑器：

```text
同时用过去、当前和未来的观测
```

适合离线重建，通常结果比滤波更平滑、更准确。

RTS（Rauch–Tung–Striebel，Rauch–Tung–Striebel）平滑器是最常见的 Kalman 平滑器之一。

### 4.3 它为什么适合轨迹？

因为轨迹本身就有运动规律：

- 位置不会无缘无故跳变；
- 速度通常连续；
- 加速度通常有限；
- 传感器观测有噪声。

Kalman 方法正好把这些信息放进同一个模型里。

### 4.4 适用场景

适合：

- 飞机轨迹；
- 车辆轨迹；
- 机器人定位；
- 雷达跟踪；
- GPS（Global Positioning System，全球定位系统）和惯性传感器融合；
- 需要同时估计速度和加速度。

不太适合：

- 完全不知道运动模型；
- 非线性很强但只用线性 Kalman 滤波器；
- 噪声分布严重非高斯且没有做鲁棒处理。

### 4.5 学习路线

#### 第一步：只学一维匀速模型

状态设为：

```text
x = [位置, 速度]
```

观测只有位置。

先理解两个步骤：

```text
预测：根据运动模型猜下一步
更新：根据观测修正预测
```

#### 第二步：学过程噪声和观测噪声

Kalman 方法里最重要的两个参数是：

- 过程噪声：你对运动模型有多不信任；
- 观测噪声：你对传感器有多不信任。

直观上：

```text
观测噪声大：更相信模型
过程噪声大：更相信观测
```

#### 第三步：扩展到二维或三维

把状态从：

```text
[位置, 速度]
```

扩展成：

```text
[x, y, z, vx, vy, vz]
```

如果要建匀加速模型，可以再加入：

```text
[ax, ay, az]
```

#### 第四步：实现 RTS（Rauch–Tung–Striebel，Rauch–Tung–Striebel）平滑器

先正向跑 Kalman 滤波器，再反向做平滑。

你会发现平滑结果通常比单向滤波更适合离线轨迹重建。

### 4.6 推荐资源

**最适合初学者：**

- Roger R. Labbe Jr., *Kalman and Bayesian Filters in Python*  
  非常适合入门，代码用 Python 和 Jupyter Notebook（交互式计算笔记本）写，直觉解释很多。  
  页面：<https://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python/>

**进阶书：**

- Simo Särkkä, *Bayesian Filtering and Smoothing*  
  更系统，覆盖非线性 Kalman 滤波、粒子滤波和平滑。  
  页面：<https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf>

**经典论文：**

- Rauch, Tung, Striebel, “Maximum likelihood estimates of linear dynamic systems”, 1965  
  RTS（Rauch–Tung–Striebel，Rauch–Tung–Striebel）平滑器的经典来源。  
  页面：<https://arc.aiaa.org/doi/10.2514/3.3166>

---

## 5. 方法五：平滑样条与 B-spline（Basis spline，基样条）

### 5.1 核心思想

样条方法不是在每个时间点都自由选择一个位置，而是用一条分段多项式曲线表示轨迹。

你可以把它想象成：

> 用几段柔和连接的曲线，去近似一堆带噪声的点。

平滑样条常见目标是：

```text
拟合误差 + λ × 曲率惩罚
```

曲率惩罚越大，曲线越不愿意弯。

### 5.2 它和差分正则化最小二乘的关系

它们思想非常接近：

- 差分正则化最小二乘是在离散时间点上惩罚二阶差分或三阶差分；
- 平滑样条是在连续曲线上惩罚二阶导数等平滑程度。

所以可以把平滑样条理解成：

> 连续曲线版本的轨迹平滑。

### 5.3 适用场景

适合：

- 离线轨迹平滑；
- 插值；
- 不规则采样时间；
- 想得到连续可求导的轨迹；
- 机器人路径、动画曲线、车辆轨迹处理。

不太适合：

- 需要显式估计运动状态和不确定性；
- 需要严格物理约束；
- 离群点很多但没有鲁棒处理。

### 5.4 学习路线

#### 第一步：学插值样条

先理解：

```text
曲线完全穿过所有点
```

这适合无噪声数据，但不适合带噪声数据。

#### 第二步：学平滑样条

再理解：

```text
曲线不一定穿过所有点，而是在贴近数据和平滑之间折中。
```

这才适合真实测量数据。

#### 第三步：学参数曲线

轨迹不是简单的 `y = f(x)`，而是：

```text
x = x(t)
y = y(t)
z = z(t)
```

所以对轨迹更常用的是分别拟合 `x(t)`、`y(t)`、`z(t)`。

#### 第四步：学 B-spline（Basis spline，基样条）控制点

B-spline（Basis spline，基样条）的好处是可以用少量控制点表示整条曲线。

这对大规模轨迹很有用，因为参数更少，曲线也天然平滑。

### 5.5 推荐资源

**实践文档：**

- SciPy（Scientific Python，科学计算 Python 库）smoothing splines 教程  
  适合从代码层面理解平滑样条。  
  页面：<https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html>

**经典书：**

- Carl de Boor, *A Practical Guide to Splines*  
  样条领域经典书，但对初学者偏难。建议先看教程，之后把它当参考书。  
  页面：<https://link.springer.com/book/9780387953663>

---

## 6. 方法六：GPR（Gaussian Process Regression，高斯过程回归）

### 6.1 核心思想

GPR（Gaussian Process Regression，高斯过程回归）可以理解成一种“带不确定性的平滑插值”。

它不仅给出预测轨迹：

```text
某个时间点的位置估计
```

还可以给出：

```text
这个估计有多不确定
```

这对缺失数据和外推尤其有用，因为你能看到越远离观测点，不确定性通常越大。

### 6.2 核函数是什么？

GPR（Gaussian Process Regression，高斯过程回归）里最重要的是 kernel（核函数）。

kernel（核函数）决定了你认为轨迹应该是什么样的：

- 很平滑；
- 可以周期变化；
- 可以有不同长度尺度；
- 可以叠加多个运动模式。

例如，如果选择很平滑的 kernel（核函数），模型会倾向于给出非常柔和的轨迹。

### 6.3 适用场景

适合：

- 数据量不是特别大；
- 需要不确定性；
- 需要自然处理缺失点；
- 想做带置信区间的插值和短期外推；
- 对模型可解释性有一定要求。

不太适合：

- 数据点非常多；
- 实时系统；
- kernel（核函数）不知道怎么选；
- 需要严格物理约束。

### 6.4 学习路线

#### 第一步：先把它当成“高级平滑器”

一开始不要急着理解全部贝叶斯推导。先知道：

```text
输入：观测时间和观测位置
输出：每个时间点的位置均值和标准差
```

#### 第二步：学习三个核心概念

只需要先掌握：

- prior（先验）：看到数据前，你认为函数大概是什么样；
- likelihood（似然）：观测噪声有多大；
- posterior（后验）：看到数据后，函数应该是什么样。

#### 第三步：学习 kernel（核函数）

先从最常用的 RBF（Radial Basis Function，径向基函数）kernel（核函数）开始。

然后再学：

- Matérn kernel（Matérn 核函数）；
- periodic kernel（周期核函数）；
- white noise kernel（白噪声核函数）。

#### 第四步：分别拟合 `x(t)`、`y(t)`、`z(t)`

先不要做复杂多输出模型。初学阶段可以把三个坐标分开拟合。

### 6.5 推荐资源

**入门实践：**

- scikit-learn 的 Gaussian Process 文档  
  可以直接跑例子，理解均值和标准差输出。  
  页面：<https://scikit-learn.org/stable/modules/gaussian_process.html>

**经典书：**

- Carl Edward Rasmussen, Christopher K. I. Williams, *Gaussian Processes for Machine Learning*  
  这是 GPR（Gaussian Process Regression，高斯过程回归）最经典的教材之一。前几章值得读。  
  页面：<https://gaussianprocess.org/gpml/>

---

## 7. 方法七：带物理约束的优化

### 7.1 核心思想

有时候，仅仅“平滑”是不够的。

例如车辆轨迹必须满足：

```text
速度不能超过上限
加速度不能超过上限
转弯半径不能太小
不能穿过墙
必须沿着道路
```

这时可以把轨迹重建写成约束优化问题：

```text
最小化：观测误差 + 平滑惩罚

满足：
速度限制
加速度限制
道路约束
动力学约束
```

### 7.2 适用场景

适合：

- 自动驾驶；
- 飞机轨迹；
- 机器人路径；
- 运动捕捉；
- 需要符合物理规律的重建；
- 对安全性要求高的系统。

不太适合：

- 只想快速画一条平滑曲线；
- 约束很复杂但没有可靠模型；
- 初学阶段还没掌握优化基础。

### 7.3 学习路线

#### 第一步：先加简单约束

从简单约束开始：

```text
速度不能超过 vmax
加速度不能超过 amax
```

不要一开始就做复杂车辆模型。

#### 第二步：用 CVXPY（Python 凸优化建模工具）实现凸优化版本

如果你的目标函数是平方误差，约束也是线性的或凸的，可以用 CVXPY（Python 凸优化建模工具）。

适合初学者，因为写出来的代码和数学公式很像。

#### 第三步：再学非线性约束

例如车辆模型里可能有：

```text
航向角
转向角
非线性运动方程
```

这时可能需要 CasADi（Computer Algebra System with Algorithmic Differentiation，算法微分与非线性优化工具）这类工具。

#### 第四步：学习 MHE（Moving Horizon Estimation，移动时域估计）

MHE（Moving Horizon Estimation，移动时域估计）可以理解成：

> 每次只看最近一段时间窗口，在窗口里做带约束的轨迹估计。

它比全局优化更适合在线系统，也更容易控制计算量。

### 7.4 推荐资源

**优化基础：**

- Stephen Boyd, Lieven Vandenberghe, *Convex Optimization*  
  页面：<https://stanford.edu/~boyd/cvxbook/>

**实践工具：**

- CVXPY（Python 凸优化建模工具）官方文档  
  页面：<https://www.cvxpy.org/>

- CasADi（Computer Algebra System with Algorithmic Differentiation，算法微分与非线性优化工具）官方文档  
  页面：<https://web.casadi.org/docs/>

---

## 8. 方法八：trend filtering（趋势滤波）

### 8.1 核心思想

前面的平滑方法大多用平方惩罚：

```text
sum((二阶差分)^2)
```

trend filtering（趋势滤波）常用 L1（一范数）惩罚：

```text
sum(|二阶差分|)
```

这个改变很关键。

平方惩罚倾向于让曲线到处都柔和变化；L1（一范数）惩罚倾向于让很多地方的二阶差分变成零，只在少数位置发生变化。

所以它适合：

> 分段线性或分段多项式的轨迹。

### 8.2 什么时候用？

适合：

- 车辆先直行、再转弯、再直行；
- 飞机轨迹有几个明显机动阶段；
- 希望自动找出“变化点”；
- 不希望过度抹平真实突变。

不太适合：

- 轨迹本来非常平滑；
- 不希望出现分段结构；
- 初学者还没有掌握凸优化。

### 8.3 学习路线

#### 第一步：比较 L2（二范数）平滑和 L1（一范数）平滑

做一个一维信号实验：

- L2（二范数）惩罚会产生整体柔和的曲线；
- L1（一范数）惩罚会产生分段结构。

#### 第二步：用 CVXPY（Python 凸优化建模工具）实现

目标可以写成：

```text
minimize sum((x_i - y_i)^2) + λ sum(abs(D2 x))
```

其中 `D2 x` 是二阶差分。

#### 第三步：扩展到二维或三维轨迹

分别对 `x(t)`、`y(t)`、`z(t)` 做趋势滤波，或者把坐标组合起来做联合惩罚。

### 8.4 推荐资源

**经典论文：**

- Seung-Jean Kim, Kwangmoo Koh, Stephen Boyd, Dimitry Gorinevsky, “L1 Trend Filtering”  
  页面：<https://web.stanford.edu/~boyd/papers/l1_trend_filter.html>

---

## 9. 推荐学习顺序：从零到能做项目

下面是一条比较稳的路线，不需要一次性学完所有方法。

### 阶段一：最小二乘和平滑直觉

目标：

- 能看懂“拟合项 + 正则项”；
- 能自己写一维平滑代码；
- 能解释 `λ` 变大或变小会发生什么。

建议学习：

1. 普通最小二乘；
2. 岭回归；
3. 差分矩阵；
4. 二阶差分和加速度的关系；
5. 三阶差分和 jerk 的关系。

练习：

- 生成一条带噪声正弦曲线；
- 用二阶差分正则化平滑；
- 画出不同 `λ` 的结果。

---

### 阶段二：加权和鲁棒

目标：

- 能处理不同可信度的观测点；
- 能处理异常点。

建议学习：

1. WLS（Weighted Least Squares，加权最小二乘）；
2. Huber 损失；
3. L1（一范数）损失；
4. IRLS（Iteratively Reweighted Least Squares，迭代重加权最小二乘）。

练习：

- 人为加入几个离群点；
- 比较平方损失、Huber 损失和 L1（一范数）损失；
- 看哪种方法更不容易被异常点带偏。

---

### 阶段三：样条

目标：

- 能用连续曲线表示轨迹；
- 能处理不规则采样时间；
- 能生成平滑且可求导的轨迹。

建议学习：

1. 三次样条；
2. 平滑样条；
3. B-spline（Basis spline，基样条）；
4. 参数曲线 `x(t), y(t), z(t)`。

练习：

- 用 SciPy（Scientific Python，科学计算 Python 库）拟合 `x(t)` 和 `y(t)`；
- 比较插值样条和平滑样条；
- 观察平滑参数对轨迹的影响。

---

### 阶段四：Kalman 方法

目标：

- 能建立状态空间模型；
- 能同时估计位置和速度；
- 能做实时滤波和离线平滑。

建议学习：

1. 一维 Kalman 滤波器；
2. 匀速模型；
3. 匀加速模型；
4. 过程噪声和观测噪声；
5. RTS（Rauch–Tung–Striebel，Rauch–Tung–Striebel）平滑器。

练习：

- 生成一条带速度的真实轨迹；
- 只观测带噪声的位置；
- 用 Kalman 滤波器估计位置和速度；
- 用 RTS（Rauch–Tung–Striebel，Rauch–Tung–Striebel）平滑器做离线优化。

---

### 阶段五：GPR（Gaussian Process Regression，高斯过程回归）

目标：

- 能输出轨迹估计和不确定性；
- 能做带置信区间的插值；
- 能理解 kernel（核函数）对结果的影响。

建议学习：

1. 高斯分布；
2. 多元高斯分布；
3. kernel（核函数）；
4. 噪声方差；
5. 预测均值和预测标准差。

练习：

- 用 scikit-learn 拟合一维轨迹；
- 画出预测均值和置信区间；
- 改变 kernel（核函数）的长度尺度；
- 观察外推区域的不确定性。

---

### 阶段六：带约束优化和 MHE（Moving Horizon Estimation，移动时域估计）

目标：

- 能把物理限制写进优化问题；
- 能处理速度、加速度、边界等约束；
- 能理解在线窗口估计。

建议学习：

1. 凸优化基本概念；
2. 线性约束和二次目标；
3. CVXPY（Python 凸优化建模工具）；
4. 非线性动力学；
5. CasADi（Computer Algebra System with Algorithmic Differentiation，算法微分与非线性优化工具）；
6. MHE（Moving Horizon Estimation，移动时域估计）。

练习：

- 在轨迹平滑里加入最大速度约束；
- 加入最大加速度约束；
- 比较有约束和无约束的结果；
- 尝试只用最近一段窗口估计轨迹。

---

## 10. 八周学习计划

### 第 1 周：最小二乘与正则化

任务：

- 学会普通最小二乘；
- 学会用矩阵写误差平方和；
- 实现一维二阶差分平滑。

产出：

- 一张图：原始噪声点 + 平滑曲线；
- 一组不同 `λ` 的对比图。

---

### 第 2 周：原始轨迹重建方法

任务：

- 加入缺失观测；
- 加入二阶差分和三阶差分；
- 用交叉验证选择 `λ1` 和 `λ2`。

产出：

- 一个可以处理缺失点的轨迹平滑脚本；
- 一张参数网格搜索表。

---

### 第 3 周：WLS（Weighted Least Squares，加权最小二乘）与鲁棒损失

任务：

- 实现 WLS（Weighted Least Squares，加权最小二乘）；
- 加入几个离群点；
- 比较平方损失和 Huber 损失。

产出：

- 一张离群点实验对比图；
- 一段文字解释为什么 Huber 损失更稳。

---

### 第 4 周：平滑样条

任务：

- 学会用 SciPy（Scientific Python，科学计算 Python 库）的平滑样条；
- 分别拟合 `x(t)`、`y(t)`；
- 比较样条和差分正则化方法。

产出：

- 同一条轨迹上两种方法的对比图；
- 对平滑参数选择的总结。

---

### 第 5 周：Kalman 滤波器

任务：

- 实现一维匀速 Kalman 滤波器；
- 扩展到二维轨迹；
- 调整过程噪声和观测噪声。

产出：

- 一个二维 Kalman 滤波轨迹例子；
- 一张不同噪声参数下的结果对比图。

---

### 第 6 周：Kalman 平滑器

任务：

- 实现 RTS（Rauch–Tung–Striebel，Rauch–Tung–Striebel）平滑器；
- 比较滤波和平滑；
- 在缺失观测情况下测试。

产出：

- 一张滤波、平滑、真实轨迹对比图；
- 一个说明“为什么平滑比滤波更适合离线重建”的小结。

---

### 第 7 周：GPR（Gaussian Process Regression，高斯过程回归）

任务：

- 跑通 scikit-learn 的 GPR（Gaussian Process Regression，高斯过程回归）例子；
- 输出均值和标准差；
- 改变 kernel（核函数）参数。

产出：

- 一张带置信区间的轨迹图；
- 一个解释外推不确定性的小结。

---

### 第 8 周：带约束优化或 trend filtering（趋势滤波）

任务二选一：

选项 A：带约束优化

- 加入最大速度约束；
- 加入最大加速度约束；
- 用 CVXPY（Python 凸优化建模工具）求解。

选项 B：trend filtering（趋势滤波）

- 用 L1（一范数）惩罚二阶差分；
- 比较 L1（一范数）和平方惩罚；
- 观察分段结构。

产出：

- 一个更接近工程场景的轨迹重建例子；
- 一页方法选择总结。

---

## 11. 推荐阅读顺序

### 最初级，先看这些

1. Roger R. Labbe Jr., *Kalman and Bayesian Filters in Python*  
   理由：直觉友好，代码友好，非常适合初学 Kalman 方法。  
   <https://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python/>

2. SciPy（Scientific Python，科学计算 Python 库）smoothing splines 教程  
   理由：可以快速上手平滑样条，不需要先读很厚的书。  
   <https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html>

3. scikit-learn Gaussian Process 文档  
   理由：适合先用代码理解 GPR（Gaussian Process Regression，高斯过程回归）。  
   <https://scikit-learn.org/stable/modules/gaussian_process.html>

---

### 有一点基础后再看

4. Stephen Boyd, Lieven Vandenberghe, *Convex Optimization*  
   理由：理解最小二乘、正则化、约束优化的统一框架。  
   <https://stanford.edu/~boyd/cvxbook/>

5. Simo Särkkä, *Bayesian Filtering and Smoothing*  
   理由：系统学习 Kalman 滤波、非线性滤波和平滑。  
   <https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf>

6. Rasmussen and Williams, *Gaussian Processes for Machine Learning*  
   理由：系统学习 GPR（Gaussian Process Regression，高斯过程回归）。  
   <https://gaussianprocess.org/gpml/>

---

### 最后作为参考书或论文看

7. Carl de Boor, *A Practical Guide to Splines*  
   理由：样条经典参考书，适合深入查理论。  
   <https://link.springer.com/book/9780387953663>

8. Peter J. Huber, “Robust Estimation of a Location Parameter”  
   理由：鲁棒估计经典论文。  
   <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full>

9. Kim, Koh, Boyd, Gorinevsky, “L1 Trend Filtering”  
   理由：trend filtering（趋势滤波）的经典论文。  
   <https://web.stanford.edu/~boyd/papers/l1_trend_filter.html>

10. Rauch, Tung, Striebel, “Maximum likelihood estimates of linear dynamic systems”  
    理由：RTS（Rauch–Tung–Striebel，Rauch–Tung–Striebel）平滑器经典论文。  
    <https://arc.aiaa.org/doi/10.2514/3.3166>

---

## 12. 一个实用项目练习

你可以用同一份模拟数据比较所有方法。

### 数据生成

生成一条二维轨迹：

```text
先直行
再缓慢转弯
再加速
最后缺失一段观测
```

然后加入：

- 高斯噪声；
- 几个离群点；
- 一些缺失点；
- 不同观测点的置信度。

### 比较方法

依次实现：

1. 带差分正则化的最小二乘；
2. WLS（Weighted Least Squares，加权最小二乘）；
3. Huber 鲁棒平滑；
4. 平滑样条；
5. Kalman 平滑器；
6. GPR（Gaussian Process Regression，高斯过程回归）；
7. 带最大速度约束的优化；
8. trend filtering（趋势滤波）。

### 比较指标

可以看：

- 位置误差；
- 速度是否合理；
- 加速度是否过大；
- 是否被离群点影响；
- 缺失区间插值是否自然；
- 外推是否发散；
- 计算速度；
- 参数是否容易调。

---

## 13. 我的最终建议

如果你现在刚开始做轨迹重建，我建议按下面的顺序走：

```text
差分正则化最小二乘
→ WLS（Weighted Least Squares，加权最小二乘）
→ Huber 鲁棒损失
→ 平滑样条
→ Kalman 滤波器和平滑器
→ GPR（Gaussian Process Regression，高斯过程回归）
→ 带约束优化或 MHE（Moving Horizon Estimation，移动时域估计）
→ trend filtering（趋势滤波）
```

真正做工程时，最常见、最稳的组合通常是：

```text
异常点预处理
+ WLS（Weighted Least Squares，加权最小二乘）或 Huber 鲁棒损失
+ Kalman 平滑器或平滑样条
+ 必要的物理约束
```

如果你的目标是论文复现或研究，可以深入 GPR（Gaussian Process Regression，高斯过程回归）、MHE（Moving Horizon Estimation，移动时域估计）和 trend filtering（趋势滤波）。

如果你的目标是工程落地，优先把 Kalman 平滑器、鲁棒损失和带约束优化练熟。
