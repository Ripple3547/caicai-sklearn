# 第四讲 复杂自变量

> [!info] 本讲概要
>
> **内容概要**：分类变量、多项式回归、样条函数、主成分分析
> **对应教材**：Weisberg Chapter 5

| 章节 | 核心问题 |
| :--- | :--- |
| 一、分类变量 | 无排序的类别如何进入线性回归 |
| 二、多项式回归 | 单个变量的弯曲关系如何刻画 |
| 三、样条函数 | 分段低阶多项式 + 光滑性约束 |
| 四、主成分分析 | 多个变量如何压缩成少数几个线性组合 |

---

## 一、分类变量

### 1. 单个分类变量

**分类变量**（*categorical variable* 或 *factor*）是指**只有有限个取值，每个值对应一个类别，且类别间无排序**的变量。设自变量 $F$ 有 $d$ 个不同的可能取值 $\{1,2,\dots,d\}$，这些取值称为 $F$ 的 $d$ 个 **levels**；因变量记为 $Y$，样本为 $(f_1,y_1),\dots,(f_n,y_n)$。那么 $Y\sim F$ 的一元回归模型可以写成按类别分别取均值的分段形式，等价地写为示性函数的加权和：

$$
y_i=\sum_{k=1}^{d}\alpha_k^{*}I_{\{f_i=k\}}+e_i,\qquad i=1,\dots,n,
\tag{1}
$$

其中 $\alpha_1^{*},\dots,\alpha_d^{*}$ 为每个 level 对应的均值，$e_i$ 为统计误差。直观上，每个 level 均值的估计应该就是自变量取值为该 level 的所有样本的 $y_i$ 的平均数，即**每个 level 内部 $y_i$ 的样本均值**：

$$
\hat{\alpha}_k=\frac{\sum_{i=1}^{n}I_{\{f_i=k\}}\,y_i}{\sum_{i=1}^{n}I_{\{f_i=k\}}}.
\tag{2}
$$

每个 level 内部 $y_i$ 的分布可以用**箱线图**（*boxplot*）表示，图中各元素的含义如下表。

| 图形元素 | 统计含义 |
| :--- | :--- |
| 箱体中间的粗线 | 中位数（50% 分位数） |
| 长方形上下两条边 | 25% 分位数、75% 分位数 |
| 箱体高度 IQR | *inter-quartile range*，即 75% 分位数减 25% 分位数 |
| 箱体外的上下两道横线 | 离中位数至少 **1.5 倍 IQR** 的最近的样本 |
| 横线之外的圆圈 | 被单独标出的样本 |

为了**在线性回归框架内**研究此问题，我们引入**虚拟变量**（*dummy variables*）。若 $F$ 有 $d$ 个类别，则引入 $d-1$ 个虚拟变量 $U_2,\dots,U_d$，其中

$$
U_k=I_{\{F=k\}},\qquad k=2,\dots,d,
$$

于是模型可以写成以下的**线性回归标准形式**：

$$
y_i=\beta_0^{*}+\sum_{k=2}^{d}\beta_k^{*}u_{ik}+e_i.
\tag{3}
$$

模型 (3) 与模型 (1) 的参数之间有一一对应关系，见下表。用最小二乘法得到 $\hat\beta_0,\hat\beta_2,\dots,\hat\beta_d$ 之后，估计值之间也满足**同样的关系**。

| 模型 (1) 的参数 | 模型 (3) 的参数 | 统计含义 |
| :--- | :--- | :--- |
| $\alpha_1^{*}$ | $\beta_0^{*}$ | 基准类（第 1 类）的均值 |
| $\alpha_k^{*}$ | $\beta_0^{*}+\beta_k^{*}$ | 第 $k$ 类的均值，$k=2,\dots,d$ |
| —— | $\beta_k^{*}$ | 第 $k$ 类相对基准类的均值偏移 |

> [!warning] 注意
>
> 加入 $d-1$ 个而非 $d$ 个虚拟变量，是为了避免与截距项**完全共线**（虚拟变量陷阱）。第一个类别被当作**基准类**（*baseline*），$\beta_k^{*}$ 因此解释为"第 $k$ 类与基准类之差"，而**不是**第 $k$ 类的绝对均值。

### 2. 单个分类变量加连续变量

假设自变量包括一个含 $d$ 个类别的分类变量 $F$、一个连续变量 $X$ 以及因变量 $Y$。最一般的做法是对 $F$ 的**每个类别内部各设定一个** $Y\sim X$ 的回归模型，这样就得到 $d$ 条*截距和斜率都不同*的直线：

$$
y_i=
\begin{cases}
\alpha_{01}^{*}+\alpha_{11}^{*}x_i+e_i, & \text{if } f_i=1,\\
\quad\cdots\\
\alpha_{0d}^{*}+\alpha_{1d}^{*}x_i+e_i, & \text{if } f_i=d.
\end{cases}
$$

引入虚拟变量后可以合成一个线性回归模型，其中前一部分称为**主效应**（*main effects*），后一部分称为**交互效应**（*interactions*）：

$$
y_i=\underbrace{\beta_0^{*}+\sum_{k=2}^{d}\beta_{0k}^{*}u_{ik}+\beta_1^{*}x_i}_{\text{主效应}}+\underbrace{\sum_{k=2}^{d}\beta_{1k}^{*}u_{ik}x_i}_{\text{交互效应}}+e_i .
$$

**交互项的存在，使得各类别可以有不同斜率**。两组参数之间的对应关系为 $\alpha_{01}^{*}=\beta_0^{*}$、$\alpha_{0k}^{*}=\beta_0^{*}+\beta_{0k}^{*}$（$k\ge 2$），以及 $\alpha_{11}^{*}=\beta_1^{*}$、$\alpha_{1k}^{*}=\beta_1^{*}+\beta_{1k}^{*}$（$k\ge 2$）。

有时为了简便只保留主效应项，得到**主效应模型**（*main effects model*）：

$$
y_i=\beta_0^{*}+\sum_{k=2}^{d}\beta_{0k}^{*}u_{ik}+\beta_1^{*}x_i+e_i,
$$

此时各类别**只有截距不同、斜率相同**，相当于一组*平行直线*。二者的差别可以概括为一句：**有无交互项，决定的是各类别的直线能否有不同斜率**。

### 3. 多个分类变量

假设自变量为 $m$ 个分类变量 $F_1,\dots,F_m$，类别数分别为 $d_1,\dots,d_m$，定义虚拟变量 $U_{jk}=I_{\{F_j=k\}}$（$j=1,\dots,m$，$k=2,\dots,d_j$）。可以构造不同**阶数**的模型，**主效应模型**（一阶模型）为

$$
y_i=\beta_0^{*}+\sum_{j=1}^{m}\sum_{k=2}^{d_j}\beta_{jk}^{*}\,u_{ijk}+e_i ;
$$

**二阶模型**在此基础上加入两两分类变量之间的乘积项：

$$
y_i=\beta_0^{*}+\sum_{j=1}^{m}\sum_{k=2}^{d_j}\beta_{jk}^{*}u_{ijk}
+\sum_{1\le j_1<j_2\le m}\ \sum_{k_1=2}^{d_{j_1}}\sum_{k_2=2}^{d_{j_2}}\beta_{j_1j_2k_1k_2}^{*}u_{ij_1k_1}u_{ij_2k_2}+e_i .
$$

依次还可以有三阶模型等，最高是**全模型**（*full model*，也叫 $m$ 阶模型）。各阶模型的结构可以对照下表理解。

| 模型         | 包含的项                  | 参数个数                      |
| :--------- | :-------------------- | :------------------------ |
| 主效应模型（一阶）  | 各分类变量的虚拟变量本身          | $1+\sum_{j=1}^{m}(d_j-1)$ |
| 二阶模型       | 一阶项 + 两两分类变量的虚拟变量乘积   | 在此基础上增加两两交叉项              |
| 全模型（$m$ 阶） | $m$ 个变量各类别**所有组合之乘积** | $\prod_{j=1}^{m}d_j$      |

模型包含哪些项可以用 **Wilkinson-Rogers 表示法**说明。在 UN Data 与 Wool Data 中，各模型的表示如下表。

| 数据 | 模型 | Wilkinson-Rogers 表示 |
| :--- | :--- | :--- |
| UN Data | 主效应 | `lifeExpF ~ group + log(ppgdp)` |
| UN Data | 含交互效应 | `lifeExpF ~ group + log(ppgdp) + group:log(ppgdp)` |
| Wool Data | 主效应 | `log(cycles) ~ len + amp + load` |
| Wool Data | 二阶 | `log(cycles) ~ len + amp + load + len:amp + len:load + amp:load` |
| Wool Data | 全模型 | 二阶再加 `+ len:amp:load` |

对应地，R 代码里先用 `Wool$len <- factor(Wool$len)` 等把三个变量转成**因子**，再用三个 `lm` 分别拟合主效应、二阶与全模型并比较。

> [!example] 例：UN Data（Weisberg §5.1.1）
>
> 用 `m1 <- lm(lifeExpF ~ group * log(ppgdp), UN11)` 拟合*含交互*的模型，用 `m2 <- lm(lifeExpF ~ group + log(ppgdp), UN11)` 拟合*主效应*模型；配合 `Boxplot(lifeExpF ~ group, data=UN11, id.n=2)` 观察各组的分布，以及 `scatterplot(lifeExpF ~ log(ppgdp) | group, ...)` 按组分面画出散点，直观比较各组*截距与斜率*的差异。

## 二、多项式回归

假设只有一个连续自变量 $X$，可以考虑 $d$ 阶的**多项式回归模型**：

$$
y_i=\beta_0^{*}+\beta_1^{*}x_i+\cdots+\beta_d^{*}x_i^{d}+e_i .
$$

若有 $m$ 个连续变量 $X_1,\dots,X_m$，可以考虑**二阶模型**

$$
y_i=\beta_0^{*}+\sum_{j=1}^{m}\beta_j^{*}x_{ij}+\sum_{1\le j_1\le j_2\le m}\beta_{j_1j_2}^{*}x_{ij_1}x_{ij_2}+e_i,
$$

其中包含**平方项**与**交叉乘积项**，当然也可以考虑更高阶的模型。

在 R 中，Geyser data 用 `lm(Interval ~ poly(Duration, d, raw=TRUE), oldfaith)` 依次拟合 $d=1,2,3,4$ 阶多项式，再用 `lines(x, predict(m2, newdata=data.frame(Duration=x)))` 把拟合曲线叠加到散点图上对比。cakes 数据中的二阶模型有*两种等价写法*：

```r
# 显式写出每一项
lm(Y ~ X1 + X2 + I(X1^2) + I(X2^2) + X1:X2, data = cakes)

# 用 polym 一次生成全部二阶项
lm(Y ~ polym(X1, X2, degree = 2, raw = TRUE), data = cakes)
```

其中 `raw = TRUE` 表示使用**原始幂次基**而不是正交多项式基，便于直接解释各项系数。

## 三、样条函数

### 1. 三次样条

多项式回归的问题在于：阶数 $d$ 太大时，在样本 $x_1,\dots,x_n$ 的**范围之外**，多项式函数可能会给出**超出常理的 $y$ 值**，即出现**过拟合**（*overfitting*）。讲义图 2 中，8 次多项式对样本点的拟合更优，但在 $X$ 超出样本取值范围（负方向）时形状明显不合理。改进的想法是：**用分段的低阶多项式做拟合**。

假设在 $X$ 的取值范围 $(a,b)$ 内有 $K$ 个**结点**（*knots*）$a<\xi_1<\cdots<\xi_K<b$（$a$ 可能为 $-\infty$，$b$ 可能为 $\infty$）。若函数 $f(x)$ 在所有区间 $(a,\xi_1),(\xi_1,\xi_2),\dots,(\xi_K,b)$ 之内都是 $d$ 阶多项式，且在 $K$ 个结点处都有 $0$ 到 $(d-1)$ 阶的**连续导数**

$$
f^{(l)}(\xi_k+)=f^{(l)}(\xi_k-),\qquad k=1,\dots,K,\quad l=0,1,\dots,d-1,
$$

则称 $f(x)$ 为一个 $d$ 次的**样条**（*spline*）函数。最常见的是**三次样条**（*cubic spline*），即 $d=3$，此时 $f(x)$ 在每个区间之内是三次多项式，在结点处连续且有连续的一阶、二阶导数。

可以进一步证明，三次样条可以写成一组**基函数**的线性组合：

$$
f(x)=\beta_0+\beta_1x+\beta_2x^{2}+\beta_3x^{3}+\beta_4(x-\xi_1)_{+}^{3}+\cdots+\beta_{K+3}(x-\xi_K)_{+}^{3},
\tag{4}
$$

其中 $(x-\xi)_{+}=x-\xi$（当 $x\ge\xi$），否则取 $0$，称为**截断幂函数**。于是 $f(x)$ 就是基函数 $\{1,x,x^{2},x^{3},(x-\xi_1)_{+}^{3},\dots,(x-\xi_K)_{+}^{3}\}$ 的线性组合。取定结点 $\xi_1,\dots,\xi_K$ 后，参数可以由**最小二乘法**估计：

$$
\min_{\beta_0,\beta_1,\dots,\beta_{K+3}}\sum_{i=1}^{n}\{y_i-f(x_i)\}^{2}.
$$

为保证 $x<\xi_1$ 和 $x>\xi_K$ 时函数不超出合理范围，可以加上约束：$f(x)$ 在 $x<\xi_1$ 和 $x>\xi_K$ 时**为线性函数**，此时 $f(x)$ 称为**自然三次样条**（*natural cubic spline*）。

结点的选择通常有两种取法，可以并列对照。

| 取法 | 端点 $\xi_1,\xi_K$ | 中间结点 |
| :--- | :--- | :--- |
| 按分位数取 | 样本最小值、最大值 | 第 $k$ 个结点取样本的第 $(k-1)/(K-1)$ **分位数**，$2\le k\le K-1$ |
| 按等距取 | 样本最小值、最大值 | $K-2$ 个结点**均匀分布**在 $\xi_1$ 与 $\xi_K$ 之间 |

结点个数 $K$ 通常用**交叉验证**选取（第五讲介绍）。

从更一般的角度看数据拟合，理想情况是找到一个*不假设任何形式*（不必是多项式或分段多项式）的函数 $f(x)$，使 $f(x)$ 既能贴合观测样本又有一定**光滑性**，于是考虑带**惩罚项**的最小二乘法：

$$
\min_{f}\left[\sum_{i=1}^{n}\{y_i-f(x_i)\}^{2}+\lambda\int f''(x)^{2}\right],
\tag{5}
$$

其中 $\lambda>0$，$\lambda\int f''(x)^2$ 一项是对 $f(x)$ **非光滑性**的惩罚。可以证明 (5) 的解就是以 $\xi_1=x_1,\dots,\xi_n=x_n$ 为结点的自然三次样条，称为**平滑样条**（*smoothing spline*）函数，其中的 $\lambda$ 通常也用交叉验证选取。

四种刻画弯曲关系的方法各有取舍，`Wage.R` 里的实现与它们的性质可以对照下表。

| 方法 | 函数形式 | R 实现 | 边界外行为 |
| :--- | :--- | :--- | :--- |
| 多项式回归 | 单一 $d$ 次多项式 | `poly(age, 4)` | 高次时**迅速失控** |
| 三次样条 | 分段三次，结点处二阶导数连续 | `bs(age, knots=c(25,40,60))` | 仍是三次多项式外推 |
| 自然三次样条 | 三次样条 + 边界外线性 | `ns(age, df=5)` | 边界外为线性，**较稳健** |
| 平滑样条 | 惩罚最小二乘的解 | `smooth.spline(age, wage, cv=TRUE)` | 结点取全部样本点，$\lambda$ 由 CV 选 |

> [!example] 例：Wage Data
>
> R 代码 `Wage.R` 完整演示了上述四类方法。用 `lm(wage ~ poly(age, 4), data=Wage)` 拟合 4 次多项式；用 `lm(wage ~ bs(age, knots=c(25,40,60)), data=Wage)` 拟合*固定结点的三次 B 样条*（`bs` 来自 `library(splines)`）；用 `lm(wage ~ ns(age, df=5), data=Wage)` 拟合自然样条，其中 `df=5` 对应 **3 个内部结点**；用 `smooth.spline(age, wage, cv=TRUE)` 拟合平滑样条，由交叉验证自动选择 $\lambda$。每段代码都在样本范围内画一次，再把 `age` 网格扩展到 `c(0,100)` 重画一次，对比可见*高次多项式在样本范围外迅速失控，而样条方法的外推要稳定得多*。

### 2. 基函数

由 (4) 式可以看出，只要把 $f(x)$ 写成**一族基函数的线性组合**，系数就可以用最小二乘法估计。一般地，若有一组**基函数**（*basis functions*）$\{\phi_1(x),\dots,\phi_M(x)\}$，令

$$
f(x)=\sum_{m=1}^{M}\beta_m\phi_m(x),
$$

则可以用最小二乘法估计参数：

$$
\min_{\beta_1,\dots,\beta_M}\sum_{i=1}^{n}\left\{y_i-\sum_{m=1}^{M}\beta_m\phi_m(x_i)\right\}^{2},
$$

这等价于解一个以 $\phi_m(x)$ 为自变量的**多元线性回归问题**。也就是说，**样条、多项式、平滑样条在计算上其实是同一个动作：换一套自变量，跑一次 OLS**。三次样条的另一组常用基函数是 **B-样条**（*B-splines*），详见 Weisberg §5.4。

## 四、主成分分析

假设有 $p$ 个连续自变量 $X_1,\dots,X_p$（均视作随机变量），希望构造它们的线性组合，尽量多地代表这些变量的"信息"。考虑系数向量 $a=(a_1,\dots,a_p)^{\top}\in\mathbb{R}^{p}$，构造随机变量 $Z=\sum_{j=1}^{p}a_jX_j$。$Z$ 中所包含的关于 $X_1,\dots,X_p$ 的信息量用 $Z$ 的**方差**衡量，于是要解

$$
\max_{a}\ \operatorname{Var}(Z)\quad \text{s.t.}\ \|a\|=1 .
$$

记 $(X_1,\dots,X_p)$ 的**协方差矩阵**为 $\Gamma\in\mathbb{R}^{p\times p}$，即 $\Gamma_{jk}=\operatorname{Cov}(X_j,X_k)$，则 $\operatorname{Var}(Z)=a^{\top}\Gamma a$。设 $\Gamma$ 的特征值从大到小为 $\lambda_1\ge\cdots\ge\lambda_p\ge 0$，相应特征向量为 $u_1,\dots,u_p$。可以证明上述优化问题的解为 $\hat a=u_1$，称 $Z_1=\sum_{j=1}^{p}u_{j1}X_j$ 为**第一个主成分**（*the first principal component*），$u_{11},\dots,u_{p1}$ 为它的**载荷**（*loadings*），而目标函数的最优值即为 $\operatorname{Var}(Z_1)=\lambda_1$。

接着在同样的约束之外要求 $a$ 与 $u_1$ **正交**，即求解

$$
\max_{a}\ \operatorname{Var}(Z)\quad \text{s.t.}\ \|a\|=1,\ \langle a,u_1\rangle=0,
$$

解为 $\hat a=u_2$，于是 $Z_2=\sum_{j}u_{j2}X_j$ 为第二个主成分，$\operatorname{Var}(Z_2)=\lambda_2$。一直进行下去可得第三至第 $p$ 个主成分 $Z_3,\dots,Z_p$，且 $\operatorname{Var}(Z_j)=\lambda_j$。因此

$$
\sum_{j=1}^{p}\operatorname{Var}(Z_j)=\sum_{j=1}^{p}\lambda_j=\operatorname{tr}(\Gamma)=\sum_{j=1}^{p}\operatorname{Var}(X_j),
$$

即**全部主成分合起来包含了原变量的全部信息**；若只取前 $q\le p$ 个主成分，则代表的信息占比为

$$
\frac{\sum_{j=1}^{q}\lambda_j}{\sum_{j=1}^{p}\lambda_j}.
$$

实际应用中，对观测样本 $(x_{11},\dots,x_{1p}),\dots,(x_{n1},\dots,x_{np})$ 先计算**样本协方差矩阵** $\hat\Gamma\in\mathbb{R}^{p\times p}$：

$$
\hat\Gamma_{jk}=\frac{1}{n-1}\sum_{i=1}^{n}(x_{ij}-\bar x_j)(x_{ik}-\bar x_k),\qquad 1\le j,k\le p,
$$

对 $\hat\Gamma$ 做**特征值分解**得到 $\hat\lambda_1\ge\cdots\ge\hat\lambda_p$ 与相应的特征向量 $\hat u_1,\dots,\hat u_p$，进而得到第 $j$ 个主成分及其样本值

$$
\hat Z_j=\sum_{k=1}^{p}\hat u_{kj}X_k
\quad\Longrightarrow\quad
\begin{bmatrix}\hat z_{1j}\\ \vdots\\ \hat z_{nj}\end{bmatrix}
=\begin{bmatrix}\sum_{k=1}^{p}\hat u_{kj}x_{1k}\\ \vdots\\ \sum_{k=1}^{p}\hat u_{kj}x_{nk}\end{bmatrix},
$$

这些样本值 $\hat z_{1j},\dots,\hat z_{nj}$ 称为第 $j$ 个主成分的**得分**（*score*）。

> [!warning] 注意
>
> 若各变量的**量纲或方差差异很大**，直接对协方差矩阵做特征值分解会让*方差大的变量主导第一主成分*。此时应改对**相关系数矩阵**做特征值分解，这等价于先把每个变量**标准化**（减去该变量均值，再除以该变量的标准差），再对标准化后的变量做主成分分析。

> [!example] 例：Professor Ratings（Weisberg §5.5）
>
> R 代码取 `x <- Rateprof[, 8:12]` 作为五个评分变量，用 `z <- prcomp(scale(x))` 对标准化后的数据做主成分分析，其中 `scale` 完成了**标准化**这一步。`summary(z)` 给出各主成分解释的**方差占比**，`predict(z)` 给出每个样本的**主成分得分**，`pairs(cbind(x, predict(z)[,1]))` 把第一主成分得分与原变量画在一起，便于解释第一主成分的实际含义（通常可理解为*教师的整体评分水平*）。

---

## 本讲小结

| 主题 | 关键对象 | 一句话要点 |
| :--- | :--- | :--- |
| 分类变量 | 虚拟变量、主效应、交互效应 | 用 $d-1$ 个虚拟变量把类别转成 0-1 自变量，**交互项决定斜率是否随类别变化** |
| 多项式回归 | $\beta_0+\beta_1x+\cdots+\beta_dx^{d}$ | 单一多项式全局刻画弯曲，但**阶数高时边界外不可靠** |
| 样条函数 | 结点、截断幂基、惩罚项 | 分段低阶多项式加光滑约束，**等价于换一组基函数做 OLS** |
| 主成分分析 | 协方差矩阵的特征值分解 | 用少数线性组合承载最大方差，**方差占比衡量信息保留度** |

**贯穿全讲的一条线索**：第四讲的四节看似分散，但都在回答同一个问题——*如何把"关系"写进线性模型的设计矩阵*。分类变量靠虚拟变量，多项式靠幂次项，样条靠截断幂基，主成分靠*从数据中算出来的*载荷方向。
