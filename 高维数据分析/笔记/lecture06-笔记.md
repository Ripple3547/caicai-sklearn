# 第六讲 正则化方法与高维回归分析

> [!info] 本讲概要
>
> **内容概要**：Stein 缩减、岭回归、lasso、弹性网、惩罚项的其它改造、调节参数的选取、高维回归示例
> **对应教材**：Weisberg Chapter 10.2.3

| 章节 | 核心问题 |
| :--- | :--- |
| 一、Stein 缩减 | 放弃一点无偏性，预测误差还能不能再降 |
| 二、正则化方法 | 岭回归、lasso、弹性网各以何种方式缩减系数 |
| 三、其它正则化方法 | 惩罚项还能按什么思路改造 |
| 四、调节参数的选取与高维回归示例 | 调节参数怎么定，方法在高维数据上表现如何 |

---

## 一、Stein 缩减

设总体服从正态分布 $N(\mu,1)$，观测到样本 $Y=(y_1,\dots,y_n)^{\top}$，其中 $n>2$，待估参数为 $\mu$。若有新的样本 $y^{\text{new}}$，它可以写成 $y^{\text{new}}=\mu+e^{\text{new}}$，其中 $e^{\text{new}}\sim N(0,1)$。用 $\mu$ 的估计量 $\hat\mu$ 去预测 $y^{\text{new}}$ 时，照上一讲的分解，预测误差为

$$
\mathrm{PE}(\hat\mu)
=\operatorname{Var}(\hat\mu)+\{\mathbb{E}\hat\mu-\mu\}^{2}+1 .
\tag{1}
$$

一个看上去很好的估计量是 $\hat\mu_0=\bar y=\sum_{i=1}^{n}y_i/n$。$\bar y$ 是 $\mu$ 的**无偏估计**，偏差项为 $0$，方差为 $\operatorname{Var}(\bar y)=1/n$，于是 $\mathrm{PE}(\bar y)=1+1/n$。

值得注意的是，$\bar y$ 同时是如下**只有截距**的最小二乘问题的解：

$$
\min_{\beta}\ \big\lVert Y-\beta\mathbf 1_n\big\rVert^{2}.
$$

也就是说，这个无偏估计本身就是不加任何约束的拟合结果。

由此引出的问题是：能否*牺牲*一些无偏性，换取预测误差的整体下降？考虑把 $\hat\mu_0$ 缩减 $c$ 倍（$0\le c\le1$），得到 $\hat\mu_0'=c\hat\mu_0$。它的方差为 $c^2/n$，偏差平方项为 $(1-c)^2\mu^2$，因此

$$
\mathrm{PE}(\hat\mu_0')=\frac{c^{2}}{n}+(1-c)^{2}\mu^{2}+1 .
\tag{2}
$$

把 (2) 对 $c$ 求导并令其为 $0$，得 $2c/n-2(1-c)\mu^{2}=0$，即最优缩减系数为 $c^{*}=\mu^{2}/(\mu^{2}+1/n)$，此时的预测误差为

$$
\mathrm{PE}(\hat\mu_0')=\frac{\mu^{2}}{\mu^{2}+1/n}\cdot\frac{1}{n}+1\le\mathrm{PE}(\bar y).
$$

只要 $\mu\ne0$，因子 $\mu^{2}/(\mu^{2}+1/n)$ 就严格小于 $1$，缩减后的预测误差**严格小于**不缩减的情形。最优缩减系数还有一个值得注意的性质：$\mu^{2}$ 越大，$c^{*}$ 越接近 $1$，即**真值离 $0$ 越远、需要缩减的幅度越小**；真值靠近 $0$ 时缩减反而更重。

$c^{*}$ 依赖未知的 $\mu^{2}$，不能直接使用。若用 $\bar y^{2}$ 代替 $\mu^{2}$，得到可以计算的缩减估计量

$$
\hat\mu_0''=\frac{\bar y^{2}}{\bar y^{2}+1/n}\,\bar y .
$$

令 $\lambda=1/\bar y^{2}$，则 $\hat\mu_0''$ 恰好是

$$
\min_{\beta}\ \big\lVert Y-\beta\mathbf 1_n\big\rVert^{2}+\lambda\beta^{2}
$$

的解。这相当于在最小二乘目标之后追加一个**惩罚项**（*penalty term*）$\lambda\beta^{2}$：它把估计量从 $\bar y$ 往 $0$ 拉，拉动的幅度由 $\lambda$ 控制，而 $\lambda$ 又由数据通过 $\bar y$ 决定，因此不必事先知道 $\mu^{2}$。

这一特例把后文要用的两个角色都摆了出来：**损失项**决定拟合的好坏，惩罚项决定收缩的强度，两者的相对权重由 $\lambda$ 调节。线性回归中的正则化方法，做的就是把 $\lambda\beta^{2}$ 换成关于整个系数向量的惩罚项。

## 二、正则化方法

受 Stein 缩减法的启发，在线性回归模型中同样可以通过对最小二乘估计进行缩减引入一些偏差，从而减小整体预测误差。把这种"目标函数 + 惩罚项"的做法写成一般形式，就是**正则化**（*regularization*）：在最小二乘的目标函数上追加一个关于回归系数的惩罚项再求最小。加在惩罚项前的系数 $\lambda>0$ 称为**调节参数**（*tuning parameter*），它控制缩减的强度。下面三个方法**只在惩罚项的形状上不同**。

### 1. 岭回归

设回归的样本为 $(x_{11},\dots,x_{1p};y_1),\dots,(x_{n1},\dots,x_{np};y_n)$。**岭回归**（*ridge regression*）考虑如下问题：

$$
\min_{\beta_0,\beta_1,\dots,\beta_p}F
=\sum_{i=1}^{n}\left(y_i-\beta_0-\sum_{j=1}^{p}x_{ij}\beta_j\right)^{2}
+\lambda\sum_{j=1}^{p}\beta_j^{2},
\tag{3}
$$

即在最小二乘的目标函数上加上惩罚项 $\lambda\sum_{j=1}^{p}\beta_j^{2}$。将 $F$ 对 $\beta_0$ 求偏导并令其为 $0$，有

$$
\frac{\partial F}{\partial\beta_0}
=-2\sum_{i=1}^{n}\left(y_i-\beta_0-\sum_{j=1}^{p}x_{ij}\beta_j\right)=0,
$$

于是 $\beta_0=\bar y-\sum_{j=1}^{p}\bar x_j\beta_j$。把它代回 (3) 的目标函数，得到

$$
\sum_{i=1}^{n}\left\{(y_i-\bar y)-\sum_{j=1}^{p}(x_{ij}-\bar x_j)\beta_j\right\}^{2}
+\lambda\sum_{j=1}^{p}\beta_j^{2}.
$$

采用第三讲的中心化符号，目标函数化为 $\lVert Y-Xb\rVert^{2}+\lambda\lVert b\rVert^{2}$。对 $b$ 求偏导并令其为 $0$，得 $-2X^{\top}(Y-Xb)+2\lambda b=0$，解得

$$
\hat b_{\text{ridge}}=\left(X^{\top}X+\lambda I\right)^{-1}X^{\top}Y .
\tag{4}
$$

当 $\lambda=0$ 时，(4) 退化为最小二乘估计量 $\hat b_{\text{ols}}=(X^{\top}X)^{-1}X^{\top}Y$，因此普通最小二乘可以看作岭回归在 $\lambda=0$ 处的特例。

> [!note] 补充
>
> 惩罚项只对 $\beta_1,\dots,\beta_p$ 求和，**不含截距** $\beta_0$；上面的推导进一步显示，$\beta_0$ 由均值吸收，缩减只作用在斜率上。另外，$X^{\top}X+\lambda I$ 对任意 $\lambda>0$ 都正定，因此岭回归在 $p>n$（此时 $X^{\top}X$ 奇异）时**依然有唯一的解**。

现在考虑一种理想化的情形：假设 $X$ 的所有 $p$ 列都标准化（即每一列的欧式模为 $1$），且列与列之间相互正交，则 $X^{\top}X=I$。此时 $\hat b_{\text{ols}}=X^{\top}Y$，而

$$
\hat b_{\text{ridge}}=(I+\lambda I)^{-1}X^{\top}Y
=\frac{1}{1+\lambda}X^{\top}Y
=\frac{1}{1+\lambda}\hat b_{\text{ols}} .
$$

可见岭回归把最小二乘估计的每一个分量都按**同一个比例** $1/(1+\lambda)$ 缩减，这与 Stein 缩减的形式一致：两者都是把无偏估计整体朝 $0$ 拉。比例恒为正，所以岭回归的系数**只会被压缩、不会恰好取零**。

### 2. Lasso

若要让估计量本身具备筛选变量的能力，惩罚项就必须能把一部分系数正好压到 $0$，而这一点岭回归做不到。Tibshirani (1996) 提出的 **lasso**（*least absolute shrinkage and selection operator*）改用 $L_1$ 惩罚：

$$
\min_{\beta_0,\beta_1,\dots,\beta_p}G
=\frac{1}{2}\sum_{i=1}^{n}\left(y_i-\beta_0-\sum_{j=1}^{p}x_{ij}\beta_j\right)^{2}
+\lambda\sum_{j=1}^{p}\lvert\beta_j\rvert .
\tag{5}
$$

经过与岭回归类似的处理，目标函数转化为

$$
\min_{b}\ \frac{1}{2}\lVert Y-Xb\rVert^{2}+\lambda\lVert b\rVert_{1},
$$

其中 $\lVert b\rVert_{1}=\sum_{j=1}^{p}\lvert\beta_j\rvert$ 为 $b$ 的 $L_1$ 模。因此 lasso 也称为 $L_1$ 模的**正则化方法**（*$L_1$-regularization*），岭回归则相应地称为 $L_2$ 模的正则化方法。

以上优化问题还等价于

$$
\min_{b}\ \frac{1}{2}\lVert Y-Xb\rVert^{2}\quad\text{s.t.}\ \lVert b\rVert_{1}\le s,
$$

其中 $s$ 可看作分配给 $\beta_1,\dots,\beta_p$ 的**总配额**。以 $p=2$ 为例，约束 $\lVert b\rVert_{1}\le s$ 在 $(\beta_1,\beta_2)$ 平面上围成一个正方形，而目标函数的等值线 $\{b:\lVert Y-Xb\rVert^{2}=c\}$ 是一族椭圆，$c$ 越大椭圆越大；需要找到最小的 $c$，使椭圆与正方形相交。这个正方形的两条对角线正好落在坐标轴上，因此椭圆常常先与某个顶点相切，对应的解里会出现恰好为 $0$ 的系数，**这正是 lasso 能做变量选择的几何来源**（讲义图 1）。

如果 $X$ 的各列标准正交，则 $X^{\top}X=I$，$\hat b_{\text{ols}}=X^{\top}Y$，此时 lasso 的目标函数变为

$$
\frac{1}{2}\lVert Y-Xb\rVert^{2}+\lambda\lVert b\rVert_{1}
=\frac{1}{2}Y^{\top}Y+\frac{1}{2}\sum_{j=1}^{p}\left(\beta_j^{2}-2\hat\beta_j^{\text{ols}}\beta_j+2\lambda\lvert\beta_j\rvert\right),
$$

右端每个分项只含单独的 $\beta_j$，于是可以对各分量分别优化，得到

$$
\hat\beta_j^{\text{lasso}}=
\begin{cases}
\operatorname{sign}\left(\hat\beta_j^{\text{ols}}\right)\left(\left\lvert\hat\beta_j^{\text{ols}}\right\rvert-\lambda\right), & \left\lvert\hat\beta_j^{\text{ols}}\right\rvert\ge\lambda,\\
0, & \left\lvert\hat\beta_j^{\text{ols}}\right\rvert<\lambda .
\end{cases}
\tag{6}
$$

由 (6) 式可以看出两点。其一，在输入变量标准正交时，lasso 只是把每个最小二乘系数朝 $0$ 平移 $\lambda$：平移后仍未越过 $0$ 的保留并缩短，越过 $0$ 的直接取 $0$，于是起到**变量选择**的作用。其二，$\lambda$ 越大，对非零系数的惩罚越重，**估计量中的 $0$ 越多、选出的变量个数越少**。讲义图 2 把两者的缩减特性画在一起：岭回归的曲线是一条过原点的斜线，lasso 的曲线则在原点附近有一段恒为 $0$ 的区间。

### 3. 弹性网

Lasso 有几个局限。其一，**最多只能选出 $n$ 个变量**；其二，在一组相关性很高的变量中通常只选其中一个，从而忽略其它的变量；其三，在传统 $n>p$ 的情况下预测效果不太好。Zou and Hastie (2005) 提出的**弹性网**（*elastic net*）把岭回归与 lasso 的惩罚项**线性组合**起来：

$$
\min_{b}\ \frac{1}{2}\lVert Y-Xb\rVert^{2}
+\lambda\left\{\alpha\lVert b\rVert_{1}+\frac{1}{2}(1-\alpha)\lVert b\rVert_{2}^{2}\right\},
\tag{7}
$$

其中 $\lambda$ 与 $\alpha$ 均为调节参数，$\alpha$ 控制 $L_1$ 成分所占的比重：$\alpha=1$ 时退化为 lasso，$\alpha=0$ 时退化为岭回归。讲义图 3 给出 $p=2$ 时三个方法惩罚项的等高线：岭回归是圆，lasso 是菱形，弹性网介于两者之间，菱形被磨圆但仍在坐标轴上保留"棱"。

若自变量矩阵 $X$ 的列都标准正交，弹性网的解为

$$
\hat\beta_j^{\text{enet}}=
\begin{cases}
\operatorname{sign}\left(\hat\beta_j^{\text{ols}}\right)\dfrac{\left\lvert\hat\beta_j^{\text{ols}}\right\rvert-\lambda\alpha}{1+\lambda(1-\alpha)}, & \left\lvert\hat\beta_j^{\text{ols}}\right\rvert\ge\lambda\alpha,\\
0, & \left\lvert\hat\beta_j^{\text{ols}}\right\rvert<\lambda\alpha .
\end{cases}
\tag{8}
$$

> [!warning] 注意
>
> 与 (6) 式一样，(8) 式成立的前提是 $X$ 的列标准正交。一般情形下 $X^{\top}X\ne I$，这两个式子不再成立，系数需要另行求解。

与 (6) 式对照可以看出，弹性网相当于先做一次 lasso 式的选择、再做一次岭回归式的缩减：置零的阈值从 $\lambda$ 降到 $\lambda\alpha$，非零部分再除以 $1+\lambda(1-\alpha)$ 做等比压缩。相对于 lasso，弹性网的一个显著特点是**一组相关性很高的变量的回归系数也会相似**，这一点见讲义图 4 的缩减特性与图 5 的变量选择对比（Zou and Hastie, 2005）。

> [!tip] 要点
>
> 三个方法的差别集中在惩罚项上：岭回归用 $L_2$ 惩罚，把系数等比压缩但一个也不剔除；lasso 用 $L_1$ 惩罚，把小的系数压到 $0$ 以完成筛选，但在强相关变量之间只保留一个；弹性网用两者的组合，用 $\alpha$ 在压缩与筛选之间调节。

## 三、其它正则化方法

上面三个方法共用一个模板：以最小二乘为主项，以一个关于系数向量的模作惩罚项。讲义随后给出几种对惩罚项的改造，各自针对前面方法的某一处不足；以下只保留惩罚项的形式与动机、以及正交情形下的主要结论，不再重述推导。

### 1. 自适应 lasso

lasso 对每个系数施加相同的惩罚力度，不论该变量是否真正重要。Zou (2006) 提出的**自适应 lasso**（*adaptive lasso*）给每个系数配上权重 $w_j>0$：

$$
\min_{\beta_1,\dots,\beta_p}\ \frac{1}{2}\sum_{i=1}^{n}\left(y_i-\sum_{j=1}^{p}x_{ij}\beta_j\right)^{2}
+\lambda\sum_{j=1}^{p}w_j\lvert\beta_j\rvert .
$$

若取 $w_j=1/\lvert\beta_j\rvert^{\gamma}$（$\gamma>0$），目标函数变成

$$
\min_{\beta_1,\dots,\beta_p}\ \frac{1}{2}\sum_{i=1}^{n}\left(y_i-\sum_{j=1}^{p}x_{ij}\beta_j\right)^{2}
+\lambda\sum_{j=1}^{p}\lvert\beta_j\rvert^{1-\gamma},
$$

相当于用 $L_{1-\gamma}$ 模作为正则化项，通常产生比 lasso **更稀疏**的解；但该目标函数**非凸**，求解比较困难。

实际做法是用一个已知的先导估计量 $\hat\beta_j$ 代替真值，取 $w_j=1/\lvert\hat\beta_j\rvert^{\gamma}$：先导估计大的系数得到小的权重、惩罚较轻，接近 $0$ 的系数得到大的权重、更容易被压到 $0$。$n>p$ 时可取 OLS 估计作为先导，$n\le p$ 时可取简单线性回归 $Y\sim X_j$ 的斜率（Huang et al., 2008）。讲义图 6 展示了自适应 lasso 的模拟实验结果。

### 2. 分组 lasso

如果有关于变量分组的信息，希望一组变量**同时被选或同时不被选**，例如变量中含一个因子时，希望代表这个因子的一组虚拟变量同时进出模型。Yuan and Lin (2006) 提出的**分组 lasso**（*group lasso*）为

$$
\min_{\beta_1,\dots,\beta_p}\ \frac{1}{2}\sum_{i=1}^{n}\left(y_i-\sum_{j=1}^{p}x_{ij}\beta_j\right)^{2}
+\lambda\sum_{\ell=1}^{L}\lVert\beta_{\ell}\rVert_{K_{\ell}},
$$

其中 $\beta_{\ell}$ 表示第 $\ell$ 组变量的系数，$\lVert\cdot\rVert_{K_{\ell}}$ 为某种模（例如可以取 $L_2$ 模）。当 $p=3$、前两个系数同组而第三个单独成组时，三个方法的惩罚项对比如下表。

| 方法 | 惩罚项 |
| :--- | :--- |
| lasso | $\lvert\beta_{11}\rvert+\lvert\beta_{12}\rvert+\lvert\beta_{2}\rvert$ |
| 岭回归 | $\beta_{11}^{2}+\beta_{12}^{2}+\beta_{2}^{2}$ |
| 分组 lasso | $\sqrt{\beta_{11}^{2}+\beta_{12}^{2}}+\lvert\beta_{2}\rvert$ |

组内的两个系数以 $L_2$ 模整体出现，组与组之间以 $L_1$ 模相加，于是同组的系数只能一起进入或一起退出惩罚；这与 lasso 逐个处理系数、逐个清零的做法形成对照（讲义图 7）。

### 3. SCAD

当 $\lambda$ 比较大时，lasso 的估计偏差通常比较大。为了修正这一点，Fan and Li (2001) 提出 **SCAD**（*smoothly clipped absolute deviation*）：

$$
\min_{\beta_1,\dots,\beta_p}\ \frac{1}{2}\sum_{i=1}^{n}\left(y_i-\sum_{j=1}^{p}x_{ij}\beta_j\right)^{2}
+\sum_{j=1}^{p}f_{\lambda}\left(\lvert\beta_j\rvert\right),
$$

其中

$$
f_{\lambda}'(\theta)=
\begin{cases}
\lambda, & \theta\le\lambda,\\
\dfrac{(a\lambda-\theta)_{+}}{a-1}, & \theta>\lambda,
\end{cases}
$$

而 $a>2$ 为选定参数。SCAD 的惩罚项**不是凸函数**。从 $f_{\lambda}'$ 可以看出，lasso 的惩罚率恒为 $\lambda$，而 SCAD 在 $\theta>\lambda$ 之后惩罚率线性衰减、到 $\theta=a\lambda$ 处降为 $0$，即**对大系数不再额外惩罚**，大系数上的偏差正是由此减轻的（讲义图 8 对比两者的惩罚项）。

若 $X$ 的各列标准正交，SCAD 的解为

$$
\hat\beta_j^{\text{scad}}=
\begin{cases}
0, & \left\lvert\hat\beta_j^{\text{ols}}\right\rvert\le\lambda,\\
\operatorname{sign}\left(\hat\beta_j^{\text{ols}}\right)\left(\left\lvert\hat\beta_j^{\text{ols}}\right\rvert-\lambda\right), & \lambda<\left\lvert\hat\beta_j^{\text{ols}}\right\rvert\le2\lambda,\\
\dfrac{(a-1)\hat\beta_j^{\text{ols}}-\operatorname{sign}\left(\hat\beta_j^{\text{ols}}\right)a\lambda}{a-2}, & 2\lambda<\left\lvert\hat\beta_j^{\text{ols}}\right\rvert\le a\lambda,\\
\hat\beta_j^{\text{ols}}, & \left\lvert\hat\beta_j^{\text{ols}}\right\rvert>a\lambda .
\end{cases}
$$

可以看出，SCAD 解的最后一段**直接取最小二乘估计**，对较大的系数不再做任何缩减；倒数第二段则是从 lasso 解向 OLS 解的线性过渡，起到偏差修正的作用（讲义图 9）。模拟实验结果见讲义图 10。

这三种改造分别针对权重、分组结构与偏差：自适应 lasso 让惩罚力度随系数大小变化，分组 lasso 让惩罚作用在整组变量上，SCAD 让惩罚在大系数处停止。它们与前面三个方法共用同一个"主项 + 惩罚项"的框架，改造的只是惩罚项本身。

## 四、调节参数的选取与高维回归示例

### 1. 调节参数的选取

正则化方法里的调节参数通常用**交叉验证**选取（第五讲）。若只有一个调节参数 $\lambda$，可以先取一组候选值 $\{\lambda_1,\dots,\lambda_M\}$，对每个 $\lambda_m$ 计算其对应的 $\mathrm{CV}(\lambda_m)$，最后取使 $\mathrm{CV}(\lambda_m)$ 最小的 $\lambda_m$。若有两个调节参数 $\lambda$ 与 $\alpha$，则各取一组值 $\{\lambda_1,\dots,\lambda_M\}$ 与 $\{\alpha_1,\dots,\alpha_L\}$，对每一对数 $(\lambda_m,\alpha_l)$ 计算 $\mathrm{CV}(\lambda_m,\alpha_l)$，再取使 CV 最小的数对。这里的"最小 CV 规则"可以替换为**1 标准误规则**，即在 CV 值靠近最小值的候选中选取正则化最强的一个。

### 2. 高维回归示例

讲义给出两个实际数据上的示例，程序在 `lecture06-code` 目录中。两个例子共用同一条流程：整理设计矩阵、划分训练集与测试集、用交叉验证选调节参数、最后在测试集上比较样本外误差。

> [!example] 例：Hitters Data
>
> `Hitters.R` 先用 `model.matrix(Salary ~ ., Hitters)[, -1]` 把数据整理成设计矩阵，再用 `glmnet(x, y, alpha = 0, lambda = grid)` 拟合岭回归，其中 `alpha = 0` 指定 $L_2$ 惩罚，`grid` 取 `10^seq(10, -2, length = 100)`。代码取出 `lambda` 网格中第 $50$ 与第 $60$ 个位置上的系数，并用 `sqrt(sum(coef(ridge.mod)[-1, 50]^2))` 计算它们的 $L_2$ 模，用来观察 $\lambda$ 变化时系数被压向 $0$ 的程度。
>
> ```r
> grid <- 10^seq(10, -2, length = 100)
> ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid)
> set.seed(1)
> train <- sample(1:nrow(x), nrow(x)/2)
> cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)
> bestlam <- cv.out$lambda.min
> ```
>
> 把样本随机对半分之后，`cv.glmnet` 在训练集上用交叉验证给出 `lambda.min`，再在测试集上计算 `mean((ridge.pred - y.test)^2)` 作为样本外 MSE。代码先取 `s = 4` 算一次，再把 `s` 换成 `1e10`（系数几乎全部收缩到 $0$）与 `s = 0`（不收缩，即 OLS，需加 `exact = TRUE`），三种设定的测试误差可以直接比较。lasso 部分只把 `alpha` 换成 $1$，流程完全相同，最后用 `lasso.coef[lasso.coef != 0]` 列出被选中的变量；`alpha` 取 $(0,1)$ 中间的值即为弹性网。

> [!example] 例：Crime Data
>
> `crime_regression.R` 从 `crime.RData` 中取出不含 police 信息的自变量子集与两个因变量 `nonViolPerPop`、`violentPerPop`，按 $7:3$ 随机划分训练集与测试集。对每个因变量依次拟合三个模型：`lm` 给出的 OLS、`glmnet(..., alpha = 0)` 给出的岭回归、`glmnet(..., alpha = 1)` 给出的 lasso，后两者的 $\lambda$ 都由 `cv.glmnet` 的 `lambda.min` 确定，样本外误差统一按测试集 RMSE 计算。
>
> ```r
> err_ols   <- sqrt(mean((y_test - y_ols)^2))
> err_ridge <- sqrt(mean((y_ridge - y_test)^2))
> err_lasso <- sqrt(mean((y_lasso - y_test)^2))
> ```
>
> `plot(fit_lasso)` 画出的**正则化路径**（*regularization path*）可以看各变量的系数随 $\lambda$ 增大依次被压到 $0$ 的次序。代码还做了两个对照：其一是把 police 相关信息加入自变量，比较同一套方法在有、无 police 信息时的测试误差；其二是只用 Pennsylvania 的样本训练 lasso，再分别在这批样本与其余各州上评估，用来观察在一个子总体上调好的模型迁移到另一部分数据时的表现。

讲义附带的 `simulation.R` 把设定推到 $p>n$ 的情形：$n=100$、$p=1000$，真实系数只有 $9$ 个非零，重复 $30$ 次模拟。每次生成的自变量为 $x=z+\eta w$，各列共享同一个公共成分 $w$，因此彼此高度相关；调节参数用 `cv.glmnet` 的 `lambda.1se` 选取，代码记录每次的样本外 RMSE 与选出的变量个数，最后输出它们的平均值，用于比较 lasso 与岭回归在变量选择与预测两个方面的差别。

这几个程序的骨架是相同的：整理设计矩阵、划分训练集与测试集、交叉验证选 $\lambda$、在测试集上评估。正则化方法在高维数据上的实施，基本上就是把这三步重复到不同的惩罚项上去。

---

## 本讲小结

| 主题 | 关键对象 | 一句话要点 |
| :--- | :--- | :--- |
| Stein 缩减 | 缩减系数 $c$、惩罚项 $\lambda\beta^{2}$ | 无偏不是最优，缩减以一点偏差换取方差的下降 |
| 正则化方法 | 岭回归、lasso、弹性网 | 惩罚项分别是 $L_2$、$L_1$ 与两者的组合 |
| 其它正则化方法 | 自适应 lasso、分组 lasso、SCAD | 改动的是惩罚项的权重、分组与形状 |
| 实施 | 交叉验证、测试集误差 | 选参数用 CV，比较方法好坏的统一尺度是样本外误差 |

**贯穿全讲的一条线索**：所有方法都在同一个式子上做文章，即损失项加上惩罚项。损失项决定拟合，惩罚项决定收缩，$\lambda$ 决定两者的相对权重；岭回归、lasso、弹性网以及后面几种改造，区别只在惩罚项把系数往 $0$ 拉的方式。Stein 缩减给出这一思路的原型，正交设计下的闭式解把"拉"的机制看得最清楚，而一般情形下这种机制要靠数值方法实现。

**参考文献**：R. Tibshirani. Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(4):267–288, 1996；H. Zou and T. Hastie. Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2):301–320, 2005；H. Zou. The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476):1418–1429, 2006.

**后续方法**：M. Yuan and Y. Lin. Model selection and estimation in regression with grouped variables. *Journal of the Royal Statistical Society: Series B*, 68:49–67, 2006；J. Fan and R. Li. Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456):1348–1360, 2001；J. Huang, S. Ma, and C.-H. Zhang. Adaptive lasso for sparse high-dimensional regression models. *Statistica Sinica*, 18:1603–1618, 2008.
