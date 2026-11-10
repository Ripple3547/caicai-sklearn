# 第七讲 Logistic 回归与非线性回归

> [!info] 本讲概要
>
> **内容概要**：Logistic 回归、广义线性模型、非线性回归
> **对应教材**：Weisberg Sections 12.1, 12.2, 11.1, 11.2, 11.3

| 章节 | 核心问题 |
| :--- | :--- |
| 一、Logistic 回归 | 成功次数或类别型因变量如何与自变量建立回归模型 |
| 二、广义线性模型（不要求） | 正态、二项与 Poisson 等分布如何纳入同一个回归框架 |
| 三、非线性回归 | 均值函数对参数非线性时，如何估计参数并作统计推断 |

---

## 一、Logistic 回归

### 1. 模型设定

设因变量 $Y$ 表示某事件成功的次数，其取值为非负整数。刻画这类计数结果的常见分布是**二项分布**（*binomial distribution*）。记试验次数为 $m$，每次试验的成功概率为 $\theta$，则 $Y\sim\operatorname{Bin}(m,\theta)$ 的概率质量函数为

$$
P(Y=j)=\binom{m}{j}\theta^{j}(1-\theta)^{m-j},\qquad 0\le j\le m,
$$

其期望与方差分别为 $\mathbb{E}(Y)=m\theta$ 和 $\operatorname{Var}(Y)=m\theta(1-\theta)$。当 $m=1$ 时，它退化为成功概率为 $\theta$ 的 **Bernoulli 分布**（*Bernoulli distribution*）；反之，$\operatorname{Bin}(m,\theta)$ 可以看作 $m$ 个独立同分布的 $\operatorname{Bernoulli}(\theta)$ 随机变量之和。

设样本为 $(x_{11},\dots,x_{1p};y_1),\dots,(x_{n1},\dots,x_{np};y_n)$；记 $x_i=(x_{i1},\dots,x_{ip})^{\top}$，并假设各条观测相互独立。二项回归模型写为

$$
\begin{aligned}
y_i&\sim\operatorname{Bin}\{m_i,\theta^{*}(x_i)\},\\
\theta^{*}(x_i)&=K\left(\beta_0^{*}+\sum_{j=1}^{p}x_{ij}\beta_j^{*}\right),
\qquad i=1,\dots,n,
\end{aligned}
\tag{1}
$$

其中 $m_i$ 已知，$\beta_0^{*},\dots,\beta_p^{*}$ 为待估参数。模型 (1) 中，$\beta_0^{*}+\sum_{j=1}^{p}x_{ij}\beta_j^{*}$ 称为**线性预测子**（*linear predictor*），$K(\cdot)$ 称为**核均值函数**（*kernel mean function*），其逆函数 $K^{-1}(\cdot)$ 称为**连接函数**（*link function*）。若记 $z_i=(1,x_{i1},\dots,x_{ip})^{\top}$、$\beta^{*}=(\beta_0^{*},\beta_1^{*},\dots,\beta_p^{*})^{\top}$，则线性预测子可简写为 $z_i^{\top}\beta^{*}$。

将 $K(\cdot)$ 取为 **Logistic 函数**（*logistic function*）

$$
S(z)=\frac{e^{z}}{1+e^{z}},
$$

便得到 **Logistic 回归模型**（*logistic regression model*）：

$$
\theta^{*}(x_i)=\frac{\exp(z_i^{\top}\beta^{*})}{1+\exp(z_i^{\top}\beta^{*})},
\qquad
z_i^{\top}\beta^{*}=\log\frac{\theta^{*}(x_i)}{1-\theta^{*}(x_i)} .
\tag{2}
$$

此时连接函数 $S^{-1}(\theta)=\log\{\theta/(1-\theta)\}$ 称为 **logit 函数**（*logit function*），而 $\theta/(1-\theta)$ 称为**成功优势**（*odds of success*）。因此，Logistic 回归并不是直接令成功概率等于线性预测子，而是令成功概率的 logit 等于线性预测子。

| 模型要素 | 在 Logistic 回归中的形式 |
| :--- | :--- |
| 分布 | $y_i\sim\operatorname{Bin}\{m_i,\theta^{*}(x_i)\}$ |
| 线性预测子 | $z_i^{\top}\beta^{*}$ |
| 连接函数 | $z_i^{\top}\beta^{*}=\log[\theta^{*}(x_i)/\{1-\theta^{*}(x_i)\}]$ |

### 2. 参数估计

Logistic 回归采用极大似然估计。设 $\theta(x_i)=S(z_i^{\top}\beta)$，则似然函数为

$$
L(\beta)=\prod_{i=1}^{n}\binom{m_i}{y_i}\theta(x_i)^{y_i}\{1-\theta(x_i)\}^{m_i-y_i}.
$$

取对数并利用 $\log[\theta(x_i)/\{1-\theta(x_i)\}]=z_i^{\top}\beta$ 与 $\log\{1-\theta(x_i)\}=-\log\{1+\exp(z_i^{\top}\beta)\}$，可得

$$
\ell(\beta)=\sum_{i=1}^{n}\left[\log\binom{m_i}{y_i}+y_i(z_i^{\top}\beta)-m_i\log\{1+\exp(z_i^{\top}\beta)\}\right].
$$

组合数项不含 $\beta$，不影响最大化。因此，最大化对数似然等价于最小化

$$
f(\beta)=\sum_{i=1}^{n}\left[m_i\log\{1+\exp(z_i^{\top}\beta)\}-y_i(z_i^{\top}\beta)\right].
\tag{3}
$$

这个目标函数通常没有最小二乘那样的显式解，需要迭代求解。设在当前估计 $\hat\beta$ 处对 $f$ 作二阶 Taylor 展开：

$$
f(\beta)\approx f(\hat\beta)+\{\nabla f(\hat\beta)\}^{\top}(\beta-\hat\beta)
+\frac{1}{2}(\beta-\hat\beta)^{\top}\{\nabla^{2}f(\hat\beta)\}(\beta-\hat\beta).
$$

直接求导可得

$$
\nabla f(\hat\beta)=\sum_{i=1}^{n}\{m_i\hat\theta(x_i)-y_i\}z_i,
\qquad
\nabla^{2}f(\hat\beta)=\sum_{i=1}^{n}m_i\hat\theta(x_i)\{1-\hat\theta(x_i)\}z_i z_i^{\top}.
$$

进一步令

$$
Y=(y_1,\dots,y_n)^{\top},\qquad
X=\begin{bmatrix}z_1^{\top}\\ \vdots\\ z_n^{\top}\end{bmatrix},
$$

并记 $\hat\mu_i=m_i\hat\theta(x_i)$、$\hat w_i=m_i\hat\theta(x_i)\{1-\hat\theta(x_i)\}$，以及 $\hat\mu=(\hat\mu_1,\dots,\hat\mu_n)^{\top}$、$\widehat W=\operatorname{diag}(\hat w_1,\dots,\hat w_n)$。这里的 $\hat\mu_i$ 与 $\hat w_i$ 分别是当前估计处二项分布的均值与方差。于是，梯度和 Hessian 矩阵可以写成

$$
\nabla f(\hat\beta)=-X^{\top}(Y-\hat\mu),
\qquad
\nabla^{2}f(\hat\beta)=X^{\top}\widehat W X .
$$

把二阶近似式对 $\beta$ 求最小，得到下一点

$$
\hat\beta'=\hat\beta+(X^{\top}\widehat W X)^{-1}X^{\top}(Y-\hat\mu).
\tag{4}
$$

由此形成**迭代重新加权最小二乘法**（*iteratively reweighted least squares*, IRLS）。所谓“重新加权”，就是每一轮都用当前的 $\hat\theta(x_i)$ 重新计算权重 $\hat w_i$，再解一次加权最小二乘形式的更新问题。

1. 取初值 $\hat\beta^{(0)}$，计算 $f(\hat\beta^{(0)})$。
2. 对 $j=0,\dots,J-1$，先计算 $\hat\theta_i^{(j)}=S(z_i^{\top}\hat\beta^{(j)})$，再计算 $\hat\mu_i^{(j)}=m_i\hat\theta_i^{(j)}$ 与 $\hat w_i^{(j)}=m_i\hat\theta_i^{(j)}(1-\hat\theta_i^{(j)})$，最后按 (4) 式更新为 $\hat\beta^{(j+1)}$。
3. 计算 $f(\hat\beta^{(j+1)})$；若 $\lvert f(\hat\beta^{(j+1)})-f(\hat\beta^{(j)})\rvert$ 小于预设的收敛标准 $\epsilon>0$，则停止循环，否则继续迭代。

可以证明 $f(\beta)$ 是凸函数，讲义据此说明 IRLS 收敛到全局最优解。也就是说，Logistic 回归的计算核心不是直接套线性回归公式，而是反复更新均值、方差权重和系数估计。

### 3. 统计推断

在真实参数 $\beta^{*}$ 附近作同样的二阶展开。记 $\mu_i^{*}=m_i\theta^{*}(x_i)$、$w_i^{*}=m_i\theta^{*}(x_i)\{1-\theta^{*}(x_i)\}$，以及 $W^{*}=\operatorname{diag}(w_1^{*},\dots,w_n^{*})$。若 IRLS 收敛到接近 $\beta^{*}$ 的解 $\hat\beta$，则

$$
\hat\beta\approx\beta^{*}+(X^{\top}W^{*}X)^{-1}X^{\top}(Y-\mu^{*}).
$$

由于 $\mathbb{E}(Y-\mu^{*})=0$，因此 $\hat\beta$ 是**渐近无偏**的。又因不同观测相互独立，且 $\operatorname{Var}(y_i-\mu_i^{*})=w_i^{*}$，所以

$$
\operatorname{Cov}(\hat\beta)\approx(X^{\top}W^{*}X)^{-1}.
$$

实际计算时用 $\widehat W$ 替代 $W^{*}$，得到协方差矩阵的估计

$$
\widehat{\operatorname{Cov}}(\hat\beta)=(X^{\top}\widehat W X)^{-1}.
\tag{5}
$$

若 $c_{jj}$ 表示 $(X^{\top}\widehat W X)^{-1}$ 的第 $j+1$ 个对角元（$0\le j\le p$），则 $\hat\beta_j$ 的标准误为 $\operatorname{se}(\hat\beta_j)=\sqrt{c_{jj}}$。据此可以构造近似置信区间与检验的近似拒绝域，其形式与线性回归中的统计推断相似。

Logistic 回归也可以包含分类自变量。设模型含一个连续变量 $X$ 和一个有 $L$ 个类别的分类变量 $F$，全模型写为 $Y\sim X+F+X:F$。对 $F$ 引入虚拟变量 $U_2,\dots,U_L$ 后，线性预测子为

$$
\beta_0+\beta_1x_i+\sum_{j=2}^{L}\beta_{0j}u_{ij}
+\sum_{j=2}^{L}\beta_{1j}u_{ij}x_i .
$$

前一组类别项允许不同类别具有不同截距，后一组交互项允许不同类别具有不同斜率；这与第四讲线性回归中的虚拟变量和交互项解释一致，只是它们现在作用于 logit 尺度。

### 4. 偏差

设 Logistic 回归得到估计量 $\hat\beta$ 与成功概率 $\hat\theta(x_i)$，则第 $i$ 条观测的拟合成功次数为 $\hat y_i=m_i\hat\theta(x_i)$，相应的对数似然记为 $\hat\ell=\ell(\hat\beta)$。作为对照，**饱和模型**（*saturated model*）允许每条观测都有独立的成功概率，其第 $i$ 条观测的成功概率估计为 $\hat\theta_i^{s}=y_i/m_i$，对数似然记为 $\ell^{s}$。

两个模型的似然比检验统计量为

$$
G^{2}=2(\ell^{s}-\hat\ell)
=2\sum_{i=1}^{n}\left\{y_i\log\frac{y_i}{\hat y_i}
+(m_i-y_i)\log\frac{m_i-y_i}{m_i-\hat y_i}\right\}.
\tag{6}
$$

$G^{2}$ 称为 Logistic 回归的**偏差**（*deviance*）。它比较当前模型与饱和模型的拟合优劣，$G^{2}$ 越小，表明当前模型拟合得越好。可以证明，当 $n$ 充分大时，$G^{2}$ 近似服从 $\chi^{2}(n-p-1)$ 分布。

> [!example] 例：Blowdown Data（Weisberg §12.2）
>
> 配套程序 `chapter12.R` 使用 `glm` 拟合二项回归。第一段代码把成功次数与失败次数组成两列因变量，以 `log(d)` 为自变量；第二段代码在 `Blowdown` 数据的 `balsam fir` 子集中拟合含 `log(d)`、`s` 及其交互项的模型。
>
> ```r
> g1 <- glm(cbind(died, m - died) ~ log(d),
>           family = binomial, data = BlowBS)
> g2 <- glm(y ~ (log(d) + s)^2, binomial, Blowdown,
>           subset = spp == "balsam fir")
> c(dev = deviance(g2), df = df.residual(g2))
> ```
>
> 其中 `family = binomial` 指定二项分布，`deviance(g2)` 给出模型偏差，`df.residual(g2)` 给出相应的残差自由度。

### 5. 分类问题

若所有样本都有 $m_i=1$，则 $y_i$ 服从 Bernoulli 分布，只能取 $0$ 或 $1$。把 $Y=0$ 与 $Y=1$ 看作两个类别，Logistic 回归便对应**二分类**（*binary classification*）问题，例如根据病人的临床症状判断是否感染疾病，或根据网络银行操作的特征判断其是否为诈骗行为。

Logistic 回归还可以推广到类别数超过 $2$ 的情形。设 $Y$ 有 $d$ 个类别，编号为 $\{1,\dots,d\}$；记第 $k$ 类的概率为 $\theta^{*(k)}(x_i)$。以第 $1$ 类为基准类，模型设定为

$$
\begin{aligned}
P(y_i=k)&=\theta^{*(k)}(x_i),\qquad k=1,\dots,d,\\
\theta^{*(1)}(x_i)&=\frac{1}{1+\sum_{\ell=2}^{d}\exp\{z_i^{\top}\beta^{*(\ell)}\}},\\
\theta^{*(k)}(x_i)&=\frac{\exp\{z_i^{\top}\beta^{*(k)}\}}
{1+\sum_{\ell=2}^{d}\exp\{z_i^{\top}\beta^{*(\ell)}\}},
\qquad k=2,\dots,d,
\end{aligned}
\tag{7}
$$

其中 $\beta^{*(\ell)}=(\beta_0^{*(\ell)},\beta_1^{*(\ell)},\dots,\beta_p^{*(\ell)})^{\top}$，$\ell=2,\dots,d$ 为待估参数。每个非基准类各有一组系数，而分母共同保证 $d$ 个类别概率之和为 $1$。因此，$y_i$ 可以看作试验次数为 $1$、成功概率向量为 $\{\theta^{*(1)}(x_i),\dots,\theta^{*(d)}(x_i)\}$ 的**多项分布**（*multinomial distribution*）观测，参数估计只需最大化相应的多项分布似然函数。

> [!warning] 注意
>
> 讲义第 6 页在“$Y$ 有 $d$ 个类别”之后将类别编号写为 $\{1,\dots,d-1\}$，但后续公式使用 $k=1,\dots,d$。由类别总数与概率公式可知，编号应为 $\{1,\dots,d\}$。

至此，二项回归、二分类与多分类共用同一个基本思想：先用线性预测子刻画自变量的作用，再通过合适的概率变换得到各类别或成功事件的概率。

## 二、广义线性模型（不要求）

### 1. 指数分布类

Logistic 回归属于一类更一般的回归模型。设因变量 $Y$ 的分布属于**指数分布类**（*exponential class*），其概率密度函数或概率质量函数具有以下形式：

$$
f(y;\eta,\phi)=\exp\left\{\frac{y\eta-b(\eta)}{a(\phi)}+h(y,\phi)\right\},
\tag{8}
$$

其中 $\eta$ 与 $\phi$ 为参数，$b(\eta)$ 与 $a(\phi)$ 为相应参数的函数。由这一形式可以计算

$$
\mathbb{E}(Y)=b'(\eta),
\qquad
\operatorname{Var}(Y)=a(\phi)b''(\eta).
$$

正态分布、二项分布、多项分布、Poisson 分布、指数分布与 Gamma 分布等都可以写成这种形式。讲义给出的三个例子如下。

| 分布 | 参数与函数取法 | $h(y,\phi)$ |
| :--- | :--- | :--- |
| $Y\sim N(\mu,\sigma^{2})$ | $\eta=\mu$，$\phi=\sigma$，$b(\eta)=\eta^{2}/2$，$a(\phi)=\phi^{2}$ | $-\dfrac{y^{2}}{2\sigma^{2}}-\log\sigma-\log\sqrt{2\pi}$ |
| $Y\sim\operatorname{Bin}(m,\theta)$，$m$ 已知 | $\eta=\log\{\theta/(1-\theta)\}$，$\phi=1$，$b(\eta)=m\log(1+e^{\eta})$，$a(\phi)=1$ | $\log\binom{m}{y}$ |
| $Y\sim\operatorname{Po}(\lambda)$ | $\eta=\log\lambda$，$\phi=1$，$b(\eta)=e^{\eta}$，$a(\phi)=1$ | $-\log(y!)$ |

> [!warning] 注意
>
> 讲义第 7 页 Poisson 例的最后一行将 $h(y,\phi)$ 印为 $\log(y!)$，但同一页上一行的展开式为 $\exp\{-\lambda+y\log\lambda-\log(y!)\}$。两相对照，应有 $h(y,\phi)=-\log(y!)$。

### 2. 模型定义与典则连接

设有数据 $(x_1,y_1),\dots,(x_n,y_n)$，其中各 $y_i$ 来自指数分布类中的某个分布，且所有观测共享同一个 $\phi$。记 $z_i=(1,x_{i1},\dots,x_{ip})^{\top}$，并假设参数 $\eta_i$ 与自变量通过均值满足

$$
\mathbb{E}(y_i)=b'(\eta_i)=g^{-1}(z_i^{\top}\beta^{*}).
\tag{9}
$$

函数 $g(\cdot)$ 称为**连接函数**，这一回归模型称为**广义线性模型**（*generalized linear model*, GLM）。若某种 $g(\cdot)$ 进一步使得 $\eta_i=z_i^{\top}\beta^{*}$，则称其为**典则连接函数**（*canonical link*）。也就是说，广义线性模型把“因变量的分布”和“均值的变换”分开设定，再用线性预测子把二者接合起来。

| 因变量分布 | 典则连接的逆函数 | 得到的回归模型 |
| :--- | :--- | :--- |
| $y_i\sim N(\eta_i,\phi^{2})$ | $g^{-1}(\eta)=\eta$ | $\mathbb{E}(y_i)=z_i^{\top}\beta^{*}$，即线性回归 |
| $y_i\sim\operatorname{Bin}(m_i,\theta_i)$ | $g^{-1}(\eta)=e^{\eta}/(1+e^{\eta})$ | $\theta_i=\exp(z_i^{\top}\beta^{*})/\{1+\exp(z_i^{\top}\beta^{*})\}$，即 Logistic 回归 |
| $y_i\sim\operatorname{Po}(\lambda_i)$ | $g^{-1}(\eta)=e^{\eta}$ | $\lambda_i=\exp(z_i^{\top}\beta^{*})$，即 Poisson 回归 |

其中二项分布的连接函数作用于成功概率 $\theta_i$，因而 $\mathbb{E}(y_i)=m_i\theta_i$；正态分布与 Poisson 分布的连接函数则分别直接作用于均值 $\eta_i$ 与 $\lambda_i$。

> [!tip] 要点
>
> 线性回归、Logistic 回归与 Poisson 回归在 GLM 框架中的差别，可以分别归因于因变量分布与连接函数的选择；三者都保留线性预测子 $z_i^{\top}\beta^{*}$。

## 三、非线性回归

### 1. 非线性模型

设自变量为 $x_i=(x_{i1},\dots,x_{ip})^{\top}$，因变量为 $y_i$，$i=1,\dots,n$。**非线性回归模型**（*nonlinear regression model*）写为

$$
y_i=m(x_i;\theta^{*})+e_i,
\qquad
e_1,\dots,e_n\overset{\mathrm{i.i.d.}}{\sim}N(0,\sigma_{*}^{2}),
\tag{10}
$$

其中均值函数 $m(x;\theta^{*})$ 是 $x$ 的非线性函数，$\theta^{*}=(\theta_1^{*},\dots,\theta_k^{*})^{\top}$ 为 $k$ 维参数。例如

$$
m(x;\theta^{*})=\theta_1^{*}+\theta_2^{*}\{1-\exp(-\theta_3^{*}x)\}
$$

含一个自变量与三个参数。这里的“非线性”是指均值函数不再具有线性回归中“设计矩阵乘参数向量”的形式。

### 2. 参数估计与 Gauss–Newton 法

在正态误差假设下，极大似然估计可以转化为最小二乘问题：

$$
\min_{\theta}f(\theta)
=\sum_{i=1}^{n}\{y_i-m(x_i;\theta)\}^{2}.
\tag{11}
$$

与线性回归不同，(11) 通常没有显式解。设在当前估计 $\hat\theta$ 处对均值函数作一阶 Taylor 展开：

$$
m(x_i;\theta)\approx m(x_i;\hat\theta)
+u_i(\hat\theta)^{\top}(\theta-\hat\theta),
\qquad
u_i(\hat\theta)=\frac{\partial m(x_i;\hat\theta)}{\partial\theta}\in\mathbb{R}^{k}.
$$

令 $\hat e_i=y_i-m(x_i;\hat\theta)$，称为**工作残差**（*working residual*），则目标函数近似为

$$
f(\theta)\approx\sum_{i=1}^{n}\{\hat e_i-u_i(\hat\theta)^{\top}(\theta-\hat\theta)\}^{2}.
$$

进一步记 $\hat e=(\hat e_1,\dots,\hat e_n)^{\top}\in\mathbb{R}^{n}$，并以 $u_i(\hat\theta)^{\top}$ 为第 $i$ 行构造 $U(\hat\theta)\in\mathbb{R}^{n\times k}$，就有

$$
f(\theta)\approx\lVert\hat e-U(\hat\theta)(\theta-\hat\theta)\rVert^{2}.
$$

右端是一个关于 $\theta-\hat\theta$ 的普通最小二乘问题，其最小值点给出 **Gauss–Newton 更新**（*Gauss–Newton method*）：

$$
\hat\theta'=\hat\theta+
\{U(\hat\theta)^{\top}U(\hat\theta)\}^{-1}U(\hat\theta)^{\top}\hat e .
\tag{12}
$$

1. 取初值 $\hat\theta^{(0)}$，计算 $f(\hat\theta^{(0)})$。
2. 对 $j=0,\dots,J-1$，计算梯度 $u_i(\hat\theta^{(j)})$ 与工作残差 $\hat e_i^{(j)}=y_i-m(x_i;\hat\theta^{(j)})$，再按 (12) 式更新为 $\hat\theta^{(j+1)}$。
3. 计算 $f(\hat\theta^{(j+1)})$；若 $\lvert f(\hat\theta^{(j+1)})-f(\hat\theta^{(j)})\rvert$ 小于预设的收敛标准 $\epsilon>0$，则停止循环，否则继续迭代。

这一算法有两个需要注意的问题。其一，梯度 $u_i(\hat\theta^{(j)})$ 在某些问题中难以计算，可能需要较复杂的数值方法；其二，初始值的选取较为重要，选取不当容易陷入局部最优。因此，非线性回归的计算核心是把当前参数附近的均值函数局部线性化，再反复求解最小二乘问题，但迭代结果仍依赖梯度计算与初值。

### 3. 统计推断

在真实参数 $\theta^{*}$ 附近作一阶展开，可以把目标函数近似为

$$
f(\theta)\approx\lVert e-U(\theta^{*})(\theta-\theta^{*})\rVert^{2},
$$

其中 $e=(e_1,\dots,e_n)^{\top}$。若 Gauss–Newton 法收敛到近似最小值点 $\hat\theta$，且 $\hat\theta$ 充分接近 $\theta^{*}$，则

$$
\hat\theta\approx\theta^{*}+
\{U(\theta^{*})^{\top}U(\theta^{*})\}^{-1}U(\theta^{*})^{\top}e .
$$

于是 $\hat\theta$ 的近似分布为

$$
N\left(\theta^{*},\sigma_{*}^{2}\{U(\theta^{*})^{\top}U(\theta^{*})\}^{-1}\right).
$$

实际推断时，用 $\hat\sigma^{2}=f(\hat\theta)/(n-k)$ 估计 $\sigma_{*}^{2}$，并用当前估计处的梯度矩阵代替 $U(\theta^{*})$，得到

$$
\widehat{\operatorname{Cov}}(\hat\theta)
=\hat\sigma^{2}\{U(\hat\theta)^{\top}U(\hat\theta)\}^{-1}.
$$

若 $c_{jj}$ 表示 $\{U(\hat\theta)^{\top}U(\hat\theta)\}^{-1}$ 的第 $j$ 个对角元，则 $\hat\theta_j$ 的标准误为 $\operatorname{se}(\hat\theta_j)=\hat\sigma\sqrt{c_{jj}}$。由此可以进一步构造近似置信区间与检验的近似拒绝域，其形式同样与线性回归中的推断相似。

> [!example] 例：Turkey Data（Weisberg §11.3）
>
> 配套程序 `chapter11.R` 使用 `nls` 拟合非线性最小二乘模型。以下代码对应均值函数 $\theta_1+\theta_2\{1-\exp(-\theta_3A)\}$，`start` 给出三个参数的初始值。
>
> ```r
> n1 <- nls(Gain ~ th1 + th2 * (1 - exp(-th3 * A)),
>           data = turk0,
>           start = list(th1 = 620, th2 = 200, th3 = 10))
> summary(n1)
> ```
>
> 程序随后用 `model.matrix(~ as.factor(S) - 1, turkey)` 构造组别指示变量，并比较几种设定：所有组共用三个参数；各组分别拥有三个参数；各组共用截距但其余参数不同；各组共用截距与渐近参数、仅速率参数不同。这些模型均通过 `weights = m` 实施加权拟合，体现了同一非线性均值函数与分类变量、权重设定的组合方式。

---

## 本讲小结

| 主题 | 关键对象 | 一句话要点 |
| :--- | :--- | :--- |
| Logistic 回归 | 二项分布、logit 连接、IRLS | 成功概率的 logit 由线性预测子刻画，参数通过迭代加权更新估计 |
| 偏差与分类 | 饱和模型、$G^{2}$、多分类概率 | 偏差比较当前模型与饱和模型，Logistic 回归也可用于二分类与多分类 |
| 广义线性模型 | 指数分布类、连接函数、典则连接 | 分布与均值变换分开设定，线性回归、Logistic 回归和 Poisson 回归由此统一 |
| 非线性回归 | 非线性均值函数、Gauss–Newton 法 | 在当前参数处局部线性化，再反复求解最小二乘更新 |

**贯穿全讲的一条线索**：线性预测子仍然是自变量进入模型的基本方式，但均值与误差结构不再局限于正态线性回归。Logistic 回归用 logit 连接二项分布，广义线性模型把这一结构推广到指数分布类，非线性回归则进一步允许均值函数以非线性方式依赖参数，并用局部线性化完成估计与推断。
