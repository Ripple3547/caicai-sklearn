# 第五讲 变量选择

> [!info] 本讲概要
>
> **内容概要**：Bias-variance Tradeoff、模型选择准则、逐步回归
> **对应教材**：Weisberg Chapter 10

| 章节 | 核心问题 |
| :--- | :--- |
| 一、Bias-variance Tradeoff | 变量多拟合好，*为什么预测反而可能变差* |
| 二、模型选择准则 | 用哪个数值标准判断"该选几个变量" |
| 三、逐步回归 | 在穷举不可行时，怎样**高效地挑出具体变量** |

---

## 一、Bias-variance Tradeoff

假设真实的线性回归模型为

$$
y_i=\beta_0^{*}+\sum_{j=1}^{p^{*}}x_{ij}\beta_j^{*}+e_i,\qquad 1\le i\le n,
$$

其中 $\operatorname{Var}(e_i)=\sigma^{2}$。但现有数据中除了以上 $p^{*}$ 个自变量以外，还有其它自变量，数据中一共有 $p>p^{*}$ 个自变量，问题是如何从中选择自变量加入模型。

由多元线性回归的知识可知，**自变量加的个数越多，残差平方和 RSS 越小、回归的 $R^{2}$ 越大**，表明回归拟合得越好。**但是预测的效果如何？** 记 $x_i=(x_{i1},\dots,x_{ip})^{\top}$、$f^{*}(x_i)=\beta_0^{*}+\sum_{j=1}^{p^{*}}x_{ij}\beta_j^{*}$，真实模型为 $y_i=f^{*}(x_i)+e_i$。假设由样本估计出来的均值函数为 $\hat f(x_i)$（估计时*可能没有用到全部 $p$ 个自变量*）。对于新观测到的自变量 $x_{\text{new}}$，因变量的真实值应为

$$
y_{\text{new}}^{*}=f^{*}(x_{\text{new}})+e_{\text{new}},
$$

而按照估计的均值函数，我们对均值函数的预测值为 $\hat y_{\text{new}}=\hat f(x_{\text{new}})$。因此可以定义**预测误差**（*Prediction Error*）

$$
\mathrm{PE}=\mathbb{E}\left(\hat y_{\text{new}}-y_{\text{new}}^{*}\right)^{2}.
$$

对 PE 做分解，可以得到

$$
\begin{aligned}
\mathrm{PE}
&=\mathbb{E}\left\{\hat f(x_{\text{new}})-f^{*}(x_{\text{new}})-e_{\text{new}}\right\}^{2}\\
&=\mathbb{E}\left\{\hat f(x_{\text{new}})-f^{*}(x_{\text{new}})\right\}^{2}+\operatorname{Var}(e_{\text{new}})\\
&=\mathbb{E}\left\{\hat f(x_{\text{new}})-\mathbb{E}\hat f(x_{\text{new}})+\mathbb{E}\hat f(x_{\text{new}})-f^{*}(x_{\text{new}})\right\}^{2}+\sigma_{*}^{2}\\
&=\operatorname{Var}\left\{\hat f(x_{\text{new}})\right\}
+\mathbb{E}\left\{\mathbb{E}\hat f(x_{\text{new}})-f^{*}(x_{\text{new}})\right\}^{2}+\sigma_{*}^{2}.
\end{aligned}
$$

分解出的三项各有名称与含义，可以整理成下表。

| 名称 | 表达式 | 含义 |
| :--- | :--- | :--- |
| **方差项**（*variance*） | $\operatorname{Var}\{\hat f(x_{\text{new}})\}$ | 重复抽样时预测值围绕其期望的波动程度 |
| **偏差平方项**（$\mathrm{bias}^{2}$） | $\mathbb{E}\{\hat f(x_{\text{new}})-f^{*}(x_{\text{new}})\}^{2}$ | 预测的期望与真实均值函数的偏离 |
| **不可消减误差**（*irreducible error*） | $\sigma_{*}^{2}$ | 新观测本身噪声的方差，**任何方法都无法消除** |

关于这三项的含义，有几点说明。第一，若对数据进行**重新抽样**，得到一组新的 $(x_1,y_1),\dots,(x_n,y_n)$，则会拟合出*不同的* $\hat f$，也就会有不同预测值 $\hat f(x_{\text{new}})$；大量重复抽样后所有 $\hat f(x_{\text{new}})$ 的平均值即为 $\mathbb{E}\hat f(x_{\text{new}})$，而重复抽样过程中 $\hat f(x_{\text{new}})$ 与其均值的偏离程度可用 $\operatorname{Var}\{\hat f(x_{\text{new}})\}$ 表示。第二，**模型灵活度**（*model flexibility*）越大（例如拥有更多参数），偏差平方项**越小**，但方差项**越大**：若取 $\hat f(x)\equiv 0$，即模型*没有任何可调节的参数*，则预测的方差项为 $0$，但偏差会较大；若让模型中有至少 $n$ 个参数，使得 $\hat f(x_i)=y_i$ 对所有 $1\le i\le n$ 都成立，则预测偏差会比较小，但方差会比较大，因为一旦换一组 $y_1,\dots,y_n$，参数值会变化较大，导致预测波动大。第三，通常**很难做到方差项与偏差平方项都小**，所以需要在两者之间寻求一种权衡，使总预测误差尽可能小，这种权衡就叫做 **bias-variance tradeoff**。

两种极端情形的对比有助于记住这个权衡：

| 模型 | 参数个数 | variance | $\mathrm{bias}^{2}$ |
| :--- | :--- | :--- | :--- |
| $\hat f(x)\equiv 0$ | $0$ | $0$ | *较大* |
| 插值模型 $\hat f(x_i)=y_i$ | 至少 $n$ | *较大* | *较小* |

讲义用**多项式回归**具体说明 bias 与 variance 的含义。假设只有一个自变量 $x$，真实模型为 $y_i=f^{*}(x_i)+e_i$，其中 $f^{*}(\cdot)$ 为 $d^{*}$ 次多项式。现在用 $d$ 次多项式函数拟合一组数据 $(x_1,y_1),\dots,(x_n,y_n)$，得到 $f^{*}(x)$ 的估计 $\hat f(x)$。**先固定 $x_1,\dots,x_n$ 不变**，每次生成一组新的 $y_1,\dots,y_n$，共生成四组数据，对应讲义图 1 上排四张图中的彩色点；对每组数据都能得到一个不同的 $\hat f(x)$（对应彩色实线）。于是对于某个 $x_{\text{new}}$，预测值 $\hat f(x_{\text{new}})$ 会有起伏，这个起伏的程度就是 **variance**：

$$
\text{Variance}=\operatorname{Var}\left\{\hat f(x_{\text{new}})\right\},
$$

它可以由各组 $\hat f(x_{\text{new}})$ 值的**样本方差**来近似。而 **bias** 为预测值 $\hat f(x_{\text{new}})$ 的期望与真实值 $f^{*}(x_{\text{new}})$ 的差，

$$
\text{Bias}=\mathbb{E}\left\{\hat f(x_{\text{new}})\right\}-f^{*}(x_{\text{new}}),
$$

它可以由各组 $\hat f(x_{\text{new}})$ 的**样本均值减去真实值** $f^{*}(x_{\text{new}})$ 来近似。讲义图 1 下排最后一张图中，四组数据对应的 $\hat f(x_{\text{new}})$ 平均值与真实值几乎重合于黑点，所以 bias **几乎为零**。

至于灵活度与这两项的关系，一般结论是：**模型越 flexible，variance 越大而 $\mathrm{bias}^{2}$ 越小；反之模型越不 flexible，variance 越小而 $\mathrm{bias}^{2}$ 越大**。模型的 flexibility 通常可以用**参数个数**来衡量。仍以多项式为例，讲义图 2 的三张子图分别从不同角度展示这一关系。

| 子图 | 内容 | 可观察到的现象 |
| :--- | :--- | :--- |
| 左图 | 样本点、真实曲线与三条拟合曲线 | 线性拟合（黄）*最差*，高次多项式（绿）对样本点拟合*最好* |
| 中图 | 横轴为多项式次数，纵轴为 MSE | 拟合误差（灰线）**单调下降**，预测误差（红线）**先减后增** |
| 右图 | 把预测误差分解为 $\mathrm{bias}^{2}$ 与 variance | $\mathrm{bias}^{2}$（青）**递减**，variance（橙）**递增** |

> [!tip] 要点
>
> RSS 或 $R^{2}$ 只看**拟合**，不看**预测**；PE 才是选择模型真正该关心的量，而它由 variance、$\mathrm{bias}^{2}$ 与 $\sigma_{*}^{2}$ 三部分构成。变量选择要做的，就是在三者之和上找**最低点**。

## 二、模型选择准则

想要决定哪些变量进入模型，这个过程叫做**模型选择**（*model selection*）。下面介绍四个模型选择准则，先给出总览。

| 准则               | 定义式                                                    | 惩罚方式                             | 选择方向    |
| :--------------- | :----------------------------------------------------- | :------------------------------- | :------ |
| Adjusted $R^{2}$ | $1-\dfrac{n-1}{n-k-1}(1-R^{2})$                        | 系数 $\dfrac{n-1}{n-k-1}$ 随 $k$ 放大 | 取**最大** |
| AIC              | $n\log\left(\mathrm{RSS}/n\right)+2(k+1)$              | 加性惩罚 $2(k+1)$                    | 取**最小** |
| BIC              | $n\log\left(\mathrm{RSS}/n\right)+(k+1)\log n$         | 加性惩罚 $(k+1)\log n$               | 取**最小** |
| $k$-fold CV      | $\dfrac{1}{k}\sum_{i=1}^{k}\widehat{\mathrm{PE}}_i(m)$ | *不显式惩罚*，直接估预测误差                  | 取**最小** |

### 1. Adjusted $R^{2}$

假设线性模型中用了 $k$ 个自变量（$1\le k\le p$），得到的**可决系数**为 $R^{2}$（$R^{2}$ 实际上依赖于 $k$ 的取值*以及选用了哪 $k$ 个自变量*，但符号中忽略了这个依赖关系）。Adjusted $R^{2}$ 定义为

$$
R_{\text{adj}}^{2}=1-\frac{n-1}{n-k-1}\left(1-R^{2}\right).
$$

它有几点性质。其一，$R_{\text{adj}}^{2}$ **不会随着 $k$ 的增加一直增大**，因为其中对较大的 $k$ 起到了"惩罚"作用，实际应用时应选取使 $R_{\text{adj}}^{2}$ **最大**的 $k$。其二，恒有关系式 $R_{\text{adj}}^{2}\le R^{2}$。其三，$R_{\text{adj}}^{2}$ *可能为负数*，但不会超过 $1$。

### 2. AIC

根据**对变量个数进行"惩罚"**的思想，Akaike (1973) 提出了 **Akaike 信息准则**（*Akaike information criterion*, AIC）：

$$
\mathrm{AIC}=-2\log(\text{maximum likelihood})+2\times(\text{number of parameters}).
$$

对于线性模型，对数似然函数的极大值为

$$
-\frac{n}{2}\log(2\pi)-\frac{n}{2}\log\left(\frac{\mathrm{RSS}}{n}\right)-\frac{n}{2},
$$

注意 AIC 是用来**比较不同变量选择方式的优劣**，故应在意其**相对大小**而非绝对大小。略去与变量选择无关的项，得到此时的 AIC 为

$$
\mathrm{AIC}=n\log\left(\frac{\mathrm{RSS}}{n}\right)+2(k+1),
$$

其中 $2(k+1)$ 可以视作对**变量个数的"惩罚"**。使用 AIC 做模型选择时，应选用使得 AIC **最小**的一组变量。

### 3. BIC

在许多时候，AIC 中的"惩罚"项力度**仍然不够**。Schwartz (1978) 提出了 **Bayes 信息准则**（*Bayes information criterion*, BIC）：

$$
\mathrm{BIC}=n\log\left(\frac{\mathrm{RSS}}{n}\right)+(k+1)\log n .
$$

使用 BIC 做模型选择时，应选用使得 BIC **最小**的一组变量。若 $\log n>2$，则 BIC 的惩罚**强于** AIC，因此 **BIC 选出的模型的变量个数不会大于 AIC 选出的模型的变量个数**。两者的差别只落在惩罚项上：

| 准则 | 惩罚项 | $n=100$ 附近的大致强度 | 倾向 |
| :--- | :--- | :--- | :--- |
| AIC | $2(k+1)$ | 每个变量约 $2$ | 变量*偏多*，偏重预测 |
| BIC | $(k+1)\log n$ | 每个变量约 $4.6$ | 变量*偏少*，偏重简约 |

> [!warning] 注意
>
> AIC、BIC 的**绝对值没有意义**，只有对*同一批数据、同一因变量*下不同模型的**相对大小**才可比。因此讲义中"略去与变量选择无关的项"是合法操作，但也意味着**不能跨数据集直接比较 AIC 数值**。

### 4. 交叉验证

模型选择的另一种方法是利用现有数据**估算出预测误差**，再选取预测误差最小的模型。基本想法是把数据分为两部分，一部分用来**拟合**模型，另一部分用来**预测**因变量，前者称为**训练集**（*training set*），后者称为**验证集**（*validation set*）。

假设一共有 $M$ 个待选模型。我们把数据分为 $k$ 份（一般保持每份样本量*相等或大致相等*）。对于一个固定的 $m\in\{1,\dots,M\}$，对每个 $i\in\{1,\dots,k\}$，用**除第 $i$ 份之外**的所有数据作为训练集拟合模型，再用**第 $i$ 份数据**作为验证集计算拟合结果对验证集样本的平均预测误差，记为 $\widehat{\mathrm{PE}}_i(m)$。令

$$
\mathrm{CV}(m)=\frac{1}{k}\sum_{i=1}^{k}\widehat{\mathrm{PE}}_i(m),
$$

最后选取使得 $\mathrm{CV}(m)$ 最小的模型：

$$
m^{*}=\arg\min_{1\le m\le M}\mathrm{CV}(m).
$$

这个过程称为 $k$ 折的**交叉验证**（*$k$-fold cross validation*）。通常取 $k=5$ 或 $k=10$，也可以取 $k=n$，此时称为**留一法交叉验证**（*leave-one-out CV*）。

> [!warning] 注意
>
> **CV 只能用于模型选择**。选好模型 $m^{*}$ 之后，还需要再用**全部数据**当作训练集重新拟合参数，作为最终的模型；用 CV 的折内估计直接当成最终参数是*错误的*。

以多项式回归为例，讲义图 3 中蓝线表示**真正的样本外预测误差**，黑线表示留一法得到的 CV 值，橙线表示 10-fold 交叉验证得到的 CV 值。可以看出，交叉验证的 CV 值与真正的样本外预测误差*可能有较大差别*，**但交叉验证选择出的模型的样本外预测误差十分接近其最小值**，这说明 CV 在"选模型"这一目的上是可靠的。

在 CV 中，有时相近模型之间的 CV 值会非常接近，为避免过拟合，可以遵循 **"1 标准误规则"**（*one standard error rule*），即选取 CV 值在 1 标准误之内 **flexibility 最小**的模型。具体来说，先计算预测误差 $\widehat{\mathrm{PE}}_i(m)$ 的**样本标准差**

$$
\mathrm{SD}(m)=\sqrt{\frac{1}{k-1}\sum_{i=1}^{k}\left\{\widehat{\mathrm{PE}}_i(m)-\mathrm{CV}(m)\right\}^{2}},
$$

再计算 CV 的**标准误**

$$
\mathrm{SE}(m)=\frac{1}{\sqrt{k}}\mathrm{SD}(m).
$$

模型选择时，在所有满足 $\mathrm{CV}(m)<\mathrm{CV}(m^{*})+\mathrm{SE}(m^{*})$ 的 $m$ 中，选择 **flexibility 最小**的模型 $m^{*}_{1\text{se}}$。讲义图 4 给出了 BIC、Adjusted $R^{2}$、CV 等几个准则在线性回归中选择变量个数的对比，图 5 则专门展示了 1 标准误规则的作用。

## 三、逐步回归

### 1. Forward Selection 与 Backward Elimination

除了**变量个数**以外，还需要决定**有哪些变量**应加入模型。在穷举法计算量过大时，常见的做法是**逐步回归**（*stepwise regression*），它包括 forward selection、backward elimination 以及 FS+BE 方法。三种做法共享同一套停止与比较逻辑，差别只在**搜索方向与起点**：

| 方法 | 起点模型 | 每一步的操作 | 额外约束 |
| :--- | :--- | :--- | :--- |
| Forward selection | 只有截距的模型 | **加入**一个变量 | 无 |
| Backward elimination | 包含全部 $p$ 个变量的模型 | **去掉**一个变量 | 需要 $p<n$ |
| FS+BE | 任意指定模型 | 加入*或*去掉一个变量 | 无 |

**Forward selection** 从*最小的*模型出发，逐步向前**加入**变量。步骤如下。

1. 拟合一个**只有截距**的模型，计算 AIC（或 BIC，或 $-R_{\text{adj}}^{2}$）。
2. 对 $m=1$ 到 $M$：从 $p$ 个变量中选取第 $j_m$ 个变量，使得当前模型加入第 $j_m$ 个变量后（*相对于加入其它变量*）AIC（或残差平方和、BIC、$-R_{\text{adj}}^{2}$）**最小**，并计算加入该变量之后的准则值。
3. 从第 1、2 步得到的 $M+1$ 个准则值中选取**最小**的，其对应的模型即为最终模型。

**Backward elimination** 则从*最大的*模型出发，逐步向后**删除**变量。

1. 拟合一个包含**所有 $p$ 个变量**的模型，计算 AIC（或 BIC，或 $-R_{\text{adj}}^{2}$）。
2. 对 $m=1$ 到 $M$：从现有模型中去掉一个变量，使得去掉此变量后（*相对于去掉其它变量*）准则值**最小**，并计算去掉变量之后的准则值。
3. 从第 1、2 步得到的 $M+1$ 个准则值中选取**最小**的，对应模型即为最终模型。

**Backward elimination 需要 $p<n$ 成立才能执行**，这一点与 forward selection 不同。

**FS+BE** 则是两者的**结合**。

1. 拟合某个模型，计算 AIC（或 BIC，或 $-R_{\text{adj}}^{2}$）。
2. 对 $m=1$ 到 $M$：**加入一个变量或去掉一个现有模型中的变量**，使得该操作后（*相对于加入或去掉其它变量*）准则值**最小**，并计算操作之后的准则值。
3. 从第 1、2 步得到的 $M+1$ 个准则值中选取**最小**的，对应模型即为最终模型。

若模型可能包括**主效应项（一阶项）和更高阶项**，三种方法在第 2 步都需要遵循 **marginality principle**：

| 方法 | marginality principle 的具体含义 |
| :--- | :--- |
| Forward selection | 所有**低阶项**都被选进来之前，**不考虑选入更高阶项** |
| Backward elimination | 所有**高阶项**都被去掉之前，**不考虑去掉更低阶项** |
| FS+BE | 加入与删除*两个方向都须遵守*上述规则 |

> [!example] 例：Highway Accidents（Weisberg §10.2.2）
>
> R 代码 `chapter10.R` 先构造变量 `sigs1 <- (sigs*len + 1)/len`，并设定候选变量集合
>
> ```r
> f <- ~ log(len) + shld + log(adt) + log(trks) + lane + slim + lwid +
>   itg + log(sigs1) + acpt + htype
> ```
>
> 三种搜索方式都以 `m0 <- lm(log(rate) ~ log(len), Highway)` 为起点：
>
> ```r
> # forward selection
> m.forward  <- step(m0, scope = f, direction = "forward")
>
> # backward elimination
> m1 <- update(m0, f)
> m.backward <- step(m1, scope = c(lower = ~ log(len)), direction = "backward")
>
> # FS+BE
> m.stepwise <- step(m1, scope = list(upper = f, lower = ~ log(len)), direction = "both")
> ```
>
> 其中 `lower` 与 `upper` **框定了搜索的范围**，`lower = ~ log(len)` 保证 `log(len)` *始终留在模型中*；`step` 函数默认按 **AIC** 逐步筛选变量。

### 2. Orthogonal Greedy Algorithm（OGA，不要求）

无论是 forward selection 还是 backward elimination，每一步都需要**重新做一次 OLS 拟合**一个多元回归模型；当变量数较大时，算法的计算成本很大。Ing and Lai (2011) 提出一种 **orthogonal greedy algorithm**（OGA），利用 **Gram-Schmidt 正交化**避免求矩阵的逆，改进计算效率。步骤如下。

1. 拟合一个只有截距的模型，取初始残差 $e^{(0)}$ 与入选变量集合 $J^{(0)}=\varnothing$。
2. 对 $m=1$ 到 $M$：
   - (a) 从变量集合 $\{1,\dots,p\}\setminus J^{(m-1)}$ 中选取第 $\hat j_m$ 个变量，使得当前模型加入该变量后（*相对于加入其它变量*）**残差平方和最小**。
   - (b) 若 $m=1$，令 $\hat X_{\langle\hat j_m\rangle}^{\perp}=X_{\langle\hat j_m\rangle}$；若 $m>1$，计算 $X_{\langle\hat j_m\rangle}$ 在 $\hat X_{\langle\hat j_1\rangle}^{\perp},\dots,\hat X_{\langle\hat j_{m-1}\rangle}^{\perp}$ 张成的线性空间上的**投影** $\hat X_{\langle\hat j_m\rangle}$，并令 $\hat X_{\langle\hat j_m\rangle}^{\perp}=X_{\langle\hat j_m\rangle}-\hat X_{\langle\hat j_m\rangle}$。
   - (c) 计算 $e^{(m-1)}$ 在 $\hat X_{\langle\hat j_m\rangle}^{\perp}$ 上的投影 $\hat e^{(m-1)}$，并令 $e^{(m)}=e^{(m-1)}-\hat e^{(m-1)}$。
   - (d) 令 $J^{(m)}=J^{(m-1)}\cup\{\hat j_m\}$。
3. 从第 2 步得到的 $M$ 个模型中，选取使得某个模型选择准则**最优**的模型。

在 OGA 的第 $m$ 步中，算法将当前选出的变量对之前选出的变量集做**正交化**，即 Gram-Schmidt 正交化，得到一组**正交基** $\hat X_{\langle\hat j_1\rangle}^{\perp},\dots,\hat X_{\langle\hat j_m\rangle}^{\perp}$。注意在计算 $X_{\langle\hat j_m\rangle}$ 的投影时，由于这些正交基**各自互相正交**，只需要做一系列**一元线性回归**

$$
X_{\langle\hat j_m\rangle}\sim\hat X_{\langle\hat j_1\rangle}^{\perp},\quad\dots,\quad X_{\langle\hat j_m\rangle}\sim\hat X_{\langle\hat j_{m-1}\rangle}^{\perp},
$$

即可，**不涉及矩阵求逆**，计算效率可以大大提升。

同时，Ing and Lai (2011) 还将经典信息准则修改为**高维信息准则**（*high-dimensional information criteria*, HDIC），在高维变量时能**更严格地控制变量个数**。定义 high-dimensional AIC（HDAIC）与 high-dimensional BIC（HDBIC）如下：

$$
\mathrm{HDAIC}=n\log\left(\frac{\mathrm{RSS}}{n}\right)+c\,k\log p,
\qquad
\mathrm{HDBIC}=n\log\left(\frac{\mathrm{RSS}}{n}\right)+k\log p\log n .
$$

与经典准则的对照如下表。二者的实质都是**在原始惩罚项上加入乘子** $\log p$，使惩罚随候选变量总数 $p$ 增长，从而对变量个数的控制更加严厉。

| 准则 | 惩罚项 | 相对经典准则的变化 |
| :--- | :--- | :--- |
| AIC | $2(k+1)$ | 与 $p$ *无关* |
| HDAIC | $c\,k\log p$ | **再乘一个** $\log p$ |
| BIC | $(k+1)\log n$ | 与 $p$ *无关* |
| HDBIC | $k\log p\log n$ | **再乘一个** $\log p$ |

讲义图 6 与图 7 的模拟实验表明，**OGA+HDIC 在高维稀疏线性模型中相对其它逐步选变量法和正则化方法均有良好表现**。

---

## 本讲小结

| 主题 | 关键对象 | 一句话要点 |
| :--- | :--- | :--- |
| Bias-variance Tradeoff | $\mathrm{PE}$ 的三项分解 | 变量越多，$\mathrm{bias}^{2}$ 降、variance 升，**需要权衡** |
| 模型选择准则 | $R_{\text{adj}}^{2}$、AIC、BIC、CV | 前三个靠**惩罚项**，CV 直接估预测误差；*AIC 偏宽，BIC 偏严* |
| 逐步回归 | forward、backward、FS+BE、OGA | 逐步搜索**替代穷举**；含高阶项时须遵守 marginality principle |

**贯穿全讲的一条线索**：本讲始终在回答一个问题——*在"拟合更好"和"预测更稳"之间，怎样划一条线*。bias-variance tradeoff 给出理论答案，模型选择准则给出可计算的判据，逐步回归给出在计算上可行的搜索策略。三者层层递进：**先有标准，再有算法**。

**参考文献**：C.-K. Ing and T. L. Lai. A stepwise regression method and consistent model selection for high-dimensional sparse linear models. *Statistica Sinica*, 21(4):1473–1513, 2011.
