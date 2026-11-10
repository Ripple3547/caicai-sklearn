# Lecture 2 笔记：Mathematical Review and Model Accuracy

> 课程：Machine Learning（Nifei Lin, SUFE, Fall 2026）
> 对应教材：ISLR 第 2 章 + 概率/线性代数预备知识

## 本讲结构

1. **Mathematical Review**：线性代数与概率论复习
2. **Statistical Learning**：统计学习的基本框架
3. **Model Accuracy & Bias–Variance Trade-off**：模型精度与偏差–方差权衡
4. **Classification**：分类问题入门（Bayes 分类器、KNN）

---

## 1. 数学复习

### 1.1 线性代数

**标量、向量、矩阵**
- 标量（scalar）：单个数。
- 向量（vector）：一维数组，$\mathbf{x} = (x_1,\dots,x_n)^\top \in \mathbb{R}^n$。
- 矩阵（matrix）：二维数组，$A \in \mathbb{R}^{m\times n}$，元素为 $A_{i,j}$。

**矩阵运算**
- 转置：$(A^\top)_{i,j} = A_{j,i}$；若 $A \in \mathbb{R}^{m\times n}$，则 $A^\top \in \mathbb{R}^{n\times m}$。
- 矩阵（点）乘：若 $C_{m\times p} = A_{m\times n} B_{n\times p}$，则 $C_{i,j} = \sum_k A_{i,k} B_{k,j}$。

**方程组**：$n$ 个未知数、$m$ 个方程可写成 $A_{m\times n} X_{n\times 1} = b_{m\times 1}$。若 $A$ 可逆，则 $X = A^{-1}b$。

### 1.2 概率论

**样本空间与事件**
- 样本空间 $\Omega$：实验所有可能结果的集合（如两次抛硬币 $\Omega=\{HH,HT,TH,TT\}$）。
- 观测 $\omega \in \Omega$：一次实现的结果。
- 事件 $E \subseteq \Omega$：样本空间的子集（如"两次相同" $E=\{HH,TT\}$）。

**概率空间** $(\Omega, \mathcal{F}, P)$：$\Omega$ 为样本空间；$\mathcal{F}$ 为 $\sigma$-代数（定义了概率的事件集合）；$P$ 为概率测度，满足三条公理：
1. $P(E) \geq 0$；
2. $P(\Omega) = 1$；
3. 若 $E_1, E_2, \dots$ 互不相交，则 $P(\bigcup_i E_i) = \sum_i P(E_i)$。

**联合概率与条件概率**
- 联合概率 $P(A,B)$；条件概率 $P(A\mid B) = P(A,B)/P(B)$。
- 乘法规则：$P(A,B) = P(A\mid B)P(B) = P(B\mid A)P(A)$。
- 例：ML 课 85 分以上的概率 50%，两门都 85+ 的概率 40%，则 $P(F\mid M) = 0.4/0.5 = 80\%$。

**随机变量**：映射 $X: \Omega \to \mathbb{R}$，把每个结果 $\omega$ 赋一个实数 $X(\omega)$。例：抛 10 次硬币，$X(\omega)$ 记正面次数。

**分布**
- 离散型：概率质量函数 $p_X(x) = P(X=x)$，满足 $p_X(x)\geq 0$，$\sum_x p_X(x)=1$。
  - Bernoulli$(p)$：$E=p$，$\mathrm{Var}=p(1-p)$
  - Binomial$(n,p)$：$E=np$，$\mathrm{Var}=np(1-p)$
  - Poisson$(\lambda)$：$E=\mathrm{Var}=\lambda$
  - Geometric$(p)$：$E=1/p$，$\mathrm{Var}=(1-p)/p^2$
- 连续型：概率密度函数 $f_X$，$P(a\leq X\leq b) = \int_a^b f_X(x)\,dx$；CDF $F_X(t) = \int_{-\infty}^t f_X(x)\,dx$。
  - Uniform$(a,b)$：$E=(a+b)/2$，$\mathrm{Var}=(b-a)^2/12$
  - Exponential$(\lambda)$：$E=1/\lambda$，$\mathrm{Var}=1/\lambda^2$
  - Normal$(\mu,\sigma^2)$：$E=\mu$，$\mathrm{Var}=\sigma^2$

**条件分布与回归视角**：数据生成分布可写为 $X \sim P_X$，$Y \mid X=x \sim P_{Y\mid X=x}$。这是回归的核心视角。

### 1.3 期望、方差与协方差

- 期望：离散 $E[Z]=\sum_z zP(Z=z)$；连续 $E[Z]=\int z f(z)\,dz$。
- 方差：$\mathrm{Var}(Z) = E[(Z-EZ)^2] = E[Z^2] - (EZ)^2$。
- 协方差：$\mathrm{Cov}(X,Y) = E[(X-EX)(Y-EY)] = E[XY] - E[X]E[Y]$。
- 恒等式：$\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$；若 $X_1,\dots,X_n$ 独立，$\mathrm{Var}(\sum_i X_i) = \sum_i \mathrm{Var}(X_i)$。

**条件期望**：$m(X) = E[Y\mid X]$ 是只依赖 $X$ 的随机变量。
- **关键结论**：$m(X) = E[Y\mid X]$ 是所有函数 $g(X)$ 中使均方预测误差 $E[(Y-g(X))^2]$ 最小的函数——条件期望是最佳的 $L_2$ 预测器。

**全期望公式与全方差公式**
- $E[Y] = E[E[Y\mid X]]$
- $\mathrm{Var}(Y) = E[\mathrm{Var}(Y\mid X)] + \mathrm{Var}(E[Y\mid X])$

### 1.4 I.I.D. 样本与极限定理

- 标准监督学习样本 $\mathcal{D}_n = \{(X_i, Y_i)\}_{i=1}^n$，$(X_i,Y_i) \overset{i.i.d.}{\sim} P_{X,Y}$。
  - 同分布：所有观测服从同一总体分布；独立：一个观测不影响另一个。
  - 意义：i.i.d. 假设使经验平均可以近似总体期望。
- **大数定律（LLN）**：若 $E|\phi(Z)| < \infty$，则 $\frac{1}{n}\sum_i \phi(Z_i) \to E[\phi(Z)]$ 几乎必然成立。
- **中心极限定理（CLT）**：若 $E[Z_i]=\mu$，$\mathrm{Var}(Z_i)=\sigma^2<\infty$，则 $\sqrt{n}\left(\frac{1}{n}\sum_i Z_i - \mu\right) \Rightarrow N(0,\sigma^2)$。样本均值的波动量级为 $O_p(n^{-1/2})$。

### 1.5 偏差–方差分解的预热例子

- $Y_i = \mu + \varepsilon_i$，$E[\varepsilon_i]=0$，$\mathrm{Var}(\varepsilon_i)=\sigma^2$，用 $\hat\mu = \frac{1}{n}\sum_i Y_i$ 估计：
  - 偏差 $E[\hat\mu] - \mu = 0$；方差 $\sigma^2/n$；MSE $= 0^2 + \sigma^2/n$。
- 若用收缩估计 $\hat\mu_c = c\hat\mu$（$0<c<1$）：
  - 偏差 $(c-1)\mu$；方差 $c^2\sigma^2/n$；MSE $= (1-c)^2\mu^2 + c^2\sigma^2/n$。
  - **核心启示**：引入一点偏差有时能大幅降低方差从而改善 MSE——这就是**正则化**的核心思想。

---

## 2. 统计学习（Statistical Learning）

### 2.1 什么是统计学习

- 例：Sales 对 TV、Radio、Newspaper 三个广告渠道的关系。能否用 $Sales \approx f(TV, Radio, Newspaper)$ 预测？

**记号**
- $Y$：响应变量（response / target），如 Sales。
- $X = (X_1, X_2, X_3)^\top$：特征/输入/预测变量（feature / input / predictor），如 TV、radio、newspaper。
- 统计学习模型：$Y = f(X) + \varepsilon$，$\varepsilon$ 捕捉测量误差和其他偏差。

### 2.2 $f(X)$ 有什么用

| 预测（Prediction） | 推断（Inference） |
|---|---|
| 能多准确地预测新响应？ | 数据背后的关系是什么？ |

- 预测：在新点 $X=x$ 处预测 $Y$。
- 推断：理解 $X = (X_1,\dots,X_p)$ 中哪些成分对解释 $Y$ 重要，哪些无关；理解每个 $X_j$ 如何影响 $Y$（取决于 $f$ 的复杂度）。

### 2.3 回归函数

- 理想的 $f$：**回归函数** $f(x) = E(Y \mid X=x)$。
- 它在均方预测误差意义下最优：$f$ 使 $E[(Y-g(X))^2 \mid X=x]$ 在所有 $g$ 上最小。
- $\varepsilon = Y - f(x)$ 是**不可约误差（irreducible error）**：即使知道 $f(x)$，每个 $X=x$ 处 $Y$ 仍有一个分布，预测仍有误差。
- 对任意估计 $\hat f(x)$：
$$E[(Y - \hat f(X))^2 \mid X=x] = \underbrace{[f(x) - \hat f(x)]^2}_{\text{可约 Reducible}} + \underbrace{\mathrm{Var}(\varepsilon)}_{\text{不可约 Irreducible}}$$

### 2.4 如何估计 $f$

- 通常没有恰好 $X=x$ 的数据点，无法直接算 $E(Y\mid X=x)$。
- 松弛定义：$\hat f(x) = \mathrm{Ave}(Y \mid X \in \mathcal{N}(x))$，$\mathcal{N}(x)$ 为 $x$ 的某个邻域。
- **最近邻平均**：$p$ 小（如 $p \leq 4$）且 $N$ 较大时效果好；$p$ 大时因**维数灾难（curse of dimensionality）**而失效——高维中最近邻其实很远，邻域不再"局部"。
- **参数化模型**：如线性模型 $f_L(X) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$，用 $p+1$ 个参数指定，通过拟合训练数据估计。线性模型几乎从不正确，但常是好的、可解释的近似。

### 2.5 灵活性与过拟合

- 模拟例子：income = f(education, seniority) + ε。
  - 线性拟合（平面）→ 合理；thin-plate spline（柔性曲面）→ 更贴合；过度柔性的曲面 → 训练误差为 0，但学到了噪声。
- **过拟合（overfitting）**：把噪声当信号学；训练误差改善而新数据表现变差。

**若干权衡**
- 预测精度 vs. 可解释性：线性模型易解释，柔性样条不易。
- 拟合好 vs. 过拟合/欠拟合：怎么知道模型"刚刚好"？
- 简约 vs. 黑箱预测：常偏好变量更少的简单模型。

**灵活性–可解释性谱**（从低灵活/高可解释到高灵活/低可解释）：
Subset Selection → Lasso → Least Squares → Generalized Additive Models → Trees → Bagging/Boosting → SVM → Deep Learning。

---

## 3. 模型精度与偏差–方差权衡

### 3.1 训练 MSE vs. 测试 MSE

- 训练 MSE：$\mathrm{MSE}_{Tr} = \mathrm{Ave}_{i\in Tr}[y_i - \hat f(x_i)]^2$——衡量对**已见过**数据的拟合，偏向过拟合的模型。
- 测试 MSE：$\mathrm{MSE}_{Te} = \mathrm{Ave}_{i\in Te}[y_i - \hat f(x_i)]^2$——估计在**新观测**上的表现，才是我们关心的。

### 3.2 灵活性与误差曲线

- 训练 MSE 随灵活性增加而**单调下降**；测试 MSE 呈 **U 形**：中等灵活性使测试误差最小。
- 真实情况平滑时：简单模型几乎和复杂模型一样好，额外灵活性主要在追噪声。
- 真实情况曲折且噪声小时：更灵活的拟合更好（欠拟合情形）。

### 3.3 偏差–方差分解（核心公式）

设测试点 $(x_0, y_0)$，真实模型 $Y = f(X) + \varepsilon$，$f(x) = E[Y\mid X=x]$，则

$$E\left[\big(y_0 - \hat f(x_0)\big)^2\right] = \mathrm{Var}\big(\hat f(x_0)\big) + \big[\mathrm{Bias}\big(\hat f(x_0)\big)\big]^2 + \mathrm{Var}(\varepsilon)$$

其中 $\mathrm{Bias}(\hat f(x_0)) = E[\hat f(x_0)] - f(x_0)$，期望对 $y_0$ 和训练集 $Tr$ 的随机性取平均。

- **规律**：灵活性增加 → 方差增大、偏差减小。
- 基于平均测试误差选择灵活性 = 做偏差–方差权衡。

---

## 4. 分类（Classification）

### 4.1 分类问题

- 响应 $Y$ 是定性的：如邮件 $\mathcal{C}=\{\text{spam}, \text{ham}\}$，数字 $\mathcal{C}=\{0,1,\dots,9\}$。
- 目标：
  1. 构建分类器 $C(X)$，给新观测 $X$ 赋类别标签；
  2. 评估每次分类的不确定性；
  3. 理解各预测变量 $X_1,\dots,X_p$ 的作用。

### 4.2 类条件概率与 Bayes 分类器

- 定义 $p_k(x) = P(Y=k \mid X=x)$，$k=1,\dots,K$。
- **Bayes 分类器**：$C_{\text{Bayes}}(x) = \arg\max_k p_k(x)$（分到最可能的类）。
- 用测试集误分类率评估：$\mathrm{Err}_{Te} = \mathrm{Ave}_{i\in Te} I[y_i \neq \hat C(x_i)]$。
- **最优性**：若真实的 $p_k(x)$ 已知，Bayes 分类器的期望误分类误差最小。
- **Bayes 决策边界**：类概率相等的点构成的边界——理论上最优但实际未知。

### 4.3 K-近邻分类（KNN）

对测试点 $x_0$：
1. 找出 $K$ 个最近的训练观测 $\mathcal{N}_0$；
2. 估计 $\hat p_k(x_0) = \frac{1}{K}\sum_{i\in\mathcal{N}_0} I(y_i = k)$；
3. 分到 $\hat p_k(x_0)$ 最大的类。

- **K 的选择**：小 $K$（如 $K=1$）→ 低偏差、高方差（边界极不规则）；大 $K$（如 $K=100$）→ 高偏差、低方差（过于平滑）。
- 用 $1/K$ 度量 KNN 的灵活性：训练误差随 $1/K$ 增大单调下降，测试误差在某个中间 $K$ 处最小。

---

## 核心要点回顾

1. **条件期望 $E[Y\mid X]$ 是最佳的 $L_2$ 预测器**——这是整个回归/监督学习的理论基石。
2. **可约误差 + 不可约误差**：我们能改进的只是 $[f(x)-\hat f(x)]^2$。
3. **测试误差 ≠ 训练误差**：训练误差随灵活性单调降，测试误差呈 U 形。
4. **偏差–方差权衡**：$E[(y_0-\hat f(x_0))^2] = \mathrm{Var} + \mathrm{Bias}^2 + \mathrm{Var}(\varepsilon)$，选模型就是选权衡点。
5. **收缩/正则化的直觉**：以一点偏差换大量方差缩减（$\hat\mu_c = c\hat\mu$ 的例子）。
6. **Bayes 分类器**是分类问题的理论最优；KNN 是简单的非参数近似，$K$ 控制灵活性。
7. **维数灾难**：高维中"局部平均"失效，非参数方法退化。
