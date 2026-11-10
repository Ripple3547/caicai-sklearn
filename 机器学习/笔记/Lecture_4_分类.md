# Lecture 4 笔记：Classification — Probabilities, Decisions, and Model Assumptions

> 课程：Machine Learning（Nifei Lin, SUFE, Fall 2026）
> 对应教材：ISLR 第 4 章

## 本讲结构

1. **Classification and Class Probabilities**：分类问题与类概率
2. **Logistic Regression**：逻辑回归
3. **Generative Models and LDA**：生成式模型与线性判别分析
4. **Classification Errors and Thresholds**：分类误差与阈值
5. **QDA and Naive Bayes**：二次判别分析与朴素贝叶斯
6. **Comparing Classification Methods**：各方法比较
7. **Generalized Linear Models**：广义线性模型

---

## 1. 分类问题与类概率

### 1.1 分类问题

- 训练观测 $(x_i, y_i)$，$x_i \in \mathbb{R}^p$，$y_i \in \{1,\dots,K\}$。
- 二分类例子：default / no default、fraud / no fraud；多分类例子：stroke、drug overdose、epileptic seizure。
- 流程：估计 $p_k(x) = \Pr(Y=k \mid X=x)$，再选类。误分类成本相等时 $\hat y(x) = \arg\max_k \hat p_k(x)$。
- 目标：在**新观测**上预测准确。

### 1.2 例：信用卡违约（Default 数据）

- $n = 10{,}000$；变量：default（Yes/No）、balance（美元）、income（美元）、student（Yes/No）。
- 333 个违约，违约率 3.33%；编码 $Y=1$ 违约，$Y=0$ 不违约。想要 $p(x) = \Pr(Y=1 \mid X=x)$。
- **基线警告**：永远预测 "No" 也有 96.67% 准确率，但一个违约者都找不出来。
- balance 对两组的区分比 income 清晰得多。

### 1.3 为什么不用线性回归？

**(1) 多分类时编码引入虚假次序**
- 把 stroke / drug overdose / epileptic seizure 编成 1/2/3：隐含"drug overdose 在两者之间"且两类差距相等；换一种编码得到完全不同的模型与预测。
- 仅当有自然次序且间隔相近（如 mild / moderate / severe）时 1/2/3 编码才合理。

**(2) 二分类时概率可能越界**
- 二分类可以编码 0/1 后做线性回归（此时 $X^\top\hat\beta$ 估计 $\Pr(Y=1\mid X)$，且翻转编码不影响最终预测）。
- 但预测值可能落到 $[0,1]$ 之外（如 balance 接近 0 时预测违约概率为负），无法解释为概率。
- → 逻辑回归直接对概率建模。

---

## 2. 逻辑回归（Logistic Regression）

### 2.1 逻辑模型

- 直接对概率建模：$p(\text{balance}) = \Pr(\text{default}=\text{Yes} \mid \text{balance})$，取值在 0 与 1 之间。
- 分类规则：$p > 0.5$ 预测 Yes；若公司想更保守地识别风险客户，可用更低阈值如 $p > 0.1$。

设线性预测子 $\eta(x) = \beta_0 + \beta_1 x$，**逻辑函数**：
$$p(x) = \frac{e^{\eta(x)}}{1 + e^{\eta(x)}} = \frac{1}{1 + e^{-\eta(x)}}$$

- $\eta \to -\infty$：$p \to 0$；$\eta = 0$：$p = 1/2$；$\eta \to +\infty$：$p \to 1$。
- 系数与预测变量有限时恒有 $0 < p(x) < 1$ —— 解决了线性回归的越界问题。
- 用**最大似然**拟合。

### 2.2 概率、赔率与对数赔率

- 斜率效应：$\frac{dp(x)}{dx} = \beta_1\, p(x)\big(1-p(x)\big)$ —— 概率的变化幅度**依赖当前 $x$**（非线性！）。
- **赔率（odds）**：$\dfrac{p(x)}{1-p(x)} = e^{\beta_0 + \beta_1 x} \in [0, +\infty)$。
- **对数赔率 / logit**：$\operatorname{logit}\{p(x)\} = \log\dfrac{p(x)}{1-p(x)} = \beta_0 + \beta_1 x$ —— 对 $x$ 线性。
- $\beta_1 > 0$ 时概率随 $x$ 增加。

### 2.3 最大似然估计

- 假设独立训练观测 $Y_i \mid X_i = x_i \sim \text{Bernoulli}(p_i)$，各 $p_i$ 服从共享系数的逻辑模型。
- 似然：$\mathcal{L}(\beta) = \prod_{i=1}^n p_i^{y_i} (1-p_i)^{1-y_i}$（违约贡献 $p_i$，不违约贡献 $1-p_i$）。
- 最大化对数似然：
$$\ell(\beta) = \sum_{i=1}^n \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big]$$
- 数值优化选择使观测结果最可能出现的系数。

### 2.4 Default 数据的估计

$\operatorname{logit}\{p(\text{balance})\} = \beta_0 + \beta_1 \text{balance}$：

| | Estimate | Std. error | z | p-value |
|---|---|---|---|---|
| Intercept | −10.6513 | 0.3612 | −29.5 | <0.0001 |
| balance | 0.0055 | 0.0002 | 24.9 | <0.0001 |

- 对 $H_0: \beta_1 = 0$ 用 **z 统计量** $z = \hat\beta_1 / \mathrm{SE}(\hat\beta_1)$。

### 2.5 定性预测变量与多重逻辑回归

- 哑变量同样适用。student[Yes]：系数 0.4049（p = 0.0004）。
  - 学生：$\hat p = e^{-3.5041+0.4049}/(1+e^{-3.5041+0.4049}) = 0.0431$；非学生：0.0292。学生违约概率更高（**边际上**）。

**多重逻辑回归**：$\operatorname{logit}\{p(X)\} = \beta_0 + \sum_{j=1}^p \beta_j X_j$

| Predictor | Estimate | Std. error | p-value |
|---|---|---|---|
| Intercept | −10.8690 | 0.4923 | <0.0001 |
| balance | 0.0057 | 0.0002 | <0.0001 |
| income | 0.0030 | 0.0082 | 0.7115 |
| student[Yes] | **−0.6468** | 0.2362 | 0.0062 |

- 学生系数**变负**了！
  - **边际比较**：全体学生 vs. 全体非学生 → 学生更危险；
  - **条件比较**：相同 balance 和 income 下学生 vs. 非学生 → 学生更安全。
- **混杂（confounding）**：学生倾向于持有更高的 balance，而高 balance 与违约相关；单变量回归把 balance 的效应算到了学生头上。同样的 balance 和 income 下，学生（0.058）比非学生（0.105）违约概率更低。
- 教训：预测变量相关时，单预测变量与多预测变量回归的结果可能非常不同。

### 2.6 多分类逻辑回归（Multinomial / Softmax）

- 选类 $K$ 为基准，$\eta_k(x) = \beta_{k0} + \sum_{j=1}^p \beta_{kj} x_j$（$k=1,\dots,K-1$）：
$$p_k(x) = \frac{e^{\eta_k(x)}}{1 + \sum_{\ell=1}^{K-1} e^{\eta_\ell(x)}}, \qquad p_K(x) = \frac{1}{1 + \sum_{\ell=1}^{K-1} e^{\eta_\ell(x)}}$$
- $\log\{p_k(x)/p_K(x)\} = \eta_k(x)$：$x_j$ 增 1 单位使"类 $k$ 对基准类的概率比"乘以 $e^{\beta_{kj}}$。
- 换基准改变系数，不改变拟合概率。
- 对称形式（softmax）：$p_k(x) = \dfrac{e^{\eta_k(x)}}{\sum_{\ell=1}^K e^{\eta_\ell(x)}}$，两两对数比线性。

---

## 3. 生成式模型与 LDA

### 3.1 另一条路线

- 逻辑回归：**直接**建模 $\Pr(Y=k \mid X=x)$（判别式）。
- **生成式模型**：
  1. 在每个类内分别建模预测变量的分布 $X \mid Y=k$；
  2. 用 Bayes 定理翻转回 $\Pr(Y=k \mid X=x)$。

### 3.2 Bayes 定理

设先验 $\pi_k = \Pr(Y=k)$，类条件密度 $f_k(x) = \Pr(X=x \mid Y=k)$，则
$$\Pr(Y=k \mid X=x) = \frac{\pi_k f_k(x)}{\sum_{\ell=1}^K \pi_\ell f_\ell(x)}$$
LDA、QDA、朴素贝叶斯的区别在于如何建模 $f_k$。

### 3.3 LDA：单预测变量

**假设**：$X \mid Y=k \sim N(\mu_k, \sigma^2)$ —— 每类有自己的均值、**共享方差**。
$$f_k(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu_k)^2}{2\sigma^2}\right)$$
（正态性是对**每个类内**假设的，不是对混合数据。）

**分类规则**：后验分母对所有类相同，只需最大化 $\pi_k f_k(x)$，等价于最大化 $\log\{\pi_k f_k(x)\}$。

**判别得分为何是线性的**：展开对数，
$$\log\{\pi_k f_k(x)\} = \underbrace{-\frac{x^2}{2\sigma^2}}_{\text{各类相同}} + \frac{x\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log\pi_k + \text{常数}$$
共享方差使 $x^2$ 项在类间比较中抵消，得到**线性判别得分**：
$$\delta_k(x) = \frac{x\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log\pi_k$$
把 $x$ 分到得分最大的类。

**例**：$\mu_1 = -1.25, \mu_2 = 1.25, \sigma^2=1, \pi_1=\pi_2=0.5$ → 决策边界 $x = (\mu_1+\mu_2)/2 = 0$；Bayes 决策边界错误率 10.6%，LDA（20 观测/类）约 11.1%。

### 3.4 LDA 的估计

1. 估计先验、类均值、合并类内方差：
$$\hat\pi_k = \frac{n_k}{n}, \qquad \hat\mu_k = \frac{1}{n_k}\sum_{i:\,y_i=k} x_i, \qquad \hat\sigma^2 = \frac{1}{n-K}\sum_{k=1}^K \sum_{i:\,y_i=k}(x_i - \hat\mu_k)^2$$
2. 代入 $\hat\delta_k(x)$；
3. 预测得分最大的类。
- 后验概率估计：$\hat p_k(x) = \exp\{\hat\delta_k(x)\} \big/ \sum_\ell \exp\{\hat\delta_\ell(x)\}$。

### 3.5 多变量 LDA

- 假设 $X \mid Y=k \sim N_p(\mu_k, \Sigma)$：类特异的均值向量 + **共享协方差矩阵**（允许预测变量相关）。
- **多变量高斯密度**（式 4.23）：
$$f(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right)$$
- **多变量 LDA 规则**：$\log\{\pi_k f_k(x)\} = C(x) + \delta_k(x)$，$C(x)$ 各类相同，
$$\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log\pi_k$$
- $\delta_k(x) = \delta_\ell(x)$ 对 $x$ 线性 → $p=2$ 时边界为直线，更高维为超平面。
- 例：三类二维（各 20 观测），Bayes 测试误差 7.46%，LDA 7.70% —— LDA 很接近最优。

---

## 4. 分类误差与阈值

### 4.1 低错误率可能掩盖漏检

Default 数据（LDA，balance + student，阈值 0.5）：

| | Actual No | Actual Yes | Total |
|---|---|---|---|
| Predicted No | 9,644 | 252 | 9,896 |
| Predicted Yes | 23 | 81 | 104 |
| Total | 9,667 | 333 | 10,000 |

- 总体错误率 = $(23+252)/10000 = 2.75\%$，看似很低；
- 但真正违约者中 $252/333 = 75.7\%$ **被漏掉**，只有 24.3% 被识别。

### 4.2 混淆矩阵与指标

| | Actual − | Actual + |
|---|---|---|
| Predicted − | TN | FN |
| Predicted + | FP | TP |

- **敏感度（Sensitivity / 召回率 / TPR）** $= \dfrac{TP}{TP+FN} = \dfrac{81}{333} = 24.3\%$ —— 问**实际阳性**；
- **特异度（Specificity）** $= \dfrac{TN}{TN+FP} = \dfrac{9644}{9667} = 99.76\%$；
- **精确度（Precision）** $= \dfrac{TP}{TP+FP} = \dfrac{81}{104} = 77.9\%$ —— 问**预测阳性**。

### 4.3 调阈值：0.5 → 0.2

保持 LDA 模型不变，$p(x) > 0.2$ 即预测 Yes：

| | t = 0.5 | t = 0.2 |
|---|---|---|
| Overall error | 2.75% | 3.73% |
| Sensitivity | 24.3% | **58.6%** |
| False positive rate | 0.24% | 2.43% |
| Precision | 77.9% | 45.3% |

- 检测到更多违约，代价是更多误报。
- **阈值控制权衡**：0.5 最小化总体错误（Bayes 分类器用的就是 0.5）；降低阈值会持续降低"违约者被误判"的错误率、提高"非违约者被误判"的错误率。总体错误率被多数类主导。最佳阈值取决于领域知识（如违约的成本）。

### 4.4 ROC 曲线与 AUC

- **ROC 曲线**：同时展示所有可能阈值下的两类错误。
  - 横轴 FPR $= \dfrac{FP}{TN+FP}$（= 1 − 特异度）；纵轴 TPR $= \dfrac{TP}{TP+FN}$（= 敏感度）。
  - 每个点对应一个阈值；降低阈值向 (1,1) 移动。
  - 左上角：高检出 + 低误报（理想）；对角线：无信息分类器。
- **AUC**（曲线下面积）：整体性能汇总。理想分类器贴近左上角、AUC 接近 1；随机猜测 AUC = 0.5。Default 数据上 LDA 的 AUC = 0.95（很好）。
- ROC 适合比较不同分类器（涵盖所有阈值）；此数据上逻辑回归的 ROC 与 LDA 几乎不可区分。

---

## 5. QDA 与朴素贝叶斯

### 5.1 二次判别分析（QDA）

- 允许**每类有自己的协方差**：$X \mid Y=k \sim N_p(\mu_k, \Sigma_k)$。
- **QDA 判别得分**：
$$\delta_k(x) = -\frac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k) - \frac{1}{2}\log|\Sigma_k| + \log\pi_k$$
- 估计 $\pi_k, \mu_k, \Sigma_k$，把观测分到得分最大的类。
- $x_j^2$、$x_j x_\ell$ 等二次项保留 → **二次决策边界**；若协方差共享则退化为 LDA。

**LDA vs. QDA（不同协方差设定下）**
- 左图：共同协方差 → Bayes 边界线性，LDA 近似得好；QDA 方差大而无偏差收益，反而更差。
- 右图：两类协方差不同（+0.7 / −0.7）→ Bayes 边界二次，QDA 更准确。

### 5.2 估计协方差的代价

- 对称 $p\times p$ 协方差矩阵有 $p(p+1)/2$ 个自由参数。

| | LDA | QDA |
|---|---|---|
| 协方差矩阵 | 1 个共享 | K 个独立 |
| 协方差参数 | $p(p+1)/2$ | $Kp(p+1)/2$ |
| $p=50, K=3$ | 1,275 | 3,825 |

- 两者都还要估计 $Kp$ 个均值参数和 $K-1$ 个自由先验。
- **偏差–方差权衡**：共享协方差降低估计方差；类形状不同时不同协方差降低偏差。每类需有足够观测才能估计自己的协方差。
- 经验法则：训练观测少 → LDA（降方差重要）；训练集很大或共享协方差假设明显不成立 → QDA。

### 5.3 朴素贝叶斯（Naive Bayes）

- **关键假设**：类内预测变量**条件独立**：
$$f_k(x) = \prod_{j=1}^p f_{kj}(x_j) \qquad\Longrightarrow\qquad p_k(x) = \frac{\pi_k \prod_{j=1}^p f_{kj}(x_j)}{\sum_{\ell=1}^K \pi_\ell \prod_{j=1}^p f_{\ell j}(x_j)}$$
- 估计许多个一维分布代替联合密度。忽略依赖会引入偏差，但大幅降低方差。
- **各类特征的建模选择**：

| 特征类型 | 类条件模型                                           |
| ---- | ----------------------------------------------- |
| 定量   | 每特征每类一个高斯密度                                     |
| 定量   | 直方图或核密度估计                                       |
| 分类   | 类内各水平的相对频率（如 (32,55,13)/100 → (0.32,0.55,0.13)） |
- **高斯朴素贝叶斯** = 每类对角协方差矩阵，是 QDA 的受限形式。

### 5.4 计算示例（三特征、两类）

对 $x^* = (0.4, 1.5, 1)^\top$，等先验：

| Class $k$ | $\hat f_{k1}(0.4)$ | $\hat f_{k2}(1.5)$ | $\hat f_{k3}(1)$ |
|---|---|---|---|
| 1 | 0.368 | 0.484 | 0.226 |
| 2 | 0.030 | 0.130 | 0.616 |

- 非归一化得分：$s_1 = 0.5(0.368)(0.484)(0.226) = 0.020126656$；$s_2 = 0.5(0.030)(0.130)(0.616) = 0.001201200$。
- 归一化：$\hat p_1 = s_1/(s_1+s_2) = 0.9437$，$\hat p_2 = 0.0563$ → 预测类 1。

### 5.5 Default 数据：朴素贝叶斯 vs. LDA

| Model | t | FP | FN | Error | Sensitivity |
|---|---|---|---|---|---|
| LDA | 0.5 | 23 | 252 | 2.75% | 24.3% |
| Naive Bayes | 0.5 | 46 | 244 | 2.90% | 26.7% |
| LDA | 0.2 | 235 | 138 | 3.73% | 58.6% |
| Naive Bayes | 0.2 | 328 | 130 | 4.58% | 61.0% |

- 各阈值下 NB 检测更多违约，但误报也更多、总错误更高。
- $n=10{,}000, p=2$ 时，条件独立带来的方差缩减收益有限。

---

## 6. 各分类方法比较

### 6.1 对数后验比的函数形式

| 方法 | 形式 |
|---|---|
| Logistic / LDA | $a_k + \sum_j b_{kj} x_j$（**线性**） |
| QDA | $a_k + \sum_j b_{kj} x_j + \sum_{j,\ell} c_{kj\ell} x_j x_\ell$（**二次 + 交互**） |
| Naive Bayes | $a_k + \sum_j g_{kj}(x_j)$（**非线性可加**） |

### 6.2 KNN 分类（回顾）

- 用 $m$ 表示邻居数（$K$ 留给类别数）：$\hat p_k(x) = \frac{1}{m}\sum_{i \in \mathcal{N}_m(x)} I(y_i = k)$。
- 例：5 个邻居中 3 正 2 负 → $\hat p_+(x) = 0.6$，多数决策为正类。
- 小 $m$：灵活边界、高方差；大 $m$：平滑、可能高偏差。
- 基于距离的预测需要合适的**特征缩放**；预测变量多使局部邻域稀疏。
- 用验证集或交叉验证选 $m$，不要用测试集。

### 6.3 六个模拟场景（核心结论）

| 场景 | 设定 | 胜者 | 输家与原因 |
|---|---|---|---|
| 1 | 类内独立高斯，线性边界，20/类 | **LDA、Logistic**（假设匹配） | KNN、QDA（灵活性白付方差）；NB 略好于 QDA（独立假设正确） |
| 2 | 类内相关 −0.5，线性边界 | LDA、Logistic 仍好 | **NB 崩了**（独立性假设被违反） |
| 3 | t 分布（重尾），线性边界，50/类 | **Logistic > LDA**（不需类内正态） | QDA 恶化（对极端点敏感）；NB 差；KNN 落后 |
| 4 | 两类协方差不同（+0.5/−0.5），二次边界 | **QDA**（假设匹配） | LDA/Logistic 有偏差；NB 忽略相关；KNN-CV 有竞争力 |
| 5 | 复杂非线性边界 | **KNN-CV 最好**；**KNN-1 最差**（平滑度选错，灵活也白搭） | QDA、NB 略胜线性方法 |
| 6 | 不同对角协方差，**仅 6 观测/类** | **NB**（假设成立且避免估相关） | QDA 估相关方差太大；LDA/Logistic 有偏差；KNN 局部信息太少 |

**总结**：没有任何方法在所有情形占优——
- 真实边界线性 → LDA、逻辑回归；
- 中等非线性 → QDA 或 NB；
- 复杂边界 + 小 $p$ + 数据充足 → 调好的 KNN；
- 非参数方法的**平滑度必须认真选择**。

### 6.4 选模型速查

- 对数赔率近似线性 → Logistic / LDA；
- 高斯类、形状不同 → QDA（每类数据足够时）；
- 特征多、数据少 → Naive Bayes 作低方差基线；
- 边界复杂、$p$ 小、数据充足 → 调优的 KNN。

---

## 7. 广义线性模型（GLM）

### 7.1 例：共享单车计数（Bikeshare）

- 新响应类型：$Y = $ bikers（华盛顿特区某小时用车人数），$Y \in \{0,1,2,\dots\}$。
- 预测变量：mnth（月，分类）、hr（小时 0–23，分类）、workingday（非周末非节假日 = 1）、temp（归一化温度，0.1 变化 ≠ 0.1°C）、weathersit（clear 为基准：cloudy/misty、light rain/snow、heavy rain/snow）。
- 线性回归系数：春秋季用车最多、冬季最少；早晚通勤高峰（9AM、6PM）最高、凌晨最低。cloudy/misty 系数 −12.89：每小时约少 12.89 人。

### 7.2 高斯响应模型的三个问题

1. **拟合均值无效**：9.6% 的拟合值为负；
2. **方差随均值变化**：12–2 月凌晨 1–4 点雨天，均值 5.05、SD 3.73；4–6 月早 7–10 点晴天，均值 243.59、SD 131.70 —— 违反恒定方差假设 $\mathrm{Var}(Y\mid X) = \sigma^2$；
3. **离散结果**：高斯模型连续，而观测是计数。

**对数变换的局限**：$\log Y$ 回归能避免负预测、缓解异方差，但 $\log 0$ 无定义，且对 $\log Y$ 拟合均值取指数一般**不等于** $E(Y\mid X)$，解释也不自然。

### 7.3 Poisson 回归

- **Poisson 分布**：$\Pr(Y=k) = \dfrac{e^{-\lambda}\lambda^k}{k!}$，$k=0,1,2,\dots$；$E(Y) = \mathrm{Var}(Y) = \lambda$ —— 均值大方差也大，正符合计数数据。
- **条件模型 + 对数链接**：
$$Y \mid X=x \sim \text{Poisson}(\lambda(x)), \qquad \log\lambda(x) = \beta_0 + \sum_{j=1}^p \beta_j x_j \;\Longleftrightarrow\; \lambda(x) = \exp\Big(\beta_0 + \textstyle\sum_j \beta_j x_j\Big)$$
  - 指数保证均值为正；0 计数仍有正概率；**不需要变换观测到的 $Y$**，允许零。
- **最大似然**：$\lambda_i = \exp(x_i^\top\beta)$，
$$\log \mathcal{L}(\beta) = \sum_{i=1}^n \big[y_i \log\lambda_i - \lambda_i - \log(y_i!)\big]$$
  数值求解；预测 $\hat\lambda(x) = \exp(x^\top\hat\beta)$。

### 7.4 系数解释：加性 vs. 乘性

| Predictor | Linear coef | Poisson coef |
|---|---|---|
| workingday | 1.27 | 0.01 |
| temp | 157.21 | 0.79 |
| Cloudy/misty | −12.89 | −0.08 |
| Light rain/snow | −66.49 | −0.58 |
| Heavy rain/snow | −109.75 | −0.93 |

- 其他变量不变，$x_j$ 变化 $\Delta$ 时 $\lambda$ 乘以 $e^{\beta_j \Delta}$；均值百分比变化为 $100(e^{\beta_j\Delta}-1)\%$。
- 例：晴 → 多云，均值变为 $e^{-0.08} = 0.923$ 倍（即 92.3%）；多云 → 雨，再乘 $e^{-0.5} \approx 0.607$。
- Poisson 模型保留了主要的季节与通勤模式；workingday 在 Poisson 下显著而线性模型下不显著。

### 7.5 GLM 框架

设 $\mu(x) = E(Y \mid X=x)$，GLM 三要素：
1. 来自**指数族**的条件响应分布；
2. **线性预测子** $\eta(x) = \beta_0 + \sum_j \beta_j x_j$；
3. **链接函数** $g$ 满足 $g(\mu(x)) = \eta(x)$。

| Model | $Y \mid X$ | Link $g(\mu)$ | Mean $\mu(x)$ |
|---|---|---|---|
| Linear | Gaussian | $\mu$（恒等） | $\eta$ |
| Logistic | Bernoulli | $\log\frac{\mu}{1-\mu}$（logit） | $\frac{e^\eta}{1+e^\eta}$ |
| Poisson | Poisson | $\log\mu$ | $e^\eta$ |

---

## 核心要点回顾

1. **分类 = 估计类概率 + 选阈值决策**；总体准确率在类不平衡时具有误导性（96.67% 的"全 No"基线）。
2. **逻辑回归**：logit 线性、概率被压在 (0,1) 内；系数用赔率解释；最大似然估计。
3. **混杂**：单变量与多变量结果可能反转（student 的例子）。
4. **LDA/QDA/NB 都是生成式**：建模 $f_k(x)$ 后用 Bayes 翻转；区别在于协方差结构（共享 / 各类独立 / 对角）。
5. **混淆矩阵指标**：Sensitivity（实际阳性中的检出率）≠ Precision（预测阳性中的正确率）；阈值调节二者的权衡。
6. **ROC/AUC**：跨所有阈值比较分类器；AUC = 0.5 为随机水平。
7. **方法选择看数据**：线性边界 → Logistic/LDA；协方差不同 → QDA；小样本高维 → NB；复杂边界低维大样本 → 调优 KNN。**没有通吃的方法**。
8. **GLM 统一视角**：指数族分布 + 线性预测子 + 链接函数；计数数据用 Poisson 回归（log 链接），避免负预测与恒定方差假设。
