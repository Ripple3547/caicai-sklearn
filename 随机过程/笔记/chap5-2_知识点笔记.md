# Chapter 5 泊松过程（Part II）知识点笔记与例题解答

> **对应讲义**：`slides/chap5-2.pdf`（23 页，Fall 2026）
> **参考教材**：Sheldon M. Ross《Introduction to Probability Models》(第 12 版) 第 5 章 §5.3.3–§5.3.4
> **讲义结构（Outline）**：(1) Split a Poisson process（泊松过程的分裂）　(2) Two Poisson processes（两个泊松过程的竞赛，§5.3.3 后半）　(3) Conditional distribution of the arrival times（到达时间的条件分布）　(4) Sampling a Poisson process（泊松过程的抽样）

---

## 第 1 部分　泊松过程的分裂（Split a Poisson Process）

### 1.1 问题的设定

设 $\{N(t),\ t\ge0\}$ 是速率为 $\lambda$ 的泊松过程。**每次事件发生时**，独立地（"与其它所有事件无关"）把这个事件标记为

- **I 型事件**，概率为 $p$；
- **II 型事件**，概率为 $1-p$。

令 $N_1(t)$、$N_2(t)$ 分别表示 $[0,t]$ 内 I 型、II 型事件的个数，则显然

$$
N(t)=N_1(t)+N_2(t)
$$

> [!NOTE] 辨析（"对每个事件掷硬币"而不是"对每个时刻掷硬币"）
> 分裂的作用对象是**事件**：每个到达的顾客独立地被判为男/女，每个移民独立地被判为英裔/非英裔。硬币的概率 $p$ **不随时间改变**，且掷硬币的结果与"什么时候到达""前面来过几个人"都无关。这两点（概率恒定、分类独立）是下面 Proposition 5.5 成立的全部前提，第 4 部分放宽第一条会得到不同的结论。

### 1.2 Proposition 5.5（分裂定理）

$\{N_1(t),\ t\ge0\}$ 与 $\{N_2(t),\ t\ge0\}$ 分别是速率为 $\lambda p$ 与 $\lambda(1-p)$ 的**泊松过程**，且这两个过程**相互独立**。等价地，对任意 $n,m\ge0$，

$$
P\{N_1(t)=n,\ N_2(t)=m\}
=P\{N_1(t)=n\}\,P\{N_2(t)=m\}
=e^{-\lambda pt}\frac{(\lambda pt)^n}{n!}\cdot
e^{-\lambda(1-p)t}\frac{(\lambda(1-p)t)^m}{m!}
$$

注意两件事同时成立才算完整：**(i) 每个分量仍是泊松过程（速率被打折）；(ii) 两个分量互相独立。** 只记住"速率变成 $\lambda p$"而忘记独立性，会在求联合概率、求协方差时出错。

### 1.3 证明要点（逐条验证 Definition 5.2）

只需验证 $\{N_1(t)\}$ 满足泊松过程的四条公理。

- $N_1(0)=0$：因为 $N(0)=0$。
- **独立增量**：区间 $(t,t+h)$ 内 I 型事件的个数由"该区间内事件总数"与"各事件的分类结果"共同决定；前者只依赖区间长度、后者每次独立。因此 $(t,t+h)$ 内的 I 型事件数与任何不重叠区间的信息无关。
- **一次事件的概率**（关键计算）：把 $h$ 内的 I 型事件按 $h$ 内事件总数分情况，

$$
P\{N_1(h)=1\}=P\{N_1(h)=1\mid N(h)=1\}P\{N(h)=1\}
+P\{N_1(h)=1\mid N(h)\ge2\}P\{N(h)\ge2\}
$$

$$
=p\,(\lambda h+o(h))+o(h)=\lambda p h+o(h)
$$

- **两次及以上的概率**：

$$
P\{N_1(h)\ge2\}\le P\{N(h)\ge2\}=o(h)
$$

**独立性**：$t$ 到 $t+h$ 之间是否出现 I 型事件，只取决于 $(t,t+h)$ 内的事件与硬币，与"II 型事件在什么时候发生"这一信息无关，故两过程独立。（教材另给出一个更漂亮的证法，见 Example 3.24：条件在 $N(t)=n$ 上，分类结果服从**多项分布**，与 $N_1+N_2=n$ 的泊松分解相乘后恰好因子化——这正是第 4 部分 Proposition 5.6 的证明技巧。）

> [!TIP] 方法（记住这个"因子化"套路）
> 分裂定理的两种证法对应两类题型：**公理验证法**适合证明"某计数过程仍是泊松过程"；**条件 + 多项分布法**适合直接算联合分布，而且它能推广到时变分类概率的情形。

### 1.4 从"泊松随机变量"看分裂与叠加（讲义 p6）

讲义 p6 把 Proposition 5.5 重新表述为两条关于**泊松随机变量**的结论：

| 运算 | 命题 | 结论 | 典型用途 |
|---|---|---|---|
| **分裂**（thinning） | $N\sim\text{Poisson}(\lambda)$，每个个体独立地以概率 $p$ 归为 I 类 | $N_1\sim\text{Poisson}(\lambda p)$，$N_2\sim\text{Poisson}(\lambda(1-p))$，且二者独立 | 求"某一类占多少"的分布 |
| **叠加**（superposition） | $N_1\sim\text{Poisson}(\lambda_1)$，$N_2\sim\text{Poisson}(\lambda_2)$，独立 | $N_1+N_2\sim\text{Poisson}(\lambda_1+\lambda_2)$ | 求"总数"的分布 |

- 分裂可以推广到**多类**：把泊松过程分成 $k$ 类（概率 $p_1,\dots,p_k$，$\sum p_i=1$），则 $k$ 个分量是相互独立、速率分别为 $\lambda p_i$ 的泊松过程。Example 5.16 正是这一推广的应用。
- 叠加之后**回看"每个事件来自哪个过程"**：合并过程仍是速率为 $\lambda_1+\lambda_2$ 的泊松过程，而每个合并事件独立地来自第一个过程的概率是

$$
\frac{\lambda_1}{\lambda_1+\lambda_2}
$$

即"分裂"与"叠加"是同一枚硬币的两面。这个观察是第 2 部分解决两个泊松过程竞赛问题的关键。

---

### Example 5.14　英裔移民人数

移民按每周 10 人的泊松速率到达 A 地区；每个移民是英裔的概率为 $1/12$。求**二月份**没有任何英裔移民到达 A 地区的概率。

**解答**：

- 二月按 4 周计（28 天），故 $[0,t]$ 内移民总数 $\sim\text{Poisson}(4\cdot10)$，其中"英裔"这一类由分裂得到，速率为 $10\cdot\frac{1}{12}$。
- 于是二月份英裔移民人数

$$
N_1\sim\text{Poisson}\left(4\cdot10\cdot\frac{1}{12}\right)=\text{Poisson}\left(\frac{10}{3}\right)
$$

- 所求概率

$$
P\{N_1=0\}=e^{-10/3}\approx\boxed{0.0357}
$$

> [!WARNING] 易错点（时间单位与"零事件"）
> - 二月份是 28 天，即**恰好 4 周**，所以二月的参数是 $4\times10=40$；若误按 30 天算会得到 $10\cdot\frac{30}{7}\cdot\frac{1}{12}=\frac{25}{7}$，结果完全不同。题目给的是"每周"，就要先把月换算成周。
> - 问的是"**没有**英裔"，即 $P\{N_1=0\}=e^{-\lambda_1 t}$，不要算成 $1-e^{-10/3}$。

---

### Example 5.15　最优卖价策略（拒绝-接受型最优停止）

非负的报价按速率 $\lambda$ 的泊松过程到达。每个报价的数值是密度为 $f(x)$ 的连续随机变量。报价一旦出现，你必须立即接受或拒绝（拒绝后只能等下一个）。在物品卖出之前，你以速率 $c$ 持续付出成本。目标是最大化**期望净收益**（收入减成本）。考虑"接受第一个大于 $y$ 的报价"的策略（$y$-策略，可证为最优策略）。求最优的 $y$ 与最大期望净收益。

**解答**：记 $\bar F(x)=P\{X>x\}=\int_x^\infty f(u)\,du$ 为报价的尾分布。

**第一步：接受时刻的分布。** 每个报价"大于 $y$"的概率是 $\bar F(y)$，且各报价独立。由 Proposition 5.5，"被接受的报价"构成速率为 $\lambda\bar F(y)$ 的泊松过程（thinning），故从此刻到接受的等待时间

$$
T_y\sim\text{Exp}\big(\lambda\bar F(y)\big),\qquad E[T_y]=\frac{1}{\lambda\bar F(y)}
$$

**第二步：写出净收益。** 接受的报价的期望值为 $E[X\mid X>y]$，成本期望为 $cE[T_y]$，故

$$
E[R(y)]=E[X\mid X>y]-\frac{c}{\lambda\bar F(y)}
=\frac{\int_y^\infty x f(x)\,dx-c/\lambda}{\bar F(y)}
\tag{5.14}
$$

**第三步：一阶条件。** 对 $y$ 求导（用商的求导法则），令导数为 0：

$$
\frac{d}{dy}E[R(y)]=0
\ \text{ 等价于 }\
-\bar F(y)\,y f(y)+\int_y^\infty x f(x)\,dx-\frac{c}{\lambda}f(y)=0
$$

整理得最优门槛 $y^*$ 满足

$$
y^*\bar F(y^*)=\int_{y^*}^\infty x f(x)\,dx-\frac{c}{\lambda}
\quad\text{（等价形式）}\quad
\boxed{\int_{y^*}^\infty (x-y^*)\,f(x)\,dx=\frac{c}{\lambda}}
$$

左端关于 $y$ 严格递减（从 $E[X]$ 递减到 0），故方程有唯一解。

**第四步：最大净收益。** 把 $y=y^*$ 代回 (5.14) 并利用一阶条件的等价形式：

$$
E[R(y^*)]=\frac{1}{\bar F(y^*)}\left(\int_{y^*}^\infty (x-y^*)f(x)\,dx+y^*\bar F(y^*)-\frac{c}{\lambda}\right)
=\frac{1}{\bar F(y^*)}\left(\frac{c}{\lambda}+y^*\bar F(y^*)-\frac{c}{\lambda}\right)
=\boxed{y^*}
$$

即**最优门槛恰好等于最大期望净收益**。

> [!IMPORTANT] 重要结论（门槛值 = 最优值）
> $y^*=E[R(y^*)]$ 不是巧合，而是"最优性方程"：设 $m$ 为最大期望净收益，则拒绝一个报价后问题**完全重新开始**（泊松过程无记忆），故从那一刻起的最大期望净收益仍是 $m$。于是"接受"优于"继续等"当且仅当报价 $x\ge m$，即最优门槛就是 $m$ 本身。这正是无穷期最优停止问题的 Bellman 方程。

> [!NOTE] 辨析（三处容易混的量）
> - **报价到达速率**是 $\lambda$；**可接受报价的到达速率**是 $\lambda\bar F(y)$（被稀释了）。
> - **等待时间**是 $1/(\lambda\bar F(y))$，不是 $1/\lambda$。
> - 一阶条件的右端是 $c/\lambda$（**成本率除以到达率**），不是 $c$，也不是 $c/\bar F(y)$：因为每多等一个"无效报价"，时间期望增加 $1/\lambda$。

---

### Example 5.16　组织内人员流动（马尔可夫链 + 泊松人数）

系统有 $r$ 个状态。每个个体按转移概率矩阵 $P_{ij}$ 的马尔可夫链转移（与历史无关），个体之间独立移动。初始时刻处于状态 $1,2,\dots,r$ 的人数分别是相互独立的泊松随机变量，均值为 $\lambda_1,\dots,\lambda_r$。求 $n$ 个时间单位后各状态人数的**联合分布**。

**解答**：记 $N_j(i)$ 为"初始在状态 $i$ 的人中，时刻 $n$ 处于状态 $j$ 的人数"，$P_{ij}^n$ 为 $n$ 步转移概率。

**第一步（分裂）。** 固定 $i$。初始在状态 $i$ 的人数为 $\text{Poisson}(\lambda_i)$，每个个体独立地以概率 $P_{ij}^n$ 落到状态 $j$。由 Proposition 5.5 的 $r$ 类推广，

$$
N_j(i)\ (j=1,\dots,r)\ \text{相互独立，且}\ N_j(i)\sim\text{Poisson}\big(\lambda_i P_{ij}^n\big)
$$

**第二步（叠加）。** 时刻 $n$ 处于状态 $j$ 的总人数是 $\sum_{i=1}^r N_j(i)$。对每个固定的 $j$，这是 $r$ 个独立泊松变量之和，故

$$
\sum_{i=1}^r N_j(i)\sim\text{Poisson}\left(\sum_{i=1}^r \lambda_i P_{ij}^n\right)
$$

**第三步（跨 $j$ 的独立性）。** 不同初始状态 $i\ne i'$ 涉及不同的人群，独立；同一 $i$ 内各 $j$ 的独立性来自分裂。两者合起来，最终状态人数向量

$$
\left(\sum_{i=1}^r N_1(i),\ \dots,\ \sum_{i=1}^r N_r(i)\right)
$$

的各分量**相互独立**，且第 $j$ 个分量为 $\text{Poisson}\left(\sum_{i=1}^r \lambda_i P_{ij}^n\right)$。于是联合分布就是这些泊松概率质量函数之积。

> [!NOTE] 辨析（两个常见错误）
> - 用的是 **$n$ 步**转移概率 $P_{ij}^n$，不是一步的 $P_{ij}$。人数分布在 $n$ 步之后才有意义。
> - 结论是"**独立**的泊松"，不是"多项分布"。多项分布出现在**总人数固定**时（把固定的人分成 $r$ 类）；这里总人数是随机的，所以是独立的泊松——这与"泊松分裂 / 多项分布"的对应关系完全一致。

> [!TIP] 方法（泊松世界的两条"封闭性"）
> 泊松分布在两件事上是封闭的：**分裂**（独立分类后仍泊松）与**叠加**（独立泊松之和仍泊松）。凡是"随机个个体 + 每个个体独立地落入某类"的模型，先分裂再叠加，通常能直接读出答案。

---

## 第 2 部分　两个泊松过程的竞赛（Two Poisson Processes）

### 2.1 问题

设 $\{N_1(t)\}$、$\{N_2(t)\}$ 是两个**相互独立**的泊松过程，速率分别为 $\lambda_1$、$\lambda_2$。记

- $S_n^{(1)}$：第一个过程中第 $n$ 次事件的时刻；
- $S_m^{(2)}$：第二个过程中第 $m$ 次事件的时刻。

求 $P\{S_n^{(1)}<S_m^{(2)}\}$，即"第一个过程先攒到 $n$ 次"的概率。

### 2.2 特例：$n=1,\ m=1$

$S_1^{(1)}\sim\text{Exp}(\lambda_1)$、$S_1^{(2)}\sim\text{Exp}(\lambda_2)$ 且独立，由两个指数分布的比较，

$$
P\{S_1^{(1)}<S_1^{(2)}\}=\boxed{\frac{\lambda_1}{\lambda_1+\lambda_2}}
\tag{5.16}
$$

### 2.3 特例：$n=2,\ m=1$

要在第二个过程出现第一次事件之前让第一个过程出现两次，必须：

1. 第一个（合并意义下的）事件来自过程 1 —— 概率 $\dfrac{\lambda_1}{\lambda_1+\lambda_2}$；
2. 给定这一事件已发生，由泊松过程的无记忆性（时刻"重新计时"），**一切从头开始**，下一个事件仍须来自过程 1 —— 概率仍是 $\dfrac{\lambda_1}{\lambda_1+\lambda_2}$。

两步独立相乘：

$$
P\{S_2^{(1)}<S_1^{(2)}\}=\left(\frac{\lambda_1}{\lambda_1+\lambda_2}\right)^2
$$

### 2.4 一般情形：把两个过程"合并"成抛硬币

上面两步推理揭示了一般结构：**每一次事件到来，它属于过程 1 的概率是 $\lambda_1/(\lambda_1+\lambda_2)$，属于过程 2 的概率是 $\lambda_2/(\lambda_1+\lambda_2)$，且与之前发生的一切无关。** 讲义 p8 的两个空正是这两个数：

$$
\text{属于 }N_1(t)\text{ 的概率}=\frac{\lambda_1}{\lambda_1+\lambda_2},\qquad
\text{属于 }N_2(t)\text{ 的概率}=\frac{\lambda_2}{\lambda_1+\lambda_2}
$$

于是"过程 1 先攒到 $n$ 次"等价于：**抛一枚正面概率 $p=\lambda_1/(\lambda_1+\lambda_2)$ 的硬币，正面（过程 1）先累计到 $n$ 个。** 这件事完全由前 $n+m-1$ 次抛掷决定（若在前 $n+m-1$ 次里正面不足 $n$ 个，则反面已至少 $m$ 个，过程 2 已经先到 $m$ 次）。故

$$
\boxed{
P\{S_n^{(1)}<S_m^{(2)}\}
=\sum_{k=n}^{n+m-1}\binom{n+m-1}{k}
\left(\frac{\lambda_1}{\lambda_1+\lambda_2}\right)^{k}
\left(\frac{\lambda_2}{\lambda_1+\lambda_2}\right)^{n+m-1-k}
}
$$

> [!NOTE] 辨析（为什么求和上限是 $n+m-1$ 而不是 $\infty$）
> 事件"过程 1 先到 $n$ 次"最迟到**第 $n+m-1$ 次合并事件**时必然已经判定：到那时两个过程合计发生 $n+m-1$ 次，由抽屉原理，要么过程 1 至少 $n$ 次（它赢），要么过程 2 至少 $m$ 次（它赢），不会都无法判定。所以只需数前 $n+m-1$ 次抛掷中的正面数是否达到 $n$。

> [!NOTE] 辨析（与负二项分布、与 $m=1$ 的关系）
> - 上式正是**负二项（Pascal）分布**的尾概率：$P\{\text{第 }n\text{ 次成功出现在第 }m\text{ 次失败之前}\}$。
> - 取 $m=1$：求和只剩 $k=n$ 一项，$\binom{n}{n}=1$，得 $p^n$，与 2.3 一致。
> - 两个概率互补：$P\{S_n^{(1)}<S_m^{(2)}\}+P\{S_m^{(2)}<S_n^{(1)}\}=1$（两者互斥且必有一个发生，两个过程的到达时刻不会恰好相等）。所以只要会算其中一个，另一个用 1 减即可；把 $\lambda_1$ 与 $\lambda_2$ 对换、$n$ 与 $m$ 对换，就从第一个得到第二个。

> [!TIP] 方法（把"过程之比"化为"硬币之比"）
> 两个独立泊松过程的一切"谁先到第几次"问题，都可以用**合并 + 分裂**来解：合并过程速率 $\lambda_1+\lambda_2$，每次事件用概率 $\lambda_1/(\lambda_1+\lambda_2)$ 判给过程 1。这个等价性同时解释了"为什么过程 2 的到达时间在过程 1 的框架里看起来无害"——判标签与时间间隔无关。

**数值核查**：取 $\lambda_1=2$、$\lambda_2=3$、$n=2$、$m=3$，则 $p=0.4$，

$$
P\{S_2^{(1)}<S_3^{(2)}\}=\sum_{k=2}^{4}\binom{4}{k}0.4^k0.6^{4-k}
=0.1536+0.0256+0.0016=0.5248
$$

用 Gamma 密度直接积分 $\int_0^\infty f_{S_2^{(1)}}(x)P\{S_3^{(2)}>x\}\,dx$ 也得 $0.5248$，两种算法一致。

---

## 第 3 部分　到达时间的条件分布（Conditional Distribution of the Arrival Times，§5.3.4）

### 3.1 单个事件：给定 $N(t)=1$ 时它发生在何时

**问**：已知 $[0,t]$ 内恰好发生 1 次事件，它发生在 $s$ 之前的概率是多少？

**解**：对 $s\le t$，

$$
P\{S_1<s\mid N(t)=1\}
=\frac{P\{S_1<s,\ N(t)=1\}}{P\{N(t)=1\}}
=\frac{P\{[0,s)\text{ 内 1 次}\}\,P\{[s,t)\text{ 内 0 次}\}}{P\{N(t)=1\}}
=\frac{\lambda s e^{-\lambda s}\cdot e^{-\lambda(t-s)}}{\lambda t e^{-\lambda t}}
$$

$$
=\frac{\lambda s e^{-\lambda t}}{\lambda t e^{-\lambda t}}
=\boxed{\frac{s}{t}}
$$

即 $S_1\mid N(t)=1\ \sim\ \text{Uniform}(0,t)$。**这一结果与 $\lambda$ 无关。**

> [!NOTE] 辨析（"均匀"来自平稳增量，不是巧合）
> 泊松过程有平稳、独立增量，所以 $[0,t]$ 内等长的每一小段"装下这一次事件"的机会相同；数学上分子中的 $e^{-\lambda(t-s)}$ 与分母中的 $e^{-\lambda t}$ 把 $\lambda$ 的指数部分全部约掉，只剩长度的比 $s/t$。

### 3.2 一般情形：给定 $N(t)=n$ 的条件联合密度

**目标**：求 $f(s_1,s_2,\dots,s_n\mid n)$，其中 $0<s_1<\cdots<s_n<t$。

**关键观察**：事件"$S_1=s_1,\dots,S_n=s_n$ 且 $N(t)=n$"等价于前 $n+1$ 个间隔满足

$$
T_1=s_1,\quad T_2=s_2-s_1,\quad \dots,\quad T_n=s_n-s_{n-1},\quad T_{n+1}>t-s_n
$$

**计算**：由 $T_i$ 独立同分布 $\text{Exp}(\lambda)$，

$$
f(s_1,\dots,s_n,n)=\lambda e^{-\lambda s_1}\lambda e^{-\lambda(s_2-s_1)}\cdots\lambda e^{-\lambda(s_n-s_{n-1})}\cdot e^{-\lambda(t-s_n)}
=\lambda^n e^{-\lambda t}
$$

（注意指数上的和 $-s_1-(s_2-s_1)-\cdots-(t-s_n)=-t$ **telescoping 相消**，只剩 $-\lambda t$。）又 $P\{N(t)=n\}=e^{-\lambda t}(\lambda t)^n/n!$，故

$$
f(s_1,\dots,s_n\mid n)=\frac{\lambda^n e^{-\lambda t}}{e^{-\lambda t}(\lambda t)^n/n!}
=\boxed{\frac{n!}{t^n}},\qquad 0<s_1<\cdots<s_n<t
$$

> [!IMPORTANT] 重要结论（条件密度不含 $\lambda$）
> 给定 $[0,t]$ 内共发生 $n$ 次事件后，这 $n$ 个时刻的**位置分布完全由 $n$ 与 $t$ 决定，与速率 $\lambda$ 无关**。速率的全部作用只体现在"$n$ 有多大"上（$n\sim\text{Poisson}(\lambda t)$）。因此可以把泊松过程看成两个独立的部分：**"有几次"（泊松分布）与"在哪里"（均匀分布）**。

### 3.3 次序统计量（Order Statistics）

设 $Y_1,\dots,Y_n$ 是 $n$ 个随机变量。把它们从小到大重排

$$
Y_{(1)}\le Y_{(2)}\le\cdots\le Y_{(n)}
$$

称 $Y_{(k)}$ 为第 $k$ 个**次序统计量**。

若 $Y_1,\dots,Y_n$ 独立同分布、有连续密度 $f$，则次序统计量的联合密度为

$$
f(y_1,y_2,\dots,y_n)=n!\prod_{i=1}^n f(y_i),\qquad y_1<y_2<\cdots<y_n
$$

**为什么是 $n!$**：$(Y_{(1)},\dots,Y_{(n)})=(y_1,\dots,y_n)$ 当且仅当 $(Y_1,\dots,Y_n)$ 取 $(y_1,\dots,y_n)$ 的**任一排列**；共有 $n!$ 个排列，每个排列的概率密度都是 $\prod_i f(y_i)$，但只在有序区域 $y_1<\cdots<y_n$ 内计数。

**特例（讲义 p13 的问题）**：若 $Y_i$ 独立同分布于 $\text{Uniform}(0,t)$，即 $f=1/t$，则

$$
f(y_1,\dots,y_n)=\boxed{\frac{n!}{t^n}},\qquad 0<y_1<\cdots<y_n<t
$$

这恰好就是 3.2 中得到的条件密度。

### 3.4 Theorem 5.2（到达时间 = 均匀样本的次序统计量）

**定理**：给定 $N(t)=n$，$n$ 个到达时刻 $S_1,S_2,\dots,S_n$ 的联合分布与"**$n$ 个独立同分布 $\text{Uniform}(0,t)$ 随机变量的次序统计量**"的联合分布相同，即

$$
f(s_1,\dots,s_n\mid n)=\frac{n!}{t^n},\qquad 0<s_1<\cdots<s_n<t
$$

**教材的另一种表述（Remark）**：在 $(0,t)$ 内发生 $n$ 次事件的条件下，这 $n$ 次事件发生的时刻，**当作无序的随机变量来看**，是相互独立且都服从 $(0,t)$ 上的均匀分布。

> [!NOTE] 辨析（"独立"的准确含义——本节最易错处）
> - 把 $n$ 个时刻**无序**地看（即当作 $U_1,\dots,U_n$，不排序）：**相互独立**，都 $\sim\text{Uniform}(0,t)$。
> - 把 $n$ 个时刻**按大小排序**为 $S_1<\cdots<S_n$：它们是次序统计量，**彼此不独立**（例如知道 $S_1$ 很大，就知道 $S_2$ 只能更大）。
> - 两者的联合密度是同一个函数 $n!/t^n$，但前者定义在立方体 $\{0<u_i<t\}$ 上，后者定义在单纯形 $\{0<s_1<\cdots<s_n<t\}$ 上。**"独立"只对无序版本成立。**

由"无序独立均匀"可以直接得到三条常用推论（考试中反复出现）：

**(1) 落入子区间的次数服从二项分布。** 给定 $N(t)=n$，对任意区间 $A\subseteq[0,t]$，落入 $A$ 的事件数服从 $\text{Binomial}\left(n,\ |A|/t\right)$。

**(2) 第 $k$ 个到达时刻的边缘分布是 Beta 分布。** 取 $k-1$ 个落在 $(0,s)$、1 个落在 $s$ 处、$n-k$ 个落在 $(s,t)$，得

$$
f_{S_k\mid N(t)=n}(s)=\frac{n!}{(k-1)!\,(n-k)!}\left(\frac{s}{t}\right)^{k-1}\left(1-\frac{s}{t}\right)^{n-k}\frac{1}{t}
=\frac{1}{t}\cdot\frac{1}{B(k,\,n-k+1)}\left(\frac{s}{t}\right)^{k-1}\left(1-\frac{s}{t}\right)^{n-k}
$$

即 $\dfrac{S_k}{t}\mid N(t)=n\ \sim\ \text{Beta}(k,\,n-k+1)$，于是

$$
E[S_k\mid N(t)=n]=\frac{kt}{n+1}
$$

（$k=1$ 时退化为 3.1 的 $t/(n+1)$ 与对称性 $E[S_n\mid N(t)=n]=nt/(n+1)$。）

**(3) 对到达时刻求和的期望公式。** 对任意可积函数 $g$，给定 $N(t)=n$，

$$
E\left[\sum_{i=1}^n g(S_i)\ \Big|\ N(t)=n\right]
=n\cdot\frac{1}{t}\int_0^t g(x)\,dx
$$

因为无序版本的 $U_1,\dots,U_n$ 独立同均匀，求和的期望是 $n$ 倍的单变量期望，与排序无关。Example 5.21 就是取 $g(x)=e^{-\alpha x}$ 的情形。

---

### Example 5.21　贴现的保险索赔总额

保险索赔按速率 $\lambda$ 的泊松过程发生；各次索赔金额是独立同分布、均值为 $\mu$ 的随机变量（分布 $G$），且与到达时刻独立。以 $S_i$、$C_i$ 记第 $i$ 次索赔的时刻与金额。到时刻 $t$ 为止所有索赔的**总贴现成本**

$$
D(t)=\sum_{i=1}^{N(t)}e^{-\alpha S_i}C_i
$$

其中 $\alpha$ 为贴现率。求 $E[D(t)]$。

**解答**：对 $N(t)$ 取条件。

**第一步**：给定 $N(t)=n$，由 Theorem 5.2，$(S_1,\dots,S_n)$ 与 $n$ 个独立均匀变量的次序统计量 $(U_{(1)},\dots,U_{(n)})$ 同分布。故

$$
E[D(t)\mid N(t)=n]=E\left[\sum_{i=1}^n C_i e^{-\alpha U_{(i)}}\right]
=\sum_{i=1}^n E[C_i]\,E\left[e^{-\alpha U_{(i)}}\right]
=n\mu\cdot\frac{1}{n}\sum_{i=1}^n E\left[e^{-\alpha U_{(i)}}\right]
$$

**第二步（消去排序的技巧）**：注意求和号内

$$
\sum_{i=1}^n e^{-\alpha U_{(i)}}=\sum_{i=1}^n e^{-\alpha U_i}
$$

（排序不改变一组数的和！）于是

$$
E[D(t)\mid N(t)=n]=\mu\,E\left[\sum_{i=1}^n e^{-\alpha U_i}\right]
=n\mu\,E\left[e^{-\alpha U}\right],\qquad U\sim\text{Uniform}(0,t)
$$

**第三步**：计算一元积分

$$
E\left[e^{-\alpha U}\right]=\frac{1}{t}\int_0^t e^{-\alpha x}\,dx=\frac{1-e^{-\alpha t}}{\alpha t}
$$

所以

$$
E[D(t)\mid N(t)]=N(t)\,\mu\,\frac{1-e^{-\alpha t}}{\alpha t}
$$

**第四步**：对 $N(t)\sim\text{Poisson}(\lambda t)$ 取期望（$E[N(t)]=\lambda t$）：

$$
E[D(t)]=\lambda t\cdot\mu\cdot\frac{1-e^{-\alpha t}}{\alpha t}
=\boxed{\frac{\lambda\mu}{\alpha}\left(1-e^{-\alpha t}\right)}
$$

> [!TIP] 方法（次序统计量问题"去排序"）
> 遇到形如 $\sum_i g(S_i)$（对到达时刻**求和**，而非对某个具体的 $S_k$ 求分布）的期望，不要一张一张地算 $S_k$ 的 Beta 分布。直接用 Theorem 5.2 把有序的 $S$ 换成无序的 $U$：**求和与排序无关**，于是问题化为 $n$ 个独立均匀变量的同一函数之和。本例中 $\sum_i e^{-\alpha U_{(i)}}=\sum_i e^{-\alpha U_i}$ 就是这一步的核心。

> [!NOTE] 辨析（结果的两个自检）
> - **量纲**：$\lambda\mu/\alpha$ 的量纲是"金额"（$\lambda$ 是每单位时间次数，$\mu$ 是每笔金额，$\alpha$ 是每单位时间的贴现率）。正确。
> - **极限**：令 $\alpha\to0$（不打折），$1-e^{-\alpha t}\approx\alpha t$，得 $E[D(t)]\to\lambda\mu t$ = （期望索赔次数）$\times$（平均金额）。正确。
> - 若 $\alpha\to\infty$，$E[D(t)]\to\lambda\mu/\alpha\to0$：久远的索赔被彻底贴现掉，只剩"贴现率倒数"这个时间尺度内的贡献。也合理。

---

### Example 5.22　最优发货时刻（一个简单的优化问题）

物品按速率 $\lambda$ 的泊松过程到达处理厂。在固定时刻 $T$ 所有物品被统一发运。现在要在 $(0,T)$ 中间选一个时刻 $t$，把**当时系统内的所有物品**先发运一次，使**所有物品的总期望等待时间最小**。求最优的 $t$。

**解答**：若在时刻 $t$ 发运一次，则总期望等待 = $(0,t)$ 段到货的等待 + $(t,T)$ 段到货的等待。

- $(0,t)$ 段：到达数期望 $\lambda t$；由 Theorem 5.2，每个到达时刻在 $(0,t)$ 上均匀分布，故平均等待 $t/2$。贡献

$$
\lambda t\cdot\frac{t}{2}=\frac{\lambda t^2}{2}
$$

- $(t,T)$ 段：同理，到达数期望 $\lambda(T-t)$，平均等待 $(T-t)/2$。贡献

$$
\lambda(T-t)\cdot\frac{T-t}{2}=\frac{\lambda(T-t)^2}{2}
$$

总期望等待

$$
W(t)=\frac{\lambda t^2}{2}+\frac{\lambda(T-t)^2}{2}
$$

求导

$$
W'(t)=\lambda t-\lambda(T-t)=\lambda(2t-T)
$$

令 $W'(t)=0$ 得

$$
\boxed{t^*=\frac{T}{2}},\qquad \text{最小总期望等待}=W(T/2)=\frac{\lambda T^2}{4}
$$

> [!NOTE] 辨析（"平均等待 = 区间长度的一半"从哪来）
> 这一步完全依赖 Theorem 5.2：**给定区间内到达次数，各到达时刻在该区间上均匀分布**，所以平均等待是该区间长度的一半。如果到达是"集中在区间起点"或"集中在区间终点"，最优 $t$ 就不再是 $T/2$。这也是本题的隐含假设（泊松到达）唯一被用到的地方。

---

### 3.5 Proposition 5.7（条件在"第 $n$ 次到达时刻"上）

**命题**：给定 $S_n=t$，集合 $\{S_1,\dots,S_{n-1}\}$ 的分布与"$n-1$ 个独立 $\text{Uniform}(0,t)$ 随机变量"的分布相同。

**证明（要点）**：$S_n=t$ 意味着 $t$ 之前恰好发生了 $n-1$ 次事件（第 $n$ 次正好落在 $t$），故可以用"条件再加条件"的办法把已知信息拆成两步：

$$
\{S_1,\dots,S_{n-1}\}\mid S_n=t
\ \overset{d}{=}\
\{S_1,\dots,S_{n-1}\}\mid S_n=t,\ N(t^-)=n-1
\ \overset{d}{=}\
\{S_1,\dots,S_{n-1}\}\mid N(t^-)=n-1
$$

其中 $t^-$ 表示略小于 $t$ 的时刻。然后直接用 Theorem 5.2（把 $t$ 换成 $t^-$）即得。

> [!NOTE] 辨析（Theorem 5.2 与 Proposition 5.7 条件的差异）
> | | 条件 | 含义 | 结论 |
> |---|---|---|---|
> | **Theorem 5.2** | $N(t)=n$ | 固定**区间内的事件总数**，$t$ 是任取的观察窗 | $S_1,\dots,S_n$ 是 $n$ 个 $\text{Uniform}(0,t)$ 的次序统计量 |
> | **Proposition 5.7** | $S_n=t$ | 固定**第 $n$ 次到达的准确时刻** | $S_1,\dots,S_{n-1}$ 是 $n-1$ 个 $\text{Uniform}(0,t)$ 的次序统计量 |
>
> 直观上：$S_n=t$ 已经把"第 $n$ 次事件"钉在 $t$ 处，剩下 $n-1$ 个时刻在 $(0,t)$ 里均匀散布，且它们与"$t$ 是第 $n$ 次"这一信息无关。常见用途：给定"第 $n$ 次事件发生在 $t$"，求前面几次的期望位置、求前面某段区间内的事件数分布等。

---

## 第 4 部分　泊松过程的抽样（Sampling a Poisson Process）

### 4.1 时变的分类概率

仍设 $\{N(t)\}$ 是速率 $\lambda$ 的泊松过程，有 $k$ 种可能的事件类型。与第 1 部分不同的是：**分类概率依赖于事件发生的时刻**。

若某事件发生在时刻 $y$，则它被归为 $i$ 型事件的概率为 $P_i(y)$（$i=1,\dots,k$），且

$$
\sum_{i=1}^k P_i(y)=1\quad\text{对一切 }y
$$

分类仍是"与之前发生的一切无关"的独立判定。

> [!NOTE] 辨析（这就是"抽样"与"分裂"的分界）
> 第 1 部分（分裂）是 $P_i(y)\equiv p_i$ 的**特例**；本部分是 $P_i(y)$ 随 $y$ 变化的**一般情形**。举两个直观例子：
> - 索赔"在时刻 $y$ 被归为大额"的概率随 $y$ 变（通胀），$P_1(y)$ 递增；
> - "在时刻 $s$ 感染的病人到时刻 $t$ 已出现症状"的概率是 $G(t-s)$，随感染时刻 $s$ 递减。
> 只要 $P_i$ 依赖 $y$，就不能直接套 Proposition 5.5。

### 4.2 Proposition 5.6（抽样定理）

记 $N_i(t)$ 为到时刻 $t$ 为止 $i$ 型事件的个数。则 $N_1(t),\dots,N_k(t)$ 是**相互独立的泊松随机变量**，且

$$
\boxed{E[N_i(t)]=\lambda\int_0^t P_i(s)\,ds},\qquad i=1,\dots,k
$$

> [!WARNING] 易错点（命题说的是"随机变量"，不是"过程"）
> Proposition 5.6 的结论只在**固定时刻 $t$ 的边缘分布**层面成立：$N_1(t),\dots,N_k(t)$ 独立且各自泊松。但当 $P_i(y)$ 依赖 $y$ 时，$\{N_i(t),\ t\ge0\}$ **一般不再是泊松过程**——因为 $P\{N_i(t+h)-N_i(t)=1\}\approx \lambda h\,P_i(t)$，这个数依赖 $t$，违反 Definition 5.2 公理 (iii) 中"$=\lambda h+o(h)$、速率 $\lambda$ 为常数"的要求，平稳增量也不成立。只有 $P_i(y)\equiv p_i$（常数）时，$\{N_i(t)\}$ 才恢复为速率 $\lambda p_i$ 的泊松过程（这时的结论比 Prop 5.6 更强，还要多一个"增量也独立"）。
>
> 一句话：**常数 $p$ 给"过程"，时变 $P_i(y)$ 只给"固定时刻的分布"。**

### 4.3 用 Theorem 5.2 证明 Proposition 5.6

**第一步**：若各类事件数为 $n_1,\dots,n_k$，则总事件数必为 $\sum_i n_i$。对 $N(t)$ 取条件：

$$
P\{N_1(t)=n_1,\dots,N_k(t)=n_k\}
=P\left\{N_i(t)=n_i\ \Big|\ N(t)=\textstyle\sum_i n_i\right\}\cdot P\left\{N(t)=\textstyle\sum_i n_i\right\}
$$

**第二步**：由 Theorem 5.2，在 $N(t)=\sum n_i$ 的条件下，每一个事件发生的时刻都服从 $[0,t]$ 上的均匀分布。于是对**任意一个**事件，"它是 $i$ 型"的概率为

$$
P_i=\frac{1}{t}\int_0^t P_i(s)\,ds
$$

（这里借用了教材的记号：**带自变量**的 $P_i(s)$ 是"在时刻 $s$ 事件为 $i$ 型"的概率，**不带自变量**的 $P_i$ 是它在一个到达上的**平均**值——注意是 $P_i(y)$ 对时间的平均，不是 $P_i$ 在某个时刻的值。）且各事件的分类相互独立。故条件分布是**多项分布**：

$$
P\left\{N_i(t)=n_i\ \Big|\ N(t)=\textstyle\sum_i n_i\right\}
=\frac{(\sum_i n_i)!}{n_1!\cdots n_k!}\,P_1^{n_1}\cdots P_k^{n_k}
$$

**第三步**：乘上 Poisson$(\lambda t)$ 的概率质量函数，并利用 $\lambda t P_i=\lambda\int_0^t P_i(s)ds$：

$$
P\{N_i(t)=n_i,\ i=1,\dots,k\}
=\frac{(\sum_i n_i)!}{\prod_i n_i!}\prod_i P_i^{n_i}\cdot e^{-\lambda t}\frac{(\lambda t)^{\sum_i n_i}}{(\sum_i n_i)!}
=\prod_{i=1}^k e^{-\lambda t P_i}\frac{(\lambda t P_i)^{n_i}}{n_i!}
$$

右端恰好是 $k$ 个泊松概率质量函数的**乘积**，故 $N_1(t),\dots,N_k(t)$ 相互独立，且 $N_i(t)\sim\text{Poisson}(\lambda t P_i)=\text{Poisson}\left(\lambda\int_0^t P_i(s)ds\right)$。证毕。

> [!IMPORTANT] 重要结论（两个"自由度"的分离）
> 这条证明把"随机性"干净地拆成两层：**总数** $N(t)\sim\text{Poisson}(\lambda t)$ 与**标签**（在给定总数下服众多项分布）。常数 $p$ 的情形与它完全同构，只是 $P_i$ 从积分退化为常数 $p_i$。记住这个"泊松 + 多项"的分解，分裂的一切结论都可现场推出。

---

### Example 5.18　无限服务台的排队系统（$M/G/\infty$）

顾客按速率 $\lambda$ 的泊松过程到达服务台，到达后立即由无穷多个服务台之一开始服务（即**永远不用排队**）。服务时间相互独立、服从共同分布 $G$。求：

- $X(t)$：到时刻 $t$ 已完成服务的顾客数的分布；
- $Y(t)$：时刻 $t$ 正在被服务的顾客数的分布；
- $Y(t)$ 与 $Y(t+s)$ 的联合分布（进而协方差）。

**解答（前两问）**：对时刻 $t$ 而言，把每个到达者分成两类：

- **I 型**：到 $t$ 已完成服务 —— 在时刻 $s\le t$ 到达者属于此类当且仅当服务时间 $<t-s$，概率 $G(t-s)$；
- **II 型**：到 $t$ 尚未完成服务 —— 概率 $\bar G(t-s)=1-G(t-s)$。

这正是"分类概率依赖到达时刻"的抽样，由 Proposition 5.6（令 $y=t-s$ 换元）：

$$
X(t)\sim\text{Poisson}\left(\lambda\int_0^t G(t-s)\,ds\right)
=\text{Poisson}\left(\lambda\int_0^t G(y)\,dy\right)
\tag{5.17}
$$

$$
Y(t)\sim\text{Poisson}\left(\lambda\int_0^t \bar G(t-s)\,ds\right)
=\text{Poisson}\left(\lambda\int_0^t \bar G(y)\,dy\right)
\tag{5.18}
$$

并且 $X(t)$ 与 $Y(t)$ **相互独立**。

**第三问（联合分布与协方差）：四类分解。** 现在有两个时刻 $t$ 与 $t+s$，把一次到达按"何时到"与"何时完成服务"分成四类：

| 类型 | 到达时刻 | 完成服务的时刻 | 分类概率 $P_i(y)$ |
|---|---|---|---|
| 1 | $y<t$ | 落在 $(t,t+s)$ 内 | $G(t+s-y)-G(t-y)$ |
| 2 | $y<t$ | 晚于 $t+s$ | $\bar G(t+s-y)$ |
| 3 | $t<y<t+s$ | 晚于 $t+s$ | $\bar G(t+s-y)$ |
| 4 | 其它 | —— | $1-P_1-P_2-P_3$ |

由 Proposition 5.6，$N_1,N_2,N_3$（各类的个数，$N_i\equiv N_i(t+s)$）是**相互独立**的泊松随机变量，均值为

$$
E[N_i]=\lambda\int_0^{t+s}P_i(y)\,dy,\qquad i=1,2,3
$$

而两个时刻的在系统人数可以写成

$$
Y(t)=N_1+N_2,\qquad Y(t+s)=N_2+N_3
$$

（$N_2$ 是"两个时刻都在系统中"的那批人，正是唯一的公共部分。）于是协方差极易计算：

$$
\mathrm{Cov}[Y(t),Y(t+s)]
=\mathrm{Cov}[N_1+N_2,\ N_2+N_3]
=\mathrm{Cov}[N_2,N_2]
=\mathrm{Var}(N_2)=E[N_2]
$$

（交叉项因 $N_1,N_2,N_3$ 独立而全为 0。）代入 $P_2(y)=\bar G(t+s-y)$（$y<t$），并用 $u=t-y$ 换元：

$$
\mathrm{Cov}[Y(t),Y(t+s)]
=\lambda\int_0^t \bar G(t+s-y)\,dy
=\lambda\int_0^t \bar G(u+s)\,du
$$

联合分布则由"$N_1+N_2=i$、$N_2+N_3=j$"对标量 $N_2=\ell$ 求和得到：

$$
P\{Y(t)=i,\ Y(t+s)=j\}
=\sum_{\ell=0}^{\min(i,j)}
P\{N_2=\ell\}\,P\{N_1=i-\ell\}\,P\{N_3=j-\ell\}
$$

> [!NOTE] 辨析（独立性靠的是"无限服务台"）
> $X(t)$ 与 $Y(t)$ 独立、$Y(t)$ 仍服从泊松分布，这两条**都依赖"无穷多服务台"**这一假设：顾客之间互不影响，所以"每人是否已完成"就是彼此独立的分类。若服务台有限，后到者要排队，其服务开始时刻取决于前人的服务时间——此时 $Y(t)$ 不再泊松，$X(t)$ 与 $Y(t)$ 也不再独立。**无限服务台排队是泊松抽样最标准的应用场景。**

> [!TIP] 方法（求两个时刻计数的协方差：找"公共部分"）
> 把 $Y(t)$、$Y(t+s)$ 都表示成同一组**互不相交类型**的计数之和，公共类型就是协方差的唯一来源：$\mathrm{Cov}[Y(t),Y(t+s)]=\mathrm{Var}(\text{两个时刻都在系统的人数})$。而泊松变量的方差等于期望，故只需算出它的均值。这个技巧在"计数过程的相关性"题里几乎通用。注意由此自动得到 $0\le\mathrm{Cov}\le\mathrm{Var}[Y(t)]$，即相关系数不超过 1。

---

### Example 5.20　追踪 HIV 感染人数

HIV 感染到出现艾滋病症状之间有很长的潜伏期，因此公共卫生部门难以确定任一时刻的感染人数。建立近似模型：

- 个体按速率 $\lambda$（**未知**）的泊松过程感染 HIV；
- 从感染到出现症状的时间服从已知分布 $G$；
- 不同感染者的潜伏期相互独立。

记 $N_1(t)$ 为到 $t$ 为止**已出现症状**的人数，$N_2(t)$ 为到 $t$ 为止**已感染但无症状**的人数。求可由观测量估计 $N_2(t)$ 的方法。

**解答（第一步：分类）。** 在时刻 $s\le t$ 感染的人，到 $t$ 时已出现症状的概率是 $G(t-s)$，未出现症状的概率是 $\bar G(t-s)$。这是"分类概率依赖感染时刻"的抽样，由 Proposition 5.6，

$$
N_1(t)\sim\text{Poisson}\left(\lambda\int_0^t G(t-s)ds\right)=\text{Poisson}\left(\lambda\int_0^t G(y)dy\right)
$$

$$
N_2(t)\sim\text{Poisson}\left(\lambda\int_0^t \bar G(t-s)ds\right)=\text{Poisson}\left(\lambda\int_0^t \bar G(y)dy\right)
$$

且二者相互独立。

**第二步（估计未知速率）。** 若 $\lambda$ 已知，$E[N_2(t)]$ 就是 $N_2(t)$ 的最佳猜测。但 $\lambda$ 未知，而 $N_1(t)$ 是**可观测**的。用观测量代替其均值：

$$
n_1\approx E[N_1(t)]=\lambda\int_0^t G(y)\,dy
\quad\text{由此解出}\quad
\hat\lambda=\frac{n_1}{\displaystyle\int_0^t G(y)\,dy}
$$

**第三步（估计 $N_2(t)$）。**

$$
\widehat{N_2(t)}=\hat\lambda\int_0^t \bar G(y)\,dy
=n_1\cdot\frac{\displaystyle\int_0^t \bar G(y)\,dy}{\displaystyle\int_0^t G(y)\,dy}
$$

**第四步（取指数潜伏期的算例）。** 若 $G$ 是均值为 $\mu$ 的指数分布，则 $G(y)=1-e^{-y/\mu}$、$\bar G(y)=e^{-y/\mu}$，简单积分给出

$$
\int_0^t G(y)\,dy=t-\mu\left(1-e^{-t/\mu}\right),\qquad
\int_0^t \bar G(y)\,dy=\mu\left(1-e^{-t/\mu}\right)
$$

故

$$
\widehat{N_2(t)}=n_1\cdot\frac{\mu\left(1-e^{-t/\mu}\right)}{t-\mu\left(1-e^{-t/\mu}\right)}
$$

取 $t=16$ 年、$\mu=10$ 年、$n_1=220$（千人）：

$$
\widehat{N_2(16)}=220\cdot\frac{10(1-e^{-1.6})}{16-10(1-e^{-1.6})}
=220\cdot\frac{7.98103}{16-7.98103}
=220\cdot 0.995269
\approx\boxed{218.96\ (\text{千人})}
$$

即若潜伏期为均值 10 年的指数分布、且前 16 年已出现症状者共 22 万人，则估计 $t=16$ 时约 **21.9 万**人已感染但无症状。

> [!NOTE] 辨析（为什么必须用 Prop 5.6 而不能用 Prop 5.5）
> 分类概率 $G(t-s)$、$\bar G(t-s)$ **依赖感染时刻 $s$**，所以这不是"每个事件以固定概率 $p$ 独立分类"的分裂。正确做法是把 $[0,t]$ 上每个到达时刻的贡献按 $P_i(s)$ **积分**起来。若图省事，把所有感染者都当成"刚在 $t$ 时刻感染"（用 $G(t)$、$\bar G(t)$ 代替 $G(t-s)$、$\bar G(t-s)$），在同一组数据下会得到 $220\cdot\dfrac{e^{-1.6}}{1-e^{-1.6}}\approx55.6$（千人），只有正确答案 $218.96$ 的四分之一左右，属严重低估。原因是这个错误做法把"潜伏期已过完大部分"的早期感染者当成"潜伏期刚开始"，从而大幅压低了无症状者的比例。

> [!WARNING] 易错点（模型假设的局限）
> 该模型假设感染速率 $\lambda$ **恒定不变**，即疫情以稳定速度扩散。对艾滋病流行而言这显然是一个弱点（实际 $\lambda$ 随时间增长）。教材也明确提示了这一点。做题时若问"模型的最强假设是什么"，答案就是这个"$\lambda$ 为常数"。

---

## 附 1：核心概念与辨析总结表

| 概念 | 条件 / 设定 | 结论 | 关键辨析 |
|---|---|---|---|
| **分裂**（Prop 5.5） | 每个事件独立地以**常数** $p$ 归为 I 型 | $\{N_1\},\{N_2\}$ 是速率 $\lambda p$、$\lambda(1-p)$ 的**泊松过程**且**独立** | 概率必须恒定；结论是"过程级"的 |
| **叠加** | 两个独立泊松过程速率 $\lambda_1,\lambda_2$ | 合并过程速率 $\lambda_1+\lambda_2$；每个事件来自过程 1 的概率 $\lambda_1/(\lambda_1+\lambda_2)$ | 与分裂互逆；是第 2 部分的工具 |
| **抽样**（Prop 5.6） | 分类概率 $P_i(y)$ **依赖事件时刻** $y$ | $N_i(t)$ 相互独立、$\text{Poisson}\left(\lambda\int_0^t P_i(s)ds\right)$ | 只有"固定 $t$ 的分布"，整条过程**不是**泊松过程 |
| **两个过程竞赛** | 独立泊松过程，求 $P\{S_n^{(1)}<S_m^{(2)}\}$ | 化为硬币：$\sum_{k=n}^{n+m-1}\binom{n+m-1}{k}p^k(1-p)^{n+m-1-k}$ | 只数前 $n+m-1$ 次抛掷；$p=\lambda_1/(\lambda_1+\lambda_2)$ |
| **单事件条件分布** | $N(t)=1$ | $S_1\mid N(t)=1\sim\text{Uniform}(0,t)$，$P\{S_1<s\}=s/t$ | 与 $\lambda$ 无关 |
| **Theorem 5.2** | $N(t)=n$ | $S_1,\dots,S_n$ = $n$ 个 $\text{Uniform}(0,t)$ 的**次序统计量**，密度 $n!/t^n$ | 无序看：独立同均匀；有序看：**不独立** |
| **次序统计量** | $Y_i$ i.i.d. 密度 $f$ | 联合密度 $n!\prod_i f(y_i)$ | $n!$ 来自 $n$ 个排列 |
| **Prop 5.7** | $S_n=t$ | $S_1,\dots,S_{n-1}$ = $n-1$ 个 $\text{Uniform}(0,t)$ 的次序统计量 | 与 Thm 5.2 的条件不同（固定末尾 vs 固定总数） |
| **可用推论** | $N(t)=n$ | 落入区间 $A$ 的个数 $\sim\text{Bin}(n,|A|/t)$；$E[S_k]=kt/(n+1)$ | 这两条考试最高频 |
| **无限服务台** | $M/G/\infty$ | $X(t),Y(t)$ 独立泊松，均值 $\lambda\int_0^t G$、$\lambda\int_0^t\bar G$ | 独立性依赖"无穷多服务台" |
| **两时刻协方差** | $Y(t),Y(t+s)$ | $=\mathrm{Var}(N_2)=\lambda\int_0^t\bar G(u+s)du$ | 只来自"两时刻都在系统中"的那批人 |

## 附 2：例题答案速查

| 例题 | 问题 | 答案 |
|---|---|---|
| **Example 5.14** | 二月无英裔移民 | $P=e^{-10/3}\approx0.0357$（均值 $4\cdot10\cdot\frac{1}{12}=\frac{10}{3}$） |
| **Example 5.15** | 最优门槛 $y^*$ | $\displaystyle\int_{y^*}^\infty(x-y^*)f(x)dx=\frac{c}{\lambda}$；最大净收益 $=y^*$ |
| | 最优策略下的净收益 | $E[R(y^*)]=y^*$（门槛值 = 最优值） |
| **Example 5.16** | $n$ 步后各状态人数 | 相互独立，第 $j$ 个为 $\text{Poisson}\left(\sum_i\lambda_i P_{ij}^n\right)$ |
| **两过程竞赛** | $P\{S_n^{(1)}<S_m^{(2)}\}$ | $\displaystyle\sum_{k=n}^{n+m-1}\binom{n+m-1}{k}\left(\frac{\lambda_1}{\lambda_1+\lambda_2}\right)^k\left(\frac{\lambda_2}{\lambda_1+\lambda_2}\right)^{n+m-1-k}$ |
| | $n=1,m=1$ | $\dfrac{\lambda_1}{\lambda_1+\lambda_2}$ |
| | $n=2,m=1$ | $\left(\dfrac{\lambda_1}{\lambda_1+\lambda_2}\right)^2$ |
| **单事件** | $P\{S_1<s\mid N(t)=1\}$ | $s/t$ |
| **Thm 5.2** | $f(s_1,\dots,s_n\mid n)$ | $n!/t^n$，$0<s_1<\cdots<s_n<t$ |
| | 落入 $A\subseteq[0,t]$ 的个数 | $\text{Bin}(n,|A|/t)$ |
| | $E[S_k\mid N(t)=n]$ | $kt/(n+1)$ |
| **Example 5.21** | $E[D(t)]$ | $\dfrac{\lambda\mu}{\alpha}\left(1-e^{-\alpha t}\right)$ |
| **Example 5.22** | 最优发运时刻 / 最小总期望等待 | $t^*=T/2$；$\lambda T^2/4$ |
| **Prop 5.7** | 给定 $S_n=t$ 时 $S_1,\dots,S_{n-1}$ | $n-1$ 个独立 $\text{Uniform}(0,t)$ 的次序统计量 |
| **Prop 5.6** | $E[N_i(t)]$ | $\lambda\displaystyle\int_0^t P_i(s)\,ds$，且各 $N_i(t)$ 独立 |
| **Example 5.18** | $E[X(t)]$ / $E[Y(t)]$ | $\lambda\int_0^t G(y)dy$ / $\lambda\int_0^t\bar G(y)dy$，且独立 |
| | $\mathrm{Cov}[Y(t),Y(t+s)]$ | $\lambda\int_0^t\bar G(u+s)du$ |
| **Example 5.20** | 无症状感染者估计 | $n_1\dfrac{\int_0^t\bar G(y)dy}{\int_0^t G(y)dy}$；指数潜伏期时 $=n_1\dfrac{\mu(1-e^{-t/\mu})}{t-\mu(1-e^{-t/\mu})}$，$t=16,\mu=10,n_1=220$ 得 $218.96$ |
