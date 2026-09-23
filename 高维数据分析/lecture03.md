多元线性回归模型：给定自变量 $X_{1} = x_{1}, \cdots, X_{p} = x_{p}$，Y 服从
$$
 Y \sim N(\beta^{*}_{0} + \beta_{1}^{*}x_{1} + \cdots  + \beta_{p}^{*} x_{p} , \sigma^{2}_{*})
$$
规定如下的参数 $Y \in \mathbb{R}^{n}, X \in \mathbb{R}^{n \times (p+1)}, \beta \in \mathbb{R}^{p+1}, e \in \mathbb{R}^{n}$，且 $n > p$， $X$ 满秩，则
$$
Y = X \beta^{*} + e
$$
MLE 估计 $\beta^{*}$，满足 $X^{\top}(Y - X\beta) = 0$，则 
$$
\hat{\beta} = (X^{\top}X)^{-1}X^{\top}Y
$$

估计量的分布为
$$
\hat{\beta} \sim N\{ \beta^{*}, \sigma_{*}^{2}(X^{\top} X)^{-1} \}
$$

和一元线性回归类似，有如下的结论
- $RSS /\sigma_{*}^{2} \sim \chi^{2}(n - p - 1)$
- RSS 和 $\hat{\beta}$ 相互独立


区间估计
$$
G = \frac{\hat{\beta}_{j} - \beta^{*}_{j}}{\mathrm{se}(\hat{\beta}_{j})} \sim t(n - p -1)
$$
标准误 $\mathrm{se}(\hat{\beta}_{j}) = \hat{\sigma}\sqrt{ c_{jj} }$（$c_{jj}$ 是 $(X^{\top}X)^{-1}$ 的第 j 对角元）


---
## 多重共线性问题
若 $p+1 > n$ 时， $X^{\top}X$ 不满秩
$$
rank(X^{\top}X) \leqslant  rank(X) \leqslant  n < p + 1
$$
OLS 无解。


接近线性相关，则 $X^{\top}X$ 接近奇异，因此 $tr((X^{\top}X)^{-1})$ 会非常大，使得估计的方差很大，造成估计不稳定。


方差膨胀因子 VIF 可以检测多重共线性
若
$$
X_{j} \sim X_{1} + \cdots  + X_{j-1} + X_{j+1} + \cdots  + X_{p}
$$
的 $R^{2}$ 为 $R^{2}_{(j)}$，则 $X_{j}$ 的 VIF 为
$$
VIF_{j} = \frac{1}{1 - R^{2}_{(j)}}
$$
- 若 $VIF_{j} > 10$，可认为存在共线性

也可通过自变量矩阵的条件数判断：
$$
\kappa = \frac{\sigma_{\max }(X)}{\sigma_{\min }(X)}
$$
条件数越大，共线性程度越高。

---
方差不同的情形： $e_{1},\cdots, e_{n}$ 方差各不相同，则称模型有**异方差性**，且 $Var(e_{i}) = \sigma_{i*}^{2}$，则
$$
\text{E}[\hat{\beta}] = \beta^{*}
$$
OLS 仍无偏，但有效性不保证。