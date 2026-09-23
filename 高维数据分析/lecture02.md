线性回归模型
$$
\begin{align}
 & y_{i} = \beta_{0}^{*} + \beta_{1}^{*} x_{i} + e_{i}, \quad i = 1 ,\cdots , n \\
 & e_{1}, \cdots , e_{n} \overset{i.i.d.}{\sim} N(0 ,\sigma_{*}^{2})
\end{align}
$$

通过 MLE 估计参数 $\beta_{0}^{*}, \beta_{1}^{*}, \sigma_{*}^{2}$ ，易得
$$
\sigma^{2} = \frac{1}{n} \sum_{i=1}^{n}(y_{i} - \beta_{0} - \beta_{1}x_{i})^{2}
$$
因此
$$
(\hat{\beta}_{0}, \hat{\beta}_{1}) = \arg \min_{\beta_{0}, \beta_{1}}  f(\beta_{0}, \beta_{1}) = \sum_{i=1}^{n}(y_{i} - \beta_{0} - \beta_{1}x_{i})^{2}
$$
可得
$$
\hat{\beta}_{0} = \bar{y} - \hat{\beta}_{1} \bar{x}, \quad \hat{\beta}_{1} = \frac{SXY}{SXX}
$$

- $SXX = \sum_{i=1}^{n} (x_{i} - \bar{x})^{2}$
- $SXY = \sum_{i=1}^{n}(x_{i} - \bar{x})(y_{i} - \bar{y})$
称
$$
\hat{e}_{i} = y_{i} - \beta_{0} - \beta_{1}x_{i}
$$
为残差（**R**esidual）。残差平方和 RSS 为 $f(\hat{\beta}_{0}, \hat{\beta}_{1}) = \sum_{i=1}^{n} \hat{e}_{i}^{2}$ ，因此 $\sigma_{*}^{2}$ 的极大似然估计 $\tilde{\sigma}^{2} = RSS / n$

---
极大似然估计有以下的性质
$$
\hat{\beta}_{1} \sim N\left(  \beta^{*}_{1}, \frac{\sigma_{*}^{2}}{SXX} \right), \hat{\beta}_{0} \sim N\left( \beta^{*}_{0}, \sigma_{*}^{2}\left( \frac{1}{n} + \frac{\bar{x}^{2}}{SXX} \right) \right)
$$

关于 RSS，有以下的结论
- $RSS / \sigma_{*}^{2} \sim \chi^{2}(n -2)$（由于自由度），因此 $\tilde{\sigma}^{2}$ 的 MLE 不是无偏的，通过纠偏可得 $\sigma_{*}^{2}$ 的无偏估计为 $\hat{\sigma}^{2} = RSS / (n - 2)$ 
- RSS 和 $\hat{\beta}_{0}$ 以及 $\hat{\beta}_{1}$ 相互独立

标准误：
- $\mathrm{se}(\hat{\beta}_{1}) = \sqrt{ \hat{Var}(\hat{\beta}_{1}) } = \frac{\hat{\sigma}}{\sqrt{ SXX }}$
- $\mathrm{se}(\hat{\beta}_{1}) = \sqrt{ \hat{Var}(\hat{\beta}_{0}) } = \hat{\sigma} \sqrt{ \frac{1}{n} + \frac{\bar{x}^{2}}{SXX} }$


$\hat{\beta}_{1}$ 的区间估计：
$$
G = \frac{\hat{\beta}_{1} - \beta_{1}^{*}}{\mathrm{se}(\hat{\beta}_{1})} \sim t(n-2)
$$
假设检验检验 $\beta^{*}_{1}$ 是否等于 $b_{1}$，检验统计量为
$$
T =\frac{ \hat{\beta}_{1} - b_{1}}{\mathrm{se}(\hat{\beta}_{1})}
$$
原假设成立时， $T \sim t(n - 2)$



---
可决系数（$R^{2}$）定义为
$$
R^{2} = \frac{SSreg}{ SYY} = 1 - \frac{RSS}{SYY}
$$
其中
- $SYY = \sum_{i=1}^{n}(y_{i} - \bar{y})^{2}$
- $SSreg = SYY - RSS$
