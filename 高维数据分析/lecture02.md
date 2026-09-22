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
\hat{\beta}_{0} = \bar{y} - \hat{\beta}_{1} \hat{x}, \quad \hat{\beta}_{1} = \frac{SXY}{SXX}
$$

- $SXX = \sum_{i=1}^{n} (x_{i} - \bar{x})^{2}$
- $SXY = \sum_{i=1}^{n}(x_{i} - \bar{x})(y_{i} - \bar{y})$
称
$$
\hat{e}_{i} = y_{i} - \beta_{0} - \beta_{1}x_{i}
$$
为残差（**R**esidual）。残差平方和 RSS 为

