## 前 7 题
![[1.jpg]]
![[2.jpg]]
![[3.jpg]]

## 第 8 题 Exercise 3.6
### 1
`BSAAM` 和 `OPBPC`、`OPRC`、`OPSLAKE` 组成的散点图矩阵如图所示，从相关性来看，相关性矩阵中两两变量之间的相关性应该都为大且正。
![[image.webp|434]]
下面是计算得到的相关性矩阵结果：
```r
            BSAAM     OPBPC      OPRC   OPSLAKE
BSAAM   1.0000000 0.8857478 0.9196270 0.9384360
OPBPC   0.8857478 1.0000000 0.8647073 0.9433474
OPRC    0.9196270 0.8647073 1.0000000 0.9191447
OPSLAKE 0.9384360 0.9433474 0.9191447 1.0000000
```
符合上述推断。

### 2
回归摘要：
```r
Call:
lm(formula = BSAAM ~ OPBPC + OPRC + OPSLAKE, data = water)

Residuals:
     Min       1Q   Median       3Q      Max 
-15964.1  -6491.8   -404.4   4741.9  19921.2 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 22991.85    3545.32   6.485  1.1e-07 ***
OPBPC          40.61     502.40   0.081  0.93599    
OPRC         1867.46     647.04   2.886  0.00633 ** 
OPSLAKE      2353.96     771.71   3.050  0.00410 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 8304 on 39 degrees of freedom
Multiple R-squared:  0.9017,    Adjusted R-squared:  0.8941 
F-statistic: 119.2 on 3 and 39 DF,  p-value: < 2.2e-16

```

其中的 `t value` 列代表 t 检验统计量 $T = \frac{\hat{\beta}_{j}-b_{j}}{\mathrm{se}(\hat{\beta}_{j})}$ 的值，原假设中 $b_{j} = 0$，因此 `t value` 足够大（对应 p 值足够小）时可以认为该变量的线性关系显著。

在该回归结果中，变量 `OPBPC` 的 `t value` 非常小，可以认为不显著。 `OPRC` 和 `OPSLAKE` 显著。

代码：
```r
library(alr4)

write.csv(water, file = "water.csv", row.names = FALSE)

scatterplotMatrix(~ BSAAM + OPBPC + OPRC + OPSLAKE, data = water)

sub <- water[, c("BSAAM", "OPBPC", "OPRC", "OPSLAKE")]

cor_mat <- cor(sub)
print(cor_mat)

res <- lm(BSAAM ~ OPBPC + OPRC + OPSLAKE, data = water)
print(summary(res))

```


## 第 9 题
### 1
回归结果：
```r
Call:
lm(formula = MEDV ~ ., data = boston)

Residuals:
    Min      1Q  Median      3Q     Max 
-15.595  -2.730  -0.518   1.777  26.199 

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  3.646e+01  5.103e+00   7.144 3.28e-12 ***
CRIM        -1.080e-01  3.286e-02  -3.287 0.001087 ** 
ZN           4.642e-02  1.373e-02   3.382 0.000778 ***
INDUS        2.056e-02  6.150e-02   0.334 0.738288    
CHAS         2.687e+00  8.616e-01   3.118 0.001925 ** 
NX          -1.777e+01  3.820e+00  -4.651 4.25e-06 ***
RM           3.810e+00  4.179e-01   9.116  < 2e-16 ***
AGE          6.922e-04  1.321e-02   0.052 0.958229    
DIS         -1.476e+00  1.995e-01  -7.398 6.01e-13 ***
RAD          3.060e-01  6.635e-02   4.613 5.07e-06 ***
TAX         -1.233e-02  3.760e-03  -3.280 0.001112 ** 
PTRATIO     -9.527e-01  1.308e-01  -7.283 1.31e-12 ***
B            9.312e-03  2.686e-03   3.467 0.000573 ***
LSTAT       -5.248e-01  5.072e-02 -10.347  < 2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 4.745 on 492 degrees of freedom
Multiple R-squared:  0.7406,    Adjusted R-squared:  0.7338 
F-statistic: 108.1 on 13 and 492 DF,  p-value: < 2.2e-16
```

结果中 `INDUS` 和 `AGE` 的 `t value` 都非常小，可以认为其线性关系不显著。其他变量可以认为都显著。

### 2
选择 `RM`、`DIS` 和 `PTRATIO` 三个变量回归，散点图矩阵如下：
![[image-1.webp|410]]
只考虑单独的自变量和因变量的关系，发现 `MEDV` 和 `RM`、`DIS` 都为正相关，和 `PTRATIO` 为负相关，而回归结果中 `DIS` 的系数为负，原因在于散点图中 `DIS` 也和其他未考虑的变量存在关联，进而会间接影响因变量；而回归结果中控制了其他变量等价，得出 `DIS` 的净效应是负的。


### 3
计算 VIF，发现 `RAD` 和 `TAX` 的 VIF 大于 5，尝试去掉这些变量对比回归效果。
```r
    CRIM       ZN    INDUS     CHAS       NX       RM      AGE      DIS      RAD      TAX  PTRATIO 
1.792192 2.298758 3.991596 1.073995 4.393720 1.933744 3.100826 3.955945 7.484496 9.008554 1.799084 
       B    LSTAT 
1.348521 2.941491 
```

- 去掉 `RAD` 后，调整后 R 方从 0.7338 降低到 0.7228，而 `TAX` 变得不显著，说明 `TAX` 的独立贡献弱。
- 去掉 `TAX` 后，调整后 R 方变为 0.7285
- 两者都去掉后，调整后 R 方变为 0.7232

综合上述回归结果，可以认为单独去掉 `TAX`，保留 `RAD`。

代码：
```r
library(alr4)
boston <- read.csv("boston.csv")

# 1. 回归结果
fit <- lm(MEDV ~ ., data = boston)

print(summary(fit))


# 2. 散点图矩阵
scatterplotMatrix(~ MEDV + RM + DIS + PTRATIO, data = boston)


# 3. 计算方差膨胀因子判断多重共线性
print(vif(fit))

fit1 <- lm(MEDV ~ . - RAD - TAX, data = boston)
print(summary(fit1))

fit2 <- lm(MEDV ~ . - TAX, data = boston)
print(summary(fit2))

fit3 <- lm(MEDV ~ . - RAD, data = boston)
print(summary(fit3))
```