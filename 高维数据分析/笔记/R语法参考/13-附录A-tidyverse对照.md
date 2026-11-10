# 附录 A　tidyverse 对照

## 一、为什么只作附录

课程代码与教材配套代码用的是 base R：`lm()`、`summary()`、`plot()`、`data.frame` 子集。作业提交时沿用这套写法，助教看得懂，也不会因为两套语法混用而出错。但读别人的代码、上网搜示例时，tidyverse 的写法很常见，所以这里给一份对照表，知道两边在做什么即可。

`dplyr` 的核心是五个动词：`filter()` 筛行、`select()` 选列、`mutate()` 加列、`arrange()` 排序、`summarise()` 汇总，配合 `group_by()` 分组。它们都接受数据框作为第一个参数，返回新的数据框，原数据不变。

| 操作 | base R | dplyr |
| :--- | :--- | :--- |
| 筛行 | `d[d$x > 1, ]` | `filter(d, x > 1)` |
| 选列 | `d[, c("x", "y")]` | `select(d, x, y)` |
| 加列 | `d$z <- d$x + d$y` | `mutate(d, z = x + y)` |
| 排序 | `d[order(d$x), ]` | `arrange(d, x)` |
| 降序 | `d[order(-d$x), ]` | `arrange(d, desc(x))` |
| 分组汇总 | `aggregate(y ~ g, d, mean)` | `d \|> group_by(g) \|> summarise(m = mean(y))` |
| 左连接 | `merge(a, b, by = "id", all.x = TRUE)` | `left_join(a, b, by = "id")` |
| 纵向拼接 | `do.call(rbind, list(a, b))` | `bind_rows(a, b)` |
| 去重 | `unique(d)` | `distinct(d)` |
| 取列向量 | `d$x` | `pull(d, x)` |
| 改名 | `names(d)[2] <- "y"` | `rename(d, y = old)` |
| 条件分支 | 嵌套 `ifelse()` | `case_when()` |

```r
library(dplyr)
mtcars |>
  filter(cyl == 4) |>
  select(mpg, wt, hp) |>
  arrange(desc(mpg)) |>
  head(3)
#>                 mpg    wt  hp
#> Toyota Corolla 33.9 1.835  65
#> Fiat 128       32.4 2.200  66
#> Honda Civic    30.4 1.615  52

mtcars |>
  group_by(cyl) |>
  summarise(n = n(), mean_mpg = mean(mpg))
#> # A tibble: 3 × 3
#>     cyl     n mean_mpg
#>   <dbl> <int>    <dbl>
#> 1     4    11     26.7
#> 2     6     7     19.7
#> 3     8    14     15.1
```

`summarise()` 返回的是 tibble，打印时带列类型与行数提示，和 `data.frame` 的显示不同，但按数据框用即可。

## 二、管道

`dplyr` 的管道 `%>%` 来自 `magrittr`，R 4.1 之后自带的 `|>` 已能覆盖大多数场合。两者的区别是：`%>%` 支持 `.` 占位符和任意位置，`|>` 用 `_` 占位且要求占位符出现在具名参数里；`|>` 不需要额外加载包，速度也更快。新代码优先用 `|>`。

```r
library(magrittr)
mtcars %>% filter(cyl == 6) %>% nrow()
#> [1] 7
mtcars |> filter(cyl == 6) |> nrow()
#> [1] 7
```

## 三、ggplot2

`ggplot2` 用"图形语法"拼图层：`ggplot(data, aes(...))` 定数据与映射，`geom_*()` 加几何对象，`+` 叠加。它和 base R 不是一套体系，不能把 `points()`、`abline()` 加到 ggplot 上。

```r
library(ggplot2)
ggplot(mtcars, aes(x = wt, y = mpg, colour = factor(cyl))) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "车重（千磅）", y = "每加仑英里数", colour = "缸数") +
  theme_bw()
```

常用图层与 base R 的对应关系：

| 目的 | base R | ggplot2 |
| :--- | :--- | :--- |
| 散点 | `plot(x, y)` | `geom_point()` |
| 折线 | `lines(x, y)` | `geom_line()` |
| 回归线 | `abline(lm(y ~ x))` | `geom_smooth(method = "lm")` |
| 直方图 | `hist(x)` | `geom_histogram()` |
| 箱线图 | `boxplot(y ~ g)` | `geom_boxplot()` |
| 条形图 | `barplot(table(x))` | `geom_bar()` |
| 分面 | `par(mfrow)` 手工分 | `facet_wrap(~ g)` |
| 坐标范围 | `xlim`、`ylim` | `coord_cartesian(xlim = ...)` |
| 保存 | `png()` + `dev.off()` | `ggsave("a.png", width = 6, height = 4)` |

`ggsave()` 保存最近一张图，默认按当前图形窗口的比例，宽高单位是英寸：

```r
ggsave(file.path(tempdir(), "mtcars.png"), width = 6, height = 4, dpi = 150)
```

## 四、混用的代价

两套语法混用最容易出三类问题。其一是**掩码**：加载 `dplyr` 后 `filter()` 变成 `dplyr::filter`，`stats::filter()` 的时间序列滤波失效，`lag()` 同理，必须写全名。其二是**数据结构**：tibble 不支持行名，`d[1, ]` 的行为与数据框略有差异，老代码依赖行名时会出错。其三是**公式环境**：在管道里写 `lm(y ~ x)`，公式捕获的是管道内部环境，后续 `update()`、`predict()` 可能找不到变量，稳妥写法是显式 `data =`。

结论很简单：**作业与课程代码用 base R，读别人的 tidyverse 代码时按上表翻译**，不要在同一份脚本里来回切换。

---

上一章：[[12-报错与调试]]　下一章：[[14-附录B-速查表]]
