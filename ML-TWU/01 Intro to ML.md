ML 的目标是寻找一个函数
- 输入 Speech 输出文字内容
- 输入 Image 输出是什么内容
- 输入 Chess board 输出下一步是什么

不同种类的函数：
- **Regression**:输出的是一个 scaler 如输入各种和 PM2.5 有关的数据，输出明天的 PM2.5 预测值。(Predict PM2.5)
- **Classification**:做选择题：给定一些选项，输入是一个选项。如垃圾邮件的分类（spam filtering）Alpha GO 也是一种 Classification，只不过这个分类十分复杂，选项规模为 19\*19.

机器学习领域的黑暗大陆：除了 **Regression** 和 **Classifcation** 的任务外的问题：*structure learning*：
- 生成结构化的内容：如生成图片和文档。
- 生成一篇文章。（让机器学会*创造*）

## How to find a function
 Youtube 频道的观看数预测：从后台的历史观看数据预测隔天的数据。

### function
- 猜测未知函数的**形式**：预测值换观测值之间的可能关系，但是不知道具体的参数。带有未知参数的函数叫做 **Model**。
$$
y = b + wx_{1}
$$
- $x_{1}$ feature
- $w$ weight
- $b$ bias


### loss
第二个步骤：定义 Loss：一个函数，输入自变量为 Model 的参数： $L(b, w)$，用来衡量*参数选取的好坏*。 
比较预测值和真实值（又称*label*）的差异计算 Loss。
$$
Loss: L = \frac{1}{N} \sum_{n}e_{n}
$$
其中 $e_{n}$ 是每个观测值和预测值之间的误差，有不同的计算方法
- **绝对误差** $e = \lvert y - \hat{y} \rvert$，称为 Mean Absolute Error(*MAE*)
- **平方误差** $e = (y - \hat{y})^2$，称为 Mean Squared Error(*MSE*)
- 如果 $y$ 和 $\hat{y}$ 是概率分布（probability distributions），则可以通过 **Cross-Entropy** 计算。

可以通过穷举 $w$ 和 $b$ 来计算每个对应的 Loss，画出一张等高线图，称作 **Error Surface**。

## Optimization
优化参数的方法，**Gradient Descent**（GD）：使寻找最优的参数使得 Loss 最小。
对于一维情况：
- 随机选取一个初始点 $w^{0}$
- 计算 Loss 的梯度 $\nabla L(w^{0})$
- 如果为*负*，需要增加 $w$ 的值；如果为*正*，需要减少 $w$ 的值。
- 这一步的跨度取决于两个因素：该点的**梯度**和 $\eta$（**learning rate**），$\eta$ 可以自己设定（称作*hyperparameters*）
- 迭代 $w^{1} \leftarrow w^{0} - \textcolor{red}{\eta} \frac{ \partial L }{ \partial w }\big|_{w = w^{0}}$
- 反复进行上述操作
- 停止条件：达到最大迭代次数（这里也是 *hyperparameter*）；微分值正好为 0。

> [!info]
> GD 找到的不一定是最优的解，称非最优解为 **Local minima**，全局最优为 **Global minima**。
> 但实际计算和训练上 Local Minima 是一个**假问题**

---
curve = sum of a set of functions + constant

**Sigmoid**:
$$
y = \textcolor{red}{c} \frac{1}{1 + e^{-( \textcolor{green}{b} + \textcolor{blue}{w}x)}} = \textcolor{red}{c}\ sigmoid ( \textcolor{green}{b} + \textcolor{blue}{w}x)
$$
- 改变 w 可以调整 slope
- 改变 b 可以调整 shift
- 改变 c 可以改变 height

组合不同形式的 sigmoid 可以拟合一个 curve（piecewise linear）。

---
1 epoch = see all the batches onece

ML framework:
1. function with unknown
2. define loss from training data
3. optimization

---
overfitting：在训练数据上变好，在未知数据上变差


---
## Backpropagation
