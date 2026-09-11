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
Chain Rule：
- $y = g(x), z = h(y)$，$\frac{\operatorname{d}\!z}{\operatorname{d}\!x} = \frac{\operatorname{d}\!z}{\operatorname{d}\!y} \frac{\operatorname{d}\!y}{\operatorname{d}\!x}$ 
- $x = g(s), y = h(s), z = k(x,y)$，$\frac{\operatorname{d}\!z}{\operatorname{d}\!s} = \frac{ \partial z }{ \partial x } \frac{\operatorname{d}\!x}{\operatorname{d}\!s} + \frac{ \partial z }{ \partial y } \frac{\operatorname{d}\!y}{\operatorname{d}\!s}$


定义误差（损失函数）
$$
L(\theta) = \sum_{i=1}^{N} C^{n} (\theta)
$$
其中 $C^{i}(\theta)$ 是 $y^{i}$ 和 $\hat{y}^{i}$ 的误差。因此计算损失函数对某个参数的偏微分时，有如下的公式
$$
\frac{ \partial L(\theta) }{ \partial w } = \sum_{i=1}^{N} \frac{ \partial C^{n}(\theta) }{ \partial w } 
$$


针对某个 neuron，其 input 为 $x_{1}, x_{2}$，定义 $z = x_{1} w_{1} + x_{2} w_{2} + b$，根据 Chain Rule，有
$$
\frac{ \partial C  }{ \partial w } = \frac{ \partial z }{ \partial w } \frac{ \partial C }{ \partial z }  
$$
分为两个部分
- Forward pass：计算 $\frac{ \partial z }{ \partial w }$
- Backward pass：计算 $\frac{ \partial C }{ \partial z }$

### Forward pass
$\frac{ \partial z }{ \partial w }$ 的值等于其中每个 $w$ 所对应的 input，如
- $\frac{ \partial z }{ \partial w_{1} } = x_{1}$
- $\frac{ \partial z }{ \partial w_{2} } = x_{2}$

对于第一层，其值等于 input layer 中的 input 值；对于其他层，其值等于前一层的 neuron 的输出

### Backward pass
接下来计算 $\frac{ \partial C }{ \partial z }$。首先定义 $a = \sigma(z)$，该值相当于下一层的一个 input，因此可以写成
$$
\frac{ \partial C }{ \partial z }  = \frac{ \partial a }{ \partial z } \frac{ \partial C }{ \partial a } 
$$
第一项 $\frac{ \partial a }{ \partial z }$ 为 sigmoid 的微分 $\sigma'(z)$。

第二项：考虑下一层中的两个 z（记为 $z' = a w_{3} + \cdots$ 和 $z'' = a w_{4} + \cdots$）根据 Chain Rule，可以写为
$$
\begin{align}
\frac{ \partial C }{ \partial a } &  = \frac{ \partial z' }{ \partial a }  \frac{ \partial C }{ \partial z' }  + \frac{ \partial z'' }{ \partial a }  \frac{ \partial C }{ \partial z'' }   \\
 & = w_{3} \frac{ \partial C }{ \partial z' } + w_{4} \frac{ \partial C }{ \partial z'' }  
\end{align}
$$

最终
$$
\frac{ \partial C }{ \partial z } = \sigma'(z)\left[  w_{3} \frac{ \partial C }{ \partial z' } + w_{4} \frac{ \partial C }{ \partial z'' }  \right]
$$

根据该公式的形式，可以反向计算微分，根据下一层的微分值得到当前层的微分值
- Output Layer：下一层是输出层，$\frac{ \partial C }{ \partial z' } = \frac{ \partial y_{1} }{ \partial z' } \frac{ \partial C }{ \partial y_{1} }$ 其中 $\frac{ \partial C }{ \partial y_{1} }$ 取决于损失函数的形式，可以是 MSE, Cross Entropy 等；而前一项取决于 activation function 的形式。
- Not Output Layer：从 Output Layer 开始计算，一层层反向得出微分


---
Regularization 正则化：在损失函数后添加正则项 $\lambda \sum (w_{i})^{2}$ 使得参数尽可能小，得到的目的函数会越平滑，输入发生变化时，函数值的变化比较小。正则化的对象是自变量前面的系数，而常数项 bias 对函数的平滑程度无影响。


---
## 分类问题
### Generative Model
- Function：$g(x) >0$ 为 Class1；否则为 Class0
- Loss Function： $L(f) = \sum_{n} \delta(f(x^{n}) \neq \hat{y}^{n})$
- Solution：从概率角度

给定一个 x，计算其属于每个 Class 的概率：
$$
P(C_{1} \mid x) = \frac{P(x \mid C_{1})P(C_{1})}{P(x \mid C_{1}) P(C_{1}) + P(x \mid C_{2}) P(C_{2})}
$$
同理可以计算 x 出现的概率 $P(x) = P(x \mid C_{1}) P(C_{1}) + P(x \mid C_{2}) P(C_{2})$



计算 $P(x \mid C)$：通过极大似然估计。所有的样本可由一个 Gaussian 分布表示，参数为 $\mu, \Sigma$，通过似然函数 $L(\mu, \Sigma)$ 可以得出使得样本出现概率最大的分布参数 $\mu^{*} = \frac{1}{n} \sum_{i=1}^{ n} x^{i}, \Sigma^{*} = \frac{1}{n} \sum_{i=1}^{n}(x^{i} - \mu^{*})(x^{i} - \mu^{*})^{\top}$ 

然后可以计算分类：当 $P(C \mid x) > 0.5$，则 x 属于 Class1.

参数优化：
- 每个 Class 的分布参数中 $\Sigma$ 不一样，模型的参数多，复杂度高，会导致 overfitting 出现。让每个 Class 中的 $\Sigma$ 相等：计算所有样本共同的似然函数，其中 $\Sigma$ 相等。注意 $\mu$ 的值和原来一致


若 x 可由若干个 feature 描述 $x = \begin{bmatrix}x_{1}  & x_{2}  &  \cdots  &  x_{K}\end{bmatrix}$ 且各维度之间相互独立，则
$$
P(x \mid C) = P(x_{1} \mid C) \cdots  P(x_{k} \mid C) \cdots 
$$
称为 Naive Bayes Classifier。


再对 $P(C_{1} \mid x)$ 改写：
$$
\begin{align}
P(C_{1} \mid x)  & = \frac{P(x \mid C_{1})P(C_{1})}{P(x \mid C_{1}) P(C_{1}) + P(x \mid C_{2}) P(C_{2})} \\
 & = \frac{1}{1 + \frac{P(x \mid C_{2})P(C_{2})}{P(x \mid C_{1})P(C_{1})}} \\
 &  = \frac{1}{1 + \exp (-z)} \\
 & = \sigma(z)
\end{align}
$$
其中
$$
z = \ln  \frac{P(x \mid C_{1})P(C_{1})}{P(x \mid C_{2})P(C_{2})}
$$
令不同分布的 $\Sigma$ 相等，即 $\Sigma^{1} = \Sigma^{2} = \Sigma$ ，则可以化简为
$$
z = (\mu^{1} - \mu^{2})^{\top}\Sigma^{-1}x - \frac{1}{2}(\mu^{1})^{\top}(\Sigma^{1})^{-1} \mu^{1} + \frac{1}{2}(\mu^{2})^{\top}(\Sigma^{2})^{-1}\mu^{2} + \ln  \frac{N_{1}}{N_{2}}
$$
- 令 $w = (\mu^{1} - \mu^{2})^{\top}\Sigma^{-1}$
- $b = - \frac{1}{2}(\mu^{1})^{\top}(\Sigma^{1})^{-1} \mu^{1} + \frac{1}{2}(\mu^{2})^{\top}(\Sigma^{2})^{-1}\mu^{2} + \ln  \frac{N_{1}}{N_{2}}$

则有
$$
P(C_{1} \mid x) = \sigma(w\cdot x + b)
$$


---
### Discriminative Model
$$
f_{w,b}(x) = \sigma\left( \sum_{i}w_{i}x_{i} + b \right)
$$

有训练数据 $(x^{n}, \hat{y}^{n})$，假设由 $f_{w,b}(x) = P_{w,b}(C_{1} \mid x)$ 产生，则可以计算产生该训练数据的概率
- 若 $x^{i}$ 的类别为 $C_{1}$，采样概率为 $P_{i} = f_{w,b}(x^{i})$
- 若 $x^{i}$ 的类别不为 $C_{1}$，采样概率为 $P_{i} = 1 - f_{w,b}(x^{i})$

定义 $L(w,b) = \prod_{i}P_{i}$，则可以找到一组最佳参数使得
$$
\begin{align}
w^{*}, b^{*}  & = \arg \max _{w,b} L(w,b)  \\
 & = \arg \min _{w,b} - \ln  L(w,b) \\
 & = \arg \min _{w,b} -\sum_{i} \ln  P_{i}
\end{align}
$$

对于求和的每一项，可以写成统一的形式：当 $x^{i}$ 属于 $C_{1}$ 时，令 $\hat{y}^{i} = 1$，否则为 0。因此有： 
$$
\ln P_{i} = \hat{y}^{i}\ln f_{w,b}(x^{i}) + (1 - \hat{y}^{i})\ln (1 - f_{w,b}(x^{i}))
$$
称为 **Cross Entropy**。


因此逻辑回归中的损失函数为将 Output 和 Target 分别当作一个 Bernoulli 分布，计算两者的接近程度（和Cross Entropy 相关）
$$
L(f) = \sum_{n} C(f(x^{n}, \hat{y}^{n}))
$$
其中
$$
C(f(x^{n}), \hat{y}^{n}) = - [\hat{y}^{n} \ln  f(x^{n}) + (1 - \hat{y}^{n})\ln (1 - f(x^{n}))]
$$

求微分：
$$
\frac{ \partial - \ln  L(w,b) }{ \partial w_{i} } = \sum_{n} - \left[  \hat{y}^{n} \frac{ \partial \ln  f_{w,b}(x^{n}) }{ \partial w_{i} }  + (1 - \hat{y}^{n})\frac{ \partial \ln (1 - f_{w,b}(x^{n})) }{ \partial w_{i} } \right] 
$$
已知：
- $f_{w,b}(x) = \sigma(z)$
- $z = w \cdot x + b$

因此结果为
$$
\begin{align}
\frac{ \partial - \ln  L(w,b) }{ \partial w_{i} } &  =\sum_{n} - [\hat{y}^{n}(1 - f_{w,b}(x^{n}))x^{n}_{i} + (1 - \hat{y}^{n})f_{w,b}(x^{n})x_{i}^{n}] \\
 & = \sum_{n} -(\hat{y}^{n} - f_{w,b}(x^{n}))x_{i}^{n}

\end{align}
$$

> [!tip] 为什么不用 Sqaure Error
> 相比 Cross Entropy，Square Error 的函数曲线更平缓 $\frac{ \partial (f_{w,b}(x) - \hat{y})^{2} }{ \partial w_{i} } = 2(f_{w,b}(x) - \hat{y})f_{w,b}(x)(1 - f_{w,b}(x))x_{i}$
> 不管是接近 Target 还是远离 Target，微分的值都会很小



计算 Logistic 回归的两种不同模型
- Discriminative：根据损失函数计算 $w$ 和 $b$
- Generative：学习数据的分布，计算 $\mu$ 和 $\Sigma^{-1}$，从而计算 $w$ 和 $b$

两者所计算出的最佳参数是不同的，Generative 模型基于一些假设，认为数据来自一个概率模型，训练数据是来自该概率模型的，并且有一些统计上的假设，如 Gaussian 分布、Naive Bayes 性质等。而 Discriminative 模型中没有任何限制，仅根据函数的特征寻找最佳的参数。


通常 Discriminative 模型的表现会比 Generative 的更好，有些情况下 Generative 模型具有优势：
- Discriminative 模型受数据量的影响大。而 Generative 模型基于概率假设，需要的数据量更少
- 先验 $P(C)$ 和条件概率 $P(x \mid C)$ 可以分开计算，从不同的数据来源计算得到更精确的结果

### Multi-class Classification
假设有 N 个 Class $C_{i}$，分别有参数 $w_{i}, b_{i}$，对于输入 $x$，计算 $z_{i} = w_{i} \cdot x + b_{i}$ 。对所有 $z_{i}$ 做 softmax 得到 $y_{i} = P(C_{i} \mid x)$

对于损失函数，可以计算 y 和 $\hat{y}$ 之间的 Cross Entropy：
$$
 - \sum_{i} \hat{y}_{i} \ln  y_{i}
$$
其中，若 $x$ 属于 Class i，则 $\hat{y}[i] = 1$


### Limitation
Logistic Regression 只能处理线性可分的数据，决策面是线性的。有时可以通过 Feature Transformation 将数据变成线性可分的，但需要手动根据数据的分布特征确定，更常用的方法是 Cascading logistic regression model：通过若干个 Logistic Regression （Sigmoid 函数）得到的值【此过程实现了 Feature Transformation】通过一个 Logistic Regression 得到结果


deep learning 的核心是通过增加 Hidden Layer ，达到比单个 Hidden Layer **更少的参数量**和更好的效果，需要更少的训练数据，从而不容易 Overfitting。