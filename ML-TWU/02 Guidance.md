模型效果不好的优化方向：
- 检查训练数据上的 Loss
	- Loss 大的原因：模型 bias，需要增加模型复杂度；优化方面的问题，找不到最优解（判断方法：更复杂的模型的 Train Error 和 Test Error 都比原模型的更高）
- 若训练数据上的 Loss 小，检查测试数据上的 Loss
	- 测试数据 Loss 大，可能是 Overfitting。解决方法：
		1. 增加训练数据 
		2. Data Augmentation：根据对问题的理解创造新的数据（如放大图片、翻转等）
		3. 限制模型的弹性，变得不那么复杂（减少参数、共用参数等）
		4. Less features, Early stopping,  Regularization, Dropout
	-  Mismatch：训练数据和测试数据的分布不一致


![[image.webp|323]]
模型的表现随模型复杂度的变化情况，一个合理的模型应该是在可以接受的 Training Loss 范围中选择一个表现好的 Testing Loss。

> [!tip] 如何挑选最适合的模型
> 在训练数据上训练多个模型，根据在 public testing set 上的表现选择最佳的模型：由于随机性，模型可能会对公开测试数据产生过拟合，而在未知测试数据上表现很差。如果在private测试集上效果好，则模型在从未见过的其他数据上效果差的概率微乎其微。
> - **Cross Validation**：将训练数据划分成 Training Set 和小部分的 Validation Set，根据 Validation Set 上的表现选择最好的模型。从而在 public Testing Set 上的结果一定程度反应 private Testing Set 的表现（不建议再调整模型）
> - **N-folde Cross Validation**：为减少单个 Validation Set 划分的随机性，将训练数据均分为 n 份，其中的一份作为 Validation Set，其他的作为训练数据，重复 n 次计算平均表现。

> [!question] Optimization Fails
> Training Loss 无法持续下降有两个原因：local minima 和 saddle point，两者都属于微分为 0 的点，称为 critical point，判定方法：写出 $L(\theta)$ 在 $\theta'$ 附近的泰勒展开： $$ L(\theta) \approx L(\theta') + (\theta - \theta')^{\top}g + \frac{1}{2} (\theta - \theta')^{\top} H (\theta - \theta') $$
> 在 critical point 附近梯度为 0，可以根据二次型的 Hessian 的特征判断具体位置
> - 二次型恒正（Hessian 正定）：当前为 local minima
> - 二次型恒负：当前为 local maxima
> - 二次型 Hessian 不定：saddle point

对于 Saddle point 的情况，Hessian 矩阵一定存在负特征值，对应的特征向量就是函数值可以下降的方向，尽管当前的梯度为 0. 而实际中由于二阶微分计算量大，一般不用该方法。

> [!note] Empirical Study
> 在一些损失函数的一维或二维可视化图中往往会有很多的 local minima 点，但是真实场景中的参数量往往很大，在高维空间的中 error surface 的构造很复杂，往往在某些子维度中会呈现许多的 local minima，但从整体维度来看，真正遇到 local minima 的情况是非常少见的
> ![[image-1.webp|400]]
> 真实训练一个网络到收敛时，其 Hessian 中正特征值的占比(minimum ratio)几乎不为 1，大多在 0.4~0.5 附近，表明收敛往往是处于 saddle point


---
### Batch 与 Momentum
![[image-2.webp]]
虽然 batch 越大，一个 update 内所耗费的时间越多，但是在一定量的 batch 内所耗费的时间相差不大（GPU 并行计算），而一个 epoch 内需要的 updates 更多，两个因素叠加之下，反而 batch 非常小时一个 epoch 所需的总时间越大。


![[image-3.webp]] 
当 batch 增加时，Training 和 Validation 上的准确率同时降低的原因：**优化不足**

大 batch 和小 batch 的对比：
![[image-4.webp]]

---
起始点 $\theta^{0}$，初始移动方向 $m^{0} = 0$，初始梯度信息 $g^{0}$，
- 移动方向 $m^{1} = \lambda m^{0} - \eta g^{0}$
- 新解 $\theta^{1} = \theta^{0} + m^{1}$
- 计算 $g^{1}$，$m^{2} = \lambda m^{1} - \eta g^{1}$
- $\theta^{2} = \theta^{1} + m^{2}$

对每个前进方向展开，可得历史的梯度信息权重按指数衰减
- $m^{0} = 0$
- $m^{1} = - \eta g^{0}$
- $m^{2} = - \lambda \eta g^{0} - \eta g^{1}$

### Learning rate
Loss 不再下降的原因是什么？梯度是否真的为 0？

自适应调整步长的方法：

**Adagrad**：
$$
\theta_{i}^{t+1} \leftarrow \theta_{i}^{t} - \frac{\eta}{\sigma_{i}^{t}} g^{t}_{i}
$$
其中
$$
\sigma_{i}^{t} = \sqrt{ \frac{1}{t+1}\sum_{i=0}^{t}(g_{i}^{t})^{2} }
$$

**RMSProp**：
$$
\sigma_{i}^{t} = \sqrt{ \alpha (\sigma_{i}^{t-1})^{2} + (1 - \alpha)(g_{i}^{t})^{2} }
$$
调整 $\alpha$ 的值，当 $\alpha$ 接近 0 时，当前的 $\sigma$ 受到当前梯度影响更大，对梯度变化的响应更快。


**Adam**：RMSProp + Momentum

#### Learning Rate Scheduling
- Learning Rate Decay：学习率随时间变小
- Warm Up：开始线性增加，之后逐渐减小

### Loss
相较于 MSE，Cross-entropy 更常用与分类问题，`torch` 中 Cross-entropy 和 softmax 绑定

