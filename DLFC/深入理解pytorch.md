## `Tensor` 类解析
常用实例化形式为
```py
w = torch.tensor([1.0], requires_grad=True)
```

包含两个重要的成员：`grad` 和 `data`
- `.grad` 只有叶子张量且设置 `requires_grad=True` 时，在 `.backward()` 后会填充。注意梯度值不会自动清零，每次循环需要注意清零。**这是个 Tensor**，**取值还须访问子成员** `.data`
- `data` 是指向底层存储但从计算图中分离的新张量，`autograd` 会基于修改后的值计算梯度。
- `detach` 和 `data` 类似，但是反向传播时遇到被修改的 `detach` 张量会报错。对其生成的新 Tensor 计算不会沿着计算图回传梯度，可用于模型训练、特征提取、强化学习。是更新、更推荐使用的方式。

注意，`backward()` 计算后的梯度会叠加，如果在下一次迭代中不需要用到之前的梯度，需要通过 `w.grad.zero_()` 清除。

`Tensor` 类具有设备管理特性，可以规定使用 CPU 或 GPU，但参与同一运算的所有张量只能在同一个设备上。包括转移设备、查询设备和设置默认设备的接口。

## 动态计算图
Dynamic 特性的体现：对 forward 计算后的 `Tensor` 执行 `backward()` 后会释放**计算图**，并返回 `requires_grad=True` 的 Tensor 的梯度值。

PyTorch 通过**动态计算图**来隐式构建计算图。在执行 Tensor 运算时，会根据计算关系动态记录计算历史。

执行计算时，会创建计算节点，包含在成员 `grad_fn` 中。
- 叶子张量没有 `grad_fn`
- `grad_fn` 是指向该 Tensor 对应的反向计算节点。多个节点可以连成一个计算图。

支持 autograd 的 tensor 运算会自动追踪计算图。


注意，进行 `loss.backward()` 后的 Tensor 携带 `grad_fn`，因此直接进行损失的求和需要访问其 `loss.item()`，否则计算图会一直保留，占用显存。


## 线性模型
### 模型定义
```py
class LinearModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = torch.nn.Linear(1,1)
		
	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred
		
model = LinearModel()
```
由于继承自 `Module` 类，因此 `backward` 方法会在会自动实现，此外该父类还提供类其他必要的工具函数。

`Linear` 类的两个参数分别代表输入 feature 的维度和输出的维度，此外还有一个 `bias` 参数表示是否包含偏置。实例化后的对象可以直接通过 `self.linear(x)` 进行调用（该类实现了 `__call__` 方法，变成可调用对象）

`__call__` 方法中会实现 `forward` 方法，因此示例中实际上进行了对父类 `forward` 方法的重写。


由于 `torch.nn.Module` 实现了可调用对象，因此在继承后对其 `forward` 方法进行重写，子类也可以通过 `model(x)` 的方式直接进行 `forward` 操作。

### 损失函数和优化器
`torch.nn.MSELoss` 继承自 `Module`，用于定义损失函数。

`torch.optim.SGD` 定义优化器（方法为 SGD）参数有：
- `params` 权重，一般为 `model.parameters()`
- `lr` 学习率

### 训练流程
```py
for epoch in range(100):
	y_pred = model(x_data)
	loss = criterion(y_pred, y_data)
	print(epoch, loss)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```

`step` 方法可以自动更新所包括的权重。


## logistic 回归
