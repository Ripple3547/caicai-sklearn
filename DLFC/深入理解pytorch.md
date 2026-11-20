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
饱和函数：越偏离中心，梯度值越小，如 logistic 
严格定义为：
> [!info] 
> 若函数满足 $\lim_{x \to +\infty} f'(x) = 0$（右饱和）或 $\lim_{x \to -\infty} f'(x) = 0$（左饱和），则具有饱和性；两端均饱和的函数称为饱和激活函数：
> - 代表函数：Sigmoid、Tanh
> - 主要特点：输出有界、平滑，适合概率映射或门控机制
> - 核心缺陷：输入过大或过小时梯度接近 0，深层网络反向传播易出现梯度消失。因此现代隐藏层更常使用 ReLU 及其变体等非饱和激活函数（可参考详解激活函数了解更系统的对比）。

![[深入理解pytorch-01.webp]]


### 模型定义

```py
import torch.nn.functional as F

class LogisticRegressionModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = torch.nn.Linear(1,1)
		
	def forward(self, x):
		y_pred = F.sigmoid(self.linear(x))
		return y_pred
		
```

Loss 为二项 CrossEntropy 函数
$$
Loss = -(y \log  \hat{y} + (1- y)\log (1 - \hat{y}))
$$
对应模块为 `torch.nn.BCELoss()`

### sigmoid 函数
- `torch.sigmoid`：最推荐的直接函数调用方式。适用于任何需要对张量进行 Sigmoid 计算的场景，用法简洁高效。
- `torch.nn.Sigmoid`：作为神经网络层（nn.Module）使用。通常用于在定义模型结构（如 nn.Sequential）时，将其作为一个激活函数层嵌入其中。
- `torch.nn.functional.sigmoid`：函数式接口。虽然功能与 torch.sigmoid 相同，但在较新的 PyTorch 版本中已被弃用，建议避免使用，直接改用 torch.sigmoid。

函数式和模块式的 sigmoid 都会在 forward 中构建计算图。

## 多特征输入
```py
class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = torch.nn.Linear(8,1)
		self.sigmoid = torch.nn.Sigmoid()
		
	def forward(self, x):
		x = self.sigmoid(self.linear(x))
		return x
		
```


`np.loadtxt` 加载数据时 `dtype=np.float32`，支持显卡。
```py
xy = np.loadtxt("dataset.csv.gz", delimiter=",", dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
```

工程细节：如果顺序式调用一系列的模块，建议只使用一个变量，如 `x`，否则不同模块的调用传参写错调试会比较困难。

单个模块复用的前提是该激活层本身不持有需要独立更新的参数，且前向计算是纯函数式的。比如 `nn.Sigmoid`，`nn.ReLU` 等，带参数的激活如 `nn.PReLU` 则不能复用。


## 数据导入
```py
from torch.utils.data import Dataset, DataLoader
```
`Dataset` 类是 ABC，需要通过继承定义自己的数据集类。

DataLoader 在开启 `num_workers > 0` 时会导致子进程无限递归启动。
Windows 不支持 fork，默认用 spawn。在 Windows/macOS 上用 `spawn` 启动 worker 时，spawn 会启动一个全新的 Python 解释器，把主模块重新 import 一遍，然后才执行子进程的任务。若 `DataLoader` 的创建逻辑写在模块顶层，则子进程在 `import` 主模块时会再次执行创建 `DataLoader` 的代码。于是它又要创建新的 worker 进程……子子孙孙无穷匮也。把会触发多进程的入口逻辑：
- 创建 `DataLoader`
- 启动训练循环
放进保护块 `if __name__ == '__main__':` 中。


如果 `dataset` 的构造本身也在顶层，且构造过程很重（比如扫描大量文件、加载大列表），同样建议把 `dataset = ...` 也放进保护块，避免子进程重复做一次昂贵的初始化。原则很简单：凡是会触发多进程的创建逻辑，以及它依赖的重资源初始化，都放进 `if __name__ == '__main__':` 里面。

```py
class DiabetesDataset(Dataset):
	def __init__(self, filepath):
		xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
		self.len = xy.shape[0]
		self.x_data = torch.from_numpy(xy[:, :-1])
		self.y_data = torch.from_numpy(xy[:, [-1]])
		
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]
		
	def __len__(self):
		return self.len
		
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
```


## Softmax 分类
Softmax 的输出 $\hat{Y}$ 计算损失，损失函数为：
$$
Loss(\hat{Y}, Y) = - Y \log  \hat{Y}
$$
Y 是真实样本标签的 one-hot 表示，target 类为 1，其他类为 0。输出 $\hat{Y}$ 越接近 one-hot 表示，损失越低。

在 pytorch 中，模块 `torch.nn.CrossEntropyLoss` 封装了 Softmax、Loss 和 One-hot 编码的部分。
```py
import torch
y = torch.LongTensor([0])
z = torch.Tensor([[0.2, 0.1, -0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z, y)
```

![[深入理解pytorch-02.webp]]

### 数据准备
使用数据集 MNIST，准备数据的流程如下：
- 导入特定的包
```py
import torch
from torchvision import transforms, datasets
from torch.utils.dat import DataLoader
import torch.nn.functional as F
import torch.optim as optim
```
- 导入并划分数据集
```py
batch_size = 64
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
```
- `transform` 定义了对原始数据的处理方式，具体为先转为 Tensor，后进行标准化
- dataset 定义中 `transform=transform` 指定了数据经过什么处理变为真正可用的数据集。
- loader 让 dataset 可以变成分 batch 训练的形式


### 模型定义
在模型中一共定义了 5 个线性层，相邻线性层之间用 ReLU 激活函数连接，最终的输出维度为 10：
```py
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.l1 = torch.nn.Linear(784, 512)
		self.l2 = torch.nn.Linear(512, 256)
		self.l3 = torch.nn.Linear(256, 128)
		self.l4 = torch.nn.Linear(128, 64)
		self.l5 = torch.nn.Linear(64, 10)
		
	def forward(self, x):
		x = x.view(-1, 784)
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l3(x))
		x = F.relu(self.l4(x))
		return self.l5(x)	
		
```



### 训练和测试 
```py
def train(epoch):
	running_loss = 0.0
	for batch_idx, data in enumerate(train_loader, 0):
		inputs, target = data
		optimizer.zero_grad()
		
		# forward + backward + update
		outputs = model(inputs)
		loss = criterion(outputs, target)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
		if batch_idx % 300 == 299:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
			running_loss = 0.0
			
```
详解：
- `enumerate` 返回的生成器中默认索引元素从 0 开始，第二个参数可以指定起始值。
- `train_loader` 按照每个 batch yield `(input, target)` 因此每个循环中的 `inputs` 是一个 batch 中的所有样本
- `optimizer.zero_grad()` 清零梯度信息（默认累加）
- 求总 loss 时取 `item` 将 Tensor 转为 float，从而防止计算图被挂载导致内存泄漏。



```py
def test():
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			outputs = model(images)
			_, predicted = torch.max(outputs, dim=1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy on test set: %d %%' % (100 * correct / total))

```
- `torch.no_grad()` 可以关闭自动求导，不会记录梯度信息和计算图，从而节省显存，加快计算。
- `outputs` 得到每个类别的得分，`torch.max(...)` 得到的是最大值和每个样本的类别索引，
- `total` 统计的是一个 batch 内的样本数，因此可以处理最后一个 batch 不满的情况。

主程序入口：
```py
if __name__ == '__main__':
	for epoch in range(10):
		train(epoch)
		test()
```


Otto Group Product Classification Challenge
- Dataset: https://www.kaggle.com/c/otto-group-product-classification￾challenge/data

### `torch.no_grad()`
`torch.no_grad()` 是上下文管理器，在 `with` 块内会强制关闭 `Autograd` 对计算图的记录


**安全原则**：只有当一段计算**完全不需要**将梯度回传给其**之前**的任何模块时，才可以用 no_grad 包裹它。典型的安全场景就是**整个推理/评估过程**，或者计算那些纯粹的“指标”（如用固定的预训练网络计算感知损失，此时预训练网络的参数本就不需要更新）。

#### `.detach()`
detach 是张量级的梯度截断器，专治“同一段计算里一部分需要梯度、另一部分不需要”的场景（TD 目标、GAN、蒸馏、stop-grad），也用于日志、缓存、EMA 等防止计算图泄漏的场合。它和 no_grad 的分工是：要精细截断用 detach，要整段省资源用 no_grad，纯粹推理用 inference_mode。




## CNN
输入张量的形状为 `width, height, channel`，若 `channel=n`，则卷积核个数为 `n` ，常见的卷积核尺寸为 `(3,3,n)`。每个卷积核在输入上滑动，计算局部区域和卷积核权重的点积，产生一个二维的 feature map。所有卷积核生成的 feature map 叠加，作为当前卷积层的输出，通道数为卷积核个数 `K`。

各种通道的关系为：
1. 特征图的通道数 = 该层卷积核的个数。每个卷积核独立地在输入上滑动，产生一张二维激活图，K 个卷积核就堆叠成 K 个通道的输出特征图。
2. 单个卷积核的通道数 = 输入特征图的通道数。卷积核不是二维的，而是三维的：形状为 $K_h \times K_w \times C_{in}$，其中 $C_{in}$ 必须与输入的通道数一致，这样才能在通道维度上做完整的加权求和。

因此可以认为卷积层是一个维度为 4 的张量，`(m,n,kernel_size_width, kernel_size_height)`，输出为 feature map，`(m, width_out, height_out)`


输出尺寸的公式为
$$
H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1
$$
- `padding=1` 外层补一层 0，通常可以配合 `stride=1` 来保证输出特征图和输入的尺寸保持一致。
- `stride` 表示步幅，包括上下左右的方向，注意不能只认为是横向上的幅度
