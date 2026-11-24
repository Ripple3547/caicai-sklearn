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
这里 `target` 是一维的，为**类别索引模式**（最常用）：`target` 形状比 `input` 少一维（例如 `input` 是 `(N, C)`，`target` 是 `(N,)`）。此时每个值是一个整数，表示该样本属于第几类。官方文档明确规定，这种模式下 `target` **的数据类型必须是** `long` （PyTorch 是预编译的，为了控制库的体积，官方没有为所有整数类型都实现损失函数的内核）。

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
	- 神经网络偏好符合正态分布的数据，原始图像为灰度表示，范围为 $\{ 0, \cdots, 255 \}$，需要转换成图像张量，范围为 $[0, 1]$。
	- 对于多通道的 RGB 图像，一般的尺寸为 `(w, h, c)`，而在 pytorch 中变成 `(c, w, h)` 处理更高效。
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



### CrossEntropyLoss 和 LogSoftMax 以及 NLLLoss 的关系
定义：
- LogSoftMax：把 logits 经过 softmax 后再取 log 后的结果，公式为： $\mathrm{log\_softmax}(z)_{i} = \log \left( \frac{e^{z_{i}}}{\sum_{j} e^{z_{j}}} \right) = z_{i} - \log \sum_{j} e^{z_{j}}$，特点：每个值不为正。
- NLLLoss（Negative Log Likelihood Loss）：按每个样本在正式类别上的对数概率求和，公式为 $\mathrm{NLLLoss} = - \frac{1}{N}\sum_{n=1}^{N} \log p_{n, y_{n}}$ 其中 $y_{n}$ 是第 n 个样本的真实标签索引，因此每一项都是该样本真实标签的对数概率
- CrossEntropyLoss：公式为 $\mathrm{C E} = - \frac{1}{N} \sum_{n=1}^{N} \log(\mathrm{soft max}(z_{n})_{y_{n}}) = - \frac{1}{N}\sum_{n=1}^{N}\left[ z_{n,y_{n}} - \log \sum_{j} e^{z_{n, j}} \right]$ 

因此三者关系为
$$
\mathrm{CrossEntropyLoss} = \mathrm{LogSoftmax} +\mathrm{ NLLLoss}
$$

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


`torch.nn.Conv2d` 调用时的参数顺序为 $(N, C_{\text{in}}, H, W)$ 
- $N$ 是 batch size
- $C_{in}$ 表示输入的通道数

详细的类构造声明为
```py
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)[source]
```


### 下采样
通常的下采样方法是 MaxPooling，每个区域内取最大值得到输出。大小为 `(2,2)` 的层，通常步长会设为 2，因此输出的大小会减半。

### 模型设计
原始输入经过若干个 Conv2d 层、ReLU 层、Pooling 层后把最终的张量展平为一维的向量，在经过一个线性层，让输出维度为和分类数对齐，从而通过交叉熵进行训练。


### 更换计算设备
更换计算设备需要对模型和张量都进行移动操作：
```py
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

inputs 和 targets 迁移
```py
inputs, target = inputs.to(device), target.to(device)
```


## CNN 进阶
### Google Net
#### 1×1 卷积
主要实现的是在不改变图像尺寸情况下，实现通道数的改变，从而减少计算压力。

1×1 卷积层每个卷积尺寸为 `(C_in, 1, 1)`，总共的卷积核个数为 $N$，因此输出结果的通道数为 $N$。经过1×1 卷积层，图像的尺寸不变，而通道数从 `C_in` 变为 $N$，同时由于卷积核为 `(1, 1)` 的大小，输出中每个位置都只包含输入中**相同位置**的信息，即**信息融合**

#### Inception
每个 Inception 由若干个分支构成
##### 1
第一个分支先后经过一个 `Average Pooling` 和 `1x1 Conv`。
- `Average Pooling`：`kernel_size=3, padding=1, stride=1`，因此输入和输出的大小一致。没有权重，因此可以复用。
- `1x1 Conv`：参数为输入通道数，输出通道数为 `24`。

```py
# __init__
self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

# forward
branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
branch_pool = self.branch_pool(branch_pool)
```

##### 2
由单独的 `1x1 Conv` 构成。输出通道数为 `16`。


##### 3
由两个卷积层构成：
1. `1x1 Conv`，输出通道数为 `16`。
2. `5x5 Conv`，输出通道数为 `24`。注意 `padding=2` 从而保持尺寸不变。

##### 4
由 3 个卷积层构成：
1. `1x1 Conv` 输出通道为 `16`。
2. `3x3 Conv` 输出通道为 `24`.
3. `3x3 Conv` 输出通道为 `24`。


##### Concatenate
将 4 个分支的张量拼接（沿着通道维度依次叠加）
```py
outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
return torch.cat(outputs, dim=1)
```

据此，以下是单个 Inception 模块的定义
```py
class InceptionA(nn.Module):
	def __init__(self, in_channels):
		super(InceptionA, self).__init__()
		self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
		
		self.branch5x5_1 = nn.Conv2d(in_channels,16, kernel_size=1)
		self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
		
		self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
		self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
		self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
		
		self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
	def forward(self, x):
		branch1x1 = self.branch1x1(x)
		
		branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)
		
		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)
		branch3x3 = self.branch3x3_3(branch3x3)
		
		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)
		
		outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
		return torch.cat(outputs, dim=1)
```


---
以下网络引入了两个 Inception 模块：
```py
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
		
		self.incep1 = InceptionA(in_channels=10)
		self.incep2 = InceptionA(in_channels=20)
	
		self.mp = nn.MaxPool2d(2)
		self.fc = nn.Linear(1408, 10)
	
	def forward(self, x):
		in_size = x.size(0)
		x = F.relu(self.mp(self.conv1(x)))
		x = self.incep1(x)
		x = F.relu(self.mp(self.conv2(x)))
		x = self.incep2(x)
		x = x.view(in_size, -1)
		x = self.fc(x)
		return x
```

> [!note] 线性层维度
> 上述的模块之间只显式规定了通道数的变化，而没有明确描述图像尺寸，而最后的线性层中需要指定输入维度的大小。可以通过随机初始化输入跑一遍不带线性层的网络，看最终的输出维度（或者 error 中的提示）可以确定线性层的输入维度。

以上的网络结构可能比较简单，引入更多的 Inception 模块可能达到更好的效果。


### ResNet
ResNet 可以解决梯度消失的问题，输入经过权重层、激活层、权重层后得到 $F(x)$，再加上输入，即 $H(x) = F(x) + x$，再经过激活层得到输出。从输入 $x$ 到输出之间的连接也称为 **shortcut 分支**（捷径分支）。

由于涉及 $F(x)$ 和 $x$ 两个张量的相加，因此需要确保维度的一致（通道数、长和宽）。卷积层可能包含**下采样**的操作，通道数和空间都改变，因此在捷径分支中可通过 `1x1 Conv` 操作调整 $x$ 的维度（卷积核数量控制通道数、stride 控制形状）
```py
class ResidualBlock(nn.Module):
	def __init__(self, channels):
		super(ResidualBlock, self).__init__()
		self.channels = channels
		self.conv1 = nn.Conv2d(channels, channels, 		kernel_size=3, padding=1)
		
		self.conv2 = nn.Conv2d(channels, channels, 		kernel_size=3, padding=1)
	def forward(self, x):
		y = F.relu(self.conv1(x))
		y = self.conv2(y)
		return F.relu(x + y)
```

---
![[深入理解pytorch-03.webp]]

```py
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
		self.mp = nn.MaxPool2d(2)
		self.rblock1 = ResidualBlock(16)
		self.rblock2 = ResidualBlock(32)
		self.fc = nn.Linear(512, 10)
	def forward(self, x):
		in_size = x.size(0)
		x = self.mp(F.relu(self.conv1(x)))
		x = self.rblock1(x)
		x = self.mp(F.relu(self.conv2(x)))
		x = self.rblock2(x)
		x = x.view(in_size, -1)
		x = self.fc(x)
		return x
```

不同的 Residual 设计：https://arxiv.org/abs/1603.05027
Dense Net：https://arxiv.org/abs/1608.06993


---
### 后续学习
1. 理论学习：《深度学习》相关理论性强的书
2. 阅读 Pytorch 文档：通读 api, reference 等，增强复现能力
3. 复现经典工作：读代码、写代码，下载后跑通不等于复现。
4. 选择领域，扩充视野，积累经验，掌握套路



## 循环神经网络
CNN 中通过卷积核的移动得到输出，权重得到复用；而全连接网络中两个线性层之间的每个单元都包含权重，因此包含的参数会更多。

RNN 专门处理序列类型的数据，并且也实现了权重共享。每个 RNN Cell 的输出和新的输入会再次进入 RNN Cell 不断循环，在序列计算中权重共享。因此实际上一个 RNN Cell 被反复用于计算。

输入为：
- `hidden`：初始化的 $h_{0}$，代表先验
- 输入序列 $\{ x_{t} \}$ 代表按时间变化的输入

### 循环细节
每一个时间步，在一个 RNN Cell 内，$x_{t}$ 和 hidden $h_{t-1}$ 分别经过线性层变换后相加，经过激活函数（一般是 $\tanh$）后得到 $h_{t}$，即
$$
h_{t} = \tanh (W_{hh}h_{t-1} + b_{hh} + W_{ih}x_{t} + b_{ih})
$$
矩阵运算等价于
$$
W_{1} h + W_{2}x = \begin{bmatrix}
W_{1}  & W_{2}
\end{bmatrix} \begin{bmatrix}
h \\
x
\end{bmatrix}
$$
### Pytorch 中的两种实现方式
#### RNN Cell
```py
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

hidden = cell(input, hidden)
```
![[深入理解pytorch-04.webp]]
维度关系为：
- `input.shape=(batch, input_size)`
- `hidden.shape=(batch, hidden_size)`
- 数据集的维度应为：`dataset.shape=(seqLen, batch, input_size)`

示例过程：
```py
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):
	hidden = cell(input, hidden)
```

以上的方式通过定义单个 RNN Cell，再手动实现循环来构造网络。

#### RNN
```py
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

out, hidden = cell(inputs, hidden)
```
 
多层的 RNN 在计算时可以看作一个二维的计算网络， $t$ 时刻某层的 hidden $h_{t}^{(l)}$ 接受的输入为：
- 当前时刻的前一个 hidden $h_{t}^{(l-1)}$ 
- 前一时刻的同层 hidden $h_{t-1}^{(l)}$ 

$$
h_t^{(l)} = \tanh\left(W_{ih}^{(l)}\, a_t^{(l)} + W_{hh}^{(l)}\, h_{t-1}^{(l)} + b^{(l)}\right)
$$
其中：
- 若 $l = 0$，则 $a_{t}^{(0)} =x_{t}$ ，且形状为 `(batch, input_size)`
- 若 $l \geqslant 1$，则 $a_{t}^{(l)} = h_{t}^{(l-1)}$，且形状为 `(batch, hidden_size)`
- 同时 $W_{ih}^{(l)}$，形状为 `(hidden_size, in_l)`， `in_0 = input_size, in_l>=1 = hidden_size`

输出中
- `out` 为 $h_{1}, \cdots, h_{N}$（**最后一层**的输出序列）即值包含最顶层的输出，不包含中间层。维度 `out.shape=(seqSize, batch, input_size)`
- `hidden` 为 $h_{N}$（**每层**在最后时间步的 hidden）维度 `hidden.shape=(numLayers, batch, hidden_size)`

因此有固定关系：
- 单层 RNN 时：`output[-1] == h_n[0]`
- 多层 RNN 时：`output[-1] == h_n[-1]`

以上情况都不考虑双向 RNN。

##### `batch_first`
设为 `True` 时，维度会设为先 `batch` 后 `seqLen` 即：`(batch, seqLen, input_size)`

### 案例
```py
import torch
input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]

# (seqLen, batch, input_size)
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# (seqLen, 1)
labels = torch.LongTensor(y_data).view(-1, 1)
```


#### 使用 RNN Cell
```py
class Model(torch.nn.Module):
	def __init__(self, input_size, hidden_size, batch_size):
		super(Model, self).__init
		self.batch_size = batch_size
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
		
	def forward(self, input, hidden):
		hidden = self.rnncell(input, hidden)
		return hidden
	
	def init_hidden(self):
		return torch.zeros(self.batch_size, self.hidden_size)

net = Model(input_size, hidden_size, batch_size)
```


##### 训练
```py
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
```


```py
for epoch in range(15):
	loss = 0
	optimizer.zero_grad()
	hidden = net.init_hidden()
	print('Predicted string: ', end='')
	for input, label in zip(inputs, labels):
		hidden = net(input, hidden)
		loss += criterion(hidden, label)
		_, idx = hidden.max(dim=1)
		print(idx2char[idx.item()], end='')
	loss.backward()
	optimizer.step()
	print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))
```

#### 使用 RNN
```py
class Model(torch.nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
		super(Model, self).__init__()
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, 		num_layers=num_layers)
		
	def forward(self, input):
		hidden = torch.zeros(self.num_layers,
							 self.batch_size,
							 self.hidden_size)
		out, _ = self.rnn(input, hidden)
		# (seqLen × batch, hidden_size)
		return out.view(-1, self.hidden_size)

net = Model(input_size, hidden_size, batch_size, num_layers)
```

##### 数据改变
```py
input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
one_hot_lookup = [[1, 0, 0, 0],
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

# (seqLen, batch, hidden_size)
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
# (seqLen x batch, 1) 
labels = torch.LongTensor(y_data)
```

##### 训练
```py
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
	optimizer.zero_grad()
	outputs = net(inputs)
	loss = criterion(outputs, labels)
	loss.backward()
	optimizer.step()
	
	_, idx = outputs.max(dim=1)
	idx = idx.data.numpy()
	print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
	print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
```

#### embedding
one-hot 编码的缺点：
- 维度高，需要为词表中每个词分配一个维度
- 每个向量分布在坐标轴上，空间稀疏
- 硬编码

EMBEDDING 表示：
- 低维
- 稠密
- 可学习

通过 embedding 可以将高维的 one-hot 向量映射到低维的 embedding 向量表示。RNN 中可在最后一层添加一个线性层来调整维度。

```py
class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
```
- `num_embeddings`：输入向量的维度
- `embedding_dim`：embedding 向量的维度。

在输出中，形状为 `(*, embedding_dim)`（ `*` 的内容可以是 `seqLen x batch`），在输入后面增加一个维度用于 embedding 表示.

同理，`torch.nn.Linear` 的输入和输出关系为 `(N, *, in_features)` 和 `(N, *, out_features)`

##### `CrossEntropyLoss` 输入参数解析
常见情况为 2D 输入：
- `input.shape=(N,C)`：N 个样本，一共 C 类候选
- `target.shape=N`：每个样本一个整数的类别索引


高维情况：
- `input.shape=(N, C, d1, d2, ...)`：类别所在维度为 dim=1
- `target.shape=(N, d1, d2, ...)`

对 `(d1, d2, ...)` 独立计算，然后对整个 batch 和所有 `(d1, d2, ...)` 元素求平均计算交叉熵：
- 语义分割：`logits: [2, 5, 8, 8]` 输入 2 张图像，类别数为 5，图像大小 8x8。计算交叉熵时，对该批次样本所有像素求平均。
- 词性标注：`RNN输出: [N, L, C]`，`标签: [N, L]`，共 L 个时间步，每步 C 类。需要转置让输出符合输入格式要求：
```py
# [N, L, C]
output = rnn(x)
# [N, C, L] vs [N, L]
loss = nn.CrossEntropyLoss()(output.transpose(1, 2), tags)
```

添加了 embedding 和 fc 的模型：
```py
# parameters
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.emb = torch.nn.Embedding(input_size, embedding_size)
		self.rnn = torch.nn.RNN(input_size=embedding_size, 		hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.fc = torch.nn.Linear(hidden_size, num_class)
		
	def forward(self, x):
		hidden = torch.zeros(num_layers, x.size(0), hidden_size)
		x = self.emb(x) # (batch, seqLen, embeddingSize)
		x, _ = self.rnn(x, hidden)
		x = self.fc(x)
		return x.view(-1, num_class)
```


---
### LSTM
LSTM 比 RNN 效果更好，但时间复杂度高。
[[04 To Learn More#Long Short-term Memory（LSTM）]]


### GRU
GRU 是 LSTM 和 RNN 的折中
![[深入理解pytorch-05.webp]]


## RNN 进阶
### 导包
```py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gzip
import csv
import time
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
```

### 构建数据集
```py
class NameDataset(Dataset):
    def __init__(self,is_train_set=True): #is_train_set 指定是否是训练集
        filename = 'names_train.csv.gz' if is_train_set else 'names_test.csv.gz'
        with gzip.open(filename,'rt') as f:#读取数据集
            reader = csv.reader(f)
            rows = list(reader)#读取每行数据，最终返回的结果为元组(name,country)形式
        self.names = [row[0] for row in rows]#从rows中去除第0个元素，也就是name
        self.len = len(self.names)#记录一下样本的数量
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))#同样，获取country信息，将来作为names的标签进行使用
        """
        set(self.countries) #将列表变成一个集合，因为集合里面不重复，也就是去重
        sorted(set(self.countries)) #排序
        list(sorted(set(self.countries))) #最后变成一个列表形式
        """
        
        self.country_dict = self.getCounrtyDict()#此时的列表中仍是英文名字，需要通过词典转换为序列索引
        self.country_num = len(self.country_list)
    
    
    def __getitem__(self,index):
        return self.names[index],self.country_dict[self.countries[index]]
        """
        self.names[index] #拿到姓名
        self.countries[index] #拿到国籍
        self.country_dict[self.countries[index]] #根据国籍，从字典中获取对应的索引
        countries(key)和index(value) 相当于字典中的键值对
        """    
    
  
    def __len__(self):
        return self.len
    
    
    def getCounrtyDict(self):
        country_dict = dict()#定义一个空字典
        for idx,country_name in enumerate(self.country_list,0):#遍历
            country_dict[country_name] = idx#构造键值对
        return country_dict#返回对应的字典
    
    
    def idx2country(self,index):#根据索引，返回国籍的字符串
        return self.country_list[index]#例如输入的是名字，得到分类是9，根据9，来返回对应的字符串，即所属的国籍
    
    def getCountriesNum(self):#看下到底有多少个国籍
        return self.country_num#返回国籍数量

```

### 数据准备
```py
hidden_size = 100#隐藏层是100维度
batch_size = 256#每批256个姓名样本
n_layer = 2#GRU使用了2层
n_epochs = 100#训练100次
n_chars = 128#因为ASCII共128个，故使用张量128来进行表示输入样本
use_gpu = False#不使用GPU

trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset,batch_size=batch_size,shuffle=False)

n_country = trainset.getCountriesNum()#决定模型最终的输出维度大小
```


### 模型设计
```py
class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=True):
        """
        GRU使用：hidden_size、n_layers、bidirectional
        embedd嵌入层使用：input_sizeinput_size、hidden_size
        
        """
        super(RNNClassifier,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1#RNN是单向还是双向，单向是1，双向是2
        
        self.embedding = torch.nn.Embedding(input_size,hidden_size)
        """
        embedding层
        输入维度：(seqLen,batchsize)  
        输出维度：(seqLen,batchsize,hiddensize)
       
        """
        self.gru = torch.nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)#bidirectional单向还是双向循环神经网络
        """
        GRU
        输入维度：(seqLen,batchsize,hiddensize)
        输出维度：(seqLen,batchsize,hiddensize * nDirections)
        
        """
        
        self.fc = torch.nn.Linear(hidden_size*self.n_directions,output_size)
        
    def _init_hidden(self,batch_size):#创建全零的初始的隐藏层
        hidden = torch.zeros(self.n_layers * self.n_directions,batch_size,self.hidden_size)
        return create_tensor(hidden)
    
    def forward(self,input,seq_lengths):
        input = input.t()#矩阵的转置，transpose，将原来的(batch×seqlen)转换成(seqlen×batch)
        batch_size = input.size(1)#保留batchsize
        
        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)#embedding维度：(seqLen,batchsize,hiddensize)
        
        gru_input = pack_padded_sequence(embedding,seq_lengths)#gru_input维度为：(seqLen,batchsize,hiddensize)
        
        output,hidden = self.gru(gru_input,hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1],hidden[-2]],dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output
```

### 数据处理：姓名编码、索引和长度的处理
```py
def name2list(name):#将每个姓名都转化为欸一个列表
    arr = [ord(c) for c in name]
    return arr,len(arr)#返回一个元组，一个是列表本身，一个是列表长度

def create_tensor(tensor):
    if use_gpu:
        device = torch.device('cuda:0')
        tensor = tensor.to(device)
        
    return tensor

def make_tensors(names,countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()
    
    seq_tensor = torch.zeros(len(name_sequences),seq_lengths.max()).long()#全零
    for idx,(seq,seq_len) in enumerate(zip(name_sequences,seq_lengths),0):#把值贴上去
        seq_tensor[idx,:seq_len] = torch.LongTensor(seq)
    
    seq_lengths, perm_idx = seq_lengths.sort(dim=0,descending=True)#按照序列的长度进行排序
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
    
    return create_tensor(seq_tensor),create_tensor(seq_lengths),create_tensor(countries)
```

### 训练函数
```py
def trainModel():
    total_loss = 0
    for i,(names,countries) in enumerate(trainloader,1):
        inputs,seq_lengths,target = make_tensors(names,countries)
        output = classifier(inputs,seq_lengths)
        loss = lossf(output,target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
        if i%10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}',end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]',end='')
            print(f'loss={total_loss / (i*len(inputs))}')
    return total_loss
```

### 测试函数
```py
def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i,(names,countries) in enumerate(testloader,1):
            inputs,seq_lengths,target = make_tensors(names,countries)
            output = classifier(inputs,seq_lengths)
            pred = output.max(dim=1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        percent = '%.2f'%(100*correct/total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct/total
```

### 计时函数
```py
def time_since(since):
    s = time.time() - since
    # time.time() 返回的是当前时间，秒为单位
    # s / 60 变成分钟
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds"%(m,s)
```

### 主函数
```py
if __name__ == '__main__':
    classifier = RNNClassifier(n_chars,hidden_size,n_country,n_layer) #创建分类器
    """
    n_chars 字符数量，输入的是英文字母，需要转换为one-hot向量，也就是整个字母表一共有多少个元素
    hidden_size 隐藏层数量，GRU将来输出的隐层的维度
    n_country 一共有多少个分类，也就是一共有多少个国家
    n_layer 使用几层GRU
    """
    if use_gpu: #是否使用GPU
        device = torch.device('cuda:0')
        classifier.to(device)
    
    lossf = torch.nn.CrossEntropyLoss() #分类问题，使用交叉熵损失函数就够了
    optim = torch.optim.Adam(classifier.parameters(),lr=0.00001) #优化器使用adam
    
    start = time.time() #计时，看看训练的时间有多长，距离开始训练已经过去了多少时间
    print("Training for %d epochs..." %n_epochs)
    acc_list = []
    for epoch in range(1,n_epochs+1):
        trainModel() #训练和测试进行封装到两个函数中
        acc = testModel()
        acc_list.append(acc) #测试的结果添加到列表中，方便后续的绘图操作
```


### 要点
- 序列长度补齐：按照最长的序列对每个样本长度补齐，用 0 填充。代码中的做法是先生成一个全 0 的张量，再把样本填入对应位置。代码中对于排序的做法
- `pack_padded_sequence`：用于加速计算，输入序列长短不一，末尾补 0 实际上可以简化计算，需要两个信息：嵌入向量序列 `embedding` 和 `seq_lengths` 输入序列**真实**长度（即未填充 0 的长度，填充 0 后做 embedding 长度一致），该函数会依次做以下的工作：
	1. 先按 `seq_lengths` 从长到短重排该 batch 的数据，并记录和原始序列的对应关系
	2. 按时间步压缩有效数据：每个时间步只取前 `batch_sizes[t]` 个样本对应位置。若用 B, T, D 分别表示 batch 大小、样本最长长度和嵌入维度，则维度变化为 `(B, T, D) -> (sum(seq_lengths), D)`。最终 `batch_sizes` 中记录了每个时间步中还有多少个样本还在算，也就是 `batch_sizes[t]` 表示 batch 中长度不超过 t 的样本个数。
	3. 初始化若干 `hidden[i]`，数量等于样本数。每个时间步按照 `batch_sizes[t]` 从**压缩后 tensor** 中取出对应区间内样本更新对应的 `hidden[i]`。因此每个 `hidden` 的更新次数严格等于样本的序列长度，填充 0 不会参与计算。
	- 返回 `PackedSequence`，可以作为 gru 的输入。
- 真正用作下一层输入的是 `hidden`，若为双向网络还须拼接两个 hidden 张量。


> [!note] `batch_sizes`
> 若嵌入向量序列为：
> ```text
>         t0   t1   t2   t3   t4
Lin   [  4,   3,   5,   0,   0]
Zhang [ 12,  11,   1,   5,   6]
Sato  [  9,   1,  10,   7,   0]
Hu    [ 11,  14,   0,   0,   0]
> ```
> 按照长度重排：
> ```text
> 排序后位置:  0        1       2       3
名字:       Zhang    Sato    Lin     Hu
长度:        5        4       3       2
> ```
> 压缩过程：
> ```text
> t=0: 4 个全在（长度都 ≥1）
t=1: 4 个全在（长度都 ≥2）
t=2: 3 个（Zhang,Sato,Lin；Hu 长 2 已结束）
t=3: 2 个（Zhang,Sato；Lin 长 3 已结束）
t=4: 1 个（仅 Zhang）
> ```
> 可得 `batch_sizes = [4, 4, 3, 2, 1]`

- 双向 RNN：分别前向和后向对序列进行一次计算有两批 output 输出，把前向和后向对应位置的 $h_{i}^{\{ b,f \}}$ 拼接得到 $h_{i}$ 作为 output。前向和后向分别得到的最终结果作为 $hidden = [h_{N}^{f}, h_{N}^{b}]$.
![[深入理解pytorch-06.webp]]


### 拓展：古诗生成器
#### 一、数据准备与处理
**1. 收集语料**
获取古诗数据集（如《全唐诗》、宋词语料），每首诗通常包含标题、作者、正文。清洗时去掉注释、异体字、标点不统一的内容，保留正文。可以按诗体（五言绝句、七言律诗）分别建库，也可以混合。

**2. 统一格式**
把每首诗转成一个字符序列。常见做法是：
- 正文按字符切分，句间用逗号、句号或换行符分隔，诗与诗之间插入特殊分隔符（如 `<eos>` 表示一首结束）
- 如果需要控制诗体，可以在开头加条件标记（如 `<5言>`、`<7言>`）
- 统一长度上限，过长截断、过短丢弃或单独处理

**3. 建立词表**
统计所有出现的字符（汉字 + 标点 + 特殊符号），按频率排序，建立"字符 → 整数索引"的映射。预留 `<pad>`、`<bos>`（开始）、`<eos>`（结束）、`<unk>`（未登录字）等特殊 token。词表通常几千到一万量级。

**4. 构造训练样本**
语言建模的标准做法是"用前 n 个字符预测第 n+1 个字符"：
- 把整首诗的索引序列切成输入 `x = [c0, c1, ..., c_{T-1}]` 和目标 `y = [c1, c2, ..., c_T]`，即目标整体右移一位
- 按固定窗口长度（如 64 或 128）滑动切段，得到大量定长或变长的训练片段
- 如果按批次训练，用 `pad_sequence` 补齐变长序列，同时记录真实长度，供 `pack_padded_sequence` 使用

**5. 构建 DataLoader**
自定义 `collate_fn`：把一批序列补齐、转成张量、记录长度、按长度降序或交给 `pack_padded_sequence` 自动排序。训练时打乱，验证集不打乱。

#### 二、模型设计
**1. 整体结构**
典型结构：`Embedding → GRU（可多层）→ Linear（映射回词表大小）`。

- **Embedding**：字符索引 → 稠密向量，维度一般 128~512
- **GRU**：1~2 层，隐藏维度 256~1024，`batch_first=True`
- **输出层**：`Linear(H, V)`，对每个时间步输出词表上每个字符的分数（logits）
- 可选：输出层与 Embedding 共享权重（weight tying），减少参数、提升效果
- 可选：在 GRU 前加 dropout 防过拟合

**2. 关键取舍**
- 单向还是双向：生成任务必须是单向（不能看到未来），所以只用单向 GRU
- 层数与维度：古诗语料不大，1~2 层、256~512 维通常够用，过大容易过拟合
- 是否用 pack：变长批次用 pack 训练更高效，且 `h_n` 语义正确；但语言建模常用定长切段，也可以不 pack

#### 三、训练过程
**1. 损失函数**
逐字符的交叉熵（`CrossEntropyLoss`）。把输出 reshape 成 `(B*T, V)`、目标 reshape 成 `(B*T,)` 后计算。如果用了 padding，需要设 `ignore_index=<pad>` 让填充位不参与损失。

**2. 优化器与学习率**
Adam 或 AdamW，学习率 1e-3 量级，配合学习率衰减或 warmup。梯度裁剪（clip norm 1.0 左右）对 RNN 很重要，防止梯度爆炸。

**3. 训练循环**
每个 epoch：
- 前向：输入 `x` → Embedding → GRU → Linear → logits
- 算 loss（mask 掉 padding）
- 反向传播、梯度裁剪、更新参数
- 记录训练 loss 和困惑度（perplexity）

**4. 验证与保存**
- 每个 epoch 在验证集上算 loss 和 perplexity
- 定期用当前模型生成几首诗，人工检查质量（loss 低不代表诗好）
- 保存验证 loss 最低或生成质量最好的 checkpoint
- 监控过拟合：训练 loss 持续降、验证 loss 回升时早停

**5. 常见问题**
- 梯度爆炸：梯度裁剪 + 合适学习率
- 过拟合：dropout、减小模型、数据增强（按诗体筛选）
- 模式崩塌（总生成同一首）：提高温度采样、增加数据多样性
- 训练慢：pack 变长序列、减小窗口长度、混合精度

#### 四、训练好之后如何应用
**1. 加载模型**
恢复 checkpoint 的参数和词表映射（词表必须和训练时完全一致，否则索引错位）。

**2. 生成方式：自回归逐字生成**
- **给定开头（续写）**：把用户输入的前几个字作为初始序列，送进模型，取最后一个时间步的输出，采样下一个字，把它拼回序列，再送进模型，循环直到遇到 `<eos>` 或达到长度上限
- **无条件生成**：从 `<bos>` 开始，逐步采样
- **控制格式**：如果训练时加了条件标记，生成时把对应标记放在开头，可控制五言/七言

**3. 采样策略**
- **贪心**：每步取概率最大的字，结果确定但容易重复、呆板
- **随机采样**：按概率分布采样，多样性好但可能不通顺
- **温度采样**：logits 除以温度 T，T<1 更保守、T>1 更随机，常用 0.7~1.0
- **Top-k 采样**：只在概率最高的 k 个字里采样，平衡质量和多样性
- **Top-p（nucleus）采样**：在累积概率达到 p 的最小集合里采样，自适应控制候选集
- **束搜索**：保留多条候选路径，适合追求最优输出，但多样性差

**4. 后处理**
- 按标点或字数切句，检查是否符合诗体（每句字数、句数）
- 过滤含 `<unk>`、重复字过多的结果
- 可加规则约束（如押韵检查）做二次筛选或重采样

**5. 部署形态**
- 命令行/脚本：输入开头，输出续写
- Web 服务：模型常驻内存，接口接收 prompt 返回生成结果
- 批量生成：给定主题词或格式，批量产出候选诗供筛选

#### 五、模型扩展方向
**1. 结构层面**
- **多层 GRU / LSTM**：增强表达能力，但收益递减且更慢
- **双向结构**：生成任务不能用，但可用于"给上句填下句"之外的判别任务
- **注意力机制**：在解码时对输入做 attention，适合"给定主题/关键词生成"这类条件生成
- **Transformer / GPT 式结构**：用自注意力替代 RNN，训练更并行、长程依赖更好，是当前主流替代方案
- **变分自编码器（VAE）**：引入隐变量，提升生成多样性
- **GAN 框架**：用判别器引导生成质量，但训练不稳定

**2. 条件控制**
- **格式条件**：五言/七言、绝句/律诗、平仄格律作为条件输入
- **主题/关键词条件**：给定"明月""思乡"等关键词，生成相关诗
- **风格条件**：按诗人（李白/杜甫风格）或朝代做条件
- **押韵约束**：在采样时对韵脚位置限制候选字集合

**3. 训练策略**
- **预训练 + 微调**：先在大规模古诗语料上预训练，再在特定诗体或风格上微调
- **课程学习**：先短诗后长诗，先简单后复杂
- **强化学习**：用格律合规性、押韵、通顺度做奖励，用策略梯度优化生成质量
- **数据增强**：按诗体筛选、同义替换、回译（对古诗较难）

**4. 评估与工程**
- **自动指标**：困惑度、押韵合规率、格律合规率、重复率、与训练集的 n-gram 重合度（检测抄袭）
- **人工评估**：通顺度、意境、是否符合主题
- **推理加速**：KV cache（对 RNN 意义有限，对 Transformer 关键）、量化、蒸馏、批量化生成
- **可控解码**：约束解码（constrained decoding），在每步屏蔽不合规的字

**5. 任务扩展**
- 从"续写"扩展到"给定标题生成全诗""给定上句对下句""诗转词""根据意境描述生成诗"
- 多模态：给图片生成配诗
- 交互式创作：人机协作，用户逐句修改、模型补全

#### 六、一句话总结流程
**收集并清洗古诗 → 字符级分词建词表 → 构造"前文预测后一字"的样本 → Embedding + GRU + Linear 建模 → 交叉熵 + 梯度裁剪训练 → 验证集选模型 → 自回归 + 温度/Top-k 采样生成 → 按诗体和押韵后处理 → 通过加条件、换结构、强化学习等方式扩展。**
