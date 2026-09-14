**图像识别任务**：输入的图像固定大小，包含 3 通道（channel），展平为一维向量后和第一次的 Neuron 做全连接会出现非常多的参数，会导致 Overfitting 等情况的发生。

实际上每个 Neuron 不需要连接全部输入，只需要关注一个区域。每个 Neuron 关注的这个范围称作 **Receptive field**，包含特定的大小以及通道数，不同的 Receptive field 之间可以重叠。让每个 Neuron 关注一个 Receptive field 的作用就是让模型可以捕捉图像中物体的 patterns，相同目标的不同样本往往有同样的 patterns，可能位置不同。

每个 Neuron 关注一个 Receptive field，但一个 Receptive field 可以被**多个** Neuron 关注。

经典的 Receptive field 的设计是包含全部的通道，关注图像的高和宽称为 **Kernel size**，一般为 3⨉3 的大小。每个 Receptive field 通过平移可以移动到上下左右下一个相邻的 Receptive field，移动的参数称为 **Stride**，一般为 2，有一定的重叠。

**Padding**：如果当前的 Receptive field 中有超出图像的部分，对该部分进行补 0 处理（或其他的填充方法）。


前面提到同一个 pattern 可能出现在图像的不同位置，而一个 Neuron 连接一个 Receptive field，若其中有覆盖该 pattern 出现的位置，那么该 Neuron 就能负责识别该 pattern，因此出现位置不同的同一 pattern 可能会被多个 Neuron 捕捉，这些 Neuron 之间的参数不是共享的，却承担相同的工作，可能会引起参数量的增多。因此工作相同的 Neuron 之间可以共享参数，对其对应的 Receptive field 上相同位置的权重相同。


共享策略：两个 Receptive field 各自对应的所有 Neuron 共享参数。一组被共享的参数称为 filter.
![[image-5.webp]]
Receptive field 和参数共享叠加称为 Convolutional Layer。对应的网络称为 Convolutional Neural Network.


---
### 第二种思路
图像经过一个 Convolutional Layer 操作，其中包含若干个 Filter，每个 Filter 的尺寸为 3⨉3⨉Channel（对于彩图为 3，对于黑白图为 1），每个 Filter 是一个 Tensor。卷积后的图像称为一个 Feature map，包含多个 Channel，数量等于 Filter 的个数。

在下一层的 Convolutional Layer 中，每个 Filter 的 Channel 数和上一层处理后的图片的 Channel 数相同
![[image-8.webp]]


对于每个 Filter，在图像上移动，步长为 Stride，




![[image-7.webp]]
为了简便起见，假定每个 Receptive field 的尺寸为 `1⨉Kernel_size`（即不考虑长度）。对于当前 Feature map 上的 Receptive field $r_{n}$，经过下一层的 Convolution Filter $n+1$ 的卷积操作，卷积核 $k_{n+1} = k, s_{n+1} = s$，因此下一层的每个卷积核会包含当前层的 $k$ 个 Receptive field，则重叠区域有 $k-1$ 个，重叠区域的大小为当前层的 Receptive field $r_{n}$ 减去总共积累的 Stride $\prod_{i=1} ^{n} s_{i}$


### Pooling--Max Pooling
Pooling 操作由一个固定形状的窗口参与，其在输入图像上滑动遍历每个位置，根据 Pooling 类型确定每个位置处窗口的输出值，通常有取 `max` 或 `mean` 两种。

通过 Pooling 后，图像的 Channel 不变，而尺寸缩小，因此 Pooling 操作后模型可能丢失对微小细节的捕捉。


在经过若干次 Colvolution-Pooling 后，输出结果 Flatten 连接全连接网络，最后可以做 Softmax 得到结果
![[image-9.webp]]


CNN 在下棋中的应用
- 有小规模的固定范式（small patterns）
- 出现的位置不固定
- 不建议做 Pooling，会导致细节的丢失

---
CNN 无法处理旋转和拉伸后的图形，除非训练数据中有相关的样本。对输入图像或 Feature map 使用 Spatial Transformer Layer 进行变换。

实际上两个 Layer 之间元素的关系可以描述为新 Layer 中的元素通过原始 Layer 所有元素的线性组合得到：
$$
a_{nm}^{l} = \sum_{i=1}^{N} \sum_{j=1}^{N} w_{nm, ij}^{l} a_{ij}^{l-1}
$$

而对于仿射变换，有更简洁的方式
$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix} \begin{bmatrix}
x \\
y
\end{bmatrix} + \begin{bmatrix}
e \\
f
\end{bmatrix}
$$
通过 6 个参数就能描述一个旋转/平移/缩放操作。

![[image-10.webp]]


从直观来看，仿射变换的参数应该从前一层映射到下一层得到，但是为了防止下一层中有部分元素没有前一层输入对应，更好的做法是通过下一层元素反推。即确立从下一层到前一层的映射关系。



计算得到的前一层坐标并不一定是整数，可以进行四舍五入处理，但此时又引入了一个问题：参数的微小变化可能不会影响四舍五入的结果，此时梯度下降方法就失效了。实际上更好的方法是通过双线性插值：
若下一层 l 中坐标 $\begin{bmatrix}2 & 2\end{bmatrix}^{\top}$ 通过仿射变换得到前一层的坐标为 $\begin{bmatrix}1.6, 2.4\end{bmatrix}^{\top}$，则有如下的关系
$$
\begin{align}
a_{22}^{l}  & = (1 - 0.4) \times(1 - 0.4) a_{22}^{l-1} \\
 & + (1 - 0.6)\times  (1 - 0.4) a_{12}^{l-1} \\
 & + (1 - 0.6)\times (1 - 0.6) a_{13}^{l-1} \\
 & + (1 - 0.4)\times (1 - 0.4) a_{23}^{l-1}
\end{align}
$$
![[image-11.webp]]