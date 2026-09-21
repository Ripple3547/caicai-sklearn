之前讲的 model 都是基于固定数量的 vector 输入的，接下来就以不定数量的一批 vector 输入的模型做讨论。

例如，
- 处理一串单词输入，可以用 One-hot 编码处理所有的词汇，也可以用 Word Embedding 将一个单词表示成一个向量，每个向量代表一部分语义。
- 音频输入，采用滑动窗口，将一个窗口内的声音信号编码为一个向量
- 一个 Graph
- 一个分子

输出的类别：
- 对应输入 N 个向量，输出 N 个向量，即输入长度和输出长度相同，如 POS Tagging（词性标注）、声音辨识（判别输入声音向量中各自对应的内容）、对 Graph 中每个节点进行判断
- 单个输出，如 Sentiment analysis、语音发言人辨认、分子功能预测
- 不确定输出长度，又称为 seq2seq，如翻译

## Sequence Labeling
考虑输入 N 个向量，输出 N 个结果的模型。一个直观的想法是对每个向量经过一个全连接网络，得到一个 Label，该做法的缺点是无法考虑上下文关系，如对 `I saw a saw` 进行词性标注，两个 `saw` 的结果应该不同，如果只考虑当前单个单词向量，会得到相同的词性。

改进策略：考虑当前向量的邻居窗口，将前后一定范围内的向量一并考虑进全连接网络。该方法也有一定的上限：
- 增大窗口的大小可以提高模型上限，但输入的长度不统一，窗口大小不好选
- 同时越大的窗口大小意味着更多的参数，会导致 overfitting 发生的可能性增加。

## Self-Attention
Self-Attention 考虑整个 Input Sequence，可以是网络的 Input，也可以是隐藏层的 Output。

这里的输入和输出长度相同，即输入为 $a^{i}$，输出为 $b^{i}$， $i = 1, \cdots, N$

### Relevance
向量的相关性 $\alpha$ 有两种计算方式
#### Dot-Product
对两个向量分别作用矩阵 $W^{q}$ 和 $W^{k}$ 得到 $q$ 和 $k$，再做向量点积得到相似度 $\alpha = q \cdot k$

#### Additive
对两个向量分别作用矩阵 $W^{q}$ 和 $W^{k}$，得到的两个向量相加，依次作用 $\tanh$ 和矩阵 $W$，得到 $\alpha$

---
具体计算过程中，假设要计算 $a^{1}$ 和其他输入之间的相关性，步骤为
- 计算 $q^{1} = W^{q}\cdot a^{1}$，称为 query
- 对于其他的向量，如 $a^{2}$，有 $k^{2} = W^{k} \cdot a^{2}$，称作 key
- 计算相似度 $\alpha_{1,2} = q^{1}\cdot k^{2}$，又称为 Attention Score，也可计算自身的 Score $a_{1,1}$
- 相似度再通过一次 Softmax 处理得到 $\alpha'$

在得到 $a^{1}$ 和其他向量之间的相关性之后，每个向量计算 $v^{i} = W^{v} \cdot a^{i}$，根据相关性加权求和得到 $b^{1}$。即 $b^{1} = \sum_{i}\alpha_{1,i}' v^{i}$ 


### 矩阵乘法角度
- Input 矩阵 $I = \begin{bmatrix}a^{1} & \cdots &  a^{N}\end{bmatrix}$， Query 矩阵 $Q = \begin{bmatrix}q^{1} & \cdots &  q^{N}\end{bmatrix}$，关系为 $Q = W^{q}I$
- 同理有 $K = W^{k}I, V = W^{v}I$
- 相似度计算：$A = K^{\top}Q$ ，其中每一列为该 Input 和其他向量的相似度（KQ 值），再经过 Softmax 后得到 $A'$，为真正的相似度矩阵
- 最后计算结果矩阵 $O = V A'$ 

![[image-12.webp]]
在整个模型中，超参数只有 $W^{q},W^{k}, W^{v}$。

## Multi-head Self-Attention
在寻找相关性时，用的是一个 q 和其他的 k 做点积得到单一的相似度，实际上可以通过增加 q 来发现更多的**相关**。

在原来的模型计算出的 $q^{i}, k^{i}, v^{i}$ 基础上，分别经过若干个矩阵作用得到多个 head 的 q,k,v。如设定 head = 2，则根据 $a^{1}$ 计算出 $q^{1}$ 后，经过两个不同的矩阵计算出 $q^{1,1}$ 和 $q^{1,2}$，其他同理。

在计算输出矩阵 $O$ 时，按照每个 head 分别计算。例如，对当前的 Input $i$，Head 共 2 个，对应的 Query 有两个 $q^{i,1}$ 和 $q^{i,2}$，在计算相应的 k 和 v 后得到 $b^{i,1}$ 和 $b^{i,2}$，此时经过 $W^{O}$ 后得到最终 $i$ 对应的输出 $b^{i}$。
![[image-13.webp]]


### Self-Attention 的缺点——Positional Encoding
在标准的 Self-Attention 模型中，每个 Input 只考虑了其和其他向量之间的相似度来计算注意力，实际上在翻译、语言理解等任务中，词的位置信息也是重要的因素。

**Positional Encoding** 在输入 $a^{i}$ 基础上加上一个位置相关的向量 $e^{i}$（人为指定）

详见 https://arxiv.org/abs/2003.09229.


### Self-Attention for Speech
在语音辨识场景中，由于输入窗口的大小有限，当语音的长度增加时，模型的输入规模也进一步扩大，而在计算相似度矩阵的 Softmax 时，若输入规模为 $L$，计算的复杂度为 $O(L^{2})$，因此需要限制 Attention 的范围，如 **Truncated Self-Attention**，不再对所有的输入计算相似度，而是截取一部分（可以是相邻的输入）。


### Self-Attention v.s. CNN
CNN 可以看作一个简化版的 Self-Attention，因为只考虑 Receptive field 中的 Attention；反过来，Self-Attention 也可以看作是可以学习的 Receptive field。
https://arxiv.org/abs/1911.03584

因此，从模型表现来看，小训练数据上，CNN 的表现会更佳，而 Self-Attention 会出现 Overfitting 的结果；而在大的训练数据上，Self-Attention 往往有更好的表现（弹性更大，上限更高）

### Self-Attention v.s. RNN
RNN 只考虑当前输入和其之前的输入，且只能**串行**计算，因此当输入序列较长时，当前的输出结果容易遗忘相隔较远的输入。
![[image-14.webp]]
