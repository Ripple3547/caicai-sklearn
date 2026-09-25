## Recurrent Neural Network
Slot Filling：将用户的输入内容映射到预定义的槽位中，形成结构化的信息。如在订票系统中，对应输入
```md
I would like to arrive *Taipei* on *November 2nd*.
```
Slot 为：
- Destination: `Taipei`
- Time of arrival: `November 2nd`

单词编码方案：
- 1 of N encoding：向量的长度为单词的总数，每一个维度
- Beyong 1 of N encoding：将一些未知的单词全部归类到一个维度 `other`
- Word hashing：用 n-gram 表示，每个维度出现记为 1


如何解决上下文语境的影响问题？如何区分 `arrive Taipei` 和 `leave Taipei`？如果使用普通的神经网络，对每个单词预测其属于每个 Slot 的概率，并不能考虑其所在的上下文。

### RNN
RNN 的处理方法为：每次隐藏层的输出会暂存到 Memory 中，下一个 Input 会同时引入 Memory 中的内容作为考虑作为隐藏层的输入
![[image-16.webp]]

不同的 Memory 定义方式：
- Elman Network：即将隐藏层的输出存入 Memory 中
- Jordan Network：将网络的输出 $y^{t}$ 存入 Memory 中，用作下一个 Input 的输入。

Bidirenctional RNN：双向 RNN，分别用两个模型从正向和反向遍历输入，最终的 Slot 结果由正向和反向模型同时决定。


### Long Short-term Memory（LSTM）
LSTM 有一个 Momery Cell，用于存放其他 Network 的输出结果。LSTM 的输入一共有 4 个部分：
- 其他 Network 的输出
- Input Gate：用于决定是否存入 Memory Cell
- Output Gate：用于决定是否外界是否可读 Memory Cell 中的内容
- Forget Cell：何时遗忘 Memory Cell 中的内容
![[image-17.webp]]
更规范的表达方式：
- 输入 $z_{i}, z_{f}, z_{o}$ 和 $z$
- 所有的激活函数都为 Sigmoid
- Input Gate 部分：$g(z) f(z_{i})$ 
- Forget Gate 部分：$c f(z_{f})$
- 更新后 Memory Cell 中的值：$c' = g(z) f(z_{i}) + c f(z_{f})$
- Output Gate 部分： $f(z_{o}) h(c')$

整个 LSTM 单元可以当作一个神经元节点，其 4 个输入可由输入序列通过不同的变换得到，因此 LSTM 的参数量会是普通简单神经网络的 4 倍。

---
![[image-18.webp]]
在完整的 LSTM 中，控制 Gate 的不止 Input，还有隐藏层的输出 $h^{t}$ 和 Memory Cell 中的值 $c^{t}$



---
![[image-19.webp]]
在训练 RNN 的过程中，Loss 的变化曲线往往不是呈下降趋势的，具有非常大的波动性。根源在于 RNN 中同一个参数会在一个时间序列上被反复使用，使得参数带来的影响呈现指数增长趋势（就如 $0.99^{999}$ 和 $1.01^{999}$ 的区别），因此 Error Surface 的形状变化非常大，有时 Loss 会小幅变化，有时的梯度变得非常大，而 $\alpha$ 没有得到调整，导致迭代距离过远，Loss 大幅波动。

LSTM 可以用于解决 RNN 中的梯度消失问题（无法解决梯度爆炸），让 Error surface 不会出现变化非常小的部分
