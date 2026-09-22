## Context Engineering
基于 SWE-Bench 的工具输出压缩策略对比实验
https://arxiv.org/abs/2508.21433

工具输出放到外部存储：
- https://manus.im/blog/Context-Engineering-for-Al-Agents-Lessons-from-Building-Manus
- https://arxiv.org/abs/2511.22729

如何管理 Agent 的长期记忆？
A-MEM:https://arxiv.org/abs/2502.12110
Mem0: https://arxiv.org/abs/2504.19413
Memory OS: https://arxiv.org/abs/2506.06326


### 压缩
提示词压缩的失败：Context Collapse，压缩后信息丢失，导致无法回答对原本能答对的问题
- [ACON (Agent Context Optimization)](https://arxiv.org/abs/2510.00615) 核心思想：用另一个模型阅读压缩失败的案例，总结出 Feedback，再用于压缩
- 另一个思路：强化学习训练模型，在真实的任务流程中学习压缩策略 [Summarization augmented Policy Optimization (SUPO)](https://arxiv.org/abs/2510.06727)

#### 让模型学会使用压缩工具
模型天生就不会主动使用压缩工具，即使通过系统提示词的约束也很难实现自主的压缩，只有微调模型才能让模型使用压缩工具。 [AgentFold](https://arxiv.org/abs/2510.24699)

#### 通过 Subagent 实现压缩
Subagent 实现自主压缩：主 Agent 可以为子 Agent 派发子任务，子 Agent 完成子任务后，向主 Agent 返回结果。此时主 Agent 的上下文中只有任务结果，和任务过程中工具调用有关的上下文都随着子 Agent 关闭而丢弃了。

而模型使用 Subagent 的能力也需要通过学习得到，通过强化学习，对模型主 Agent 过长的上下文做出惩罚，对子 Agent 做出限定范围外的任务做出惩罚，让模型学会使用 Subagent。


#### 从根源减少过长上下文
- 在整个上下文中，占据最多的是 observation，有至少 80%，比如文档的全部内容、工具的输出结果等， https://arxiv.org/abs/2508.21433
- 在 Software Engineering 中，有至少 70%的上下文用于存放程序代码内容 https://arxiv.org/abs/2601.16746

[过滤](https://arxiv.org/abs/2601.16746)方法通过实现更加智能的工具，在读一些大型的文件时，针对性地选取其中有关的部分放入上下文。
在 OpenClaw 中，关于 Memory 的处理策略如下：
```md
## Memory Recall
Before answering anything about prior work, decisions, dates, people,
preferences, or todos: run **memory_search** on MEMORY.md + memory/*.md;
then use **memory_get** to pull only the needed lines.
```


另一种过滤策略：**按需加载** [MCP Zero](https://arxiv.org/abs/2506.01056)，用户输入的需求可能是模糊的，让模型分析需要的工具，在通过搜索将工具的信息添加到上下文中。和 SKILL 的加载方式类似。

---- 
通过 LLM 管理上下文：
- [Agentic Context Engineering](https://arxiv.org/abs/2510.04618) 
- 通过 Prompt Engineering 实现 [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952) 让模型总结当前的上下文，整理出关键内容、可复用的策略等。
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) 模型不能看到完整的上下文，只能看到其中的摘要信息（Meta Data）根据这些信息调用搜索程序在存储中查找相关的上下文，从而更改 Meta Data。