1. 什么是Embedding？
嵌入是表示内容（如自然语言或代码）中概念的数字序列，便于嵌入测量文本字符串的相关性 。嵌入使机器学习模型和其他算法能够轻松理解内容之间的关系并执行聚类或检索等任务。
[text2Vector](./Embedding模型对RAG的重要性_20241118/text2Vector.png)

嵌入是浮点数的向量（列表）。两个向量之间的距离衡量它们的相关性。小距离表示高相关性，大距离表示低相关性。

2.Embedding主要使用场景

- 搜索（其中结果按与查询字符串的相关性排名）
- 聚类（其中文本字符串按相似性分组）
- 推荐（推荐包含相关文本字符串的项目）
- 异常检测（识别出相关性不大的异常值）
- 多样性测量（分析相似性分布）
- 分类（其中文本字符串按其最相似的标签进行分类）

3. 不同的Embedding模型的区别

什么嵌入模型最适合你自己的语言？许多研究和排行榜调查了什么是英语中最好的嵌入模型，但那里的语言太多了？
如果找到一种在西班牙语、日语、印度尼西亚语或法语，或者中文中表现良好的嵌入模型。
这就需要一些测试数据来做为参考。

- 模型版本和维度大小

首先看看OAI的基准测试
[OAI_Embedding_Model_Performance_table](./Embedding模型对RAG的重要性_20241118/OAI_Embedding_Model_Performance_table.webp)
[OAI_Embedding_performance](./Embedding模型对RAG的重要性_20241118/OAI_Embedding_performance.webp)

正如预期的那样，对于大型模型，使用较大的嵌入大小 3072 可以观察到更好的性能。
但是如果使用新的版本的话，即使相同的维度信息也可以得到更好的效果。

***在评估中，大型、小型和 Ada 模型之间的性能差异远不如 MTEB 基准那么明显，这反映了这样一个事实，即在大型基准中观察到的平均性能不一定反映在自定义数据集上获得的性能。***

- 更多开源Embedding模型对比，可以基于自己的需求场景：特定语言，或者多语言，多模态，特定行业等

围绕嵌入的开源研究非常活跃，并且定期发布新模型。[Hugging Face 😊 MTEB](https://huggingface.co/spaces/mteb/leaderboard) 排行榜是了解最新发布模型的好地方。
[Embedding-leader-board](./Embedding模型对RAG的重要性_20241118/Embedding-leader-board.png)






