1. RAG 和 大语言模型发展

LLM已经从上下文窗口小到几千个token到最新的一次处理Gemini 1.5 Pro多达一百万个令牌。
包括不同大语言模型的编码器，编码效能也在不断的提升。
OpenAI的模型中，相比较于gpt3.5 和gpt-4-turbo，gpt-4o模型，很重要的一个更新就是变更了编码集。 从原来的cl100k_base 变成了o200k_base。
方法给出的20中语言的token优化如下：
![token_Optimization_o200k_base](./上下文感知RAG_EnhancingRAGwithContextualRetrieval_20241118/token_Optimization_o200k_base.png)

更多的说明链接可以参考官方文档：
[hello-gpt-4o](https://openai.com/index/hello-gpt-4o/)

所以从context windows还是tokenizer的优化，都得到了巨大的飞跃。为了寻找处理超出初始上下文窗口所能处理的信息的方法，人们开发了检索增强生成 （RAG），它将与他们的语言模型相关联，并帮助从外部文档中检索实时和准确的答案。但是现在我们有大约 1000 倍大的上下文窗口可以处理整个百科全书，一个问题自然而然地出现了：我们还需要 RAG 吗？

简短的回答是肯定的。
长一点的答案是 
    – 这不仅是关于拥有更多信息，而且是关于正确的信息，以做出更明智的决策。

如图是long-context LLM回答和RAG的简单对比：
![contextLLM_VS_RAG](./上下文感知RAG_EnhancingRAGwithContextualRetrieval_20241118/Long-contextLLM_VS_RAG.webp)

Long-context LLM pros 长上下文 LLM 优点:

- Quick retrieval 快速检索: 长上下文模型可以持续接收输入、推理和动态检索信息。另一方面，RAG 需要附加外部文档，然后使用相同的数据来完成其所有任务。这意味着，虽然您可以不断将新信息放入上下文中，但对于 RAG，这需要更多的步骤。
- Easier to use 更易于使用: RAG 涉及多个组件 – 检索、嵌入模型和语言模型。这意味着，为了使您的 RAG 发挥最佳效果，您需要设置嵌入参数、分块策略，然后尝试测试您的模型是否真的给出了正确的答案。总的来说，比仅仅输入长提示要多花点功夫。
- Handy for simple tasks 方便执行简单任务: 如果情况不太复杂，并且需要从大量文本中相对简单地检索，则长上下文模型可以快速方便地使用。在这种情况下，如果您尝试使用 LLM，它可能工作得很好。也会不起作用的情况，RAG能做相应的增强。

RAG 优点:
- Complex RAG 将继续存在: 更简单的 RAG 形式（以微不足道的方式对数据进行分块和检索）可能会有所下降。但更复杂的 RAG将继续存在并且进化。当今的 RAG 系统包括复杂的工具，如查询重写、数据块重新排序、数据清理和优化的向量搜索，这些工具增强了它们的功能并扩展了它们的覆盖范围。
- RAG 可以更高效: 扩展 LLM 的上下文窗口以包含大量文本肯定有其自身的一系列障碍，尤其是当您考虑到较慢的响应时间和计算成本的上升时。环境越大，需要处理的数据就越多，这真的会开始累积起来。另一方面，RAG 通过仅检索相关和必要的信息来保持精简和平均.
- RAG 对资源更友好:与长上下文窗口所涉及的大量处理相比，RAG 仍然是更实惠、更快速的解决方案。它允许开发人员使用额外的上下文来增强 LLMs而无需花费大量时间和成本来处理大量数据块。
- RAG 更易于调试和评估: RAG 是一本打开的书——这意味着，您可以轻松地从一个问题到另一个答案。这对于大型文档或复杂的推理任务特别有用。这意味着 RAG 可以帮助您轻松调试答案，而放置太多上下文可能会难以处理并导致错误/幻觉。
- RAG 能够保持最新状态: RAG 的最大优势之一是它将最新的数据集成到 LLM 的决策过程中。通过直接连接到更新的数据库或进行外部调用，RAG 确保正在使用的信息是最新的可用信息，这对于及时性至关重要的应用程序至关重要。
- RAG 战略性地处理信息: 通常，当关键信息位于输入的开头或结尾时，LLMs 的性能最佳。这意味着，根据最近的研究，如果你问的问题涉及上下文的其余部分，你可能会对答案感到失望。同时，使用 RAG，有一些技术，例如对文档进行重新排序，您可以使用这些技术根据文档的优先级战略性地更改文档的位置。如果在上下文中完成，这将是一个很大的障碍。
![contextLLM_VS_RAG](./上下文感知RAG_EnhancingRAGwithContextualRetrieval_20241118/RAG_0-1729080789691.gif)

2. 如果更有效利用LLM context windwos来加强RAG

开发人员通常使用 Retrieval-Augmented Generation （RAG） 来增强 AI 模型的知识。RAG 是一种从知识库中检索相关信息并将其附加到用户提示符的方法，从而显著增强了模型的响应。问题在于，传统的 RAG 解决方案在编码信息时会删除上下文，这通常会导致系统无法从知识库中检索相关信息。
![standardRAG](./上下文感知RAG_EnhancingRAGwithContextualRetrieval_20241118/standardRAG.webp)

同时OpenAI和Azure OpenAI给出了提示词缓存[prompt-caching](https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/prompt-caching)
可以进一步的降低长context的成本，提升相应速度。这样构建一个固定比较长的context，来对每个chunk进行上下文的补全。

通过一个简单的提示，让它对文章的内容上下文进行说明:
编写了一个提示，指示模型提供简洁的、特定于 chunk 的上下文，该上下文使用整个文档的上下文来解释 chunk。
```prompt
<document> 
{{WHOLE_DOCUMENT}} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{{CHUNK_CONTENT}} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
```
架构如图：
![contexttual_retrieval_Preprocessing](./上下文感知RAG_EnhancingRAGwithContextualRetrieval_20241118/contexttual_retrieval_Preprocessing.webp)

3. 初步评测数据

使用embedding检索并检索前 20 个块的所有知识域的平均性能。衡量在前 20 个区块中无法检索的相关文档的百分比。
![benchmark_data_top20](./上下文感知RAG_EnhancingRAGwithContextualRetrieval_20241118/benchmark_data_top20.webp)

结论：
1. 上下文嵌入将前 20 个块的检索失败率降低了 35%（5.7% → 3.7%）
2. 结合上下文嵌入和上下文 BM25 将前 20 个块的检索失败率降低了 49%（5.7% → 2.9%）。

***向上下文窗口中添加更多数据块会增加包含相关信息的机会。但是，更多信息可能会分散模型的注意力，因此这是有限制的。我们尝试了交付 5、10 和 20 个 chunk，发现使用 20 是这些选项中性能最高的**

4. 通过重新排名进一步提高性能

重新排名是一种常用的筛选技术，用于确保仅将最相关的数据块传递给模型。重新排名可提供更好的响应，并降低成本和延迟，因为模型处理的信息更少。
![rerank_contextual_retrieval](./上下文感知RAG_EnhancingRAGwithContextualRetrieval_20241118/rerank_contextual_retrieval.webp)

发现 Reranked 上下文嵌入和上下文 BM25 将前 20 个块的检索失败率降低了 67% （5.7% → 1.9%）。
