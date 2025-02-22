## LLM 漂移、提示漂移、链接和级联

参考论文: [How Is ChatGPT’s Behavior Changing over Time?](https://arxiv.org/pdf/2307.09009)

提示链（Prompt Chaining）可以手动或自动执行;
手动需要通过 GUI 链构建工具手动制作链（比如prompt flow 或dify之类的平台框架）。
Autonomous Agent 在执行和利用他们所掌握的工具时动态创建链。
这两种方法都容易受到级联、LLM 和 prompt Drift 的影响。


### LLM 漂移

LLM 漂移是 LLM 响应在短时间内的明确变化。
这与 LLMs本质上是不确定（概率随机和temperature等）无关，也与轻微的提示工程措辞更改无关;
而是对 LLM根本性的改变。

最近的一项研究发现，在四个月的时间里，GPT-4 和 GPT-3.5 的响应准确性在积极但更令人担忧的情况下波动很大，而且是负面性--降智。
研究发现，GPT-3.5 和 GPT-4 都有很大差异，并且某些任务的性能会下降。

这些任务下图显示了四个月期间模型准确性的波动。在某些情况下，弃用是相当明显的，准确性损失了 60% 以上。
![](./LLM&PromptDrift/LLM_Drift.webp)


### Prompt Drift 提示漂移

Prompt Drift 是由于模型更改、模型迁移或推理时提示注入数据的变化而随着时间的推移产生不同响应的现象。

LLMs是不确定的，这意味着同一 LLM在不同时间很可能随着时间的推移产生不同的响应。
从本质上讲，这不是问题，措辞可能会有所不同，而基本事实保持不变。

在某些情况下，LLM。例如，LLMs 已被弃用，通常需要迁移，正如我们最近看到的 OpenAI 弃用了一些模型。因此，提示保持不变，但使用的模型会发生变化。

在推理时注入提示的数据有时也可能不同。可以说，所有这些因素都导致了一种称为瞬时漂移的现象。

简而言之，提示偏差可能由以下原因引起：
- Model-inspired tangents, 模型启发式切线
- 错误的问题提取
- LLM 随机性和创意惊喜


*** 出现了提示管理和测试接口，比如 ChainForge，最近 LangChain 推出了 LangSmith，以及 Vellum 等商业产品。***
*** 市场明确需要确保在大型语言模型迁移/弃用之前可以测试生成应用程序。如果一个模型在很大程度上与底层 LLM，那就更好了。***

#### Cascading 级 联

级联是指链中的一个节点引入畸变或偏差，并且这种意外的异常会延续到下一个节点，在那里异常很可能会加剧。
每个节点输出都离预期结果越来越远，这种现象通常称为级联。

考虑下图的因素：

- 1. 在链接的应用程序中，用户输入可能是意外的或计划外的，因此会从节点产生不可预见的输出。
- 2. 前一个节点的输出可能不准确或产生一定程度的偏差，这在当前节点中会加剧。
- 3. LLM 响应也可能是意外的，因为 LLMs 是不确定的。第三点是可以引入提示漂移或 LLM。
- 4. 然后，节点 2 的输出被转移并导致偏差的级联。

### 总结

Prompt Chaining 不应该孤立地看待，而应该将 Prompt Engineering 视为一门由几个分支组成的学科。

提示 LLM也很重要，并且对输出质量有明显的影响。

Prompt Engineering 是 Chaining 的基础，而 Prompt Engineering 的学科非常简单易懂。

随着 LLM 环境的发展，提示正变得可编程（通过 RAG 的模板和上下文注入）并合并到越来越复杂的结构中。因此，链接应该得到 Agents、Pipelines、Chain-of-Thought Reasoning 等元素的支持。

