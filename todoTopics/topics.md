### TODO topics

#### 2024-31W-topics

1. 数据合成与模型崩溃

很多公司模型训练遇到训练数据的缺失，导致没有办法进行下去。
所以出现通过模型本身来生成数据，或者所谓的数据蒸馏等，来产生更多的训练和测试数据。

[Synthetic Data (Almost) from Scratch](https://arxiv.org/abs/2402.13064)

[quickstart-github-sample](https://github.com/Azure/synthetic-qa-generation/)

同时也出现另外一个声音，就是合成数据到时候语言模型的崩塌。

[AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y)



2. AppAgent: Multimodal Agents as Smartphone Users

不通过function call，而是通过 appAgent 方式来操作app，然后整合autogen，来实现对客户和应用的整合。

[AppAgent: Multimodal Agents as Smartphone Users](https://arxiv.org/abs/2312.13771)

[AppAgent-github-repo](https://github.com/mnotgod96/AppAgent)


3. RouteLLM--通过不同的任务来选择不同的模型

不同的任务选择不同的模型，达到性价比的最优化。这个和LLM-gateway的思路和出发点不一样，但是模式类似，看怎么融合在一起。

[RouteLLM](https://github.com/lm-sys/RouteLLM)


4. Azure Assitant API 与 bot service framework整合

如果把一个azure openAI的一个Assitant，快速和bot service framework整合，快速变成一个助手。

