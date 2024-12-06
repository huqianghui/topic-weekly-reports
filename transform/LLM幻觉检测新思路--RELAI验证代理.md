大型语言模型（LLM）已经在各个领域中展现出其强大的文本生成和理解能力。
伴随着这些技术的广泛应用，一个不可忽视的问题也逐渐浮出水面——LLM幻觉。幻觉，即LLM(LLM的擅长与不擅长：深入剖析大语言模型的能力边界)生成的错误或虚构信息，可能会误导用户，尤其在医疗健康、金融等关键领域，其潜在风险更是不可估量。
为了应对这一挑战，RELAI代理应用而生，为实时检测LLM幻觉提供了新的解决方案。

## LLM幻觉：不容忽视的问题

近年来，OpenAI等科技公司发布的最新研究指出，即使是像GPT-4和Claude-3.5-Sonnet这样的顶级LLM，也存在较高的幻觉生成率。特别是在OpenAI推出的SimpleQA数据集中，这些问题被更加直观地展现出来。SimpleQA是一个专注于短答案问题的数据集，其设计目的在于提供一个健壮的基准来评估LLM回答的事实准确性。该数据集经过精心设计和严格的质量检查，最小化了模糊性，使得检测LLM幻觉变得更为可行。

在SimpleQA数据集的测试中，GPT-4和Claude-3.5-Sonnet等顶级LLM的表现并不理想。它们在高幻觉率上的挣扎揭示了现有LLM在事实准确性方面的不足。例如，对于“Bil Keane赢得国家漫画家协会最佳辛迪加面板奖的次数”这一问题，GPT-4给出了错误的答案“三次”，而实际答案是“四次”。这种幻觉的产生，源于LLM复杂的训练过程、输入分词以及模型架构等多种因素。

## RELAI验证代理：多层次检测，提升可靠性

针对LLM幻觉这一严峻问题，RELAI公司推出了其LLM验证代理。这些专业化的代理能够实时检测并标记LLM输出中的幻觉，从而提升LLM输出的可靠性。RELAI的验证框架包括了多种互补的验证代理，每个代理都具有独特的检测能力，共同构成了一个多层次的验证体系。

### 幻觉验证代理（Hallucination Verifier Agent）

该代理专注于分析 LLM 生成内容的统计模式。通过对大量文本数据的学习，它能够识别出那些在统计上缺乏事实依据的线索。例如，某些词汇的组合、语义模式在真实知识体系中出现的概率极低，如果在 LLM 的输出中频繁出现这类模式，就可能暗示着幻觉的存在。它并不直接判断答案的对错，而是从统计角度为其他代理提供潜在问题的提示，起到初步筛查的作用。

### LLM 验证代理（LLM Verifier Agent）

RELAI 利用其专有的 LLM 作为辅助模型，将原始 LLM 的响应与之进行交叉参考。它能够识别出在两个模型输出之间存在的不一致之处，从而标记出可能存在事实不准确的答案。这种基于模型对比的方式，利用了不同模型在知识理解和表达上的差异，通过相互校验来提高检测的准确性。例如，当原始 LLM 给出一个答案时，LLM 验证代理会从其辅助模型的角度出发，判断该答案是否符合一般的知识逻辑和语言模式，如果存在明显差异，则可能存在幻觉。

### 基于事实的 LLM 验证代理（Grounded LLM Verifier Agent）

此代理会从可靠的、预先批准的数据源（如权威知识库、学术数据库等）中检索信息，并将 LLM 生成的答案与之进行精确匹配。这一过程类似于传统的事实核查，通过与可靠来源的对比，为 LLM 的输出增加了额外的验证层。例如，对于一个历史事件相关的问题，它会在权威历史资料中查找准确答案，并与 LLM 的回答进行比对，若不一致则标记为可能的幻觉。

这三种验证代理可以根据用户的需要选择操作模式：“常规模式”（默认）和“强模式”。在常规模式下，代理主要针对响应中的主要不准确之处；而在强模式下，代理会进行更深入的分析，甚至识别出细微的不准确之处。

## 组合优势：RELAI集成验证代理

RELAI的验证代理不仅可以单独使用，还可以组合成集成验证代理，进一步提升检测效果。集成验证代理包括两种类型：

RELAI集成验证代理-I（RELAI Ensemble Verifier-I）：当所有个体代理都检测到幻觉时，该代理会标记幻觉。

RELAI集成验证代理-U（RELAI Ensemble Verifier-U）：当至少一个个体代理检测到幻觉时，该代理就会标记幻觉。

通过组合不同代理的信号，RELAI的集成验证代理能够提供更全面、更可靠的幻觉检测。这种多层次的验证体系，确保了从多个角度对LLM输出进行审查，大大提高了检测的准确性和可靠性。

## 评估与实验：超越现有基线

### 评估设置与指标   
1、数据集与样本选择

实验在 SimpleQA 数据集上进行，该数据集具有高质量和低歧义性的特点，能够有效测试幻觉检测方法的性能。从数据集中随机抽取 200 个提示（prompt）用于评估，确保样本的随机性和代表性。

2、评估指标

检测率（Detection rate，又称真阳性率 True Positive Rate）：指正确标记为幻觉的不正确响应的百分比。这一指标直接反映了验证代理对幻觉的识别能力，检测率越高，说明能够发现更多的幻觉情况。

误报率（False Positive rate）：指被错误标记为幻觉的正确响应的百分比。误报率越低，说明验证代理的准确性越高，避免了将正确答案误判为幻觉的情况。理想的幻觉检测器应实现 100% 的检测率和 0% 的误报率。


###  在 GPT-4o 上的实验结果  

1、与现有基线方法对比

在对 GPT - 4o 的响应进行评估时，将 RELAI 的验证代理与 SelfCheckGPT（使用 NLI 和 LLM Prompt 两种方式）以及 INSIDE 等现有基线方法进行比较。结果显示，RELAI 的方法在不同误报率水平下均显著优于现有基线方法。例如，RELAI 的基于事实的 LLM 验证代理在约 5% 的误报率下，检测率达到了 78%。而 RELAI 集成验证代理 - I 在接近 0% 误报率时，检测率为 28.6%，这意味着添加该代理到 LLM 中可以在不引入任何误报的情况下将幻觉率降低三分之一。


2、具体示例分析

以 “Bil Keane 赢得美国国家漫画家协会最佳联合漫画奖的次数” 这一问题为例，地面真相（Ground truth）为四次，而 GPT - 4o 回答为三次。RELAI 的 LLM 验证代理指出正确答案应为四次而非三次；幻觉验证代理提示该答案缺乏支持，应进行交叉验证；基于事实的 LLM 验证代理则明确指出该响应不准确，并提供了准确的获奖年份信息，参考来源为 Wikipedia。在这个示例中，所有三个验证代理都成功标记了幻觉，展示了 RELAI 验证代理在实际检测中的有效性。

### 在 Claude-3.5-Sonnet 上的实验结果

1、性能趋势相似性
为测试 RELAI 验证代理的通用性，在另一个流行的 LLM Claude - 3.5 - Sonnet 上进行了相同的实验。结果发现，RELAI 代理在 Claude - 3.5 - Sonnet 上的性能趋势与在 GPT - 4o 上相似。

2. 具体数据表现

在约 10% 的误报率下，RELAI 的基于事实的 LLM 验证代理实现了 81% 的检测率；在接近 0% 误报率时，RELAI 集成验证代理 - I 达到了 27.5% 的检测率。同样，RELAI 代理在 Claude - 3.5 - Sonnet 上也显著优于现有基线方法，进一步证明了其在不同 LLM 模型上的有效性和通用性。

## 使用与接入：轻松实现实时检测

RELAI还提供了一个易于使用的平台，使用户能够轻松利用这些实时幻觉检测代理。用户只需选择基础模型（如GPT-4），然后添加一个或多个验证代理，即可在实时聊天中标记潜在幻觉。此外，RELAI还为企业用户提供了API接入服务，使其能够无缝集成RELAI的AI可靠性解决方案。这使得企业能够在自己的应用程序或系统中嵌入RELAI的验证代理，从而提升整个系统的准确性和可靠性。RELAI的LLM验证代理为LLM幻觉检测提供了新的解决方案，显著超越了现有方法。通过多层次、多角度的验证体系，RELAI的验证代理能够实时检测并标记LLM输出中的幻觉，从而提升LLM输出的可靠性和准确性。

![rela-ai](./LLM幻觉检测新思路--RELAI验证代理/rela-ai-小李子.png)

## 考虑如果实现自定义agent

考虑如果自我构建构建一个agent 或者multi-agent来实现一个自定义的agent群体，实现对结果的检测。








