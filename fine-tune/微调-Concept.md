## 什么是微调
根据上面解释，微调（Fine-tuning）是一种迁移学习的方法，用于在一个预训练模型的基础上，通过在特定任务的数据上进行有监督训练，来适应该任务的要求并提高模型性能。
微调利用了预训练模型在大规模通用数据上学习到的语言知识和表示能力，将其迁移到特定任务上。

### 微调的一般步骤

1. 预训练模型选择：选择一个在大规模数据上进行预训练的模型作为基础模型。例如，可以选择一种预训练的语言模型，如BERT、GPT等。

2. 数据准备：准备用于微调的特定任务数据集。这些数据集应包含任务相关的样本和相应的标签或目标。确保数据集与任务的特定领域或问题相关。

3. 构建任务特定的模型头：根据任务的要求，构建一个特定的模型头（task-specific head）。模型头是添加到预训练模型之上的额外层或结构，用于根据任务要求进行输出预测或分类。例如，对于文本分类任务，可以添加一个全连接层和softmax激活函数。

4. 参数初始化：将预训练模型的参数作为初始参数加载到微调模型中。这些参数可以被视为模型已经学习到的通用语言表示。

5. 微调训练：使用特定任务的数据集对模型进行有监督训练。这包括将任务数据输入到模型中，计算损失函数，并通过反向传播和优化算法（如梯度下降）更新模型参数。在微调过程中，只有模型头的参数会被更新，而预训练模型的参数会保持不变。

6. 调整超参数：微调过程中，可以根据需要调整学习率、批量大小、训练迭代次数等超参数，以达到更好的性能。

7. 评估和验证：在微调完成后，使用验证集或测试集对微调模型进行评估，以评估其在特定任务上的性能。可以使用各种指标，如准确率、精确率、召回率等。

8. 可选的后续微调：根据实际情况，可以选择在特定任务的数据上进行进一步的微调迭代，以进一步提高模型性能。
   
微调的关键是在预训练模型的基础上进行训练，从而将模型的知识迁移到特定任务上。
通过这种方式，可以在较少的数据和计算资源下，快速构建和训练高性能的模型。

### Parameter-Efficient Fine-Tuning（PEFT）

PEFT是一种旨在减少模型参数调整量的微调技术，特别适用于大规模预训练模型的迁移学习场景。
PEFT 通过冻结大部分预训练模型的参数，只调整小部分参数来适应目标任务，显著降低计算成本和存储需求，同时保持性能接近全量微调。

主要思想：
	1.	减少参数调整量：只优化模型的一小部分参数，如添加的任务特定层或选择性地解冻某些模块。
	2.	高效利用预训练知识：利用大模型的通用表示能力，通过少量参数调整适应新任务。
	3.	节省计算和存储：适合硬件资源受限或需要在多任务间共享模型的场景。

优势：
	1.	参数效率高：PEFT 通常只需调整 0.1%-1% 的总参数量。
	2.	跨任务适用性强：支持多任务学习，每个任务可拥有独立的轻量化参数模块。
	3.	易于部署：微调后可以在原始预训练模型的基础上加载额外参数，避免重新训练整个模型。
	4.	硬件友好：降低显存占用，适合在内存受限的设备上应用。

#### 常见的PEFT方法

1. Adapter
   
适配器层是插入预训练模型层之间的小型神经网络。
在微调过程中，只训练这些适配器层，保持预先训练的参数冻结。
通过这种方式，适配器学习将预先训练的模型提取的特征适应新任务。
Adapter 技术 是一种微调预训练模型的方式，它通过在模型原有结构中引入 额外的参数模块（即 Adapter 模块），让这些模块在训练过程中进行微调，而原始模型的参数保持不变或少量更新，从而减少计算开销。

在模型的每一层中插入轻量级的适配器模块（Adapter），仅优化这些模块的参数。
•	关键点：
	•	Adapter 通常是一个小的前馈网络（如瓶颈结构），其输入是主模型的激活值。
	•	通过 Adapter 学习任务特定表示，而冻结主模型参数。

•	优点：
	•	支持多任务学习（不同任务使用不同的 Adapter）。
	•	模型共享性好。

•	应用场景：
	•	多任务学习和领域适配（Domain Adaptation）。

##### 1.1 LoRA 低秩分解，适合减少参数量
LoRA 就是 Adapter 技术的一个变种，它通过 低秩矩阵分解 来实现参数高效微调，将原始模型权重的变化限制在一个 低秩子空间 内，从而减少训练过程中需要更新的参数量。
LoRA (Low-Rank Adaptation of Large Language Models)是其中的一种。
LoRA 通过在模型的权重矩阵上添加低秩矩阵的参数分解，只更新额外引入的矩阵参数，从而减少训练开销。

•	权重 $W$ 被分解为两部分：$W = W_0 + A \times B$，其中 $A$ 和 $B$ 是低秩矩阵。
•	预训练权重 $W_0$ 被固定，仅优化 $A$ 和 $B$。

优点：
	•	参数开销小（仅需存储 $A$ 和 $B$）。
	•	性能与全量微调接近。

应用场景：
	•	大语言模型（如 GPT、BERT）的高效微调。

矩阵的秩是衡量其行或列之间线性独立性的一个指标，也可以理解为矩阵中存储的有效信息的数量。
低秩矩阵是一个秩较低的矩阵，相对于它的尺寸而言，其行或列并不完全独立。它可以被分解成多个小型矩阵的乘积，从而有效压缩表示。

LoRA 使用低秩矩阵分解来高效地调整模型权重：
	•	假设一个权重矩阵 $W \in \mathbb{R}^{d \times k}$ 需要微调。
	•	LoRA 使用低秩矩阵 $A \in \mathbb{R}^{d \times r}$ 和 $B \in \mathbb{R}^{r \times k}$ 表示权重更新 $\Delta W$：

\Delta W = A \cdot B

	•	这里，$r \ll d, k$，因此只需要训练 $A$ 和 $B$，大大减少参数量。

例如：
	•	如果 $d = 1024$，$k = 1024$，而 $r = 8$，全矩阵的参数量为 $1024 \times 1024 = 1,048,576$，而低秩分解后的参数量仅为 $1024 \times 8 + 8 \times 1024 = 16,384$，减少了 98.4%。

局限性：

- 1.	信息损失：
	•	低秩分解是近似的，对于某些复杂任务或强领域偏移的场景，可能会丢失关键信息。
- 2.	适用性依赖分布：
	•	低秩假设对某些高度非线性或高维特征分布的任务可能不成立。
- 3.	秩选择困难：
	•	需要实验确定合适的秩 $r$，否则可能导致性能下降。

##### 1.2  Prefix-Tuning 对输入或中间前缀向量进行微调

在输入序列前添加一段可学习的前缀向量，仅优化这些向量以适应目标任务。

•	关键点：
	•	在 Transformer 的输入和注意力层前注入固定长度的“前缀”。
	•	不调整模型权重，只调整前缀向量的参数。
•	优点：
	•	不影响预训练模型本体。
	•	非常高效，尤其适合文本生成任务。
•	应用场景：
	•	任务包括文本生成、对话系统和机器翻译。

##### 1.3 BitFit (Bias Tuning)仅微调偏置项

只调整预训练模型中的偏置参数（bias），而冻结所有其他权重。
•	关键点：
	•	偏置参数的数量相比全模型参数少得多，因此优化负担小。
•	优点：
	•	极低的参数更新量。
	•	对许多任务表现良好。
•	应用场景：
	•	小规模任务、资源受限场景。

##### 1.4 Prompt-Tuning 对输入或中间前缀向量进行微调

直接优化输入的可学习提示（prompt），而不调整模型权重。
•	关键点：
	•	用一组可学习的嵌入替代静态的任务描述。
	•	Prompt 嵌入被训练来优化特定任务。

•	优点：
	•	数据量需求低。
	•	非侵入性，对预训练模型完全无影响。
•	应用场景：
	•	Few-shot 和 Zero-shot 任务。

还有包括：

5. 选择性层调整（Selective Layer Tuning）：可以只微调层的一个子集，而不是微调模型的所有层。这减少了需要更新的参数数量。
6. 稀疏微调（Sparse Fine-Tuning）：传统的微调会略微调整所有参数，但稀疏微调只涉及更改模型参数的一个子集。这通常是基于一些标准来完成的，这些标准标识了与新任务最相关的参数。
7. 正则化技术（Regularization Techniques）：可以将正则化项添加到损失函数中，以阻止参数发生较大变化，从而以更“参数高效”的方式有效地微调模型。
8. 任务特定的头（Task-specific Heads）：有时，在预先训练的模型架构中添加一个任务特定的层或“头”，只对这个头进行微调，从而减少需要学习的参数数量。 

可以根据任务特点选择合适的方法，例如使用 LoRA 或 Adapter 微调大模型，同时显著节省计算资源。具体实现还可以结合像 Hugging Face 的 peft 库来快速上手！


### 高效微调技术存在的一些问题

当前的高效微调技术很难在类似方法之间进行直接比较并评估它们的真实性能，主要的原因如下所示：

1. 参数计算口径不一致：参数计算可以分为三类：

1) 可训练参数的数量、
2) 微调模型与原始模型相比改变的参数的数量、
3) 微调模型和原始模型之间差异的等级。
例如，DiffPruning更新0.5%的参数，但是实际参与训练的参数量是200%。这为比较带来了困难。
尽管可训练的参数量是最可靠的存储高效指标，但是也不完美。 Ladder-side Tuning使用一个单独的小网络，参数量高于LoRA或BitFit，但是因为反向传播不经过主网络，其消耗的内存反而更小。

2. 缺乏模型大小的考虑：

已有工作表明，大模型在微调中需要更新的参数量更小（无论是以百分比相对而论还是以绝对数量而论），因此（基）模型大小在比较不同PEFT方法时也要考虑到。

3. 缺乏测量基准和评价标准:

不同方法所使用的的模型/数据集组合都不一样，评价指标也不一样，难以得到有意义的结论。

4. 代码实现可读性差

很多开源代码都是简单拷贝Transformer代码库，然后进行小修小补。这些拷贝也不使用git fork，难以找出改了哪里。即便是能找到，可复用性也比较差（通常指定某个Transformer版本，没有说明如何脱离已有代码库复用这些方法）


### 高效微调技术最佳实践

针对以上存在的问题，研究高效微调技术时，建议按照最佳实践进行实施：

1. 明确指出参数数量类型。
   比如 LoRA、Adapter 和 BitFit 等方法：
   参数一般分为以下几类：
	1) 	权重参数（Weights）：模型中主要的可训练参数，通常是权重矩阵，如 Transformer 中的 W_q、W_k、W_v 和 W_o。
	2) 	偏置参数（Bias）：模型中的偏置项（如 b），与权重相比参数量少。
	3) 	Adapter 模块参数：微调过程中额外插入的小网络（如 LoRA 的低秩矩阵或瓶颈模块中的参数）。
	4) 	Prompt 参数：在 Prompt-Tuning 中引入的可学习的 嵌入向量。
	5) 	缩放参数（Scaling Factors）：如 IA3 方法中的缩放因子。
    
2. 使用不同大小的模型进行评估。
3. 和类似方法进行比较。
4. 标准化PEFT测量基准。
5. 重视代码清晰度，以最小化进行实现。