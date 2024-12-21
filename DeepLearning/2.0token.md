## tokenize分词

### 总览

在使用GPT BERT模型输入词语常常会先进行tokenize ，tokenize的目标是把输入的文本流，切分成一个个子串，每个子串相对有完整的语义，便于学习embedding表达和后续模型的使用。

tokenize有三种粒度：word/subword/char

word/词，词，是最自然的语言单元。

对于英文等自然语言来说，存在着天然的分隔符，如空格或一些标点符号等，对词的切分相对容易。

但是对于一些东亚文字包括中文来说，就需要某种分词算法才行。

- Tokenizers库中，基于规则切分部分，采用了spaCy和Moses两个库。如果基于词来做词汇表，由于长尾现象的存在，这个词汇表可能会超大。
像Transformer XL库就用到了一个26.7万个单词的词汇表。
这需要极大的embedding matrix才能存得下。embedding matrix是用于查找取用token的embedding vector的。这对于内存或者显存都是极大的挑战。常规的词汇表，一般大小不超过5万。

- char/字符，即最基本的字符，如英语中的'a','b','c'或中文中的'你'，'我'，'他'等。而一般来讲，字符的数量是少量有限的。这样做的问题是，由于字符数量太小，我们在为每个字符学习嵌入向量的时候，每个向量就容纳了太多的语义在内，学习起来非常困难。

- subword/子词级，它介于字符和单词之间。比如说'Transformers'可能会被分成'Transform'和'ers'两个部分。这个方案平衡了词汇量和语义独立性，是相对较优的方案。它的处理原则是，常用词应该保持原状，生僻词应该拆分成子词以共享token压缩空间。


### 常用的tokenize算法

| 分词方法          | 特点                     | 被提出的时间 | 典型模型          |
| ------------- | ---------------------- | ------ | ------------- |
| BPE           | 采用合并规则，可以适应未知词         | 2016年  | GPT-2、RoBERTa |
| WordPiece     | 采用逐步拆分的方法，可以适应未知词      | 2016年  | BERT          |
| Unigram LM    | 采用无序语言模型，训练速度快         | 2018年  | XLM           |
| SentencePiece | 采用汉字、字符和子词三种分词方式，支持多语言 | 2018年  | T5、ALBERT     |


最常用的三种tokenize算法：BPE（Byte-Pair Encoding)，WordPiece和SentencePiece

#### BPE（Byte-Pair Encoding)
BPE，即字节对编码。其核心思想在于将最常出现的子词对合并，直到词汇表达到预定的大小时停止。

BPE是一种基于数据压缩算法的分词方法。它通过不断地合并出现频率最高的字符或者字符组合，来构建一个词表。具体来说，BPE的运算过程如下：

- 将所有单词按照字符分解为字母序列。例如：“hello”会被分解为["h","e","l","l","o"]。
- 统计每个字母序列出现的频率，将频率最高的序列合并为一个新序列。
- 重复第二步，直到达到预定的词表大小或者无法再合并。

词表大小通常先增加后减小

每次合并后词表可能出现3种变化：

- +1，表明加入合并后的新字词，同时原来的2个子词还保留（2个字词不是完全同时连续出现）
- +0，表明加入合并后的新字词，同时原来的2个子词中一个保留，一个被消解（一个字词完全随着另一个字词的出现而紧跟着出现）
- -1，表明加入合并后的新字词，同时原来的2个子词都被消解（2个字词同时连续出现）


#### WordPiece

WordPiece，从名字好理解，它是一种子词粒度的tokenize算法subword tokenization algorithm，很多著名的Transformers模型，比如BERT/DistilBERT/Electra都使用了它。

wordpiece算法可以看作是BPE的变种。不同的是，WordPiece基于概率生成新的subword而不是下一最高频字节对。WordPiece算法也是每次从词表中选出两个子词合并成新的子词。**BPE选择频数最高的相邻子词合并，而WordPiece选择使得语言模型概率最大的相邻子词加入词表。**即它每次合并的两个字符串A和B，应该具有最大的$\frac{P(A B)}{P(A) P(B)}$值。合并AB之后，所有原来切成A+B两个tokens的就只保留AB一个token，整个训练集上最大似然变化量与$\frac{P(A B)}{P(A) P(B)}$成正比。

比如说 
P(ed)的概率比$P(e) + P(d)$ 单独出现的概率更大，可能比他们具有最大的互信息值，也就是两子词在语言模型上具有较强的关联性。

那wordPiece和BPE的区别：

BPE： apple 当词表有appl 和 e的时候，apple优先编码为 appl和e（即使原始预料中 app 和 le 的可能性更大）
wordPiece：根据原始语料， app和le的概率更大

#### Unigram
与BPE或者WordPiece不同，Unigram的算法思想是从一个巨大的词汇表出发，再逐渐删除trim down其中的词汇，直到size满足预定义。

初始的词汇表可以采用所有预分词器分出来的词，再加上所有高频的子串。

每次从词汇表中删除词汇的原则是使预定义的损失最小。训练时，计算loss的公式为：

假设训练文档中的所有词分别为$x_{1} ; x_{2}, \ldots, x_{N}$，而每个词tokenize的方法是一个集合$S\left(x_{i}\right)$

当一个词汇表确定时，每个词tokenize的方法集合$S\left(x_{i}\right)$就是确定的，而每种方法对应着一个概率$P(x)$.

如果从词汇表中删除部分词，则某些词的tokenize的种类集合就会变少，log( *)中的求和项就会减少，从而增加整体loss。

Unigram算法每次会从词汇表中挑出使得loss增长最小的10%~20%的词汇来删除。

一般Unigram算法会与SentencePiece算法连用。

#### SentencePiece
SentencePiece，顾名思义，它是把一个句子看作一个整体，再拆成片段，而没有保留天然的词语的概念。一般地，它把空格space也当作一种特殊字符来处理，再用BPE或者Unigram算法来构造词汇表。

比如，XLNetTokenizer就采用了_来代替空格，解码的时候会再用空格替换回来。

目前，Tokenizers库中，所有使用了SentencePiece的都是与Unigram算法联合使用的，比如ALBERT、XLNet、Marian和T5.

