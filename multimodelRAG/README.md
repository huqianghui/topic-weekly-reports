#### 多模态文档的RAG

##### 文本和图片混合

1. PDF，word或者PPT中的图片和文字的混合。

特别是图片不仅仅需要里面的文字，还是需要里面图片

2. 文本和图片分开的，但是需要查询的时候也需要返回图片

a. 涉及到图片的处理： 使用统一的向量模型，还是分别用不同的向量模型

b. 查询时候，怎么抽取多模态的内容，怎么排序等。

c. 怎么把结果处理，同一返回给用户



##### 多模态的存储

[LanceDB-Store, query and filter vectors, metadata and multi-modal data (text, images, videos, point clouds, and more).](https://github.com/lancedb/lancedb)

[ColPALI for RAG](https://github.com/illuin-tech/colpali)

[AzureAI-ImageSearch](https://github.com/ambarishg/AzureAI-ImageSearch/blob/main/04.image_search.ipynb)

[gpt4v_multi_modal_retrieval](https://colab.research.google.com/gist/seldo/057406cf3b49a3ed41f9f17a02930996/gpt4v_multi_modal_retrieval.ipynb#scrollTo=b383f38e)