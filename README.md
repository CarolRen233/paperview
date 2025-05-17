# 文献主题分析工具

这个工具可以分析CSV格式的文献数据集，自动识别主要研究主题，并为每个主题找出最相关的论文。

## 功能

- 从CSV文件读取文献数据
- 使用LDA（潜在狄利克雷分配）算法进行主题建模
- 为每个主题识别关键词
- 为每个主题找出最相关的3篇论文
- 生成主题分布可视化图像
- 保存分析结果到CSV文件

## 使用方法

1. 确保安装了所需的Python库：

```
pip install pandas numpy scikit-learn matplotlib
```

2. 运行分析脚本：

```
python topic_analysis.py
```

3. 查看分析结果：
   - 控制台将显示每个主题的关键词和相关论文
   - 结果文件保存在 `./results/PPCF1681_replaced_synonyms/` 目录下
   - 主题分析结果：`topic_analysis_results.csv`
   - 主题分布图：`topic_distribution.png`

## 自定义

如需分析其他文献数据集或调整分析参数，可以修改 `topic_analysis.py` 文件：

- 更改输入文件路径：修改 `file_path` 变量
- 调整主题数量：修改 `n_topics` 变量
- 修改TF-IDF参数：调整 `TfidfVectorizer` 的参数
- 修改每个主题显示的论文数量：调整 `get_top_papers_for_topic` 函数的 `n` 参数

## 文件格式要求

输入CSV文件应包含以下列：
- `文献标题`：论文标题
- `摘要`：论文摘要
- `作者`：作者信息
- `发表年`：发表年份
- `被引频次`：引用次数

## 注意事项

- 脚本默认使用英文停用词，适用于英文文献分析
- 确保输入CSV文件使用UTF-8编码
- 主题数量可根据具体需求调整，默认为8个主题



```
对提供的论文列表（file\_path = './results/PPCF1681/PPCF1681\_replaced\_synonyms.csv'）进行深入分析（你不必阅读这里面的内容，我只需要告诉你这个.csv的head包含作者	Author full names	作者 ID	文献标题	年份	来源出版物名称	卷	期	论文编号	起始页码	结束页码	页码计数	施引文献	DOI	链接	归属机构	带归属机构的作者	摘要	作者关键字	索引关键字	通讯地址	文献类型	出版阶段	开放获取	来源出版物	EID
），根据论文的标题进行聚类，分两个进行聚类：

1. 聚类所有的论文，聚类的类别是30个，然后统计这30个类别分别涉及的论文数量，找出论文数量最多的5个类别，并找到每个类别涉及的所有论文

2. 聚类最近5年的论文，聚类的类别是30个，然后统计这30个类别分别涉及的论文数量，找出论文数量最多的5个类别，并找到每个类别涉及的所有论文

然后分别对以上类别用deepseek进行总结主题，并且根据每个类别涉及到的论文的标题和摘要，还有关键词，总结出此类别的研究重点、使用的技术、典型应用、尚待解决的问题

然后找到此类别中引用量最高的5篇文章

最后用@[https://echarts.apache.org/examples/en/editor.html?c=graph-label-overlap](https://echarts.apache.org/examples/en/editor.html?c=graph-label-overlap) 这个风格将这个类别中的cite量前20的文章的所有关键词进行可视化

最终形成一个html的文件，这个文件里面的内容应当是这样的：

# \[论文话题]主题分析报告（\[论文话题]是这个输入文档的目录的名字）

## 所有年份主题总结：

每个主题涉及的论文数量的图表，参考@[https://echarts.apache.org/examples/en/editor.html?c=dataset-encode0](https://echarts.apache.org/examples/en/editor.html?c=dataset-encode0)

### 1.  所有年份主题1

关键词：....
研究重点：....
使用的技术：
典型应用：
尚待解决的问题：

关键词可视化（将这个类别中的cite量前20的文章的所有关键词进行可视化，用@[https://echarts.apache.org/examples/en/editor.html?c=graph-label-overlap](https://echarts.apache.org/examples/en/editor.html?c=graph-label-overlap) ）

此类别中引用量最高的5篇文章列表，包含标题，作者，cite，doi链接

### 2.  所有年份主题2

### 3.  所有年份主题3

### 4.  所有年份主题4

### 5.  所有年份主题5

## 最近5年主题总结：

一样的，以此类推

```
