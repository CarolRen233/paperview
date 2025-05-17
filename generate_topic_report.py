import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
from openai import OpenAI
import time
import jinja2
import math

# 配置参数
data_path = './results/PPCF1681/PPCF1681_replaced_synonyms.csv'
output_html = './results/PPCF1681/PPCF1681_topic_report.html'
num_clusters = 30
recent_years = 5

# 1. 读取数据
print('读取数据...')
df = pd.read_csv(data_path, low_memory=False)
print(f'共读取{len(df)}条文献')

# 2. 预处理：只保留有标题的论文
df = df.dropna(subset=['文献标题'])

# 3. 标题向量化
print('加载句向量模型...')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
titles = df['文献标题'].astype(str).tolist()
print('计算标题向量...')
title_vecs = model.encode(titles, show_progress_bar=True)

# 4. KMeans聚类
print(f'对所有年份论文聚类为{num_clusters}类...')
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(title_vecs)
df['cluster_all'] = labels

# 5. 统计每类数量，找出前5大类
cluster_counts = df['cluster_all'].value_counts().sort_values(ascending=False)
top5_clusters = cluster_counts.head(5).index.tolist()

# 6. 收集每类论文
all_clusters_papers = {cid: df[df['cluster_all'] == cid] for cid in top5_clusters}

# 7. 最近5年论文筛选
this_year = datetime.now().year
recent_df = df[df['年份'].apply(lambda x: str(x).isdigit() and int(x) >= this_year - recent_years + 1)]

# 8. 最近5年聚类
if len(recent_df) > 0:
    print(f'对最近{recent_years}年论文聚类为{num_clusters}类...')
    recent_titles = recent_df['文献标题'].astype(str).tolist()
    recent_vecs = model.encode(recent_titles, show_progress_bar=True)
    kmeans_recent = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels_recent = kmeans_recent.fit_predict(recent_vecs)
    recent_df['cluster_recent'] = labels_recent
    cluster_counts_recent = recent_df['cluster_recent'].value_counts().sort_values(ascending=False)
    top5_clusters_recent = cluster_counts_recent.head(5).index.tolist()
    recent_clusters_papers = {cid: recent_df[recent_df['cluster_recent'] == cid] for cid in top5_clusters_recent}
else:
    print('最近5年无论文数据！')
    recent_clusters_papers = {}

# 9. 保存中间结果，便于后续deepseek和HTML渲染
import pickle
with open('./results/PPCF1681/cluster_results.pkl', 'wb') as f:
    pickle.dump({
        'all_clusters_papers': all_clusters_papers,
        'recent_clusters_papers': recent_clusters_papers,
        'df': df,
        'recent_df': recent_df
    }, f)

print('数据聚类与统计完成，已保存中间结果。')

# ========== 10. DeepSeek主题总结 ========== #
DEEPSEEK_API_KEY = 'sk-7b20f5557661403aaaea619acdf5d85c'
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def deepseek_summarize(papers_df):
    # 拼接标题、摘要、关键词
    texts = []
    for _, row in papers_df.iterrows():
        title = str(row.get('文献标题', ''))
        abstract = str(row.get('摘要', ''))
        keywords = str(row.get('作者关键字', ''))
        texts.append(f"标题: {title}\n摘要: {abstract}\n关键词: {keywords}")
    prompt = '''
请根据以下论文的标题、摘要和关键词，总结该类别的：
1. 主题名称（10字以内，简明扼要，适合做标题）
2. 主要关键词
3. 研究重点
4. 使用的技术
5. 典型应用
6. 尚待解决的问题
请严格分条输出：\n主题名称：...\n主要关键词：...\n研究重点：...\n使用的技术：...\n典型应用：...\n尚待解决的问题：...\n
论文内容如下：\n''' + "\n\n".join(texts[:30])  # 只取前30篇，防止超长
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek API error: {e}")
        return "[DeepSeek API调用失败]"

# ========== 11. 统计高被引论文和关键词 ========== #
def get_top_cited_papers(papers_df, topn=5):
    # cite字段可能叫施引文献或Cited_By
    for col in ['施引文献', 'Cited_By', '被引量', 'Cite', 'Citations']:
        if col in papers_df.columns:
            cite_col = col
            break
    else:
        cite_col = None
    if cite_col:
        papers_df = papers_df.copy()
        papers_df[cite_col] = pd.to_numeric(papers_df[cite_col], errors='coerce').fillna(0)
        return papers_df.sort_values(cite_col, ascending=False).head(topn)
    else:
        return papers_df.head(topn)

def get_top_keywords_and_cooccurrence(papers_df, topn=10):
    # 首先按引用量选择前30篇论文
    cite_cols = ['施引文献', 'Cited_By', '被引量', 'Cite', 'Citations']
    cite_col = next((col for col in cite_cols if col in papers_df.columns), None)
    
    if cite_col:
        # 使用.copy()避免SettingWithCopyWarning
        papers_df_copy = papers_df.copy()
        papers_df_copy[cite_col] = pd.to_numeric(papers_df_copy[cite_col], errors='coerce').fillna(0)
        # 选择引用量最高的至多30篇论文
        top_cited_papers = papers_df_copy.nlargest(30, cite_col)
    else:
        # 如果没有引用量列，就直接取前30篇
        top_cited_papers = papers_df.head(30)
    
    # 统计高频关键词及共现关系
    from collections import Counter, defaultdict
    import re
    all_keywords = []
    cooccur = defaultdict(Counter)
    for _, row in top_cited_papers.iterrows():
        kws = []
        for col in ['作者关键字', '索引关键字']:
            if col in top_cited_papers.columns and isinstance(row[col], str) and row[col].strip():
                kws += [k.strip().lower() for k in re.split(r'[;,，；]', row[col]) if k.strip()]
        kws = list(set(kws))
        all_keywords += kws
        # 共现统计
        for i, k1 in enumerate(kws):
            for j, k2 in enumerate(kws):
                if i != j:
                    cooccur[k1][k2] += 1
    kw_counter = Counter(all_keywords)
    top_keywords = [k for k, _ in kw_counter.most_common(topn)]
    return top_keywords, kw_counter, cooccur

def deepseek_translate(words):
    # 用deepseek翻译关键词为中文
    prompt = '请将以下英文关键词翻译为中文，按顺序输出，用逗号分隔：' + ', '.join(words)
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"DeepSeek翻译API error: {e}")
        return ''

def build_keyword_graph_data(center, top_keywords, cooccur, kw_counter):
    # 生成Echarts节点和边，中心节点为center，相关节点为top_keywords
    nodes = [{
        'id': center,
        'name': center,
        'symbolSize': 60,
        'value': kw_counter[center] if center in kw_counter else 1,
        'category': 0,
        'itemStyle': {'color': '#FF0000'}
    }]
    # 计算最大最小频率
    freqs = [kw_counter[kw] for kw in top_keywords if kw != center]
    minf, maxf = min(freqs) if freqs else 1, max(freqs) if freqs else 1
    min_size, max_size = 30, 60
    for kw in top_keywords:
        if kw == center:
            continue
        freq = kw_counter[kw]
        # 线性缩放到[min_size, max_size]
        if maxf > minf:
            size = min_size + (freq - minf) * (max_size - min_size) / (maxf - minf)
        else:
            size = (min_size + max_size) / 2
        nodes.append({
            'id': kw,
            'name': kw,
            'symbolSize': size,
            'value': int(freq),
            'category': 1,
            'itemStyle': {'color': '#3399FF'}
        })
    links = []
    for kw in top_keywords:
        if kw == center:
            continue
        links.append({
            'source': center,
            'target': kw,
            'value': cooccur[center][kw] if center in cooccur and kw in cooccur[center] else 1
        })
    # 相关节点之间的共现边
    for i, k1 in enumerate(top_keywords):
        for j, k2 in enumerate(top_keywords):
            if i < j and k1 != center and k2 != center:
                v = cooccur[k1][k2]
                if v > 0:
                    links.append({'source': k1, 'target': k2, 'value': v})
    return nodes, links

def build_topic_obj(papers_df, topic_type, topic_idx, topic_title):
    # topic_type: '所有年份' or '最近5年'
    # topic_idx: 1-based
    summary = deepseek_summarize(papers_df)
    parsed = parse_deepseek_result(summary)
    # 主题名
    display_title = f"{topic_type}主题{topic_idx}：{parsed.get('topic_title', topic_title)}"
    # 关键词统计与翻译
    top_keywords, kw_counter, cooccur = get_top_keywords_and_cooccurrence(papers_df, 10)
    cn_keywords = deepseek_translate(top_keywords)
    # 论文数
    paper_count = len(papers_df)
    # 关键词可视化
    nodes, edges = build_keyword_graph_data(top_keywords[0] if top_keywords else display_title, top_keywords, cooccur, kw_counter)
    # APA论文
    top_papers = get_top_cited_papers(papers_df, 5)
    top_papers_apa = [build_apa(row) for _, row in top_papers.iterrows()]
    return {
        'display_title': display_title,
        'keywords': ', '.join(top_keywords),
        'keywords_cn': cn_keywords,
        'focus': parsed.get('focus', ''),
        'tech': parsed.get('tech', ''),
        'app': parsed.get('app', ''),
        'problem': parsed.get('problem', ''),
        'paper_count': paper_count,
        'kw_nodes': nodes,
        'kw_edges': edges,
        'top_papers_apa': top_papers_apa
    }

# ========== 13. Jinja2渲染HTML ========== #
TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>{{ topic_name }}主题分析报告</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <style>body{font-family:sans-serif;max-width:1200px;margin:auto;}h2{margin-top:2em;} .apa{margin-bottom:8px;}</style>
</head>
<body>
<h1>{{ topic_name }}主题分析报告</h1>

<h2>所有年份主题总结：</h2>
{% for topic in all_topics %}
<h3>{{ topic.display_title }}</h3>
<div><b>本主题包含论文：</b>{{ topic.paper_count }} 篇</div>
<div><b>关键词：</b>{{ topic.keywords }}</div>
<div><b>关键词中文：</b>{{ topic.keywords_cn }}</div>
<div><b>研究重点：</b>{{ topic.focus }}</div>
<div><b>使用的技术：</b>{{ topic.tech }}</div>
<div><b>典型应用：</b>{{ topic.app }}</div>
<div><b>尚待解决的问题：</b>{{ topic.problem }}</div>
<div id="all_kw_graph_{{ loop.index0 }}" style="width: 800px; height: 400px;"></div>
<script>
var chart = echarts.init(document.getElementById('all_kw_graph_{{ loop.index0 }}'));
var option = {
    title: {text: '关键词关系图'},
    tooltip: {},
    legend: [{data:['中心关键词','相关关键词']}],
    series: [{
        type: 'graph',
        layout: 'force',
        roam: true,
        label: {show: true},
        data: {{ topic.kw_nodes|safe }},
        edges: {{ topic.kw_edges|safe }},
        categories: [{name:'中心关键词'},{name:'相关关键词'}],
        force: {repulsion: 400, edgeLength: 120}
    }]
};
chart.setOption(option);
</script>
<div><b>引用量最高的5篇文章：</b><ul>
{% for apa in topic.top_papers_apa %}
<li class="apa">{{ apa|safe }}</li>
{% endfor %}
</ul></div>
{% endfor %}

<h2>最近5年主题总结：</h2>
{% for topic in recent_topics %}
<h3>{{ topic.display_title }}</h3>
<div><b>本主题包含论文：</b>{{ topic.paper_count }} 篇</div>
<div><b>关键词：</b>{{ topic.keywords }}</div>
<div><b>关键词中文：</b>{{ topic.keywords_cn }}</div>
<div><b>研究重点：</b>{{ topic.focus }}</div>
<div><b>使用的技术：</b>{{ topic.tech }}</div>
<div><b>典型应用：</b>{{ topic.app }}</div>
<div><b>尚待解决的问题：</b>{{ topic.problem }}</div>
<div id="recent_kw_graph_{{ loop.index0 }}" style="width: 800px; height: 400px;"></div>
<script>
var chart = echarts.init(document.getElementById('recent_kw_graph_{{ loop.index0 }}'));
var option = {
    title: {text: '关键词关系图'},
    tooltip: {},
    legend: [{data:['中心关键词','相关关键词']}],
    series: [{
        type: 'graph',
        layout: 'force',
        roam: true,
        label: {show: true},
        data: {{ topic.kw_nodes|safe }},
        edges: {{ topic.kw_edges|safe }},
        categories: [{name:'中心关键词'},{name:'相关关键词'}],
        force: {repulsion: 400, edgeLength: 120}
    }]
};
chart.setOption(option);
</script>
<div><b>引用量最高的5篇文章：</b><ul>
{% for apa in topic.top_papers_apa %}
<li class="apa">{{ apa|safe }}</li>
{% endfor %}
</ul></div>
{% endfor %}

</body>
</html>
'''

# ========== 14. 汇总并渲染 ========== #
def parse_deepseek_result(text):
    # 解析主题名称和各项内容
    result = {'topic_title': '', 'keywords': '', 'focus': '', 'tech': '', 'app': '', 'problem': ''}
    import re
    patterns = {
        'topic_title': r'主题名称[:：]\s*(.*)',
        'keywords': r'主要关键词[:：]\s*(.*)',
        'focus': r'研究重点[:：]\s*(.*)',
        'tech': r'使用的技术[:：]\s*(.*)',
        'app': r'典型应用[:：]\s*(.*)',
        'problem': r'尚待解决的问题[:：]\s*(.*)'
    }
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[k] = m.group(1).strip()
    return result

def build_apa(row):
    # 作者, 年份, 标题, 期刊, 卷(期), 页码, DOI, 引用量
    authors = row.get('作者', '') or row.get('Author full names', '')
    year = str(row.get('年份', ''))
    title = row.get('文献标题', '')
    journal = row.get('来源出版物名称', '')
    vol = str(row.get('卷', ''))
    issue = str(row.get('期', ''))
    pages = f"{row.get('起始页码', '')}-{row.get('结束页码', '')}" if row.get('起始页码', '') and row.get('结束页码', '') else ''
    doi = row.get('DOI', '')
    
    # Find the citation count column
    cite_cols = ['施引文献', 'Cited_By', '被引量', 'Cite', 'Citations']
    cite_col = next((col for col in cite_cols if col in row.index), None)
    citations = row.get(cite_col, 0) if cite_col else 0
    
    apa = f"{authors} ({year}). {title}. <i>{journal}</i>, {vol}({issue}), {pages}. <a href='https://doi.org/{doi}' target='_blank'>https://doi.org/{doi}</a> <b>[引用量: {citations}]</b>"
    return apa

# 主题名
while True:
    topic_name = input("请输入本主题的名称（用于报告标题）: ").strip()
    if topic_name:
        break
    print("主题名称不能为空，请重新输入。")

# 修改输出HTML路径以包含用户输入的主题名
output_html = f'./results/{topic_name}_topic_report.html'

# 所有年份主题
all_topic_names = [f'主题{idx+1}' for idx in top5_clusters]
all_topic_counts = [len(all_clusters_papers[cid]) for cid in top5_clusters]
all_topics = []
for i, cid in enumerate(top5_clusters):
    print(f'正在总结所有年份主题{cid}...')
    all_topics.append(build_topic_obj(all_clusters_papers[cid], '所有年份', i+1, all_topic_names[i]))
    time.sleep(2)

# 最近5年主题
recent_topics = []
if recent_clusters_papers:
    for i, cid in enumerate(top5_clusters_recent):
        print(f'正在总结最近5年主题{cid}...')
        recent_topics.append(build_topic_obj(recent_clusters_papers[cid], '最近5年', i+1, all_topic_names[i] if i < len(all_topic_names) else ''))
        time.sleep(2)

# 渲染HTML
env = jinja2.Environment()
tmpl = env.from_string(TEMPLATE)
html = tmpl.render(
    topic_name=topic_name,
    all_topic_names=all_topic_names,
    all_topic_counts=all_topic_counts,
    all_topics=all_topics,
    recent_topics=recent_topics
)
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'HTML报告已生成：{output_html}') 