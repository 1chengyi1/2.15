import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from zhipuai import ZhipuAI
import os
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

# 配置页面
st.set_page_config(
    page_title="科研诚信风险预警平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式中新增/修改按钮样式部分
st.markdown("""
<style>
/* 统一按钮样式（高级感天青色系） */
.stButton>button, .stDownloadButton>button {  /* 新增 .stDownloadButton>button */
    background-color: #4a90e2;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.2);
    transition: all 0.3s ease;
}

.stButton>button:hover, .stDownloadButton>button:hover {  /* 新增悬停状态 */
    background-color: #357abd;
    box-shadow: 0 6px 18px rgba(74, 144, 226, 0.3);
    transform: translateY(-1px);
}

.stButton>button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 6px rgba(74, 144, 226, 0.4);
}

/* 侧边栏按钮样式调整（与主按钮统一） */
.sidebar .stButton>button {
    background-color: #4a90e2;
    box-shadow: none;
    padding: 8px 16px;
}

.sidebar .stButton>button:hover {
    background-color: #357abd;
}
.main {
    max-width: 90%;
    margin: 0 auto; /* 水平居中 */
}

/* 调整侧边栏宽度（可选，保持与主内容比例协调） */
.sidebar .sidebar-content {
    max-width: 280px; /* 适当缩小侧边栏宽度 */
}

/* 确保宽屏设备下内容不溢出 */
.stApp {
    padding: 20px; /* 增加内边距提升舒适感 */
}

/* 表格滚动条优化 */
.scrollable-table {
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# 初始化智谱API
client = ZhipuAI(api_key="89c41de3c3a34f62972bc75683c66c72.ZGwzmpwgMfjtmksz")

# 数据预处理和风险值计算模块
@st.cache_data(show_spinner=False)
def process_risk_data():
    # 不端原因严重性权重
    misconduct_weights = {
        '伪造、篡改图片': 6,
        '篡改图片': 3,
        '篡改数据': 3,
        '篡改数据、图片': 6,
        '编造研究过程': 4,
        '编造研究过程、不当署名': 7,
        '篡改数据、不当署名': 6,
        '伪造通讯作者邮箱': 2,
        '实验流程不规范': 2,
        '数据审核不严': 2,
        '署名不当、实验流程不规范': 5,
        '篡改数据、代写代投、伪造通讯作者邮箱、不当署名': 13,
        '篡改数据、伪造通讯作者邮箱、不当署名': 8,
        '第三方代写、伪造通讯作者邮箱': 7,
        '第三方代写代投、伪造数据': 8,
        '一稿多投': 2,
        '第三方代写代投、伪造数据、一稿多投': 10,
        '篡改数据、剽窃': 8,
        '伪造图片': 3,
        '伪造图片、不当署名': 6,
        '委托实验、不当署名': 6,
        '伪造数据': 3,
        '伪造数据、篡改图片': 6,
        '伪造数据、不当署名、伪造通讯作者邮箱等': 8,
        '伪造数据、一图多用、伪造图片、代投问题': 14,
        '伪造数据、署名不当': 6,
        '抄袭剽窃他人项目申请书内容': 6,
        '伪造通讯作者邮箱、篡改数据和图片': 8,
        '篡改数据、不当署名': 6,
        '抄袭他人基金项目申请书': 6,
        '结题报告中存在虚假信息': 5,
        '抄袭剽窃': 5,
        '造假、抄袭': 5,
        '第三方代写代投': 5,
        '署名不当': 3,
        '第三方代写代投、署名不当': 8,
        '抄袭剽窃、伪造数据': 8,
        '买卖图片数据': 3,
        '买卖数据': 3,
        '买卖论文': 5,
        '买卖论文、不当署名': 8,
        '买卖论文数据': 8,
        '买卖论文数据、不当署名': 11,
        '买卖图片数据、不当署名': 6,
        '图片不当使用、伪造数据': 6,
        '图片不当使用、数据造假、未经同意使用他人署名': 9,
        '图片不当使用、数据造假、未经同意使用他人署名、编造研究过程': 13,
        '图片造假、不当署名': 9,
        '图片造假、不当署名、伪造通讯作者邮箱等': 11,
        '买卖数据、不当署名': 6,
        '伪造论文、不当署名': 6,
        '其他轻微不端行为': 1
    }

    # 读取原始数据
    papers_df = pd.read_excel('实验数据.xlsx', sheet_name='论文')
    projects_df = pd.read_excel('实验数据.xlsx', sheet_name='项目')

    # 网络构建函数
    @st.cache_resource(show_spinner=False)
    def build_networks(papers, projects):
        # 作者-论文网络
        G_papers = nx.Graph()
        for _, row in papers.iterrows():
            authors = [row['姓名']]
            weight = misconduct_weights.get(row['不端原因'], 1)
            G_papers.add_edge(row['姓名'], row['不端内容'], weight=weight)

        # 作者-项目网络
        G_projects = nx.Graph()
        for _, row in projects.iterrows():
            authors = [row['姓名']]
            weight = misconduct_weights.get(row['不端原因'], 1)
            G_projects.add_edge(row['姓名'], row['不端内容'], weight=weight)

        # 作者-作者网络
        G_authors = nx.Graph()

        # 共同项目/论文连接
        for df in [papers, projects]:
            for _, row in df.iterrows():
                authors = [row['姓名']]
                weight = misconduct_weights.get(row['不端原因'], 1)
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        if G_authors.has_edge(authors[i], authors[j]):
                            G_authors[authors[i]][authors[j]]['weight'] += weight
                        else:
                            G_authors.add_edge(authors[i], authors[j], weight=weight)

        # 研究方向相似性连接
        research_areas = papers.groupby('姓名')['研究方向'].apply(lambda x: ' '.join(x)).reset_index()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(research_areas['研究方向'])
        similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

        for i in range(len(research_areas)):
            for j in range(i + 1, len(research_areas)):
                if similarity_matrix[i, j] > 0.7:
                    a1 = research_areas.iloc[i]['姓名']
                    a2 = research_areas.iloc[j]['姓名']
                    G_authors.add_edge(a1, a2, weight=similarity_matrix[i, j], reason='研究方向相似')

        # 共同机构连接
        institution_map = papers.set_index('姓名')['研究机构'].to_dict()
        for a1 in institution_map:
            for a2 in institution_map:
                if a1 != a2 and institution_map[a1] == institution_map[a2]:
                    G_authors.add_edge(a1, a2, weight=1, reason='研究机构相同')

        return G_authors

    # Word2Vec（Skip-gram）模型定义
    class SkipGramModel(nn.Module):
        def __init__(self, vocab_size, embedding_size):
            super(SkipGramModel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_size)
            self.out = nn.Linear(embedding_size, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs)
            outputs = self.out(embeds)
            return outputs

    # 数据集定义
    class SkipGramDataset(Dataset):
        def __init__(self, walks, node2id):
            self.walks = walks
            self.node2id = node2id

        def __len__(self):
            return len(self.walks)

        def __getitem__(self, idx):
            walk = self.walks[idx]
            input_ids = [self.node2id[node] for node in walk[:-1]]
            target_ids = [self.node2id[node] for node in walk[1:]]
            return torch.tensor(input_ids), torch.tensor(target_ids)

    # DeepWalk实现
    @st.cache_resource(show_spinner=False)
    def deepwalk(_graph, walk_length=30, num_walks=100, embedding_size=64):
        graph = _graph
        walks = []
        nodes = list(graph.nodes())

        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [str(node)]
                current = node
                for _ in range(walk_length - 1):
                    neighbors = list(graph.neighbors(current))
                    if neighbors:
                        current = random.choice(neighbors)
                        walk.append(str(current))
                    else:
                        break
                walks.append(walk)

        # 构建节点到ID的映射
        node2id = {node: idx for idx, node in enumerate(set([node for walk in walks for node in walk]))}
        id2node = {idx: node for node, idx in node2id.items()}

        # 构建数据集
        dataset = SkipGramDataset(walks, node2id)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 模型初始化
        model = SkipGramModel(len(node2id), embedding_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        for epoch in range(3):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, len(node2id)), targets.view(-1))
                loss.backward()
                optimizer.step()

        # 获取嵌入
        embeddings = {}
        with torch.no_grad():
            for node, idx in node2id.items():
                embeddings[node] = model.embeddings(torch.tensor([idx])).squeeze().numpy()

        return embeddings

    # 执行计算流程
    with st.spinner('正在构建合作网络...'):
        G_authors = build_networks(papers_df, projects_df)

    with st.spinner('正在训练DeepWalk模型...'):
        embeddings = deepwalk(G_authors)

    with st.spinner('正在计算风险指标...'):
        # 构建分类数据集
        X, y = [], []
        for edge in G_authors.edges():
            X.append(np.concatenate([embeddings[edge[0]], embeddings[edge[1]]]))
            y.append(1)

        non_edges = list(nx.non_edges(G_authors))
        non_edges = random.sample(non_edges, len(y))
        for edge in non_edges:
            X.append(np.concatenate([embeddings[edge[0]], embeddings[edge[1]]]))
            y.append(0)

        # 训练分类器
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)

        # 计算节点风险值
        risk_scores = {node: np.linalg.norm(emb) for node, emb in embeddings.items()}

    risk_df = pd.DataFrame({
        '作者': list(risk_scores.keys()),
        '风险值': list(risk_scores.values())
    })
    risk_df.to_parquet('risk_scores.parquet', engine='pyarrow')
    return risk_df, papers_df, projects_df

# 调用智谱大模型进行评价
def get_zhipu_evaluation(selected, paper_records, project_records, related_people):
    # 构建输入文本
    related_people_str = ", ".join(related_people) if related_people else "无"
    input_text = f"请对科研人员 {selected} 进行评价，其论文不端记录为：{paper_records.to_csv(sep=chr(9), na_rep='nan')}，项目不端记录为：{project_records.to_csv(sep=chr(9), na_rep='nan')}。同时，请提及国家的一些科研诚信政策，并列举出与 {selected} 有关的一些人（{related_people_str}）。"
    try:
        response = client.chat.completions.create(
            model="glm-4v-plus",
            messages=[{"role": "user", "content": input_text}]
        )
        # 检查响应是否成功
        if response:
            return response.choices[0].message.content
        else:
            return f"请求失败，可能是网络问题或API调用异常"
    except Exception as e:
        return f"发生异常：{str(e)}"

# 分页显示表格
def show_paginated_table(df, page_size=10, key="pagination"):
    total_pages = (len(df) // page_size) + 1
    page = st.number_input('选择页码', 1, total_pages, 1, key=key)
    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(df.iloc[start:end], use_container_width=True)


# 初始化会话状态
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'search_name' not in st.session_state:
    st.session_state.search_name = ''
if 'search_institution' not in st.session_state:
    st.session_state.search_institution = ''
if 'search_button_clicked' not in st.session_state:
    st.session_state.search_button_clicked = False
if 'selected' not in st.session_state:
    st.session_state.selected = None
if 'author_risk' not in st.session_state:
    st.session_state.author_risk = None
if 'paper_records' not in st.session_state:
    st.session_state.paper_records = pd.DataFrame()
if 'project_records' not in st.session_state:
    st.session_state.project_records = pd.DataFrame()
if 'related_people' not in st.session_state:
    st.session_state.related_people = []
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None

# 侧边栏导航
with st.sidebar:
    st.title("导航")
    if st.button("🏠 首页"):
        st.session_state.page = 'home'
    if st.button("🔍 风险查询"):
        st.session_state.page = 'search'
# 主内容区域
st.markdown("<div class='navbar'><h1>科研诚信风险预警平台</h1></div>", unsafe_allow_html=True)

# 确保 risk_df 被正确加载
try:
    risk_df = pd.read_parquet('risk_scores.parquet')
    papers = pd.read_excel('C:\\Users\\86130\\Desktop\\project\\马丹薇\\实验数据.xlsx', sheet_name='论文')
    projects = pd.read_excel('C:\\Users\\86130\\Desktop\\project\\马丹薇\\实验数据.xlsx', sheet_name='项目')
except:
    with st.spinner("首次运行需要初始化数据..."):
        risk_df, papers, projects = process_risk_data()

if st.session_state.page == 'home':
    fig = go.Figure(data=[go.Scatter(
        x=risk_df['作者'],
        y=risk_df['风险值'],
        mode='markers',
        text=risk_df['风险值'],
        hoverinfo='text+x'
    )])
    fig.update_layout(
        title='部分作者风险值散点图',
        xaxis_title='作者',
        yaxis_title='风险值'
    )
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == 'search':
    # 搜索模块
    with st.container():
        st.subheader("🔍 研究人员查询")
        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            st.session_state.search_name = st.text_input("姓名", placeholder="输入研究人员姓名",
                                                         value=st.session_state.search_name)
        with col2:
            st.session_state.search_institution = st.text_input("机构", placeholder="输入研究机构",
                                                                value=st.session_state.search_institution)
        with col3:
            search_button = st.button("查询", type="primary", use_container_width=True)

    if search_button:
        if st.session_state.search_name and not st.session_state.search_institution:
            st.session_state.search_button_clicked = True
            # 只根据姓名模糊匹配
            name_candidates = risk_df[risk_df['作者'].str.contains(st.session_state.search_name)]
            paper_matches = papers[papers['姓名'].str.contains(st.session_state.search_name)]
            project_matches = projects[projects['姓名'].str.contains(st.session_state.search_name)]

            if len(paper_matches) == 0 and len(project_matches) == 0:
                st.warning("未找到匹配的研究人员")
                st.session_state.search_button_clicked = False
                st.stop()

            # 直接选择第一个匹配人员
            st.session_state.selected = name_candidates['作者'].iloc[0]

            # 获取详细信息
            st.session_state.author_risk = risk_df[risk_df['作者'] == st.session_state.selected].iloc[0]['风险值']
            st.session_state.paper_records = papers[papers['姓名'] == st.session_state.selected]
            st.session_state.project_records = projects[projects['姓名'] == st.session_state.selected]

            # 查找与查询作者有关的人
            st.session_state.related_people = papers[
                (papers['研究机构'] == papers[papers['姓名'] == st.session_state.selected]['研究机构'].iloc[0]) |
                (papers['研究方向'] == papers[papers['姓名'] == st.session_state.selected]['研究方向'].iloc[0]) |
                (papers['不端内容'] == papers[papers['姓名'] == st.session_state.selected]['不端内容'].iloc[0])
            ]['姓名'].unique()
            st.session_state.related_people = [person for person in st.session_state.related_people if
                                               person != st.session_state.selected]

        elif st.session_state.search_name and st.session_state.search_institution:
            st.session_state.search_button_clicked = True
            # 模糊匹配
            name_candidates = risk_df[risk_df['作者'].str.contains(st.session_state.search_name)]
            paper_matches = papers[papers['姓名'].str.contains(st.session_state.search_name) & papers[
                '研究机构'].str.contains(st.session_state.search_institution)]
            project_matches = projects[projects['姓名'].str.contains(st.session_state.search_name) & projects[
                '研究机构'].str.contains(st.session_state.search_institution)]

            if len(paper_matches) == 0 and len(project_matches) == 0:
                st.warning("未找到匹配的研究人员")
                st.session_state.search_button_clicked = False
                st.stop()

            # 直接选择第一个匹配人员
            st.session_state.selected = name_candidates['作者'].iloc[0]

            # 获取详细信息
            st.session_state.author_risk = risk_df[risk_df['作者'] == st.session_state.selected].iloc[0]['风险值']
            st.session_state.paper_records = papers[papers['姓名'] == st.session_state.selected]
            st.session_state.project_records = projects[projects['姓名'] == st.session_state.selected]

            # 查找与查询作者有关的人
            st.session_state.related_people = papers[
                (papers['研究机构'] == papers[papers['姓名'] == st.session_state.selected]['研究机构'].iloc[0]) |
                (papers['研究方向'] == papers[papers['姓名'] == st.session_state.selected]['研究方向'].iloc[0]) |
                (papers['不端内容'] == papers[papers['姓名'] == st.session_state.selected]['不端内容'].iloc[0])
            ]['姓名'].unique()
            st.session_state.related_people = [person for person in st.session_state.related_people if
                                               person != st.session_state.selected]

        elif not st.session_state.search_name and st.session_state.search_institution:
            st.warning("不支持这种检索，请输入研究人员姓名进行查询。")
            st.session_state.search_button_clicked = False
            st.stop()

    if st.session_state.search_button_clicked:
        # 研究人员基本信息卡片（保持原有位置，位于核心指标之前）
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 2, 2])
            with col2:
                st.markdown(f"<h3 style='font-size: 24px;'>{st.session_state.selected}</h3>", unsafe_allow_html=True)
                st.markdown(f"**研究方向：** {st.session_state.paper_records['研究方向'].iloc[0]}" if not st.session_state.paper_records.empty else "—")
            with col3:
                st.markdown(f"**研究机构：** {st.session_state.paper_records['研究机构'].iloc[0]}" if not st.session_state.paper_records.empty else "—")
        # 数据表格区域（论文记录）
        with st.container():
            st.subheader("📄 论文记录")
        if not st.session_state.paper_records.empty:
            # 添加竖向滚动条
            st.markdown(
                """
                <style>
                .scrollable-table {
                    max-height: 300px;  /* 设置最大高度 */
                    overflow-y: auto;   /* 添加竖向滚动条 */
                    display: block;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            # 将 DataFrame 转换为 HTML，并添加滚动条样式
            st.markdown(
                f'<div class="scrollable-table">{st.session_state.paper_records.to_html(escape=False, index=False)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("暂无论文不端记录",icon="ℹ️")

        # 项目记录表格
        with st.container():
            st.subheader("📋 项目记录")
        if not st.session_state.project_records.empty:
            st.markdown(
                f'<div class="scrollable-table">{st.session_state.project_records.to_html(escape=False, index=False, col_space=50)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("暂无项目不端记录", icon="ℹ️")

        # 移动核心指标到项目记录之后
        with st.container(border=True):
            st.markdown("<h3>核心指标</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("风险值", f"{st.session_state.author_risk:.2f}", delta_color="inverse")
            with col2:
                risk_level = "high" if st.session_state.author_risk > 8.5 else "low"
                st.metric("等级", f"{'⚠️ 高风险' if risk_level == 'high' else '✅ 低风险'}",
                          help="风险值超过8.5为高风险", label_visibility="collapsed")



        # 大模型评价区域
        with st.container(border=True):
            st.subheader("📝 智谱大模型评价")
            executor = ThreadPoolExecutor(max_workers=2)

            # 异步获取评价的函数，将 selected 作为参数传入
            def async_evaluation(selected, paper_records, project_records, related_people):
                return get_zhipu_evaluation(
                    selected,
                    paper_records,
                    project_records,
                    related_people
                )

            # 新增：调用智谱大模型的按钮
            if st.session_state.search_button_clicked and st.session_state.selected:
                if st.button(f"📝 获取 {st.session_state.selected} 的大模型评价"):
                    future = executor.submit(
                        async_evaluation,
                        st.session_state.selected,
                        st.session_state.paper_records,
                        st.session_state.project_records,
                        st.session_state.related_people
                    )
                    with st.spinner("正在调用智谱大模型进行评价..."):
                        st.session_state.evaluation = future.result()
            if st.session_state.evaluation is not None:
                st.info(st.session_state.evaluation, icon="💡")

        # 新增：合作关系网络图按钮
        if st.session_state.search_button_clicked and st.session_state.selected:
            if st.button("🕸️ 查看合作关系网络"):
                def build_network_graph(author):
                    G = nx.Graph()
                    G.add_node(author)

                    # 查找与查询作者有共同研究机构、研究方向或不端内容的作者
                    related = papers[
                        (papers['研究机构'] == papers[papers['姓名'] == author]['研究机构'].iloc[0]) |
                        (papers['研究方向'] == papers[papers['姓名'] == author]['研究方向'].iloc[0]) |
                        (papers['不端内容'] == papers[papers['姓名'] == author]['不端内容'].iloc[0])
                    ]['姓名'].unique()

                    for person in related:
                        if person != author:
                            reason = ''
                            if papers[(papers['姓名'] == author) & (papers['研究机构'] == papers[
                                papers['姓名'] == person]['研究机构'].iloc[0])].shape[0] > 0:
                                reason = '研究机构相同'
                            elif papers[(papers['姓名'] == author) & (papers['研究方向'] == papers[
                                papers['姓名'] == person]['研究方向'].iloc[0])].shape[0] > 0:
                                reason = '研究方向相似'
                            else:
                                reason = '不端内容相关'
                            G.add_node(person)
                            G.add_edge(author, person, label=reason)

                    # 使用 plotly 绘制网络图
                    pos = nx.spring_layout(G, k=0.5)  # 布局
                    edge_trace = []
                    edge_annotations = []  # 用于存储边的标注信息
                    for edge in G.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_trace.append(go.Scatter(
                            x=[x0, x1, None], y=[y0, y1, None],
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='text',
                            mode='lines'
                        ))

                        # 计算边的中点位置，用于放置标注文字
                        mid_x = (x0 + x1) / 2
                        mid_y = (y0 + y1) / 2
                        edge_annotations.append(
                            dict(
                                x=mid_x,
                                y=mid_y,
                                xref='x',
                                yref='y',
                                text=edge[2]['label'],  # 相连的原因作为标注文字
                                showarrow=False,
                                font=dict(size=10, color='black')
                            )
                        )

                    node_trace = go.Scatter(
                        x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
                        marker=dict(
                            showscale=True,
                            colorscale='YlGnBu',
                            size=10,
                        )
                    )
                    for node in G.nodes():
                        x, y = pos[node]
                        node_trace['x'] += tuple([x])
                        node_trace['y'] += tuple([y])
                        node_trace['text'] += tuple([node])

                    fig = go.Figure(
                        data=edge_trace + [node_trace],
                        layout=go.Layout(
                            title='<br>合作关系网络图',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            annotations=edge_annotations  # 添加边的标注信息
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                build_network_graph(st.session_state.selected)

        # 下载查询结果
        result_dict = {
            '论文记录': st.session_state.paper_records,
            '项目记录': st.session_state.project_records,
            '风险分析': pd.DataFrame({
                '作者': [st.session_state.selected],
                '信用风险值': [st.session_state.author_risk],
                '风险等级': ['高风险' if risk_level == 'high' else '低风险']
            })
        }
        if st.session_state.evaluation:
            result_dict['智谱大模型评价'] = pd.DataFrame({'评价内容': [st.session_state.evaluation]})

        with pd.ExcelWriter('查询结果.xlsx') as writer:
            for sheet_name, df in result_dict.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

        with open('查询结果.xlsx', 'rb') as file:
            st.download_button(
                label="📥 下载查询结果",
                data=file,
                file_name='查询结果.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                
            )
