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
import zhipuai

# 设置智谱清言 API 密钥
zhipuai.api_key = "89c41de3c3a34f62972bc75683c66c72.ZGwzmpwgMfjtmksz"  # 请替换为你自己的 API 密钥

# ==========================
# 数据预处理和风险值计算模块
# ==========================
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
    papers_df = pd.read_excel('data3.xlsx', sheet_name='论文')
    projects_df = pd.read_excel('data3.xlsx', sheet_name='项目')

    # ======================
    # 网络构建函数
    # ======================
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
        similarity_matrix = cosine_similarity(tfidf_matrix)

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

    # ======================
    # Word2Vec（Skip-gram）模型定义
    # ======================
    class SkipGramModel(nn.Module):
        def __init__(self, vocab_size, embedding_size):
            super(SkipGramModel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_size)
            self.out = nn.Linear(embedding_size, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs)
            outputs = self.out(embeds)
            return outputs

    # ======================
    # 数据集定义
    # ======================
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

    # ======================
    # DeepWalk实现
    # ======================
    def deepwalk(graph, walk_length=30, num_walks=200, embedding_size=128):
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
        for epoch in range(10):
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

    # ======================
    # 执行计算流程
    # ======================
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

    return pd.DataFrame({
        '作者': list(risk_scores.keys()),
        '风险值': list(risk_scores.values())
    }), papers_df, projects_df

# 调用智谱清言 API 生成简历和评价
def generate_resume_and_evaluation(author, paper_records, project_records, risk_value):
    prompt = f"请为科研人员 {author} 生成一份简历和评价。该科研人员的论文不端记录如下：{paper_records.to_csv(sep='\t', na_rep='nan')}，项目不端记录如下：{project_records.to_csv(sep='\t', na_rep='nan')}，信用风险值为 {risk_value}。"
    response = zhipuai.model_api.invoke(
        model="chatglm_turbo",
        prompt=[{"role": "user", "content": prompt}]
    )
    if response['code'] == 200:
        return response['data']['choices'][0]['content']
    else:
        return f"请求失败，错误代码：{response['code']}，错误信息：{response['msg']}"

# ==========================
# 可视化界面模块
# ==========================
def main():
    st.set_page_config(
        page_title="科研人员诚信风险预警平台",
        page_icon="🔬",
        layout="wide"
    )

    # 自定义CSS样式
    st.markdown("""
    <style>
.high - risk { color: red; font - weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 0% {opacity:1;} 50% {opacity:0;} 100% {opacity:1;} }
.metric - box { padding: 20px; border - radius: 10px; background: #f0f2f6; margin: 10px; }
    table {
        table - layout: fixed;
    }
    table td {
        white - space: normal;
    }
 .stDataFrame tbody tr {
        display: block;
        overflow - y: auto;
        height: 200px;
    }
 .stDataFrame tbody {
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    # 侧边栏控制面板上方添加智谱清言大模型按钮
    if st.sidebar.button("🧠 智谱清言生成简历和评价", help="查找科研人员后点击此按钮生成简历和评价"):
        if 'selected_author' in st.session_state:
            author = st.session_state.selected_author
            author_risk = st.session_state.author_risk
            paper_records = st.session_state.paper_records
            project_records = st.session_state.project_records
            with st.spinner("正在调用智谱清言生成简历和评价..."):
                result = generate_resume_and_evaluation(author, paper_records, project_records, author_risk)
            st.subheader("📋 智谱清言生成的简历和评价")
            st.write(result)
        else:
            st.warning("请先搜索并选择一个科研人员")

    # 侧边栏控制面板
    with st.sidebar:
        st.title("控制面板")
        if st.button("🔄 重新计算风险值", help="当原始数据更新后点击此按钮"):
            with st.spinner("重新计算中..."):
                risk_df, papers, projects = process_risk_data()
                risk_df.to_excel('risk_scores.xlsx', index=False)
            st.success("风险值更新完成！")

        # 添加“返回首页”按钮
        if st.button("🏠 返回首页", help="点击返回首页"):
            st.markdown("[点击这里返回首页](https://chengyi10.wordpress.com/)", unsafe_allow_html=True)

    # 尝试加载现有数据
    try:
        risk_df = pd.read_excel('risk_scores.xlsx')
        papers = pd.read_excel('data3.xlsx', sheet_name='论文')
        projects = pd.read_excel('data3.xlsx', sheet_name='项目')
    except:
        with st.spinner("首次运行需要初始化数据..."):
            risk_df, papers, projects = process_risk_data()
            risk_df.to_excel('risk_scores.xlsx', index=False)

    # 主界面
    st.title("🔍 科研人员信用风险预警系统")

    # 搜索框
    search_term = st.text_input("输入研究人员姓名：", placeholder="支持模糊搜索...")

    if search_term:
        # 模糊匹配
        candidates = risk_df[risk_df['作者'].str.contains(search_term)]
        if len(candidates) == 0:
            st.warning("未找到匹配的研究人员")
            return

        # 直接选择第一个匹配人员
        selected = candidates['作者'].iloc[0]

        # 获取详细信息
        author_risk = risk_df[risk_df['作者'] == selected].iloc[0]['风险值']
        paper_records = papers[papers['姓名'] == selected]
        project_records = projects[projects['姓名'] == selected]

        # 保存选中的科研人员信息到 session_state
        st.session_state.selected_author = selected
        st.session_state.author_risk = author_risk
        st.session_state.paper_records = paper_records
        st.session_state.project_records = project_records

        # ======================
        # 信息展示
        # ======================
        st.subheader("📄 论文记录")
        if not paper_records.empty:
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
                f'<div class="scrollable-table">{paper_records.to_html(escape=False, index=False)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("暂无论文不端记录")

        st.subheader("📋 项目记录")
        if not project_records.empty:
            st.markdown(project_records.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.info("暂无项目不端记录")

        # 风险指标
        st.subheader("📊 风险分析")
        risk_level = "high" if author_risk > 12 else "low"
        cols = st.columns(4)
        cols[0].metric("信用风险值", f"{author_risk:.2f}",
                       delta_color="inverse" if risk_level == "high" else "normal")
        cols[1].metric("风险等级",
                       f"{'⚠️ 高风险' if risk_level == 'high' else '✅ 低风险'}",
                       help="高风险阈值：12")

        # ======================
        # 关系网络可视化
        # ======================
        with st.expander("🕸️ 展开合作关系网络", expanded=True):
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
                        if papers[(papers['姓名'] == author) & (papers['研究机构'] == papers[papers['姓名'] == person]['研究机构'].iloc[0])].shape[0] > 0:
                            reason = '研究机构相同'
                        elif papers[(papers['姓名'] == author) & (papers['研究方向'] == papers[papers['姓名'] == person]['研究方向'].iloc[0])].shape[0] > 0:
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

            build_network_graph(selected)


if __name__ == "__main__":
    main()
