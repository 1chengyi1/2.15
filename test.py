import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zhipuai import ZhipuAI
import os

# 设置智谱 API 密钥
client = ZhipuAI(api_key="89c41de3c3a34f62972bc75683c66c72.ZGwzmpwgMfjtmksz")

# ==========================
# 数据预处理模块
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

    # 读取数据并构建网络（保持原有网络构建逻辑不变）
    # 读取原始数据
    papers_df = pd.read_excel('实验数据.xlsx', sheet_name='论文')
    projects_df = pd.read_excel('实验数据.xlsx', sheet_name='项目')

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

# ==========================
# 智谱大模型交互模块
# ==========================
def get_zhipu_evaluation(selected, paper_records, project_records):
    """获取包含网络搜索的深度分析报告"""
    prompt_template = f"""
请为科研人员【{selected}】生成深度分析报告，需包含以下内容：

一、学术背景分析（基于网络公开信息）
1. 教育经历：毕业院校、学位信息
2. 任职机构：当前及历史任职情况
3. 研究方向：主要研究领域及细分方向
4. 学术成果：代表性论文、专利、项目（列举3-5个重点成果）

二、科研诚信评估（结合国家政策）
根据以下政策分析历史记录：
- 《科研诚信案件调查处理规则（试行）》
- 《关于进一步加强科研诚信建设的若干意见》
- 《科学技术活动违规行为处理暂行规定》
评估维度：
1. 行为严重性分析
2. 整改情况追踪
3. 潜在影响评估

三、合作网络分析（基于公开数据）
1. 高频合作者（列出5-10人）
2. 合作形式分析（论文/项目/专利等）
3. 机构关联网络
4. 国际合作情况

四、风险预警建议
1. 监管关注建议
2. 合作风险提示
3. 项目评审建议

格式要求：
## 学术背景
...
## 诚信评估  
...
## 合作网络
...
## 风险预警
..."""

    try:
        response = client.chat.completions.create(
            model="glm-4v-plus",
            messages=[{
                "role": "user",
                "content": prompt_template,
                "temperature": 0.3
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"## 服务异常\n{str(e)}"

# ==========================
# 可视化界面模块
# ==========================
def main():
    st.set_page_config(
        page_title="科研诚信智能分析平台",
        page_icon="🔬",
        layout="wide"
    )

    # 自定义样式
    st.markdown("""
    <style>
    .report-section { 
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 15px 0;
        background: white;
    }
    .section-title {
        color: #1e3d6d;
        border-left: 4px solid #1e3d6d;
        padding-left: 10px;
    }
    .risk-alert {
        color: #d32f2f;
        background: #ffebee;
        padding: 15px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 侧边栏（保持原有数据加载逻辑不变）
    # ...

    # 主界面
    st.title("🔍 科研人员深度分析系统")
    
    # 搜索功能（保持原有搜索逻辑不变）
    search_term = st.text_input("输入研究人员姓名：", placeholder="支持中英文姓名搜索...")
    
    if search_term:
        # ...（保持原有数据匹配逻辑）
        
        # ======================
        # 智能分析报告生成
        # ======================
        if st.button(f"🕵️ 生成{selected}的智能分析报告"):
            with st.spinner("正在通过学术大数据生成深度分析..."):
                try:
                    report = get_zhipu_evaluation(selected, paper_records, project_records)
                    
                    # 结构化显示报告
                    sections = {
                        "## 学术背景": "academic",
                        "## 诚信评估": "integrity",
                        "## 合作网络": "collab",
                        "## 风险预警": "risk"
                    }
                    
                    current_section = None
                    content_buffer = []
                    
                    for line in report.split('\n'):
                        line = line.strip()
                        if line in sections:
                            if current_section:
                                # 输出缓冲内容
                                with st.container():
                                    st.markdown(f'<div class="report-section" id="{sections[current_section]}">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="section-title">{current_section}</div>', unsafe_allow_html=True)
                                    st.markdown('\n'.join(content_buffer))
                                    st.markdown('</div>', unsafe_allow_html=True)
                            current_section = line
                            content_buffer = []
                        else:
                            content_buffer.append(line)
                    
                    # 处理最后一个部分
                    if current_section:
                        with st.container():
                            st.markdown(f'<div class="report-section" id="{sections[current_section]}">', unsafe_allow_html=True)
                            st.markdown(f'<div class="section-title">{current_section}</div>', unsafe_allow_html=True)
                            st.markdown('\n'.join(content_buffer))
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # 特别标注风险预警
                            if "风险预警" in current_section:
                                st.markdown('<div class="risk-alert">⚠️ 请重点关注风险预警内容</div>', unsafe_allow_html=True)
                                
                except Exception as e:
                    st.error(f"报告生成失败：{str(e)}")

        # ...（保持原有的网络可视化等组件）

if __name__ == "__main__":
    main()
