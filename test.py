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
        # ...（保持原有的权重字典不变）
    }

    # 读取数据并构建网络（保持原有网络构建逻辑不变）
    # ...
    return risk_df, papers_df, projects_df

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
