import requests  # 用于调用API

# ==========================
# 智谱清言大模型API调用函数
# ==========================
def generate_research_report(author_name, papers, projects):
    # 假设智谱清言大模型的API端点和API密钥
    API_URL = "https://api.zhiqingyan.com/generate_report"
    API_KEY = "89c41de3c3a34f62972bc75683c66c72.ZGwzmpwgMfjtmksz"
    
    # 准备请求数据
    data = {
        "author_name": author_name,
        "papers": papers.to_dict(orient='records'),
        "projects": projects.to_dict(orient='records'),
        "api_key": API_KEY
    }
    
    # 发送请求
    response = requests.post(API_URL, json=data)
    
    if response.status_code == 200:
        return response.json().get("report", "无法生成报告")
    else:
        return "请求失败，无法生成报告"

# ==========================
# 可视化界面模块
# ==========================
def main():
    # ...（之前的代码保持不变）

    # 主界面
    st.title("🔍 科研人员信用风险分析系统")

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

        # ======================
        # 信息展示
        # ======================
        st.subheader("📄 论文记录")
        if not paper_records.empty:
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
        # 智谱清言大模型生成报告
        # ======================
        if st.button("📝 生成科研诚信报告（智谱清言大模型）"):
            with st.spinner("正在生成报告..."):
                report = generate_research_report(selected, paper_records, project_records)
                st.subheader("📜 科研诚信报告")
                st.write(report)

        # ======================
        # 关系网络可视化
        # ======================
        with st.expander("🕸️ 展开合作关系网络", expanded=True):
            build_network_graph(selected)

if __name__ == "__main__":
    main()
