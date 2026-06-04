import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Finance AI Analyst", page_icon="📊", layout="wide")

st.sidebar.title("Finance AI Analyst")
st.sidebar.markdown("AI 驱动的财报 + 舆情综合分析")

market = st.sidebar.selectbox("市场", ["auto", "a_share", "us"], format_func=lambda x: {"auto": "自动识别", "a_share": "A股", "us": "美股"}[x])
symbol = st.sidebar.text_input("股票代码", placeholder="如 600519 或 AAPL")
include_sentiment = st.sidebar.checkbox("包含舆情分析", value=True)

if "has_report" not in st.session_state:
    st.session_state.has_report = False

if st.sidebar.button("开始分析", type="primary", use_container_width=True):
    if not symbol.strip():
        st.error("请输入股票代码")
    else:
        with st.spinner(f"正在分析 {symbol} ..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"symbol": symbol.strip(), "market": market, "include_sentiment": include_sentiment},
                    timeout=180,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("report"):
                        from frontend.components.report_view import render_report
                        from backend.schemas.models import AnalysisReport
                        report = AnalysisReport(**data["report"])
                        render_report(report)
                        st.session_state.has_report = True
                    else:
                        st.error(data.get("error", "未知错误"))
                else:
                    st.error(f"请求失败: {resp.status_code} {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("无法连接到后端，请先启动 backend：`python -m backend.main`")
            except Exception as e:
                st.error(f"分析失败: {e}")

st.sidebar.divider()

if st.sidebar.button("查看历史", use_container_width=True):
    if not symbol.strip():
        st.sidebar.warning("请输入股票代码后点击")
    else:
        try:
            resp = requests.get(f"{API_BASE}/history", params={"symbol": symbol.strip()}, timeout=10)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                st.sidebar.markdown(f"**{symbol} 的历史分析 ({len(items)} 条)**")
                for item in items:
                    ts = item.get("timestamp", "")[:16]
                    st.sidebar.caption(f"{ts} — {item.get('summary', '')}")
            else:
                st.sidebar.error("获取历史失败")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("无法连接到后端")

st.sidebar.divider()
st.sidebar.caption("Powered by LangChain + FastAPI + Streamlit")

if not st.session_state.has_report:
    st.title("Finance AI Analyst")
    st.markdown("""
    ### AI 驱动的上市公司财报 + 舆情综合分析平台

    输入 A股代码（如 `600519` 贵州茅台）或美股代码（如 `AAPL` 苹果），
    AI 将自动完成：
    1. **基本面分析** — 拉取最新财报数据，分析核心财务指标
    2. **舆情分析** — 汇总近期新闻动态，判断市场情绪
    3. **风险交叉验证** — 综合基本面与舆情，识别潜在风险

    ---
    👈 在左侧输入股票代码开始分析
    """)
