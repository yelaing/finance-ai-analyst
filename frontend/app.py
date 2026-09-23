import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import streamlit as st

from frontend.style import inject

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Finance AI Analyst", page_icon="▲", layout="wide")
inject()

# ---- sidebar ----
st.sidebar.title("Finance AI Analyst")
st.sidebar.markdown("财报 · 舆情 · 多空辩论")

market = st.sidebar.selectbox(
    "市场",
    ["auto", "a_share", "us"],
    format_func=lambda x: {"auto": "自动识别", "a_share": "A股", "us": "美股"}[x],
)
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
                    json={
                        "symbol": symbol.strip(),
                        "market": market,
                        "include_sentiment": include_sentiment,
                    },
                    timeout=180,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("report"):
                        from backend.schemas.models import AnalysisReport
                        from frontend.components.report_view import render_report

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
            resp = requests.get(
                f"{API_BASE}/history", params={"symbol": symbol.strip()}, timeout=10
            )
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
st.sidebar.caption("Data from akshare & Yahoo Finance")

# ---- landing page ----
if not st.session_state.has_report:
    st.markdown(
        '<div class="landing-hero">'
        "<h1>Finance AI Analyst</h1>"
        '<p class="subtitle">'
        "上市公司财报与市场舆情的交叉分析工具。输入股票代码获取基本面指标、技术面数据与多空辩论结果。"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature-list">'
        '<div class="feature-item">'
        '<div class="feature-num">01</div>'
        '<div class="feature-body">'
        "<h3>基本面 + 技术面</h3>"
        "<p>营收、利润、现金流等核心指标，叠加 MA/MACD/RSI 技术面数据，一次拉取双维度。</p>"
        "</div>"
        "</div>"
        '<div class="feature-item">'
        '<div class="feature-num">02</div>'
        '<div class="feature-body">'
        "<h3>多空辩论</h3>"
        "<p>多头与空头研究员并行分析同一组数据，风控仲裁官独立裁决，暴露分歧而非掩盖分歧。</p>"
        "</div>"
        "</div>"
        '<div class="feature-item">'
        '<div class="feature-num">03</div>'
        '<div class="feature-body">'
        "<h3>舆情 + 风险交叉验证</h3>"
        "<p>近期新闻与市场传闻的智能情绪判断，基本面与舆情交叉比对，识别潜在风险点。</p>"
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="footer-hint">A股（600519）&nbsp;·&nbsp;美股（AAPL、NVDA）— 左侧输入代码开始</p>',
        unsafe_allow_html=True,
    )
