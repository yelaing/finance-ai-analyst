import streamlit as st
import pandas as pd


def _calc_change(price_history):
    """Calculate price change from last two data points."""
    if not price_history or len(price_history) < 2:
        return None, None
    last = price_history[-1]["close"]
    prev = price_history[-2]["close"]
    change = last - prev
    pct = (change / prev) * 100 if prev else 0
    return round(last, 2), round(change, 2), round(pct, 2)


def render_snapshot(report):
    """Price snapshot bar — at-a-glance key numbers."""
    price = change = chg_pct = None
    if report.technical and report.technical.price_history:
        price, change, chg_pct = _calc_change(report.technical.price_history)

    if not price:
        return

    chg_color = "var(--positive)" if change >= 0 else "var(--negative)"
    chg_sign = "+" if change >= 0 else ""

    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:20px;'
        f'padding:8px 0 16px 0;border-bottom:1px solid var(--border);margin-bottom:16px;'
        f'font-family:var(--font-mono);">'
        f'<span style="font-size:2rem;font-weight:700;color:var(--ink);">{price}</span>'
        f'<span style="font-size:1.1rem;color:{chg_color};">{chg_sign}{change} ({chg_sign}{chg_pct}%)</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_metrics(metrics: list):
    cols = st.columns(min(len(metrics), 4))
    for i, m in enumerate(metrics):
        with cols[i % 4]:
            emoji = {"positive": "▲", "neutral": "—", "negative": "▼"}.get(
                m.assessment, "·"
            )
            st.metric(
                label=m.label,
                value=m.value,
                delta=m.yoy_change,
            )
            st.caption(f"{emoji} {m.assessment}")


def render_sentiment(sentiment):
    if not sentiment:
        return

    st.subheader("舆情分析")
    emoji = {"bullish": "▲", "neutral": "—", "bearish": "▼"}.get(
        sentiment.overall, "·"
    )
    score_pct = int((sentiment.score + 1) / 2 * 100)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(
            f'<div style="text-align:center;padding:16px 0;">'
            f'<div class="sentiment-score-display">{score_pct}%</div>'
            f'<div class="sentiment-label-display">{emoji} {sentiment.overall}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.progress(score_pct / 100, text=f"市场倾向: {sentiment.overall}")
        if sentiment.key_drivers:
            st.markdown("**关键驱动因素**")
            tags_html = "".join(
                f'<span class="driver-tag">{d}</span>'
                for d in sentiment.key_drivers
            )
            st.markdown(tags_html, unsafe_allow_html=True)


def render_risks(risks: list):
    st.subheader("风险提示")
    for r in risks:
        st.markdown(
            f'<div class="risk-row">'
            f'<span class="risk-badge {r.level}">[{r.level}]</span>'
            f'<strong style="color:var(--ink)">{r.category}</strong>'
            f'<span style="color:var(--ink-secondary);margin-left:auto;text-align:right;font-size:0.88rem;">{r.detail}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_technical(technical):
    if not technical:
        st.info("暂无技术面数据")
        return

    # Candlestick chart
    if technical.price_history:
        st.subheader("K 线图 (60 日)")
        df = pd.DataFrame(technical.price_history)
        df["date"] = pd.to_datetime(df["date"])
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )])
        fig.update_layout(
            height=360,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#6b6b6b", gridcolor="#242424"),
            yaxis=dict(color="#6b6b6b", gridcolor="#242424"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Indicator values
    st.subheader("技术指标")
    cols = st.columns(4)
    with cols[0]:
        st.metric("MA5", technical.ma_5 or "—")
    with cols[1]:
        st.metric("MA20", technical.ma_20 or "—")
    with cols[2]:
        st.metric("MA60", technical.ma_60 or "—")
    with cols[3]:
        rsi = technical.rsi_14
        if rsi is not None:
            rsi_color = "var(--negative)" if rsi > 70 else ("var(--positive)" if rsi < 30 else "var(--ink)")
            st.markdown(f"**RSI(14)**&nbsp;<span style='color:{rsi_color}'>{rsi}</span>", unsafe_allow_html=True)
        else:
            st.metric("RSI(14)", "—")

    cols2 = st.columns(4)
    with cols2[0]:
        st.metric("MACD DIF", technical.macd_dif or "—")
    with cols2[1]:
        st.metric("MACD DEA", technical.macd_dea or "—")
    with cols2[2]:
        st.metric("MACD 柱", technical.macd_bar or "—")
    with cols2[3]:
        k = technical.kdj_k or "—"
        d = technical.kdj_d or "—"
        j = technical.kdj_j or "—"
        st.caption(f"**KDJ**: K:{k} D:{d} J:{j}")

    if technical.trend_summary:
        st.info(technical.trend_summary)


def render_debate(bull_thesis, bear_thesis, verdict):
    if not bull_thesis or not bear_thesis:
        return

    st.subheader("多空辩论")

    bull_points = bull_thesis.key_evidence
    bear_points = bear_thesis.key_evidence
    max_len = max(len(bull_points), len(bear_points))

    rows_html = ""
    for i in range(max_len):
        bull_item = bull_points[i] if i < len(bull_points) else "—"
        bear_item = bear_points[i] if i < len(bear_points) else "—"
        rows_html += (
            f'<tr>'
            f'<td class="debate-bull">◆ {bull_item}</td>'
            f'<td class="debate-bear">◆ {bear_item}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="margin:12px 0;">'
        f'<div style="display:flex;gap:16px;margin-bottom:8px;">'
        f'<div style="flex:1;font-weight:600;color:oklch(0.62 0.15 150);">'
        f'▲ 多头 · 置信度 {bull_thesis.confidence:.0%}</div>'
        f'<div style="flex:1;font-weight:600;color:oklch(0.58 0.20 22);">'
        f'▼ 空头 · 置信度 {bear_thesis.confidence:.0%}</div>'
        f'</div>'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<colgroup><col style="width:50%"><col style="width:50%"></colgroup>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(f"**多头核心论点** {bull_thesis.viewpoint[:200]}...")
    st.markdown(f"**空头核心论点** {bear_thesis.viewpoint[:200]}...")

    if verdict:
        st.divider()
        st.markdown("**仲裁判断**")
        st.info(verdict)


def render_download_button(report):
    symbol = report.symbol
    st.markdown(
        f'<a href="http://localhost:8000/api/v1/export/{symbol}" '
        f'download style="text-decoration:none;">'
        f'<button style="background:var(--amber);color:#000;border:none;'
        f'padding:8px 16px;border-radius:6px;cursor:pointer;font-size:0.85rem;">'
        f'⬇ 下载 Markdown 报告</button></a>',
        unsafe_allow_html=True,
    )


def render_report(report):
    # Header
    st.markdown(
        f'<div class="report-header">'
        f"<h1>{report.name} ({report.symbol})</h1>"
        f'<span class="report-time">分析时间: {report.timestamp.strftime("%Y-%m-%d %H:%M")}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Price snapshot
    render_snapshot(report)

    tabs = ["基本面", "技术面", "舆情与风险", "多空辩论", "结论"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

    with tab1:
        st.subheader("核心财务指标")
        render_metrics(report.fundamental_metrics)
        st.divider()
        st.subheader("基本面总结")
        st.info(report.fundamental_summary)

    with tab2:
        render_technical(report.technical)

    with tab3:
        render_sentiment(report.sentiment)
        st.divider()
        render_risks(report.risks)

    with tab4:
        render_debate(report.bull_thesis, report.bear_thesis, report.debate_verdict)

    with tab5:
        st.subheader("综合结论")
        st.success(report.conclusion)
        if report.sources:
            st.caption("数据来源: " + ", ".join(report.sources))
        st.divider()
        render_download_button(report)
