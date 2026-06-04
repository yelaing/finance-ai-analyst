import streamlit as st


def render_metrics(metrics: list):
    cols = st.columns(min(len(metrics), 4))
    for i, m in enumerate(metrics):
        with cols[i % 4]:
            emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(m.assessment, "⚪")
            st.metric(
                label=m.label,
                value=m.value,
                delta=m.yoy_change,
                help=f"评估: {m.assessment}",
            )
            st.caption(f"{emoji} {m.assessment}")


def render_sentiment(sentiment):
    if not sentiment:
        return
    st.subheader("舆情分析")
    c1, c2 = st.columns([1, 3])
    emoji = {"bullish": "📈", "neutral": "➡️", "bearish": "📉"}.get(sentiment.overall, "❓")
    score_pct = int((sentiment.score + 1) / 2 * 100)
    with c1:
        st.metric("舆情分数", f"{score_pct}%")
        st.caption(f"{emoji} {sentiment.overall}")
    with c2:
        st.progress(score_pct / 100, text=f"倾向: {sentiment.overall}")
        for d in sentiment.key_drivers:
            st.caption(f"• {d}")


def render_risks(risks: list):
    st.subheader("风险提示")
    for r in risks:
        color = {"low": "green", "medium": "orange", "high": "red"}.get(r.level, "grey")
        st.markdown(
            f"| <span style='color:{color}'>[{r.level.upper()}]</span> | **{r.category}** | {r.detail} |",
            unsafe_allow_html=True,
        )


def render_report(report):
    st.title(f"{report.name} ({report.symbol})")
    st.caption(f"分析时间: {report.timestamp.strftime('%Y-%m-%d %H:%M')}")

    tab1, tab2, tab3 = st.tabs(["基本面", "舆情与风险", "结论"])

    with tab1:
        st.subheader("核心财务指标")
        render_metrics(report.fundamental_metrics)
        st.markdown("---")
        st.subheader("基本面总结")
        st.info(report.fundamental_summary)

    with tab2:
        render_sentiment(report.sentiment)
        st.markdown("---")
        render_risks(report.risks)

    with tab3:
        st.subheader("综合结论")
        st.success(report.conclusion)
        if report.sources:
            st.caption("数据来源: " + ", ".join(report.sources))
