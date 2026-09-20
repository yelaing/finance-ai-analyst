from backend.schemas.models import AnalysisReport


def render_markdown(report: AnalysisReport) -> str:
    lines = [
        f"# {report.name}（{report.symbol}）分析报告",
        f"",
        f"**分析时间**: {report.timestamp.strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"---",
        f"",
        f"## 一、基本面分析",
        f"",
    ]

    # Metrics table
    lines.append("| 指标 | 数值 | 同比变化 | 评估 |")
    lines.append("|------|------|----------|------|")
    for m in report.fundamental_metrics:
        yoy = m.yoy_change or "-"
        emoji = {"positive": "▲", "neutral": "—", "negative": "▼"}.get(m.assessment, "·")
        lines.append(f"| {m.label} | {m.value} | {yoy} | {emoji} {m.assessment} |")

    lines.append("")
    lines.append(f"**基本面总结**：{report.fundamental_summary}")
    lines.append("")

    # Technical
    if report.technical:
        lines.append("---")
        lines.append("")
        lines.append("## 二、技术面分析")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        tech = report.technical
        for label, val in [
            ("MA5", tech.ma_5), ("MA20", tech.ma_20), ("MA60", tech.ma_60),
            ("MACD DIF", tech.macd_dif), ("MACD DEA", tech.macd_dea),
            ("MACD 柱", tech.macd_bar), ("RSI(14)", tech.rsi_14),
            ("KDJ-K", tech.kdj_k), ("KDJ-D", tech.kdj_d), ("KDJ-J", tech.kdj_j),
        ]:
            if val is not None:
                lines.append(f"| {label} | {val} |")
        lines.append("")
        if tech.trend_summary:
            lines.append(f"**技术面解读**：{tech.trend_summary}")
            lines.append("")

    # Sentiment
    lines.append("---")
    lines.append("")
    lines.append("## 三、舆情分析")
    lines.append("")
    if report.sentiment:
        lines.append(f"- 整体倾向：{report.sentiment.overall}")
        lines.append(f"- 舆情分数：{report.sentiment.score}")
        lines.append(f"- 关键驱动因素：{'、'.join(report.sentiment.key_drivers)}")
    else:
        lines.append("（本次分析未包含舆情数据）")
    lines.append("")

    # Debate
    if report.bull_thesis:
        lines.append("---")
        lines.append("")
        lines.append("## 四、多空辩论")
        lines.append("")
        lines.append(f"### 多头观点（置信度 {report.bull_thesis.confidence:.0%}）")
        lines.append(f"> {report.bull_thesis.viewpoint}")
        lines.append("")
        lines.append("**证据**：")
        for e in report.bull_thesis.key_evidence:
            lines.append(f"- {e}")
        lines.append("")
        lines.append(f"### 空头观点（置信度 {report.bear_thesis.confidence:.0%}）")
        lines.append(f"> {report.bear_thesis.viewpoint}")
        lines.append("")
        lines.append("**证据**：")
        for e in report.bear_thesis.key_evidence:
            lines.append(f"- {e}")
        lines.append("")
        if report.debate_verdict:
            lines.append(f"**仲裁判断**：{report.debate_verdict}")
            lines.append("")

    # Risks
    lines.append("---")
    lines.append("")
    lines.append("## 五、风险提示")
    lines.append("")
    lines.append("| 类别 | 等级 | 说明 |")
    lines.append("|------|------|------|")
    for r in report.risks:
        lines.append(f"| {r.category} | {r.level} | {r.detail} |")
    lines.append("")

    # Conclusion
    lines.append("---")
    lines.append("")
    lines.append("## 六、综合结论")
    lines.append("")
    lines.append(report.conclusion)
    lines.append("")

    # Sources
    if report.sources:
        lines.append("---")
        lines.append("")
        lines.append(f"**数据来源**：{'、'.join(report.sources)}")
        lines.append("")

    lines.append("---")
    lines.append("*本报告由 Finance AI Analyst 自动生成，仅供参考，不构成投资建议。*")

    return "\n".join(lines)
