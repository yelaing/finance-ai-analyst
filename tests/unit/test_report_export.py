import pytest

from backend.pipeline.report_export import render_markdown
from backend.schemas.models import MetricItem, TechnicalIndicators


def test_full_report_contains_all_six_sections(report_factory):
    report = report_factory().model_copy(
        update={
            "technical": TechnicalIndicators(
                ma_5=1500.0, ma_20=1450.0, rsi_14=55.0, trend_summary="短期均线偏多。"
            )
        }
    )
    md = render_markdown(report)

    assert md.startswith("# 贵州茅台（600519）分析报告")
    assert "**分析时间**: 2026-09-24 10:00" in md
    for section in (
        "## 一、基本面分析",
        "## 二、技术面分析",
        "## 三、舆情分析",
        "## 四、多空辩论",
        "## 五、风险提示",
        "## 六、综合结论",
    ):
        assert section in md
    assert md.rstrip().endswith(
        "*本报告由 Finance AI Analyst 自动生成，仅供参考，不构成投资建议。*"
    )


@pytest.mark.parametrize(
    ("assessment", "mark"),
    [("positive", "▲"), ("neutral", "—"), ("negative", "▼"), ("?weird", "·")],
)
def test_metric_assessment_markers(report_factory, assessment, mark):
    report = report_factory().model_copy(
        update={
            "fundamental_metrics": [
                MetricItem(label="ROE", value="30%", yoy_change="+2pct", assessment=assessment)
            ]
        }
    )
    assert f"| ROE | 30% | +2pct | {mark} {assessment} |" in render_markdown(report)


def test_metric_without_yoy_falls_back_to_dash(report_factory):
    report = report_factory().model_copy(
        update={"fundamental_metrics": [MetricItem(label="ROE", value="30%", assessment="neutral")]}
    )
    assert "| ROE | 30% | - | — neutral |" in render_markdown(report)


def test_technical_section_is_omitted_when_absent(report_factory):
    assert "## 二、技术面分析" not in render_markdown(report_factory())


def test_technical_section_skips_none_values_and_empty_summary(report_factory):
    report = report_factory().model_copy(update={"technical": TechnicalIndicators(ma_5=1500.0)})
    md = render_markdown(report)
    assert "| MA5 | 1500.0 |" in md
    assert "MA20" not in md
    assert "**技术面解读**" not in md


def test_technical_trend_summary_is_rendered_when_present(report_factory):
    report = report_factory().model_copy(
        update={"technical": TechnicalIndicators(trend_summary="短期均线偏多。")}
    )
    assert "**技术面解读**：短期均线偏多。" in render_markdown(report)


def test_missing_sentiment_is_stated_explicitly(report_factory):
    md = render_markdown(report_factory(with_sentiment=False))
    assert "（本次分析未包含舆情数据）" in md
    assert "整体倾向" not in md


def test_sentiment_details_are_rendered(report_factory):
    md = render_markdown(report_factory())
    assert "- 整体倾向：neutral" in md
    assert "- 舆情分数：0.1" in md
    assert "- 关键驱动因素：渠道压力" in md


def test_debate_section_is_omitted_when_both_sides_absent(report_factory):
    md = render_markdown(report_factory(with_debate=False))
    assert "## 四、多空辩论" not in md
    assert "仲裁判断" not in md


def test_debate_section_survives_one_side_missing(report_factory):
    """原先只判断 bull_thesis 却无条件解引用 bear_thesis，一边缺失就 AttributeError。"""
    only_bull = report_factory().model_copy(update={"bear_thesis": None})
    only_bear = report_factory().model_copy(update={"bull_thesis": None})

    for report in (only_bull, only_bear):
        md = render_markdown(report)
        assert "## 四、多空辩论" not in md
        assert "## 五、风险提示" in md  # 后续小节照常渲染


def test_debate_verdict_is_optional(report_factory):
    report = report_factory().model_copy(update={"debate_verdict": None})
    md = render_markdown(report)
    assert "### 多头观点（置信度 70%）" in md
    assert "仲裁判断" not in md


def test_risk_table_lists_every_risk(report_factory):
    md = render_markdown(report_factory())
    assert "| 增长可持续性 | high | 增速回落 |" in md


def test_sources_line_is_omitted_when_empty(report_factory):
    report = report_factory().model_copy(update={"sources": []})
    assert "**数据来源**" not in render_markdown(report)


def test_sources_line_joins_multiple_sources(report_factory):
    report = report_factory().model_copy(update={"sources": ["同花顺财务摘要", "东方财富新闻"]})
    assert "**数据来源**：同花顺财务摘要、东方财富新闻" in render_markdown(report)
