"""四阶段流水线集成测试。

只替换外部边界：fetch_stock_data、三条 LCEL 链、calculate_indicators。
run_analysis 的编排逻辑、字段装配、TechnicalIndicators 组装全部真实执行。
"""

import logging
import types
from datetime import datetime

import pytest

from backend.core.errors import InvalidSymbolError
from backend.pipeline import analyzer
from backend.pipeline.data_fetcher import FetchResult, StockInfo
from backend.pipeline.technical import TechnicalData

pytestmark = pytest.mark.integration


class FakeChain:
    """记录调用顺序与入参；可配置返回结果或抛异常。"""

    def __init__(self, name, result, calls, configs=None):
        self.name = name
        self.result = result
        self.calls = calls
        self.configs = configs if configs is not None else []

    def invoke(self, payload, **kwargs):
        self.calls.append((self.name, payload))
        self.configs.append(kwargs.get("config"))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def called(self) -> bool:
        return any(name == self.name for name, _ in self.calls)


FUND_JSON = {
    "metrics": [
        {"label": "营收增速", "value": "6.3%", "yoy_change": "-3pct", "assessment": "neutral"}
    ],
    "summary": "基本面总结",
}
SENT_JSON = {"overall": "neutral", "score": 0.2, "key_drivers": ["渠道压力"]}
BULL_JSON = {"viewpoint": "看多论点", "key_evidence": ["毛利高"], "confidence": 0.8}
BEAR_JSON = {"viewpoint": "看空论点", "key_evidence": ["增速降"], "confidence": 0.6}
ARB_JSON = {
    "risks": [{"category": "增长可持续性", "level": "high", "detail": "增速回落"}],
    "conclusion": "综合结论",
    "debate_verdict": "分歧在增速",
}


@pytest.fixture
def pipeline(monkeypatch):
    """装好全部替身，返回 (calls, fetch 替身可改的容器)。"""
    calls: list[tuple[str, dict]] = []
    state = {
        "fetch": FetchResult(
            info=StockInfo("600519", "贵州茅台", "a_share"),
            financial_text="财务数据",
            news_text="近期新闻",
            sources=["同花顺财务摘要"],
        ),
        "tech": TechnicalData(
            ma_5=1500.0, ma_20=1450.0, rsi_14=55.0, price_history=[{"close": 1510.0}]
        ),
    }

    monkeypatch.setattr(analyzer, "fetch_stock_data", lambda symbol, market: state["fetch"])
    monkeypatch.setattr(analyzer, "calculate_indicators", lambda symbol, market: state["tech"])
    monkeypatch.setattr(analyzer, "fundamental_chain", FakeChain("fundamental", FUND_JSON, calls))
    monkeypatch.setattr(analyzer, "sentiment_chain", FakeChain("sentiment", SENT_JSON, calls))
    monkeypatch.setattr(
        analyzer, "debate_chain", FakeChain("debate", {"bull": BULL_JSON, "bear": BEAR_JSON}, calls)
    )
    monkeypatch.setattr(analyzer, "arbitrator_chain", FakeChain("arbitrator", ARB_JSON, calls))
    return calls, state


def test_full_pipeline_assembles_complete_report(pipeline):
    _, _ = pipeline
    report = analyzer.run_analysis("600519")

    assert report.symbol == "600519"
    assert report.name == "贵州茅台"
    assert report.sources == ["同花顺财务摘要"]
    assert report.fundamental_summary == "基本面总结"
    assert report.fundamental_metrics[0].label == "营收增速"
    assert report.conclusion == "综合结论"
    assert report.debate_verdict == "分歧在增速"
    assert report.risks[0].category == "增长可持续性"
    assert report.sentiment is not None and report.sentiment.score == 0.2
    assert isinstance(report.timestamp, datetime)


def test_stages_run_in_order(pipeline):
    calls, _ = pipeline
    analyzer.run_analysis("600519")
    assert [name for name, _ in calls] == ["fundamental", "sentiment", "debate", "arbitrator"]


def test_debate_chain_is_invoked_once_for_both_sides(pipeline):
    calls, _ = pipeline
    analyzer.run_analysis("600519")

    debate_payloads = [payload for name, payload in calls if name == "debate"]
    assert len(debate_payloads) == 1
    assert set(debate_payloads[0]) == {
        "name",
        "symbol",
        "fundamental_summary",
        "technical_data",
        "news_data",
    }


def test_debate_theses_are_labelled_by_side(pipeline):
    report = analyzer.run_analysis("600519")
    assert report.bull_thesis.analyst == "bull"
    assert report.bull_thesis.viewpoint == "看多论点"
    assert report.bull_thesis.confidence == 0.8
    assert report.bear_thesis.analyst == "bear"
    assert report.bear_thesis.confidence == 0.6


def test_arbitrator_receives_both_viewpoints(pipeline):
    calls, _ = pipeline
    analyzer.run_analysis("600519")

    payload = next(payload for name, payload in calls if name == "arbitrator")
    assert payload["bull_thesis"] == "看多论点"
    assert payload["bear_thesis"] == "看空论点"
    assert payload["fundamental_summary"] == "基本面总结"


def test_fundamental_chain_receives_fetched_text(pipeline):
    calls, _ = pipeline
    analyzer.run_analysis("600519")

    payload = next(payload for name, payload in calls if name == "fundamental")
    assert payload == {"name": "贵州茅台", "symbol": "600519", "financial_data": "财务数据"}


def test_technical_summary_reaches_debate_and_arbitrator(pipeline):
    calls, _ = pipeline
    analyzer.run_analysis("600519")

    for stage in ("debate", "arbitrator"):
        payload = next(p for name, p in calls if name == stage)
        assert "最新收盘价: 1510.0" in payload["technical_data"]
        assert "MA5" not in payload["technical_data"] or "1510.0" in payload["technical_data"]


# ---------- 舆情分支 ----------


def test_sentiment_skipped_when_disabled(pipeline):
    calls, _ = pipeline
    report = analyzer.run_analysis("600519", include_sentiment=False)

    assert report.sentiment is None
    assert not any(name == "sentiment" for name, _ in calls)


@pytest.mark.parametrize("news_text", ["", "暂无近期新闻"])
def test_sentiment_skipped_when_no_news(pipeline, news_text):
    calls, state = pipeline
    state["fetch"].news_text = news_text
    report = analyzer.run_analysis("600519")

    assert report.sentiment is None
    assert not any(name == "sentiment" for name, _ in calls)


def test_empty_news_is_replaced_by_placeholder_for_downstream(pipeline):
    calls, state = pipeline
    state["fetch"].news_text = ""
    analyzer.run_analysis("600519")

    payload = next(p for name, p in calls if name == "debate")
    assert payload["news_data"] == "无近期新闻数据"


# ---------- 技术面缺失 ----------


def test_missing_technical_data_degrades_gracefully(pipeline):
    calls, state = pipeline
    state["tech"] = None
    report = analyzer.run_analysis("600519")

    assert report.technical is None
    for stage in ("debate", "arbitrator"):
        payload = next(p for name, p in calls if name == stage)
        assert payload["technical_data"] == "暂无技术面数据"


def test_technical_without_price_history_uses_zero_close(pipeline):
    calls, state = pipeline
    state["tech"] = TechnicalData(ma_5=1500.0, price_history=[])
    report = analyzer.run_analysis("600519")

    assert report.technical is not None
    payload = next(p for name, p in calls if name == "debate")
    assert "最新收盘价: 0" in payload["technical_data"]


def test_technical_indicators_are_copied_into_report(pipeline):
    report = analyzer.run_analysis("600519")
    assert report.technical.ma_5 == 1500.0
    assert report.technical.ma_20 == 1450.0
    assert report.technical.rsi_14 == 55.0
    assert report.technical.price_history == [{"close": 1510.0}]


# ---------- 缺省与失败传播 ----------


def test_debate_verdict_defaults_to_empty_string(pipeline, monkeypatch):
    payload = {k: v for k, v in ARB_JSON.items() if k != "debate_verdict"}
    monotonic = FakeChain("arbitrator", payload, [])
    monkeypatch.setattr(analyzer, "arbitrator_chain", monotonic)
    assert analyzer.run_analysis("600519").debate_verdict == ""


def test_stage_failure_propagates(pipeline, monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        analyzer, "arbitrator_chain", FakeChain("arbitrator", RuntimeError("仲裁失败"), calls)
    )
    with pytest.raises(RuntimeError, match="仲裁失败"):
        analyzer.run_analysis("600519")


def test_fetch_failure_propagates_before_any_llm_call(pipeline, monkeypatch):
    calls, _ = pipeline

    def boom(symbol, market):
        raise InvalidSymbolError("无法自动识别")

    monkeypatch.setattr(analyzer, "fetch_stock_data", boom)
    with pytest.raises(InvalidSymbolError):
        analyzer.run_analysis("600519")
    assert calls == []


def test_sentiment_without_news_still_reports_news_text_for_debate(pipeline):
    """舆情被跳过不影响争议阶段拿到原始新闻。"""
    calls, _ = pipeline
    analyzer.run_analysis("600519")
    payload = next(p for name, p in calls if name == "debate")
    assert payload["news_data"] == "近期新闻"


def test_run_analysis_is_repeatable(pipeline):
    """同一进程内多次分析不应互相污染：每次都产出独立对象。"""
    first = analyzer.run_analysis("600519")
    second = analyzer.run_analysis("600519")

    assert first is not second
    assert first.symbol == second.symbol == "600519"
    assert first.timestamp.tzinfo is None  # 存的是本地朴素时间
    assert second.timestamp >= first.timestamp
    assert first.bull_thesis is not second.bull_thesis


# ---------- 阶段缓存 ----------


class CountingChain:
    def __init__(self, result) -> None:
        self.result = result
        self.invocations = 0
        self.configs: list[dict] = []

    def invoke(self, payload, **kwargs):
        self.invocations += 1
        self.configs.append(kwargs.get("config") or {})
        return self.result


def make_callbacks():
    from backend.core.tokens import TokenUsageCallback

    return {stage: TokenUsageCallback(stage) for stage in analyzer._STAGES}


def test_run_stage_caches_result():
    chain = CountingChain({"v": 1})
    payload = {"prompt": "这个 payload 只在本用例出现"}
    callbacks = make_callbacks()

    first = analyzer._run_stage("fundamental", chain, payload, callbacks)
    second = analyzer._run_stage("fundamental", chain, payload, callbacks)

    assert first == second == {"v": 1}
    assert chain.invocations == 1


def test_run_stage_cache_key_depends_on_model(monkeypatch):
    chain = CountingChain({"v": 1})
    payload = {"prompt": "换模型必须让旧缓存失效"}
    callbacks = make_callbacks()

    analyzer._run_stage("fundamental", chain, payload, callbacks)
    monkeypatch.setattr(analyzer, "_settings", types.SimpleNamespace(llm_model="另一个模型"))
    analyzer._run_stage("fundamental", chain, payload, callbacks)

    assert chain.invocations == 2


def test_run_stage_cache_key_depends_on_payload():
    chain = CountingChain({"v": 1})
    callbacks = make_callbacks()

    analyzer._run_stage("debate", chain, {"p": "a"}, callbacks)
    analyzer._run_stage("debate", chain, {"p": "b"}, callbacks)

    assert chain.invocations == 2


def test_run_stage_passes_stage_specific_callback():
    from backend.core.tokens import TokenUsageCallback

    chain = CountingChain({"v": 1})
    analyzer._run_stage("sentiment", chain, {"p": "回调接线"}, make_callbacks())

    callback = chain.configs[0]["callbacks"][0]
    assert isinstance(callback, TokenUsageCallback)
    assert callback.stage == "sentiment"


def test_run_stage_marks_cache_hit_in_usage_accumulator():
    from backend.core.tokens import TokenUsageAccumulator, usage_var

    accumulator = TokenUsageAccumulator()
    token = usage_var.set(accumulator)
    try:
        chain = CountingChain({"v": 1})
        payload = {"p": "命中要记进汇总"}
        analyzer._run_stage("arbitrator", chain, payload, make_callbacks())
        analyzer._run_stage("arbitrator", chain, payload, make_callbacks())
    finally:
        usage_var.reset(token)

    assert accumulator.cached_stages == ["arbitrator"]


def test_second_identical_run_makes_no_llm_calls(pipeline):
    """缓存的核心收益：同样输入第二次不再花 LLM 的钱。"""
    calls, _ = pipeline
    analyzer.run_analysis("600519")
    assert len(calls) == 4  # fundamental / sentiment / debate / arbitrator

    calls.clear()
    analyzer.run_analysis("600519")
    assert calls == []


def test_usage_summary_explains_zero_tokens_on_cache_hit(pipeline, caplog):
    """token 为 0 时必须能看出是因为命中缓存，而不是统计坏了。"""
    calls, _ = pipeline
    analyzer.run_analysis("600519")

    with caplog.at_level(logging.INFO):
        analyzer.run_analysis("600519")

    record = [r for r in caplog.records if "分析用量汇总" in r.getMessage()][-1]
    assert record.llm_calls == 0
    assert set(record.cached_stages.split(",")) == set(analyzer._STAGES)


# ---------- 超时/失败降级 ----------


def test_llm_client_has_request_timeout_configured():
    """「超时降级」的前提是超时真的配上了。"""
    from backend.config import get_settings

    assert analyzer._llm.request_timeout == get_settings().llm_timeout


def test_sentiment_failure_degrades_instead_of_failing(pipeline, monkeypatch, caplog):
    calls, _ = pipeline
    monkeypatch.setattr(
        analyzer, "sentiment_chain", FakeChain("sentiment", RuntimeError("超时"), calls)
    )

    with caplog.at_level(logging.INFO):
        report = analyzer.run_analysis("600519")

    assert report.sentiment is None
    assert "Step 3/4" in caplog.text  # 后续阶段照常执行
    assert report.conclusion == "综合结论"
    assert report.risks


def test_debate_failure_degrades_instead_of_failing(pipeline, monkeypatch):
    calls, _ = pipeline
    monkeypatch.setattr(analyzer, "debate_chain", FakeChain("debate", RuntimeError("超时"), calls))

    report = analyzer.run_analysis("600519")

    assert report.bull_thesis is None
    assert report.bear_thesis is None
    assert report.conclusion == "综合结论"  # 仲裁仍然跑完
    assert report.risks


def test_degraded_stages_are_logged(pipeline, monkeypatch, caplog):
    calls, _ = pipeline
    monkeypatch.setattr(analyzer, "debate_chain", FakeChain("debate", RuntimeError("超时"), calls))

    with caplog.at_level(logging.WARNING):
        analyzer.run_analysis("600519")

    assert "本次分析有阶段被降级" in caplog.text
    record = next(r for r in caplog.records if hasattr(r, "degraded_stages"))
    assert record.degraded_stages == "debate"


def test_arbitrator_gets_placeholder_when_debate_missing(pipeline, monkeypatch):
    calls, _ = pipeline
    monkeypatch.setattr(analyzer, "debate_chain", FakeChain("debate", RuntimeError("超时"), calls))

    analyzer.run_analysis("600519")

    payload = next(p for name, p in calls if name == "arbitrator")
    assert "未完成" in payload["bull_thesis"]
    assert "未完成" in payload["bear_thesis"]


@pytest.mark.parametrize("stage", ["fundamental", "arbitrator"])
def test_required_stage_failure_still_fails_the_request(pipeline, monkeypatch, stage):
    """基本面与风控仲裁的字段是必需的 —— 缺了就产不出合法报告。

    这种情况必须让异常上抛成 502，不能降级后给出一个"看似完整、实则缺结论"的报告。
    """
    calls, _ = pipeline
    monkeypatch.setattr(analyzer, f"{stage}_chain", FakeChain(stage, RuntimeError("超时"), calls))

    with pytest.raises(RuntimeError, match="超时"):
        analyzer.run_analysis("600519")


def test_usage_summary_is_logged_even_when_pipeline_fails(pipeline, monkeypatch, caplog):
    calls, _ = pipeline
    monkeypatch.setattr(
        analyzer, "arbitrator_chain", FakeChain("arbitrator", RuntimeError("仲裁失败"), calls)
    )

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="仲裁失败"):
        analyzer.run_analysis("600519")

    assert any("分析用量汇总" in r.getMessage() for r in caplog.records)


def test_logging_reports_each_stage(pipeline, caplog):
    with caplog.at_level(logging.INFO):
        analyzer.run_analysis("600519")
    for stage in ("Step 1/4", "Step 2/4", "Step 3/4", "Step 4/4"):
        assert stage in caplog.text
