import logging
import time
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.core.cache import LLM, cache_key, get_cache
from backend.core.tokens import (
    TokenUsageAccumulator,
    TokenUsageCallback,
    log_summary,
    usage_var,
)
from backend.pipeline.data_fetcher import fetch_stock_data
from backend.pipeline.prompts import (
    ARBITRATOR_PROMPT,
    BEAR_ANALYST_PROMPT,
    BULL_ANALYST_PROMPT,
    FUNDAMENTAL_ANALYSIS_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
)
from backend.pipeline.technical import _build_summary, calculate_indicators
from backend.schemas.models import (
    AnalysisReport,
    DebateThesis,
    MetricItem,
    RiskItem,
    SentimentSummary,
    TechnicalIndicators,
)

logger = logging.getLogger(__name__)

_settings = get_settings()

_TEMPERATURE = 0.3

_llm = ChatOpenAI(
    base_url=_settings.llm_base_url,
    api_key=_settings.llm_api_key,
    model=_settings.llm_model,
    temperature=_TEMPERATURE,
    max_tokens=2048,
)

_parser = JsonOutputParser()


def _chain(prompt):
    """Assemble prompt | llm | parser, retrying network errors and JSON parse failures alike."""
    return (prompt | _llm | _parser).with_retry(
        stop_after_attempt=3,
        exponential_jitter_params={"initial": 1, "max": 2, "jitter": 0},
    )


fundamental_chain = _chain(FUNDAMENTAL_ANALYSIS_PROMPT)
sentiment_chain = _chain(SENTIMENT_ANALYSIS_PROMPT)
debate_chain = RunnableParallel(
    bull=_chain(BULL_ANALYST_PROMPT),
    bear=_chain(BEAR_ANALYST_PROMPT),
)
arbitrator_chain = _chain(ARBITRATOR_PROMPT)

# 4 个调用点。辩论那一点内部并行跑两条链，共用同一个标签，无法区分多空各自的用量。
_STAGES = ("fundamental", "sentiment", "debate", "arbitrator")


def _run_stage(stage: str, chain, payload: dict, callbacks: dict) -> dict:
    """带缓存的阶段调用。

    缓存键含 model 与 temperature：只拿 prompt 当键的话，换了模型或温度仍会命中
    旧结果，那是错的。命中缓存时回调不触发，因此 token 统计为 0 属正常（汇总里会
    把命中的阶段列出来，避免被误读成统计失效）。
    """
    key = cache_key(payload, model=_settings.llm_model, temperature=_TEMPERATURE)
    cached = get_cache().get(LLM, key)
    if cached is not None:
        logger.info("阶段命中缓存，跳过 LLM 调用", extra={"cache_hit": True, "stage": stage})
        accumulator = usage_var.get()
        if accumulator is not None:
            accumulator.mark_cached(stage)
        return cached

    result = chain.invoke(payload, config={"callbacks": [callbacks[stage]]})
    get_cache().set(LLM, key, result)
    return result


def run_analysis(
    symbol: str, market: str = "auto", include_sentiment: bool = True
) -> AnalysisReport:
    """跑完整四阶段流水线。

    在这里建立 token 用量累加器并放进 ContextVar，让并行分支共用同一份统计；
    无论成功还是失败，最后都会打一条请求级用量汇总。
    """
    accumulator = TokenUsageAccumulator()
    callbacks = {stage: TokenUsageCallback(stage) for stage in _STAGES}
    token = usage_var.set(accumulator)
    started = time.perf_counter()
    try:
        return _run_pipeline(symbol, market, include_sentiment, callbacks)
    finally:
        usage_var.reset(token)
        log_summary(symbol, accumulator, (time.perf_counter() - started) * 1000)


def _run_pipeline(
    symbol: str, market: str, include_sentiment: bool, callbacks: dict
) -> AnalysisReport:
    data = fetch_stock_data(symbol, market)

    # ---- Step 1: 基本面 + 技术面 ----
    logger.info("Step 1/4: 基本面 + 技术面分析")
    fund_json = _run_stage(
        "fundamental",
        fundamental_chain,
        {
            "name": data.info.name,
            "symbol": data.info.symbol,
            "financial_data": data.financial_text,
        },
        callbacks,
    )
    metrics = [MetricItem(**m) for m in fund_json["metrics"]]
    fund_summary = fund_json["summary"]

    # Technical indicators
    tech_data = calculate_indicators(symbol, data.info.market)
    tech_summary = ""
    if tech_data:
        last_close = tech_data.price_history[-1]["close"] if tech_data.price_history else 0
        tech_summary = _build_summary(tech_data, last_close)

    # ---- Step 2: 舆情分析（可选） ----
    sentiment_summary = None
    if include_sentiment and data.news_text and "暂无" not in data.news_text:
        logger.info("Step 2/4: 舆情分析")
        sent_json = _run_stage(
            "sentiment",
            sentiment_chain,
            {
                "name": data.info.name,
                "symbol": data.info.symbol,
                "news_data": data.news_text,
            },
            callbacks,
        )
        sentiment_summary = SentimentSummary(**sent_json)

    # ---- Step 3: 多头 + 空头辩论（并行） ----
    logger.info("Step 3/4: 多空辩论")
    news_text = data.news_text if data.news_text else "无近期新闻数据"
    debate = _run_stage(
        "debate",
        debate_chain,
        {
            "name": data.info.name,
            "symbol": data.info.symbol,
            "fundamental_summary": fund_summary,
            "technical_data": tech_summary or "暂无技术面数据",
            "news_data": news_text,
        },
        callbacks,
    )
    bull_json = debate["bull"]
    bear_json = debate["bear"]

    bull_thesis = DebateThesis(
        analyst="bull",
        viewpoint=bull_json["viewpoint"],
        key_evidence=bull_json["key_evidence"],
        confidence=bull_json["confidence"],
    )
    bear_thesis = DebateThesis(
        analyst="bear",
        viewpoint=bear_json["viewpoint"],
        key_evidence=bear_json["key_evidence"],
        confidence=bear_json["confidence"],
    )

    # ---- Step 4: 风控仲裁 ----
    logger.info("Step 4/4: 风控仲裁")
    arb_json = _run_stage(
        "arbitrator",
        arbitrator_chain,
        {
            "fundamental_summary": fund_summary,
            "technical_data": tech_summary or "暂无技术面数据",
            "bull_thesis": bull_json["viewpoint"],
            "bear_thesis": bear_json["viewpoint"],
        },
        callbacks,
    )
    risks = [RiskItem(**r) for r in arb_json["risks"]]

    # ---- 组装 TechnicalIndicators ----
    technical = None
    if tech_data:
        technical = TechnicalIndicators(
            ma_5=tech_data.ma_5,
            ma_20=tech_data.ma_20,
            ma_60=tech_data.ma_60,
            macd_dif=tech_data.macd_dif,
            macd_dea=tech_data.macd_dea,
            macd_bar=tech_data.macd_bar,
            rsi_14=tech_data.rsi_14,
            kdj_k=tech_data.kdj_k,
            kdj_d=tech_data.kdj_d,
            kdj_j=tech_data.kdj_j,
            trend_summary=tech_summary,
            price_history=tech_data.price_history,
        )

    return AnalysisReport(
        symbol=data.info.symbol,
        name=data.info.name,
        timestamp=datetime.now(),
        fundamental_metrics=metrics,
        fundamental_summary=fund_summary,
        sentiment=sentiment_summary,
        risks=risks,
        conclusion=arb_json["conclusion"],
        sources=data.sources,
        technical=technical,
        bull_thesis=bull_thesis,
        bear_thesis=bear_thesis,
        debate_verdict=arb_json.get("debate_verdict", ""),
    )
