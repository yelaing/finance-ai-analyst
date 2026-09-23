import logging
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

from backend.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from backend.pipeline.data_fetcher import fetch_stock_data
from backend.pipeline.technical import calculate_indicators, _build_summary
from backend.pipeline.prompts import (
    FUNDAMENTAL_ANALYSIS_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
    BULL_ANALYST_PROMPT,
    BEAR_ANALYST_PROMPT,
    ARBITRATOR_PROMPT,
)
from backend.schemas.models import (
    AnalysisReport,
    MetricItem,
    SentimentSummary,
    RiskItem,
    TechnicalIndicators,
    DebateThesis,
)

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    temperature=0.3,
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


def run_analysis(
    symbol: str, market: str = "auto", include_sentiment: bool = True
) -> AnalysisReport:
    data = fetch_stock_data(symbol, market)

    # ---- Step 1: 基本面 + 技术面 ----
    logger.info("Step 1/4: 基本面 + 技术面分析")
    fund_json = fundamental_chain.invoke(
        {
            "name": data.info.name,
            "symbol": data.info.symbol,
            "financial_data": data.financial_text,
        }
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
        sent_json = sentiment_chain.invoke(
            {
                "name": data.info.name,
                "symbol": data.info.symbol,
                "news_data": data.news_text,
            }
        )
        sentiment_summary = SentimentSummary(**sent_json)

    # ---- Step 3: 多头 + 空头辩论（并行） ----
    logger.info("Step 3/4: 多空辩论")
    news_text = data.news_text if data.news_text else "无近期新闻数据"
    debate = debate_chain.invoke(
        {
            "name": data.info.name,
            "symbol": data.info.symbol,
            "fundamental_summary": fund_summary,
            "technical_data": tech_summary or "暂无技术面数据",
            "news_data": news_text,
        }
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
    arb_json = arbitrator_chain.invoke(
        {
            "fundamental_summary": fund_summary,
            "technical_data": tech_summary or "暂无技术面数据",
            "bull_thesis": bull_json["viewpoint"],
            "bear_thesis": bear_json["viewpoint"],
        }
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
