import json
import re
import logging
from datetime import datetime

from backend.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from backend.pipeline.data_fetcher import fetch_stock_data
from backend.pipeline.prompts import (
    FUNDAMENTAL_ANALYSIS_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
    CROSS_VALIDATION_PROMPT,
)
from backend.schemas.models import (
    AnalysisReport,
    MetricItem,
    SentimentSummary,
    RiskItem,
)

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    return json.loads(text)


def _call_llm(prompt: str) -> str:
    """Call LLM via OpenAI-compatible API and return raw text."""
    from openai import OpenAI

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


def run_analysis(symbol: str, market: str = "auto", include_sentiment: bool = True) -> AnalysisReport:
    data = fetch_stock_data(symbol, market)

    logger.info("Step 1/3: 基本面分析")
    fund_prompt = FUNDAMENTAL_ANALYSIS_PROMPT.format(
        name=data.info.name,
        symbol=data.info.symbol,
        financial_data=data.financial_text,
    )
    fund_raw = _call_llm(fund_prompt)
    fund_json = _extract_json(fund_raw)
    metrics = [MetricItem(**m) for m in fund_json["metrics"]]
    fund_summary = fund_json["summary"]

    sentiment_summary = None
    if include_sentiment and data.news_text and "暂无" not in data.news_text:
        logger.info("Step 2/3: 舆情分析")
        sent_prompt = SENTIMENT_ANALYSIS_PROMPT.format(
            name=data.info.name,
            symbol=data.info.symbol,
            news_data=data.news_text,
        )
        sent_raw = _call_llm(sent_prompt)
        sent_json = _extract_json(sent_raw)
        sentiment_summary = SentimentSummary(**sent_json)

    logger.info("Step 3/3: 交叉验证")
    cross_prompt = CROSS_VALIDATION_PROMPT.format(
        fundamental_summary=fund_summary,
        sentiment_summary=sentiment_summary.model_dump_json(indent=2) if sentiment_summary else "无舆情数据",
    )
    cross_raw = _call_llm(cross_prompt)
    cross_json = _extract_json(cross_raw)
    risks = [RiskItem(**r) for r in cross_json["risks"]]

    return AnalysisReport(
        symbol=data.info.symbol,
        name=data.info.name,
        timestamp=datetime.now(),
        fundamental_metrics=metrics,
        fundamental_summary=fund_summary,
        sentiment=sentiment_summary,
        risks=risks,
        conclusion=cross_json["conclusion"],
        sources=data.sources,
    )
