from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalysisRequest(BaseModel):
    symbol: str = Field(
        description="股票代码：A 股为 6 位数字（600519），美股为字母代码（AAPL）",
        examples=["600519"],
    )
    market: str = Field(
        default="auto",
        description="市场：'a_share' / 'us' / 'auto'（按代码格式自动识别）",
        examples=["auto"],
    )
    include_sentiment: bool = Field(
        default=True,
        description="是否跑舆情分析；关闭可省下一次 LLM 调用",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"symbol": "600519", "market": "auto", "include_sentiment": True}]
        }
    )


class MetricItem(BaseModel):
    label: str
    value: str
    yoy_change: str | None = None
    assessment: str  # "positive", "neutral", "negative"


class SentimentSummary(BaseModel):
    overall: str  # "bullish", "neutral", "bearish"
    score: float = Field(ge=-1.0, le=1.0)
    key_drivers: list[str]


class RiskItem(BaseModel):
    category: str
    level: str  # "low", "medium", "high"
    detail: str


class TechnicalIndicators(BaseModel):
    ma_5: float | None = None
    ma_20: float | None = None
    ma_60: float | None = None
    macd_dif: float | None = None
    macd_dea: float | None = None
    macd_bar: float | None = None
    rsi_14: float | None = None
    kdj_k: float | None = None
    kdj_d: float | None = None
    kdj_j: float | None = None
    trend_summary: str = ""
    price_history: list[dict] = []


class DebateThesis(BaseModel):
    analyst: str  # "bull" or "bear"
    viewpoint: str
    key_evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# 给 /docs 一份完整报告的形状示例。它必须是 schema 合法的（有测试逐个校验），
# 且把可选字段（舆情/技术面/多空）都填上，读者一次就能看全报告结构。
_REPORT_EXAMPLE: dict[str, Any] = {
    "symbol": "600519",
    "name": "贵州茅台",
    "timestamp": "2026-09-24T10:00:00",
    "fundamental_metrics": [
        {
            "label": "营收增速",
            "value": "6.34%",
            "yoy_change": "-3.2个百分点",
            "assessment": "neutral",
        }
    ],
    "fundamental_summary": "盈利质量优异、现金流充沛，但增速回落已成事实。",
    "sentiment": {"overall": "neutral", "score": 0.0, "key_drivers": ["渠道压力"]},
    "risks": [{"category": "增长可持续性", "level": "high", "detail": "净利增速仅 1.47%"}],
    "conclusion": "基本面依然稳健但增长斜率放缓，属「健康但失速」状态。",
    "sources": ["同花顺财务摘要"],
    "technical": {
        "ma_5": 1256.34,
        "ma_20": 1280.1,
        "rsi_14": 37.09,
        "trend_summary": "最新收盘价 1240.0，位于 MA20 (1280.1) 下方，短期均线偏空。",
        "price_history": [],
    },
    "bull_thesis": {
        "analyst": "bull",
        "viewpoint": "盈利质量与护城河未变",
        "key_evidence": ["毛利率 89.76%"],
        "confidence": 0.7,
    },
    "bear_thesis": {
        "analyst": "bear",
        "viewpoint": "增速换挡，估值溢价承压",
        "key_evidence": ["净利增速 1.47%"],
        "confidence": 0.6,
    },
    "debate_verdict": "分歧在于增速下移是阶段性还是结构性。",
}


class AnalysisReport(BaseModel):
    symbol: str
    name: str
    timestamp: datetime
    fundamental_metrics: list[MetricItem]
    fundamental_summary: str
    sentiment: SentimentSummary | None = None
    risks: list[RiskItem]
    conclusion: str
    sources: list[str] = []
    technical: TechnicalIndicators | None = None
    bull_thesis: DebateThesis | None = None
    bear_thesis: DebateThesis | None = None
    debate_verdict: str | None = None

    model_config = ConfigDict(json_schema_extra={"examples": [_REPORT_EXAMPLE]})


class AnalysisResponse(BaseModel):
    status: str
    report: AnalysisReport | None = None
    error: str | None = None
    from_cache: bool = False
    cached_at: datetime | None = None


class HistoryItem(BaseModel):
    id: str
    symbol: str
    name: str
    timestamp: datetime
    summary: str


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "items": [
                        {
                            "id": "600519_20260924_100000",
                            "symbol": "600519",
                            "name": "贵州茅台",
                            "timestamp": "2026-09-24T10:00:00",
                            "summary": "同花顺财务摘要",
                        }
                    ],
                    "total": 1,
                }
            ]
        }
    )


class SearchHit(BaseModel):
    id: str
    symbol: str
    name: str
    timestamp: datetime
    score: float = Field(description="余弦相似度（0~1），越大越相似")
    text: str = Field(description="被向量化的关键文本，便于判断命中原因")


class SearchResponse(BaseModel):
    items: list[SearchHit]
    total: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "items": [
                        {
                            "id": "002594_20260924_090000",
                            "symbol": "002594",
                            "name": "比亚迪",
                            "timestamp": "2026-09-24T09:00:00",
                            "score": 0.8123,
                            "text": "比亚迪（002594） 基本面：公司主营新能源汽车与动力电池业务。",
                        }
                    ],
                    "total": 1,
                }
            ]
        }
    )
