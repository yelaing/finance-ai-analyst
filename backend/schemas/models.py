from datetime import datetime

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. 600519 (A-share) or AAPL (US)")
    market: str = Field(default="auto", description="Market: 'a_share', 'us', or 'auto'")
    include_sentiment: bool = Field(default=True)


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


class SearchHit(BaseModel):
    id: str
    symbol: str
    name: str
    timestamp: datetime
    score: float  # 余弦相似度 0~1，越大越相似
    text: str


class SearchResponse(BaseModel):
    items: list[SearchHit]
    total: int
