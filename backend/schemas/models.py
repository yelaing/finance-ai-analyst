from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class AnalysisRequest(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. 600519 (A-share) or AAPL (US)")
    market: str = Field(default="auto", description="Market: 'a_share', 'us', or 'auto'")
    include_sentiment: bool = Field(default=True)


class MetricItem(BaseModel):
    label: str
    value: str
    yoy_change: Optional[str] = None
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
    ma_5: Optional[float] = None
    ma_20: Optional[float] = None
    ma_60: Optional[float] = None
    macd_dif: Optional[float] = None
    macd_dea: Optional[float] = None
    macd_bar: Optional[float] = None
    rsi_14: Optional[float] = None
    kdj_k: Optional[float] = None
    kdj_d: Optional[float] = None
    kdj_j: Optional[float] = None
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
    sentiment: Optional[SentimentSummary] = None
    risks: list[RiskItem]
    conclusion: str
    sources: list[str] = []
    technical: Optional[TechnicalIndicators] = None
    bull_thesis: Optional[DebateThesis] = None
    bear_thesis: Optional[DebateThesis] = None
    debate_verdict: Optional[str] = None


class AnalysisResponse(BaseModel):
    status: str
    report: Optional[AnalysisReport] = None
    error: Optional[str] = None
    from_cache: bool = False
    cached_at: Optional[datetime] = None


class HistoryItem(BaseModel):
    id: str
    symbol: str
    name: str
    timestamp: datetime
    summary: str


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
