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
