import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from backend.pipeline.analyzer import run_analysis
from backend.pipeline.report_export import render_markdown
from backend.schemas.models import (
    AnalysisReport,
    AnalysisRequest,
    AnalysisResponse,
    HistoryResponse,
    SearchResponse,
)
from backend.store.chroma_store import get_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest):
    """Run financial analysis on a stock."""
    store = get_store()

    try:
        cached = store.get_latest(req.symbol)
        if cached:
            ts = cached.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts.date() == datetime.now().date():
                from backend.schemas.models import AnalysisReport
                report = AnalysisReport(**cached)
                return AnalysisResponse(status="ok", report=report, from_cache=True, cached_at=ts)

        report = run_analysis(req.symbol, req.market, req.include_sentiment)
        store.save(report)
        return AnalysisResponse(status="ok", report=report)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed for %s", req.symbol)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=HistoryResponse)
def history(symbol: str = Query(description="Stock ticker"), limit: int = Query(default=20, le=50)):
    store = get_store()
    items = store.get_history(symbol=symbol, limit=limit)
    return HistoryResponse(items=items, total=len(items))


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(description="自然语言检索词，如「高风险的消费类股票分析」"),
    limit: int = Query(default=5, le=20),
    symbol: str | None = Query(default=None, description="限定在某支股票范围内检索"),
):
    """Semantic search over stored reports."""
    store = get_store()
    items = store.search_semantic(q, n=limit, symbol=symbol)
    return SearchResponse(items=items, total=len(items))


@router.get("/export/{symbol}")
def export_report(symbol: str):
    """Export latest analysis report as Markdown."""
    store = get_store()
    cached = store.get_latest(symbol)
    if not cached:
        raise HTTPException(status_code=404, detail="未找到该股票的分析报告")
    report = AnalysisReport(**cached)
    md = render_markdown(report)
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={symbol}_report.md"},
    )
