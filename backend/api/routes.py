import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from backend.pipeline.analyzer import run_analysis
from backend.schemas.models import (
    AnalysisRequest,
    AnalysisResponse,
    HistoryResponse,
)
from backend.store.chroma_store import get_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest):
    """Run financial analysis on a stock."""
    store = get_store()

    try:
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
