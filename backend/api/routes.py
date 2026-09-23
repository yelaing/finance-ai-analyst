from datetime import datetime

from fastapi import APIRouter, Query
from fastapi.responses import Response

from backend.core.errors import AppError, InvalidRequestError, ReportNotFoundError
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

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["analysis"],
    summary="分析一支股票",
    description="跑完整的四阶段流水线（基本面+技术面 → 舆情 → 多空辩论 → 风控仲裁）。"
    "当天已分析过的股票会直接返回缓存结果（from_cache=true），不再重复调用 LLM。",
)
def analyze(req: AnalysisRequest) -> AnalysisResponse:
    store = get_store()

    try:
        cached = store.get_latest(req.symbol)
        if cached:
            ts = cached.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            # 历史记录缺 timestamp 时不当作缓存命中，直接重新分析
            if isinstance(ts, datetime) and ts.date() == datetime.now().date():
                report = AnalysisReport(**cached)
                return AnalysisResponse(status="ok", report=report, from_cache=True, cached_at=ts)

        report = run_analysis(req.symbol, req.market, req.include_sentiment)
        store.save(report)
        return AnalysisResponse(status="ok", report=report)
    except AppError:
        # 业务异常自带错误码与状态码，交给全局处理器
        raise
    except ValueError as e:
        # 存量代码里的参数类 ValueError 仍按 400 处理，保持原有契约
        raise InvalidRequestError(str(e)) from e
    except Exception as e:
        # 真实异常只进日志，响应里不暴露内部细节
        raise AppError("分析失败，请稍后重试") from e


@router.get(
    "/history",
    response_model=HistoryResponse,
    tags=["history"],
    summary="按股票代码查历史分析记录",
    description="精确检索（不是语义检索）。按时间倒序返回，summary 字段是数据来源列表。",
)
def history(
    symbol: str = Query(description="股票代码，如 600519 或 AAPL"),
    limit: int = Query(default=20, le=50, description="最多返回多少条"),
) -> HistoryResponse:
    store = get_store()
    items = store.get_history(symbol=symbol, limit=limit)
    return HistoryResponse(items=items, total=len(items))


@router.get(
    "/search",
    response_model=SearchResponse,
    tags=["history"],
    summary="语义检索历史报告",
    description="把 query 向量化后在 ChromaDB 里做余弦相似度检索，可跨股票；"
    "传入 symbol 则限定在该股票范围内。score 为余弦相似度，越大越相似。",
)
def search(
    q: str = Query(description="自然语言检索词，如「高风险的消费类股票分析」"),
    limit: int = Query(default=5, le=20, description="最多返回多少条"),
    symbol: str | None = Query(default=None, description="限定在某支股票范围内检索"),
) -> SearchResponse:
    store = get_store()
    items = store.search_semantic(q, n=limit, symbol=symbol)
    return SearchResponse(items=items, total=len(items))


@router.get(
    "/export/{symbol}",
    tags=["history"],
    summary="导出最新报告为 Markdown",
    description="把该股票最近一次分析报告渲染成 Markdown 文件下载。",
)
def export_report(symbol: str) -> Response:
    store = get_store()
    cached = store.get_latest(symbol)
    if not cached:
        raise ReportNotFoundError("未找到该股票的分析报告")
    report = AnalysisReport(**cached)
    md = render_markdown(report)
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={symbol}_report.md"},
    )
