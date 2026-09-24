import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.api.errors import register_exception_handlers
from backend.api.health import check_all, is_ready
from backend.api.middleware import RequestContextMiddleware
from backend.api.routes import router as api_router
from backend.config import get_settings
from backend.core.logging import setup_logging
from backend.core.metrics import render
from backend.core.net import apply_network_env

settings = get_settings()
setup_logging(settings)

logger = logging.getLogger(__name__)

if settings.is_prod and settings.cors_origins == ["*"]:
    logger.warning("生产环境 CORS_ORIGINS 仍为 ['*']，建议显式列出允许的来源")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """网络环境变量在**服务真正启动时**才应用。

    刻意不放模块顶层：那会让任何 `import backend.main` 的一方（测试、脚本）
    都被动改掉进程的全局环境变量 —— 实测这会污染测试断言。放这里则
    import 无副作用，uvicorn 启动时会正常执行。
    """
    apply_network_env(settings)
    yield


app = FastAPI(
    title="Finance AI Analyst",
    version="0.1.0",
    description="上市公司财报 + 舆情 + 多空辩论的自动化分析服务。",
    lifespan=lifespan,
)

# 内层：业务请求都经过它，用于分配 request_id 与记录耗时
app.add_middleware(RequestContextMiddleware)
# 外层：保证错误响应（含中间件自身产生的 500）也带上 CORS 头
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["system"], summary="存活探针")
def health() -> dict[str, object]:
    """进程活着就返回 200，顺带报各依赖的状态。

    **不因为依赖不可用而改状态码** —— 存活探针失败会让编排去重启容器，
    而上游限流/短暂不可用重启本地进程是解决不了的，只会打断正在处理的请求。
    需要严格判定的场景用 /health/ready。
    """
    statuses = check_all()
    return {
        "status": "ok" if is_ready(statuses) else "degraded",
        "dependencies": [status.as_dict() for status in statuses],
    }


@app.get("/health/ready", tags=["system"], summary="就绪探针")
def health_ready(response: Response) -> dict[str, object]:
    """依赖不可用返回 503，便于接监控告警与流量摘除。

    这里刻意不走统一错误响应体：探针要的是机器可读的逐依赖状态，
    而不是面向调用方的错误码结构。
    """
    statuses = check_all()
    ready = is_ready(statuses)
    if not ready:
        response.status_code = 503
    return {
        "status": "ready" if ready else "not_ready",
        "dependencies": [status.as_dict() for status in statuses],
    }


@app.get("/metrics", summary="Prometheus 指标", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus 文本格式。不进 OpenAPI —— 它不是业务接口。"""
    content, content_type = render()
    return Response(content=content, media_type=content_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=not settings.is_prod,
    )
