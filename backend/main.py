import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.errors import register_exception_handlers
from backend.api.middleware import RequestContextMiddleware
from backend.api.routes import router as api_router
from backend.config import get_settings
from backend.core.logging import setup_logging
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
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=not settings.is_prod,
    )
