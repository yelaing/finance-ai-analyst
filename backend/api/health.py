"""依赖连通性检查。

`/health` 是**存活**探针：进程活着就 200，只顺带报各依赖的状态。
`/health/ready` 是**就绪**探针：真去检依赖，不可用就 503。

分开的理由不是形式主义：容器编排通常用存活探针决定「要不要重启容器」。
让 LLM 的一次短暂限流触发容器重启是错误的 —— 重启解决不了上游限流，
还会把正在处理的请求全部打断。就绪探针则适合接入监控告警与流量摘除。
"""

import logging
from dataclasses import dataclass
from typing import Literal

import chromadb
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from backend.config import get_settings
from backend.core.cache import HEALTH, get_cache
from backend.store.chroma_store import COLLECTION_NAME

logger = logging.getLogger(__name__)

# 探测本身要快：卡住的探针比不探还糟（会把健康检查拖成超时）
_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


class HealthResponse(BaseModel):
    """存活探针响应。status 只会是 ok / degraded —— **依赖不可用也不改 HTTP 状态码**。"""

    status: Literal["ok", "degraded"]
    dependencies: list[DependencyStatus]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "dependencies": [
                        {"name": "chromadb", "ok": True, "detail": "12 条向量"},
                        {"name": "llm", "ok": True, "detail": "261 个模型可用"},
                    ],
                }
            ]
        }
    )


class ReadyResponse(BaseModel):
    """就绪探针响应。status 为 not_ready 时 HTTP 状态码是 503。"""

    status: Literal["ready", "not_ready"]
    dependencies: list[DependencyStatus]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "not_ready",
                    "dependencies": [
                        {"name": "chromadb", "ok": True, "detail": "12 条向量"},
                        {
                            "name": "llm",
                            "ok": False,
                            "detail": "APITimeoutError: Request timed out.",
                        },
                    ],
                }
            ]
        }
    )


def check_llm() -> DependencyStatus:
    """打一次兼容接口的 /models 列表：能验证 key 有效 + 端点可达，且不花 token。

    结果缓存若干秒 —— 探针可能被容器编排高频调用，没必要每次都打 provider。
    失败结果同样缓存：上游挂了的时候更要避免被探针反复冲击。
    """
    cache = get_cache()
    cached = cache.get(HEALTH, "llm")
    if cached is not None:
        return cached

    settings = get_settings()
    try:
        client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        models = client.models.list()
        status = DependencyStatus("llm", True, f"{len(models.data)} 个模型可用")
    except Exception as exc:
        logger.warning("LLM 连通性探测失败: %s", exc)
        status = DependencyStatus("llm", False, f"{type(exc).__name__}: {exc}")

    cache.set(HEALTH, "llm", status)
    return status


def check_chroma() -> DependencyStatus:
    """只读探测：开一个独立的客户端数一下集合。

    刻意**不用** app 的 store 单例：它初始化时会做索引回填（可能触发 embedding
    调用），健康探针不该有这种副作用。已实测两个 PersistentClient 并存无争用。
    """
    settings = get_settings()
    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        names = [collection.name for collection in client.list_collections()]
        if COLLECTION_NAME not in names:
            return DependencyStatus("chromadb", True, "可访问，集合尚未创建")
        count = client.get_collection(COLLECTION_NAME, embedding_function=None).count()
        return DependencyStatus("chromadb", True, f"{count} 条向量")
    except Exception as exc:
        logger.warning("ChromaDB 连通性探测失败: %s", exc)
        return DependencyStatus("chromadb", False, f"{type(exc).__name__}: {exc}")


def check_all() -> list[DependencyStatus]:
    return [check_chroma(), check_llm()]


def is_ready(statuses: list[DependencyStatus]) -> bool:
    return all(status.ok for status in statuses)
