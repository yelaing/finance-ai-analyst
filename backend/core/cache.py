"""进程内缓存：按命名空间隔离的 TTL + LRU。

命名空间各自持有独立的 cache，因为 cachetools.TTLCache 的 ttl 是**整个 cache 一个**、
不支持逐条指定；而数据源要短 TTL（新鲜度）与 LLM 要长 TTL（确定性记忆化）语义不同。

**已知且接受的局限：缓存击穿。** 两个线程同时未命中同一键会各算一次（LLM 多花一次钱）。
本项目单用户、单 worker，不值得为此引入分布式锁。

切片 5 引入 Redis 时，只需让 get_cache() 返回另一个实现 —— 调用点只用 get/set。
"""

import functools
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from cachetools import TTLCache

from backend.core.metrics import get_metrics

FETCH = "fetch"
LLM = "llm"
EMBED = "embed"
HEALTH = "health"

# 新增调用点时同步加到这里：NoOpCache 靠它给出各命名空间的尺寸，漏加会让两者的键集合悄悄分叉
NAMESPACES = (FETCH, LLM, EMBED, HEALTH)


def cache_key(*parts: Any, **params: Any) -> str:
    """稳定哈希。不用内置 hash() —— 它对 str 每进程随机化，跨进程不一致。"""
    material = json.dumps([parts, params], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class MemoryCache:
    """按命名空间分桶的 TTL + LRU 缓存。

    timer 可注入，测试用假时钟验证过期，不必真的 sleep。
    """

    def __init__(
        self,
        specs: Mapping[str, tuple[int, float]],
        timer: Callable[[], float] | None = None,
    ) -> None:
        self._timer = timer or time.monotonic
        self._caches: dict[str, TTLCache[str, Any]] = {
            namespace: TTLCache(maxsize=maxsize, ttl=ttl, timer=self._timer)
            for namespace, (maxsize, ttl) in specs.items()
        }

    def _bucket(self, namespace: str) -> TTLCache[str, Any]:
        bucket = self._caches.get(namespace)
        if bucket is None:
            # 拼错命名空间会导致「永远不缓存」这种静默失效，所以直接报错
            raise KeyError(f"未知缓存命名空间 {namespace!r}，已注册的：{sorted(self._caches)}")
        return bucket

    def get(self, namespace: str, key: str) -> Any | None:
        value = self._bucket(namespace).get(key)
        # 在缓存自身计数，而不是各调用点 —— 覆盖率天然完整，以后新增调用点也不会漏
        get_metrics().observe_cache(namespace, value is not None)
        return value

    def set(self, namespace: str, key: str, value: Any) -> None:
        self._bucket(namespace)[key] = value

    def clear(self) -> None:
        for bucket in self._caches.values():
            bucket.clear()

    def sizes(self) -> dict[str, int]:
        return {namespace: len(bucket) for namespace, bucket in self._caches.items()}


class NoOpCache:
    """CACHE_ENABLED=false 时替代 MemoryCache，调用点无需写分支。"""

    def get(self, namespace: str, key: str) -> Any | None:
        return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        return None

    def clear(self) -> None:
        return None

    def sizes(self) -> dict[str, int]:
        return dict.fromkeys(NAMESPACES, 0)


@functools.lru_cache
def get_cache() -> MemoryCache | NoOpCache:
    """进程内单例。测试改配置后需 get_cache.cache_clear()。"""
    from backend.config import get_settings

    settings = get_settings()
    if not settings.cache_enabled:
        return NoOpCache()
    return MemoryCache(
        {
            FETCH: (settings.cache_maxsize, settings.cache_ttl_seconds),
            LLM: (settings.cache_maxsize, settings.cache_llm_ttl_seconds),
            EMBED: (settings.cache_maxsize, settings.cache_llm_ttl_seconds),
            HEALTH: (settings.cache_maxsize, settings.health_probe_ttl_seconds),
        }
    )
