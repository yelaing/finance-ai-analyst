"""缓存层：TTL / LRU 淘汰 / 命名空间隔离 / 键稳定性 / 关闭模式。

TTL 用注入的假时钟验证，不真的 sleep —— 依赖真实时间的测试不可靠。
"""

import pytest

from backend.core.cache import (
    FETCH,
    LLM,
    NAMESPACES,
    MemoryCache,
    NoOpCache,
    cache_key,
    get_cache,
)


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def reset_caches() -> None:
    from backend.config import get_settings

    get_settings.cache_clear()
    get_cache.cache_clear()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache(clock: FakeClock) -> MemoryCache:
    return MemoryCache({FETCH: (4, 30.0), LLM: (8, 3600.0)}, timer=clock)


def test_set_then_get(cache):
    cache.set(FETCH, "k", {"v": 1})
    assert cache.get(FETCH, "k") == {"v": 1}


def test_miss_returns_none(cache):
    assert cache.get(FETCH, "missing") is None


def test_entry_expires_after_its_ttl(cache, clock):
    cache.set(FETCH, "k", 1)
    clock.advance(29.0)
    assert cache.get(FETCH, "k") == 1
    clock.advance(2.0)  # 累计 31s > 30s
    assert cache.get(FETCH, "k") is None


def test_namespaces_use_independent_ttl(cache, clock):
    """数据源 30s 过期时，LLM 的 1 小时条目必须还在。"""
    cache.set(FETCH, "k", "fetch")
    cache.set(LLM, "k", "llm")
    clock.advance(31.0)

    assert cache.get(FETCH, "k") is None
    assert cache.get(LLM, "k") == "llm"


def test_namespaces_use_independent_maxsize(cache):
    for index in range(5):
        cache.set(FETCH, f"f{index}", index)  # maxsize 4
    for index in range(5):
        cache.set(LLM, f"l{index}", index)  # maxsize 8

    assert cache.sizes() == {FETCH: 4, LLM: 5}


def test_maxsize_evicts_oldest(cache):
    for index in range(4):
        cache.set(FETCH, f"k{index}", index)
    cache.set(FETCH, "overflow", 99)

    assert cache.sizes()[FETCH] == 4
    assert cache.get(FETCH, "k0") is None
    assert cache.get(FETCH, "overflow") == 99


def test_unknown_namespace_raises(cache):
    """拼错命名空间会导致「永远不缓存」这种静默失效，必须直接报错。"""
    with pytest.raises(KeyError, match="未知缓存命名空间"):
        cache.get("typo", "k")
    with pytest.raises(KeyError, match="未知缓存命名空间"):
        cache.set("typo", "k", 1)


def test_clear_empties_every_namespace(cache):
    cache.set(FETCH, "a", 1)
    cache.set(LLM, "b", 2)
    cache.clear()
    assert cache.sizes() == {FETCH: 0, LLM: 0}


# ---------- 键的稳定性 ----------


def test_cache_key_is_order_independent_for_dicts():
    assert cache_key({"a": 1, "b": 2}) == cache_key({"b": 2, "a": 1})


def test_cache_key_distinguishes_payloads():
    assert cache_key({"prompt": "a"}) != cache_key({"prompt": "b"})


def test_cache_key_distinguishes_model_and_temperature():
    """只拿 prompt 做键的话，换了模型或温度仍会命中旧结果 —— 那是错的。"""
    payload = {"prompt": "同一段 prompt"}
    keys = {
        cache_key(payload, model="qwen-plus", temperature=0.3),
        cache_key(payload, model="gpt-4o", temperature=0.3),
        cache_key(payload, model="qwen-plus", temperature=0.7),
    }
    assert len(keys) == 3


def test_cache_key_is_stable_across_calls():
    """不能用内置 hash()：它对 str 每进程随机化，跨进程不一致。"""
    assert cache_key("文本", model="m") == cache_key("文本", model="m")


# ---------- 关闭模式 ----------


def test_noop_cache_never_stores():
    cache = NoOpCache()
    cache.set(FETCH, "k", 1)
    assert cache.get(FETCH, "k") is None
    cache.clear()
    assert cache.sizes() == dict.fromkeys(NAMESPACES, 0)


def test_get_cache_returns_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "false")
    reset_caches()
    assert isinstance(get_cache(), NoOpCache)


def test_get_cache_returns_memory_cache_by_default(monkeypatch):
    monkeypatch.delenv("CACHE_ENABLED", raising=False)
    reset_caches()
    cache = get_cache()
    assert isinstance(cache, MemoryCache)
    # 用的是共享的 NAMESPACES 常量：漏加命名空间会让 NoOp 与 Memory 的键集合悄悄分叉
    assert set(cache.sizes()) == set(NAMESPACES)


def test_get_cache_is_a_singleton(monkeypatch):
    monkeypatch.delenv("CACHE_ENABLED", raising=False)
    reset_caches()
    assert get_cache() is get_cache()
