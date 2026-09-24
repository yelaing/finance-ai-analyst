"""依赖探测本身（端点测试替换了 check_all，所以这里覆盖真实探测逻辑）。"""

import types

import chromadb
import pytest

from backend.api import health
from backend.config import Settings
from backend.store.chroma_store import COLLECTION_NAME


class FakeModels:
    def __init__(self, count: int) -> None:
        self._count = count
        self.calls = 0

    def list(self) -> types.SimpleNamespace:
        self.calls += 1
        return types.SimpleNamespace(data=list(range(self._count)))


@pytest.fixture
def fake_openai(monkeypatch):
    """替换 SDK 客户端，避免探测打真实网络。返回构造参数的记录表。"""
    constructed: list[dict] = []
    models = FakeModels(3)

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            constructed.append(kwargs)
            self.models = models

    monkeypatch.setattr(health, "OpenAI", FakeClient)
    return constructed, models


def test_check_llm_reports_ok(fake_openai):
    constructed, _ = fake_openai
    status = health.check_llm()

    assert status.ok is True
    assert "3 个模型" in status.detail
    assert constructed[0]["timeout"] == health._PROBE_TIMEOUT_SECONDS


def test_check_llm_caches_result_so_probes_do_not_hammer_provider(fake_openai):
    constructed, models = fake_openai

    first = health.check_llm()
    second = health.check_llm()

    assert second is first
    assert models.calls == 1  # 第二次读缓存，没再打 provider
    assert len(constructed) == 1


def test_check_llm_caches_failures_too(monkeypatch):
    """上游挂掉的时候更要避免被高频探针反复冲击。"""
    attempts = []

    class Exploding:
        def __init__(self, **kwargs) -> None:
            attempts.append(1)
            raise RuntimeError("连接失败")

    monkeypatch.setattr(health, "OpenAI", Exploding)

    first = health.check_llm()
    second = health.check_llm()

    assert first.ok is False
    assert "RuntimeError" in first.detail
    assert second is first
    assert len(attempts) == 1


def test_check_chroma_reports_vector_count(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    client.create_collection(
        name=COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(
        health,
        "get_settings",
        lambda: Settings(_env_file=None, chroma_persist_dir=str(chroma_dir)),
    )

    status = health.check_chroma()
    assert status.ok is True
    assert "0 条向量" in status.detail


def test_check_chroma_treats_missing_collection_as_reachable(monkeypatch, tmp_path):
    """全新安装还没跑过分析时集合不存在 —— 那是「还没用」不是「不可用」。"""
    monkeypatch.setattr(
        health,
        "get_settings",
        lambda: Settings(_env_file=None, chroma_persist_dir=str(tmp_path / "chroma")),
    )

    status = health.check_chroma()
    assert status.ok is True
    assert "尚未创建" in status.detail


def test_check_chroma_reports_unusable_path(monkeypatch, tmp_path):
    not_a_dir = tmp_path / "file-not-dir"
    not_a_dir.write_text("我是文件，不是目录", encoding="utf-8")
    monkeypatch.setattr(
        health,
        "get_settings",
        lambda: Settings(_env_file=None, chroma_persist_dir=str(not_a_dir)),
    )

    status = health.check_chroma()
    assert status.ok is False
    assert status.detail


def test_is_ready_requires_every_dependency():
    ok = health.DependencyStatus("a", True, "")
    down = health.DependencyStatus("b", False, "炸了")

    assert health.is_ready([ok, ok]) is True
    assert health.is_ready([ok, down]) is False
    assert health.is_ready([]) is True  # 没有依赖时不该判为不可用


def test_dependency_status_serializes():
    status = health.DependencyStatus("llm", True, "3 个模型")
    assert status.as_dict() == {"name": "llm", "ok": True, "detail": "3 个模型"}
