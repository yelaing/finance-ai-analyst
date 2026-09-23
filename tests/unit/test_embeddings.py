import logging

import pytest

import backend.store.embeddings as emb


@pytest.fixture
def sleep_calls(monkeypatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(emb.time, "sleep", lambda seconds: calls.append(seconds))
    return calls


def test_embed_texts_chunks_by_provider_limit(monkeypatch):
    """DashScope text-embedding-v3 单次上限 10 条，11 条就会 400。"""
    sizes: list[int] = []

    def fake_once(batch):
        sizes.append(len(batch))
        return [[0.0] * 4 for _ in batch]

    monkeypatch.setattr(emb, "_embed_once", fake_once)
    result = emb.embed_texts([f"t{i}" for i in range(25)])

    assert sizes == [10, 10, 5]
    assert len(result) == 25


def test_embed_texts_preserves_input_order(monkeypatch):
    monkeypatch.setattr(emb, "_embed_once", lambda b: [[float(len(t))] * 2 for t in b])
    assert [v[0] for v in emb.embed_texts(["a", "bbbbb", "ccc"])] == [1.0, 5.0, 3.0]


def test_embed_texts_empty_input_makes_no_call(monkeypatch):
    calls = []

    def fake_once(batch):
        calls.append(batch)
        return []

    monkeypatch.setattr(emb, "_embed_once", fake_once)
    assert emb.embed_texts([]) == []
    assert calls == []


def test_embed_batch_retries_then_succeeds(monkeypatch, sleep_calls, caplog):
    attempts = []

    def flaky(batch):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError(f"第 {len(attempts)} 次失败")
        return [[1.0]]

    monkeypatch.setattr(emb, "_embed_once", flaky)
    with caplog.at_level(logging.WARNING):
        assert emb._embed_batch(["x"]) == [[1.0]]

    assert len(attempts) == 3
    assert sleep_calls == [1, 2]  # 退避序列 1s / 2s
    assert len(caplog.records) == 2
    assert "第 1/3 次尝试" in caplog.text
    assert "第 2/3 次尝试" in caplog.text


def test_embed_batch_raises_the_last_error(monkeypatch, sleep_calls):
    attempts = []

    def always_fail(batch):
        attempts.append(1)
        raise ValueError(f"第 {len(attempts)} 次失败")

    monkeypatch.setattr(emb, "_embed_once", always_fail)
    with pytest.raises(ValueError, match="第 3 次失败"):
        emb._embed_batch(["x"])

    assert len(attempts) == 3
    assert sleep_calls == [1, 2]  # 最后一次失败不再退避，直接抛出


def test_embed_batch_does_not_log_on_final_failure(monkeypatch, sleep_calls, caplog):
    """最后一次失败的告警由调用方决定怎么处理，这里只负责抛。"""
    monkeypatch.setattr(emb, "_embed_once", lambda b: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        emb._embed_batch(["x"])
    assert len(caplog.records) == 2  # 只有前两次尝试记了日志


def test_embed_once_uses_configured_model_and_maps_results(monkeypatch):
    """_embed_once 是唯一真正打到 provider 的地方，替换 client 验证入参出参。"""
    import types

    captured = {}

    class FakeEmbeddings:
        def create(self, model, input):
            captured["model"] = model
            captured["input"] = input
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=[0.1, 0.2]),
                    types.SimpleNamespace(embedding=[0.3, 0.4]),
                ]
            )

    monkeypatch.setattr(emb, "_client", types.SimpleNamespace(embeddings=FakeEmbeddings()))
    assert emb._embed_once(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert captured == {"model": emb._settings.embedding_model, "input": ["a", "b"]}
