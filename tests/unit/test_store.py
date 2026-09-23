"""报告存储测试：双写、自愈、模型不变量、fail-soft、语义检索链路。

全部在 tmp 目录里跑，`data_file` / `chroma_dir` 由 fixture 注入，绝不碰真实 data/。
"""

import json
import logging
import types
from datetime import datetime
from pathlib import Path

import pytest

import backend.store.chroma_store as cs
from backend.store.chroma_store import COLLECTION_NAME, ReportStore
from tests.conftest import build_report


@pytest.fixture
def counting_embed(monkeypatch) -> list[int]:
    """记录每次向量化调用涉及的文本条数，用于断言「有没有白做功」。"""
    from tests.conftest import fake_embed

    calls: list[int] = []

    def wrapper(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    monkeypatch.setattr(cs, "embed_texts", wrapper)
    return calls


# ---------- 文本与元数据构造 ----------


def test_title_prefers_company_name():
    assert cs._title({"symbol": "600519", "name": "贵州茅台"}) == "贵州茅台（600519）"


def test_title_falls_back_to_symbol_when_name_is_symbol():
    """历史数据里有些记录的 name 直接就是代码，别渲染成 600519（600519）。"""
    assert cs._title({"symbol": "600519", "name": "600519"}) == "600519"
    assert cs._title({"symbol": "600519"}) == "600519"


def test_meta_carries_exactly_symbol_name_timestamp():
    meta = cs._meta({"symbol": "600519", "name": "贵州茅台", "timestamp": "2026-09-24T10:00:00"})
    assert meta == {
        "symbol": "600519",
        "name": "贵州茅台",
        "timestamp": "2026-09-24T10:00:00",
    }


def test_build_doc_includes_all_field_groups():
    record = {
        "id": "x",
        "symbol": "600519",
        "name": "贵州茅台",
        "timestamp": "2026-09-24T10:00:00",
        "fundamental_summary": "营收稳健",
        "conclusion": "综合结论在此",
        "full_report": build_report().model_dump(mode="json"),
    }
    doc = cs._build_doc(record)
    assert "贵州茅台（600519） 基本面：营收稳健" in doc
    assert "综合结论：综合结论在此" in doc
    assert "风险点：增长可持续性(high): 增速回落" in doc
    assert "舆情：neutral，得分 0.1" in doc
    assert "驱动因素：渠道压力" in doc
    assert "多空分歧：分歧在于增速是否结构性下移。" in doc


def test_build_doc_skips_absent_field_groups():
    """四阶段流水线之前的旧记录没有 technical / 多空 / 舆情，不能崩。"""
    record = {
        "id": "old",
        "symbol": "600519",
        "name": "600519",
        "timestamp": "2026-06-04T10:46:11",
        "fundamental_summary": "老记录",
        "conclusion": "老结论",
        "full_report": {"risks": [], "sentiment": None},
    }
    assert cs._build_doc(record) == "600519 基本面：老记录\n综合结论：老结论"


def test_build_doc_handles_missing_conclusion_and_full_report():
    """最残缺的记录：没有结论、没有 full_report，也不该崩。"""
    record = {"id": "x", "symbol": "600519", "name": "600519", "fundamental_summary": "只有基本面"}
    assert cs._build_doc(record) == "600519 基本面：只有基本面"


# ---------- save：双写 ----------


def test_save_writes_json_and_vector(store, tmp_data_dir):
    doc_id = store.save(build_report())

    assert doc_id == "600519_20260924_100000"
    records = json.loads((tmp_data_dir / "reports.json").read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    assert set(record) == {
        "id",
        "symbol",
        "name",
        "timestamp",
        "sentiment_score",
        "sources",
        "fundamental_summary",
        "conclusion",
        "full_report",
    }
    assert record["sentiment_score"] == 0.1

    col = store._collection
    assert col.count() == 1
    stored = col.get(ids=[doc_id], include=["documents", "metadatas"])
    assert "贵州茅台（600519） 基本面：" in stored["documents"][0]
    assert stored["metadatas"][0]["symbol"] == "600519"


def test_save_twice_with_same_id_does_not_duplicate_vector(store):
    report = build_report()
    store.save(report)
    store.save(report)
    assert store._collection.count() == 1


def test_save_without_sentiment_records_null_score(store, tmp_data_dir):
    store.save(build_report(with_sentiment=False))
    records = json.loads((tmp_data_dir / "reports.json").read_text(encoding="utf-8"))
    assert records[0]["sentiment_score"] is None


def test_save_is_fail_soft_when_embedding_fails(store, tmp_data_dir, monkeypatch, caplog):
    """向量索引写失败不能连累报告落盘，也不能抛异常给调用方。"""

    def boom(texts):
        raise RuntimeError("embedding 服务不可用")

    monkeypatch.setattr(cs, "embed_texts", boom)
    with caplog.at_level(logging.WARNING):
        doc_id = store.save(build_report())

    assert doc_id == "600519_20260924_100000"
    records = json.loads((tmp_data_dir / "reports.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in records] == [doc_id]
    assert store._collection.count() == 0  # 索引里确实没有
    assert "写入向量索引失败" in caplog.text


# ---------- 精确检索（读 JSON） ----------


def test_get_latest_returns_most_recent(store):
    store.save(build_report(timestamp=datetime(2026, 9, 22, 9, 0, 0), conclusion="旧"))
    store.save(build_report(timestamp=datetime(2026, 9, 24, 9, 0, 0), conclusion="新"))

    latest = store.get_latest("600519")
    assert latest is not None and latest["conclusion"] == "新"


def test_get_latest_returns_none_for_unknown_symbol(store):
    assert store.get_latest("NOPE") is None


def test_get_history_sorted_desc_and_limited(store):
    for day, name in ((22, "旧"), (23, "中"), (24, "新")):
        store.save(build_report(timestamp=datetime(2026, 9, day, 9, 0, 0), name=name))

    items = store.get_history(symbol="600519", limit=2)
    assert [i.name for i in items] == ["新", "中"]
    assert items[0].summary == "同花顺财务摘要"
    assert items[0].timestamp == datetime(2026, 9, 24, 9, 0, 0)


def test_get_history_without_symbol_returns_all(store):
    store.save(build_report(symbol="600519"))
    store.save(build_report(symbol="AAPL", name="Apple", timestamp=datetime(2026, 9, 24, 11, 0, 0)))
    assert len(store.get_history(limit=20)) == 2
    assert len(store.get_history(symbol="AAPL", limit=20)) == 1


# ---------- 语义检索（读 ChromaDB） ----------


def test_search_semantic_ranks_lexically_closest_first(store):
    store.save(
        build_report(
            symbol="002594",
            name="比亚迪",
            summary="公司主营新能源汽车与动力电池业务。",
            timestamp=datetime(2026, 9, 24, 9, 0, 0),
        )
    )
    store.save(
        build_report(
            symbol="600519",
            name="贵州茅台",
            summary="白酒行业盈利质量优异，毛利率极高。",
            timestamp=datetime(2026, 9, 24, 10, 0, 0),
        )
    )

    hits = store.search_semantic("新能源汽车与动力电池", n=2)
    assert hits[0]["symbol"] == "002594"
    assert 0.0 < hits[0]["score"] <= 1.0
    assert hits[0]["score"] >= hits[1]["score"]
    assert set(hits[0]) == {"id", "symbol", "name", "timestamp", "score", "text"}
    assert isinstance(hits[0]["timestamp"], datetime)


def test_search_semantic_respects_limit(store):
    for i in range(5):
        store.save(build_report(timestamp=datetime(2026, 9, 24, 9, i, 0)))
    assert len(store.search_semantic("基本面", n=3)) == 3
    assert len(store.search_semantic("基本面", n=99)) == 5  # 超过总数也不会报错


def test_search_semantic_filters_by_symbol(store):
    store.save(build_report(symbol="600519", timestamp=datetime(2026, 9, 24, 9, 0, 0)))
    store.save(build_report(symbol="AAPL", name="Apple", timestamp=datetime(2026, 9, 24, 10, 0, 0)))
    hits = store.search_semantic("基本面", n=5, symbol="AAPL")
    assert [h["symbol"] for h in hits] == ["AAPL"]
    assert store.search_semantic("基本面", n=5, symbol="NOPE") == []


def test_search_semantic_on_empty_collection_returns_empty(store):
    assert store.search_semantic("随便问问") == []


# ---------- 索引自愈 ----------


def test_partial_missing_vectors_are_backfilled_on_next_start(store, tmp_data_dir):
    """fail-soft 造成的 JSON 与索引不一致必须能自愈，而不是永久漏检。"""
    store.save(build_report(timestamp=datetime(2026, 9, 24, 9, 0, 0)))
    store.save(build_report(timestamp=datetime(2026, 9, 24, 10, 0, 0)))
    victim = "600519_20260924_090000"
    store._collection.delete(ids=[victim])
    assert store._collection.count() == 1

    reopened = ReportStore(
        data_file=str(tmp_data_dir / "reports.json"),
        chroma_dir=str(tmp_data_dir / "chroma"),
    )
    assert reopened._collection.count() == 2
    assert victim in set(reopened._collection.get(include=[])["ids"])


def test_startup_is_a_noop_when_index_is_consistent(store, tmp_data_dir, counting_embed):
    """模型一致且向量齐备时不该重建、也不该调用 embedding。"""
    store.save(build_report())
    counting_embed.clear()

    ReportStore(
        data_file=str(tmp_data_dir / "reports.json"),
        chroma_dir=str(tmp_data_dir / "chroma"),
    )
    assert counting_embed == []


def test_embedding_model_change_triggers_rebuild_with_warning(
    store, tmp_data_dir, monkeypatch, caplog, counting_embed
):
    """混用不同模型的向量会让相似度静默失去意义，必须清空重建并告警。"""
    store.save(build_report())
    counting_embed.clear()
    monkeypatch.setattr(
        cs,
        "COLLECTION_METADATA",
        {"hnsw:space": "cosine", "embedding_model": "另一个模型"},
    )

    with caplog.at_level(logging.WARNING):
        rebuilt = ReportStore(
            data_file=str(tmp_data_dir / "reports.json"),
            chroma_dir=str(tmp_data_dir / "chroma"),
        )

    assert counting_embed == [1]  # 重新向量化了一次
    assert rebuilt._collection.count() == 1  # 没有被清空后漏掉
    assert rebuilt._collection.metadata["embedding_model"] == "另一个模型"
    assert "不一致，已清空重建" in caplog.text


def test_unknown_provenance_with_existing_vectors_is_rebuilt(
    store, tmp_data_dir, monkeypatch, caplog
):
    """索引里有向量却查不到模型名时，无法保证一致性，只能重建。"""
    store.save(build_report())
    monkeypatch.setattr(
        cs, "COLLECTION_METADATA", {"hnsw:space": "cosine", "embedding_model": "未知"}
    )
    with caplog.at_level(logging.WARNING):
        ReportStore(
            data_file=str(tmp_data_dir / "reports.json"),
            chroma_dir=str(tmp_data_dir / "chroma"),
        )
    assert "（未记录）" in caplog.text or "未知" in caplog.text


def test_store_points_at_injected_paths_not_real_data(store, tmp_data_dir):
    """把「测试不碰真实数据」这件事本身也断言掉。"""
    store.save(build_report())
    assert (tmp_data_dir / "reports.json").exists()
    assert (tmp_data_dir / "chroma" / "chroma.sqlite3").exists()
    assert store._data_file == str(tmp_data_dir / "reports.json")
    assert store._collection.name == COLLECTION_NAME

    # 真实仓库的 data/ 里不该出现这次测试写入的记录。
    # data/ 被 gitignore，CI 上不存在，所以这里必须容忍缺失 ——
    # 否则会变成「本机通过、CI 报 FileNotFoundError」。
    real = Path("data/reports.json")
    if real.exists():
        records = json.loads(real.read_text(encoding="utf-8"))
        assert all(r["id"] != "600519_20260924_100000" for r in records)


# ---------- 空库接管、回填失败、单例 ----------


def test_legacy_empty_collection_is_adopted_without_warning(
    tmp_data_dir, stub_embed, counting_embed, caplog
):
    """真实场景：仓库里遗留一个没有模型标记的空 collection，应直接采用并补标记。"""
    import chromadb

    legacy = chromadb.PersistentClient(path=str(tmp_data_dir / "chroma"))
    legacy.create_collection(
        name=COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )

    with caplog.at_level(logging.WARNING):
        store = ReportStore(
            data_file=str(tmp_data_dir / "reports.json"),
            chroma_dir=str(tmp_data_dir / "chroma"),
        )

    assert store._collection.count() == 0
    assert (
        store._collection.metadata["embedding_model"] == cs.COLLECTION_METADATA["embedding_model"]
    )
    assert "不一致，已清空重建" not in caplog.text  # 空库重建没有意义，不该告警
    assert counting_embed == []  # 也没有需要回填的东西


def test_index_empty_list_is_a_noop(store):
    store._index([])
    assert store._collection.count() == 0


def test_startup_backfill_failure_is_logged_not_raised(
    tmp_data_dir, stub_embed, monkeypatch, caplog
):
    """回填失败不能让服务起不来，下次启动还会重试。"""
    data_file = tmp_data_dir / "reports.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    "id": "600519_20260101_090000",
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "timestamp": "2026-01-01T09:00:00",
                    "fundamental_summary": "旧记录",
                    "conclusion": "旧结论",
                    "full_report": {"risks": []},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def boom(texts):
        raise RuntimeError("embedding 服务不可用")

    monkeypatch.setattr(cs, "embed_texts", boom)
    with caplog.at_level(logging.WARNING):
        store = ReportStore(data_file=str(data_file), chroma_dir=str(tmp_data_dir / "chroma"))

    assert store._collection.count() == 0
    assert "下次启动会重试" in caplog.text


def test_get_store_returns_process_wide_singleton(tmp_data_dir, stub_embed, monkeypatch):
    """get_store 是进程内单例；这里把路径指向 tmp，避免碰到真实数据。"""
    monkeypatch.setattr(
        cs,
        "_settings",
        types.SimpleNamespace(chroma_persist_dir=str(tmp_data_dir / "chroma")),
    )
    monkeypatch.setattr(cs, "DATA_FILE", str(tmp_data_dir / "reports.json"))
    monkeypatch.setattr(cs, "_store", None)

    first = cs.get_store()
    assert first is cs.get_store()
    assert first._data_file == str(tmp_data_dir / "reports.json")
