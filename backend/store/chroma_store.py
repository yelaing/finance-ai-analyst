import json
import logging
import os
import threading
from datetime import datetime

import chromadb

from backend.config import get_settings
from backend.schemas.models import AnalysisReport, HistoryItem
from backend.store.embeddings import embed_texts

logger = logging.getLogger(__name__)

_settings = get_settings()

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reports.json")
COLLECTION_NAME = "analysis_reports"
# embedding_model 记在 collection 上，用来防止不同模型的向量混进同一个索引
COLLECTION_METADATA = {"hnsw:space": "cosine", "embedding_model": _settings.embedding_model}
_lock = threading.Lock()


def _title(record: dict) -> str:
    name = record.get("name")
    if name and name != record["symbol"]:
        return f"{name}（{record['symbol']}）"
    return record["symbol"]


def _build_doc(record: dict) -> str:
    """把一份报告压成用于向量化的关键文本。老记录缺字段就跳过对应行。"""
    full = record.get("full_report") or {}
    lines = [f"{_title(record)} 基本面：{record.get('fundamental_summary') or ''}"]

    if record.get("conclusion"):
        lines.append(f"综合结论：{record['conclusion']}")

    risks = full.get("risks") or []
    if risks:
        lines.append(
            "风险点："
            + "；".join(f"{r.get('category')}({r.get('level')}): {r.get('detail')}" for r in risks)
        )

    sentiment = full.get("sentiment") or {}
    if sentiment:
        lines.append(
            f"舆情：{sentiment.get('overall')}，得分 {sentiment.get('score')}，"
            f"驱动因素：{'、'.join(sentiment.get('key_drivers') or [])}"
        )

    if full.get("debate_verdict"):
        lines.append(f"多空分歧：{full['debate_verdict']}")

    return "\n".join(lines)


def _meta(record: dict) -> dict:
    return {
        "symbol": record["symbol"],
        "name": record.get("name") or record["symbol"],
        "timestamp": record["timestamp"],
    }


class ReportStore:
    def __init__(self):
        self._records: list[dict] = self._load()
        self._client = chromadb.PersistentClient(path=_settings.chroma_persist_dir)
        self._collection = self._open_collection()
        self._sync_index()

    # ---------- JSON 文件：权威源，精确检索 / 完整报告 / 导出都读这里 ----------

    def _load(self) -> list[dict]:
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _dump(self):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2, default=str)

    # ---------- ChromaDB：语义索引 ----------

    def _open_collection(self):
        existing = self._client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=None, metadata=COLLECTION_METADATA
        )
        stored = (existing.metadata or {}).get("embedding_model")
        current = COLLECTION_METADATA["embedding_model"]
        if stored == current:
            return existing
        # 同一个 collection 里混入不同模型的向量，相似度会静默失去意义，只能清空重建
        if existing.count():
            logger.warning(
                "向量索引的 embedding 模型为 %s，与当前配置 %s 不一致，已清空重建",
                stored or "（未记录）",
                current,
            )
        self._client.delete_collection(COLLECTION_NAME)
        return self._client.create_collection(
            name=COLLECTION_NAME, embedding_function=None, metadata=COLLECTION_METADATA
        )

    def _sync_index(self):
        """把 JSON 里存在、索引里缺失的记录补齐：首次回填与写入失败后的自愈都走这里。"""
        indexed = set(self._collection.get(include=[])["ids"])
        missing = [r for r in self._records if r["id"] not in indexed]
        if not missing:
            return
        logger.info("向量索引补齐 %d/%d 条历史报告", len(missing), len(self._records))
        try:
            self._index(missing)
        except Exception:
            logger.warning("历史报告写入向量索引失败，下次启动会重试", exc_info=True)

    def _index(self, records: list[dict]):
        if not records:
            return
        docs = [_build_doc(r) for r in records]
        self._collection.upsert(
            ids=[r["id"] for r in records],
            documents=docs,
            embeddings=embed_texts(docs),
            metadatas=[_meta(r) for r in records],
        )

    # ---------- 对外接口 ----------

    def save(self, report: AnalysisReport) -> str:
        doc_id = f"{report.symbol}_{report.timestamp.strftime('%Y%m%d_%H%M%S')}"
        record = {
            "id": doc_id,
            "symbol": report.symbol,
            "name": report.name,
            "timestamp": report.timestamp.isoformat(),
            "sentiment_score": report.sentiment.score if report.sentiment else None,
            "sources": report.sources,
            "fundamental_summary": report.fundamental_summary,
            "conclusion": report.conclusion,
            "full_report": report.model_dump(mode="json"),
        }
        with _lock:
            self._records.append(record)
            self._dump()
        try:
            self._index([record])
        except Exception:
            logger.warning(
                "报告 %s 已存入 JSON，但写入向量索引失败，下次启动会补齐", doc_id, exc_info=True
            )
        return doc_id

    def search_semantic(self, query: str, n: int = 5, symbol: str | None = None) -> list[dict]:
        """语义检索历史报告。symbol 非空时限定在该股票范围内检索。"""
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=embed_texts([query]),
            n_results=min(n, total),
            where={"symbol": symbol} if symbol else None,
            include=["metadatas", "documents", "distances"],
        )
        return [
            {
                "id": result["ids"][0][i],
                "symbol": result["metadatas"][0][i]["symbol"],
                "name": result["metadatas"][0][i]["name"],
                "timestamp": datetime.fromisoformat(result["metadatas"][0][i]["timestamp"]),
                "score": round(1 - result["distances"][0][i], 4),
                "text": result["documents"][0][i],
            }
            for i in range(len(result["ids"][0]))
        ]

    def get_latest(self, symbol: str) -> dict | None:
        """Return the most recent full report for a symbol, or None."""
        records = [r for r in self._records if r["symbol"] == symbol]
        if not records:
            return None
        records.sort(key=lambda x: x["timestamp"], reverse=True)
        return records[0].get("full_report")

    def get_history(self, symbol: str | None = None, limit: int = 20) -> list[HistoryItem]:
        records = self._records
        if symbol:
            records = [r for r in records if r["symbol"] == symbol]
        records.sort(key=lambda x: x["timestamp"], reverse=True)
        return [
            HistoryItem(
                id=r["id"],
                symbol=r["symbol"],
                name=r["name"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                summary="; ".join(r.get("sources", [])),
            )
            for r in records[:limit]
        ]


_store: ReportStore | None = None


def get_store() -> ReportStore:
    global _store
    if _store is None:
        _store = ReportStore()
    return _store
