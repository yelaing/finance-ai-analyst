import json
import logging
import os
import threading
from datetime import datetime
from typing import Optional

from backend.schemas.models import AnalysisReport, HistoryItem

logger = logging.getLogger(__name__)
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reports.json")
_lock = threading.Lock()


class ReportStore:

    def __init__(self):
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _dump(self):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2, default=str)

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
        return doc_id

    def search_similar(self, symbol: str, query: str, n: int = 3) -> list[dict]:
        symbol_records = [r for r in self._records if r["symbol"] == symbol]
        symbol_records.sort(key=lambda x: x["timestamp"], reverse=True)
        return [
            {"id": r["id"], "text": r.get("fundamental_summary", ""), "meta": r}
            for r in symbol_records[:n]
        ]

    def get_latest(self, symbol: str) -> Optional[dict]:
        """Return the most recent full report for a symbol, or None."""
        records = [r for r in self._records if r["symbol"] == symbol]
        if not records:
            return None
        records.sort(key=lambda x: x["timestamp"], reverse=True)
        return records[0].get("full_report")

    def get_history(self, symbol: Optional[str] = None, limit: int = 20) -> list[HistoryItem]:
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


_store: Optional[ReportStore] = None


def get_store() -> ReportStore:
    global _store
    if _store is None:
        _store = ReportStore()
    return _store
