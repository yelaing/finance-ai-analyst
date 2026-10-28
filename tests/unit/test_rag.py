"""报告问答（RAG）的单元测试：上下文拼装与「无命中不调用 LLM」的降级行为。"""

from datetime import datetime

import pytest

import backend.pipeline.rag as rag


def _hit(
    symbol: str = "600519",
    name: str = "贵州茅台",
    timestamp: datetime | str = datetime(2026, 9, 24, 10, 0, 0),
    text: str = "贵州茅台（600519） 基本面：营收稳健",
    score: float = 0.81,
) -> dict:
    return {
        "id": f"{symbol}_20260924_100000",
        "symbol": symbol,
        "name": name,
        "timestamp": timestamp,
        "score": score,
        "text": text,
    }


class RecordingChain:
    def __init__(self, answer: str = "根据报告 [1]，营收稳健。"):
        self.answer = answer
        self.payloads: list[dict] = []

    def invoke(self, payload, **kwargs):
        self.payloads.append(payload)
        return self.answer


class BoomChain:
    def invoke(self, payload, **kwargs):
        raise RuntimeError("模型超时")


def test_build_context_numbers_hits_and_includes_metadata():
    context = rag.build_context(
        [
            _hit(),
            _hit(
                symbol="002594",
                name="比亚迪",
                timestamp=datetime(2026, 9, 25, 9, 30, 0),
                text="比亚迪（002594） 基本面：新能源汽车销量增长",
            ),
        ]
    )

    assert "[1] 贵州茅台（600519） · 2026-09-24" in context
    assert "贵州茅台（600519） 基本面：营收稳健" in context
    assert "[2] 比亚迪（002594） · 2026-09-25" in context
    assert "新能源汽车销量增长" in context


def test_build_context_accepts_iso_timestamp_from_json_records():
    context = rag.build_context([_hit(timestamp="2026-09-24T10:00:00")])
    assert "[1] 贵州茅台（600519） · 2026-09-24" in context


def test_answer_question_passes_question_and_context(monkeypatch):
    chain = RecordingChain()
    monkeypatch.setattr(rag, "qa_chain", chain)

    answer = rag.answer_question("营收怎么样？", [_hit()])

    assert answer == "根据报告 [1]，营收稳健。"
    assert chain.payloads == [
        {
            "question": "营收怎么样？",
            "context": "[1] 贵州茅台（600519） · 2026-09-24\n贵州茅台（600519） 基本面：营收稳健",
        }
    ]


def test_answer_question_without_hits_skips_llm(monkeypatch):
    chain = RecordingChain()
    monkeypatch.setattr(rag, "qa_chain", chain)

    answer = rag.answer_question("随便问问", [])

    assert answer == rag.NO_HITS_ANSWER
    assert chain.payloads == []


def test_answer_question_propagates_llm_failure(monkeypatch):
    monkeypatch.setattr(rag, "qa_chain", BoomChain())

    with pytest.raises(RuntimeError, match="模型超时"):
        rag.answer_question("营收怎么样？", [_hit()])
