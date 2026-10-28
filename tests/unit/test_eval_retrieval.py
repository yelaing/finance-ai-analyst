"""检索评测脚本的指标计算，用假向量验证，不调用 embedding。"""

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "eval_retrieval.py"


def load_eval():
    spec = importlib.util.spec_from_file_location("eval_retrieval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_ranks_relevant_document_and_computes_mrr():
    ev = load_eval()
    results = ev.evaluate(
        ["a", "b", "c"],
        np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]),
        [{"question": "q", "relevant": ["a"]}],
        np.array([[1.0, 0.0]]),
    )

    assert results[0]["ranked"] == ["a", "c", "b"]
    assert results[0]["hit@1"] is True
    assert results[0]["hit@3"] is True
    assert results[0]["rr"] == 1.0


def test_evaluate_handles_unanswerable_question_and_summary():
    ev = load_eval()
    results = ev.evaluate(
        ["a", "b"],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        [
            {"question": "known", "relevant": ["b"]},
            {"question": "unknown", "relevant": []},
        ],
        np.array([[0.0, 1.0], [0.9, 0.1]]),
    )

    assert results[0]["rr"] == 1.0
    assert results[1]["hit@1"] is None
    assert results[1]["rr"] is None

    summary = ev.summarize(results)
    assert summary["answerable"] == 1
    assert summary["unanswerable"] == 1
    assert summary["hit@1"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["unanswerable_top_scores"] == [results[1]["top_score"]]
