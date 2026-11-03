"""版本对比：指标 diff 与失败 case 对齐。全部离线，不调用任何外部接口。"""

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))

import eval_compare as ec  # noqa: E402


def _result(case_id, category, expected, ranked, answerable=True):
    return {
        "id": case_id,
        "category": category,
        "answerable": answerable,
        "expected": expected,
        "ranked": ranked,
        "scores": [0.9] * len(ranked),
        "top1": ranked[0] if ranked else None,
        "top1_score": 0.9,
        "first_hit": 1,
        "question": f"q-{case_id}",
    }


def _retrieval(summary, results):
    return {
        "dataset": {"report_names": {}},
        "results": results,
        "summary": {"__overall__": {"count": len(results), **summary}},
        "threshold": {},
    }


def _generation(summary, failures):
    return {
        "results": [
            {
                "id": case_id,
                "category": category,
                "failure": {"primary": primary, "reasons": [primary]},
            }
            for case_id, category, primary in failures
        ],
        "summary": summary,
    }


def _write_run(root, name, retrieval, generation=None):
    run = root / name
    run.mkdir()
    (run / "retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
    if generation is not None:
        (run / "generation.json").write_text(json.dumps(generation), encoding="utf-8")
    (run / "meta.json").write_text(
        json.dumps({"mode": "full" if generation else "retrieval"}), encoding="utf-8"
    )
    return run


def test_compare_metrics_and_failure_alignment(tmp_path):
    prev = _write_run(
        tmp_path,
        "20260101-000000",
        _retrieval(
            {"hit@1": 0.5, "hit@3": 0.8, "hit@5": 0.9, "mrr": 0.6, "recall@5": 0.5},
            [
                _result("Q1", "factual", ["R1"], ["R2"]),
                _result("Q2", "opinion", ["R9"], ["R8", "R9"]),
            ],
        ),
        _generation(
            {"citation_accuracy": 0.8, "refusal_accuracy": 1.0, "relevance": 4.6},
            [("G1", "comparison", "模型编造"), ("G2", "opinion", "引用不支撑")],
        ),
    )
    curr = _write_run(
        tmp_path,
        "20260102-000000",
        _retrieval(
            {"hit@1": 0.6, "hit@3": 0.85, "hit@5": 0.95, "mrr": 0.7, "recall@5": 0.55},
            [
                _result("Q2", "opinion", ["R9"], ["R8", "R9"]),
                _result("Q3", "factual", ["R3"], ["R4"]),
            ],
        ),
        _generation(
            {"citation_accuracy": 0.7, "refusal_accuracy": 0.9, "relevance": 4.4},
            [("G1", "comparison", "模型编造"), ("G3", "factual", "模型编造")],
        ),
    )

    text = ec.compare_runs(prev, curr)

    assert "hit@1" in text
    assert "+0.10" in text and "↑" in text
    assert "新增失败" in text and "Q3" in text and "G3" in text
    assert "已修复" in text and "Q1" in text and "G2" in text
    assert "仍失败" in text and "Q2" in text and "G1" in text


def test_compare_without_generation_side(tmp_path):
    prev = _write_run(
        tmp_path,
        "20260101-000000",
        _retrieval({"hit@1": 0.5, "hit@3": 0.8, "hit@5": 0.9, "mrr": 0.6, "recall@5": 0.5}, []),
    )
    curr = _write_run(
        tmp_path,
        "20260102-000000",
        _retrieval({"hit@1": 0.6, "hit@3": 0.85, "hit@5": 0.95, "mrr": 0.7, "recall@5": 0.55}, []),
    )

    text = ec.compare_runs(prev, curr)

    assert "生成侧未参与对比" in text
    assert "hit@1" in text
