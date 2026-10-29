"""检索评测的指标计算，用假分数验证，不调用 embedding。"""

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


def _query(identifier, category="factual", expected=("a",), answerable=True, subtype=None):
    return {
        "id": identifier,
        "category": category,
        "answerable": answerable,
        "unanswerable_type": subtype,
        "question": identifier,
        "expected_hits": list(expected),
    }


def load_suite():
    import sys

    tests_dir = SCRIPT.parent
    sys.path.insert(0, str(tests_dir))
    suite_script = tests_dir / "eval_suite.py"
    spec = importlib.util.spec_from_file_location("eval_suite", suite_script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _retrieval_result(identifier, expected, ranked):
    return {
        "id": identifier,
        "category": "opinion",
        "answerable": True,
        "question": identifier,
        "expected": list(expected),
        "ranked": list(ranked),
        "scores": [0.9, 0.8, 0.7, 0.6, 0.5],
    }


def test_retrieval_failure_library_ignores_partial_multi_answer_hits():
    suite = load_suite()
    retrieval = {
        "results": [
            _retrieval_result("O01", ["a", "b", "c", "d", "e", "f"], ["a", "x", "y", "z", "w"])
        ],
        "dataset": {"report_names": {}},
    }

    text = suite.render_retrieval_failures(retrieval)

    assert "O01" not in text


def test_retrieval_failure_library_lists_top1_misses_with_top5_hits():
    suite = load_suite()
    retrieval = {
        "results": [
            _retrieval_result("C01", ["a", "b"], ["x", "a", "y", "z", "w"]),
        ],
        "dataset": {"report_names": {}},
    }

    text = suite.render_retrieval_failures(retrieval)

    assert "top1 未命中" in text
    assert "C01" in text


def test_retrieval_failure_library_lists_zero_hit_questions():
    suite = load_suite()
    retrieval = {
        "results": [_retrieval_result("O02", ["a", "b"], ["x", "y", "z", "w", "v"])],
        "dataset": {"report_names": {}},
    }

    text = suite.render_retrieval_failures(retrieval)

    assert "O02" in text


def test_cosine_scores_rank_by_similarity():
    ev = load_eval()
    scores = ev.cosine_scores(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]),
    )

    assert scores[0] > scores[2] > scores[1]


def test_evaluate_query_computes_hit_at_k_and_mrr():
    ev = load_eval()
    result = ev.evaluate_query(
        _query("q1"),
        ["a", "c", "b"],
        [0.9, 0.8, 0.7],
    )

    assert result["hit@1"] is True
    assert result["hit@3"] is True
    assert result["hit@5"] is True
    assert result["rr"] == 1.0
    assert result["recall@5"] == 1.0
    assert result["top1"] == "a"
    assert result["top1_score"] == 0.9


def test_evaluate_query_handles_multiple_answers_and_misses():
    ev = load_eval()
    result = ev.evaluate_query(
        _query("q2", expected=("a", "c")),
        ["b", "a", "c"],
        [0.9, 0.8, 0.7],
    )

    assert result["hit@1"] is False
    assert result["hit@3"] is True
    assert result["rr"] == 0.5
    assert result["recall@5"] == 1.0


def test_evaluate_query_does_not_score_unanswerable_questions():
    ev = load_eval()
    result = ev.evaluate_query(
        _query("q3", expected=(), answerable=False, subtype="out_of_scope"),
        ["a", "b"],
        [0.6, 0.5],
    )

    assert result["hit@1"] is None
    assert result["rr"] is None
    assert result["recall@5"] is None
    assert result["top1_score"] == 0.6


def test_aggregate_splits_categories_and_unanswerable_subtypes():
    ev = load_eval()
    results = [
        ev.evaluate_query(_query("f1", "factual"), ["a", "b"], [0.9, 0.8]),
        ev.evaluate_query(_query("f2", "factual"), ["b", "a"], [0.9, 0.8]),
        ev.evaluate_query(_query("o1", "opinion", expected=("a", "b")), ["b", "a"], [0.9, 0.8]),
        ev.evaluate_query(
            _query("u1", "unanswerable", expected=(), answerable=False, subtype="out_of_scope"),
            ["a"],
            [0.4],
        ),
        ev.evaluate_query(
            _query("u2", "unanswerable", expected=("a",), answerable=False, subtype="missing_fact"),
            ["a"],
            [0.7],
        ),
    ]

    summary = ev.aggregate(results)

    assert summary["factual"]["hit@1"] == 0.5
    assert summary["factual"]["count"] == 2
    assert summary["opinion"]["hit@1"] == 1.0
    assert summary["__overall__"]["count"] == 3
    assert summary["unanswerable"]["by_type"]["out_of_scope"]["count"] == 1
    assert summary["unanswerable"]["by_type"]["missing_fact"]["top1_scores"]["median"] == 0.7


def test_threshold_analysis_uses_p90_and_reports_tradeoff():
    ev = load_eval()
    analysis = ev.threshold_analysis(
        [0.6, 0.7, 0.8, 0.9],
        [0.4, 0.5, 0.6, 0.7],
        [0.65, 0.85],
    )

    assert analysis["threshold"] == 0.67
    assert analysis["false_accept"] == 0.25
    assert analysis["false_reject"] == 0.25
    assert analysis["retention"] == 0.75
    assert analysis["missing_fact_above"] == 0.5
    assert any(row["source"] == "无关问题 top1 P90" for row in analysis["candidates"])


def test_render_markdown_contains_category_and_threshold_sections():
    ev = load_eval()
    results = [
        ev.evaluate_query(_query("f1", "factual"), ["a"], [0.9]),
        ev.evaluate_query(
            _query("u1", "unanswerable", expected=(), answerable=False, subtype="out_of_scope"),
            ["a"],
            [0.4],
        ),
        ev.evaluate_query(
            _query("u2", "unanswerable", expected=("a",), answerable=False, subtype="missing_fact"),
            ["a"],
            [0.8],
        ),
    ]
    markdown = ev.render_retrieval_markdown(
        {
            "dataset": {"report_ids": ["a"], "queries": [{}] * 3},
            "results": results,
            "summary": ev.aggregate(results),
            "threshold": ev.threshold_analysis([0.9], [0.4], [0.8]),
        }
    )

    assert "事实型" in markdown
    assert "无法回答型" in markdown
    assert "相似度阈值建议" in markdown
    assert "报告未包含该事实" in markdown
