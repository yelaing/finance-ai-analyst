"""运行归档、badcase 合并、历史表与失败 case 聚合。全部离线。"""

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))

import eval_suite as es  # noqa: E402


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
        "first_hit": 2 if expected and expected[0] in ranked else None,
        "question": f"q-{case_id}",
    }


def _retrieval(results):
    return {
        "dataset": {"report_names": {"R9": "示例公司"}, "queries": [], "report_ids": []},
        "results": results,
        "summary": {
            "__overall__": {
                "count": 1,
                "hit@1": 0.63,
                "hit@3": 0.87,
                "hit@5": 0.92,
                "mrr": 0.76,
                "recall@5": 0.67,
            }
        },
        "threshold": {},
    }


def _generation(failures):
    return {
        "results": [
            {
                "id": case_id,
                "category": category,
                "failure": {"primary": primary, "reasons": [primary]},
                "question": f"q-{case_id}",
                "reference_answer": "ref",
                "answer": "ans",
                "citations": [],
                "judge": {"relevance": 4, "refused": False, "fabricated": False, "notes": ""},
                "citation_metrics": {
                    "indices": [],
                    "supported_indices": [],
                    "invalid_indices": [],
                    "accuracy": None,
                },
            }
            for case_id, category, primary in failures
        ],
        "summary": {
            "count": 30,
            "citation_accuracy": 0.8,
            "refusal_accuracy": 1.0,
            "relevance": 4.63,
        },
    }


def test_failure_rows_severity_and_types():
    retrieval = _retrieval(
        [
            _result("Q1", "factual", ["R1"], ["R2"]),
            _result("Q2", "opinion", ["R9"], ["R8", "R9"]),
        ]
    )
    generation = _generation([("Q2", "opinion", "模型编造")])

    rows = es.failure_rows(retrieval, generation)
    by_key = {(row["id"], row["failure_type"]): row for row in rows}

    assert by_key[("Q1", "retrieval_top5_miss")]["severity"] == "high"
    assert by_key[("Q2", "retrieval_top1_miss")]["severity"] == "low"
    assert by_key[("Q2", "generation_failure")]["severity"] == "high"


def test_merge_badcases_preserves_status_and_first_seen():
    existing = {
        "C02:generation_failure": {
            "id": "C02",
            "category": "comparison",
            "failure_type": "generation_failure",
            "reason": "旧原因",
            "severity": "high",
            "first_seen": "old-run",
            "last_seen": "old-run",
            "status": "fixed",
        }
    }
    rows = [
        {
            "id": "C02",
            "category": "comparison",
            "failure_type": "generation_failure",
            "reason": "新原因",
            "severity": "high",
        },
        {
            "id": "C05",
            "category": "opinion",
            "failure_type": "retrieval_top1_miss",
            "reason": "top1 未命中",
            "severity": "low",
        },
    ]

    merged = es.merge_badcases(existing, rows, "new-run")

    assert merged["C02:generation_failure"]["status"] == "fixed"
    assert merged["C02:generation_failure"]["first_seen"] == "old-run"
    assert merged["C02:generation_failure"]["last_seen"] == "new-run"
    assert merged["C02:generation_failure"]["reason"] == "新原因"
    assert merged["C05:retrieval_top1_miss"]["status"] == "open"
    assert merged["C05:retrieval_top1_miss"]["first_seen"] == "new-run"


def test_append_history_creates_header_once(tmp_path):
    path = tmp_path / "eval_history.md"
    meta = {"run_id": "20260101-000000", "commit": "a1b2c3d", "mode": "full"}
    retrieval = _retrieval([])
    generation = _generation([])

    es.append_history(path, meta, retrieval, generation)
    es.append_history(path, {**meta, "run_id": "20260102-000000"}, retrieval, generation)

    text = path.read_text(encoding="utf-8")
    assert text.count("| 时间 | commit |") == 1
    assert text.count("| 2026-") == 2
    assert "100%" in text and "4.63" in text


def test_append_history_marks_retrieval_only(tmp_path):
    path = tmp_path / "eval_history.md"
    meta = {"run_id": "20260101-000000", "commit": "a1b2c3d", "mode": "retrieval"}

    es.append_history(path, meta, _retrieval([]), None)

    assert "| retrieval |" in path.read_text(encoding="utf-8")


def test_write_run_dir_layout(tmp_path):
    meta = {"run_id": "20260101-000000", "commit": "a1b2c3d", "mode": "full"}

    run_dir = es.write_run_dir(
        tmp_path,
        meta,
        retrieval=_retrieval([]),
        generation=_generation([]),
        report="# report",
        failures="# failures",
    )

    assert (run_dir / "report.md").read_text(encoding="utf-8") == "# report"
    assert (run_dir / "failures.md").read_text(encoding="utf-8") == "# failures"
    assert json.loads((run_dir / "retrieval.json").read_text(encoding="utf-8"))["summary"]
    assert json.loads((run_dir / "generation.json").read_text(encoding="utf-8"))["summary"]
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["run_id"]


def test_failures_markdown_groups_same_id_once():
    retrieval = _retrieval(
        [
            _result("Q2", "opinion", ["R9"], ["R8", "R9"]),
            _result("Q5", "factual", ["R5"], ["R6"]),
        ]
    )
    generation = _generation([("Q2", "opinion", "模型编造")])

    text = es.render_failures_unified(retrieval, generation)

    assert text.count("## Q2") == 1
    assert "retrieval_top1_miss" in text
    assert "generation_failure" in text
    assert "严重度" in text
