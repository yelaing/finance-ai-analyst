"""回归门禁：阈值判定、缺失指标、retrieval-only、新增 badcase。"""

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))

import eval_gate as eg  # noqa: E402

THRESHOLDS = {
    "retrieval": {
        "hit@1": 0.55,
        "hit@3": 0.80,
        "hit@5": 0.85,
        "mrr": 0.70,
        "recall@5": 0.50,
    },
    "generation": {
        "citation_accuracy": 0.70,
        "refusal_accuracy": 0.90,
        "relevance": 4.40,
    },
}


def _write_run(tmp_path, mode, retrieval_summary, generation_summary=None, badcases_before=0):
    run = tmp_path / "runs" / "20260101-000000"
    run.mkdir(parents=True)
    retrieval = {
        "dataset": {},
        "results": [],
        "summary": {"__overall__": retrieval_summary},
        "threshold": {},
    }
    (run / "retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
    if generation_summary is not None:
        generation = {"results": [], "summary": generation_summary}
        (run / "generation.json").write_text(json.dumps(generation), encoding="utf-8")
    (run / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "20260101-000000",
                "mode": mode,
                "badcases_before": badcases_before,
            }
        ),
        encoding="utf-8",
    )
    return run


def _thresholds(tmp_path):
    path = tmp_path / "eval_thresholds.json"
    path.write_text(json.dumps(THRESHOLDS), encoding="utf-8")
    return path


def _badcases(tmp_path, first_seen):
    path = tmp_path / "eval_badcases.json"
    path.write_text(
        json.dumps(
            {
                "C02:generation_failure": {
                    "id": "C02",
                    "failure_type": "generation_failure",
                    "first_seen": first_seen,
                    "last_seen": first_seen,
                    "status": "open",
                    "severity": "high",
                    "category": "comparison",
                    "reason": "引用不支撑",
                }
            }
        ),
        encoding="utf-8",
    )
    return path


FULL_SUMMARY = {"hit@1": 0.63, "hit@3": 0.87, "hit@5": 0.92, "mrr": 0.76, "recall@5": 0.67}
FULL_GENERATION = {"citation_accuracy": 0.80, "refusal_accuracy": 1.0, "relevance": 4.63}


def test_gate_passes_when_all_above(tmp_path, capsys):
    run = _write_run(tmp_path, "full", FULL_SUMMARY, FULL_GENERATION)
    badcases = _badcases(tmp_path, "older-run")

    code = eg.main(
        ["--run", str(run), "--thresholds", str(_thresholds(tmp_path)), "--badcases", str(badcases)]
    )

    assert code == 0
    assert "通过" in capsys.readouterr().out


def test_gate_fails_on_metric_drop(tmp_path, capsys):
    run = _write_run(tmp_path, "full", {**FULL_SUMMARY, "hit@1": 0.50}, FULL_GENERATION)
    badcases = _badcases(tmp_path, "older-run")

    code = eg.main(
        ["--run", str(run), "--thresholds", str(_thresholds(tmp_path)), "--badcases", str(badcases)]
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "hit@1" in out and "0.500" in out and "-0.050" in out


def test_gate_skips_generation_for_retrieval_only_run(tmp_path):
    run = _write_run(tmp_path, "retrieval", FULL_SUMMARY)
    badcases = _badcases(tmp_path, "older-run")

    code = eg.main(
        ["--run", str(run), "--thresholds", str(_thresholds(tmp_path)), "--badcases", str(badcases)]
    )

    assert code == 0


def test_gate_fails_when_full_run_misses_generation_file(tmp_path, capsys):
    run = _write_run(tmp_path, "full", FULL_SUMMARY)
    badcases = _badcases(tmp_path, "older-run")

    code = eg.main(
        ["--run", str(run), "--thresholds", str(_thresholds(tmp_path)), "--badcases", str(badcases)]
    )

    assert code == 1
    assert "generation.json" in capsys.readouterr().out


def test_gate_fails_on_new_badcase(tmp_path, capsys):
    run = _write_run(tmp_path, "full", FULL_SUMMARY, FULL_GENERATION, badcases_before=3)
    badcases = _badcases(tmp_path, "20260101-000000")

    code = eg.main(
        ["--run", str(run), "--thresholds", str(_thresholds(tmp_path)), "--badcases", str(badcases)]
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "C02:generation_failure" in out


def test_gate_does_not_fail_on_baseline_run(tmp_path, capsys):
    run = _write_run(tmp_path, "full", FULL_SUMMARY, FULL_GENERATION, badcases_before=0)
    badcases = _badcases(tmp_path, "20260101-000000")

    code = eg.main(
        ["--run", str(run), "--thresholds", str(_thresholds(tmp_path)), "--badcases", str(badcases)]
    )

    assert code == 0
    assert "基线" in capsys.readouterr().out
