"""真实数据评测的一键入口。

默认不跑（`pyproject.toml` 里 `-m 'not eval'`）；需要时：

    py -3 -m pytest -m eval
    py -3 tests/eval_suite.py
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "eval_suite.py"
REPORT = Path(__file__).resolve().parent / "eval_report.md"
FAILURES = Path(__file__).resolve().parent / "eval_failures.md"

pytestmark = pytest.mark.eval


def test_real_eval_suite_writes_markdown_reports():
    spec = importlib.util.spec_from_file_location("eval_suite", SCRIPT)
    assert spec and spec.loader
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    result = suite.run_suite()

    assert len(result["retrieval"]["results"]) == 80
    assert result["generation"]["summary"]["count"] == 30
    assert REPORT.exists()
    assert FAILURES.exists()
    assert "检索评测" in REPORT.read_text(encoding="utf-8")
    assert "失败 case" in FAILURES.read_text(encoding="utf-8")
