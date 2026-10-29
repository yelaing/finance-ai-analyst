"""采集脚本的 manifest 兼容逻辑，不联网。"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "eval_collect.py"


def load_collect():
    spec = importlib.util.spec_from_file_location("eval_collect", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_failures_accepts_list_from_manifest():
    collect = load_collect()
    payload = {
        "failures": [
            {"symbol": "601318", "category": "金融", "reason": "boom"},
        ]
    }

    failures = collect._load_failures(payload)

    assert failures == {"601318": {"symbol": "601318", "category": "金融", "reason": "boom"}}


def test_only_force_recollects_even_when_target_already_met():
    collect = load_collect()
    universe = [("600519", "白酒"), ("601318", "金融")]

    pending = collect._pending_symbols(
        universe,
        completed={"600519", "601318"},
        only=True,
        force=True,
    )

    assert pending == universe


def test_only_without_force_skips_completed_symbols():
    collect = load_collect()
    universe = [("600519", "白酒"), ("601318", "金融")]

    pending = collect._pending_symbols(
        universe,
        completed={"600519"},
        only=True,
        force=False,
    )

    assert pending == [("601318", "金融")]


def test_full_run_stops_when_target_is_met():
    collect = load_collect()
    universe = [("600519", "白酒"), ("601318", "金融")]

    pending = collect._pending_symbols(
        universe,
        completed={"600519", "601318"},
        only=False,
        force=False,
    )

    assert pending == []


def test_load_failures_accepts_dict_and_missing():
    collect = load_collect()

    assert collect._load_failures({"failures": {"600519": {"symbol": "600519"}}}) == {
        "600519": {"symbol": "600519"}
    }
    assert collect._load_failures({}) == {}
