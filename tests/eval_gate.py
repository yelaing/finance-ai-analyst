"""回归门禁：读一次运行的结果，低于阈值或出现新增 badcase 就非零退出。

    python tests/eval_gate.py
    python tests/eval_gate.py --run tests/eval_runs/20261103-153000

阈值写在 tests/eval_thresholds.json，初始值由多次运行的均值与波动推导。
"""

import argparse
import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from eval_compare import extract_metrics, load_run  # noqa: E402

RUNS_DIR = TESTS_DIR / "eval_runs"
THRESHOLDS_PATH = TESTS_DIR / "eval_thresholds.json"
BADCASES_PATH = TESTS_DIR / "eval_badcases.json"

_METRIC_LABELS = {
    "hit@1": "hit@1",
    "hit@3": "hit@3",
    "hit@5": "hit@5",
    "mrr": "MRR",
    "recall@5": "recall@5",
    "citation_accuracy": "引用准确率",
    "refusal_accuracy": "拒答准确率",
    "relevance": "相关性",
}


def find_latest_run(runs_dir: Path = RUNS_DIR) -> Path | None:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return None
    dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    return dirs[-1] if dirs else None


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_badcases(run_id: str, badcases_path: Path = BADCASES_PATH) -> list[dict]:
    path = Path(badcases_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [entry for entry in data.values() if entry.get("first_seen") == run_id]


def evaluate(run_dir: Path, thresholds: dict, badcases_path: Path = BADCASES_PATH) -> dict:
    run = load_run(run_dir)
    meta = run.get("meta") or {}
    mode = meta.get("mode", "full")
    run_id = meta.get("run_id") or Path(run_dir).name

    checked: list[dict] = []
    breaches: list[dict] = []
    notes: list[str] = []

    if run.get("retrieval") is None:
        breaches.append({"name": "retrieval.json", "missing": True})
    else:
        metrics = extract_metrics(run)
        for name, floor in (thresholds.get("retrieval") or {}).items():
            actual = metrics.get(name)
            item = {"name": name, "actual": actual, "threshold": floor, "missing": actual is None}
            checked.append(item)
            if actual is None or actual < floor:
                breaches.append(item)

        if thresholds.get("generation"):
            if mode != "full":
                notes.append("retrieval-only 运行，跳过生成指标")
            elif run.get("generation") is None:
                breaches.append({"name": "generation.json", "missing": True})
            else:
                for name, floor in thresholds["generation"].items():
                    actual = metrics.get(name)
                    item = {
                        "name": name,
                        "actual": actual,
                        "threshold": floor,
                        "missing": actual is None,
                    }
                    checked.append(item)
                    if actual is None or actual < floor:
                        breaches.append(item)

    new_cases = new_badcases(run_id, badcases_path)
    badcases_before = meta.get("badcases_before", 0)
    if new_cases and badcases_before == 0:
        notes.append(
            f"基线运行（badcases_before=0），记录 {len(new_cases)} 条新 badcase，不做门禁判定"
        )
    elif new_cases:
        breaches.extend(
            {"name": f"新增 badcase {entry.get('id')}", "badcase": entry} for entry in new_cases
        )

    return {
        "run_id": run_id,
        "mode": mode,
        "checked": checked,
        "breaches": breaches,
        "new_badcases": new_cases,
        "notes": notes,
    }


def _print_report(result: dict) -> None:
    print(f"评测门禁：{result['run_id']}（mode={result['mode']}）")
    for item in result["checked"]:
        label = _METRIC_LABELS.get(item["name"], item["name"])
        if item["actual"] is None:
            print(f"✗ {label} 缺失（阈值 {item['threshold']}）")
        elif item["actual"] < item["threshold"]:
            print(
                f"✗ {label} = {item['actual']:.3f} < {item['threshold']:.3f}"
                f"（{item['actual'] - item['threshold']:+.3f}）"
            )
        else:
            print(f"✓ {label} = {item['actual']:.3f} >= {item['threshold']:.3f}")

    for breach in result["breaches"]:
        if breach.get("missing") and breach["name"] in ("retrieval.json", "generation.json"):
            print(f"✗ {breach['name']} 缺失，无法校验对应指标")
        elif "badcase" in breach:
            entry = breach["badcase"]
            print(
                f"✗ 新增 badcase：{entry.get('id')}:{entry.get('failure_type')}"
                f"（{entry.get('severity')}，{entry.get('reason')}）"
            )

    for note in result["notes"]:
        print(f"! {note}")

    failures = [breach for breach in result["breaches"] if "badcase" not in breach]
    failures += [breach for breach in result["breaches"] if "badcase" in breach]
    if failures:
        print(f"门禁失败：{len(failures)} 项")
    else:
        print("门禁通过")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测回归门禁")
    parser.add_argument("--run", type=Path, default=None, help="run 目录，默认取最新一次")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    parser.add_argument("--badcases", type=Path, default=BADCASES_PATH)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    run_dir = args.run or find_latest_run(args.runs_dir)
    if run_dir is None:
        print(f"找不到运行记录：{args.runs_dir}")
        return 2

    thresholds = load_thresholds(args.thresholds)
    result = evaluate(run_dir, thresholds, args.badcases)
    _print_report(result)
    return 1 if result["breaches"] else 0


if __name__ == "__main__":
    sys.exit(main())
