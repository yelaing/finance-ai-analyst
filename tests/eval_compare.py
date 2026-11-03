"""评测运行对比：指标 diff 与失败 case 对齐。

输入是两个 `tests/eval_runs/<timestamp>/` 目录（由 eval_suite.py 产出），
只读 json，不调用任何外部接口。
"""

import argparse
import json
import sys
from pathlib import Path

RETRIEVAL_METRICS = ("hit@1", "hit@3", "hit@5", "mrr", "recall@5")
GENERATION_METRICS = ("citation_accuracy", "refusal_accuracy", "relevance")

# 失败类型 -> 严重度。自动推导，不让人工填：
# top5 全漏说明正确报告根本没进上下文；生成失败说明链路已经出错。
# top1 未命中但 top3/top5 仍命中，链路未必受影响，标 low。
SEVERITY = {
    "retrieval_top5_miss": "high",
    "generation_failure": "high",
    "retrieval_top1_miss": "low",
}
_SEVERITY_RANK = {"high": 0, "low": 1}


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)

    def read(name: str):
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    return {
        "retrieval": read("retrieval.json"),
        "generation": read("generation.json"),
        "meta": read("meta.json") or {},
    }


def _top1(result: dict):
    ranked = result.get("ranked") or []
    return result.get("top1") or (ranked[0] if ranked else None)


def failure_entries(retrieval: dict | None, generation: dict | None) -> list[dict]:
    """把检索与生成两侧的失败统一成 (id, failure_type) 粒度的条目。"""
    rows: list[dict] = []

    for result in (retrieval or {}).get("results", []):
        if not result.get("answerable") or not result.get("expected"):
            continue
        expected = set(result["expected"])
        top5 = set((result.get("ranked") or [])[:5])
        if not top5 & expected:
            failure_type = "retrieval_top5_miss"
            reason = "top5 全漏"
        elif _top1(result) not in expected:
            failure_type = "retrieval_top1_miss"
            reason = "top1 未命中"
        else:
            continue
        rows.append(
            {
                "id": result["id"],
                "category": result.get("category"),
                "failure_type": failure_type,
                "reason": reason,
                "severity": SEVERITY[failure_type],
            }
        )

    for result in (generation or {}).get("results", []):
        failure = result.get("failure")
        if not failure:
            continue
        rows.append(
            {
                "id": result["id"],
                "category": result.get("category"),
                "failure_type": "generation_failure",
                "reason": failure.get("primary", "生成失败"),
                "severity": SEVERITY["generation_failure"],
            }
        )

    return sorted(
        rows, key=lambda row: (_SEVERITY_RANK[row["severity"]], row["id"], row["failure_type"])
    )


def extract_metrics(run: dict) -> dict[str, float | None]:
    retrieval = run.get("retrieval") or {}
    overall = (retrieval.get("summary") or {}).get("__overall__") or {}
    generation = run.get("generation") or {}
    summary = generation.get("summary") or {}
    metrics: dict[str, float | None] = {name: overall.get(name) for name in RETRIEVAL_METRICS}
    metrics.update({name: summary.get(name) for name in GENERATION_METRICS})
    return metrics


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "—"
    change = after - before
    if abs(change) < 0.005:
        return "0.00 →"
    arrow = "↑" if change > 0 else "↓"
    return f"{change:+.2f} {arrow}"


def _failure_types_by_id(rows: list[dict]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row["failure_type"])
    return {case_id: "、".join(sorted(types)) for case_id, types in grouped.items()}


def compare_runs(prev_dir: Path, curr_dir: Path) -> str:
    prev = load_run(prev_dir)
    curr = load_run(curr_dir)
    prev_metrics = extract_metrics(prev)
    curr_metrics = extract_metrics(curr)

    lines = [f"# 评测对比：{Path(prev_dir).name} → {Path(curr_dir).name}", ""]

    prev_count = len((prev.get("retrieval") or {}).get("results", []))
    curr_count = len((curr.get("retrieval") or {}).get("results", []))
    if prev_count != curr_count:
        lines.append(f"> 数据集题数不同（{prev_count} → {curr_count}），指标不可直接比较。")
        lines.append("")

    lines.extend(["## 指标", "", "| 指标 | 旧 | 新 | Δ |", "|---|---:|---:|---:|"])
    for name in RETRIEVAL_METRICS + GENERATION_METRICS:
        lines.append(
            f"| {name} | {_fmt(prev_metrics.get(name))} | "
            f"{_fmt(curr_metrics.get(name))} | "
            f"{_delta(prev_metrics.get(name), curr_metrics.get(name))} |"
        )
    lines.append("")

    has_both_generation = prev.get("generation") is not None and curr.get("generation") is not None
    if not has_both_generation:
        lines.extend(["生成侧未参与对比（至少一侧缺 generation.json）。", ""])

    prev_rows = failure_entries(prev.get("retrieval"), prev.get("generation"))
    curr_rows = failure_entries(curr.get("retrieval"), curr.get("generation"))
    prev_ids = {row["id"] for row in prev_rows}
    curr_ids = {row["id"] for row in curr_rows}
    curr_types = _failure_types_by_id(curr_rows)

    new_ids = sorted(curr_ids - prev_ids)
    fixed_ids = sorted(prev_ids - curr_ids)
    still_ids = sorted(prev_ids & curr_ids)

    def render(ids: list[str], with_type: bool = False) -> str:
        if not ids:
            return "无"
        if with_type:
            return "、".join(f"{case_id}（{curr_types.get(case_id, '')}）" for case_id in ids)
        return "、".join(ids)

    lines.extend(
        [
            "## 失败对齐（按 case id）",
            "",
            f"- 新增失败 {len(new_ids)}：{render(new_ids, with_type=True)}",
            f"- 已修复 {len(fixed_ids)}：{render(fixed_ids)}",
            f"- 仍失败 {len(still_ids)}：{render(still_ids)}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对比两次评测运行")
    parser.add_argument("prev", type=Path, help="旧 run 目录")
    parser.add_argument("curr", type=Path, help="新 run 目录")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(compare_runs(args.prev, args.curr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
