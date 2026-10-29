"""真实数据检索评测：分品类 hit@k / MRR / recall@5，以及相似度阈值建议。

评测对象是 `_build_doc` 产出的报告关键文本（与 ChromaDB 实际索引的文本一致），
向量化走项目真实的 embedding 模型。数据集与报告快照都在 `tests/fixtures/`，
不重新拉数据、不调用 LLM，所以复跑只花 embedding 的钱。

用法：

    python tests/eval_retrieval.py
    python tests/eval_retrieval.py --dataset tests/fixtures/eval_dataset.json

阈值只给建议，不自动改 `/ask` 的行为；阈值能不能救、误召回和漏召回怎么权衡，
都在输出的 markdown 里写清楚。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_DATASET = FIXTURE_DIR / "eval_dataset.json"
DEFAULT_REPORTS = FIXTURE_DIR / "eval_reports" / "reports.json"

ANSWERABLE_CATEGORIES = ("factual", "comparison", "opinion")
CATEGORY_LABELS = {
    "factual": "事实型",
    "comparison": "跨股票对比型",
    "opinion": "观点型",
    "unanswerable": "无法回答型",
}


def cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """两个矩阵都按行做 L2 归一化后点积，等价于余弦相似度。"""
    query = query / np.linalg.norm(query)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    return docs @ query


def evaluate_query(
    query: dict,
    ranked: list[str],
    scores: list[float],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """记录一条问题的排名、hit@k、MRR 与多答案 recall@k。"""
    expected = set(query.get("expected_hits") or [])
    answerable = bool(query.get("answerable"))
    first_hit = next(
        (rank for rank, doc_id in enumerate(ranked, 1) if doc_id in expected),
        None,
    )

    def hit_at(k: int) -> bool | None:
        if not answerable or not expected:
            return None
        return first_hit is not None and first_hit <= k

    result = {
        "id": query["id"],
        "category": query["category"],
        "answerable": answerable,
        "unanswerable_type": query.get("unanswerable_type"),
        "question": query["question"],
        "expected": sorted(expected),
        "ranked": ranked,
        "scores": [round(float(score), 4) for score in scores],
        "top1": ranked[0] if ranked else None,
        "top1_score": round(float(scores[0]), 4) if scores else 0.0,
        "first_hit": first_hit,
    }
    for k in k_values:
        result[f"hit@{k}"] = hit_at(k)
    result["rr"] = (1 / first_hit if first_hit else 0.0) if answerable and expected else None
    result["recall@5"] = (
        len(expected & set(ranked[:5])) / len(expected) if answerable and expected else None
    )
    return result


def score_stats(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    values = np.asarray(scores, dtype=float)
    return {
        "count": len(scores),
        "min": round(float(values.min()), 3),
        "p25": round(float(np.percentile(values, 25)), 3),
        "median": round(float(np.median(values)), 3),
        "p75": round(float(np.percentile(values, 75)), 3),
        "max": round(float(values.max()), 3),
    }


def _mean(values: list[bool | float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 4) if clean else 0.0


def aggregate(results: list[dict]) -> dict:
    """按品类汇总；无法回答型只统计 top1 分布，不强行算 hit@k。"""
    groups: dict[str, list[dict]] = {}
    for result in results:
        groups.setdefault(result["category"], []).append(result)

    summary: dict[str, dict] = {}
    for category, items in groups.items():
        answerable = [item for item in items if item["answerable"]]
        if answerable:
            summary[category] = {
                "count": len(items),
                "answerable": len(answerable),
                "hit@1": _mean([item["hit@1"] for item in answerable]),
                "hit@3": _mean([item["hit@3"] for item in answerable]),
                "hit@5": _mean([item["hit@5"] for item in answerable]),
                "mrr": _mean([item["rr"] for item in answerable]),
                "recall@5": _mean([item["recall@5"] for item in answerable]),
                "top1_scores": score_stats([item["top1_score"] for item in answerable]),
            }
        else:
            by_type: dict[str, list[dict]] = {}
            for item in items:
                by_type.setdefault(item.get("unanswerable_type") or "unspecified", []).append(item)
            summary[category] = {
                "count": len(items),
                "answerable": 0,
                "by_type": {
                    key: {
                        "count": len(value),
                        "top1_scores": score_stats([item["top1_score"] for item in value]),
                    }
                    for key, value in by_type.items()
                },
            }

    answerable_results = [result for result in results if result["answerable"]]
    summary["__overall__"] = {
        "count": len(answerable_results),
        "hit@1": _mean([item["hit@1"] for item in answerable_results]),
        "hit@3": _mean([item["hit@3"] for item in answerable_results]),
        "hit@5": _mean([item["hit@5"] for item in answerable_results]),
        "mrr": _mean([item["rr"] for item in answerable_results]),
        "recall@5": _mean([item["recall@5"] for item in answerable_results]),
    }
    return summary


def _percentile(scores: list[float], q: float) -> float | None:
    if not scores:
        return None
    return round(float(np.percentile(np.asarray(scores, dtype=float), q)), 4)


def threshold_candidates(
    answerable_scores: list[float],
    out_of_scope_scores: list[float],
) -> list[dict]:
    """在几个有解释性的分位点上枚举阈值，算误召回 / 漏召回。"""
    if not answerable_scores or not out_of_scope_scores:
        return []

    candidates: list[tuple[str, float]] = []
    for q in (50, 75, 90, 95, 100):
        value = _percentile(out_of_scope_scores, q)
        if value is not None:
            candidates.append((f"无关问题 top1 P{q}", value))
    for q in (0, 5, 10, 25):
        value = _percentile(answerable_scores, q)
        if value is not None:
            candidates.append((f"可回答问题 top1 P{q}", value))

    rows: dict[float, dict] = {}
    for source, threshold in candidates:
        threshold = round(threshold, 3)
        false_accept = sum(score >= threshold for score in out_of_scope_scores) / len(
            out_of_scope_scores
        )
        false_reject = sum(score < threshold for score in answerable_scores) / len(
            answerable_scores
        )
        rows[threshold] = {
            "source": source,
            "threshold": threshold,
            "false_accept": round(false_accept, 4),
            "false_reject": round(false_reject, 4),
            "retention": round(1 - false_reject, 4),
            "balanced_accuracy": round(1 - (false_accept + false_reject) / 2, 4),
        }
    return sorted(rows.values(), key=lambda row: (row["threshold"], row["source"]))


def threshold_analysis(
    answerable_scores: list[float],
    out_of_scope_scores: list[float],
    missing_fact_scores: list[float],
) -> dict:
    """推荐阈值 = 无关问题 top1 的 P90。

    这个项目的设计优先级是「宁可说未覆盖，也不要编」：先把误召回压到 10% 以内，
    再看可回答问题被误拒多少。单看平衡准确率会把这批 60 条可回答问题
    和 10 条无关问题等权，和实际风险不对称，所以只作为参考列。
    """
    candidates = threshold_candidates(answerable_scores, out_of_scope_scores)
    if not candidates or not out_of_scope_scores:
        return {"threshold": None, "policy": "样本不足", "candidates": candidates}

    threshold = round(float(np.percentile(np.asarray(out_of_scope_scores), 90)), 3)
    false_accept = sum(score >= threshold for score in out_of_scope_scores) / len(
        out_of_scope_scores
    )
    false_reject = sum(score < threshold for score in answerable_scores) / len(answerable_scores)
    missing_above = (
        sum(score >= threshold for score in missing_fact_scores) / len(missing_fact_scores)
        if missing_fact_scores
        else None
    )
    return {
        "threshold": threshold,
        "policy": "无关问题 top1 的 P90，优先避免误召回",
        "false_accept": round(false_accept, 4),
        "false_reject": round(false_reject, 4),
        "retention": round(1 - false_reject, 4),
        "missing_fact_above": round(missing_above, 4) if missing_above is not None else None,
        "missing_fact_count": len(missing_fact_scores),
        "candidates": candidates,
    }


def load_dataset(path: Path = DEFAULT_DATASET) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reports(path: Path = DEFAULT_REPORTS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_retrieval(dataset_path: Path = DEFAULT_DATASET, reports_path: Path = DEFAULT_REPORTS):
    """用真实 embedding 对快照做全量排序，返回可直接渲染的结果。"""
    dataset = load_dataset(dataset_path)
    report_ids = dataset.get("report_ids") or []
    records = [record for record in load_reports(reports_path) if record["id"] in report_ids]
    if len(records) != len(report_ids):
        missing = sorted(set(report_ids) - {record["id"] for record in records})
        raise RuntimeError(f"快照缺报告：{missing}")

    from backend.config import get_settings
    from backend.core.net import apply_network_env

    apply_network_env(get_settings())

    from backend.store.chroma_store import _build_doc
    from backend.store.embeddings import embed_texts

    doc_ids = [record["id"] for record in records]
    doc_texts = [_build_doc(record) for record in records]
    questions = dataset["queries"]
    doc_vectors = embed_texts(doc_texts)
    query_vectors = embed_texts([query["question"] for query in questions])
    docs = np.asarray(doc_vectors, dtype=float)

    results = []
    for query, vector in zip(questions, query_vectors, strict=True):
        scores = cosine_scores(np.asarray(vector, dtype=float), docs)
        order = np.argsort(-scores)
        ranked = [doc_ids[index] for index in order]
        results.append(evaluate_query(query, ranked, [float(scores[index]) for index in order]))

    answerable_scores = [result["top1_score"] for result in results if result["answerable"]]
    out_of_scope_scores = [
        result["top1_score"]
        for result in results
        if not result["answerable"] and result.get("unanswerable_type") == "out_of_scope"
    ]
    missing_fact_scores = [
        result["top1_score"]
        for result in results
        if not result["answerable"] and result.get("unanswerable_type") == "missing_fact"
    ]
    return {
        "dataset": dataset,
        "results": results,
        "summary": aggregate(results),
        "threshold": threshold_analysis(
            answerable_scores, out_of_scope_scores, missing_fact_scores
        ),
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def render_retrieval_markdown(result: dict) -> str:
    dataset = result["dataset"]
    summary = result["summary"]
    threshold = result["threshold"]
    lines = [
        "## 检索评测",
        "",
        f"数据：{len(dataset.get('report_ids', []))} 份真实报告，"
        f"{len(dataset['queries'])} 条问题；快照与标注见 "
        f"`tests/fixtures/eval_reports/`、`tests/eval_labels.md`。",
        "",
        "| 品类 | 题数 | hit@1 | hit@3 | hit@5 | MRR | recall@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in ANSWERABLE_CATEGORIES:
        item = summary.get(category)
        if not item:
            continue
        lines.append(
            f"| {CATEGORY_LABELS[category]} | {item['answerable']} | "
            f"{item['hit@1']:.2f} | {item['hit@3']:.2f} | {item['hit@5']:.2f} | "
            f"{item['mrr']:.2f} | {item['recall@5']:.2f} |"
        )
    overall = summary["__overall__"]
    lines.append(
        f"| **合计（可回答）** | {overall['count']} | {overall['hit@1']:.2f} | "
        f"{overall['hit@3']:.2f} | {overall['hit@5']:.2f} | "
        f"{overall['mrr']:.2f} | {overall['recall@5']:.2f} |"
    )
    lines.extend(
        [
            "",
            "hit@k / MRR 对多答案题按「至少命中一个期望报告」计算；recall@5 按"
            "「top5 覆盖了多少期望报告」计算。观点题的期望集合可能大于 5，"
            "此时 recall@5 的结构上限就是 5/期望数，不能单独拿来判断检索好坏。",
        ]
    )

    unanswerable = summary.get("unanswerable")
    if unanswerable:
        lines.extend(
            [
                "",
                "无法回答型只统计 top1 分数分布；它们没有「应该命中的报告」，不能算 hit@k。",
                "",
                "| 子类 | 题数 | top1 min | P25 | 中位数 | P75 | max |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, item in unanswerable.get("by_type", {}).items():
            stats = item["top1_scores"]
            label = {"out_of_scope": "完全超出覆盖范围", "missing_fact": "报告未包含该事实"}.get(
                key, key
            )
            lines.append(
                f"| {label} | {item['count']} | {stats['min']} | {stats['p25']} | "
                f"{stats['median']} | {stats['p75']} | {stats['max']} |"
            )

    lines.extend(
        [
            "",
            "### 相似度阈值建议",
            "",
            f"- 推荐阈值：**{threshold['threshold']}**（{threshold['policy']}）",
            f"- 在该阈值下，无关问题误召回：{_pct(threshold['false_accept'])}；"
            f"可回答问题漏召回：{_pct(threshold['false_reject'])}；"
            f"保留率：{_pct(threshold['retention'])}。",
        ]
    )
    if threshold.get("missing_fact_above") is not None:
        lines.append(
            f"- 报告缺少事实型问题的 top1 高于阈值的比例："
            f"{_pct(threshold['missing_fact_above'])}。这类问题检索到了正确公司，"
            "但报告里没有该字段，阈值救不了，只能靠 prompt / 拒答判断。"
        )
    if threshold.get("candidates"):
        lines.extend(
            [
                "",
                "| 候选口径 | 阈值 | 误召回 | 漏召回 | 保留率 | 平衡准确率（参考） |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in threshold["candidates"]:
            lines.append(
                f"| {row['source']} | {row['threshold']} | {_pct(row['false_accept'])} | "
                f"{_pct(row['false_reject'])} | {_pct(row['retention'])} | "
                f"{row['balanced_accuracy']:.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实数据检索评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result = run_retrieval(args.dataset, args.reports)
    markdown = render_retrieval_markdown(result)
    print(markdown)
    output = Path(__file__).resolve().parent / "eval_retrieval_report.md"
    output.write_text(markdown, encoding="utf-8")
    print(f"已写入 {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
