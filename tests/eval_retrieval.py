"""检索评测：用真实 embedding 算 hit@k / MRR，并给出相关与无关问题的分数分布。

用法：

    python tests/eval_retrieval.py

只测检索召回，不测生成是否忠实；embedding 调用约 22 次（有进程内缓存，重跑同一批很便宜）。
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rag_eval.json"


def cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """两个矩阵都按行做 L2 归一化后点积，等价于余弦相似度。"""
    query = query / np.linalg.norm(query)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    return docs @ query


def evaluate(
    doc_ids: list[str],
    doc_vectors: list[list[float]] | np.ndarray,
    queries: list[dict],
    query_vectors: list[list[float]] | np.ndarray,
) -> list[dict]:
    """逐条问题排序文档，记录 top1、hit@k 与首次命中的倒数排名。"""
    docs = np.asarray(doc_vectors, dtype=float)
    query_matrix = np.asarray(query_vectors, dtype=float)
    results: list[dict] = []

    for query, vector in zip(queries, query_matrix, strict=True):
        scores = cosine_scores(vector, docs)
        order = np.argsort(-scores)
        ranked = [doc_ids[index] for index in order]
        relevant = set(query["relevant"])
        first_hit = next(
            (rank for rank, doc_id in enumerate(ranked, 1) if doc_id in relevant),
            None,
        )

        results.append(
            {
                "question": query["question"],
                "relevant": sorted(relevant),
                "ranked": ranked,
                "top_score": float(scores[order[0]]),
                "hit@1": (first_hit is not None and first_hit <= 1) if relevant else None,
                "hit@3": (first_hit is not None and first_hit <= 3) if relevant else None,
                "hit@5": (first_hit is not None and first_hit <= 5) if relevant else None,
                "rr": (1 / first_hit if first_hit else 0.0) if relevant else None,
            }
        )

    return results


def summarize(results: list[dict]) -> dict:
    """只对可回答问题算 hit@k / MRR；无关问题只保留分数，用来判断阈值。"""
    answerable = [result for result in results if result["relevant"]]
    unanswerable = [result for result in results if not result["relevant"]]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "hit@1": mean([float(result["hit@1"]) for result in answerable]),
        "hit@3": mean([float(result["hit@3"]) for result in answerable]),
        "hit@5": mean([float(result["hit@5"]) for result in answerable]),
        "mrr": mean([float(result["rr"]) for result in answerable]),
        "answerable_top_scores": [result["top_score"] for result in answerable],
        "unanswerable_top_scores": [result["top_score"] for result in unanswerable],
    }


def score_stats(scores: list[float]) -> str:
    if not scores:
        return "n/a"
    return f"min {min(scores):.3f} / median {float(np.median(scores)):.3f} / max {max(scores):.3f}"


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    documents = data["documents"]
    queries = data["queries"]

    # 代理必须在 OpenAI client 创建前写入环境变量
    from backend.config import get_settings
    from backend.core.net import apply_network_env

    apply_network_env(get_settings())

    from backend.store.embeddings import embed_texts

    doc_vectors = embed_texts([document["text"] for document in documents])
    query_vectors = embed_texts([query["question"] for query in queries])
    results = evaluate(
        [document["id"] for document in documents],
        doc_vectors,
        queries,
        query_vectors,
    )
    summary = summarize(results)

    print(
        f"文档 {len(documents)} 条，问题 {len(queries)} 条"
        f"（可回答 {summary['answerable']}，不可回答 {summary['unanswerable']}）"
    )
    print(
        f"hit@1 {summary['hit@1']:.2f}  "
        f"hit@3 {summary['hit@3']:.2f}  "
        f"hit@5 {summary['hit@5']:.2f}  "
        f"MRR {summary['mrr']:.2f}"
    )
    print(f"相关 top1 分数：{score_stats(summary['answerable_top_scores'])}")
    print(f"无关 top1 分数：{score_stats(summary['unanswerable_top_scores'])}")
    print()

    for result in results:
        if not result["relevant"]:
            mark = "?"
        elif result["hit@1"]:
            mark = "+"
        else:
            mark = "-"
        print(f"{mark} {result['question']}")
        print(
            f"   top1={result['ranked'][0]}  score={result['top_score']:.3f}"
            f"  expected={result['relevant']}"
        )


if __name__ == "__main__":
    main()
