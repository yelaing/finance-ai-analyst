"""生成侧真实评测：跑 `/api/v1/ask`，判引用是否成立、是否拒答、是否编造。

- 30 条问题固定抽样：8 事实 + 7 对比 + 5 观点 + 10 无法回答（5 超纲 + 5 缺字段）；
- 回答全部来自真实接口，检索库是 `tests/fixtures/eval_reports/` 的 30 份快照；
- 引用逐条交给 LLM judge，并用 10 条人工抽查校准 judge 一致率；
- 失败 case 按「检索没召回 / 召回了但无关 / 召回了正确报告但模型没用 / 模型编造」
  归因；「引用不支撑」作为引用级细因单列，写进 `tests/eval_failures.md`。

不为了让指标好看改 prompt、改检索参数或改标签；阈值只做模拟，不自动改 `/ask`。
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_DATASET = FIXTURE_DIR / "eval_dataset.json"
DEFAULT_REPORTS = FIXTURE_DIR / "eval_reports" / "reports.json"
ANSWER_CACHE = FIXTURE_DIR / "eval_answers.json"
JUDGE_CACHE = FIXTURE_DIR / "eval_judge.json"
MANUAL_SPOTCHECK = FIXTURE_DIR / "eval_spotcheck_manual.json"
SPOTCHECK_MD = Path(__file__).resolve().parent / "eval_judge_spotcheck.md"

# 与 backend/pipeline/rag.py 保持一致；运行时会断言一次，防止漂移。
NO_HITS_ANSWER = "没有找到相关的历史分析报告。请先分析相关股票，或换一种问法。"
CACHE_VERSION = 1
JUDGE_VERSION = 2
LIMIT = 5

logger = logging.getLogger("eval_generation")

JUDGE_PROMPT = """你是严格的 RAG 评测员，只做判定，不回答用户问题。

## 问题
{question}

## 参考答案（用于判断相关性，不要要求回答逐字一致）
{reference}

## 系统回答
{answer}

## 引用片段
{context}

## 判定要求
1. citations：只对「系统回答」中实际出现的 [n] 逐个判定；同一编号只判一次；
   回答中出现的每个 [n] 都必须返回一条判定，不得省略。
   supported=true 表示该片段直接支撑引用所在句子的关键事实，而不是仅仅主题相关。
   正向列举、排除项、对比项都要按引用所在句子判断：句子给出具体数字/事实，
   且片段中能找到对应内容，就判 true，即使该句用于排除某个对象。
   句子声称「未提及」「仅有」「均未」「唯一」等范围性断言时，只有片段本身明确
   支持这个范围或完整性才判 true；片段只是没出现某个词，不能证明「未提及」。
   index 必须在 1 到 {count} 之间；没有引用就返回空数组。
2. relevance：1-5 分。回答是否切题、是否覆盖参考答案的关键点；
   对本身无法回答的问题，明确说明未覆盖也可以得高分。
3. refused：回答是否明确表示资料不足 / 现有报告未覆盖，而不是给出具体答案。
4. fabricated：回答是否包含引用片段中没有的具体事实、数字或公司；笼统建议不算。

只输出 JSON，不要 markdown 代码块：
{{"citations": [{{"index": 1, "supported": true, "reason": "..."}}],
 "relevance": 4, "refused": false, "fabricated": false, "notes": "..."}}
"""


# ---------- 纯函数：抽样、引用、阈值、归因 ----------


def select_generation_sample(dataset: dict) -> list[dict]:
    """固定抽样，避免每次评测换题导致指标不可比。"""
    queries = dataset["queries"]
    factual = [query for query in queries if query["category"] == "factual"]
    comparison = [query for query in queries if query["category"] == "comparison"]
    opinion = [query for query in queries if query["category"] == "opinion"]
    unanswerable = [query for query in queries if query["category"] == "unanswerable"]
    out_of_scope = [query for query in unanswerable if query["unanswerable_type"] == "out_of_scope"]
    missing_fact = [query for query in unanswerable if query["unanswerable_type"] == "missing_fact"]

    by_industry = []
    for industry in ("金融", "白酒", "新能源", "医药"):
        by_industry.extend([query for query in comparison if query.get("industry") == industry][:2])

    sample = (
        factual[::2][:8]
        + by_industry[:7]
        + opinion[::3][:5]
        + out_of_scope[::2][:5]
        + missing_fact[::2][:5]
    )
    ids = [query["id"] for query in sample]
    if len(ids) != len(set(ids)) or len(sample) != 30:
        raise RuntimeError(f"生成评测抽样应得到 30 条唯一问题，实际 {len(sample)} 条")
    return sample


def extract_citation_indices(answer: str) -> list[int]:
    """按首次出现顺序返回回答里的 [n]，重复编号只算一次。"""
    seen = set()
    indices = []
    for match in re.finditer(r"\[(\d+)\]", answer or ""):
        index = int(match.group(1))
        if index not in seen:
            seen.add(index)
            indices.append(index)
    return indices


def citation_metrics(answer: str, citations: list[dict], judge: dict) -> dict:
    """逐条引用判定；越界编号算不支持，但保留在分母里。"""
    indices = extract_citation_indices(answer)
    valid = [index for index in indices if 1 <= index <= len(citations)]
    invalid = [index for index in indices if index not in valid]
    judge_map = {
        int(item["index"]): bool(item.get("supported"))
        for item in judge.get("citations") or []
        if isinstance(item, dict) and str(item.get("index", "")).isdigit()
    }
    supported = [index for index in valid if judge_map.get(index)]
    cited_ids = [citations[index - 1]["id"] for index in valid]
    return {
        "indices": indices,
        "valid_indices": valid,
        "invalid_indices": invalid,
        "supported_indices": supported,
        "unsupported_indices": [index for index in valid if not judge_map.get(index)],
        "cited_ids": cited_ids,
        "accuracy": (len(supported) / len(indices)) if indices else None,
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def value_matched(answer: str, expected_value: str) -> bool:
    return _normalize(expected_value) in _normalize(answer)


def apply_threshold(result: dict, threshold: float | None) -> dict:
    """模拟「top1 低于阈值就当没检索到」，不修改生产接口。"""
    top1 = result.get("top1_score")
    if top1 is not None and threshold is not None and top1 < threshold:
        return {"applied": True, "answer": NO_HITS_ANSWER}
    return {"applied": False, "answer": result["answer"]}


def classify_failure(query: dict, result: dict, judge: dict, metrics: dict) -> dict | None:
    """归因优先级：编造 > 检索没召回 > 召回了但无关 > 召回了正确报告但模型没用。

    「引用不支撑」是引用级细因：如果正确报告已经召回并被使用，只是某条引用
    不成立，单独归为此类，不强行算成「模型没用」。
    """
    answerable = bool(query.get("answerable"))
    expected = set(query.get("expected_hits") or [])
    retrieved = {
        citation.get("id") for citation in result.get("citations") or [] if citation.get("id")
    }
    cited = set(metrics.get("cited_ids") or [])
    fabricated = bool(judge.get("fabricated"))
    refused = bool(judge.get("refused"))
    relevance = judge.get("relevance") or 0
    reasons: list[str] = []

    if fabricated:
        reasons.append("模型编造")

    if answerable:
        if expected and not expected & retrieved:
            reasons.append("检索没召回")
        elif expected and not expected & cited:
            reasons.append("召回了正确报告但模型没用")
        if refused:
            reasons.append("错误拒答")
        if relevance <= 2:
            reasons.append("回答不相关")
        if metrics.get("accuracy") is not None and metrics["accuracy"] < 1:
            reasons.append("引用不支撑")
        if query.get("expected_value") and not value_matched(
            result.get("answer", ""), query["expected_value"]
        ):
            reasons.append("未答出参考值")
    else:
        if not refused:
            reasons.append("未拒答")
        if query.get("unanswerable_type") == "out_of_scope":
            # 超纲题检索到报告是常态；只要模型正确拒答，就不算失败。
            if not refused:
                reasons.append("召回了但无关")
        elif expected and not expected & retrieved:
            reasons.append("检索没召回")
        elif expected and expected & retrieved and not refused:
            reasons.append("召回了正确报告但模型没用")

    if not reasons:
        return None

    priority = (
        "模型编造",
        "检索没召回",
        "召回了但无关",
        "召回了正确报告但模型没用",
        "未拒答",
        "错误拒答",
        "引用不支撑",
        "回答不相关",
        "未答出参考值",
    )
    primary = next((reason for reason in priority if reason in reasons), reasons[0])
    # 失败库主因以四类为骨架；引用不支撑无法归入「模型没用」时单列，
    # 避免把「用了正确报告但引用写错」误标成「正确报告没被使用」。
    if primary == "引用不支撑":
        pass
    elif primary in {"回答不相关", "未答出参考值", "错误拒答"}:
        primary = "召回了正确报告但模型没用" if expected & retrieved else "检索没召回"
    elif primary == "未拒答":
        if query.get("unanswerable_type") == "out_of_scope":
            primary = "召回了但无关"
        elif expected & retrieved:
            primary = "召回了正确报告但模型没用"
        else:
            primary = "检索没召回"
    return {"primary": primary, "reasons": reasons}


# ---------- 缓存与真实接口 ----------


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash(payload) -> str:
    dumped = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def dataset_hash(dataset: dict, model: str) -> str:
    return _hash(
        {
            "version": CACHE_VERSION,
            "model": model,
            "limit": LIMIT,
            "snapshot_generated_at": dataset.get("snapshot_generated_at"),
            "report_ids": dataset.get("report_ids"),
            "queries": dataset.get("queries"),
        }
    )


def _make_store(dataset: dict, reports_path: Path):
    from backend.store.chroma_store import ReportStore

    report_ids = set(dataset.get("report_ids") or [])
    records = [record for record in _read_json(reports_path, []) if record.get("id") in report_ids]
    if len(records) != len(report_ids):
        missing = sorted(report_ids - {record["id"] for record in records})
        raise RuntimeError(f"快照缺报告：{missing}")

    temporary = tempfile.TemporaryDirectory(
        prefix="finance_eval_chroma_",
        ignore_cleanup_errors=True,
    )
    data_file = Path(temporary.name) / "reports.json"
    data_file.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    store = ReportStore(
        data_file=str(data_file),
        chroma_dir=str(Path(temporary.name) / "chroma"),
    )
    return store, temporary


def _ask_once(client, question: str) -> dict:
    response = client.post(
        "/api/v1/ask",
        json={"question": question, "limit": LIMIT},
    )
    if response.status_code != 200:
        raise RuntimeError(f"/ask 返回 {response.status_code}: {response.text[:300]}")
    data = response.json()
    citations = data.get("citations") or []
    return {
        "answer": data.get("answer") or "",
        "citations": citations,
        "top1_score": citations[0].get("score") if citations else None,
    }


def collect_answers(
    sample: list[dict],
    dataset: dict,
    reports_path: Path = DEFAULT_REPORTS,
    refresh: bool = False,
) -> dict[str, dict]:
    """优先读缓存；缓存不匹配或 --refresh 时才真实调用接口并落盘。"""
    from backend.config import get_settings
    from backend.core.net import apply_network_env

    apply_network_env(get_settings())
    model = get_settings().llm_model
    current_hash = dataset_hash(dataset, model)
    cache = _read_json(ANSWER_CACHE, {})
    answers = cache.get("answers") or {} if cache.get("dataset_hash") == current_hash else {}
    missing = [query for query in sample if refresh or query["id"] not in answers]
    if not missing:
        logger.info("回答全部命中缓存（%d 条）", len(sample))
        return {query["id"]: answers[query["id"]] for query in sample}

    import backend.api.routes as routes
    from backend.main import app

    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确错误
        raise RuntimeError("需要 fastapi/httpx 才能跑生成评测") from exc

    store, temporary = _make_store(dataset, reports_path)
    original_get_store = routes.get_store
    routes.get_store = lambda: store
    try:
        with TestClient(app) as client:
            for query in missing:
                started = time.perf_counter()
                answer = _ask_once(client, query["question"])
                answers[query["id"]] = answer
                _write_json(
                    ANSWER_CACHE,
                    {
                        "dataset_hash": current_hash,
                        "model": model,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "answers": answers,
                    },
                )
                logger.info(
                    "[ask] %s %.1fs top1=%s",
                    query["id"],
                    time.perf_counter() - started,
                    answer["top1_score"],
                )
    finally:
        routes.get_store = original_get_store
        temporary.cleanup()

    return {query["id"]: answers[query["id"]] for query in sample}


# ---------- LLM judge ----------


_judge_chain = None


def _get_judge_chain():
    global _judge_chain
    if _judge_chain is not None:
        return _judge_chain

    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    from backend.config import get_settings

    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=900,
        request_timeout=settings.llm_timeout,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是严格、简洁的 RAG 评测员，只输出 JSON。"),
            ("human", JUDGE_PROMPT),
        ]
    )
    _judge_chain = (prompt | llm | JsonOutputParser()).with_retry(
        stop_after_attempt=2,
        exponential_jitter_params={"initial": 1, "max": 2, "jitter": 0},
    )
    return _judge_chain


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "是"}
    return bool(value)


def _format_context(citations: list[dict]) -> str:
    return (
        "\n\n".join(
            f"[{index}] {hit.get('name')}（{hit.get('symbol')}）\n{hit.get('text', '')}"
            for index, hit in enumerate(citations, 1)
        )
        or "（无引用片段）"
    )


def judge_answer(query: dict, answer: dict, refresh: bool = False) -> dict:
    from backend.config import get_settings

    cache = _read_json(JUDGE_CACHE, {})
    key = _hash(
        {
            "version": JUDGE_VERSION,
            "model": get_settings().llm_model,
            "question": query["question"],
            "reference": query.get("reference_answer"),
            "answer": answer["answer"],
            "citations": answer["citations"],
        }
    )
    if not refresh and key in cache:
        return cache[key]

    raw = _get_judge_chain().invoke(
        {
            "question": query["question"],
            "reference": query.get("reference_answer") or "",
            "answer": answer["answer"],
            "context": _format_context(answer["citations"]),
            "count": len(answer["citations"]),
        }
    )
    if not isinstance(raw, dict):
        raise ValueError(f"judge 返回不是 JSON 对象：{raw!r}")
    relevance = raw.get("relevance")
    if not isinstance(relevance, (int, float)):
        raise ValueError(f"judge 缺少 relevance：{raw!r}")

    citations = []
    for item in raw.get("citations") or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        citations.append(
            {
                "index": index,
                "supported": _as_bool(item.get("supported")),
                "reason": str(item.get("reason") or ""),
            }
        )
    verdict = {
        "citations": citations,
        "relevance": max(1, min(5, int(round(float(relevance))))),
        "refused": _as_bool(raw.get("refused")),
        "fabricated": _as_bool(raw.get("fabricated")),
        "notes": str(raw.get("notes") or ""),
    }
    cache[key] = verdict
    _write_json(JUDGE_CACHE, cache)
    return verdict


# ---------- 汇总 ----------


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def evaluate_generation(
    sample: list[dict],
    answers: dict[str, dict],
    dataset: dict,
    threshold: float | None,
    refresh: bool = False,
) -> dict:
    query_by_id = {query["id"]: query for query in dataset["queries"]}
    results = []
    for query in sample:
        answer = answers[query["id"]]
        verdict = judge_answer(query, answer, refresh=refresh)
        metrics = citation_metrics(answer["answer"], answer["citations"], verdict)
        failure = classify_failure(query, answer, verdict, metrics)
        simulated = apply_threshold(answer, threshold)
        simulated_refused = True if simulated["applied"] else verdict["refused"]
        results.append(
            {
                "id": query["id"],
                "category": query["category"],
                "answerable": bool(query.get("answerable")),
                "unanswerable_type": query.get("unanswerable_type"),
                "question": query["question"],
                "reference_answer": query.get("reference_answer"),
                "answer": answer["answer"],
                "citations": answer["citations"],
                "top1_score": answer["top1_score"],
                "judge": verdict,
                "citation_metrics": metrics,
                "failure": failure,
                "simulated": {
                    "applied": simulated["applied"],
                    "refused": simulated_refused,
                    # 阈值触发时输出就是拒答文案，不应继承原回答的编造判定
                    "fabricated": False if simulated["applied"] else verdict["fabricated"],
                    "answer": simulated["answer"],
                },
                "query": query_by_id[query["id"]],
            }
        )

    answerable = [result for result in results if result["answerable"]]
    unanswerable = [result for result in results if not result["answerable"]]
    answerable_with_citations = [
        result for result in answerable if result["citation_metrics"]["indices"]
    ]
    total_citations = sum(len(result["citation_metrics"]["indices"]) for result in answerable)
    total_supported = sum(
        len(result["citation_metrics"]["supported_indices"]) for result in answerable
    )
    refusal_correct = [
        result
        for result in unanswerable
        if result["judge"]["refused"] and not result["judge"]["fabricated"]
    ]

    categories = {}
    for category in ("factual", "comparison", "opinion", "unanswerable"):
        subset = [result for result in results if result["category"] == category]
        if not subset:
            continue
        with_citations = [result for result in subset if result["citation_metrics"]["indices"]]
        categories[category] = {
            "count": len(subset),
            "citation_accuracy": _mean(
                [result["citation_metrics"]["accuracy"] for result in with_citations]
            ),
            "citation_coverage": round(len(with_citations) / len(subset), 4),
            "relevance": _mean([result["judge"]["relevance"] for result in subset]),
            "fabricated": sum(1 for result in subset if result["judge"]["fabricated"]),
            "refused": sum(1 for result in subset if result["judge"]["refused"]),
        }

    return {
        "results": results,
        "summary": {
            "count": len(results),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "citation_accuracy": (
                round(total_supported / total_citations, 4) if total_citations else None
            ),
            "citation_coverage": (
                round(len(answerable_with_citations) / len(answerable), 4) if answerable else None
            ),
            "citation_total": total_citations,
            "citation_supported": total_supported,
            "relevance": _mean([result["judge"]["relevance"] for result in results]),
            "relevance_answerable": _mean([result["judge"]["relevance"] for result in answerable]),
            "relevance_unanswerable": _mean(
                [result["judge"]["relevance"] for result in unanswerable]
            ),
            "fabricated": sum(1 for result in results if result["judge"]["fabricated"]),
            "fabricated_answerable": sum(
                1 for result in answerable if result["judge"]["fabricated"]
            ),
            "fabricated_unanswerable": sum(
                1 for result in unanswerable if result["judge"]["fabricated"]
            ),
            "wrong_refusal": sum(1 for result in answerable if result["judge"]["refused"]),
            "refusal_correct": len(refusal_correct),
            "refusal_total": len(unanswerable),
            "refusal_accuracy": (
                round(len(refusal_correct) / len(unanswerable), 4) if unanswerable else None
            ),
            "simulated_wrong_refusal": sum(
                1 for result in answerable if result["simulated"]["refused"]
            ),
            "simulated_refusal_correct": sum(
                1
                for result in unanswerable
                if result["simulated"]["refused"] and not result["simulated"]["fabricated"]
            ),
            "categories": categories,
        },
        "threshold": threshold,
    }


# ---------- 人工抽查 ----------


def select_spotcheck(generation: dict, count: int = 10) -> list[dict]:
    results = generation["results"]
    samples = []
    step = max(1, len(results) // count)
    candidates = results[::step] + results
    seen = set()
    for result in candidates:
        if len(samples) >= count:
            break
        for index in result["citation_metrics"]["indices"]:
            if not (1 <= index <= len(result["citations"])):
                continue
            key = (result["id"], index)
            if key in seen:
                continue
            seen.add(key)
            citation = result["citations"][index - 1]
            judge_item = next(
                (item for item in result["judge"]["citations"] if item["index"] == index),
                None,
            )
            answer_sentence = next(
                (
                    sentence.strip()
                    for sentence in re.split(r"[。；\n]", result["answer"])
                    if f"[{index}]" in sentence
                ),
                result["answer"][:120],
            )
            samples.append(
                {
                    "sample_id": f"S{len(samples) + 1:02d}",
                    "query_id": result["id"],
                    "category": result["category"],
                    "question": result["question"],
                    "citation_index": index,
                    "answer_sentence": answer_sentence,
                    "citation_text": citation.get("text", "")[:500],
                    "citation_name": citation.get("name"),
                    "judge_supported": bool(judge_item and judge_item["supported"]),
                    "judge_reason": judge_item["reason"] if judge_item else "judge 未返回该编号",
                }
            )
            break
    return samples


def spotcheck_agreement(samples: list[dict]) -> dict | None:
    manual = _read_json(MANUAL_SPOTCHECK, {})
    entries = {item.get("sample_id"): item for item in manual.get("samples") or []}
    if not samples or not all(sample["sample_id"] in entries for sample in samples):
        return None
    agree = sum(
        1
        for sample in samples
        if bool(entries[sample["sample_id"]].get("supported")) == sample["judge_supported"]
    )
    return {
        "total": len(samples),
        "agree": agree,
        "agreement": round(agree / len(samples), 4),
        "manual": entries,
    }


def _escape_cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_spotcheck(samples: list[dict], agreement: dict | None) -> str:
    lines = [
        "# Judge 人工抽查",
        "",
        "从 30 条生成评测中抽 10 条引用，人工判断引用片段是否直接支撑回答中的句子，"
        "与 judge 的 `supported` 对比。人工结论填在 "
        f"`{MANUAL_SPOTCHECK.relative_to(REPO_ROOT)}`，再重跑本脚本即可算出并写入一致率。",
        "",
    ]
    if agreement:
        lines.append(
            f"**一致率：{agreement['agree']}/{agreement['total']} = {agreement['agreement']:.0%}**"
        )
        lines.append("")
    lines.extend(
        [
            "| 样本 | 问题 | 引用句 | 引用片段 | Judge | 人工 | 备注 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for sample in samples:
        manual = (agreement or {}).get("manual", {}).get(sample["sample_id"], {})
        manual_text = (
            "支持"
            if manual.get("supported") is True
            else "不支持"
            if manual.get("supported") is False
            else "待填"
        )
        judge_text = "支持" if sample["judge_supported"] else "不支持"
        lines.append(
            f"| {sample['sample_id']} | {_escape_cell(sample['question'][:40])} | "
            f"{_escape_cell(sample['answer_sentence'][:80])} | "
            f"{_escape_cell(sample['citation_text'][:100])} | {judge_text} | {manual_text} | "
            f"{_escape_cell(manual.get('note', sample['judge_reason'])[:60])} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------- Markdown ----------


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def render_generation_markdown(generation: dict) -> str:
    summary = generation["summary"]
    lines = [
        "## 生成评测",
        "",
        f"数据：{summary['count']} 条问题（可回答 {summary['answerable']}，"
        f"无法回答 {summary['unanswerable']}）；真实调用 `/api/v1/ask`，limit=5。",
        "",
        "| 指标 | 数值 | 口径 |",
        "|---|---:|---|",
        f"| 引用准确率 | {_pct(summary['citation_accuracy'])} | "
        f"{summary['citation_supported']}/{summary['citation_total']} 条实际引用被 judge 判为支撑 |",
        f"| 引用覆盖率 | {_pct(summary['citation_coverage'])} | 可回答回答中至少有一个 [n] 的比例 |",
        f"| 相关性（1-5） | {_num(summary['relevance'])} | 全部 30 条；可回答 "
        f"{_num(summary['relevance_answerable'])} / 无法回答 "
        f"{_num(summary['relevance_unanswerable'])} |",
        f"| 拒答准确率 | {_pct(summary['refusal_accuracy'])} | "
        f"{summary['refusal_correct']}/{summary['refusal_total']} 条无法回答题正确说明未覆盖 |",
        f"| 错误拒答 | {summary['wrong_refusal']} | 可回答问题被拒答的条数 |",
        f"| 编造 | {summary['fabricated']} | judge 判定回答含引用片段之外的具体事实/数字；"
        f"可回答 {summary['fabricated_answerable']} / 无法回答 {summary['fabricated_unanswerable']} |",
        "",
        "### 分品类",
        "",
        "| 品类 | 题数 | 引用准确率 | 引用覆盖率 | 相关性 | 编造 | 拒答 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "factual": "事实型",
        "comparison": "对比型",
        "opinion": "观点型",
        "unanswerable": "无法回答型",
    }
    for category, item in summary["categories"].items():
        lines.append(
            f"| {labels.get(category, category)} | {item['count']} | "
            f"{_pct(item['citation_accuracy'])} | {_pct(item['citation_coverage'])} | "
            f"{_num(item['relevance'])} | {item['fabricated']} | {item['refused']} |"
        )

    threshold = generation.get("threshold")
    lines.extend(
        [
            "",
            "### 阈值模拟（不修改 `/ask`）",
            "",
        ]
    )
    if threshold is None:
        lines.append("- 未提供阈值，跳过模拟。")
    else:
        lines.append(
            f"- 阈值 {threshold}：模拟「top1 低于阈值直接返回未覆盖」。"
            f"无法回答题正确拒答 {summary['simulated_refusal_correct']}/"
            f"{summary['refusal_total']}；可回答题被误拒 {summary['simulated_wrong_refusal']}/"
            f"{summary['answerable']}。"
        )
        lines.append(
            "- 这只是模拟，生产 `/ask` 仍按原逻辑：检索到任何报告就调用 LLM。"
            "缺失事实型问题的 top1 通常很高，阈值救不了，必须靠 prompt/拒答判断。"
        )
    lines.append("")
    return "\n".join(lines)


def render_failures_markdown(generation: dict) -> str:
    failures = [result for result in generation["results"] if result["failure"]]
    counts = Counter(result["failure"]["primary"] for result in failures)
    lines = [
        "# 生成评测失败 case",
        "",
        f"共 {len(failures)} 条失败（judge 口径）。归因优先级：编造 > 检索没召回 > "
        "召回了但无关 > 召回了正确报告但模型没用；「引用不支撑」是四类之外的引用级细因，"
        "单列，避免把「用了正确报告但引用写错」误标成「模型没用」。",
        "",
        "已人工复核：C02 的「引用不支撑」是 judge 漏判（引用[5]片段明确含中信证券 82.54%，"
        "judge 理由也写了应判 true，最终仍返回 false），这里保留原始判定、不手工改分；"
        "但 C02 回答首句把招商银行列入、后文又排除，存在自相矛盾，未计入编造。",
        "",
        "| 主因 | 条数 |",
        "|---|---:|",
    ]
    for reason, count in counts.most_common():
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    if not failures:
        lines.append("本次 30 条生成评测没有失败 case。")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| ID | 品类 | 问题 | 主因 | 全部原因 | 是否修复 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for result in failures:
        lines.append(
            f"| {result['id']} | {result['category']} | "
            f"{result['question'][:60]} | {result['failure']['primary']} | "
            f"{'、'.join(result['failure']['reasons'])} | 否（本次只记录） |"
        )
    lines.append("")

    for result in failures:
        lines.extend(
            [
                f"## {result['id']} {result['category']}",
                "",
                f"**问题**：{result['question']}",
                "",
                f"**期望**：{result['reference_answer']}",
                "",
                f"**实际回答**：{result['answer']}",
                "",
                "**检索 top5**："
                + "；".join(
                    f"[{index}] {hit.get('name')}（{hit.get('symbol')}，{hit.get('score')}）"
                    for index, hit in enumerate(result["citations"], 1)
                ),
                "",
                f"**输出引用**：{result['citation_metrics']['indices'] or '无'}；"
                f"judge 支持 {result['citation_metrics']['supported_indices'] or '无'}；"
                f"越界 {result['citation_metrics']['invalid_indices'] or '无'}",
                "",
                f"**Judge**：relevance={result['judge']['relevance']}，"
                f"refused={result['judge']['refused']}，"
                f"fabricated={result['judge']['fabricated']}；"
                f"{result['judge']['notes']}",
                "",
                f"**失败原因**：{'、'.join(result['failure']['reasons'])}",
                "",
                "**是否修复**：否。本次只做真实评测，没有为了让指标好看改 prompt、检索参数或标签。",
                "",
            ]
        )
    return "\n".join(lines)


# ---------- 入口 ----------


def run_generation(
    dataset_path: Path = DEFAULT_DATASET,
    reports_path: Path = DEFAULT_REPORTS,
    threshold: float | None = None,
    refresh: bool = False,
) -> dict:
    from backend.config import get_settings
    from backend.core.net import apply_network_env

    apply_network_env(get_settings())
    from backend.pipeline.rag import NO_HITS_ANSWER as production_no_hits

    if production_no_hits != NO_HITS_ANSWER:
        raise RuntimeError("eval_generation.NO_HITS_ANSWER 与生产常量不一致，请同步")

    dataset = _read_json(dataset_path, None)
    if not dataset:
        raise SystemExit(f"缺少评测集 {dataset_path}，先跑 tests/eval_build_dataset.py")
    sample = select_generation_sample(dataset)
    answers = collect_answers(sample, dataset, reports_path, refresh=refresh)
    generation = evaluate_generation(sample, answers, dataset, threshold, refresh=refresh)
    samples = select_spotcheck(generation)
    agreement = spotcheck_agreement(samples)
    generation["spotcheck"] = samples
    generation["spotcheck_agreement"] = agreement
    SPOTCHECK_MD.write_text(render_spotcheck(samples, agreement), encoding="utf-8")
    return generation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成侧真实评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--refresh", action="store_true", help="忽略回答与 judge 缓存")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    threshold = args.threshold
    if threshold is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_retrieval import run_retrieval

        threshold = run_retrieval(args.dataset, args.reports)["threshold"]["threshold"]

    generation = run_generation(
        args.dataset,
        args.reports,
        threshold=threshold,
        refresh=args.refresh,
    )
    report = render_generation_markdown(generation)
    failures = render_failures_markdown(generation)
    Path(__file__).resolve().parent.joinpath("eval_generation_report.md").write_text(
        report, encoding="utf-8"
    )
    Path(__file__).resolve().parent.joinpath("eval_failures.md").write_text(
        failures, encoding="utf-8"
    )
    print(report)
    print(f"失败 case 已写入 {Path(__file__).resolve().parent / 'eval_failures.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
