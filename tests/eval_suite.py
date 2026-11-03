"""一条命令跑完真实数据评测：检索 80 条 + 生成 30 条 + 失败 case。

    python tests/eval_suite.py
    python -m pytest -m eval

只读 `tests/fixtures/eval_reports/` 的固定快照，不重新采集数据；
生成评测会真实调用 `/api/v1/ask` 与 judge，默认走本地缓存，`--refresh` 才重跑。
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_compare import failure_entries  # noqa: E402
from eval_generation import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_REPORTS,
    render_generation_markdown,
    run_generation,
)
from eval_retrieval import render_retrieval_markdown, run_retrieval  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "eval_report.md"
FAILURES_PATH = Path(__file__).resolve().parent / "eval_failures.md"
RETRIEVAL_REPORT_PATH = Path(__file__).resolve().parent / "eval_retrieval_report.md"
RUNS_DIR = Path(__file__).resolve().parent / "eval_runs"
HISTORY_PATH = Path(__file__).resolve().parent / "eval_history.md"
BADCASES_PATH = Path(__file__).resolve().parent / "eval_badcases.json"

HISTORY_HEADER = (
    "# 评测运行历史\n\n"
    "| 时间 | commit | mode | hit@1 | hit@3 | hit@5 | 拒答 | 相关性 |\n"
    "|---|---|---|---:|---:|---:|---:|---:|\n"
)


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_report(retrieval: dict, generation: dict) -> str:
    retrieval_summary = retrieval["summary"]
    factual = retrieval_summary.get("factual", {})
    opinion = retrieval_summary.get("opinion", {})
    threshold = retrieval["threshold"]
    generation_summary = generation["summary"]
    agreement = generation.get("spotcheck_agreement")
    failures = [result for result in generation["results"] if result["failure"]]

    lines = [
        "# 真实数据检索与生成评测",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}。"
        "所有数据来自 `tests/fixtures/eval_reports/` 的真实 A 股快照，"
        "指标是实际跑出来的；除下方披露的一次 judge rubric 纠错外，"
        "没有为了让数字好看调整 RAG prompt、检索参数或标签。",
        "",
        "## 结论摘要",
        "",
        f"- 检索 hit@1：事实型 **{_num(factual.get('hit@1'))}**，"
        f"观点型 **{_num(opinion.get('hit@1'))}**；"
        "事实型问题有唯一正确报告，观点型问题允许多份报告，且问法更抽象，"
        "两者差距反映的是任务难度，不是同一把尺子。",
        f"- 推荐相似度阈值 **{threshold.get('threshold')}**"
        f"（{threshold.get('policy')}）：无关问题误召回 "
        f"{_pct(threshold.get('false_accept'))}，可回答问题漏召回 "
        f"{_pct(threshold.get('false_reject'))}。缺失事实型问题中 "
        f"{_pct(threshold.get('missing_fact_above'))} 的 top1 高于阈值，"
        "阈值救不了这类问题。",
        f"- 生成侧：引用准确率 **{_pct(generation_summary.get('citation_accuracy'))}**，"
        f"拒答准确率 **{_pct(generation_summary.get('refusal_accuracy'))}**"
        f"（{generation_summary.get('refusal_correct')}/"
        f"{generation_summary.get('refusal_total')}），"
        f"相关性 **{_num(generation_summary.get('relevance'))}/5**，"
        f"编造 **{generation_summary.get('fabricated')}** 条。",
        f"- 生成评测共 {len(failures)}/{generation_summary.get('count')} 条失败（judge 口径），"
        "其中 C02 的引用归因已确认是 judge 漏判；明细见 `tests/eval_failures.md`。",
        "",
    ]
    if agreement:
        lines.append(
            f"- Judge 人工抽查一致率：{agreement['agree']}/{agreement['total']} = "
            f"{agreement['agreement']:.0%}（样本见 `tests/eval_judge_spotcheck.md`）。"
        )
        lines.append("")

    lines.extend(
        [
            render_retrieval_markdown(retrieval),
            render_generation_markdown(generation),
            "## 失败 case",
            "",
            f"- 生成侧失败 {len(failures)} 条，按主因分布："
            + (
                "、".join(
                    f"{reason} {count}" for reason, count in _failure_counts(failures).most_common()
                )
                if failures
                else "无"
            )
            + "。",
            "- 检索侧失败（top5 漏召回）见 `tests/eval_failures.md`。",
            "",
            "## 不调参声明",
            "",
            "- 数据集、阈值、关键词规则在跑检索指标之前冻结；开测前人工复核 evidence，"
            "修正过明显语义误标（例如把毛利率优化标成下滑），修正依据是句子语义，"
            "不是 hit@k；指标出来后没有回头改题、改期望集合或调检索参数。",
            "- 生成评测没有改 RAG prompt、没有加阈值、没有重排检索结果；"
            "阈值只做模拟，生产 `/ask` 行为保持不变。",
            "- Judge prompt 在首轮评测后修正过一次：原 rubric 把排除项的引用误判为"
            "不支撑，并漏判回答中的部分 `[n]`；修正依据是引用片段中存在明确数字，"
            "不是分数。修正后只重跑 judge，30 条回答与检索结果未变。",
            "- 回答、引用和 judge 判定均落盘缓存；复跑优先复用，避免重复花费，"
            "但这不影响首次结果的真实性。",
            "",
            "## 局限",
            "",
            "- 30 份报告、80 条问题，四个主行业各 7 家，另有家电、食品饮料各 1 家；"
            "样本量足以暴露问题，不足以代表全市场。",
            "- 检索是报告级、不切分：一份报告一个向量，长报告里的单一事实会被稀释；"
            "观点型问题相对占优，事实型问题更依赖财务表是否完整进索引。",
            "- 事实型只取 2024 年报一行；对比题限定在行业内部，阈值是人为写死的业务口径。",
            "- 观点题用关键词规则标注，可能漏掉语义相近但用词不同的句子；"
            "这是保守口径，不做主观扩展。",
            "- Judge 与生成模型同为 qwen-plus，存在自偏好风险；"
            "用 10 条人工抽查校准，样本很小，一致率只能作为方向性参考。",
            "- 已人工确认一次 judge 漏判：C02 的引用[5]片段明确含中信证券 82.54%，"
            "judge 理由也写了应判 true，最终 `supported` 仍是 false。"
            "本报告保留 judge 原始判定、不手工改分，所以引用准确率是偏保守口径，"
            "C02 的「引用不支撑」归因应视为 judge 噪声；但回答首句把招商银行列入、"
            "后文又排除，存在自相矛盾，未计入编造。",
            "- Judge 对「先写错、后文自我修正」的中间句较宽容：C06 首句称仅有贵州茅台和山西汾酒，"
            "后文修正为三家，最终结论正确；这类不一致未计入编造。",
            "- 生成侧抽样 30 条，其中无法回答 10 条（5 超纲 + 5 缺字段）；"
            "拒答准确率的置信区间很宽。",
            "- 数据源存在已知外部故障：东方财富接口不可达、akshare 新闻正则报错，"
            "报告已回退到巨潮/同花顺；快照生成时间与数据源版本见 `manifest.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def _failure_counts(failures: list[dict]):
    from collections import Counter

    return Counter(result["failure"]["primary"] for result in failures)


# ---------- 运行归档 / badcase / 历史表 ----------


def failure_rows(retrieval: dict, generation: dict | None) -> list[dict]:
    """检索与生成失败统一成 (id, failure_type) 条目，severity 自动推导。"""
    return failure_entries(retrieval, generation)


def merge_badcases(existing: dict, rows: list[dict], run_id: str) -> dict:
    """合并本次失败：保留人工维护的 status 与 first_seen，只更新 last_seen/reason。"""
    merged = dict(existing)
    for row in rows:
        key = f"{row['id']}:{row['failure_type']}"
        entry = merged.get(key)
        if entry is None:
            merged[key] = {
                **row,
                "first_seen": run_id,
                "last_seen": run_id,
                "status": "open",
            }
        else:
            entry.update(
                {
                    "category": row["category"],
                    "reason": row["reason"],
                    "severity": row["severity"],
                    "last_seen": run_id,
                }
            )
    return dict(sorted(merged.items()))


def append_history(path: Path, meta: dict, retrieval: dict, generation: dict | None) -> None:
    """每次运行 append 一行；文件不存在时先写表头。"""
    overall = retrieval["summary"]["__overall__"]
    refusal = _pct(generation["summary"]["refusal_accuracy"]) if generation else "—"
    relevance = _num(generation["summary"]["relevance"]) if generation else "—"
    stamp = meta.get("started_at") or datetime.now().isoformat(timespec="minutes")
    row = (
        f"| {stamp[:16].replace('T', ' ')} | "
        f"{(meta.get('commit') or 'unknown')[:7]} | {meta.get('mode', 'full')} | "
        f"{_num(overall.get('hit@1'))} | {_num(overall.get('hit@3'))} | "
        f"{_num(overall.get('hit@5'))} | {refusal} | {relevance} |\n"
    )
    path = Path(path)
    if path.exists():
        with path.open("a", encoding="utf-8") as handle:
            handle.write(row)
    else:
        path.write_text(HISTORY_HEADER + row, encoding="utf-8")


def write_run_dir(
    runs_dir: Path,
    meta: dict,
    retrieval: dict,
    generation: dict | None,
    report: str,
    failures: str,
) -> Path:
    run_dir = Path(runs_dir) / meta["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    (run_dir / "failures.md").write_text(failures, encoding="utf-8")
    (run_dir / "retrieval.json").write_text(
        json.dumps(retrieval, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if generation is not None:
        (run_dir / "generation.json").write_text(
            json.dumps(generation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _top1(result: dict):
    ranked = result.get("ranked") or []
    return result.get("top1") or (ranked[0] if ranked else None)


def render_retrieval_only_report(retrieval: dict) -> str:
    return "\n".join(
        [
            "# 检索评测（retrieval-only）",
            "",
            "本次只跑检索，未调用生成模型与 judge。",
            "",
            render_retrieval_markdown(retrieval),
        ]
    )


def render_failures_unified(retrieval: dict, generation: dict | None = None) -> str:
    """按 case id 聚合检索与生成失败，同一题只出现一个小节。"""
    rows = failure_rows(retrieval, generation)
    if not rows:
        return "# 评测失败 case\n\n本次没有失败 case。\n"

    retrieval_by_id = {result["id"]: result for result in retrieval.get("results", [])}
    generation_by_id = {
        result["id"]: result
        for result in (generation or {}).get("results", [])
        if result.get("failure")
    }
    names = (retrieval.get("dataset") or {}).get("report_names") or {}

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row)

    high = sum(1 for row in rows if row["severity"] == "high")
    low = len(rows) - high
    lines = [
        "# 评测失败 case",
        "",
        f"共 {len(grouped)} 个 case（high {high} / low {low}），按 case id 聚合；"
        "同一题的检索与生成失败合在同一小节，不再重复行。"
        "严重度自动推导：top5 全漏、生成失败为 high，top1 未命中但 top5 命中为 low。",
        "",
        "已人工复核：C02 的「引用不支撑」是 judge 漏判（引用[5]片段明确含中信证券 82.54%），"
        "这里保留原始判定、不手工改分。",
        "",
        "| ID | 品类 | 严重度 | 失败类型 | 失败原因 |",
        "|---|---|---|---|---|",
    ]
    for case_id, case_rows in grouped.items():
        types = "、".join(row["failure_type"] for row in case_rows)
        reasons = "；".join(row["reason"] for row in case_rows)
        lines.append(
            f"| {case_id} | {case_rows[0]['category']} | {case_rows[0]['severity']} | "
            f"{types} | {_escape_cell(reasons)} |"
        )
    lines.append("")

    for case_id, case_rows in grouped.items():
        sample = retrieval_by_id.get(case_id) or generation_by_id.get(case_id) or {}
        lines.extend(
            [
                f"## {case_id} {case_rows[0]['category']}（{case_rows[0]['severity']}）",
                "",
                f"**问题**：{sample.get('question', '')}",
                "",
                f"**失败类型**：{'、'.join(row['failure_type'] for row in case_rows)}",
                "",
            ]
        )

        retrieval_row = next(
            (row for row in case_rows if row["failure_type"].startswith("retrieval_")), None
        )
        if retrieval_row is not None:
            result = retrieval_by_id[case_id]
            expected = [names.get(identifier, identifier) for identifier in result["expected"]]
            top5 = "；".join(
                f"{names.get(identifier, identifier)}（{identifier}，{score}）"
                for identifier, score in zip(
                    result["ranked"][:5], result["scores"][:5], strict=False
                )
            )
            top1 = _top1(result)
            lines.extend(
                [
                    f"**检索侧**：{retrieval_row['reason']}",
                    "",
                    f"- 期望报告：{'、'.join(expected)}",
                    f"- top1：{names.get(top1, top1)}（{result['scores'][0] if result['scores'] else '—'}）",
                    f"- top5：{top5}",
                    f"- 首个命中排名：{result.get('first_hit') or '无'}",
                    "",
                ]
            )

        generation_row = next(
            (row for row in case_rows if row["failure_type"] == "generation_failure"), None
        )
        if generation_row is not None:
            result = generation_by_id[case_id]
            metrics = result["citation_metrics"]
            lines.extend(
                [
                    f"**期望**：{result.get('reference_answer')}",
                    "",
                    f"**实际回答**：{result.get('answer')}",
                    "",
                    "**检索 top5**："
                    + "；".join(
                        f"[{index}] {hit.get('name')}（{hit.get('symbol')}，{hit.get('score')}）"
                        for index, hit in enumerate(result.get("citations") or [], 1)
                    ),
                    "",
                    f"**输出引用**：{metrics['indices'] or '无'}；"
                    f"judge 支持 {metrics['supported_indices'] or '无'}；"
                    f"越界 {metrics['invalid_indices'] or '无'}",
                    "",
                    f"**Judge**：relevance={result['judge']['relevance']}，"
                    f"refused={result['judge']['refused']}，"
                    f"fabricated={result['judge']['fabricated']}；"
                    f"{result['judge']['notes']}",
                    "",
                ]
            )

        lines.extend(
            [
                f"**失败原因**：{'；'.join(row['reason'] for row in case_rows)}",
                "",
            ]
        )
    return "\n".join(lines)


def render_retrieval_failures(retrieval: dict) -> str:
    """兼容旧调用：只渲染检索侧失败，输出与统一渲染器一致。"""
    return render_failures_unified(retrieval)


def run_suite(
    dataset_path: Path = DEFAULT_DATASET,
    reports_path: Path = DEFAULT_REPORTS,
    refresh: bool = False,
    mode: str = "full",
    runs_dir: Path = RUNS_DIR,
    history_path: Path = HISTORY_PATH,
    badcases_path: Path = BADCASES_PATH,
) -> dict:
    if mode not in ("full", "retrieval"):
        raise ValueError(f"未知模式：{mode}")

    started = datetime.now()
    started_monotonic = time.monotonic()
    retrieval = run_retrieval(dataset_path, reports_path)
    generation = None
    if mode == "full":
        threshold = retrieval["threshold"]["threshold"]
        generation = run_generation(
            dataset_path,
            reports_path,
            threshold=threshold,
            refresh=refresh,
        )
    duration = time.monotonic() - started_monotonic

    report = (
        render_report(retrieval, generation)
        if generation is not None
        else render_retrieval_only_report(retrieval)
    )
    failures = render_failures_unified(retrieval, generation)

    if generation is not None:
        REPORT_PATH.write_text(report, encoding="utf-8")
        FAILURES_PATH.write_text(failures, encoding="utf-8")
    RETRIEVAL_REPORT_PATH.write_text(render_retrieval_markdown(retrieval), encoding="utf-8")

    run_id = started.strftime("%Y%m%d-%H%M%S")
    suffix = 1
    while (Path(runs_dir) / run_id).exists():
        run_id = f"{started.strftime('%Y%m%d-%H%M%S')}-{suffix}"
        suffix += 1

    badcases_path = Path(badcases_path)
    existing = (
        json.loads(badcases_path.read_text(encoding="utf-8")) if badcases_path.exists() else {}
    )
    meta = {
        "run_id": run_id,
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "started_at": started.isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 1),
        "refresh": bool(refresh),
        "mode": mode,
        "badcases_before": len(existing),
    }
    merged = merge_badcases(existing, failure_rows(retrieval, generation), run_id)
    badcases_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_dir = write_run_dir(runs_dir, meta, retrieval, generation, report, failures)
    append_history(history_path, meta, retrieval, generation)
    return {
        "retrieval": retrieval,
        "generation": generation,
        "run_dir": run_dir,
        "meta": meta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实数据检索与生成评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--refresh", action="store_true", help="忽略回答与 judge 缓存")
    parser.add_argument(
        "--mode",
        choices=("full", "retrieval"),
        default="full",
        help="retrieval 只跑检索，不调用生成模型与 judge",
    )
    parser.add_argument("--compare", type=Path, default=None, help="跑完与指定 run 目录对比")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result = run_suite(args.dataset, args.reports, refresh=args.refresh, mode=args.mode)
    print((result["run_dir"] / "report.md").read_text(encoding="utf-8"))
    print(f"运行归档已写入 {result['run_dir']}")
    print(f"失败 case 已写入 {result['run_dir'] / 'failures.md'}")
    print(f"检索评测已写入 {RETRIEVAL_REPORT_PATH}")
    generation = result["generation"]
    print(
        f"检索题数={len(result['retrieval']['results'])}，"
        f"生成题数={generation['summary']['count'] if generation else 0}"
    )
    if args.compare:
        from eval_compare import compare_runs

        print(compare_runs(args.compare, result["run_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
