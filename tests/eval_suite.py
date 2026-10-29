"""一条命令跑完真实数据评测：检索 80 条 + 生成 30 条 + 失败 case。

    python tests/eval_suite.py
    python -m pytest -m eval

只读 `tests/fixtures/eval_reports/` 的固定快照，不重新采集数据；
生成评测会真实调用 `/api/v1/ask` 与 judge，默认走本地缓存，`--refresh` 才重跑。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_generation import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_REPORTS,
    render_failures_markdown,
    render_generation_markdown,
    run_generation,
)
from eval_retrieval import render_retrieval_markdown, run_retrieval  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "eval_report.md"
FAILURES_PATH = Path(__file__).resolve().parent / "eval_failures.md"
RETRIEVAL_REPORT_PATH = Path(__file__).resolve().parent / "eval_retrieval_report.md"


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _retrieval_top1(result: dict) -> str | None:
    ranked = result.get("ranked") or []
    return result.get("top1") or (ranked[0] if ranked else None)


def _escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_retrieval_failures(retrieval: dict) -> str:
    """列出两类检索失败：top5 全漏，以及 top1 未命中但 top5 命中。

    多答案题的期望集合可能大于 5，top5 不可能全覆盖；那属于 recall@5 的
    统计范围，不是「检索没召回」。top1 未命中的题单独列出，用来解释
    hit@1/MRR 为什么低，同时说明 `/ask` 仍可能拿到正确上下文。
    """
    full_misses = []
    top1_misses = []
    for result in retrieval["results"]:
        if not result["answerable"] or not result["expected"]:
            continue
        expected = set(result["expected"])
        top5 = set(result["ranked"][:5])
        if not top5 & expected:
            full_misses.append(result)
        elif _retrieval_top1(result) not in expected:
            top1_misses.append(result)

    names = retrieval["dataset"].get("report_names") or {}
    lines = [
        "# 检索侧失败 case",
        "",
        f"共 {len(full_misses)} 条题目 top5 完全没有命中期望报告，"
        f"{len(top1_misses)} 条 top1 未命中但 top5 内有期望报告。"
        "多答案题的部分漏召回看检索报告里的 recall@5，不在这里重复列。",
        "",
    ]
    if not full_misses and not top1_misses:
        lines.append("本次 80 条检索评测没有 top5 全漏或 top1 未命中的题目。")
        lines.append("")
        return "\n".join(lines)

    if full_misses:
        lines.extend(
            [
                "## top5 全漏",
                "",
                "| ID | 品类 | 问题 | 期望报告（前 5） | top5 |",
                "|---|---|---|---|---|",
            ]
        )
        for result in full_misses:
            top5 = "、".join(
                f"{names.get(identifier, identifier)}（{identifier}，{score}）"
                for identifier, score in zip(
                    result["ranked"][:5], result["scores"][:5], strict=False
                )
            )
            expected_names = [
                names.get(identifier, identifier) for identifier in result["expected"]
            ]
            expected = "、".join(expected_names[:5])
            if len(expected_names) > 5:
                expected += f" 等 {len(expected_names)} 份"
            lines.append(
                f"| {result['id']} | {result['category']} | "
                f"{_escape_cell(result['question'][:60])} | {expected} | {top5} |"
            )
        lines.extend(
            [
                "",
                "原因：报告级向量把整份报告压成一个点，长报告里的单一事实容易被其他主题稀释；"
                "这不是调阈值能解决的，需要切分或混合检索。",
                "",
            ]
        )

    if top1_misses:
        lines.extend(
            [
                "## top1 未命中但 top5 命中",
                "",
                "| ID | 品类 | 问题 | top1（分数） | 首个命中排名 | 期望报告数 |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for result in top1_misses:
            top1 = _retrieval_top1(result)
            top1_score = result["scores"][0] if result.get("scores") else None
            first_hit = result.get("first_hit")
            if first_hit is None:
                first_hit = next(
                    (
                        rank
                        for rank, identifier in enumerate(result["ranked"], 1)
                        if identifier in set(result["expected"])
                    ),
                    None,
                )
            lines.append(
                f"| {result['id']} | {result['category']} | "
                f"{_escape_cell(result['question'][:60])} | "
                f"{names.get(top1, top1)}（{top1_score}） | {first_hit} | "
                f"{len(result['expected'])} |"
            )
        lines.extend(
            [
                "",
                "原因：同一行业、财务结构相近的报告向量距离很近，top1 被其中一份"
                "未命中期望集合的报告占住；`/ask` 仍能从 top5 拿到正确报告，"
                "但 hit@1/MRR 会下降。要改善得靠 rerank、按公司/指标拆分子块"
                "或混合检索，不是调阈值。",
                "",
            ]
        )
    return "\n".join(lines)


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


def run_suite(
    dataset_path: Path = DEFAULT_DATASET,
    reports_path: Path = DEFAULT_REPORTS,
    refresh: bool = False,
) -> dict:
    retrieval = run_retrieval(dataset_path, reports_path)
    threshold = retrieval["threshold"]["threshold"]
    generation = run_generation(
        dataset_path,
        reports_path,
        threshold=threshold,
        refresh=refresh,
    )
    report = render_report(retrieval, generation)
    failures = render_failures_markdown(generation) + "\n" + render_retrieval_failures(retrieval)
    REPORT_PATH.write_text(report, encoding="utf-8")
    FAILURES_PATH.write_text(failures, encoding="utf-8")
    RETRIEVAL_REPORT_PATH.write_text(render_retrieval_markdown(retrieval), encoding="utf-8")
    return {"retrieval": retrieval, "generation": generation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实数据检索与生成评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--refresh", action="store_true", help="忽略回答与 judge 缓存")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result = run_suite(args.dataset, args.reports, refresh=args.refresh)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"失败 case 已写入 {FAILURES_PATH}")
    print(f"检索评测已写入 {RETRIEVAL_REPORT_PATH}")
    print(
        f"检索题数={len(result['retrieval']['results'])}，"
        f"生成题数={result['generation']['summary']['count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
