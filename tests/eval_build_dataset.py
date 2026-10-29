"""从真实报告快照构造 80 条可人工核对的评测问题。

原则：

- 数值题只取报告里的「近三年关键财务」表，答案就是表里的原始字符串；
- 对比题的阈值写在问题里，期望集合由快照逐家筛选，不靠主观感觉；
- 观点题的期望集合由固定的文本匹配规则产生，命中句写进 evidence，方便复核；
- 无法回答题分两类：完全超出 30 份报告覆盖范围（out_of_scope），
  以及报告里确实没有这个字段（missing_fact）。后者检索会命中正确公司，
  但生成阶段必须拒答，不能算成「检索失败」。

标注完成后才能跑检索评测；如果先看指标再改标签，评测就失去意义。
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_REPORTS = FIXTURE_DIR / "eval_reports" / "reports.json"
DEFAULT_MANIFEST = FIXTURE_DIR / "eval_reports" / "manifest.json"
DEFAULT_OUTPUT = FIXTURE_DIR / "eval_dataset.json"
DEFAULT_LABELS = Path(__file__).resolve().parent / "eval_labels.md"

INDUSTRIES = ("金融", "白酒", "新能源", "医药")

# 只有这些字段会进 `_build_doc`，观点题必须在索引文本里能查到证据，
# 否则标注依据和 RAG 实际能看到的内容不一致。
DOC_FIELDS = ("基本面总结", "综合结论", "风险点", "舆情驱动", "多空仲裁")

FACTUAL_METRICS = (
    "净资产收益率",
    "销售毛利率",
    "销售净利率",
    "资产负债率",
    "营业总收入同比增长率",
    "净利润同比增长率",
    "每股经营现金流",
    "流动比率",
    "营业总收入",
    "净利润",
)

# 对比题只用比率/增速，绝对金额的阈值不直观；每个指标给一组业务上说得通的阈值。
COMPARISON_METRICS = (
    ("净资产收益率", "%", (5, 10, 15, 20, 25, 30)),
    ("销售毛利率", "%", (20, 30, 40, 50, 60, 70, 80)),
    ("销售净利率", "%", (5, 10, 15, 20, 30, 40)),
    ("资产负债率", "%", (20, 30, 40, 50, 60, 70, 80, 90)),
    ("营业总收入同比增长率", "%", (-10, 0, 5, 10, 20, 30)),
    ("净利润同比增长率", "%", (-10, 0, 5, 10, 20, 30)),
    ("流动比率", "", (0.8, 1.0, 1.5, 2.0, 3.0, 4.0)),
    ("每股经营现金流", "元", (0, 1, 2, 3, 5, 10)),
)

OPINION_PATTERNS = (
    {
        "key": "cashflow_risk",
        "question": "哪些报告提示了现金流风险？",
        "fields": ("风险点",),
        "required": ("现金流",),
        "context_any": (
            "风险",
            "压力",
            "紧张",
            "下滑",
            "恶化",
            "承压",
            "流出",
            "依赖",
            "波动",
            "覆盖",
            "弱化",
            "断裂",
            "骤降",
            "下降",
            "减少",
            "为负",
            "转负",
            "偏低",
            "仅",
            "难以覆盖",
            "缺口",
            "覆盖不足",
            "回落",
        ),
    },
    {
        "key": "policy_risk",
        "question": "哪些报告提示了政策或监管风险？",
        "fields": ("风险点",),
        "match_any": ("政策", "监管"),
        "context_any": ("风险", "约束", "趋严", "压力", "挑战", "超预期", "不确定性", "收紧"),
    },
    {
        "key": "competition",
        "question": "哪些报告提示了行业竞争加剧？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("竞争",),
        "context_any": ("加剧", "激烈", "压力", "内卷"),
    },
    {
        "key": "demand_slowdown",
        "question": "哪些报告提示了需求放缓？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("需求",),
        "context_any": ("放缓", "疲软", "不足", "下行", "承压"),
    },
    {
        "key": "inventory_pressure",
        "question": "哪些报告提示了库存或存货压力？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("库存", "存货"),
        "context_any": ("压力", "风险", "高", "积压", "去化", "减值"),
        "allow_following_context": False,
    },
    {
        "key": "margin_decline",
        "question": "哪些报告提示了毛利率下滑？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        # 只认毛利率自己后面跟的下降语义；不把同一小句里费用率、收入或资产负债率的下降算进来。
        "regex": r"毛利率(?:[、与][^，。；]{0,20})?[^，。；]{0,4}(?:下滑|下降|回落|承压|收窄|走弱)|毛利率回调压力",
    },
    {
        "key": "governance_risk",
        "question": "哪些报告提示了管理层或治理风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("管理层", "治理"),
        "context_any": ("风险", "更迭", "过渡", "不稳定", "变动", "缺位", "落差", "滞后"),
    },
    {
        "key": "sentiment_negative",
        "question": "哪些报告对舆情给出负面判断？",
        "special": "sentiment_negative",
        "fields": (),
    },
    {
        "key": "technical_strong",
        "question": "哪些报告认为技术面偏强？",
        "fields": ("技术面",),
        "match_any": ("多头", "偏强", "上涨", "走强", "突破", "转强"),
    },
    {
        "key": "technical_weak",
        "question": "哪些报告认为技术面偏弱？",
        "fields": ("技术面",),
        "match_any": ("空头", "偏弱", "下跌", "走弱", "承压", "回调", "跌破"),
    },
    {
        "key": "dividend",
        "question": "哪些报告提到了分红或股东回报？",
        "fields": (
            "基本面总结",
            "综合结论",
            "风险点",
            "舆情驱动",
            "多空仲裁",
            "多头观点",
            "空头观点",
        ),
        "match_any": ("分红", "股东回报", "派息", "现金分红"),
    },
    {
        "key": "overseas_risk",
        "question": "哪些报告提示了海外或国际化风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("海外",),
        "context_any": ("风险", "不确定", "波动", "承压", "贸易", "汇率"),
    },
    {
        "key": "raw_material",
        "question": "哪些报告提到了原材料价格波动？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("原材料",),
        "context_any": ("价格", "成本", "波动", "上涨", "下行"),
    },
    {
        "key": "rd_risk",
        "question": "哪些报告提示了研发或管线风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("研发",),
        "context_any": ("风险", "投入", "失败", "不及预期", "管线", "费用"),
    },
    {
        "key": "customer_concentration",
        "question": "哪些报告提到了客户集中度风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("客户集中", "大客户", "客户依赖", "前五大客户"),
    },
    {
        "key": "fx_risk",
        "question": "哪些报告提示了汇率风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("汇率",),
        "context_any": ("风险", "波动", "贬值", "升值", "敞口"),
    },
    {
        "key": "safety_environment",
        "question": "哪些报告提示了安全生产或环保风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("安全生产", "环保", "安全事故", "环保风险"),
    },
    {
        "key": "goodwill",
        "question": "哪些报告提示了商誉减值风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("商誉",),
        "context_any": ("减值", "风险"),
    },
    {
        "key": "capacity",
        "question": "哪些报告提到了产能扩张或产能过剩？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("产能",),
        "context_any": ("过剩", "扩张", "投放", "利用率", "消化"),
    },
    {
        "key": "procurement",
        "question": "哪些报告提到了集采或医保政策？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("集采", "医保"),
    },
    {
        "key": "channel_inventory",
        "question": "哪些报告提到了渠道库存？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("渠道",),
        "context_any": ("库存", "压货", "积压"),
    },
    {
        "key": "valuation_high",
        "question": "哪些报告提示了估值偏高？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("估值",),
        "context_any": ("偏高", "高企", "泡沫", "回调压力", "处于高位", "高位", "透支"),
    },
    {
        "key": "cashflow_healthy",
        "question": "哪些报告认为现金流健康？",
        "fields": ("基本面总结", "综合结论"),
        "required": ("现金流",),
        "context_any": ("健康", "强劲", "优异", "充裕", "稳健"),
    },
    {
        "key": "brand_moat",
        "question": "哪些报告提到了品牌力或护城河？",
        "fields": ("基本面总结", "综合结论", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("品牌力", "护城河", "品牌壁垒"),
    },
    {
        "key": "price_war",
        "question": "哪些报告提到了价格战或降价压力？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("价格战", "降价", "内卷"),
    },
    {
        "key": "shareholder_risk",
        "question": "哪些报告提示了股东减持或质押风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("股东",),
        "match_any": ("减持", "质押"),
        "context_any": ("风险", "压力", "平仓", "关注", "紧张", "触发", "预警", "违约", "承压"),
    },
    {
        "key": "debt_risk",
        "question": "哪些报告提示了债务或偿债风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "required": ("债务",),
        "context_any": ("风险", "压力", "到期", "偿债", "难以覆盖"),
    },
    {
        "key": "project_progress",
        "question": "哪些报告提到了项目投产或建设进度？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("投产", "在建工程", "项目进度"),
    },
    {
        "key": "quality_safety",
        "question": "哪些报告提示了产品质量或食品安全风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("食品安全", "产品质量", "质量风险"),
    },
    {
        "key": "tax_policy",
        "question": "哪些报告提到了税收政策或税率变化？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁", "多头观点", "空头观点"),
        "match_any": ("税收优惠", "税率", "税收政策"),
    },
    {
        "key": "receivable_risk",
        "question": "哪些报告提示了应收账款或回款风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁"),
        "required": ("应收账款",),
        "context_any": ("风险", "回款", "账期", "减值", "压力"),
    },
    {
        "key": "inventory_write_down",
        "question": "哪些报告提示了存货跌价或减值风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁"),
        "required": ("存货",),
        "context_any": ("跌价", "减值", "积压", "压力"),
    },
    {
        "key": "interest_rate",
        "question": "哪些报告提示了利率或息差风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁"),
        # 负向后顾排除「净利率」「毛利率」，否则盈利指标会被误当成利率风险。
        "regex": r"(?<!净)(?<!毛)利率[^，。；]{0,6}(下行|上行|波动|风险|敞口|上升)"
        r"|息差[^，。；]{0,6}(收窄|下行|压力|风险|承压)",
    },
    {
        "key": "asset_quality",
        "question": "哪些报告提到了资产质量或不良贷款？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁"),
        "match_any": ("资产质量", "不良贷款", "不良率", "拨备"),
    },
    {
        "key": "litigation",
        "question": "哪些报告提示了诉讼、仲裁或处罚风险？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁"),
        "match_any": ("诉讼", "仲裁", "处罚", "合规"),
    },
    {
        "key": "financing",
        "question": "哪些报告提到了融资或再融资压力？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁"),
        "match_any": ("融资", "定增", "可转债", "发债"),
        "context_any": ("风险", "压力", "收紧", "失败", "成本"),
    },
    {
        "key": "consumption",
        "question": "哪些报告提示了消费疲软或需求不足？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁"),
        "required": ("消费",),
        "context_any": ("疲软", "复苏", "需求", "承压", "不足"),
    },
    {
        "key": "related_party",
        "question": "哪些报告提到了关联交易或大股东占用？",
        "fields": ("基本面总结", "综合结论", "风险点", "多空仲裁"),
        "match_any": ("关联交易", "大股东占用", "资金占用"),
    },
)

OUT_OF_SCOPE_QUESTIONS = (
    ("美联储 2026 年 9 月降息了多少个基点？", "海外货币政策"),
    ("伦敦金现最新价格是多少美元每盎司？", "国际黄金现货价格"),
    ("港股通本周净流入多少亿港元？", "港股通资金流"),
    ("2026 年生猪现货价格是多少元每公斤？", "生猪现货价格"),
    ("华安张江 REIT 的现金分派率是多少？", "公募 REITs 分派率"),
    ("A 股证券交易印花税税率是多少？", "证券交易税费"),
    ("人民币兑美元中间价今天是多少？", "外汇中间价"),
    ("北向资金今日净买入多少亿元？", "北向资金"),
    ("美国 2026 年 8 月 CPI 同比涨幅是多少？", "美国通胀数据"),
    ("同仁堂 2024 年的净资产收益率是多少？", "未收录公司"),
)

MISSING_FACT_TEMPLATES = (
    ("员工人数", "员工人数"),
    ("研发投入金额", "研发投入"),
    ("经销商数量", "经销商"),
    ("前五大客户销售占比", "前五大客户"),
    ("门店数量", "门店"),
    ("产能利用率", "产能利用率"),
    ("境外收入占比", "境外收入"),
    ("高管薪酬", "高管薪酬"),
    ("回购金额", "回购"),
    ("专利数量", "专利"),
)


# ---------- 财务表解析 ----------


def parse_history(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in (text or "").splitlines() if "|" in line]
    if not lines:
        return []
    header = [cell.strip() for cell in lines[0].split("|")]
    rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def metric_value(row: dict[str, str], metric: str) -> float | None:
    raw = (row.get(metric) or "").strip()
    if not raw or raw in {"—", "-", "--", "不适用", "N/A"}:
        return None
    text = raw.replace(",", "").strip()
    for suffix in ("%", "亿", "万", "元"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    try:
        return float(text)
    except ValueError:
        return None


def annual_row(report: dict, period: str = "2024-12-31") -> dict[str, str] | None:
    history = (report.get("full_report") or {}).get("financial_history") or ""
    return next((row for row in parse_history(history) if row.get("报告期") == period), None)


# ---------- 观点题文本匹配 ----------


def _split_sentences(text: str | None) -> list[str]:
    return [part.strip() for part in re.split(r"[。；\n]", text or "") if part.strip()]


def _split_clauses(sentence: str) -> list[str]:
    """按逗号切分句内小句，用来判断关键词和风险语境是否真的在说同一件事。"""
    return [part.strip() for part in re.split(r"[，,]", sentence) if part.strip()]


# 风险词经常出现在结果小句里，例如「现金流为负，存在流动性压力」。
# 只允许紧邻的下一小句以这些因果/判断词开头时补充语境，
# 避免「毛利率优化，资产负债率下降」被误判成毛利率下滑。
_CONSEQUENCE_MARKERS = (
    "显示",
    "反映",
    "表明",
    "说明",
    "导致",
    "引发",
    "加剧",
    "带来",
    "意味着",
    "存在",
    "形成",
    "触发",
    "暴露",
    "削弱",
    "制约",
    "承压",
    "提示",
    "暗示",
    "指向",
)


def _context_hit(
    clauses: list[str],
    index: int,
    context_any: tuple[str, ...],
    allow_following: bool = True,
) -> bool:
    if any(term in clauses[index] for term in context_any):
        return True
    if allow_following and index + 1 < len(clauses):
        following = clauses[index + 1]
        if following.startswith(_CONSEQUENCE_MARKERS) and any(
            term in following for term in context_any
        ):
            return True
    return False


def narrative_sentences(report: dict) -> list[tuple[str, str]]:
    """把报告里可引用的叙述拆成 (字段, 句子)，字段名用于限定匹配范围。"""
    full = report.get("full_report") or {}
    items: list[tuple[str, str]] = []

    def add(label: str, text: str | None) -> None:
        for sentence in _split_sentences(text):
            items.append((label, sentence))

    add("基本面总结", report.get("fundamental_summary"))
    add("综合结论", report.get("conclusion"))

    for risk in full.get("risks") or []:
        category = risk.get("category") or "未分类"
        add(f"风险点·{category}", risk.get("detail"))

    sentiment = full.get("sentiment") or {}
    for driver in sentiment.get("key_drivers") or []:
        add("舆情驱动", driver)

    add("多空仲裁", full.get("debate_verdict"))
    for key, label in (("bull_thesis", "多头观点"), ("bear_thesis", "空头观点")):
        thesis = full.get(key) or {}
        add(label, thesis.get("viewpoint"))
        for evidence in thesis.get("key_evidence") or []:
            add(label, evidence)

    technical = full.get("technical") or {}
    add("技术面", technical.get("trend_summary"))
    return [(label, sentence) for label, sentence in items if label.startswith(DOC_FIELDS)]


def opinion_matches(report: dict, pattern: dict) -> list[tuple[str, str]]:
    """返回命中 (字段, 句子)；同一报告只取第一条，避免重复计分。"""
    if pattern.get("special") == "sentiment_negative":
        sentiment = (report.get("full_report") or {}).get("sentiment") or {}
        if sentiment.get("overall") == "negative" or (sentiment.get("score") or 0) < 0:
            return [
                (
                    "舆情",
                    f"overall={sentiment.get('overall')}, score={sentiment.get('score')}",
                )
            ]
        return []

    fields = tuple(pattern.get("fields") or ())
    required = tuple(pattern.get("required") or ())
    match_any = tuple(pattern.get("match_any") or ())
    context_any = tuple(pattern.get("context_any") or ())
    regex = pattern.get("regex")
    allow_following = bool(pattern.get("allow_following_context", True))
    if not fields or not (required or match_any or regex):
        return []

    matches = []
    for label, sentence in narrative_sentences(report):
        if fields and not any(label.startswith(prefix) for prefix in fields):
            continue

        clauses = _split_clauses(sentence)
        hit = False
        for index, clause in enumerate(clauses):
            if regex:
                if not re.search(regex, clause):
                    continue
            else:
                if required and not all(term in clause for term in required):
                    continue
                if match_any and not any(term in clause for term in match_any):
                    continue
            if context_any and not _context_hit(
                clauses, index, context_any, allow_following=allow_following
            ):
                continue
            hit = True
            break
        if hit:
            matches.append((label, sentence))
    return matches


# ---------- 四类问题 ----------


def _report_meta(report: dict, manifest_by_symbol: dict[str, dict]) -> dict:
    item = manifest_by_symbol.get(report["symbol"], {})
    # 同花顺偶尔返回「五 粮 液」这种带空格的简称，问题里读起来很别扭。
    company = re.sub(r"\s+", "", report.get("name") or report["symbol"])
    return {
        "symbol": report["symbol"],
        "company": company,
        "industry": item.get("category") or "其他",
    }


def _factual_question(report: dict, meta: dict, metric: str, row: dict[str, str]) -> dict:
    value = row[metric]
    report_id = report["id"]
    return {
        "id": "",  # 由调用方统一编号
        "category": "factual",
        "answerable": True,
        "unanswerable_type": None,
        "symbol": meta["symbol"],
        "company": meta["company"],
        "industry": meta["industry"],
        "question": f"{meta['company']} 2024 年的{metric}是多少？",
        "expected_hits": [report_id],
        "reference_answer": f"2024-12-31 的{metric}为 {value}。",
        "rationale": "直接取自报告「近三年关键财务」表的 2024-12-31 行，数值与报告原文一致。",
        "evidence": {report_id: f"近三年财务表 2024-12-31 行：{metric} = {value}"},
        "period": "2024-12-31",
        "metric": metric,
        "expected_value": value,
    }


def build_factual_queries(reports: list[dict], manifest_by_symbol: dict[str, dict]) -> list[dict]:
    """每个行业取 5 家，指标轮换；同一家公司只出一题。"""
    by_industry: dict[str, list[dict]] = {industry: [] for industry in INDUSTRIES}
    others = []
    for report in reports:
        meta = _report_meta(report, manifest_by_symbol)
        if meta["industry"] in by_industry:
            by_industry[meta["industry"]].append(report)
        else:
            others.append(report)

    selected: list[tuple[dict, dict]] = []
    used: set[str] = set()
    for industry in INDUSTRIES:
        group = by_industry[industry]
        for report in group:
            if len([item for item in selected if item[1]["industry"] == industry]) >= 5:
                break
            if report["symbol"] in used:
                continue
            selected.append((report, _report_meta(report, manifest_by_symbol)))
            used.add(report["symbol"])
    for report in others + [item for group in by_industry.values() for item in group]:
        if len(selected) >= 20:
            break
        if report["symbol"] in used:
            continue
        selected.append((report, _report_meta(report, manifest_by_symbol)))
        used.add(report["symbol"])

    queries = []
    for index, (report, meta) in enumerate(selected):
        row = annual_row(report)
        if not row:
            continue
        offset = index % len(FACTUAL_METRICS)
        metric = next(
            (
                FACTUAL_METRICS[(offset + step) % len(FACTUAL_METRICS)]
                for step in range(len(FACTUAL_METRICS))
                if metric_value(row, FACTUAL_METRICS[(offset + step) % len(FACTUAL_METRICS)])
                is not None
            ),
            None,
        )
        if not metric:
            continue
        query = _factual_question(report, meta, metric, row)
        query["id"] = f"F{len(queries) + 1:02d}"
        queries.append(query)

    if len(queries) < 20:
        raise RuntimeError(f"事实型只构造出 {len(queries)} 条，检查报告是否有 2024-12-31 行")
    return queries[:20]


def _fmt_threshold(value: float, unit: str) -> str:
    if unit == "":
        return f"{value:g}"
    return f"{value:g}{unit}"


def build_comparison_queries(
    reports: list[dict], manifest_by_symbol: dict[str, dict]
) -> list[dict]:
    """每个行业 5 题；阈值写进问题，期望集合由快照逐家筛选。"""
    by_industry: dict[str, list[dict]] = {industry: [] for industry in INDUSTRIES}
    for report in reports:
        meta = _report_meta(report, manifest_by_symbol)
        if meta["industry"] in by_industry:
            by_industry[meta["industry"]].append(report)

    queries: list[dict] = []
    for industry in INDUSTRIES:
        group = by_industry[industry]
        if len(group) < 3:
            raise RuntimeError(f"{industry}行业只有 {len(group)} 家，无法构造对比题")

        candidates = []
        seen_expected: set[tuple[str, ...]] = set()
        for metric, unit, thresholds in COMPARISON_METRICS:
            for operator in (">", "<"):
                for threshold in thresholds:
                    matched = []
                    for report in group:
                        row = annual_row(report)
                        value = metric_value(row, metric) if row else None
                        if value is None:
                            continue
                        if (operator == ">" and value > threshold) or (
                            operator == "<" and value < threshold
                        ):
                            matched.append((report, row[metric], value))
                    if not (2 <= len(matched) <= 5):
                        continue
                    expected = tuple(sorted(report["id"] for report, _, _ in matched))
                    if expected in seen_expected:
                        continue
                    seen_expected.add(expected)
                    candidates.append(
                        {
                            "metric": metric,
                            "unit": unit,
                            "operator": operator,
                            "threshold": threshold,
                            "matched": matched,
                        }
                    )

        candidates.sort(
            key=lambda item: (
                abs(len(item["matched"]) - 3),
                [name for name, _, _ in COMPARISON_METRICS].index(item["metric"]),
                item["operator"] == "<",
                item["threshold"],
            )
        )
        selected: list[dict] = []
        metric_counts: Counter[str] = Counter()
        for candidate in candidates:
            if metric_counts[candidate["metric"]] >= 2:
                continue
            selected.append(candidate)
            metric_counts[candidate["metric"]] += 1
            if len(selected) == 5:
                break
        if len(selected) < 5:
            for candidate in candidates:
                if candidate in selected:
                    continue
                selected.append(candidate)
                if len(selected) == 5:
                    break

        if len(selected) < 5:
            raise RuntimeError(f"{industry}行业只构造出 {len(selected)} 道对比题")

        for candidate in selected:
            metric = candidate["metric"]
            unit = candidate["unit"]
            operator = candidate["operator"]
            threshold = candidate["threshold"]
            op_text = "高于" if operator == ">" else "低于"
            threshold_text = _fmt_threshold(threshold, unit)
            matched = candidate["matched"]
            expected = [report["id"] for report, _, _ in matched]
            detail = "、".join(
                f"{report.get('name') or report['symbol']}（{raw}）" for report, raw, _ in matched
            )
            evidence = {
                report["id"]: f"近三年财务表 2024-12-31 行：{metric} = {raw}"
                for report, raw, _ in matched
            }
            queries.append(
                {
                    "id": "",
                    "category": "comparison",
                    "answerable": True,
                    "unanswerable_type": None,
                    "symbol": None,
                    "company": None,
                    "industry": industry,
                    "question": (
                        f"{industry}行业 2024 年{metric}{op_text}{threshold_text}的公司有哪些？"
                    ),
                    "expected_hits": expected,
                    "reference_answer": (
                        f"{industry}行业 2024 年{metric}{op_text}{threshold_text}的公司：{detail}。"
                    ),
                    "rationale": (
                        f"限定在{industry}行业，阈值 {threshold_text} 直接写在问题里；"
                        f"期望集合由快照 2024-12-31 行逐家筛选，共 {len(expected)} 家。"
                    ),
                    "evidence": evidence,
                    "period": "2024-12-31",
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "unit": unit,
                }
            )

    for index, query in enumerate(queries, 1):
        query["id"] = f"C{index:02d}"
    if len(queries) != 20:
        raise RuntimeError(f"对比型应为 20 条，实际 {len(queries)} 条")
    return queries


def build_opinion_queries(reports: list[dict], manifest_by_symbol: dict[str, dict]) -> list[dict]:
    """按固定顺序取前 20 个至少命中 1 份报告的规则，规则不参考检索指标。

    观点题天然可能是多答案（例如「哪些报告提示了现金流风险」），
    因此不设 5 份的上限；recall@5 会因此偏低，但那是真实结果。
    """
    selected = []
    for pattern in OPINION_PATTERNS:
        matched = []
        for report in reports:
            hits = opinion_matches(report, pattern)
            if hits:
                matched.append((report, hits[0]))
        if matched:
            selected.append((pattern, matched))
        if len(selected) == 20:
            break

    if len(selected) < 20:
        yields = [
            (pattern["key"], sum(1 for report in reports if opinion_matches(report, pattern)))
            for pattern in OPINION_PATTERNS
        ]
        raise RuntimeError(f"观点型只构造出 {len(selected)} 条，各规则命中数：{yields}")

    queries = []
    for index, (pattern, matched) in enumerate(selected, 1):
        expected = [report["id"] for report, _ in matched]
        evidence = {report["id"]: f"{label}：{sentence}" for report, (label, sentence) in matched}
        detail = "；".join(
            f"{report.get('name') or report['symbol']}：{sentence[:80]}"
            for report, (_, sentence) in matched
        )
        conditions = []
        if pattern.get("required"):
            conditions.append(f"同时出现 {'、'.join(pattern['required'])}")
        if pattern.get("match_any"):
            conditions.append(f"出现 {'/'.join(pattern['match_any'])} 之一")
        if pattern.get("regex"):
            conditions.append(f"匹配正则 {pattern['regex']}")
        if pattern.get("context_any"):
            conditions.append(f"上下文出现 {'/'.join(pattern['context_any'])} 之一")
        if pattern.get("allow_following_context") is False:
            conditions.append("语境词必须与关键词在同一小句")
        if pattern.get("special"):
            conditions.append("结构化舆情字段 overall=negative 或 score<0")
        queries.append(
            {
                "id": f"O{index:02d}",
                "category": "opinion",
                "answerable": True,
                "unanswerable_type": None,
                "symbol": None,
                "company": None,
                "industry": None,
                "question": pattern["question"],
                "expected_hits": expected,
                "reference_answer": f"相关报告：{detail}。",
                "rationale": (
                    f"在{('、'.join(pattern.get('fields') or ('结构化字段',)))}中，"
                    f"{'，'.join(conditions)}；命中句见 evidence，共 {len(expected)} 份。"
                ),
                "evidence": evidence,
                "pattern": pattern["key"],
            }
        )
    return queries


def build_unanswerable_queries(
    reports: list[dict],
    manifest_by_symbol: dict[str, dict],
) -> list[dict]:
    from backend.store.chroma_store import _build_doc

    docs = {report["id"]: _build_doc(report) for report in reports}
    queries = []
    for index, (question, topic) in enumerate(OUT_OF_SCOPE_QUESTIONS, 1):
        queries.append(
            {
                "id": f"UO{index:02d}",
                "category": "unanswerable",
                "answerable": False,
                "unanswerable_type": "out_of_scope",
                "symbol": None,
                "company": None,
                "industry": None,
                "question": question,
                "expected_hits": [],
                "reference_answer": "30 份报告均未覆盖该主题，正确回答应说明未覆盖。",
                "rationale": (
                    f"主题为{topic}，不在 30 份报告的覆盖范围内；没有 expected_hits，"
                    "检索到任何报告都属于误召回。"
                ),
                "evidence": {},
            }
        )

    used_symbols: set[str] = set()
    missing = []
    for fact, term in MISSING_FACT_TEMPLATES:
        chosen = None
        for report in reports:
            if report["symbol"] in used_symbols:
                continue
            if term in docs[report["id"]]:
                continue
            chosen = report
            break
        if chosen is None:
            for report in reports:
                if term in docs[report["id"]]:
                    continue
                chosen = report
                break
        if chosen is None:
            continue
        used_symbols.add(chosen["symbol"])
        meta = _report_meta(chosen, manifest_by_symbol)
        report_id = chosen["id"]
        missing.append(
            {
                "id": "",
                "category": "unanswerable",
                "answerable": False,
                "unanswerable_type": "missing_fact",
                "symbol": meta["symbol"],
                "company": meta["company"],
                "industry": meta["industry"],
                "question": f"{meta['company']} 2025 年的{fact}是多少？",
                "expected_hits": [report_id],
                "reference_answer": (
                    "现有报告未包含该数据；正确回答应说明「现有报告未覆盖」，不能给出具体数字。"
                ),
                "rationale": (
                    f"报告字段只有财务摘要、近三年关键财务、舆情与风险，没有{fact}；"
                    "检索应命中该公司报告，但生成阶段必须拒答。"
                ),
                "evidence": {
                    report_id: f"已检查该报告的索引文本，不含「{term}」；"
                    "因此即使检索命中，也无法从报告回答。"
                },
                "missing_fact": fact,
                "fact_term": term,
            }
        )

    if len(missing) < 10:
        raise RuntimeError(f"缺失事实型只构造出 {len(missing)} 条")
    for index, query in enumerate(missing[:10], 1):
        query["id"] = f"UM{index:02d}"
    return queries + missing[:10]


# ---------- 校验与标注文档 ----------


def validate_dataset(dataset: dict, report_ids: set[str]) -> None:
    queries = dataset.get("queries") or []
    if len(queries) != 80:
        raise ValueError(f"评测集应为 80 条，实际 {len(queries)} 条")

    counts = Counter(query["category"] for query in queries)
    expected_counts = {"factual": 20, "comparison": 20, "opinion": 20, "unanswerable": 20}
    if counts != expected_counts:
        raise ValueError(f"品类分布不正确：{dict(counts)}")

    subtypes = Counter(
        query.get("unanswerable_type") for query in queries if query["category"] == "unanswerable"
    )
    if subtypes != {"out_of_scope": 10, "missing_fact": 10}:
        raise ValueError(f"无法回答子类分布不正确：{dict(subtypes)}")

    seen_ids = set()
    for query in queries:
        identifier = query.get("id")
        if not identifier or identifier in seen_ids:
            raise ValueError(f"问题 ID 缺失或重复：{identifier}")
        seen_ids.add(identifier)
        for field in ("question", "reference_answer", "rationale"):
            if not query.get(field):
                raise ValueError(f"{identifier} 缺少 {field}")

        expected = set(query.get("expected_hits") or [])
        unknown = expected - report_ids
        if unknown:
            raise ValueError(f"{identifier} 引用了不存在的报告 ID：{sorted(unknown)}")

        if query["category"] in {"factual", "comparison", "opinion"}:
            if not query.get("answerable"):
                raise ValueError(f"{identifier} 应标记为可回答")
            if not expected:
                raise ValueError(f"{identifier} 没有期望报告")
        else:
            if query.get("answerable"):
                raise ValueError(f"{identifier} 应标记为不可回答")
            if query.get("unanswerable_type") == "out_of_scope" and expected:
                raise ValueError(f"{identifier} 是超出覆盖范围题，不应有期望报告")
            if query.get("unanswerable_type") == "missing_fact" and len(expected) != 1:
                raise ValueError(f"{identifier} 是缺失事实题，应恰好有 1 个正确报告")


def _escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_labels(dataset: dict) -> str:
    names = dataset.get("report_names") or {}
    lines = [
        "# 评测集标注说明",
        "",
        "## 数据来源",
        "",
        f"- 报告快照：`tests/fixtures/eval_reports/reports.json`，"
        f"{dataset.get('report_count')} 份，采集时间 {dataset.get('snapshot_generated_at')}。",
        "- 数据源：项目已有的 akshare 接口（巨潮公司概况、同花顺财务摘要、巨潮个股公告）；"
        "东方财富不可达时自动回退，具体来源逐报告记录在 `manifest.json`。",
        "- 报告由现有四阶段流水线生成，未人工改写；近三年财务表来自同花顺财务摘要。",
        f"- 标注生成时间：{dataset.get('generated_at')}；生成脚本：`tests/eval_build_dataset.py`。",
        "",
        "## 标注口径",
        "",
        "- **事实型**：答案取报告「近三年关键财务」表 2024-12-31 行的原始字符串，"
        "期望报告就是该公司的报告，可逐项对照 `financial_history`。",
        "- **对比型**：问题里写明行业与阈值（例如「资产负债率高于 60%」），"
        "期望集合由快照逐家筛选；不把「明显偏高」这类主观措辞留给标注者自由解释。",
        "- **观点型**：用固定关键词规则在报告叙述字段中匹配，命中句写入每题 evidence；"
        "期望集合是文本匹配结果，不做语义扩展。关键词与风险语境必须落在同一小句，"
        "只有紧邻的下一小句以「显示/反映/存在」等因果判断词开头时才补充语境；"
        "毛利率、利率两条规则用受限正则，避免把费用率下降、净利率误当成目标风险。"
        "规则在跑检索指标之前确定；开测前人工复核 evidence，修正过明显语义误标，"
        "但从未查看 hit@k 或根据指标回头改标签。",
        "- **无法回答型**：",
        "  - `out_of_scope`：主题不在 30 份报告覆盖范围内，`expected_hits` 为空；"
        "检索到任何报告都算误召回。",
        "  - `missing_fact`：报告是正确检索目标，但报告里确实没有该字段（已检查索引文本）；"
        "`answerable=false`，正确生成行为是说明未覆盖，不能编造。",
        "- 标注只依赖报告快照，不依赖检索结果；没有为了让指标好看而改标签或阈值。",
        "",
        "## 品类分布",
        "",
        "| 品类 | 题数 |",
        "|---|---:|",
    ]
    counts = Counter(query["category"] for query in dataset["queries"])
    labels = {
        "factual": "事实型",
        "comparison": "跨股票对比型",
        "opinion": "观点型",
        "unanswerable": "无法回答型",
    }
    for category in ("factual", "comparison", "opinion", "unanswerable"):
        lines.append(f"| {labels[category]} | {counts.get(category, 0)} |")
    lines.append("")

    for category in ("factual", "comparison", "opinion", "unanswerable"):
        subset = [query for query in dataset["queries"] if query["category"] == category]
        if not subset:
            continue
        lines.extend(
            [
                f"## {labels[category]}（{len(subset)} 条）",
                "",
                "| ID | 问题 | 期望报告 | 标注依据 |",
                "|---|---|---|---|",
            ]
        )
        for query in subset:
            expected = query.get("expected_hits") or []
            if expected:
                expected_text = "、".join(
                    f"{names.get(identifier, identifier)}（`{identifier}`）"
                    for identifier in expected
                )
            else:
                expected_text = "无（应拒答）"
            if query.get("unanswerable_type") == "missing_fact":
                expected_text = "正确检索目标：" + expected_text + "（但报告无该事实）"
            lines.append(
                f"| {query['id']} | {_escape_cell(query['question'])} | "
                f"{_escape_cell(expected_text)} | {_escape_cell(query['rationale'])} |"
            )
        lines.append("")
        lines.append("### 逐题证据")
        lines.append("")
        for query in subset:
            lines.append(f"**{query['id']}** {query['question']}")
            lines.append("")
            lines.append(f"- 参考答案：{query['reference_answer']}")
            if query.get("evidence"):
                for identifier, evidence in query["evidence"].items():
                    lines.append(f"- `{identifier}`：{evidence}")
            else:
                lines.append("- 证据：无（该题的正确行为是说明报告未覆盖）")
            lines.append("")
    lines.extend(
        [
            "## 已知局限",
            "",
            "- 30 份报告覆盖四个行业，但每行业只有 7 家左右，对比题的阈值结论不具市场代表性。",
            "- 观点题的期望集合由关键词规则产生，规则可能漏掉语义相近但用词不同的句子；"
            "这是刻意选择的保守口径，宁可少标，不用主观扩展。"
            "受限正则和同小句约束也会带来漏标，例如把「毛利承压」写成其他措辞的报告可能未计入。",
            "- 事实型只取 2024 年报一行，未覆盖 2025 年报与 2026 中报的时序问题。",
            "- 缺失事实题检查的是报告索引文本不含关键词，不保证模型不知道答案；"
            "评测的是模型是否遵守「只依据资料」的约束。",
            "",
        ]
    )
    return "\n".join(lines)


# ---------- 组装 ----------


def build_dataset(reports: list[dict], manifest: dict) -> dict:
    manifest_by_symbol = {item["symbol"]: item for item in manifest.get("symbols") or []}
    report_ids = [report["id"] for report in reports]
    queries = (
        build_factual_queries(reports, manifest_by_symbol)
        + build_comparison_queries(reports, manifest_by_symbol)
        + build_opinion_queries(reports, manifest_by_symbol)
        + build_unanswerable_queries(reports, manifest_by_symbol)
    )
    dataset = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "tests/fixtures/eval_reports/reports.json",
        "snapshot_generated_at": manifest.get("generated_at"),
        "report_count": len(reports),
        "report_ids": report_ids,
        "report_names": {
            report["id"]: re.sub(r"\s+", "", report.get("name") or report["symbol"])
            for report in reports
        },
        "queries": queries,
    }
    validate_dataset(dataset, set(report_ids))
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构造真实报告评测集")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    reports = json.loads(args.reports.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(reports) != 30:
        raise SystemExit(f"快照应有 30 份报告，实际 {len(reports)} 份；先跑 tests/eval_collect.py")

    dataset = build_dataset(reports, manifest)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.labels.write_text(render_labels(dataset), encoding="utf-8")

    counts = Counter(query["category"] for query in dataset["queries"])
    print(f"已写入 {args.output}")
    print(f"已写入 {args.labels}")
    print(f"品类分布：{dict(counts)}")
    for industry in INDUSTRIES:
        sizes = [
            len(query["expected_hits"])
            for query in dataset["queries"]
            if query["category"] == "comparison" and query["industry"] == industry
        ]
        print(f"对比型 {industry}：期望报告数 {sizes}")
    print(
        "观点型规则：",
        [query["pattern"] for query in dataset["queries"] if query["category"] == "opinion"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
