"""评测集构造的解析、文本匹配与校验逻辑，用构造的报告对象验证，不联网。"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "eval_build_dataset.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("eval_build_dataset", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(risks=None, summary="", conclusion="", sentiment=None, technical=None):
    return {
        "id": "600519_20261029_185253",
        "symbol": "600519",
        "name": "贵州茅台",
        "fundamental_summary": summary,
        "conclusion": conclusion,
        "full_report": {
            "risks": risks or [],
            "sentiment": sentiment,
            "technical": technical,
        },
    }


def test_report_meta_normalizes_company_name_for_questions():
    builder = load_builder()

    meta = builder._report_meta({"symbol": "000858", "name": "五 粮 液"}, {})

    assert meta["company"] == "五粮液"


def test_parse_history_reads_header_and_rows():
    builder = load_builder()
    text = """--- 近三年关键财务 ---
报告期 | 营业总收入 | 净资产收益率
2024-12-31 | 1741.44亿 | 36.02%
2025-12-31 | 1720.54亿 | 32.53%"""

    rows = builder.parse_history(text)

    assert len(rows) == 2
    assert rows[0]["报告期"] == "2024-12-31"
    assert rows[0]["净资产收益率"] == "36.02%"


def test_metric_value_parses_percent_and_number_and_skips_dirty():
    builder = load_builder()
    row = {"净资产收益率": "36.02%", "流动比率": "4.45", "资产负债率": "—", "净利润": "862.28亿"}

    assert builder.metric_value(row, "净资产收益率") == 36.02
    assert builder.metric_value(row, "流动比率") == 4.45
    assert builder.metric_value(row, "净利润") == 862.28
    assert builder.metric_value(row, "资产负债率") is None


def test_opinion_pattern_only_matches_risk_field_and_context():
    builder = load_builder()
    report = _report(
        risks=[{"category": "现金流", "detail": "经营性现金流持续为负，存在流动性压力。"}],
        summary="公司现金流健康，经营质量优异。",
    )
    pattern = {
        "key": "cashflow_risk",
        "question": "哪些报告提示了现金流风险？",
        "fields": ("风险点",),
        "required": ("现金流",),
        "context_any": ("压力", "风险", "紧张"),
    }

    matches = builder.opinion_matches(report, pattern)

    assert len(matches) == 1
    assert matches[0][0] == "风险点·现金流"
    assert "流动性压力" in matches[0][1]


def test_margin_decline_ignores_decline_in_unrelated_clause():
    builder = load_builder()
    report = _report(
        summary="盈利能力指标（毛利率、净利率）持续优化，资产负债率下降反映财务结构趋稳。",
    )
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "margin_decline")

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_cashflow_healthy_ignores_healthy_structure_when_cashflow_is_under_pressure():
    builder = load_builder()
    report = _report(
        summary="经营现金流同比有所承压，资产负债率温和上升但仍处极低水平，整体财务结构健康。",
    )
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "cashflow_healthy")

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_policy_risk_ignores_accounting_policy_and_policy_credibility():
    builder = load_builder()
    report = _report(
        risks=[
            {"category": "盈利质量", "detail": "利润依赖会计政策调整，可持续性存疑。"},
            {"category": "治理", "detail": "存在资本运作与经营实质脱钩风险，削弱政策可信度。"},
        ],
    )
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "policy_risk")

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_valuation_high_ignores_valuation_repair_without_high_valuation_evidence():
    builder = load_builder()
    report = _report(
        conclusion="估值修复缺乏扎实盈利质量支撑，短期面临压力。",
    )
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "valuation_high")

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_margin_decline_ignores_unrelated_cost_ratio_decline():
    builder = load_builder()
    report = _report(
        risks=[
            {
                "category": "盈利质量",
                "detail": "销售净利率与毛利率提升主要源于费用压降（如销售费用率下降）及结构微调。",
            }
        ],
    )
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "margin_decline")

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_margin_decline_matches_metric_list_decline():
    builder = load_builder()
    report = _report(summary="盈利能力指标（毛利率、净利率、ROE）均出现回落。")
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "margin_decline")

    matches = builder.opinion_matches(report, pattern)

    assert len(matches) == 1


def test_interest_rate_ignores_profit_margin_substring():
    builder = load_builder()
    report = _report(
        risks=[{"category": "盈利", "detail": "销售净利率回落面临压力。"}],
    )
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "interest_rate")

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_inventory_pressure_ignores_inventory_turnover_in_execution_risk():
    builder = load_builder()
    report = _report(
        risks=[
            {
                "category": "治理",
                "detail": "缺乏可量化的执行证据（如经销商库存周转加快），存在时间差风险。",
            }
        ],
    )
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "inventory_pressure")

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_shareholder_risk_ignores_broker_pledge_business():
    builder = load_builder()
    report = _report(
        risks=[
            {
                "category": "杠杆质量",
                "detail": "未披露两融/股票质押业务不良率变化，高杠杆缺乏底层资产质量验证。",
            }
        ],
    )
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "shareholder_risk")

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_opinion_context_can_sit_in_following_consequence_clause():
    builder = load_builder()
    report = _report(
        risks=[{"category": "现金流", "detail": "经营性现金流持续为负，存在流动性压力。"}],
    )
    pattern = {
        "key": "cashflow_risk",
        "question": "哪些报告提示了现金流风险？",
        "fields": ("风险点",),
        "required": ("现金流",),
        "context_any": ("压力", "风险", "紧张"),
    }

    matches = builder.opinion_matches(report, pattern)

    assert len(matches) == 1


def test_opinion_rationale_documents_regex_rule(monkeypatch):
    builder = load_builder()
    report = _report(summary="毛利率与ROE明显回落。")
    pattern = next(item for item in builder.OPINION_PATTERNS if item["key"] == "margin_decline")
    monkeypatch.setattr(builder, "OPINION_PATTERNS", (pattern,) * 20)

    queries = builder.build_opinion_queries([report], {})

    assert "匹配正则" in queries[0]["rationale"]


def test_opinion_pattern_ignores_fields_not_in_rag_index():
    builder = load_builder()
    report = _report(
        technical={"trend_summary": "技术面偏强，突破关键阻力位。"},
        summary="基本面一般。",
    )
    pattern = {
        "key": "technical_strong",
        "question": "哪些报告认为技术面偏强？",
        "fields": ("技术面", "基本面总结"),
        "match_any": ("偏强",),
    }

    matches = builder.opinion_matches(report, pattern)

    assert matches == []


def test_opinion_questions_are_not_capped_at_five_reports(monkeypatch):
    builder = load_builder()
    reports = []
    for index in range(6):
        report = _report(
            risks=[{"category": "现金流", "detail": "经营性现金流承压，存在流动性风险。"}],
        )
        report["id"] = f"doc{index}"
        report["symbol"] = f"00000{index}"
        report["name"] = f"公司{index}"
        reports.append(report)
    pattern = {
        "key": "cashflow_risk",
        "question": "哪些报告提示了现金流风险？",
        "fields": ("风险点",),
        "required": ("现金流",),
        "context_any": ("风险", "压力"),
    }
    monkeypatch.setattr(builder, "OPINION_PATTERNS", (pattern,) * 20)

    queries = builder.build_opinion_queries(reports, {})

    assert len(queries) == 20
    assert len(queries[0]["expected_hits"]) == 6


def test_opinion_pattern_supports_structured_negative_sentiment():
    builder = load_builder()
    report = _report(sentiment={"overall": "negative", "score": -0.4, "key_drivers": ["政策收紧"]})
    pattern = {"key": "sentiment_negative", "special": "sentiment_negative", "fields": ()}

    matches = builder.opinion_matches(report, pattern)

    assert len(matches) == 1
    assert "negative" in matches[0][1]


def _query(identifier, category, answerable=True, subtype=None, expected=("doc1",)):
    return {
        "id": identifier,
        "category": category,
        "answerable": answerable,
        "unanswerable_type": subtype,
        "question": f"问题 {identifier}",
        "expected_hits": list(expected),
        "reference_answer": "参考答案",
        "rationale": "标注理由",
        "evidence": {},
    }


def _base_dataset():
    queries = []
    for index in range(20):
        queries.append(_query(f"F{index:02d}", "factual"))
        queries.append(_query(f"C{index:02d}", "comparison"))
        queries.append(_query(f"O{index:02d}", "opinion"))
        if index < 10:
            queries.append(
                _query(
                    f"UO{index:02d}",
                    "unanswerable",
                    answerable=False,
                    subtype="out_of_scope",
                    expected=(),
                )
            )
        else:
            queries.append(
                _query(
                    f"UM{index:02d}",
                    "unanswerable",
                    answerable=False,
                    subtype="missing_fact",
                )
            )
    return {"report_ids": ["doc1"], "queries": queries}


def test_validate_dataset_accepts_balanced_dataset():
    builder = load_builder()

    builder.validate_dataset(_base_dataset(), {"doc1"})


def test_validate_dataset_rejects_unknown_expected_report():
    builder = load_builder()
    dataset = _base_dataset()
    dataset["queries"][0]["expected_hits"] = ["missing-doc"]

    try:
        builder.validate_dataset(dataset, {"doc1"})
    except ValueError as exc:
        assert "missing-doc" in str(exc)
    else:
        raise AssertionError("应当拒绝不存在的报告 ID")


def test_metric_value_skips_boolean_placeholder():
    builder = load_builder()

    assert builder.metric_value({"流动比率": "False"}, "流动比率") is None
