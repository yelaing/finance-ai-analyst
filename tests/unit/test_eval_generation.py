"""生成侧评测的纯函数：抽样、引用解析、阈值模拟与失败归因。不调用 LLM。"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "eval_generation.py"


def load_generation():
    spec = importlib.util.spec_from_file_location("eval_generation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset():
    queries = []
    for index in range(20):
        queries.append(
            {
                "id": f"F{index:02d}",
                "category": "factual",
                "answerable": True,
                "expected_hits": ["doc1"],
                "question": f"事实 {index}",
            }
        )
    for index in range(20):
        queries.append(
            {
                "id": f"C{index:02d}",
                "category": "comparison",
                "answerable": True,
                "expected_hits": ["doc1"],
                "industry": ("金融", "白酒", "新能源", "医药")[index % 4],
                "question": f"对比 {index}",
            }
        )
    for index in range(20):
        queries.append(
            {
                "id": f"O{index:02d}",
                "category": "opinion",
                "answerable": True,
                "expected_hits": ["doc1"],
                "question": f"观点 {index}",
            }
        )
    for index in range(10):
        queries.append(
            {
                "id": f"UO{index:02d}",
                "category": "unanswerable",
                "answerable": False,
                "unanswerable_type": "out_of_scope",
                "expected_hits": [],
                "question": f"超纲 {index}",
            }
        )
    for index in range(10):
        queries.append(
            {
                "id": f"UM{index:02d}",
                "category": "unanswerable",
                "answerable": False,
                "unanswerable_type": "missing_fact",
                "expected_hits": ["doc1"],
                "question": f"缺字段 {index}",
            }
        )
    return {"queries": queries}


def test_select_generation_sample_is_balanced_and_deterministic():
    generation = load_generation()
    dataset = _dataset()

    sample = generation.select_generation_sample(dataset)

    assert len(sample) == 30
    counts = {}
    for query in sample:
        counts[query["category"]] = counts.get(query["category"], 0) + 1
    assert counts == {"factual": 8, "comparison": 7, "opinion": 5, "unanswerable": 10}
    subtypes = [
        query.get("unanswerable_type") for query in sample if query["category"] == "unanswerable"
    ]
    assert subtypes.count("out_of_scope") == 5
    assert subtypes.count("missing_fact") == 5


def test_extract_citation_indices_deduplicates_and_keeps_order():
    generation = load_generation()

    assert generation.extract_citation_indices("结论 [2]，依据 [1][2]，无效 [x]。") == [2, 1]


def test_citation_metrics_counts_out_of_range_as_unsupported():
    generation = load_generation()
    citations = [{"id": "doc1"}, {"id": "doc2"}]
    judge = {"citations": [{"index": 1, "supported": True}, {"index": 3, "supported": True}]}

    metrics = generation.citation_metrics("说法 [1][3]。", citations, judge)

    assert metrics["indices"] == [1, 3]
    assert metrics["valid_indices"] == [1]
    assert metrics["invalid_indices"] == [3]
    assert metrics["supported_indices"] == [1]
    assert metrics["accuracy"] == 0.5
    assert metrics["cited_ids"] == ["doc1"]


def test_value_matched_ignores_spaces_but_not_wrong_numbers():
    generation = load_generation()

    assert generation.value_matched("2024 年 ROE 为 36.02%", "36.02%")
    assert generation.value_matched("营业收入 1.03 万亿", "1.03万亿")
    assert not generation.value_matched("2024 年 ROE 为 36.0%", "36.02%")


def test_apply_threshold_turns_low_score_into_no_hits():
    generation = load_generation()

    applied = generation.apply_threshold({"top1_score": 0.4, "answer": "编的答案"}, 0.5)
    kept = generation.apply_threshold({"top1_score": 0.7, "answer": "正常答案"}, 0.5)

    assert applied["applied"] is True
    assert "没有找到相关的历史分析报告" in applied["answer"]
    assert kept["applied"] is False
    assert kept["answer"] == "正常答案"


def test_classify_failure_recognizes_correct_report_from_citations():
    generation = load_generation()
    query = {
        "category": "factual",
        "answerable": True,
        "expected_hits": ["doc1"],
        "expected_value": "36.02%",
    }
    judge = {
        "fabricated": False,
        "refused": False,
        "relevance": 5,
        "citations": [{"index": 1, "supported": True}],
    }
    result = {"citations": [{"id": "doc1"}], "answer": "ROE 为 36.02% [1]"}

    failure = generation.classify_failure(
        query,
        result,
        judge,
        generation.citation_metrics(result["answer"], result["citations"], judge),
    )

    assert failure is None


def test_classify_failure_keeps_unsupported_citation_out_of_unused_report():
    generation = load_generation()
    query = {"category": "comparison", "answerable": True, "expected_hits": ["doc1"]}
    judge = {
        "fabricated": False,
        "refused": False,
        "relevance": 4,
        "citations": [
            {"index": 1, "supported": True},
            {"index": 2, "supported": False},
        ],
    }
    result = {
        "citations": [{"id": "doc1"}, {"id": "doc2"}],
        "answer": "正确事实 [1]，额外断言 [2]",
    }

    failure = generation.classify_failure(
        query,
        result,
        judge,
        generation.citation_metrics(result["answer"], result["citations"], judge),
    )

    assert failure["primary"] == "引用不支撑"
    assert "召回了正确报告但模型没用" not in failure["reasons"]


def test_classify_failure_identifies_retrieval_miss_and_unused_report():
    generation = load_generation()
    query = {
        "category": "factual",
        "answerable": True,
        "expected_hits": ["doc1"],
        "expected_value": "36.02%",
    }
    judge = {"fabricated": False, "refused": False, "relevance": 4, "citations": []}

    miss = generation.classify_failure(
        query,
        {"citations": [{"id": "doc2"}], "answer": "不知道"},
        judge,
        {"accuracy": None, "cited_ids": [], "invalid_indices": []},
    )
    assert miss["primary"] == "检索没召回"

    unused = generation.classify_failure(
        query,
        {"citations": [{"id": "doc1"}, {"id": "doc2"}], "answer": "没有引用正确内容"},
        judge,
        {"accuracy": None, "cited_ids": [], "invalid_indices": []},
    )
    assert unused["primary"] == "召回了正确报告但模型没用"


def test_dataset_hash_changes_when_snapshot_is_refreshed():
    generation = load_generation()
    dataset = _dataset()

    first = generation.dataset_hash(
        {**dataset, "snapshot_generated_at": "2026-10-29T10:00:00"}, "qwen"
    )
    second = generation.dataset_hash(
        {**dataset, "snapshot_generated_at": "2026-10-30T10:00:00"}, "qwen"
    )

    assert first != second


def test_render_spotcheck_escapes_pipes_from_financial_tables():
    generation = load_generation()
    sample = {
        "sample_id": "S01",
        "question": "ROE 是多少",
        "answer_sentence": "ROE 为 36.02% [1]",
        "citation_text": "| 报告期 | ROE |\n| 2024 | 36.02% |",
        "judge_supported": True,
        "judge_reason": "数值一致",
    }

    text = generation.render_spotcheck([sample], None)

    assert "| 报告期 | ROE |" not in text
    assert "\\| 报告期 \\| ROE \\|" in text


def test_classify_failure_marks_fabrication_and_ignores_correct_refusal():
    generation = load_generation()
    judge = {"fabricated": True, "refused": False, "relevance": 2, "citations": []}

    failure = generation.classify_failure(
        {
            "category": "unanswerable",
            "answerable": False,
            "unanswerable_type": "missing_fact",
            "expected_hits": ["doc1"],
        },
        {"citations": [{"id": "doc1"}], "answer": "员工人数是 12345 人"},
        judge,
        {"accuracy": None, "cited_ids": [], "invalid_indices": []},
    )
    assert failure["primary"] == "模型编造"

    assert (
        generation.classify_failure(
            {
                "category": "unanswerable",
                "answerable": False,
                "unanswerable_type": "out_of_scope",
                "expected_hits": [],
            },
            {"citations": [{"id": "doc1"}], "answer": "现有报告未覆盖"},
            {"fabricated": False, "refused": True, "relevance": 5, "citations": []},
            {"accuracy": None, "cited_ids": [], "invalid_indices": []},
        )
        is None
    )
