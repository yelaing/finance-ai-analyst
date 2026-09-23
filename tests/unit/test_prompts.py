"""prompt 是这个项目的核心资产，改动必须显式确认 —— 因此有 golden 基线。

有意修改 prompt 后，运行 `python tests/regen_prompts_golden.py` 重新生成基线，
并让 diff 进入 code review。
"""

import json
from pathlib import Path

import pytest

from backend.pipeline import prompts as p

TEMPLATES = {
    "FUNDAMENTAL_ANALYSIS_PROMPT": ["name", "symbol", "financial_data"],
    "SENTIMENT_ANALYSIS_PROMPT": ["name", "symbol", "news_data"],
    "BULL_ANALYST_PROMPT": [
        "name",
        "symbol",
        "fundamental_summary",
        "technical_data",
        "news_data",
    ],
    "BEAR_ANALYST_PROMPT": [
        "name",
        "symbol",
        "fundamental_summary",
        "technical_data",
        "news_data",
    ],
    "ARBITRATOR_PROMPT": [
        "fundamental_summary",
        "technical_data",
        "bull_thesis",
        "bear_thesis",
    ],
    "CROSS_VALIDATION_PROMPT": ["fundamental_summary", "sentiment_summary"],
}

# 注意不能放在 tests/data/：.gitignore 里的 `data/` 会把任意层级同名目录一起忽略，
# 那样基线文件进不了仓库，本地通过而 CI 找不到文件
GOLDEN_FILE = Path(__file__).resolve().parent.parent / "fixtures" / "prompts_expected.json"


def render(name: str) -> list[list[str]]:
    """每个变量统一填 X，得到与变量名无关的稳定渲染结果。"""
    template = getattr(p, name)
    return [
        [m.type, m.content]
        for m in template.format_messages(**dict.fromkeys(template.input_variables, "X"))
    ]


@pytest.mark.parametrize(("name", "expected"), TEMPLATES.items())
def test_input_variables_match_expected(name, expected):
    """analyzer 按这些变量名传参，改名字就会 KeyError。"""
    assert sorted(getattr(p, name).input_variables) == sorted(expected)


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_is_a_chat_prompt_template(name):
    from langchain_core.prompts import ChatPromptTemplate

    assert isinstance(getattr(p, name), ChatPromptTemplate)


@pytest.mark.parametrize("name", TEMPLATES)
def test_rendered_prompts_match_golden(name):
    expected = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    assert render(name) == expected[name], f"{name} 的渲染结果变了"


def test_braces_are_escaped_to_literals():
    """模板里写的是 {{ }}，渲染后必须是单个花括号，否则 JSON 示例会坏掉。"""
    messages = render("FUNDAMENTAL_ANALYSIS_PROMPT")
    human = messages[1][1]  # [system, human] 里的 human 正文
    assert "{{" not in human and "}}" not in human
    assert '"metrics"' in human
    assert '"summary"' in human


def test_system_prompt_is_baked_into_the_template():
    messages = p.ARBITRATOR_PROMPT.format_messages(
        fundamental_summary="a", technical_data="b", bull_thesis="c", bear_thesis="d"
    )
    assert [m.type for m in messages] == ["system", "human"]
    assert "风控仲裁官" in messages[0].content


def test_cross_validation_has_no_system_message():
    messages = p.CROSS_VALIDATION_PROMPT.format_messages(
        fundamental_summary="a", sentiment_summary="b"
    )
    assert [m.type for m in messages] == ["human"]


@pytest.mark.parametrize(
    ("name", "needle"),
    [
        ("BULL_ANALYST_PROMPT", "看多"),
        ("BEAR_ANALYST_PROMPT", "看空"),
        ("ARBITRATOR_PROMPT", "独立风控仲裁官"),
        ("SENTIMENT_ANALYSIS_PROMPT", "-1.0 到 1.0"),
    ],
)
def test_key_instructions_survive(name, needle):
    """防止某次润色把关键约束（角色/输出范围）改没了。"""
    text = "\n".join(content for _, content in render(name))
    assert needle in text
