"""token 用量统计：解析 / 累计 / 成本换算开关 / 汇总日志。"""

import logging
import types

import pytest

from backend.core.tokens import (
    LlmCall,
    TokenUsageAccumulator,
    TokenUsageCallback,
    estimate_cost,
    extract_usage,
    log_summary,
    usage_var,
)


def make_response(
    *,
    input_tokens: int = 10,
    output_tokens: int = 2,
    model: str = "qwen-plus",
    with_usage_metadata: bool = True,
    with_token_usage: bool = True,
):
    message = types.SimpleNamespace(
        usage_metadata=(
            {"input_tokens": input_tokens, "output_tokens": output_tokens}
            if with_usage_metadata
            else None
        ),
        response_metadata={"model_name": model} if model else {},
    )
    llm_output = {"model_name": model}
    if with_token_usage:
        llm_output["token_usage"] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }
    return types.SimpleNamespace(
        generations=[[types.SimpleNamespace(message=message)]],
        llm_output=llm_output,
    )


# ---------- 解析 ----------


def test_extract_usage_prefers_usage_metadata():
    assert extract_usage(make_response(input_tokens=14, output_tokens=1)) == ("qwen-plus", 14, 1)


def test_extract_usage_falls_back_to_llm_output_token_usage():
    assert extract_usage(make_response(with_usage_metadata=False)) == ("qwen-plus", 10, 2)


def test_extract_usage_returns_none_when_nothing_available():
    """拿不到就不编造 —— 宁可没有数字，也不要假数字。"""
    response = make_response(with_usage_metadata=False, with_token_usage=False)
    assert extract_usage(response) is None


def test_extract_usage_tolerates_empty_generations():
    response = types.SimpleNamespace(generations=[], llm_output={})
    assert extract_usage(response) is None


# ---------- 回调 ----------


def test_callback_records_call_into_accumulator():
    accumulator = TokenUsageAccumulator()
    token = usage_var.set(accumulator)
    try:
        TokenUsageCallback("fundamental").on_llm_end(make_response())
    finally:
        usage_var.reset(token)

    assert accumulator.calls == [LlmCall("fundamental", "qwen-plus", 10, 2)]


def test_callback_works_without_accumulator():
    """链可能被单独调用（没有 run_analysis 的上下文），不该因此报错。"""
    TokenUsageCallback("debate").on_llm_end(make_response())


def test_callback_warns_when_usage_unparseable(caplog):
    response = make_response(with_usage_metadata=False, with_token_usage=False)
    with caplog.at_level(logging.WARNING):
        TokenUsageCallback("fundamental").on_llm_end(response)
    assert "未能从响应中解析 token 用量" in caplog.text


def test_unparseable_usage_is_still_counted_as_a_call():
    """解析不出用量也要计数，否则「调用了但没记到 token」会从指标上凭空消失。"""
    from backend.core.metrics import get_metrics

    response = make_response(with_usage_metadata=False, with_token_usage=False)
    TokenUsageCallback("fundamental").on_llm_end(response)

    assert (
        get_metrics().registry.get_sample_value(
            "llm_calls_total", {"stage": "fundamental", "outcome": "usage_unparsed"}
        )
        == 1
    )


def test_callback_counts_failures_via_metrics():
    """失败的调用不会走 on_llm_end —— 不在这条路径上记，失败率就永远是 0。"""
    from backend.core.metrics import get_metrics

    TokenUsageCallback("debate").on_llm_error(TimeoutError("超时"))

    assert (
        get_metrics().registry.get_sample_value(
            "llm_calls_total", {"stage": "debate", "outcome": "TimeoutError"}
        )
        == 1
    )


def test_callback_logs_structured_fields(caplog):
    with caplog.at_level(logging.INFO):
        TokenUsageCallback("sentiment").on_llm_end(make_response(input_tokens=7, output_tokens=3))

    record = next(r for r in caplog.records if "LLM 调用完成" in r.getMessage())
    assert (record.stage, record.tokens_in, record.tokens_out) == ("sentiment", 7, 3)
    assert record.model == "qwen-plus"


# ---------- 累加器 ----------


def test_totals_sum_every_call():
    accumulator = TokenUsageAccumulator()
    accumulator.add(LlmCall("a", "m", 10, 1))
    accumulator.add(LlmCall("b", "m", 20, 2))
    assert accumulator.totals() == (30, 3)


def test_cached_stages_are_tracked_separately_from_calls():
    accumulator = TokenUsageAccumulator()
    accumulator.mark_cached("sentiment")
    assert accumulator.cached_stages == ["sentiment"]
    assert accumulator.calls == []  # 命中缓存不该产生 token 记录


def test_calls_property_returns_a_copy():
    """外部改动返回值不该影响内部状态。"""
    accumulator = TokenUsageAccumulator()
    accumulator.add(LlmCall("a", "m", 1, 1))
    accumulator.calls.clear()
    assert len(accumulator.calls) == 1


# ---------- 成本换算 ----------


def test_estimate_cost_requires_both_prices():
    calls = [LlmCall("a", "m", 1000, 500)]
    assert estimate_cost(calls, None, 0.002) is None
    assert estimate_cost(calls, 0.001, None) is None


def test_estimate_cost_with_configured_prices():
    calls = [LlmCall("a", "m", 1000, 1000)]
    assert estimate_cost(calls, 0.001, 0.002) == pytest.approx(0.003)


def test_estimate_cost_of_no_calls_is_zero():
    assert estimate_cost([], 0.001, 0.002) == 0.0


# ---------- 汇总日志 ----------


def test_log_summary_lists_cached_stages(caplog):
    """命中缓存时 token 为 0 是正常的 —— 汇总必须把命中阶段列出来，否则会被误读成统计失效。"""
    accumulator = TokenUsageAccumulator()
    accumulator.mark_cached("fundamental")

    with caplog.at_level(logging.INFO):
        log_summary("600519", accumulator, 12.3)

    record = next(r for r in caplog.records if "分析用量汇总" in r.getMessage())
    assert record.llm_calls == 0
    assert record.cached_stages == "fundamental"
    assert (record.tokens_in, record.tokens_out) == (0, 0)
    assert record.duration_ms == 12.3


def test_log_summary_omits_cost_when_prices_unset(monkeypatch, caplog):
    monkeypatch.delenv("LLM_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("LLM_PRICE_OUTPUT_PER_1K", raising=False)
    from backend.config import get_settings

    get_settings.cache_clear()

    accumulator = TokenUsageAccumulator()
    accumulator.add(LlmCall("a", "m", 1000, 1000))
    with caplog.at_level(logging.INFO):
        log_summary("600519", accumulator, 1.0)

    record = next(r for r in caplog.records if "分析用量汇总" in r.getMessage())
    assert not hasattr(record, "est_cost")


def test_log_summary_includes_cost_when_prices_configured(monkeypatch, caplog):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_1K", "0.001")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_1K", "0.002")
    from backend.config import get_settings

    get_settings.cache_clear()

    accumulator = TokenUsageAccumulator()
    accumulator.add(LlmCall("a", "m", 1000, 1000))
    with caplog.at_level(logging.INFO):
        log_summary("600519", accumulator, 1.0)

    record = next(r for r in caplog.records if "分析用量汇总" in r.getMessage())
    assert record.est_cost == pytest.approx(0.003)


def test_blank_price_env_is_treated_as_unset(monkeypatch):
    """.env 里写成 LLM_PRICE_INPUT_PER_1K= 应当视为未配置，而不是启动报错。"""
    from backend.config import Settings

    settings = Settings(_env_file=None, llm_price_input_per_1k="", llm_price_output_per_1k="  ")
    assert settings.llm_price_input_per_1k is None
    assert settings.llm_price_output_per_1k is None
