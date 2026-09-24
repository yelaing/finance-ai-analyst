"""指标记法。

每个用例用独立 registry —— 默认 registry 是全局的，跨用例会累积计数，
断言就只能写成「大于等于」这种没意义的形状。
"""

import pytest
from prometheus_client import CollectorRegistry

from backend.core.metrics import Metrics


@pytest.fixture
def metrics() -> Metrics:
    return Metrics(CollectorRegistry())


def value(metrics: Metrics, name: str, **labels) -> float:
    sample = metrics.registry.get_sample_value(name, labels or None)
    assert sample is not None, f"没有采到 {name}{labels}"
    return sample


def test_http_observation_records_count_and_duration(metrics):
    metrics.observe_http(method="GET", path="/health", status=200, seconds=0.25)

    assert value(metrics, "http_requests_total", method="GET", path="/health", status="200") == 1
    assert value(metrics, "http_request_duration_seconds_count", method="GET", path="/health") == 1
    assert value(
        metrics, "http_request_duration_seconds_sum", method="GET", path="/health"
    ) == pytest.approx(0.25)


def test_http_observations_accumulate(metrics):
    for _ in range(3):
        metrics.observe_http(method="POST", path="/api/v1/analyze", status=200, seconds=1.0)
    assert (
        value(metrics, "http_requests_total", method="POST", path="/api/v1/analyze", status="200")
        == 3
    )


def test_cache_hit_and_miss_are_distinguished(metrics):
    metrics.observe_cache("llm", hit=True)
    metrics.observe_cache("llm", hit=True)
    metrics.observe_cache("llm", hit=False)

    assert value(metrics, "cache_operations_total", namespace="llm", result="hit") == 2
    assert value(metrics, "cache_operations_total", namespace="llm", result="miss") == 1


def test_llm_tokens_counted_per_kind_and_model(metrics):
    metrics.observe_llm_tokens("fundamental", "qwen-plus", input_tokens=100, output_tokens=30)

    assert value(metrics, "llm_calls_total", stage="fundamental", outcome="success") == 1
    assert (
        value(metrics, "llm_tokens_total", stage="fundamental", model="qwen-plus", kind="input")
        == 100
    )
    assert (
        value(metrics, "llm_tokens_total", stage="fundamental", model="qwen-plus", kind="output")
        == 30
    )


def test_llm_failure_counted_separately_from_success(metrics):
    metrics.observe_llm_failure("sentiment", "APITimeoutError")

    assert value(metrics, "llm_calls_total", stage="sentiment", outcome="APITimeoutError") == 1
    assert (
        metrics.registry.get_sample_value(
            "llm_calls_total", {"stage": "sentiment", "outcome": "success"}
        )
        is None
    )


def test_fetch_outcome_distinguishes_degraded(metrics):
    metrics.observe_fetch("a_share", healthy=True, seconds=0.8)
    metrics.observe_fetch("a_share", healthy=False, seconds=0.3)

    assert value(metrics, "data_fetch_seconds_count", market="a_share", outcome="ok") == 1
    assert value(metrics, "data_fetch_seconds_count", market="a_share", outcome="degraded") == 1


def test_analysis_outcome_counted(metrics):
    metrics.observe_analysis(ok=True)
    metrics.observe_analysis(ok=False)
    metrics.observe_analysis(ok=False)

    assert value(metrics, "analysis_runs_total", outcome="ok") == 1
    assert value(metrics, "analysis_runs_total", outcome="failed") == 2


def test_registries_are_isolated():
    first = Metrics(CollectorRegistry())
    second = Metrics(CollectorRegistry())

    first.observe_analysis(ok=True)

    assert first.registry.get_sample_value("analysis_runs_total", {"outcome": "ok"}) == 1
    assert second.registry.get_sample_value("analysis_runs_total", {"outcome": "ok"}) is None
