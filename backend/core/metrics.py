"""Prometheus 指标。

**标签基数必须保持有界。** HTTP 的 path 一律用**路由模板**
（`/api/v1/export/{symbol}`）而不是原始 URL —— 用原始 URL 的话每个股票代码都会
长出一条独立的时间序列，几百个股票就能把 Prometheus 的内存和查询拖垮。
其余标签（stage / model / namespace / outcome）都是有限集合。

registry 可注入：测试用独立 registry，避免计数器跨用例累积。
"""

import functools

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# 分析终态
OUTCOME_OK = "ok"
OUTCOME_FAILED = "failed"


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        self.http_requests = Counter(
            "http_requests_total",
            "HTTP 请求数",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP 请求耗时",
            ["method", "path"],
            registry=self.registry,
        )
        self.llm_calls = Counter(
            "llm_calls_total",
            "LLM 调用数（含失败与无法解析用量）",
            ["stage", "outcome"],
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "llm_tokens_total",
            "LLM token 用量",
            ["stage", "model", "kind"],
            registry=self.registry,
        )
        self.cache_ops = Counter(
            "cache_operations_total",
            "缓存读取次数",
            ["namespace", "result"],
            registry=self.registry,
        )
        self.data_fetch = Histogram(
            "data_fetch_seconds",
            "数据源抓取耗时",
            ["market", "outcome"],
            registry=self.registry,
        )
        self.degraded_stages = Counter(
            "analysis_degraded_stages_total",
            "被降级为缺失的分析阶段",
            ["stage"],
            registry=self.registry,
        )
        self.analysis_runs = Counter(
            "analysis_runs_total",
            "分析请求终态",
            ["outcome"],
            registry=self.registry,
        )

    # ---------- 便捷封装：把「怎么记」收在一处，调用点只表达「发生了什么」 ----------

    def observe_http(self, method: str, path: str, status: int, seconds: float) -> None:
        self.http_requests.labels(method=method, path=path, status=str(status)).inc()
        self.http_duration.labels(method=method, path=path).observe(seconds)

    def observe_cache(self, namespace: str, hit: bool) -> None:
        self.cache_ops.labels(namespace=namespace, result="hit" if hit else "miss").inc()

    def observe_llm_tokens(
        self, stage: str, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        self.llm_calls.labels(stage=stage, outcome="success").inc()
        self.llm_tokens.labels(stage=stage, model=model, kind="input").inc(input_tokens)
        self.llm_tokens.labels(stage=stage, model=model, kind="output").inc(output_tokens)

    def observe_llm_failure(self, stage: str, outcome: str) -> None:
        """outcome 用异常类名（APITimeoutError / RateLimitError …）—— 有限集合，基数有界。"""
        self.llm_calls.labels(stage=stage, outcome=outcome).inc()

    def observe_fetch(self, market: str, healthy: bool, seconds: float) -> None:
        self.data_fetch.labels(market=market, outcome="ok" if healthy else "degraded").observe(
            seconds
        )

    def observe_analysis(self, ok: bool) -> None:
        self.analysis_runs.labels(outcome=OUTCOME_OK if ok else OUTCOME_FAILED).inc()


@functools.lru_cache
def get_metrics() -> Metrics:
    """进程内单例。测试改不了它，用 Metrics(CollectorRegistry()) 自建；或 cache_clear()。"""
    return Metrics()


def render() -> tuple[bytes, str]:
    return generate_latest(get_metrics().registry), CONTENT_TYPE_LATEST
