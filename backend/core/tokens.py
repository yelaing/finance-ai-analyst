"""LLM token 用量统计。

每次调用记一条（阶段 / 模型 / 输入输出 token），流水线结束打一条请求级汇总。
成本默认**只统计 token、不换算金额** —— 各提供方价格变动频繁，硬编码一张价目表
等于编造事实；配置了 LLM_PRICE_*_PER_1K 才额外给出估算金额。
"""

import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmCall:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int


class TokenUsageAccumulator:
    """一个请求一份。

    放在 ContextVar 里能让并行分支共用同一个对象：RunnableParallel 的工作线程
    拿到的是 context 副本，但副本里装的是**同一个对象引用**，就地修改父线程可见。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[LlmCall] = []
        self._cached_stages: list[str] = []

    def add(self, call: LlmCall) -> None:
        with self._lock:
            self._calls.append(call)

    def mark_cached(self, stage: str) -> None:
        """记录「这一阶段命中缓存、没真正调用」。汇总里要能区分，否则 0 token 会被误读成统计坏了。"""
        with self._lock:
            self._cached_stages.append(stage)

    @property
    def calls(self) -> list[LlmCall]:
        with self._lock:
            return list(self._calls)

    @property
    def cached_stages(self) -> list[str]:
        with self._lock:
            return list(self._cached_stages)

    def totals(self) -> tuple[int, int]:
        calls = self.calls
        return (
            sum(call.input_tokens for call in calls),
            sum(call.output_tokens for call in calls),
        )


usage_var: ContextVar[TokenUsageAccumulator | None] = ContextVar("llm_usage", default=None)


def extract_usage(response: Any) -> tuple[str, int, int] | None:
    """从 LLMResult 里解析出 (model, input_tokens, output_tokens)；解析不出来就返回 None。

    实测 langchain-openai 1.x + DashScope 会同时给出 message.usage_metadata 与
    llm_output["token_usage"]，优先用前者。
    """
    llm_output = getattr(response, "llm_output", None) or {}
    model = llm_output.get("model_name") or ""

    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        message = getattr(generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None)
        if usage:
            metadata = getattr(message, "response_metadata", None) or {}
            return (
                model or metadata.get("model_name", ""),
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
            )

    token_usage = llm_output.get("token_usage")
    if token_usage:
        return (
            model,
            int(token_usage.get("prompt_tokens", 0)),
            int(token_usage.get("completion_tokens", 0)),
        )
    return None


class TokenUsageCallback(BaseCallbackHandler):
    """挂在链的 config 上：每次 LLM 调用记一条，并打一条结构化日志。

    没有它就只能知道「花了多少钱」，不知道花在哪一阶段。
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        parsed = extract_usage(response)
        if parsed is None:
            logger.warning("未能从响应中解析 token 用量 stage=%s", self.stage)
            return

        model, input_tokens, output_tokens = parsed
        logger.info(
            "LLM 调用完成 stage=%s model=%s tokens_in=%d tokens_out=%d",
            self.stage,
            model,
            input_tokens,
            output_tokens,
            extra={
                "stage": self.stage,
                "model": model,
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
            },
        )

        accumulator = usage_var.get()
        if accumulator is not None:
            accumulator.add(LlmCall(self.stage, model, input_tokens, output_tokens))


def estimate_cost(
    calls: list[LlmCall],
    input_price_per_1k: float | None,
    output_price_per_1k: float | None,
) -> float | None:
    """两个单价都配了才换算；缺一个就返回 None（不猜、不编造）。"""
    if input_price_per_1k is None or output_price_per_1k is None:
        return None

    input_tokens = sum(call.input_tokens for call in calls)
    output_tokens = sum(call.output_tokens for call in calls)
    return round(
        input_tokens / 1000 * input_price_per_1k + output_tokens / 1000 * output_price_per_1k,
        6,
    )


def log_summary(symbol: str, accumulator: TokenUsageAccumulator, elapsed_ms: float) -> None:
    """请求级汇总。cached_stages 必须打出来 —— 命中缓存时 token 为 0 是正常的，不是统计失效。"""
    from backend.config import get_settings

    settings = get_settings()
    calls = accumulator.calls
    input_tokens, output_tokens = accumulator.totals()
    cost = estimate_cost(
        calls,
        settings.llm_price_input_per_1k,
        settings.llm_price_output_per_1k,
    )

    extra: dict[str, Any] = {
        "llm_calls": len(calls),
        "tokens_in": input_tokens,
        "tokens_out": output_tokens,
        "cached_stages": ",".join(accumulator.cached_stages) or "-",
        "duration_ms": round(elapsed_ms, 1),
    }
    if cost is not None:
        extra["est_cost"] = cost

    logger.info(
        "分析用量汇总 symbol=%s llm_calls=%d cached_stages=%s tokens_in=%d tokens_out=%d%s",
        symbol,
        len(calls),
        extra["cached_stages"],
        input_tokens,
        output_tokens,
        f" est_cost={cost}" if cost is not None else "",
        extra=extra,
    )
