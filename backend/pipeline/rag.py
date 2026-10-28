"""报告问答（RAG）：先检索历史报告，再把命中片段拼进 prompt 让模型作答。"""

import logging
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.core.tokens import TokenUsageCallback
from backend.pipeline.prompts import RAG_QA_PROMPT

logger = logging.getLogger(__name__)

_settings = get_settings()

_TEMPERATURE = 0.2

_llm = ChatOpenAI(
    base_url=_settings.llm_base_url,
    api_key=_settings.llm_api_key,
    model=_settings.llm_model,
    temperature=_TEMPERATURE,
    max_tokens=1024,
    request_timeout=_settings.llm_timeout,
)

qa_chain = (RAG_QA_PROMPT | _llm | StrOutputParser()).with_retry(
    stop_after_attempt=3,
    exponential_jitter_params={"initial": 1, "max": 2, "jitter": 0},
)

NO_HITS_ANSWER = "没有找到相关的历史分析报告。请先分析相关股票，或换一种问法。"


def _hit_date(hit: dict) -> str:
    timestamp = hit.get("timestamp")
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d")
    return str(timestamp)[:10] if timestamp else "未知日期"


def build_context(hits: list[dict]) -> str:
    """把检索结果编成带编号的上下文，编号与回答里的 [n] 引用对应。"""
    return "\n\n".join(
        f"[{index}] {hit['name']}（{hit['symbol']}） · {_hit_date(hit)}\n{hit['text']}"
        for index, hit in enumerate(hits, 1)
    )


def answer_question(question: str, hits: list[dict]) -> str:
    """检索结果为空时不调用 LLM —— 没有资料时让模型自由发挥只会产生幻觉。"""
    if not hits:
        return NO_HITS_ANSWER
    return qa_chain.invoke(
        {"question": question, "context": build_context(hits)},
        config={"callbacks": [TokenUsageCallback("qa")]},
    )
