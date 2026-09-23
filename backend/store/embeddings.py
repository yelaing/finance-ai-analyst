import logging
import time

from openai import OpenAI

from backend.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# DashScope text-embedding-v3 单次请求硬上限：实测 11 条即 400
_BATCH_SIZE = 10
_MAX_RETRIES = 2

_client = OpenAI(
    base_url=_settings.llm_base_url,
    api_key=_settings.llm_api_key.get_secret_value(),
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本向量化，按 provider 上限自动分批。返回向量顺序与输入一致。"""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start : start + _BATCH_SIZE]))
    return vectors


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """前 _MAX_RETRIES 次失败会记录日志并退避；最后一次失败直接把异常抛给调用方。"""
    for attempt in range(_MAX_RETRIES):
        try:
            return _embed_once(batch)
        except Exception as e:
            delay = 2**attempt
            logger.warning(
                "embedding 调用失败（第 %d/%d 次尝试），%ds 后重试：%s",
                attempt + 1,
                _MAX_RETRIES + 1,
                delay,
                e,
            )
            time.sleep(delay)
    return _embed_once(batch)


def _embed_once(batch: list[str]) -> list[list[float]]:
    resp = _client.embeddings.create(model=_settings.embedding_model, input=batch)
    return [item.embedding for item in resp.data]
