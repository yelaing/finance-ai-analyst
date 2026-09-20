import logging
import time

from openai import OpenAI

from backend.config import LLM_BASE_URL, LLM_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# DashScope text-embedding-v3 单次请求硬上限：实测 11 条即 400
_BATCH_SIZE = 10
_MAX_RETRIES = 2

_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本向量化，按 provider 上限自动分批。返回向量顺序与输入一致。"""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start:start + _BATCH_SIZE]))
    return vectors


def _embed_batch(batch: list[str]) -> list[list[float]]:
    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            return [item.embedding for item in resp.data]
        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                delay = 2 ** attempt
                logger.warning("embedding 调用失败（第 %d/%d 次尝试），%ds 后重试：%s",
                               attempt + 1, _MAX_RETRIES + 1, delay, e)
                time.sleep(delay)
    raise last_error
