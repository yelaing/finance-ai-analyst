"""全部配置集中在这里，来源是环境变量与 .env（.env 不进版本库）。"""

import functools
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 运行环境。prod 下日志默认输出 JSON，且 uvicorn 关闭热重载
    env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] | None = None

    # LLM
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr = SecretStr("sk-placeholder")
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-v3"

    # 存储
    chroma_persist_dir: str = "./data/chroma"

    # 服务
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: list[str] = ["*"]

    # 美股数据源走 yfinance，国内网络通常需要代理；容器内应留空
    http_proxy: str | None = None

    # 缓存（进程内 TTL + LRU）。数据源短 TTL 保新鲜度，LLM/embedding 长 TTL 做确定性记忆化
    cache_enabled: bool = True
    cache_maxsize: int = 512
    cache_ttl_seconds: float = 1800.0
    cache_llm_ttl_seconds: float = 86400.0

    # LLM 价格（每 1000 token）。留空则只统计 token，不换算金额
    llm_price_input_per_1k: float | None = None
    llm_price_output_per_1k: float | None = None

    @field_validator("llm_price_input_per_1k", "llm_price_output_per_1k", mode="before")
    @classmethod
    def _blank_price_means_unset(cls, value: object) -> object:
        """.env 里写成 `LLM_PRICE_INPUT_PER_1K=` 这种空值应当视为未配置，而不是启动报错。"""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    @property
    def resolved_log_format(self) -> str:
        if self.log_format:
            return self.log_format
        return "json" if self.is_prod else "text"


@functools.lru_cache
def get_settings() -> Settings:
    """进程内单例。测试或脚本可在改环境变量后调用 get_settings.cache_clear()。"""
    return Settings()
