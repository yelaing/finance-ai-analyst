"""统一日志：单行文本 / JSON 行两种格式，每条记录自动带上当前请求的 request_id。

业务代码只需要 logger.info("...", extra={"duration_ms": 12})，
两种格式各自决定怎么把附加字段呈现出来。
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.config import Settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# LogRecord 自带属性，不算「附加字段」
_STANDARD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "request_id",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

# 这些库在 INFO 级别会淹没业务日志
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "chromadb", "chromadb.telemetry")


class RequestIdFilter(logging.Filter):
    """把当前请求的 request_id 注入每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}


class TextFormatter(logging.Formatter):
    """给人看的格式，附加字段以 key=value 追加在行尾。"""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-5s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extra = _extra_fields(record)
        if extra:
            line += "  " + " ".join(f"{k}={v}" for k, v in extra.items())
        return line


class JsonFormatter(logging.Formatter):
    """给日志采集看的格式，每行一个合法 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(settings: "Settings") -> None:
    """安装唯一一个 StreamHandler，替换掉 logging 的默认配置。"""
    formatter: logging.Formatter = (
        JsonFormatter() if settings.resolved_log_format == "json" else TextFormatter()
    )

    # 日志里全是中文。Windows 管道默认 cp936、容器常见 POSIX/ASCII，两种情况都会
    # 让中文日志乱码甚至抛 UnicodeEncodeError，所以强制 stdout 用 UTF-8
    stdout = sys.stdout
    if hasattr(stdout, "reconfigure"):
        stdout.reconfigure(encoding="utf-8")

    handler = logging.StreamHandler(stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    noisy_level = logging.INFO if settings.env == "dev" else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)
