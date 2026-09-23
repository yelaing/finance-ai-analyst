import json
import logging
import sys

import pytest

from backend.config import Settings
from backend.core.logging import (
    JsonFormatter,
    RequestIdFilter,
    TextFormatter,
    request_id_var,
    setup_logging,
)


@pytest.fixture(autouse=True)
def restore_logging():
    """setup_logging 会改 root logger，测试完必须还原，免得污染其它测试。"""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="backend.demo",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="测试消息",
        args=(),
        exc_info=None,
    )
    RequestIdFilter().filter(record)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_text_formatter_shape():
    line = TextFormatter().format(make_record())
    assert "INFO" in line
    assert "backend.demo" in line
    assert "测试消息" in line
    assert "[-]" in line  # 无请求上下文时的默认 request_id


def test_text_formatter_appends_extra_as_key_value():
    line = TextFormatter().format(make_record(duration_ms=12, tokens=340))
    assert "duration_ms=12" in line
    assert "tokens=340" in line


def test_text_formatter_has_no_extra_suffix_when_no_extras():
    line = TextFormatter().format(make_record())
    assert "=" not in line.split("backend.demo:")[1]


def test_json_formatter_emits_valid_json_with_required_keys():
    payload = json.loads(JsonFormatter().format(make_record(duration_ms=7)))
    assert set(payload) >= {"ts", "level", "logger", "request_id", "msg"}
    assert payload["level"] == "INFO"
    assert payload["logger"] == "backend.demo"
    assert payload["msg"] == "测试消息"
    assert payload["duration_ms"] == 7


def test_json_formatter_includes_traceback():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = make_record()
        record.exc_info = sys.exc_info()
    payload = json.loads(JsonFormatter().format(record))
    assert "RuntimeError" in payload["exc"]
    assert "boom" in payload["exc"]


def test_json_formatter_tolerates_unserializable_extra():
    """附加字段可能是任意对象，序列化必须兜住而不是让日志本身抛异常。"""
    payload = json.loads(JsonFormatter().format(make_record(obj=object())))
    assert "obj" in payload


def test_request_id_filter_reads_contextvar_and_resets():
    assert request_id_var.get() == "-"
    token = request_id_var.set("abc12345")
    try:
        assert make_record().request_id == "abc12345"
    finally:
        request_id_var.reset(token)
    assert make_record().request_id == "-"


@pytest.mark.parametrize(("fmt", "expect_json"), [("text", False), ("json", True)])
def test_setup_logging_installs_exactly_one_handler(fmt, expect_json):
    """重复调用不能累积 handler —— 这是最容易漏掉的一类 bug。"""
    settings = Settings(_env_file=None, log_format=fmt, log_level="WARNING")
    setup_logging(settings)
    setup_logging(settings)

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING
    handler = root.handlers[0]
    assert any(isinstance(f, RequestIdFilter) for f in handler.filters)
    assert isinstance(handler.formatter, JsonFormatter if expect_json else TextFormatter)


def test_setup_logging_uses_env_derived_format_when_unspecified():
    setup_logging(Settings(_env_file=None, env="prod"))
    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)


def test_setup_logging_tolerates_stream_without_reconfigure(monkeypatch):
    """pytest 的捕获对象、某些部署 shim 没有 reconfigure，此时应静默跳过而非崩。"""

    class BareStream:
        def write(self, text): ...

        def flush(self): ...

    stream = BareStream()
    monkeypatch.setattr(sys, "stdout", stream)
    setup_logging(Settings(_env_file=None, log_format="text"))

    assert logging.getLogger().handlers[0].stream is stream


def test_setup_logging_quiets_noisy_loggers_in_prod_only():
    setup_logging(Settings(_env_file=None, log_format="text"))
    assert logging.getLogger("httpx").level == logging.INFO

    setup_logging(Settings(_env_file=None, env="prod"))
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("chromadb.telemetry").level == logging.WARNING
