import pytest

from backend.core.errors import (
    AppError,
    ErrorCode,
    ErrorResponse,
    InvalidRequestError,
    InvalidSymbolError,
    ReportNotFoundError,
    UnsupportedMarketError,
)


def test_error_code_is_str_enum():
    assert ErrorCode.INVALID_SYMBOL == "INVALID_SYMBOL"
    assert isinstance(ErrorCode.INVALID_SYMBOL, str)
    assert ErrorCode.INVALID_SYMBOL.value == "INVALID_SYMBOL"


def test_error_code_taxonomy_is_stable():
    """错误码是对外契约，改名或删项都属于破坏性变更，必须显式确认。"""
    assert {c.value for c in ErrorCode} == {
        "VALIDATION_ERROR",
        "INVALID_REQUEST",
        "INVALID_SYMBOL",
        "UNSUPPORTED_MARKET",
        "NOT_FOUND",
        "REPORT_NOT_FOUND",
        "DATA_SOURCE_ERROR",
        "LLM_ERROR",
        "INTERNAL_ERROR",
    }


def test_app_error_defaults_to_internal_error():
    exc = AppError()
    assert exc.code is ErrorCode.INTERNAL_ERROR
    assert exc.status_code == 500
    assert exc.message == "服务器内部错误"
    assert str(exc) == "服务器内部错误"


def test_app_error_custom_message():
    exc = AppError("分析失败，请稍后重试")
    assert exc.message == "分析失败，请稍后重试"
    assert str(exc) == "分析失败，请稍后重试"


@pytest.mark.parametrize(
    ("cls", "code", "status"),
    [
        (InvalidSymbolError, ErrorCode.INVALID_SYMBOL, 400),
        (UnsupportedMarketError, ErrorCode.UNSUPPORTED_MARKET, 400),
        (InvalidRequestError, ErrorCode.INVALID_REQUEST, 400),
        (ReportNotFoundError, ErrorCode.REPORT_NOT_FOUND, 404),
    ],
)
def test_subclasses_carry_code_and_status(cls, code, status):
    exc = cls("具体说明")
    assert isinstance(exc, AppError)
    assert exc.code is code
    assert exc.status_code == status
    assert exc.message == "具体说明"


def test_error_response_has_exactly_three_fields():
    body = ErrorResponse(detail="x", code="INVALID_SYMBOL", request_id="abc12345")
    assert body.model_dump() == {
        "detail": "x",
        "code": "INVALID_SYMBOL",
        "request_id": "abc12345",
    }
