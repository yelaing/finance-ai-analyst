"""错误码与业务异常。客户端可以只依据 code 分支处理，不必解析文案。"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    UNSUPPORTED_MARKET = "UNSUPPORTED_MARKET"
    NOT_FOUND = "NOT_FOUND"
    REPORT_NOT_FOUND = "REPORT_NOT_FOUND"
    DATA_SOURCE_ERROR = "DATA_SOURCE_ERROR"
    LLM_ERROR = "LLM_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """业务异常基类：自带错误码与 HTTP 状态码，由全局异常处理器统一输出。

    子类只需要覆盖 code / status_code，不必各自处理响应格式。
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = 500
    default_message: str = "服务器内部错误"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class InvalidSymbolError(AppError):
    """无法识别或不被支持的股票代码。"""

    code = ErrorCode.INVALID_SYMBOL
    status_code = 400


class UnsupportedMarketError(AppError):
    """market 参数取值非法。"""

    code = ErrorCode.UNSUPPORTED_MARKET
    status_code = 400


class InvalidRequestError(AppError):
    """其它参数类错误（含存量代码抛出的 ValueError）。"""

    code = ErrorCode.INVALID_REQUEST
    status_code = 400


class ReportNotFoundError(AppError):
    """该股票还没有任何已存储的分析报告。"""

    code = ErrorCode.REPORT_NOT_FOUND
    status_code = 404


class ErrorResponse(BaseModel):
    """统一错误响应体。保留 FastAPI 原生的 detail 键，另外补上 code 与 request_id。"""

    detail: str
    code: str
    request_id: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detail": "无法自动识别 !!!bad 的市场，请手动指定 market 参数",
                    "code": "INVALID_SYMBOL",
                    "request_id": "3f9a1c2e",
                }
            ]
        }
    )


_ERROR_DESCRIPTIONS: dict[int, str] = {
    400: "参数不合法（INVALID_SYMBOL / UNSUPPORTED_MARKET / INVALID_REQUEST）",
    404: "资源不存在（NOT_FOUND / REPORT_NOT_FOUND）",
    422: "请求校验失败（VALIDATION_ERROR）",
    500: "服务器内部错误（INTERNAL_ERROR），响应体不含内部细节",
}


def error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    """生成挂到端点 `responses=` 上的声明，让 /docs 展示统一的错误形状。

    这些只是文档：运行时的错误体由 backend/api/errors.py 的处理器产出。
    两者必须一致，所以响应体模型与错误码都取自本模块，避免两处各写一份。
    """
    return {
        code: {
            "model": ErrorResponse,
            "description": _ERROR_DESCRIPTIONS.get(code, "错误"),
        }
        for code in codes
    }
