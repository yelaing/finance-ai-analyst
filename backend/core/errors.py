"""错误码与业务异常。客户端可以只依据 code 分支处理，不必解析文案。"""

from enum import StrEnum

from pydantic import BaseModel


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
