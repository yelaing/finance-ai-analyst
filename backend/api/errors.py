"""全局异常处理器：把各类异常收敛成同一种响应体（detail + code + request_id）。"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.core.errors import AppError, ErrorCode, ErrorResponse
from backend.core.logging import request_id_var

logger = logging.getLogger(__name__)


def _payload(detail: str, code: ErrorCode) -> dict[str, str]:
    return ErrorResponse(
        detail=detail, code=code.value, request_id=request_id_var.get()
    ).model_dump()


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    if exc.status_code >= 500:
        logger.exception("业务异常 code=%s: %s", exc.code.value, exc.message)
    return JSONResponse(status_code=exc.status_code, content=_payload(exc.message, exc.code))


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_REQUEST
    return JSONResponse(status_code=exc.status_code, content=_payload(str(exc.detail), code))


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # FastAPI 原生 detail 是结构化数组；这里压成一行可读文案，完整明细进日志，
    # 保证 detail 字段类型在两种格式之间保持稳定（始终是字符串）
    summary = "; ".join(
        f"{'.'.join(str(part) for part in err.get('loc', []))}: {err.get('msg', '')}"
        for err in exc.errors()
    )
    logger.warning("请求参数校验失败: %s", summary)
    return JSONResponse(
        status_code=422,
        content=_payload(summary or "请求参数校验失败", ErrorCode.VALIDATION_ERROR),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
