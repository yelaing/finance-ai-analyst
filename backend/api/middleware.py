"""请求上下文中间件：分配 request_id、记录访问日志与耗时。"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.core.errors import ErrorCode, ErrorResponse
from backend.core.logging import request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """每个请求分配（或沿用调用方传入的）request_id，并在响应头回传。

    未捕获异常在这里就地转成统一 500，而不是注册一个 Exception 异常处理器：
    FastAPI 的 Exception 处理器挂在 ServerErrorMiddleware 上，位于用户中间件的外层，
    走到那里时 ContextVar 已被重置，响应体里就拿不到 request_id 了。
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:8]
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            try:
                response = await call_next(request)
            except Exception:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                logger.exception(
                    "%s %s 未捕获异常",
                    request.method,
                    request.url.path,
                    extra={"duration_ms": elapsed_ms},
                )
                body = ErrorResponse(
                    detail="服务器内部错误",
                    code=ErrorCode.INTERNAL_ERROR.value,
                    request_id=request_id,
                )
                response = JSONResponse(status_code=500, content=body.model_dump())

            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.info(
                "%s %s %s",
                request.method,
                request.url.path,
                response.status_code,
                extra={"duration_ms": elapsed_ms},
            )
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            request_id_var.reset(token)
