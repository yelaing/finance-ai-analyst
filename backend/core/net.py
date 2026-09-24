"""启动时**一次性**把代理相关的环境变量写入 os.environ。

为什么必须是"一次性"：requests / httpx / curl_cffi 都在**发起请求时**读这些变量。
历史上 technical.py 为了 A 股绕过代理，在请求线程里临时把 NO_PROXY 改成 "*"，
这在 FastAPI 的同步端点线程池下就是竞态 —— 并发的另一个请求（比如 yfinance）
会读到被改过的值、绕过代理然后失败。

实测依据（2026-09-24）：A 股的同花顺与新浪在**走系统代理**时都正常
（basic.10jqka.com.cn 0.6s / finance.sina.com.cn 0.9s），所以那个绕行已无必要，
直接删掉。确实需要绕过的场景，改在 .env 里配 NO_PROXY，由本模块在启动时应用。

结果：运行期对 os.environ 零写入，并发安全是**结构性成立**的，不靠加锁或纪律维持。
"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.config import Settings

logger = logging.getLogger(__name__)


def apply_network_env(settings: "Settings") -> None:
    """由应用入口（backend.main）在启动时调用一次。

    只设置有值的项：留空表示"不干预"，交给环境原本的配置（比如系统代理）。
    值本身可能是 http://user:pass@host 形式，所以不在 INFO 级别打印。
    """
    applied: list[str] = []
    for name, value in (
        ("HTTP_PROXY", settings.http_proxy),
        ("HTTPS_PROXY", settings.http_proxy),
        ("NO_PROXY", settings.no_proxy),
    ):
        if value:
            os.environ[name] = value
            applied.append(name)

    if applied:
        logger.info("已应用网络环境变量（仅启动时）: %s", ", ".join(applied))
        logger.debug(
            "网络环境变量取值: %s",
            {name: os.environ[name] for name in applied},
        )
