"""技术面数据抓取路径：_calc_a_share / _calc_us。

akshare / yfinance 以假模块注入，K 线整理与代理设置逻辑真实执行。
"""

import os
import sys
import types

import pandas as pd
import pytest

from backend.pipeline import technical as t


def kline(n: int = 70) -> pd.DataFrame:
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "date": [f"2026-01-{i + 1:02d}" for i in range(n)],
            "open": [c - 0.5 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000 + i for i in range(n)],
        }
    )


@pytest.fixture
def proxyless(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)


# ---------- _calc_a_share ----------


def fake_akshare(df):
    module = types.ModuleType("akshare")
    calls = []

    def stock_zh_a_daily(symbol, adjust):
        calls.append({"symbol": symbol, "adjust": adjust, "no_proxy": os.environ.get("NO_PROXY")})
        if isinstance(df, Exception):
            raise df
        return df

    module.stock_zh_a_daily = stock_zh_a_daily
    module.calls = calls
    return module


@pytest.mark.parametrize(("symbol", "expected"), [("600519", "sh600519"), ("000001", "sz000001")])
def test_calc_a_share_builds_prefixed_symbol(monkeypatch, proxyless, symbol, expected):
    module = fake_akshare(kline())
    monkeypatch.setitem(sys.modules, "akshare", module)
    t._calc_a_share(symbol)
    assert module.calls[0]["symbol"] == expected
    assert module.calls[0]["adjust"] == "qfq"


PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")


def env_snapshot() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in PROXY_VARS}


def test_calc_a_share_does_not_mutate_process_env(monkeypatch, proxyless):
    """抓取过程不得改动全局环境变量。

    历史上这里是临时把 NO_PROXY 改成 "*"，但那在多线程下是竞态：并发的另一个请求
    （比如 yfinance）会读到被改过的值、绕过代理然后失败。代理环境改由
    backend.core.net 在服务启动时一次性应用。
    """
    module = fake_akshare(kline())
    monkeypatch.setitem(sys.modules, "akshare", module)
    before = env_snapshot()

    t._calc_a_share("600519")

    assert env_snapshot() == before
    assert module.calls[0]["no_proxy"] is None  # 不再注入 NO_PROXY


def test_calc_a_share_returns_none_when_no_data(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare(pd.DataFrame()))
    with pytest.raises(ValueError, match="No K-line data"):
        t._calc_a_share("600519")


def test_calc_a_share_computes_indicators(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare(kline(70)))
    result = t._calc_a_share("600519")
    assert result.ma_5 is not None and result.ma_60 is not None
    assert len(result.price_history) == 60


# ---------- _calc_us ----------


class FakeTicker:
    def __init__(self, df):
        self._df = df

    def history(self, period):
        if isinstance(self._df, Exception):
            raise self._df
        return self._df


def fake_yfinance(df):
    module = types.ModuleType("yfinance")
    module.Ticker = lambda symbol: FakeTicker(df)
    return module


def yahoo_frame(n: int = 70) -> pd.DataFrame:
    """模仿 yfinance 的返回：DatetimeIndex 名为 Date，列名首字母大写。"""
    idx = pd.date_range("2026-01-01", periods=n, freq="D", name="Date")
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [c - 0.5 for c in closes],
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [1000 + i for i in range(n)],
        },
        index=idx,
    )


def test_calc_us_renames_yahoo_columns(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(yahoo_frame(70)))
    result = t._calc_us("AAPL")

    assert result.ma_5 is not None and result.ma_20 is not None
    assert len(result.price_history) == 60
    assert "close" in result.price_history[0]


def test_calc_us_does_not_mutate_process_env(monkeypatch, proxyless):
    """同 _calc_a_share：运行期不改全局环境变量，代理由启动时统一应用。"""
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(yahoo_frame(70)))
    before = env_snapshot()

    t._calc_us("AAPL")

    assert env_snapshot() == before


def test_calc_us_raises_when_no_data(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(pd.DataFrame()))
    with pytest.raises(ValueError, match="No K-line data"):
        t._calc_us("AAPL")


def test_calculate_indicators_returns_none_when_us_fetch_fails(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(ConnectionError("Yahoo 挂了")))
    assert t.calculate_indicators("AAPL", "us") is None
