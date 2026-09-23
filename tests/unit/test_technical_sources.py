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


def test_calc_a_share_bypasses_proxy_during_fetch(monkeypatch, proxyless):
    """A 股用国内站点，抓取时必须设 NO_PROXY=*，否则会被 Clash 拦成 ProxyError。"""
    module = fake_akshare(kline())
    monkeypatch.setitem(sys.modules, "akshare", module)
    t._calc_a_share("600519")

    assert module.calls[0]["no_proxy"] == "*"
    assert "NO_PROXY" not in os.environ  # 抓完必须还原


def test_calc_a_share_restores_previous_no_proxy(monkeypatch, proxyless):
    monkeypatch.setenv("NO_PROXY", "example.com")
    module = fake_akshare(kline())
    monkeypatch.setitem(sys.modules, "akshare", module)
    t._calc_a_share("600519")

    assert module.calls[0]["no_proxy"] == "*"
    assert os.environ["NO_PROXY"] == "example.com"


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
    monkeypatch.setattr(t, "_settings", types.SimpleNamespace(http_proxy=None))
    result = t._calc_us("AAPL")

    assert result.ma_5 is not None and result.ma_20 is not None
    assert len(result.price_history) == 60
    assert "close" in result.price_history[0]


def test_calc_us_sets_proxy_from_config(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(yahoo_frame(70)))
    monkeypatch.setattr(t, "_settings", types.SimpleNamespace(http_proxy="http://127.0.0.1:7897"))
    t._calc_us("AAPL")

    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7897"


def test_calc_us_does_not_set_proxy_when_unconfigured(monkeypatch, proxyless):
    """容器里没有代理，硬编码地址会把请求打到不存在的端口。"""
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(yahoo_frame(70)))
    monkeypatch.setattr(t, "_settings", types.SimpleNamespace(http_proxy=None))
    t._calc_us("AAPL")

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ


def test_calc_us_raises_when_no_data(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(pd.DataFrame()))
    monkeypatch.setattr(t, "_settings", types.SimpleNamespace(http_proxy=None))
    with pytest.raises(ValueError, match="No K-line data"):
        t._calc_us("AAPL")


def test_calculate_indicators_returns_none_when_us_fetch_fails(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(ConnectionError("Yahoo 挂了")))
    monkeypatch.setattr(t, "_settings", types.SimpleNamespace(http_proxy=None))
    assert t.calculate_indicators("AAPL", "us") is None
