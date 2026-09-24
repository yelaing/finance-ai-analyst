"""数据抓取层测试。

akshare / yfinance 通过注入假模块替换（它们在函数体内 import），
数据整理与降级逻辑本身真实执行。
"""

import logging
import sys
import types

import pandas as pd
import pytest

from backend.core.errors import InvalidSymbolError, UnsupportedMarketError
from backend.pipeline import data_fetcher as df

# ---------- 假数据源 ----------


def fake_akshare(*, info=True, financial=True, news=True):
    module = types.ModuleType("akshare")

    def stock_individual_info_em(symbol):
        if not info:
            raise ConnectionError("push2.eastmoney.com 不可达")
        return pd.DataFrame(
            {
                "item": ["股票简称", "公司名称", "行业"],
                "value": ["贵州茅台", "贵州茅台酒股份", "白酒"],
            }
        )

    def stock_financial_abstract_ths(symbol, indicator):
        if not financial:
            raise ValueError("同花顺接口异常")
        return pd.DataFrame(
            {"营业总收入": ["500亿"], "净利润": ["250亿"]},
            index=pd.Index(["2026-03-31"]),
        )

    def stock_news_em(symbol):
        if not news:
            raise ValueError("Invalid regular expression: invalid escape sequence: \\u")
        return pd.DataFrame({"标题": ["茅台发布一季报"], "发布时间": ["2026-04-01"]})

    module.stock_individual_info_em = stock_individual_info_em
    module.stock_financial_abstract_ths = stock_financial_abstract_ths
    module.stock_news_em = stock_news_em
    return module


class FakeTicker:
    def __init__(self, *, info=None, news=None, financials=None, raise_on_info=False):
        self._info = info
        self._news = news
        self._financials = financials
        self._raise = raise_on_info

    @property
    def info(self):
        if self._raise:
            raise ConnectionError("Yahoo 不可达")
        return self._info

    @property
    def quarterly_financials(self):
        return self._financials

    @property
    def news(self):
        return self._news


def fake_yfinance(ticker):
    module = types.ModuleType("yfinance")
    module.Ticker = lambda symbol: ticker
    return module


@pytest.fixture
def proxyless(monkeypatch):
    """清掉进程里可能已有的代理变量，让「是否设置代理」的断言不受环境干扰。"""
    for name in ("HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)


# ---------- detect_market ----------


@pytest.mark.parametrize("symbol", ["600519", "000001", "12345", "999999"])
def test_detect_market_a_share(symbol):
    assert df.detect_market(symbol) == "a_share"


@pytest.mark.parametrize("symbol", ["AAPL", "aapl", "TSLA", "F"])
def test_detect_market_us(symbol):
    assert df.detect_market(symbol) == "us"


@pytest.mark.parametrize("symbol", ["BRK.A", "!!!", "", "600519.SH", "1234567"])
def test_detect_market_rejects_unknown(symbol):
    with pytest.raises(InvalidSymbolError, match="无法自动识别"):
        df.detect_market(symbol)


# ---------- fetch_stock_data 分派 ----------


def test_fetch_stock_data_rejects_unknown_market():
    with pytest.raises(UnsupportedMarketError, match="不支持的市场类型: mars"):
        df.fetch_stock_data("600519", "mars")


def test_fetch_stock_data_autodetects_market(monkeypatch):
    sentinel = df.FetchResult(
        info=df.StockInfo("AAPL", "AAPL", "us"), financial_text="", news_text=""
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(FakeTicker(info={})))
    monkeypatch.setattr(df, "_fetch_us", lambda symbol: sentinel)
    assert df.fetch_stock_data("AAPL", "auto") is sentinel


def test_fetch_stock_data_rejects_bad_symbol_before_fetching(monkeypatch):
    def should_not_run(symbol):
        raise AssertionError("不该走到抓取")

    monkeypatch.setattr(df, "_fetch_a_share", should_not_run)
    with pytest.raises(InvalidSymbolError):
        df.fetch_stock_data("!!!", "auto")


# ---------- _fetch_a_share ----------


def test_fetch_a_share_happy_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare())
    result = df._fetch_a_share("600519")

    assert result.info.name == "贵州茅台"
    assert result.info.market == "a_share"
    assert result.sources == ["东方财富个股信息", "同花顺财务摘要", "东方财富新闻"]
    assert "公司全称：贵州茅台酒股份" in result.financial_text
    assert "行业板块：白酒" in result.financial_text
    assert "--- 最新财务摘要 ---" in result.financial_text
    assert "报告期：2026-03-31" in result.financial_text
    assert "茅台发布一季报" in result.news_text


@pytest.mark.parametrize(
    ("kwargs", "expect_text", "expect_source", "expect_log"),
    [
        ({"info": False}, "[个股信息获取失败", "东方财富个股信息", "A股个股信息获取失败"),
        ({"financial": False}, None, "同花顺财务摘要", "A股财务摘要获取失败"),
        ({"news": False}, None, "东方财富新闻", "A股新闻获取失败"),
    ],
)
def test_fetch_a_share_degrades_on_partial_failure(
    monkeypatch, caplog, kwargs, expect_text, expect_source, expect_log
):
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare(**kwargs))
    with caplog.at_level(logging.WARNING):
        result = df._fetch_a_share("600519")

    assert expect_source not in result.sources
    assert expect_log in caplog.text  # 失败必须留痕，不能静默吞掉
    if expect_text:
        assert expect_text in result.financial_text


def test_fetch_a_share_falls_back_when_news_is_empty(monkeypatch):
    module = fake_akshare()
    module.stock_news_em = lambda symbol: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "akshare", module)
    assert df._fetch_a_share("600519").news_text == "暂无近期新闻"


# ---------- _fetch_us ----------


def test_fetch_us_happy_path(monkeypatch, proxyless):
    ticker = FakeTicker(
        info={"longName": "Apple Inc.", "sector": "Technology", "trailingPE": 30.1},
        news=[{"title": "Apple ships", "publisher": "Reuters"}],
        financials=pd.DataFrame({"2026Q1": [1.0]}),
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(ticker))
    monkeypatch.setattr(df, "_settings", types.SimpleNamespace(http_proxy=None))
    result = df._fetch_us("AAPL")

    assert result.info.name == "Apple Inc."
    assert result.info.symbol == "AAPL"
    assert result.sources == ["Yahoo Finance", "Yahoo Finance News"]
    assert "公司名称：Apple Inc." in result.financial_text
    assert "Apple ships" in result.news_text


def test_fetch_us_falls_back_when_news_is_empty(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(FakeTicker(info={}, news=[])))
    monkeypatch.setattr(df, "_settings", types.SimpleNamespace(http_proxy=None))
    assert df._fetch_us("AAPL").news_text == "暂无近期新闻"


def test_fetch_us_degrades_when_source_unreachable(monkeypatch, proxyless, caplog):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(FakeTicker(raise_on_info=True)))
    monkeypatch.setattr(df, "_settings", types.SimpleNamespace(http_proxy=None))
    with caplog.at_level(logging.ERROR):
        result = df._fetch_us("AAPL")

    assert "[数据获取失败" in result.financial_text
    assert result.sources == []
    assert "美股数据获取失败" in caplog.text


def test_fetch_us_sets_proxy_from_config(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(FakeTicker(info={})))
    monkeypatch.setattr(df, "_settings", types.SimpleNamespace(http_proxy="http://127.0.0.1:7897"))
    df._fetch_us("AAPL")

    import os

    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7897"


def test_fetch_us_does_not_set_proxy_when_unconfigured(monkeypatch, proxyless):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(FakeTicker(info={})))
    monkeypatch.setattr(df, "_settings", types.SimpleNamespace(http_proxy=None))
    df._fetch_us("AAPL")

    import os

    assert "HTTP_PROXY" not in os.environ


HEALTHY_SOURCES = ["东方财富个股信息", "同花顺财务摘要", "东方财富新闻"]


def _result(*, symbol="600519", sources=None) -> df.FetchResult:
    return df.FetchResult(
        info=df.StockInfo(symbol, "贵州茅台", "a_share"),
        financial_text="财务数据",
        news_text="近期新闻",
        sources=HEALTHY_SOURCES if sources is None else sources,
    )


def test_fetch_stock_data_logs_structured_fields(monkeypatch, caplog):
    monkeypatch.setattr(df, "_fetch_a_share", lambda symbol: _result())
    with caplog.at_level(logging.INFO):
        df.fetch_stock_data("600519", "a_share")

    record = next(r for r in caplog.records if "数据抓取完成" in r.getMessage())
    # symbol / market / sources 现在走 extra 成为结构化字段，不再拼进消息文本
    assert record.symbol == "600519"
    assert record.market == "a_share"
    assert record.sources == HEALTHY_SOURCES
    assert record.duration_ms >= 0
    assert "贵州茅台" not in record.getMessage()


# ---------- 数据源缓存 ----------


def test_healthy_fetch_is_cached_and_second_call_skips_network(monkeypatch):
    calls = []

    def counting(symbol):
        calls.append(symbol)
        return _result()

    monkeypatch.setattr(df, "_fetch_a_share", counting)
    first = df.fetch_stock_data("600519", "a_share")
    second = df.fetch_stock_data("600519", "a_share")

    assert calls == ["600519"]  # 第二次没再抓
    assert second is first


def test_degraded_fetch_is_not_cached(monkeypatch, caplog):
    """缓存降级结果等于把一次偶发故障固化成半小时的持续降级。"""
    calls = []

    def partial(symbol):
        calls.append(symbol)
        return _result(sources=["同花顺财务摘要"])  # 三个源只成功一个

    monkeypatch.setattr(df, "_fetch_a_share", partial)
    with caplog.at_level(logging.WARNING):
        df.fetch_stock_data("600519", "a_share")
        df.fetch_stock_data("600519", "a_share")

    assert calls == ["600519", "600519"]  # 每次都重新抓
    assert "结果不写入缓存" in caplog.text


@pytest.mark.parametrize(
    ("sources", "healthy"),
    [
        (["东方财富个股信息", "同花顺财务摘要"], False),  # 缺新闻
        (["同花顺财务摘要", "东方财富新闻"], False),  # 缺个股信息
        ([], False),  # 全失败
        (["东方财富个股信息", "同花顺财务摘要", "东方财富新闻", "额外源"], True),
    ],
)
def test_health_requires_all_expected_sources(sources, healthy):
    """健康度用 sources 这个结构化信号判断，不嗅探文本标记 ——
    财务摘要失败时文本里根本不留标记。"""
    assert df._is_healthy("a_share", _result(sources=sources)) is healthy


def test_cache_hit_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(df, "_fetch_a_share", lambda symbol: _result())
    df.fetch_stock_data("600519", "a_share")
    with caplog.at_level(logging.INFO):
        df.fetch_stock_data("600519", "a_share")

    record = next(r for r in caplog.records if "命中缓存" in r.getMessage())
    assert record.cache_hit is True
    assert record.symbol == "600519"


def test_cache_is_keyed_per_symbol_and_market(monkeypatch):
    from backend.core.cache import FETCH, cache_key, get_cache

    monkeypatch.setattr(df, "_fetch_a_share", lambda symbol: _result(symbol=symbol))
    df.fetch_stock_data("600519", "a_share")
    assert get_cache().get(FETCH, cache_key("a_share", "600519")) is not None
    assert get_cache().get(FETCH, cache_key("a_share", "000001")) is None
    assert get_cache().get(FETCH, cache_key("us", "600519")) is None


def test_cache_disabled_always_fetches(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "false")
    from backend.config import get_settings
    from backend.core.cache import get_cache

    get_settings.cache_clear()
    get_cache.cache_clear()

    calls = []

    def counting(symbol):
        calls.append(symbol)
        return _result()

    monkeypatch.setattr(df, "_fetch_a_share", counting)
    df.fetch_stock_data("600519", "a_share")
    df.fetch_stock_data("600519", "a_share")
    assert calls == ["600519", "600519"]


# ---------- 边界：空数据与脏值 ----------


def test_fetch_a_share_skips_financial_block_when_empty(monkeypatch):
    module = fake_akshare()
    module.stock_financial_abstract_ths = lambda symbol, indicator: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "akshare", module)

    result = df._fetch_a_share("600519")
    assert "同花顺财务摘要" not in result.sources
    assert "--- 最新财务摘要 ---" not in result.financial_text


@pytest.mark.parametrize("dirty", ["nan", "None", "", None])
def test_fetch_a_share_filters_dirty_financial_values(monkeypatch, dirty):
    """同花顺偶尔会返回 nan / None / 空串，这些不该进 prompt。"""
    module = fake_akshare()
    module.stock_financial_abstract_ths = lambda symbol, indicator: pd.DataFrame(
        {"脏列": [dirty], "正常列": ["42"]}, index=pd.Index(["2026-03-31"])
    )
    monkeypatch.setitem(sys.modules, "akshare", module)

    text = df._fetch_a_share("600519").financial_text
    assert "正常列：42" in text
    assert f"脏列：{dirty}" not in text


def test_fetch_us_logs_quarterly_financial_failure(monkeypatch, proxyless, caplog):
    class Exploding:
        @property
        def info(self):
            return {}

        @property
        def quarterly_financials(self):
            raise ValueError("季度财报解析失败")

        @property
        def news(self):
            return []

    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(Exploding()))
    monkeypatch.setattr(df, "_settings", types.SimpleNamespace(http_proxy=None))
    with caplog.at_level(logging.WARNING):
        df._fetch_us("AAPL")

    assert "美股季度财务获取失败" in caplog.text
    assert "Yahoo Finance" in caplog.records[0].getMessage() or caplog.text


def test_fetch_us_logs_news_failure_and_falls_back(monkeypatch, proxyless, caplog):
    class ExplodingNews:
        @property
        def info(self):
            return {}

        @property
        def quarterly_financials(self):
            return None

        @property
        def news(self):
            raise KeyError("news 字段结构变了")

    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance(ExplodingNews()))
    monkeypatch.setattr(df, "_settings", types.SimpleNamespace(http_proxy=None))
    with caplog.at_level(logging.WARNING):
        result = df._fetch_us("AAPL")

    assert "美股新闻获取失败" in caplog.text
    assert result.news_text == "[新闻获取失败]"
