import logging
import re
import time
from dataclasses import dataclass, field

from backend.core.cache import FETCH, cache_key, get_cache
from backend.core.errors import InvalidSymbolError, UnsupportedMarketError

logger = logging.getLogger(__name__)

# 各市场「成功抓取」应当产出的数据源。用 sources 这个结构化信号判断健康度，
# 而不是去嗅探文本里的失败标记 —— 财务摘要失败时文本里根本不留标记。
_EXPECTED_SOURCES = {
    "a_share": {"东方财富个股信息", "同花顺财务摘要", "东方财富新闻"},
    "us": {"Yahoo Finance", "Yahoo Finance News"},
}


@dataclass
class StockInfo:
    symbol: str
    name: str
    market: str  # "a_share" or "us"


@dataclass
class FetchResult:
    info: StockInfo
    financial_text: str
    news_text: str
    sources: list[str] = field(default_factory=list)


def detect_market(symbol: str) -> str:
    """Auto-detect market from symbol pattern."""
    if re.match(r"^\d{5,6}$", symbol):
        return "a_share"
    if re.match(r"^[A-Za-z]{1,5}$", symbol):
        return "us"
    raise InvalidSymbolError(f"无法自动识别 {symbol} 的市场，请手动指定 market 参数")


def _fetch_a_share(symbol: str) -> FetchResult:
    info = StockInfo(symbol=symbol, name=symbol, market="a_share")
    financial_parts = []
    news_parts = []
    sources = []

    try:
        import akshare as ak

        stock_df = ak.stock_individual_info_em(symbol=symbol)
        info_dict = dict(zip(stock_df["item"], stock_df["value"], strict=False))
        info.name = info_dict.get("股票简称", symbol)

        financial_parts.append(f"公司全称：{info_dict.get('公司名称', 'N/A')}")
        financial_parts.append(f"行业板块：{info_dict.get('行业', 'N/A')}")
        financial_parts.append(f"总市值：{info_dict.get('总市值', 'N/A')}")
        financial_parts.append(f"流通市值：{info_dict.get('流通市值', 'N/A')}")
        financial_parts.append(f"市盈率(动态)：{info_dict.get('市盈率-动态', 'N/A')}")
        financial_parts.append(f"市净率：{info_dict.get('市净率', 'N/A')}")
        financial_parts.append(f"营业收入：{info_dict.get('营业收入', 'N/A')}")
        financial_parts.append(f"营业利润：{info_dict.get('营业利润', 'N/A')}")
        financial_parts.append(f"净利润：{info_dict.get('净利润', 'N/A')}")
        financial_parts.append(f"总股本：{info_dict.get('总股本', 'N/A')}")
        financial_parts.append(f"每股净资产：{info_dict.get('每股净资产', 'N/A')}")
        financial_parts.append(f"每股收益：{info_dict.get('每股收益', 'N/A')}")
        financial_parts.append(f"总资产：{info_dict.get('总资产', 'N/A')}")
        sources.append("东方财富个股信息")

    except Exception as e:
        logger.warning("A股个股信息获取失败 symbol=%s: %s", symbol, e)
        financial_parts.append(f"[个股信息获取失败: {e}]")

    try:
        import akshare as ak

        fin_df = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
        if fin_df is not None and not fin_df.empty:
            latest = fin_df.iloc[-1]
            financial_parts.append("\n--- 最新财务摘要 ---")
            financial_parts.append(f"报告期：{fin_df.index[-1]}")
            for col in fin_df.columns[:15]:
                val = latest.get(col)
                if val is not None and str(val) not in ("nan", "None", ""):
                    financial_parts.append(f"{col}：{val}")
            sources.append("同花顺财务摘要")
    except Exception as e:
        logger.warning("A股财务摘要获取失败 symbol=%s: %s", symbol, e)

    try:
        import akshare as ak

        news_df = ak.stock_news_em(symbol=symbol)
        if news_df is not None and not news_df.empty:
            recent = news_df.head(15)
            for _, row in recent.iterrows():
                title = row.get("标题", "")
                time = row.get("发布时间", "")
                news_parts.append(f"- [{time}] {title}")
            sources.append("东方财富新闻")
    except Exception as e:
        logger.warning("A股新闻获取失败 symbol=%s: %s", symbol, e)
        news_parts.append("[新闻获取失败]")

    return FetchResult(
        info=info,
        financial_text="\n".join(financial_parts) or "暂无财务数据",
        news_text="\n".join(news_parts) or "暂无近期新闻",
        sources=sources,
    )


def _fetch_us(symbol: str) -> FetchResult:
    # 代理环境由 backend.core.net 在启动时统一设置；这里不再改动 os.environ，
    # 否则并发请求会读到被改过的全局变量
    info = StockInfo(symbol=symbol.upper(), name=symbol.upper(), market="us")
    financial_parts = []
    news_parts = []
    sources = []

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        stock_info = ticker.info or {}
        info.name = stock_info.get("longName") or stock_info.get("shortName") or symbol.upper()

        financial_parts.append(f"公司名称：{info.name}")
        financial_parts.append(f"行业：{stock_info.get('industry', 'N/A')}")
        financial_parts.append(f"板块：{stock_info.get('sector', 'N/A')}")
        financial_parts.append(f"市值：{stock_info.get('marketCap', 'N/A')}")
        financial_parts.append(f"市盈率：{stock_info.get('trailingPE', 'N/A')}")
        financial_parts.append(f"市净率：{stock_info.get('priceToBook', 'N/A')}")
        financial_parts.append(f"营收增长率：{stock_info.get('revenueGrowth', 'N/A')}")
        financial_parts.append(f"ROE：{stock_info.get('returnOnEquity', 'N/A')}")
        financial_parts.append(f"负债率：{stock_info.get('debtToEquity', 'N/A')}")
        financial_parts.append(f"每股收益：{stock_info.get('trailingEps', 'N/A')}")
        financial_parts.append(f"股息率：{stock_info.get('dividendYield', 'N/A')}")
        financial_parts.append(f"52周高：{stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
        financial_parts.append(f"52周低：{stock_info.get('fiftyTwoWeekLow', 'N/A')}")
        sources.append("Yahoo Finance")

        try:
            fin = ticker.quarterly_financials
            if fin is not None and not fin.empty:
                latest_q = fin.iloc[:, 0]
                financial_parts.append("\n--- 最新季度财务 ---")
                financial_parts.append(str(latest_q.to_dict()))
        except Exception as e:
            logger.warning("美股季度财务获取失败 symbol=%s: %s", symbol, e)

        try:
            news_list = ticker.news or []
            for n in news_list[:15]:
                title = n.get("title", "")
                provider = n.get("publisher", "")
                news_parts.append(f"- [{provider}] {title}")
            sources.append("Yahoo Finance News")
        except Exception as e:
            logger.warning("美股新闻获取失败 symbol=%s: %s", symbol, e)
            news_parts.append("[新闻获取失败]")

    except Exception as e:
        logger.error("美股数据获取失败 symbol=%s: %s", symbol, e)
        financial_parts.append(f"[数据获取失败: {e}]")

    return FetchResult(
        info=info,
        financial_text="\n".join(financial_parts) or "暂无财务数据",
        news_text="\n".join(news_parts) or "暂无近期新闻",
        sources=sources,
    )


def _is_healthy(market: str, result: FetchResult) -> bool:
    return _EXPECTED_SOURCES[market] <= set(result.sources)


def fetch_stock_data(symbol: str, market: str = "auto") -> FetchResult:
    if market == "auto":
        market = detect_market(symbol)
    if market not in _EXPECTED_SOURCES:
        raise UnsupportedMarketError(f"不支持的市场类型: {market}")

    cache = get_cache()
    key = cache_key(market, symbol)
    cached = cache.get(FETCH, key)
    if cached is not None:
        logger.info(
            "数据源命中缓存",
            extra={"cache_hit": True, "symbol": symbol, "market": market},
        )
        return cached

    started = time.perf_counter()
    result = _fetch_a_share(symbol) if market == "a_share" else _fetch_us(symbol)
    logger.info(
        "数据抓取完成",
        extra={
            "symbol": symbol,
            "market": market,
            "sources": result.sources,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )

    if _is_healthy(market, result):
        cache.set(FETCH, key, result)
    else:
        # 缓存降级结果等于把一次偶发故障固化成半小时的持续降级，比不缓存更糟
        logger.warning(
            "数据源部分失败，结果不写入缓存",
            extra={"symbol": symbol, "market": market, "sources": result.sources},
        )
    return result
