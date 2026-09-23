import json
import re
from typing import Optional
from dataclasses import dataclass, field


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
    raise ValueError(f"无法自动识别 {symbol} 的市场，请手动指定 market 参数")


def _fetch_a_share(symbol: str) -> FetchResult:
    info = StockInfo(symbol=symbol, name=symbol, market="a_share")
    financial_parts = []
    news_parts = []
    sources = []

    try:
        import akshare as ak

        stock_df = ak.stock_individual_info_em(symbol=symbol)
        info_dict = dict(zip(stock_df["item"], stock_df["value"]))
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
    except Exception:
        pass

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
    except Exception:
        news_parts.append("[新闻获取失败]")

    return FetchResult(
        info=info,
        financial_text="\n".join(financial_parts) or "暂无财务数据",
        news_text="\n".join(news_parts) or "暂无近期新闻",
        sources=sources,
    )


def _fetch_us(symbol: str) -> FetchResult:
    import os

    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")

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
        except Exception:
            pass

        try:
            news_list = ticker.news or []
            for n in news_list[:15]:
                title = n.get("title", "")
                provider = n.get("publisher", "")
                news_parts.append(f"- [{provider}] {title}")
            sources.append("Yahoo Finance News")
        except Exception:
            news_parts.append("[新闻获取失败]")

    except Exception as e:
        financial_parts.append(f"[数据获取失败: {e}]")

    return FetchResult(
        info=info,
        financial_text="\n".join(financial_parts) or "暂无财务数据",
        news_text="\n".join(news_parts) or "暂无近期新闻",
        sources=sources,
    )


def fetch_stock_data(symbol: str, market: str = "auto") -> FetchResult:
    if market == "auto":
        market = detect_market(symbol)
    if market == "a_share":
        return _fetch_a_share(symbol)
    if market == "us":
        return _fetch_us(symbol)
    raise ValueError(f"不支持的市场类型: {market}")
