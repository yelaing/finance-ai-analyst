import logging
import re
import time
from dataclasses import dataclass, field

from backend.core.cache import FETCH, cache_key, get_cache
from backend.core.errors import InvalidSymbolError, UnsupportedMarketError
from backend.core.metrics import get_metrics

logger = logging.getLogger(__name__)

# 各市场的「必需数据类别」：报告必需字段依赖它们 —— name 来自公司概况、
# 基本面分析来自财报。新闻/公告只喂**可选的**舆情阶段，缺失是合法状态
# （公司真的没发公告），所以不在必需之列。
#
# 每个类别是可接受源名的集合：有主源也有备源，任一成功即满足该类别。
_REQUIRED_SOURCE_GROUPS: dict[str, tuple[frozenset[str], ...]] = {
    "a_share": (
        frozenset({"东方财富个股信息", "巨潮资讯公司概况"}),
        frozenset({"同花顺财务摘要"}),
    ),
    "us": (frozenset({"Yahoo Finance"}),),
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


@dataclass
class _QueryResult:
    """单个数据类别的查询结果。

    `source` 只在**调用成功**时给出，与是否返回内容无关 —— 这个区分是健康度判据
    与缓存正确性的基础。`name` 用于公司概况带回股票简称。
    """

    source: str | None
    lines: list[str] = field(default_factory=list)
    name: str | None = None


def _is_healthy(market: str, result: FetchResult) -> bool:
    """所有必需类别都至少有一个源**成功查询过**。

    判据是「查询成功」而不是「返回了内容」：源正常返回但数据为空（公司真没发公告）
    属于真实状态，不该判为降级 —— 否则那些日子缓存永远不会生效。

    与 sources 配套的空/失败编码（缓存里靠这两者组合消歧）：
      成功且有数据 → 源名在 sources，文本是实际内容
      成功但为空   → 源名在 sources，文本是「暂无…」
      查询失败     → 源名不在 sources，文本是「[xxx失败]」
    """
    queried = set(result.sources)
    return all(queried & group for group in _REQUIRED_SOURCE_GROUPS[market])


def detect_market(symbol: str) -> str:
    """Auto-detect market from symbol pattern."""
    if re.match(r"^\d{5,6}$", symbol):
        return "a_share"
    if re.match(r"^[A-Za-z]{1,5}$", symbol):
        return "us"
    raise InvalidSymbolError(f"无法自动识别 {symbol} 的市场，请手动指定 market 参数")


def _company_overview(symbol: str) -> _QueryResult:
    """公司概况：东方财富为主源，不可达时退到巨潮。

    巨潮只给公司全称/简称一类的静态信息，没有行业与估值 —— 但东财那条路在这台
    机器上本来就一直不可达（主机不通，实测直连与走代理都失败），所以是纯增益。
    """
    try:
        import akshare as ak

        stock_df = ak.stock_individual_info_em(symbol=symbol)
        info_dict = dict(zip(stock_df["item"], stock_df["value"], strict=False))
        lines = [
            f"公司全称：{info_dict.get('公司名称', 'N/A')}",
            f"行业板块：{info_dict.get('行业', 'N/A')}",
            f"总市值：{info_dict.get('总市值', 'N/A')}",
            f"流通市值：{info_dict.get('流通市值', 'N/A')}",
            f"市盈率(动态)：{info_dict.get('市盈率-动态', 'N/A')}",
            f"市净率：{info_dict.get('市净率', 'N/A')}",
            f"营业收入：{info_dict.get('营业收入', 'N/A')}",
            f"营业利润：{info_dict.get('营业利润', 'N/A')}",
            f"净利润：{info_dict.get('净利润', 'N/A')}",
            f"总股本：{info_dict.get('总股本', 'N/A')}",
            f"每股净资产：{info_dict.get('每股净资产', 'N/A')}",
            f"每股收益：{info_dict.get('每股收益', 'N/A')}",
            f"总资产：{info_dict.get('总资产', 'N/A')}",
        ]
        return _QueryResult("东方财富个股信息", lines, info_dict.get("股票简称"))
    except Exception as exc:
        logger.warning("A股个股信息失败，改用巨潮 symbol=%s: %s", symbol, exc)

    try:
        import akshare as ak

        profile = ak.stock_profile_cninfo(symbol=symbol)
        if profile is None or profile.empty:
            return _QueryResult("巨潮资讯公司概况", [])
        row = profile.iloc[0]
        lines = [
            f"公司全称：{row.get('公司名称', 'N/A')}",
            f"英文名称：{row.get('英文名称', 'N/A')}",
            f"所属行业：{row.get('所属行业', 'N/A')}",
            f"注册地址：{row.get('注册地址', 'N/A')}",
            f"主营业务：{row.get('主营业务', 'N/A')}",
        ]
        return _QueryResult("巨潮资讯公司概况", lines, row.get("A股简称"))
    except Exception as exc:
        logger.warning("A股公司概况两个源都失败 symbol=%s: %s", symbol, exc)
        return _QueryResult(None)


def _financial_abstract(symbol: str) -> _QueryResult:
    """同花顺财务摘要。调用成功就记源名，即使这一期没有数据。"""
    try:
        import akshare as ak

        fin_df = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
    except Exception as exc:
        logger.warning("A股财务摘要失败 symbol=%s: %s", symbol, exc)
        return _QueryResult(None)

    if fin_df is None or fin_df.empty:
        return _QueryResult("同花顺财务摘要", [])

    latest = fin_df.iloc[-1]
    lines = ["\n--- 最新财务摘要 ---", f"报告期：{fin_df.index[-1]}"]
    for col in fin_df.columns[:15]:
        val = latest.get(col)
        if val is not None and str(val) not in ("nan", "None", ""):
            lines.append(f"{col}：{val}")
    return _QueryResult("同花顺财务摘要", lines)


def _strip_company_prefix(title: str, display_name: str) -> str:
    """去掉巨潮公告标题里的「公司名:」前缀。

    头条标题形如「贵州茅台:贵州茅台关于召开…」，公司名重复两遍，白占 prompt token。
    """
    prefix = f"{display_name}:"
    if display_name and title.startswith(prefix):
        return title[len(prefix) :].strip()
    return title


def _company_news(symbol: str, display_name: str) -> _QueryResult:
    """个股动态：东方财富新闻为主源，解析失败时退到巨潮个股公告。

    东财新闻接口在 akshare 1.18.97 上有 pyarrow 解析 bug（实测升级后仍在），
    而公告是公司自己的动态（业绩说明会/董事会决议/分红），比全市快讯贴切。
    """
    try:
        import akshare as ak

        news_df = ak.stock_news_em(symbol=symbol)
        if news_df is not None and not news_df.empty:
            lines = [
                f"- [{row.get('发布时间', '')}] {row.get('标题', '')}"
                for _, row in news_df.head(15).iterrows()
            ]
            return _QueryResult("东方财富新闻", lines)
        return _QueryResult("东方财富新闻", [])
    except Exception as exc:
        logger.warning("A股新闻失败，改用巨潮公告 symbol=%s: %s", symbol, exc)

    try:
        import akshare as ak

        notice_df = ak.stock_individual_notice_report(security=symbol)
        if notice_df is None or notice_df.empty:
            return _QueryResult("巨潮资讯个股公告", [])
        lines = [
            f"- [{row.get('公告日期', '')}] "
            f"{_strip_company_prefix(str(row.get('公告标题', '')), display_name)}"
            f"（{row.get('公告类型', '')}）"
            for _, row in notice_df.head(15).iterrows()
        ]
        return _QueryResult("巨潮资讯个股公告", lines)
    except Exception as exc:
        logger.warning("A股个股动态两个源都失败 symbol=%s: %s", symbol, exc)
        return _QueryResult(None)


def _fetch_a_share(symbol: str) -> FetchResult:
    info = StockInfo(symbol=symbol, name=symbol, market="a_share")
    financial_parts: list[str] = []
    news_parts: list[str] = []
    sources: list[str] = []

    overview = _company_overview(symbol)
    if overview.source:
        sources.append(overview.source)
    if overview.name:
        info.name = overview.name
    if overview.lines:
        financial_parts.extend(overview.lines)
    else:
        financial_parts.append("[个股信息获取失败：东方财富与巨潮均不可用]")

    financials = _financial_abstract(symbol)
    if financials.source:
        sources.append(financials.source)
    financial_parts.extend(financials.lines)

    news = _company_news(symbol, info.name)
    if news.source:
        sources.append(news.source)
    if news.lines:
        news_parts.extend(news.lines)
    elif news.source is None:
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


def fetch_stock_data(symbol: str, market: str = "auto") -> FetchResult:
    if market == "auto":
        market = detect_market(symbol)
    if market not in _REQUIRED_SOURCE_GROUPS:
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
    elapsed = time.perf_counter() - started
    healthy = _is_healthy(market, result)
    get_metrics().observe_fetch(market, healthy, elapsed)
    logger.info(
        "数据抓取完成",
        extra={
            "symbol": symbol,
            "market": market,
            "sources": result.sources,
            "duration_ms": round(elapsed * 1000, 1),
        },
    )

    if healthy:
        cache.set(FETCH, key, result)
    else:
        # 缓存降级结果等于把一次偶发故障固化成半小时的持续降级，比不缓存更糟
        logger.warning(
            "数据源部分失败，结果不写入缓存",
            extra={"symbol": symbol, "market": market, "sources": result.sources},
        )
    return result
