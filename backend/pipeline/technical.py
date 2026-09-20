import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TechnicalData:
    ma_5: Optional[float] = None
    ma_20: Optional[float] = None
    ma_60: Optional[float] = None
    macd_dif: Optional[float] = None
    macd_dea: Optional[float] = None
    macd_bar: Optional[float] = None
    rsi_14: Optional[float] = None
    kdj_k: Optional[float] = None
    kdj_d: Optional[float] = None
    kdj_j: Optional[float] = None
    price_history: list[dict] = field(default_factory=list)


def _calc_ma(close: pd.Series, period: int) -> Optional[float]:
    if len(close) >= period:
        return round(float(close.rolling(window=period).mean().iloc[-1]), 2)
    return None


def _calc_macd(close: pd.Series):
    if len(close) < 26:
        return None, None, None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    bar = 2 * (dif - dea)
    return round(float(dif.iloc[-1]), 4), round(float(dea.iloc[-1]), 4), round(float(bar.iloc[-1]), 4)


def _calc_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def _calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9):
    if len(close) < period:
        return None, None, None
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return round(float(k.iloc[-1]), 2), round(float(d.iloc[-1]), 2), round(float(j.iloc[-1]), 2)


def _build_summary(td: TechnicalData, close: float) -> str:
    parts = [f"最新收盘价: {close}"]

    if td.ma_5 and td.ma_20:
        if close > td.ma_20:
            parts.append(f"价格位于 MA20 ({td.ma_20}) 上方，短期均线偏多")
        else:
            parts.append(f"价格位于 MA20 ({td.ma_20}) 下方，短期均线偏空")
        if td.ma_5 > td.ma_20:
            parts.append("MA5 上穿 MA20，短期金叉信号")
        elif td.ma_5 < td.ma_20:
            parts.append("MA5 低于 MA20，短期死叉信号")

    if td.macd_dif is not None and td.macd_dea is not None:
        if td.macd_dif > td.macd_dea:
            parts.append(f"MACD DIF ({td.macd_dif}) > DEA ({td.macd_dea})，多头排列")
        else:
            parts.append(f"MACD DIF ({td.macd_dif}) < DEA ({td.macd_dea})，空头排列")

    if td.rsi_14 is not None:
        if td.rsi_14 > 70:
            parts.append(f"RSI(14)={td.rsi_14}，处于超买区域")
        elif td.rsi_14 < 30:
            parts.append(f"RSI(14)={td.rsi_14}，处于超卖区域")
        else:
            parts.append(f"RSI(14)={td.rsi_14}，处于中性区间")

    if td.kdj_k is not None and td.kdj_d is not None:
        if td.kdj_k > td.kdj_d:
            parts.append(f"KDJ K({td.kdj_k}) > D({td.kdj_d})，短期动能偏强")
        else:
            parts.append(f"KDJ K({td.kdj_k}) < D({td.kdj_d})，短期动能偏弱")

    return "；".join(parts) + "。"


def calculate_indicators(symbol: str, market: str) -> Optional[TechnicalData]:
    try:
        if market == "a_share":
            return _calc_a_share(symbol)
        elif market == "us":
            return _calc_us(symbol)
    except Exception as e:
        logger.warning("Technical indicator calculation failed for %s: %s", symbol, e)
    return None


def _calc_a_share(symbol: str) -> TechnicalData:
    # A 股用国内站点，不走代理，避免 Clash 干扰
    import os
    import akshare as ak
    old_no_proxy = os.environ.get("NO_PROXY", "")
    os.environ["NO_PROXY"] = "*"
    try:
        exch = "sh" if symbol.startswith("6") else "sz"
        full_symbol = f"{exch}{symbol}"
        df = ak.stock_zh_a_daily(symbol=full_symbol, adjust="qfq")
    finally:
        if old_no_proxy:
            os.environ["NO_PROXY"] = old_no_proxy
        else:
            os.environ.pop("NO_PROXY", None)

    if df is None or df.empty:
        raise ValueError(f"No K-line data for {symbol}")

    df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"})
    return _compute_indicators(df)


def _calc_us(symbol: str) -> TechnicalData:
    import os
    import yfinance as yf
    # 确保走代理访问 Yahoo Finance
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="6mo")
    if df is None or df.empty:
        raise ValueError(f"No K-line data for {symbol}")

    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    return _compute_indicators(df)


def _compute_indicators(df: pd.DataFrame) -> TechnicalData:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    last_close = round(float(close.iloc[-1]), 2)

    dif, dea, bar = _calc_macd(close)
    k, d, j = _calc_kdj(high, low, close)

    td = TechnicalData(
        ma_5=_calc_ma(close, 5),
        ma_20=_calc_ma(close, 20),
        ma_60=_calc_ma(close, 60),
        macd_dif=dif,
        macd_dea=dea,
        macd_bar=bar,
        rsi_14=_calc_rsi(close),
        kdj_k=k,
        kdj_d=d,
        kdj_j=j,
    )

    td.price_history = [
        {"date": str(row["date"]), "open": row["open"], "high": row["high"],
         "low": row["low"], "close": row["close"]}
        for _, row in df.tail(60).iterrows()
    ]

    return td
