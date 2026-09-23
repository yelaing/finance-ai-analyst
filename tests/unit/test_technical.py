import math

import pandas as pd
import pytest

from backend.pipeline import technical as t

FLAT = pd.Series([100.0] * 40)
RISING = pd.Series([float(i) for i in range(1, 40)])
FALLING = pd.Series([float(40 - i) for i in range(1, 40)])


def make_df(rows: int = 70) -> pd.DataFrame:
    """构造一段合成 K 线：价格缓慢上行，高低价包住收盘价。"""
    closes = [100.0 + i * 0.5 for i in range(rows)]
    return pd.DataFrame(
        {
            "date": [f"2026-01-{i + 1:02d}" for i in range(rows)],
            "open": [c - 0.3 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000 + i for i in range(rows)],
        }
    )


# ---------- MA ----------


def test_calc_ma_hand_computed():
    assert t._calc_ma(pd.Series([1.0, 2, 3, 4, 5]), 5) == 3.0
    assert t._calc_ma(pd.Series([1.0, 2, 3, 4, 6]), 5) == 3.2


def test_calc_ma_returns_none_when_not_enough_data():
    assert t._calc_ma(pd.Series([1.0, 2, 3, 4]), 5) is None


# ---------- MACD ----------


def test_calc_macd_flat_series_is_all_zero():
    """价格恒定 → 两条 EMA 都等于该常数 → DIF/DEA/BAR 全为 0（可手工验证）。"""
    assert t._calc_macd(FLAT) == (0.0, 0.0, 0.0)


def test_calc_macd_rising_series_is_positive():
    dif, dea, bar = t._calc_macd(RISING)
    assert dif is not None and dea is not None and bar is not None
    assert dif > 0 and dea > 0
    # BAR 的定义是 2*(DIF-DEA)
    assert bar == pytest.approx(2 * (dif - dea), abs=1e-4)


def test_calc_macd_returns_none_tuple_when_too_short():
    assert t._calc_macd(pd.Series([1.0] * 10)) == (None, None, None)


# ---------- RSI ----------


def test_calc_rsi_all_falling_is_zero():
    assert t._calc_rsi(FALLING) == 0.0


def test_calc_rsi_returns_none_when_too_short():
    assert t._calc_rsi(pd.Series([1.0] * 10)) is None


def test_calc_rsi_stays_in_range_for_mixed_series():
    close = pd.Series([10.0 + (i % 5) - 2 for i in range(40)])
    rsi = t._calc_rsi(close)
    assert rsi is not None and 0.0 <= rsi <= 100.0


def test_calc_rsi_on_monotonic_rise_is_nan_known_defect():
    """刻画现状：全程无下跌时 RSI 应为 100，实际返回 NaN。

    成因：`avg_loss.replace(0, np.nan)` 让 rs 变 NaN，进而污染 RSI。
    连涨 15 个交易日以上的股票会触达，届时报告里会写成「RSI(14)=nan，处于中性区间」。
    这是既有缺陷，不在「测试与 CI」切片范围内 —— 修好它时必须同步更新本断言。
    """
    assert math.isnan(t._calc_rsi(RISING))


# ---------- KDJ ----------


def test_calc_kdj_returns_none_tuple_when_too_short():
    assert t._calc_kdj(pd.Series([1.0] * 5), pd.Series([1.0] * 5), pd.Series([1.0] * 5)) == (
        None,
        None,
        None,
    )


def test_calc_kdj_j_is_derived_from_k_and_d():
    high = pd.Series([10.0, 12, 11, 15, 14, 16, 13, 18, 17, 19, 20, 21])
    low = pd.Series([5.0, 6, 4, 7, 8, 6, 9, 10, 8, 11, 12, 13])
    close = (high + low) / 2
    k, d, j = t._calc_kdj(high, low, close)

    assert k is not None and d is not None and j is not None
    # J = 3K - 2D 是定义式；K/D 各保留两位小数，故留 0.05 容差
    assert j == pytest.approx(3 * k - 2 * d, abs=0.05)
    assert 0.0 <= k <= 100.0 and 0.0 <= d <= 100.0


# ---------- _build_summary 分支 ----------


def test_build_summary_always_starts_with_close_price():
    summary = t._build_summary(t.TechnicalData(), 123.45)
    assert summary.startswith("最新收盘价: 123.45")
    assert summary.endswith("。")


def test_build_summary_above_ma20_and_golden_cross():
    td = t.TechnicalData(ma_5=12.0, ma_20=10.0)
    summary = t._build_summary(td, 15.0)
    assert "位于 MA20 (10.0) 上方" in summary
    assert "偏多" in summary
    assert "金叉" in summary


def test_build_summary_below_ma20_and_death_cross():
    td = t.TechnicalData(ma_5=8.0, ma_20=10.0)
    summary = t._build_summary(td, 9.0)
    assert "位于 MA20 (10.0) 下方" in summary
    assert "偏空" in summary
    assert "死叉" in summary


@pytest.mark.parametrize(
    ("dif", "dea", "expected"),
    [(1.0, 0.5, "多头排列"), (0.5, 1.0, "空头排列")],
)
def test_build_summary_macd_branches(dif, dea, expected):
    assert expected in t._build_summary(t.TechnicalData(macd_dif=dif, macd_dea=dea), 10.0)


@pytest.mark.parametrize(("rsi", "expected"), [(75.0, "超买"), (25.0, "超卖"), (50.0, "中性区间")])
def test_build_summary_rsi_branches(rsi, expected):
    assert expected in t._build_summary(t.TechnicalData(rsi_14=rsi), 10.0)


@pytest.mark.parametrize(("k", "d", "expected"), [(30.0, 20.0, "偏强"), (20.0, 30.0, "偏弱")])
def test_build_summary_kdj_branches(k, d, expected):
    summary = t._build_summary(t.TechnicalData(kdj_k=k, kdj_d=d), 10.0)
    assert "动能" in summary and expected in summary


def test_build_summary_skips_missing_indicators():
    summary = t._build_summary(t.TechnicalData(), 10.0)
    assert "MA" not in summary
    assert "MACD" not in summary
    assert "RSI" not in summary
    assert "KDJ" not in summary


def test_build_summary_says_nothing_about_crossing_when_mas_are_equal():
    summary = t._build_summary(t.TechnicalData(ma_5=10.0, ma_20=10.0), 10.0)
    assert "MA20" in summary
    assert "金叉" not in summary and "死叉" not in summary


def test_build_summary_skips_ma_clause_when_only_one_ma_present():
    summary = t._build_summary(t.TechnicalData(ma_5=12.0), 15.0)
    assert "MA20" not in summary


# ---------- calculate_indicators 分派与容错 ----------


def test_calculate_indicators_dispatches_to_a_share(monkeypatch):
    sentinel = t.TechnicalData(ma_5=1.0)
    monkeypatch.setattr(t, "_calc_a_share", lambda symbol: sentinel)
    assert t.calculate_indicators("600519", "a_share") is sentinel


def test_calculate_indicators_dispatches_to_us(monkeypatch):
    sentinel = t.TechnicalData(ma_5=2.0)
    monkeypatch.setattr(t, "_calc_us", lambda symbol: sentinel)
    assert t.calculate_indicators("AAPL", "us") is sentinel


def test_calculate_indicators_returns_none_for_unknown_market():
    assert t.calculate_indicators("600519", "mars") is None


def test_calculate_indicators_swallows_errors(monkeypatch, caplog):
    def boom(symbol):
        raise RuntimeError("K 线拉取失败")

    monkeypatch.setattr(t, "_calc_a_share", boom)
    assert t.calculate_indicators("600519", "a_share") is None
    assert "Technical indicator calculation failed" in caplog.text


# ---------- _compute_indicators ----------


def test_compute_indicators_fills_indicators_and_keeps_last_60_days():
    td = t._compute_indicators(make_df(70))
    assert td.ma_5 is not None and td.ma_20 is not None and td.ma_60 is not None
    assert td.macd_dif is not None and td.rsi_14 is not None and td.kdj_k is not None
    assert len(td.price_history) == 60
    assert set(td.price_history[0]) == {"date", "open", "high", "low", "close"}
    # tail(60) 取的是最后一段，最后一根应等于输入的最后一行
    assert td.price_history[-1]["close"] == pytest.approx(100.0 + 69 * 0.5)


def test_compute_indicators_handles_short_history():
    td = t._compute_indicators(make_df(5))
    assert td.ma_5 is not None
    assert td.ma_60 is None  # 不足 60 天
    assert td.macd_dif is None  # 不足 26 天
    assert len(td.price_history) == 5
