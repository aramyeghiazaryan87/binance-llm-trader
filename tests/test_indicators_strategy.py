import pandas as pd
import pytest

from trader.indicators import atr, ema, rolling_high, rsi
from trader.market_data import ohlcv_to_frame
from trader.strategy import StrategyParams, compute_features, entry_conditions, entry_signal, evaluate_exit

from .conftest import make_candles


def frame(rows):
    return ohlcv_to_frame(rows, "1h")


def test_ema_of_constant_is_constant():
    s = pd.Series([5.0] * 50)
    assert ema(s, 10).dropna().eq(5.0).all()


def test_rsi_extremes_and_balance():
    up = pd.Series([float(i) for i in range(1, 40)])
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    flat = pd.Series([10.0] * 40)
    assert rsi(flat).iloc[-1] == pytest.approx(50.0)
    zigzag = pd.Series([10.0 + (1 if i % 2 else 0) for i in range(200)])
    assert 45 < rsi(zigzag).iloc[-1] < 55


def test_atr_of_constant_range():
    df = pd.DataFrame({"high": [11.0] * 50, "low": [9.0] * 50, "close": [10.0] * 50})
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_rolling_high_excludes_current_bar():
    high = pd.Series([1.0, 2.0, 3.0, 10.0])
    assert rolling_high(high, 3).iloc[-1] == 3.0


def test_unclosed_candle_is_dropped():
    rows = make_candles(250)
    df = frame(rows)
    assert len(df) == 250  # из 251 строки последняя (текущая) отброшена


def test_breakout_in_uptrend_gives_signal():
    params = StrategyParams()
    f = compute_features("SOL/USDT", frame(make_candles(300, breakout=True)), params)
    assert f is not None and f.trend == "up"
    conditions = entry_conditions(f, params)
    assert all(conditions.values()), conditions
    sig = entry_signal(f, params)
    assert sig is not None
    assert sig.stop_price < sig.price
    assert sig.stop_pct == pytest.approx(params.atr_stop_mult * f.atr / f.price * 100)
    assert sig.id.startswith("SOLUSDT-")


def test_no_breakout_no_signal():
    params = StrategyParams()
    f = compute_features("BTC/USDT", frame(make_candles(300, breakout=False)), params)
    assert f is not None
    assert entry_signal(f, params) is None


def test_downtrend_never_signals():
    params = StrategyParams()
    f = compute_features("XRP/USDT", frame(make_candles(300, drift=-0.001, breakout=True)), params)
    assert f is not None and f.trend == "down"
    assert entry_signal(f, params) is None


def test_short_history_returns_none():
    assert compute_features("X/USDT", frame(make_candles(100)), StrategyParams()) is None


def test_exit_rules():
    params = StrategyParams(atr_trail_mult=3.0)
    common = dict(entry_price=100.0, atr_at_entry=1.0, params=params, max_hold_hours=96)
    # стоп-лосс
    d = evaluate_exit(stop_price=98.0, highest_price=100.0, price=97.9, held_hours=1, **common)
    assert d.should_exit and d.reason == "stop_loss"
    # трейлинг поднимает стоп: максимум 110 -> стоп 107
    d = evaluate_exit(stop_price=98.0, highest_price=100.0, price=110.0, held_hours=5, **common)
    assert not d.should_exit and d.stop_price == pytest.approx(107.0) and d.highest_price == 110.0
    d = evaluate_exit(stop_price=107.0, highest_price=110.0, price=106.5, held_hours=6, **common)
    assert d.should_exit and d.reason == "trailing_stop"
    # стоп по времени
    d = evaluate_exit(stop_price=98.0, highest_price=101.0, price=100.5, held_hours=97, **common)
    assert d.should_exit and d.reason == "time_stop"
