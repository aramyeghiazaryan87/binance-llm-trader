"""Вторая стратегия «откат в тренде» и запуск нескольких ботов рядом."""
from dataclasses import replace
from datetime import timedelta

import pytest

from trader.backtest import BacktestConfig, run_backtest
from trader.config import ConfigError, Settings
from trader.market_data import ohlcv_to_frame
from trader.strategy import (
    STRATEGIES,
    StrategyParams,
    compute_features,
    entry_signal,
    pullback_conditions,
    pullback_signal,
)

from .conftest import HOUR_MS, make_candles

NOW_MS = 1_800_000_000_000 // HOUR_MS * HOUR_MS


def pullback_rows(dip_bars: int = 6, tail: int = 0):
    """Ровный рост, откат на dip_bars свечей по -0.4%, зелёная свеча отскока, затем tail свечей вбок."""
    rows = make_candles(400, now_ms=NOW_MS)[:-1]
    ts, c = rows[-1][0], rows[-1][4]
    for _ in range(dip_bars):
        o, c, ts = c, c * 0.996, ts + HOUR_MS
        rows.append([ts, o, o * 1.001, c * 0.999, c, 100.0])
    o, c, ts = c, c * 1.004, ts + HOUR_MS
    rows.append([ts, o, c * 1.001, o * 0.999, c, 120.0])
    bounce_ts = ts
    for _ in range(tail):
        ts += HOUR_MS
        rows.append([ts, c, c * 1.001, c * 0.999, c, 100.0])
    return rows, bounce_ts, ts


def features(rows, last_ts):
    df = ohlcv_to_frame(rows, "1h", now_ms=last_ts + 2 * HOUR_MS)
    return compute_features("SOL/USDT", df.iloc[-300:], StrategyParams())


def test_pullback_signal_on_bounce_near_ema50():
    rows, _, last = pullback_rows(6)
    f = features(rows, last)
    assert all(pullback_conditions(f, StrategyParams()).values())
    sig = pullback_signal(f, StrategyParams())
    assert sig is not None and sig.stop_price == pytest.approx(f.price - 2 * f.atr)
    assert "откат к EMA50" in sig.reason
    assert entry_signal(f, StrategyParams()) is None  # пробоя здесь нет: стратегии ловят разное


def test_no_pullback_signal_when_dip_too_shallow_or_too_deep_or_no_bounce():
    for dip in (3, 10):  # цена далеко от EMA50 / RSI слишком низкий
        rows, _, last = pullback_rows(dip)
        assert pullback_signal(features(rows, last), StrategyParams()) is None
    rows, _, last = pullback_rows(6)
    red = rows[:-1] + [[rows[-1][0], rows[-1][4] * 1.001, rows[-1][2], rows[-1][3], rows[-1][4], 120.0]]
    assert pullback_signal(features(red, last), StrategyParams()) is None  # свеча закрылась ниже открытия


def test_settings_select_strategy_and_separate_files(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    s = Settings.from_env(None)
    assert s.strategy == "breakout" and s.db_path.name == "trader_paper.db"  # прежние файлы не меняются

    monkeypatch.setenv("STRATEGY", "Pullback")
    monkeypatch.setenv("BOT_NAME", "pullback")
    s = Settings.from_env(None)
    assert s.strategy == "pullback"
    assert s.db_path.name == "trader_paper_pullback.db" and s.log_path.name == "trader_paper_pullback.log"

    monkeypatch.setenv("STRATEGY", "scalping")
    with pytest.raises(ConfigError, match="STRATEGY"):
        Settings.from_env(None)
    monkeypatch.setenv("STRATEGY", "pullback")
    monkeypatch.setenv("BOT_NAME", "../x")
    with pytest.raises(ConfigError, match="BOT_NAME"):
        Settings.from_env(None)


def test_engine_and_prompt_use_selected_strategy(settings, now):
    from .test_engine import build
    from .conftest import approving_responder

    engine, _, _, _, client = build(replace(settings, strategy="pullback"), now, responder=approving_responder())
    assert engine.signal_fn is STRATEGIES["pullback"] is pullback_signal
    engine.advisor.strategy = "pullback"
    assert "pullback strategy" in engine.advisor.system_prompt
    assert "trend-breakout" not in engine.advisor.system_prompt

    default, *_ = build(settings, now, responder=approving_responder())
    assert default.signal_fn is entry_signal and "trend-breakout" in default.advisor.system_prompt


def test_backtest_runs_pullback_strategy():
    rows, bounce_ts, last = pullback_rows(6, tail=4)
    frame = ohlcv_to_frame(rows, "1h", now_ms=last + 2 * HOUR_MS)
    signal_time = frame.index[frame.index.get_loc(frame.index[-1]) - 4]
    start = signal_time - timedelta(hours=3)

    breakout = run_backtest({"SOL/USDT": frame}, BacktestConfig(), start)
    pullback = run_backtest({"SOL/USDT": frame}, BacktestConfig(strategy="pullback"), start)
    assert breakout.signals == 0
    assert pullback.signals == 1 and len(pullback.trades) == 1
    assert pullback.trades[0].entry_time == signal_time + timedelta(hours=1)
    assert BacktestConfig().with_overrides({"STRATEGY": "pullback"}).strategy == "pullback"


def test_prompt_states_real_timeframe(settings, now):
    from .conftest import approving_responder
    from .test_engine import build

    engine, *_ = build(settings, now, responder=approving_responder())
    assert "closed 1h candles" in engine.advisor.system_prompt
    engine.advisor.timeframe = "4h"
    prompt = engine.advisor.system_prompt
    assert "After every closed 4h candle" in prompt and "1h" not in prompt
