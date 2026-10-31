from dataclasses import replace
from datetime import timedelta

import pytest

from trader.backtest import (
    BacktestConfig,
    fetch_history,
    format_comparison,
    format_result,
    parse_variant,
    run_backtest,
    select_pool,
)
from trader.market_data import ohlcv_to_frame

from .conftest import HOUR_MS, make_candles

NOW_MS = 1_800_000_000_000 // HOUR_MS * HOUR_MS


def frame_with_tail(tail: list[tuple[float, float, float, float]], n: int = 400, now_ms: int = NOW_MS):
    """Плавный рост, на последней свече пробой с объёмом (сигнал), затем свечи tail (open, high, low, close)."""
    rows = make_candles(n, breakout=True, now_ms=now_ms)[:-1]  # без незакрытой свечи
    ts = rows[-1][0]
    for o, h, low, c in tail:
        ts += HOUR_MS
        rows.append([ts, o, h, low, c, 100.0])
    frame = ohlcv_to_frame(rows, "1h", now_ms=ts + 2 * HOUR_MS)
    signal_time = frame.index[n - 1]
    return frame, signal_time


def start_before(frame, signal_time, bars: int = 5):
    return signal_time - timedelta(hours=bars)


def test_stop_loss_exit_with_fees_and_cooldown():
    frame, sig_time = frame_with_tail([])
    entry = float(frame.loc[sig_time, "close"])
    # гэп вниз на 10%: выход по цене открытия, а не по стопу
    gap = entry * 0.9
    frame, sig_time = frame_with_tail([(gap, gap * 1.001, gap * 0.99, gap)] + [(gap, gap * 1.001, gap * 0.999, gap)] * 3)
    cfg = BacktestConfig(cooldown_hours=6)
    result = run_backtest({"SOL/USDT": frame}, cfg, start_before(frame, sig_time))

    assert result.signals == 1 and len(result.trades) == 1
    trade = result.trades[0]
    assert trade.reason == "stop_loss"
    assert trade.entry_time == sig_time + timedelta(hours=1)
    assert trade.exit_price == pytest.approx(gap)
    # комиссия 0.1% и проскальзывание 0.05% на каждой стороне
    expected = trade.cost / (entry * 1.0005) * 0.999 * gap * 0.9995 * 0.999
    assert trade.proceeds == pytest.approx(expected)
    assert result.final_equity == pytest.approx(cfg.start_usdt + trade.pnl)
    # размер от риска: потеря на стопе около 1% капитала (тут больше из-за гэпа)
    assert trade.cost == pytest.approx(min(10_000 * 0.01 / ((entry - trade.stop) / entry), 10_000 * 0.2), rel=1e-3)


def test_trailing_stop_locks_profit():
    frame, sig_time = frame_with_tail([])
    p = float(frame.loc[sig_time, "close"])
    tail = []
    for _ in range(6):  # ралли
        tail.append((p, p * 1.03, p * 0.999, p * 1.03))
        p *= 1.03
    tail.append((p, p * 1.001, p * 0.85, p * 0.86))  # резкий откат
    frame, sig_time = frame_with_tail(tail)
    result = run_backtest({"SOL/USDT": frame}, BacktestConfig(), start_before(frame, sig_time))

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.reason == "trailing_stop"
    assert trade.pnl > 0
    assert trade.exit_price == pytest.approx(trade.stop)


def test_max_open_positions_skips_extra_signals():
    frame, sig_time = frame_with_tail([])
    p = float(frame.loc[sig_time, "close"])
    frame, sig_time = frame_with_tail([(p, p * 1.001, p * 0.999, p)] * 3)
    cfg = BacktestConfig(limits=replace(BacktestConfig().limits, max_open_positions=1))
    result = run_backtest({"SOL/USDT": frame, "ETH/USDT": frame.copy()}, cfg, start_before(frame, sig_time))

    assert result.signals == 2
    assert result.skipped == {"лимит позиций": 1}
    assert [t.symbol for t in result.trades] == ["SOL/USDT"]  # приоритет у пары выше в списке
    assert result.trades[0].reason == "end_of_test"
    assert "Сделок: 1" in format_result(result)
    assert "текущие" in format_comparison([("текущие", result)])


def test_universe_size_limits_symbols():
    frame, sig_time = frame_with_tail([])
    result = run_backtest({"A/USDT": frame, "B/USDT": frame.copy()}, BacktestConfig(universe_size=1), sig_time)
    assert result.symbols == ["A/USDT"]


def test_overrides_and_variant_parsing():
    name, overrides = parse_variant("широкий:UNIVERSE_SIZE=25, min_volume_ratio=1.0")
    assert name == "широкий" and overrides == {"UNIVERSE_SIZE": "25", "min_volume_ratio": "1.0"}
    cfg = BacktestConfig().with_overrides({**overrides, "MAX_OPEN_POSITIONS": "6", "RSI_MAX": "80"})
    assert cfg.universe_size == 25 and isinstance(cfg.universe_size, int)
    assert cfg.params.min_volume_ratio == 1.0 and cfg.params.rsi_max == 80.0
    assert cfg.limits.max_open_positions == 6
    assert parse_variant("BREAKOUT_LOOKBACK=15")[0] == "BREAKOUT_LOOKBACK=15"

    with pytest.raises(ValueError, match="неизвестный"):
        BacktestConfig().with_overrides({"LEVERAGE": "10"})
    with pytest.raises(ValueError, match="число"):
        BacktestConfig().with_overrides({"UNIVERSE_SIZE": "много"})
    with pytest.raises(ValueError):
        parse_variant("имя:UNIVERSE_SIZE")


def test_fetch_history_paginates_and_drops_open_candle():
    rows = make_candles(2500, now_ms=NOW_MS)  # последняя строка — незакрытая свеча

    class Paged:
        calls = 0

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            Paged.calls += 1
            return [r for r in rows if r[0] >= since][:limit]

    frame = fetch_history(Paged(), "SOL/USDT", "1h", rows[0][0], now_ms=NOW_MS)
    assert len(frame) == 2500 and Paged.calls == 3
    assert frame.index.is_monotonic_increasing


def test_dynamic_universe_follows_volume_at_the_time():
    """Сигнал засчитывается, только если пара в тот час была в топе по объёму, а не сегодня."""
    frame, sig_time = frame_with_tail([])
    loud, quiet = frame.copy(), frame.copy()
    loud["volume"] *= 1e6  # объём в USDT ~ 1e10 за сутки: проходит MIN_QUOTE_VOLUME
    cfg = BacktestConfig(dynamic_universe=True, universe_size=1, min_quote_volume=1e6)
    result = run_backtest({"QUIET/USDT": quiet, "LOUD/USDT": loud}, cfg, start_before(frame, sig_time))

    assert set(result.universe_hours) == {"LOUD/USDT"}
    assert result.signals == 1 and [t.symbol for t in result.trades] == ["LOUD/USDT"]

    # тот же сигнал не учитывается, если объём пары ниже порога
    none = run_backtest({"QUIET/USDT": quiet}, cfg, start_before(frame, sig_time))
    assert none.signals == 0 and none.universe_hours == {}


def test_select_pool_keeps_past_leaders():
    import pandas as pd

    days = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")

    def daily(volumes):
        return pd.DataFrame({"close": 1.0, "volume": volumes}, index=days)

    pool = select_pool(
        {
            "OLD/USDT": daily([9e8, 9e8, 0.0, 0.0]),  # лидер в начале, потом делистинг
            "BIG/USDT": daily([5e8, 5e8, 5e8, 5e8]),
            "NEW/USDT": daily([0.0, 0.0, 1e9, 1e9]),
            "TINY/USDT": daily([1e6, 1e6, 1e6, 1e6]),  # ниже MIN_QUOTE_VOLUME
            "MID/USDT": daily([1e8, 1e8, 1e8, 1e8]),  # ни разу не в топ-2
        },
        start=days[1].to_pydatetime(),
        top_n=2,
        min_quote_volume=2e7,
    )
    assert pool == ["BIG/USDT", "NEW/USDT", "OLD/USDT"]


def test_bool_override():
    assert BacktestConfig().with_overrides({"DYNAMIC_UNIVERSE": "true"}).dynamic_universe is True


def crash_then_recovery_frame():
    """Сигнал, затем обвал (стоп), затем долгое боковое движение и новый пробой."""
    frame, sig_time = frame_with_tail([])
    p = float(frame.loc[sig_time, "close"])
    low = p * 0.5
    tail = [(low, low, low * 0.99, low)] + [(low, low * 1.001, low * 0.999, low)] * 30
    return frame_with_tail(tail)


def test_kill_switch_stops_trading_or_resets_after_pause():
    frame, sig_time = crash_then_recovery_frame()
    big_risk = replace(BacktestConfig().limits, risk_per_trade_pct=5.0, max_position_pct=100.0, max_drawdown_pct=10.0)
    cfg = BacktestConfig(limits=big_risk)
    result = run_backtest({"SOL/USDT": frame}, cfg, start_before(frame, sig_time))
    # гэп на 50% вниз: стоп по цене открытия, просадка > 10% -> аварийная остановка
    assert result.trades[0].reason == "stop_loss"
    assert len(result.kill_switch_times) == 1 and result.kill_switch_time == result.trades[0].exit_time

    # у второй пары сигнал через 20 часов после обвала первой
    later, later_sig = frame_with_tail([], now_ms=NOW_MS + 20 * HOUR_MS)
    p = float(later.loc[later_sig, "close"])
    later, later_sig = frame_with_tail([(p, p * 1.001, p * 0.999, p)] * 3, now_ms=NOW_MS + 20 * HOUR_MS)
    frames = {"SOL/USDT": frame, "ETH/USDT": later}
    start = start_before(frame, sig_time)

    stopped = run_backtest(frames, cfg, start)  # как живой бот: после остановки сделок нет
    assert [t.symbol for t in stopped.trades] == ["SOL/USDT"]
    assert stopped.skipped == {"аварийная остановка": 1}

    reset = run_backtest(frames, replace(cfg, kill_switch_reset_hours=6), start)  # пауза 6 ч, потом снова торгуем
    assert [t.symbol for t in reset.trades] == ["SOL/USDT", "ETH/USDT"]
    assert reset.trades[1].entry_time == later_sig + timedelta(hours=1)
    assert len(reset.kill_switch_times) == 1  # пик капитала сброшен: старая просадка не вызывает новую остановку


def test_monthly_breakdown_sums_to_total():
    from trader.backtest import format_breakdown, monthly_breakdown

    frame, sig_time = frame_with_tail([])
    start = sig_time - timedelta(days=40)
    result = run_backtest({"BTC/USDT": frame}, BacktestConfig(), start)
    rows = monthly_breakdown(result)
    assert len(rows) >= 2
    total = 1.0
    for row in rows:
        total *= 1 + row.return_pct / 100
    assert (total - 1) * 100 == pytest.approx(result.return_pct, abs=1e-6)
    assert sum(r.trades for r in rows) == len(result.trades)
    assert all(r.btc_pct is not None for r in rows)
    assert rows[0].label in format_breakdown(rows)


def test_candle_cache_fetches_only_new_tail(tmp_path):
    from trader.backtest import CandleCache, fetch_many

    rows = make_candles(1500, now_ms=NOW_MS)
    calls = []

    class Paged:
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            calls.append(since)
            return [r for r in rows if r[0] >= since][:limit]

    cache = CandleCache(tmp_path / "candles")
    first_now = rows[1200][0]
    first = cache.history(Paged(), "SOL/USDT", "1h", rows[0][0], now_ms=first_now)
    assert len(first) == 1200 and len(calls) == 2

    calls.clear()
    frames, errors = fetch_many(lambda: Paged(), ["SOL/USDT"], "1h", rows[100][0], cache=cache, now_ms=NOW_MS)
    assert errors == {}
    assert calls == [rows[1200][0]]  # докачан только хвост
    assert len(frames["SOL/USDT"]) == 1500 - 100
    assert frames["SOL/USDT"].index.is_unique and frames["SOL/USDT"].index.is_monotonic_increasing


def test_fetch_many_reports_errors_and_retries(monkeypatch):
    import ccxt

    from trader import backtest

    monkeypatch.setattr(backtest.time, "sleep", lambda _s: None)
    rows = make_candles(50, now_ms=NOW_MS)
    attempts = {"FLAKY/USDT": 0}

    class Exchange:
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            if symbol == "BAD/USDT":
                raise ccxt.BadSymbol("нет такой пары")
            if symbol == "FLAKY/USDT" and attempts[symbol] < 2:
                attempts[symbol] += 1
                raise ccxt.NetworkError("timeout")
            return [r for r in rows if r[0] >= since][:limit]

    seen = []
    frames, errors = backtest.fetch_many(
        Exchange, ["FLAKY/USDT", "BAD/USDT"], "1h", rows[0][0], workers=2, progress=lambda d, t: seen.append(d), now_ms=NOW_MS
    )
    assert list(frames) == ["FLAKY/USDT"] and len(frames["FLAKY/USDT"]) == 50
    assert "нет такой пары" in errors["BAD/USDT"]
    assert sorted(seen) == [1, 2]


def test_regime_filter_in_backtest():
    import pandas as pd

    frame, sig_time = frame_with_tail([])
    p = float(frame.loc[sig_time, "close"])
    frame, sig_time = frame_with_tail([(p, p * 1.001, p * 0.999, p)] * 3)
    days = pd.date_range(end=sig_time.normalize() - pd.Timedelta(days=1), periods=300, freq="D")
    rising = pd.DataFrame({"close": [100.0 + i for i in range(300)]}, index=days)
    falling = rising.assign(close=rising["close"][::-1].to_numpy())
    cfg = BacktestConfig(regime_filter=True, regime_ema=200)
    start = start_before(frame, sig_time)

    ok = run_backtest({"SOL/USDT": frame}, cfg, start, regime_daily=rising)
    assert len(ok.trades) == 1 and ok.regime_off_hours == 0

    blocked = run_backtest({"SOL/USDT": frame}, cfg, start, regime_daily=falling)
    assert blocked.trades == [] and blocked.skipped == {"фильтр рынка": 1}
    assert blocked.regime_off_hours == len(blocked.equity)
    assert "Фильтр рынка запрещал входы 100%" in format_result(blocked)

    with pytest.raises(ValueError, match="BTC"):
        run_backtest({"SOL/USDT": frame}, cfg, start)
