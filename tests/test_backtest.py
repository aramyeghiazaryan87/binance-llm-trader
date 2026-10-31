from dataclasses import replace
from datetime import timedelta

import pytest

from trader.backtest import BacktestConfig, fetch_history, format_comparison, format_result, parse_variant, run_backtest
from trader.market_data import ohlcv_to_frame

from .conftest import HOUR_MS, make_candles

NOW_MS = 1_800_000_000_000 // HOUR_MS * HOUR_MS


def frame_with_tail(tail: list[tuple[float, float, float, float]], n: int = 400):
    """Плавный рост, на последней свече пробой с объёмом (сигнал), затем свечи tail (open, high, low, close)."""
    rows = make_candles(n, breakout=True, now_ms=NOW_MS)[:-1]  # без незакрытой свечи
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
