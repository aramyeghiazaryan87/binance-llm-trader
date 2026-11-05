"""Стратегия «Трендовый пробой» (только лонг, спот).

Вход (на закрытой свече, все условия одновременно):
  1. Цена выше EMA200: долгосрочный тренд вверх.
  2. EMA20 выше EMA50: среднесрочный тренд вверх.
  3. Закрытие выше максимума предыдущих 20 свечей: пробой.
  4. Объём свечи >= 1.2 x средний объём за 20 свечей: пробой подтверждён объёмом.
  5. RSI(14) в диапазоне 50..78: импульс есть, но без сильной перекупленности.
  6. ATR в диапазоне 0.3%..6% от цены: волатильность в разумных пределах.

Выход:
  * начальный стоп = вход - 2 x ATR;
  * трейлинг-стоп = максимум с момента входа - 3 x ATR (стоп только поднимается);
  * стоп по времени: позиция закрывается через MAX_HOLD_HOURS;
  * досрочное закрытие по совету LLM (только уменьшает риск).

LLM не может открыть сделку без сигнала стратегии: он только одобряет или отклоняет сигналы.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .indicators import atr, ema, rolling_high, rsi, sma


@dataclass(frozen=True)
class StrategyParams:
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    breakout_lookback: int = 20
    volume_lookback: int = 20
    min_volume_ratio: float = 1.2
    rsi_period: int = 14
    rsi_min: float = 50.0
    rsi_max: float = 78.0
    atr_period: int = 14
    min_atr_pct: float = 0.3
    max_atr_pct: float = 6.0
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 3.0
    # стратегия «откат в тренде» (pullback)
    pullback_rsi_min: float = 35.0
    pullback_rsi_max: float = 55.0
    pullback_ema_atr: float = 1.0  # цена не дальше стольких ATR от EMA50


@dataclass(frozen=True)
class Features:
    symbol: str
    price: float
    candle_time: datetime  # время открытия последней закрытой свечи (UTC)
    change_24h_pct: float
    change_7d_pct: float
    rsi: float
    atr: float
    atr_pct: float
    volume_ratio: float
    ema_fast: float
    ema_mid: float
    ema_slow: float
    trend: str  # up | down | mixed
    breakout_level: float
    dist_to_breakout_pct: float
    open_price: float = float("nan")  # открытие последней закрытой свечи
    prev_close: float = float("nan")  # закрытие предыдущей свечи


@dataclass(frozen=True)
class EntrySignal:
    id: str
    symbol: str
    price: float
    atr: float
    stop_price: float
    stop_pct: float
    reason: str
    features: Features


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str
    stop_price: float
    highest_price: float


def timeframe_minutes(timeframe: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    try:
        return int(timeframe[:-1]) * units[timeframe[-1]]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"неподдерживаемый таймфрейм: {timeframe!r}") from exc


def _pct_change(close: pd.Series, bars_back: int) -> float:
    if len(close) <= bars_back:
        return float("nan")
    base = float(close.iloc[-1 - bars_back])
    return (float(close.iloc[-1]) / base - 1.0) * 100.0 if base else float("nan")


def compute_features(symbol: str, df: pd.DataFrame, params: StrategyParams, timeframe: str = "1h") -> Features | None:
    """Считает индикаторы по закрытым свечам. None, если истории мало."""
    if df is None or len(df) < params.ema_slow + 5:
        return None
    close = df["close"]
    e_fast = ema(close, params.ema_fast)
    e_mid = ema(close, params.ema_mid)
    e_slow = ema(close, params.ema_slow)
    r = rsi(close, params.rsi_period)
    a = atr(df, params.atr_period)
    vol_avg = sma(df["volume"].shift(1), params.volume_lookback)
    level = rolling_high(df["high"], params.breakout_lookback)

    price = float(close.iloc[-1])
    last_atr = float(a.iloc[-1])
    last_vol_avg = float(vol_avg.iloc[-1])
    last_level = float(level.iloc[-1])
    values = (float(e_fast.iloc[-1]), float(e_mid.iloc[-1]), float(e_slow.iloc[-1]), float(r.iloc[-1]), last_atr)
    if price <= 0 or any(pd.isna(v) for v in values) or pd.isna(last_level):
        return None
    f_fast, f_mid, f_slow, f_rsi, _ = values

    if f_fast > f_mid > f_slow:
        trend = "up"
    elif f_fast < f_mid < f_slow:
        trend = "down"
    else:
        trend = "mixed"

    bars_per_day = max(1, 1440 // timeframe_minutes(timeframe))
    ts = df.index[-1]
    candle_time = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    return Features(
        symbol=symbol,
        price=price,
        candle_time=candle_time,
        change_24h_pct=_pct_change(close, bars_per_day),
        change_7d_pct=_pct_change(close, bars_per_day * 7),
        rsi=f_rsi,
        atr=last_atr,
        atr_pct=last_atr / price * 100.0,
        volume_ratio=float(df["volume"].iloc[-1]) / last_vol_avg if last_vol_avg > 0 else 0.0,
        ema_fast=f_fast,
        ema_mid=f_mid,
        ema_slow=f_slow,
        trend=trend,
        breakout_level=last_level,
        dist_to_breakout_pct=(price / last_level - 1.0) * 100.0 if last_level > 0 else 0.0,
        open_price=float(df["open"].iloc[-1]),
        prev_close=float(close.iloc[-2]),
    )


def regime_series(daily: pd.DataFrame, ema_period: int) -> pd.Series:
    """Фильтр рынка по дневным свечам BTC: True, если закрытие выше дневной EMA.

    Индекс: время ЗАКРЫТИЯ дневной свечи (открытие + 1 день), чтобы в момент t использовать
    только уже закрытые дни. Пока истории не хватает на EMA, значение False (входов нет).
    """
    if daily is None or daily.empty:
        return pd.Series(dtype=bool)
    close = daily["close"]
    line = ema(close, ema_period)
    ok = (close > line) & line.notna()
    ok.index = daily.index + pd.Timedelta(days=1)
    return ok.astype(bool)


def regime_ok(daily: pd.DataFrame, ema_period: int) -> bool:
    """Текущее состояние фильтра рынка по последнему закрытому дню."""
    series = regime_series(daily, ema_period)
    return bool(series.iloc[-1]) if len(series) else False


def entry_conditions(f: Features, params: StrategyParams) -> dict[str, bool]:
    return {
        "above_ema200": f.price > f.ema_slow,
        "ema20_above_ema50": f.ema_fast > f.ema_mid,
        "breakout": f.price > f.breakout_level,
        "volume_confirmed": f.volume_ratio >= params.min_volume_ratio,
        "rsi_in_range": params.rsi_min <= f.rsi <= params.rsi_max,
        "atr_in_range": params.min_atr_pct <= f.atr_pct <= params.max_atr_pct,
    }


def entry_signal(f: Features, params: StrategyParams) -> EntrySignal | None:
    conditions = entry_conditions(f, params)
    if not all(conditions.values()):
        return None
    stop = f.price - params.atr_stop_mult * f.atr
    if stop <= 0:
        return None
    signal_id = f"{f.symbol.replace('/', '')}-{f.candle_time:%Y%m%d%H%M}"
    reason = (
        f"пробой {params.breakout_lookback}-свечного максимума {f.breakout_level:.6g}, "
        f"объём x{f.volume_ratio:.1f}, RSI {f.rsi:.0f}, ATR {f.atr_pct:.2f}%"
    )
    return EntrySignal(
        id=signal_id,
        symbol=f.symbol,
        price=f.price,
        atr=f.atr,
        stop_price=stop,
        stop_pct=(1.0 - stop / f.price) * 100.0,
        reason=reason,
        features=f,
    )


def pullback_conditions(f: Features, params: StrategyParams) -> dict[str, bool]:
    return {
        "above_ema200": f.price > f.ema_slow,
        "ema50_above_ema200": f.ema_mid > f.ema_slow,
        "rsi_pullback": params.pullback_rsi_min <= f.rsi <= params.pullback_rsi_max,
        "near_ema50": abs(f.price - f.ema_mid) <= params.pullback_ema_atr * f.atr,
        "bounce": f.price > f.open_price and f.price > f.prev_close,
        "atr_in_range": params.min_atr_pct <= f.atr_pct <= params.max_atr_pct,
    }


def pullback_signal(f: Features, params: StrategyParams) -> EntrySignal | None:
    """Стратегия «Откат в тренде»: в устойчивом восходящем тренде (цена и EMA50 выше EMA200)
    цена откатилась к EMA50 (RSI 35..55), и свеча закрылась отскоком вверх. Покупаем отскок."""
    if not all(pullback_conditions(f, params).values()):
        return None
    stop = f.price - params.atr_stop_mult * f.atr
    if stop <= 0:
        return None
    reason = (
        f"откат к EMA50 {f.ema_mid:.6g} ({(f.price / f.ema_mid - 1) * 100:+.1f}%) и отскок, "
        f"RSI {f.rsi:.0f}, ATR {f.atr_pct:.2f}%"
    )
    return EntrySignal(
        id=f"{f.symbol.replace('/', '')}-{f.candle_time:%Y%m%d%H%M}",
        symbol=f.symbol,
        price=f.price,
        atr=f.atr,
        stop_price=stop,
        stop_pct=(1.0 - stop / f.price) * 100.0,
        reason=reason,
        features=f,
    )


STRATEGIES = {"breakout": entry_signal, "pullback": pullback_signal}
STRATEGY_DESCRIPTIONS = {
    "breakout": "a rule-based trend-breakout strategy (buys when the close breaks above the 20-candle high on "
                "rising volume in an uptrend)",
    "pullback": "a rule-based pullback strategy (in an established uptrend it buys the first bounce after the price "
                "dips back to the 50-period EMA)",
}


def evaluate_exit(
    *,
    entry_price: float,
    stop_price: float,
    highest_price: float,
    atr_at_entry: float,
    price: float,
    held_hours: float,
    params: StrategyParams,
    max_hold_hours: float,
) -> ExitDecision:
    """Проверка стопов. Все цены в одной шкале (источник рыночных данных)."""
    highest = max(highest_price, price)
    trail = highest - params.atr_trail_mult * atr_at_entry
    stop = max(stop_price, trail)
    if price <= stop:
        reason = "trailing_stop" if stop > entry_price else "stop_loss"
        return ExitDecision(True, reason, stop, highest)
    if held_hours >= max_hold_hours:
        return ExitDecision(True, "time_stop", stop, highest)
    return ExitDecision(False, "", stop, highest)
