"""Бэктест стратегии «Трендовый пробой» на истории свечей (без LLM).

Повторяет логику живого бота, используя те же функции:
  * сигнал: compute_features + entry_signal на окне из CANDLES_LIMIT закрытых свечей,
    как в engine (EMA зависят от длины окна, поэтому окно то же самое);
  * вход: по цене закрытия сигнальной свечи + проскальзывание, комиссия как в paper-режиме;
  * размер и допуск сделки: RiskManager (риск на сделку, лимиты позиций и экспозиции,
    дневной лимит убытка, аварийная остановка по просадке);
  * выход: evaluate_exit — стоп-лосс, трейлинг-стоп, стоп по времени; пауза после стоп-лосса.

Внутри часовой свечи порядок цен неизвестен, поэтому проверка стопа осторожная:
сначала минимум свечи против старого стопа (при гэпе вниз выход по цене открытия),
затем подтягиваем трейлинг по максимуму и, если закрытие ниже нового стопа, выходим
по закрытию (живой бот проверяет стопы каждые 5 минут и поймал бы это внутри часа).

Claude здесь не участвует: каждый сигнал считается одобренным с множителем размера
SIZE_FACTOR. Это проверка самих правил: если они не зарабатывают, фильтр LLM не спасёт.

Список пар (dynamic_universe=True) выбирается каждый час заново, как у живого бота:
топ UNIVERSE_SIZE по объёму торгов за предыдущие 24 часа среди пула кандидатов.
Пул собирает select_pool по дневным свечам всех USDT-пар, включая делистнутые, поэтому
в тест попадают и монеты, которые были в топе тогда, а сегодня упали или исчезли.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .indicators import rolling_high, sma
from .market_data import ohlcv_to_frame, with_retries
from .risk import AccountState, RiskLimits, RiskManager
from .strategy import StrategyParams, compute_features, entry_signal, evaluate_exit, regime_series, timeframe_minutes

log = logging.getLogger(__name__)

EXIT_NAMES = {
    "stop_loss": "стоп-лосс",
    "trailing_stop": "трейлинг-стоп",
    "time_stop": "по времени",
    "kill_switch": "аварийная остановка",
    "end_of_test": "конец теста",
}


@dataclass(frozen=True)
class BacktestConfig:
    params: StrategyParams = StrategyParams()
    limits: RiskLimits = RiskLimits()
    universe_size: int = 15
    start_usdt: float = 10_000.0
    fee_pct: float = 0.1
    slippage_pct: float = 0.05
    cooldown_hours: float = 6.0
    max_hold_hours: float = 96.0
    size_factor: float = 1.0
    candles_window: int = 300
    timeframe: str = "1h"
    min_quote_volume: float = 20_000_000.0
    dynamic_universe: bool = False  # False: пары в порядке frames, первые universe_size
    # 0 = как у живого бота: после аварийной остановки сделок больше нет (снимается только вручную).
    # >0 = остановка снимается через столько часов, пик капитала сбрасывается (как reset-kill-switch).
    kill_switch_reset_hours: float = 0.0
    regime_filter: bool = False  # входы только когда BTC выше дневной EMA(regime_ema), как REGIME_FILTER
    regime_ema: int = 200

    @classmethod
    def from_settings(cls, s: Any) -> "BacktestConfig":
        return cls(
            params=StrategyParams(
                breakout_lookback=s.breakout_lookback,
                min_volume_ratio=s.min_volume_ratio,
                atr_stop_mult=s.atr_stop_mult,
                atr_trail_mult=s.atr_trail_mult,
            ),
            limits=RiskLimits(
                risk_per_trade_pct=s.risk_per_trade_pct,
                max_position_pct=s.max_position_pct,
                max_open_positions=s.max_open_positions,
                max_total_exposure_pct=s.max_total_exposure_pct,
                daily_loss_limit_pct=s.daily_loss_limit_pct,
                max_drawdown_pct=s.max_drawdown_pct,
                min_order_usdt=s.min_order_usdt,
            ),
            universe_size=s.universe_size,
            start_usdt=s.paper_start_usdt,
            fee_pct=s.paper_fee_pct,
            slippage_pct=s.paper_slippage_pct,
            cooldown_hours=s.cooldown_hours,
            max_hold_hours=s.max_hold_hours,
            candles_window=s.candles_limit,
            timeframe=s.timeframe,
            min_quote_volume=s.min_quote_volume,
            regime_filter=s.regime_filter,
            regime_ema=s.regime_ema,
        )

    def with_overrides(self, overrides: dict[str, str]) -> "BacktestConfig":
        """KEY=VALUE в стиле .env (регистр не важен): поля StrategyParams, RiskLimits или самого конфига."""
        groups: list[tuple[str, Any]] = [("params", self.params), ("limits", self.limits), ("", self)]
        changes: dict[str, dict[str, Any]] = {"params": {}, "limits": {}, "": {}}
        for raw_key, raw_value in overrides.items():
            key = raw_key.strip().lower()
            for group, obj in groups:
                types = {f.name: f.type for f in fields(obj) if f.name not in ("params", "limits")}
                if key in types:
                    kind = types[key] if isinstance(types[key], str) else types[key].__name__
                    try:
                        if kind == "bool":
                            value: Any = str(raw_value).strip().lower() in ("1", "true", "yes", "on", "да")
                        elif kind == "int":
                            value = int(raw_value)
                        elif kind == "float":
                            value = float(raw_value)
                        else:
                            value = str(raw_value)
                    except ValueError as exc:
                        raise ValueError(f"{raw_key}: ожидается число, получено {raw_value!r}") from exc
                    changes[group][key] = value
                    break
            else:
                raise ValueError(f"неизвестный параметр {raw_key!r}")
        return replace(
            self,
            params=replace(self.params, **changes["params"]),
            limits=replace(self.limits, **changes["limits"]),
            **changes[""],
        )


def parse_variant(spec: str) -> tuple[str, dict[str, str]]:
    """'имя:KEY=VAL,KEY=VAL' или 'KEY=VAL,KEY=VAL' (тогда имя = сами настройки)."""
    name, _, body = spec.partition(":") if ":" in spec else ("", "", spec)
    overrides: dict[str, str] = {}
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"ожидается KEY=VALUE, получено {part!r}")
        overrides[key.strip()] = value.strip()
    if not overrides:
        raise ValueError(f"пустой вариант {spec!r}")
    return (name.strip() or body.strip()), overrides


@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    entry_price: float  # цена рынка на входе (без проскальзывания)
    cost: float  # потрачено USDT
    qty: float
    atr: float
    stop: float
    highest: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    proceeds: float = 0.0
    reason: str = ""

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.cost * 100.0 if self.cost else 0.0

    @property
    def hold_hours(self) -> float:
        end = self.exit_time or self.entry_time
        return (end - self.entry_time).total_seconds() / 3600.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    symbols: list[str]
    start: datetime
    end: datetime
    trades: list[Trade] = field(default_factory=list)
    equity: list[tuple[datetime, float]] = field(default_factory=list)
    signals: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    kill_switch_times: list[datetime] = field(default_factory=list)
    benchmark_pct: float | None = None  # BTC buy & hold за тот же период
    benchmark_closes: pd.Series | None = None  # закрытия BTC за период, для помесячной разбивки
    universe_hours: dict[str, int] = field(default_factory=dict)  # сколько часов пара была в списке
    regime_off_hours: int = 0  # часов, когда фильтр рынка запрещал входы

    @property
    def kill_switch_time(self) -> datetime | None:
        return self.kill_switch_times[0] if self.kill_switch_times else None

    @property
    def pnl_by_symbol(self) -> dict[str, tuple[int, float]]:
        out: dict[str, tuple[int, float]] = {}
        for t in self.trades:
            n, pnl = out.get(t.symbol, (0, 0.0))
            out[t.symbol] = (n + 1, pnl + t.pnl)
        return out

    @property
    def weeks(self) -> float:
        return max((self.end - self.start).total_seconds() / (7 * 86_400), 1e-9)

    @property
    def final_equity(self) -> float:
        return self.equity[-1][1] if self.equity else self.config.start_usdt

    @property
    def return_pct(self) -> float:
        return (self.final_equity / self.config.start_usdt - 1.0) * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        peak, worst = -math.inf, 0.0
        for _, value in self.equity:
            peak = max(peak, value)
            if peak > 0:
                worst = max(worst, (1.0 - value / peak) * 100.0)
        return worst

    @property
    def win_rate(self) -> float:
        return sum(t.pnl > 0 for t in self.trades) / len(self.trades) * 100.0 if self.trades else 0.0

    @property
    def avg_trade_pct(self) -> float:
        return sum(t.pnl_pct for t in self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = -sum(t.pnl for t in self.trades if t.pnl < 0)
        if losses == 0:
            return math.inf if gains > 0 else 0.0
        return gains / losses

    @property
    def exit_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.trades:
            counts[t.reason] = counts.get(t.reason, 0) + 1
        return counts


def _prefilter(df: pd.DataFrame, params: StrategyParams) -> pd.Series:
    """Быстрый отсев: пробой и объём не зависят от длины окна, их можно посчитать сразу по всей истории.

    Полную проверку (EMA, RSI, ATR) делает entry_signal только там, где отсев пройден.
    """
    level = rolling_high(df["high"], params.breakout_lookback)
    vol_avg = sma(df["volume"].shift(1), params.volume_lookback)
    return (df["close"] > level) & (df["volume"] >= params.min_volume_ratio * vol_avg)


def run_backtest(
    frames: dict[str, pd.DataFrame],
    config: BacktestConfig,
    start: datetime,
    end: datetime | None = None,
    regime_daily: pd.DataFrame | None = None,
) -> BacktestResult:
    """frames: свечи по парам в порядке приоритета (ликвидности), включая историю для разогрева индикаторов.

    regime_daily: дневные свечи BTC (с запасом истории на EMA), нужны при cfg.regime_filter.
    """
    cfg = config
    params = cfg.params
    risk = RiskManager(cfg.limits)
    fee = cfg.fee_pct / 100.0
    slip = cfg.slippage_pct / 100.0
    tf = timedelta(minutes=timeframe_minutes(cfg.timeframe))
    symbols = [s for s in frames if not frames[s].empty]
    if not cfg.dynamic_universe:
        symbols = symbols[: cfg.universe_size]
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    times = sorted({ts for s in symbols for ts in frames[s].index if ts >= start and (end is None or ts < end)})
    result = BacktestResult(cfg, symbols, start, end or start)
    if not times:
        return result
    result.end = times[-1] + tf

    positions_idx = {s: {ts: i for i, ts in enumerate(frames[s].index)} for s in symbols}
    prefilter = {s: _prefilter(frames[s], params).to_numpy() for s in symbols}
    closes = {s: frames[s]["close"].to_numpy() for s in symbols}
    volume_rank = _volume_ranking(frames, symbols, times, cfg) if cfg.dynamic_universe else None
    regime_at = None
    if cfg.regime_filter:
        if regime_daily is None or regime_daily.empty:
            raise ValueError("для regime_filter нужны дневные свечи BTC (regime_daily)")
        regime = regime_series(regime_daily, cfg.regime_ema)
        regime_times, regime_values = regime.index, regime.to_numpy()

        def regime_at(t: datetime) -> bool:
            pos = regime_times.searchsorted(t, side="right") - 1  # последний день, закрытый к моменту t
            return bool(regime_values[pos]) if pos >= 0 else False
    last_close: dict[str, float] = {}
    cash = cfg.start_usdt
    open_trades: dict[str, Trade] = {}
    cooldowns: dict[str, datetime] = {}
    peak = cfg.start_usdt
    day, day_start_equity, daily_blocked = None, cfg.start_usdt, False
    kill = False
    kill_until: datetime | None = None

    def close_trade(trade: Trade, price: float, when: datetime, reason: str) -> None:
        nonlocal cash
        fill = price * (1 - slip)
        trade.exit_price = price
        trade.exit_time = when
        trade.proceeds = trade.qty * fill * (1 - fee)
        trade.reason = reason
        cash += trade.proceeds
        result.trades.append(trade)
        del open_trades[trade.symbol]
        if reason == "stop_loss" and cfg.cooldown_hours > 0:
            cooldowns[trade.symbol] = when + timedelta(hours=cfg.cooldown_hours)

    def skip(reason: str) -> None:
        result.skipped[reason] = result.skipped.get(reason, 0) + 1

    for ts in times:
        bar_close = ts + tf
        if kill and kill_until is not None and bar_close >= kill_until:
            kill, kill_until, peak = False, None, cash  # позиций нет: капитал = cash
        # 1. стопы открытых позиций по свече ts
        for symbol in list(open_trades):
            i = positions_idx[symbol].get(ts)
            if i is None:
                continue
            bar = frames[symbol].iloc[i]
            trade = open_trades[symbol]
            o, h, low, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
            if low <= trade.stop:
                reason = "trailing_stop" if trade.stop > trade.entry_price else "stop_loss"
                close_trade(trade, min(o, trade.stop), bar_close, reason)
                continue
            decision = evaluate_exit(
                entry_price=trade.entry_price,
                stop_price=trade.stop,
                highest_price=max(trade.highest, h),
                atr_at_entry=trade.atr,
                price=c,
                held_hours=(bar_close - trade.entry_time).total_seconds() / 3600.0,
                params=params,
                max_hold_hours=cfg.max_hold_hours,
            )
            trade.stop, trade.highest = decision.stop_price, decision.highest_price
            if decision.should_exit:
                close_trade(trade, c, bar_close, decision.reason)

        # 2. оценка капитала на закрытии свечи
        for symbol in open_trades:
            i = positions_idx[symbol].get(ts)
            if i is not None:
                last_close[symbol] = float(closes[symbol][i])
        value = sum(t.qty * last_close.get(t.symbol, t.entry_price) for t in open_trades.values())
        equity = cash + value
        if bar_close.date() != day:
            day, day_start_equity, daily_blocked = bar_close.date(), equity, False
        peak = max(peak, equity)
        drawdown = (1.0 - equity / peak) * 100.0 if peak > 0 else 0.0
        day_pnl = (equity / day_start_equity - 1.0) * 100.0 if day_start_equity > 0 else 0.0
        if not daily_blocked and risk.daily_loss_hit(day_pnl):
            daily_blocked = True
        if not kill and risk.drawdown_hit(drawdown):
            kill = True
            result.kill_switch_times.append(bar_close)
            if cfg.kill_switch_reset_hours > 0:
                kill_until = bar_close + timedelta(hours=cfg.kill_switch_reset_hours)
            for symbol in list(open_trades):
                close_trade(open_trades[symbol], last_close[symbol], bar_close, "kill_switch")
            equity = cash
        result.equity.append((bar_close, equity))

        # 3. сигналы на закрытой свече ts и входы по её закрытию
        universe = volume_rank(ts) if volume_rank else symbols
        for symbol in universe:
            result.universe_hours[symbol] = result.universe_hours.get(symbol, 0) + 1
        market_ok = regime_at(bar_close) if regime_at else True
        if not market_ok:
            result.regime_off_hours += 1
        for symbol in universe:
            i = positions_idx[symbol].get(ts)
            if i is None or not prefilter[symbol][i] or i + 1 < cfg.candles_window:
                continue
            window = frames[symbol].iloc[i + 1 - cfg.candles_window : i + 1]
            features = compute_features(symbol, window, params, cfg.timeframe)
            sig = entry_signal(features, params) if features else None
            if sig is None:
                continue
            if symbol in open_trades:
                continue  # живой бот не считает сигналом пару, по которой уже есть позиция
            until = cooldowns.get(symbol)
            if until and until > bar_close:
                continue
            result.signals += 1
            if not market_ok:
                skip("фильтр рынка")
                continue
            positions_value = sum(t.qty * last_close.get(t.symbol, t.entry_price) for t in open_trades.values())
            account = AccountState(
                equity=cash + positions_value,
                cash=cash,
                positions_value=positions_value,
                open_positions=len(open_trades),
                day_pnl_pct=day_pnl,
                drawdown_pct=drawdown,
                kill_switch=kill,
                daily_blocked=daily_blocked,
            )
            gate = risk.can_open(account, symbol, set(open_trades), cooldowns, bar_close)
            if not gate.ok:
                skip("аварийная остановка" if kill else "дневной лимит" if daily_blocked else "лимит позиций")
                continue
            price = sig.price
            stop = price - params.atr_stop_mult * sig.atr
            sizing = risk.size_position(
                account, price=price, stop_price=stop, size_factor=cfg.size_factor, min_cost=0.0, free_quote=cash
            )
            if not sizing.ok:
                skip("нет свободных средств")
                continue
            amount = sizing.amount_usdt
            qty = amount / (price * (1 + slip)) * (1 - fee)
            cash -= amount
            open_trades[symbol] = Trade(symbol, bar_close, price, amount, qty, sig.atr, stop, price)
            last_close[symbol] = price

    end_time = result.end
    for symbol in list(open_trades):
        close_trade(open_trades[symbol], last_close[symbol], end_time, "end_of_test")
    if result.equity:
        result.equity[-1] = (result.equity[-1][0], cash)

    btc = next((s for s in ("BTC/USDT", "BTC/USDC") if s in frames), None)
    if btc:
        period = frames[btc][(frames[btc].index >= start) & (frames[btc].index < end_time)]
        if not period.empty and period["open"].iloc[0] > 0:
            result.benchmark_pct = (float(period["close"].iloc[-1]) / float(period["open"].iloc[0]) - 1.0) * 100.0
            closes_btc = period["close"].copy()
            closes_btc.index = closes_btc.index + tf
            result.benchmark_closes = closes_btc
    return result


@dataclass(frozen=True)
class PeriodStats:
    label: str
    return_pct: float
    trades: int
    win_rate: float
    btc_pct: float | None
    kill_switch: bool


def monthly_breakdown(result: BacktestResult) -> list[PeriodStats]:
    """Доход по календарным месяцам (UTC) по кривой капитала; сделки относятся к месяцу закрытия."""
    if not result.equity:
        return []
    equity = pd.Series([v for _, v in result.equity], index=pd.DatetimeIndex([t for t, _ in result.equity]))
    out: list[PeriodStats] = []
    prev_end = result.config.start_usdt
    months = equity.index.tz_convert("UTC").tz_localize(None).to_period("M")
    for period, values in equity.groupby(months):
        start_value, end_value = prev_end, float(values.iloc[-1])
        prev_end = end_value
        first, last = values.index[0], values.index[-1]
        trades = [t for t in result.trades if t.exit_time and first <= t.exit_time <= last]
        btc = None
        if result.benchmark_closes is not None:
            before = result.benchmark_closes[result.benchmark_closes.index < first]
            inside = result.benchmark_closes[(result.benchmark_closes.index >= first) & (result.benchmark_closes.index <= last)]
            base = before.iloc[-1] if not before.empty else (inside.iloc[0] if not inside.empty else None)
            if base and not inside.empty:
                btc = (float(inside.iloc[-1]) / float(base) - 1.0) * 100.0
        out.append(
            PeriodStats(
                label=str(period),
                return_pct=(end_value / start_value - 1.0) * 100.0 if start_value > 0 else 0.0,
                trades=len(trades),
                win_rate=sum(t.pnl > 0 for t in trades) / len(trades) * 100.0 if trades else 0.0,
                btc_pct=btc,
                kill_switch=any(first <= k <= last for k in result.kill_switch_times),
            )
        )
    return out


def format_breakdown(rows: list[PeriodStats]) -> str:
    header = f"{'месяц':<8} {'доход%':>7} {'сделок':>6} {'приб.%':>6} {'BTC%':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:
        btc = f"{r.btc_pct:>+7.1f}" if r.btc_pct is not None else f"{'—':>7}"
        mark = "  ⛔ аварийная остановка" if r.kill_switch else ""
        lines.append(f"{r.label:<8} {r.return_pct:>+7.2f} {r.trades:>6} {r.win_rate:>6.0f} {btc}{mark}")
    return "\n".join(lines)


def _volume_ranking(frames: dict[str, pd.DataFrame], symbols: list[str], times: list[datetime], cfg: BacktestConfig):
    """Функция ts -> топ пар по объёму в USDT за 24 часа, закончившиеся закрытием свечи ts."""
    bars = max(1, 1440 // timeframe_minutes(cfg.timeframe))
    index = pd.DatetimeIndex(times)
    table = pd.DataFrame(
        {s: (frames[s]["close"] * frames[s]["volume"]).rolling(bars, min_periods=bars).sum().reindex(index) for s in symbols}
    )
    names = list(table.columns)
    values = table.to_numpy()
    row_of = {ts: i for i, ts in enumerate(times)}

    def top(ts: datetime) -> list[str]:
        row = values[row_of[ts]]
        ranked = sorted(
            (
                (v, names[j])
                for j, v in enumerate(row)
                if v == v and v >= cfg.min_quote_volume  # v == v: не NaN
            ),
            reverse=True,
        )
        return [name for _, name in ranked[: cfg.universe_size]]

    return top


def select_pool(daily: dict[str, pd.DataFrame], start: datetime, top_n: int, min_quote_volume: float) -> list[str]:
    """Пары, которые хотя бы один день периода были в топ-top_n по дневному объёму в USDT.

    Порядок: по суммарному объёму за период. Так в пул попадают монеты, которые были
    лидерами тогда, даже если сегодня они упали или делистнуты.
    """
    volume = pd.DataFrame({s: d["close"] * d["volume"] for s, d in daily.items() if not d.empty})
    if volume.empty:
        return []
    volume = volume[volume.index >= pd.Timestamp(start).normalize() - pd.Timedelta(days=1)]
    ranks = volume.where(volume >= min_quote_volume).rank(axis=1, ascending=False, method="first")
    ever_top = ranks.le(top_n).any(axis=0)
    chosen = [s for s in volume.columns if ever_top[s]]
    totals = volume[chosen].sum(axis=0)
    return sorted(chosen, key=lambda s: -totals[s])


def fetch_history(exchange: Any, symbol: str, timeframe: str, since_ms: int, now_ms: int | None = None) -> pd.DataFrame:
    """Все закрытые свечи с since_ms до текущего момента, постранично по 1000 штук."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    step = timeframe_minutes(timeframe) * 60_000
    rows: list[list[float]] = []
    cursor = since_ms
    while cursor < now_ms:
        batch = with_retries(
            lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000),
            what=f"история {symbol}",
            attempts=4,
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return ohlcv_to_frame(rows, timeframe, now_ms=now_ms)


def format_result(result: BacktestResult, title: str = "") -> str:
    r = result
    trades = len(r.trades)
    pf = "∞" if math.isinf(r.profit_factor) else f"{r.profit_factor:.2f}"
    lines = [
        f"=== {title or 'текущие настройки'} ===",
        f"Сделок: {trades} ({trades / r.weeks:.1f} в неделю), сигналов: {r.signals}",
        f"Прибыльных: {r.win_rate:.0f}%, средняя сделка {r.avg_trade_pct:+.2f}% (после комиссий), profit factor {pf}",
        f"Итог: {r.return_pct:+.2f}% ({r.config.start_usdt:,.0f} → {r.final_equity:,.0f} USDT), "
        f"макс. просадка {r.max_drawdown_pct:.2f}%",
    ]
    if trades:
        avg_hold = sum(t.hold_hours for t in r.trades) / trades
        reasons = ", ".join(f"{EXIT_NAMES.get(k, k)} {v}" for k, v in sorted(r.exit_reasons.items(), key=lambda x: -x[1]))
        lines.append(f"Среднее удержание: {avg_hold:.0f} ч; выходы: {reasons}")
    if r.skipped:
        lines.append("Пропущено сигналов: " + ", ".join(f"{k} {v}" for k, v in r.skipped.items()))
    if r.config.regime_filter and r.equity:
        lines.append(f"Фильтр рынка запрещал входы {r.regime_off_hours / len(r.equity):.0%} времени")
    if r.kill_switch_times:
        when = ", ".join(f"{t:%Y-%m-%d}" for t in r.kill_switch_times)
        lines.append(f"⛔ Аварийная остановка по просадке: {len(r.kill_switch_times)} раз ({when})")
    if r.trades:
        ranked = sorted(r.pnl_by_symbol.items(), key=lambda x: -x[1][1])
        fmt = lambda items: ", ".join(f"{s.split('/')[0]} {p:+,.0f}" for s, (_, p) in items)  # noqa: E731
        lines.append(f"Лучшие пары (USDT): {fmt(ranked[:5])}")
        lines.append(f"Худшие пары (USDT): {fmt(ranked[-5:][::-1])}")
    return "\n".join(lines)


def format_comparison(results: list[tuple[str, BacktestResult]]) -> str:
    header = f"{'вариант':<34} {'сделок':>6} {'в нед.':>6} {'приб.%':>6} {'сред.%':>7} {'PF':>5} {'итог%':>7} {'просадка%':>9}"
    lines = [header, "-" * len(header)]
    for name, r in results:
        pf = "∞" if math.isinf(r.profit_factor) else f"{r.profit_factor:.2f}"
        lines.append(
            f"{name[:34]:<34} {len(r.trades):>6} {len(r.trades) / r.weeks:>6.1f} {r.win_rate:>6.0f} "
            f"{r.avg_trade_pct:>+7.2f} {pf:>5} {r.return_pct:>+7.2f} {r.max_drawdown_pct:>9.2f}"
        )
    return "\n".join(lines)


class CandleCache:
    """Свечи на диске (data/candles/<пара>_<таймфрейм>.csv.gz): следующие запуски докачивают только новое.

    Папка data/ в .gitignore. В облачной сессии кэш живёт до конца сессии, на своём компьютере — постоянно.
    """

    def __init__(self, directory: Path | str | None):
        self.dir = Path(directory) if directory else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, timeframe: str) -> Path:
        assert self.dir is not None
        return self.dir / f"{symbol.replace('/', '_')}_{timeframe}.csv.gz"

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        if not self.dir:
            return None
        path = self._path(symbol, timeframe)
        if not path.exists():
            return None
        try:
            frame = pd.read_csv(path, index_col=0)
        except Exception as exc:  # noqa: BLE001 — битый файл просто качаем заново
            log.debug("кэш %s не читается: %s", path, exc)
            return None
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = "ts"
        return frame.astype(float)

    def save(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        if self.dir:
            frame.to_csv(self._path(symbol, timeframe))

    def history(
        self, exchange: Any, symbol: str, timeframe: str, since_ms: int, now_ms: int | None = None
    ) -> pd.DataFrame:
        """Свечи с since_ms: из кэша, докачивая только недостающий хвост (или всё, если кэш короче)."""
        cached = self.load(symbol, timeframe)
        since_ts = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        step = timeframe_minutes(timeframe) * 60_000
        if cached is not None and not cached.empty and cached.index[0] <= since_ts:
            tail_since = int(cached.index[-1].timestamp() * 1000) + step
            fresh = fetch_history(exchange, symbol, timeframe, tail_since, now_ms)
            frame = pd.concat([cached, fresh]) if not fresh.empty else cached
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        else:
            frame = fetch_history(exchange, symbol, timeframe, since_ms, now_ms)
        if not frame.empty:
            self.save(symbol, timeframe, frame)
        return frame[frame.index >= since_ts]


def fetch_many(
    make_exchange: Callable[[], Any],
    symbols: list[str],
    timeframe: str,
    since_ms: int,
    *,
    cache: CandleCache | None = None,
    workers: int = 6,
    progress: Callable[[int, int], None] | None = None,
    now_ms: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Свечи по многим парам в несколько потоков (у каждого потока свой клиент ccxt).

    Возвращает (свечи в порядке symbols, ошибки по парам). 6 потоков укладываются в лимит
    Binance на публичные запросы с запасом.
    """
    cache = cache or CandleCache(None)
    local = threading.local()
    lock = threading.Lock()
    done = 0

    def one(symbol: str) -> tuple[str, pd.DataFrame | None, str]:
        nonlocal done
        if not hasattr(local, "exchange"):
            local.exchange = make_exchange()
        try:
            frame, error = cache.history(local.exchange, symbol, timeframe, since_ms, now_ms), ""
        except Exception as exc:  # noqa: BLE001
            frame, error = None, str(exc)
        with lock:
            done += 1
            if progress:
                progress(done, len(symbols))
        return symbol, frame, error

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, symbols))
    frames = {s: f for s, f, _ in results if f is not None}
    errors = {s: e for s, _, e in results if e}
    return frames, errors
