"""Главный цикл бота.

* Раз в час (после закрытия часовой свечи): данные -> сигналы -> LLM -> риск-менеджер -> ордера.
* Каждые 5 минут: проверка стопов и лимитов без LLM (бесплатно).
* Раз в день: отчёт.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .broker import Broker, BrokerError
from .config import Settings
from .llm import DecisionContext, LLMAdvisor, LLMResult, PositionView, SignalVerdict
from .market_data import MarketData
from .news import NewsProvider
from .notifier import Notifier
from .reports import EXIT_REASONS, daily_report_text, status_text
from .risk import AccountState, RiskLimits, RiskManager
from .storage import Position, Shadow, Storage, from_iso, to_iso, utcnow
from .strategy import (
    EntrySignal,
    Features,
    StrategyParams,
    compute_features,
    entry_signal,
    evaluate_exit,
    regime_ok,
    timeframe_minutes,
)

log = logging.getLogger(__name__)


@dataclass
class CycleSummary:
    time: datetime
    universe: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    llm_status: str = ""
    llm_report: str = ""
    llm_cost: float = 0.0
    regime: str = ""  # состояние фильтра рынка, пусто если фильтр выключен


class TradingEngine:
    def __init__(
        self,
        *,
        settings: Settings,
        storage: Storage,
        broker: Broker,
        data: MarketData,
        advisor: LLMAdvisor | None,
        news: NewsProvider,
        notifier: Notifier,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.s = settings
        self.storage = storage
        self.broker = broker
        self.data = data
        self.advisor = advisor
        self.news = news
        self.notifier = notifier
        self.clock = clock
        self.params = StrategyParams(
            breakout_lookback=settings.breakout_lookback,
            min_volume_ratio=settings.min_volume_ratio,
            atr_stop_mult=settings.atr_stop_mult,
            atr_trail_mult=settings.atr_trail_mult,
        )
        self.risk = RiskManager(
            RiskLimits(
                risk_per_trade_pct=settings.risk_per_trade_pct,
                max_position_pct=settings.max_position_pct,
                max_open_positions=settings.max_open_positions,
                max_total_exposure_pct=settings.max_total_exposure_pct,
                daily_loss_limit_pct=settings.daily_loss_limit_pct,
                max_drawdown_pct=settings.max_drawdown_pct,
                min_order_usdt=settings.min_order_usdt,
            )
        )
        self._stop = False
        self._deviation_check = settings.market_data_source == "mainnet" and settings.mode in ("testnet", "demo")

    # ------------------------------------------------------------------ запуск
    def startup(self) -> None:
        self.data.load_markets()
        self.broker.load_markets()
        self._init_capital()
        self.reconcile()

    def _init_capital(self) -> None:
        if self.storage.get("starting_capital") is not None:
            return
        quote = self.s.quote_asset
        free = self.broker.free_balance(quote)
        capital = min(free, self.s.budget_usdt) if self.s.budget_usdt > 0 else free
        if capital < self.s.min_order_usdt * 2:
            raise RuntimeError(
                f"на счёте {free:.2f} {quote}: этого мало для торговли. Пополните баланс "
                f"(на testnet баланс выдаётся бесплатно) или уменьшите MIN_ORDER_USDT."
            )
        capital = round(capital, 2)
        self.storage.set("starting_capital", capital)
        self.storage.set("peak_equity", capital)
        self.notifier.send(f"Стартовый капитал бота: {capital:,.2f} {quote} (свободно на счёте {free:,.2f})")

    def reconcile(self) -> None:
        """Сверка с биржей: позиция в базе есть, а монет на счёте нет (сброс testnet, ручная продажа)."""
        for pos in self.storage.open_positions():
            base = pos.symbol.split("/")[0]
            try:
                free = self.broker.free_balance(base)
            except Exception as exc:  # noqa: BLE001
                log.warning("сверка %s не удалась: %s", pos.symbol, exc)
                continue
            if free < pos.qty * 0.5:
                price = self._safe_exec_price(pos.symbol) or pos.entry_price
                self._book_close(pos, price=price, proceeds=pos.qty * price, reason="reconcile", fee=0.0, order_id=None)
                self.notifier.send(
                    f"⚠️ {pos.symbol}: в базе {pos.qty:.6g} {base}, на бирже {free:.6g}. "
                    "Позиция закрыта в учёте по текущей цене без ордера (сверка)."
                )

    # ------------------------------------------------------------------ учёт
    def _safe_exec_price(self, symbol: str) -> float | None:
        try:
            return self.broker.price(symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("нет цены %s на бирже исполнения: %s", symbol, exc)
            return None

    def _cooldowns(self) -> dict[str, datetime]:
        raw = self.storage.get("cooldowns", {}) or {}
        now = self.clock()
        result = {s: from_iso(t) for s, t in raw.items()}
        return {s: t for s, t in result.items() if t and t > now}

    def _set_cooldown(self, symbol: str, until: datetime) -> None:
        raw = {s: to_iso(t) for s, t in self._cooldowns().items()}
        raw[symbol] = to_iso(until)
        self.storage.set("cooldowns", raw)

    def account_state(self, exec_prices: dict[str, float] | None = None) -> AccountState:
        positions = self.storage.open_positions()
        if exec_prices is None:
            exec_prices = {}
            if positions:
                try:
                    exec_prices = self.broker.prices([p.symbol for p in positions])
                except Exception as exc:  # noqa: BLE001
                    log.warning("цены для оценки позиций недоступны: %s", exc)
        start = float(self.storage.get("starting_capital", 0.0) or 0.0)
        cash = start + self.storage.realized_pnl() - sum(p.cost_usdt for p in positions)
        value = sum(p.qty * exec_prices.get(p.symbol, p.entry_price) for p in positions)
        equity = cash + value

        today = self.clock().date().isoformat()
        if self.storage.get("day_date") != today:
            self.storage.set("day_date", today)
            self.storage.set("day_start_equity", equity)
        day_start = float(self.storage.get("day_start_equity", equity) or equity)
        day_pnl = (equity / day_start - 1.0) * 100.0 if day_start > 0 else 0.0

        peak = max(float(self.storage.get("peak_equity", equity) or equity), equity)
        self.storage.set("peak_equity", peak)
        drawdown = (1.0 - equity / peak) * 100.0 if peak > 0 else 0.0

        daily_blocked = self.storage.get("daily_block_date") == today
        if not daily_blocked and self.risk.daily_loss_hit(day_pnl):
            self.storage.set("daily_block_date", today)
            daily_blocked = True
            self.notifier.send(
                f"⏸ Дневной лимит убытка: {day_pnl:+.2f}% (лимит -{self.s.daily_loss_limit_pct}%). "
                "Новых входов до конца дня (UTC) не будет, открытые позиции защищают стопы."
            )
        return AccountState(
            equity=equity,
            cash=cash,
            positions_value=value,
            open_positions=len(positions),
            day_pnl_pct=day_pnl,
            drawdown_pct=drawdown,
            kill_switch=bool(self.storage.get("kill_switch", False)),
            daily_blocked=daily_blocked,
        )

    def _enforce_limits(self, account: AccountState) -> None:
        if account.kill_switch or not self.risk.drawdown_hit(account.drawdown_pct):
            return
        self.storage.set("kill_switch", True)
        self.notifier.send(
            f"⛔ АВАРИЙНАЯ ОСТАНОВКА: просадка {account.drawdown_pct:.2f}% от пика "
            f"(лимит {self.s.max_drawdown_pct}%). Закрываю все позиции, новые сделки запрещены. "
            "Разберитесь в причинах, затем: python -m trader reset-kill-switch"
        )
        self.close_all("kill_switch")

    # ------------------------------------------------------------------ сделки
    def _book_close(
        self, pos: Position, *, price: float, proceeds: float, reason: str, fee: float, order_id: str | None, qty: float | None = None
    ) -> None:
        now = self.clock()
        pnl = proceeds - pos.cost_usdt
        pos.status = "closed"
        pos.exit_price = price
        pos.exit_time = now
        pos.exit_reason = reason
        pos.proceeds_usdt = proceeds
        pos.pnl_usdt = pnl
        pos.pnl_pct = pnl / pos.cost_usdt * 100.0 if pos.cost_usdt else 0.0
        self.storage.update_position(pos)
        if order_id is not None:
            self.storage.add_trade(
                time=now, position_id=pos.id, symbol=pos.symbol, side="sell", qty=qty if qty is not None else pos.qty,
                price=price, quote_amount=proceeds + fee, fee_usdt=fee, order_id=order_id, mode=self.s.mode,
            )
        if reason == "stop_loss" and self.s.cooldown_hours > 0:
            self._set_cooldown(pos.symbol, now + timedelta(hours=self.s.cooldown_hours))

    def close_position(self, pos: Position, reason: str, note: str = "") -> bool:
        base = pos.symbol.split("/")[0]
        try:
            free = self.broker.free_balance(base)
        except Exception as exc:  # noqa: BLE001
            self.notifier.error(f"{pos.symbol}: не удалось получить баланс для продажи: {exc}")
            return False
        qty = self.broker.round_qty(pos.symbol, min(pos.qty, free))
        price_now = self._safe_exec_price(pos.symbol)
        min_cost = self.broker.min_order_cost(pos.symbol)
        if qty <= 0 or (price_now and qty * price_now < min_cost):
            price = price_now or pos.entry_price
            self._book_close(pos, price=price, proceeds=pos.qty * price, reason="reconcile", fee=0.0, order_id=None)
            self.notifier.send(f"⚠️ {pos.symbol}: продавать нечего или остаток меньше минимального ордера; позиция закрыта в учёте.")
            return True
        try:
            fill = self.broker.market_sell(pos.symbol, qty)
        except BrokerError as exc:
            self.notifier.error(f"{pos.symbol}: продажа не удалась ({EXIT_REASONS.get(reason, reason)}): {exc}")
            return False
        self._book_close(
            pos, price=fill.avg_price, proceeds=fill.net_quote, reason=reason, fee=fill.fee_usdt, order_id=fill.order_id, qty=fill.qty
        )
        icon = "✅" if (pos.pnl_usdt or 0) >= 0 else "🔻"
        text = (
            f"{icon} ПРОДАЖА {pos.symbol}: {fill.qty:.6g} по {fill.avg_price:.6g}, "
            f"результат {pos.pnl_usdt:+.2f} USDT ({pos.pnl_pct:+.2f}%). Причина: {EXIT_REASONS.get(reason, reason)}"
        )
        if note:
            text += f" — {note}"
        self.notifier.send(text)
        return True

    def close_all(self, reason: str = "manual") -> int:
        closed = 0
        for pos in self.storage.open_positions():
            if self.close_position(pos, reason):
                closed += 1
        return closed

    def _open_position(self, sig: EntrySignal, confidence: float | None, size_factor: float, why: str, held: set[str]) -> str | None:
        now = self.clock()
        account = self.account_state()
        gate = self.risk.can_open(account, sig.symbol, held, self._cooldowns(), now)
        if not gate.ok:
            return f"{sig.symbol}: {gate.reason}"

        try:
            ref_price = self.data.price(sig.symbol)
        except Exception as exc:  # noqa: BLE001
            return f"{sig.symbol}: нет текущей цены ({exc})"
        if ref_price > sig.price * 1.03:
            return f"{sig.symbol}: цена ушла на {ref_price / sig.price * 100 - 100:.1f}% выше сигнала, вход пропущен"
        if self._deviation_check:
            exec_price = self._safe_exec_price(sig.symbol)
            if exec_price is None:
                return f"{sig.symbol}: нет цены на бирже исполнения"
            deviation = abs(exec_price / ref_price - 1.0) * 100.0
            if deviation > self.s.max_price_deviation_pct:
                return (
                    f"{sig.symbol}: цена на {self.s.mode} ({exec_price:.6g}) отличается от mainnet "
                    f"({ref_price:.6g}) на {deviation:.1f}%, вход пропущен"
                )

        stop = ref_price - self.params.atr_stop_mult * sig.atr
        try:
            free_quote = self.broker.free_balance(self.s.quote_asset)
        except Exception as exc:  # noqa: BLE001
            return f"{sig.symbol}: баланс недоступен ({exc})"
        sizing = self.risk.size_position(
            account,
            price=ref_price,
            stop_price=stop,
            size_factor=size_factor,
            min_cost=self.broker.min_order_cost(sig.symbol),
            free_quote=free_quote,
        )
        if not sizing.ok:
            return f"{sig.symbol}: {sizing.reason}"

        try:
            fill = self.broker.market_buy(sig.symbol, sizing.amount_usdt)
        except BrokerError as exc:
            self.notifier.error(f"{sig.symbol}: покупка не удалась: {exc}")
            return f"{sig.symbol}: ошибка покупки"
        if fill.qty <= 0:
            return f"{sig.symbol}: исполнено 0"

        pos = Position(
            symbol=sig.symbol,
            qty=fill.qty,
            entry_price=fill.net_quote / fill.qty,
            cost_usdt=fill.net_quote,
            entry_ref_price=ref_price,
            stop_ref_price=stop,
            highest_ref_price=ref_price,
            atr_at_entry=sig.atr,
            entry_time=now,
            signal_id=sig.id,
            llm_confidence=confidence,
        )
        self.storage.add_position(pos)
        self.storage.add_trade(
            time=now, position_id=pos.id, symbol=sig.symbol, side="buy", qty=fill.qty, price=fill.avg_price,
            quote_amount=fill.quote_amount, fee_usdt=fill.fee_usdt, order_id=fill.order_id, mode=self.s.mode,
        )
        held.add(sig.symbol)
        conf_text = f", уверенность LLM {confidence:.0%}" if confidence is not None else ""
        self.notifier.send(
            f"🟢 ПОКУПКА {sig.symbol}: {fill.qty:.6g} по {fill.avg_price:.6g} на {fill.net_quote:.2f} USDT. "
            f"Стоп {stop:.6g} (-{(1 - stop / ref_price) * 100:.1f}%){conf_text}.\nСигнал: {sig.reason}\nLLM: {why}"
        )
        return None

    # ------------------------------------------------------------------ циклы
    def run_risk_check(self) -> list[str]:
        """Стопы и лимиты. LLM не используется."""
        closed: list[str] = []
        positions = self.storage.open_positions()
        if positions:
            try:
                ref_prices = self.data.prices([p.symbol for p in positions])
            except Exception as exc:  # noqa: BLE001
                log.warning("проверка стопов: цены недоступны: %s", exc)
                ref_prices = {}
            now = self.clock()
            for pos in positions:
                price = ref_prices.get(pos.symbol)
                if not price:
                    continue
                decision = evaluate_exit(
                    entry_price=pos.entry_ref_price,
                    stop_price=pos.stop_ref_price,
                    highest_price=pos.highest_ref_price,
                    atr_at_entry=pos.atr_at_entry,
                    price=price,
                    held_hours=(now - pos.entry_time).total_seconds() / 3600.0,
                    params=self.params,
                    max_hold_hours=self.s.max_hold_hours,
                )
                if decision.stop_price != pos.stop_ref_price or decision.highest_price != pos.highest_ref_price:
                    pos.stop_ref_price = decision.stop_price
                    pos.highest_ref_price = decision.highest_price
                    self.storage.update_position(pos)
                if decision.should_exit and self.close_position(pos, decision.reason):
                    closed.append(f"{pos.symbol} ({EXIT_REASONS.get(decision.reason, decision.reason)})")
        try:
            self.update_shadows()
        except Exception:  # noqa: BLE001 — учёт теней не должен мешать стопам
            log.exception("ошибка учёта отклонённых сигналов")
        self._enforce_limits(self.account_state())
        return closed

    def _prior_verdicts(self, signal_ids: set[str]) -> dict[str, SignalVerdict]:
        """Последние решения Claude по этим сигналам из журнала. id сигнала содержит время свечи,
        поэтому найденное решение относится к тем же данным, а не к прошлому часу."""
        found: dict[str, SignalVerdict] = {}
        for decision in self.storage.decisions(kind="llm", limit=24):
            for raw in decision["payload"].get("verdicts") or []:
                sid = raw.get("signal_id")
                if sid in signal_ids and sid not in found:
                    try:
                        found[sid] = SignalVerdict.model_validate(raw)
                    except ValueError:
                        continue
        if found:
            log.info("Claude уже оценивал %s: беру его прошлое решение", ", ".join(sorted(found)))
        return found

    # ------------------------------------------------------------------ тень
    def _shadow(self, sig: EntrySignal, source: str, reason: str, confidence: float | None = None) -> None:
        """Запомнить отклонённый сигнал, чтобы потом посчитать, чем закончилась бы сделка."""
        if any(sh.symbol == sig.symbol for sh in self.storage.shadows(status="open")):
            return  # как и бот, вторую позицию по той же паре не открываем
        shadow = Shadow(
            signal_id=sig.id,
            symbol=sig.symbol,
            source=source,
            entry_time=self.clock(),
            entry_price=sig.price,
            stop_price=sig.stop_price,
            highest_price=sig.price,
            atr=sig.atr,
            reason=reason,
            confidence=confidence,
        )
        if self.storage.add_shadow(shadow):
            log.info("тень: %s отклонён (%s), следим, чем закончилась бы сделка", sig.symbol, source)

    def update_shadows(self) -> list[Shadow]:
        """Те же правила выхода, что у настоящих позиций, по тем же ценам. Ордеров нет."""
        shadows = self.storage.shadows(status="open")
        if not shadows:
            return []
        try:
            prices = self.data.prices(sorted({sh.symbol for sh in shadows}))
        except Exception as exc:  # noqa: BLE001
            log.warning("тень: цены недоступны: %s", exc)
            return []
        now = self.clock()
        costs_pct = 2 * (self.s.paper_fee_pct + self.s.paper_slippage_pct)  # вход и выход
        closed: list[Shadow] = []
        for sh in shadows:
            price = prices.get(sh.symbol)
            if not price:
                continue
            decision = evaluate_exit(
                entry_price=sh.entry_price,
                stop_price=sh.stop_price,
                highest_price=sh.highest_price,
                atr_at_entry=sh.atr,
                price=price,
                held_hours=(now - sh.entry_time).total_seconds() / 3600.0,
                params=self.params,
                max_hold_hours=self.s.max_hold_hours,
            )
            sh.stop_price, sh.highest_price = decision.stop_price, decision.highest_price
            if decision.should_exit:
                sh.status, sh.exit_time, sh.exit_price, sh.exit_reason = "closed", now, price, decision.reason
                sh.pnl_pct = (price / sh.entry_price - 1.0) * 100.0 - costs_pct
                closed.append(sh)
                log.info("тень: %s закрыта бы (%s) %+.2f%%", sh.symbol, decision.reason, sh.pnl_pct)
            self.storage.update_shadow(sh)
        return closed

    def market_regime(self) -> tuple[bool, str]:
        """(можно ли открывать сделки, пояснение). При ошибке данных входы запрещены."""
        sym, period = self.s.regime_symbol, self.s.regime_ema
        try:
            daily = self.data.ohlcv(sym, "1d", max(400, period * 2))
        except Exception as exc:  # noqa: BLE001
            log.warning("фильтр рынка: дневные свечи %s недоступны: %s", sym, exc)
            return False, f"нет данных {sym}, входы запрещены"
        base = sym.split("/")[0]
        if regime_ok(daily, period):
            return True, f"{base} выше дневной EMA{period}"
        return False, f"{base} ниже дневной EMA{period}, новых входов нет"

    def _collect_features(self, symbols: list[str]) -> dict[str, Features]:
        features: dict[str, Features] = {}
        for symbol in symbols:
            try:
                frame = self.data.ohlcv(symbol, self.s.timeframe, self.s.candles_limit)
                f = compute_features(symbol, frame, self.params, self.s.timeframe)
            except Exception as exc:  # noqa: BLE001
                log.warning("свечи %s недоступны: %s", symbol, exc)
                continue
            if f is not None:
                features[symbol] = f
        return features

    def _ask_llm(
        self,
        now: datetime,
        account: AccountState,
        candidates: list[Features],
        signals: list[EntrySignal],
        positions: list[Position],
        features: dict[str, Features],
    ) -> LLMResult:
        if not self.s.llm_enabled or self.advisor is None:
            return LLMResult(status="disabled", detail="LLM выключен или нет ANTHROPIC_API_KEY / TRADER_ANTHROPIC_API_KEY")
        if account.kill_switch:
            return LLMResult(status="skipped", detail="аварийная остановка включена, LLM не нужен")
        if not candidates and not positions:
            return LLMResult(status="skipped", detail="нет рыночных данных для анализа")
        last_ok = self.storage.last_llm_call(status="ok")
        if last_ok:
            age_min = (now - from_iso(last_ok["time"])).total_seconds() / 60.0
            if age_min < self.s.llm_min_minutes_between_calls:
                return LLMResult(status="skipped", detail=f"прошлый вызов LLM был {age_min:.0f} мин назад")

        try:
            ref_prices = self.data.prices([p.symbol for p in positions]) if positions else {}
        except Exception:  # noqa: BLE001
            ref_prices = {}
        views = []
        for p in positions:
            price = ref_prices.get(p.symbol) or (features[p.symbol].price if p.symbol in features else p.entry_ref_price)
            views.append(PositionView.from_position(p, price, now))
        headlines = []
        try:
            headlines = self.news.headlines(now)
        except Exception as exc:  # noqa: BLE001
            log.warning("новости недоступны: %s", exc)
        fear_greed = None
        try:
            fear_greed = self.news.fear_greed()
        except Exception as exc:  # noqa: BLE001
            log.warning("индекс страха недоступен: %s", exc)
        ctx = DecisionContext(
            now=now,
            mode=self.s.mode,
            account=account,
            max_positions=self.s.max_open_positions,
            candidates=candidates,
            signals=signals,
            positions=views,
            headlines=headlines,
            fear_greed=fear_greed,
            news_lookback_hours=self.s.news_lookback_hours,
        )
        try:
            return self.advisor.decide(ctx)
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM: непредвиденная ошибка")
            return LLMResult(status="error", detail=str(exc))

    def run_hourly_cycle(self) -> CycleSummary:
        now = self.clock()
        summary = CycleSummary(time=now)
        self.storage.set("last_cycle", to_iso(now))

        # 1. стопы по свежим ценам
        summary.closed.extend(self.run_risk_check())

        # 2. universe: самые ликвидные пары mainnet, которые есть на бирже исполнения
        allowed = self.broker.symbols()
        universe = self.data.top_symbols(
            quote=self.s.quote_asset, limit=self.s.universe_size, min_quote_volume=self.s.min_quote_volume, allowed=allowed
        )
        summary.universe = universe
        positions = self.storage.open_positions()
        held = {p.symbol for p in positions}
        features = self._collect_features(list(dict.fromkeys(universe + sorted(held))))
        candidates = [features[s] for s in universe if s in features]

        # 3. сигналы стратегии
        cooldowns = self._cooldowns()
        signals = []
        for f in candidates:
            if f.symbol in held or f.symbol in cooldowns:
                continue
            sig = entry_signal(f, self.params)
            if sig is not None:
                signals.append(sig)
        summary.signals = [f"{s.symbol} ({s.id})" for s in signals]

        # 3a. фильтр рынка: на падающем рынке новых входов нет (LLM это не обходит)
        if self.s.regime_filter:
            allowed_by_regime, summary.regime = self.market_regime()
            log.info("фильтр рынка: %s", summary.regime)
            if not allowed_by_regime and signals:
                summary.rejected.extend(f"{s.symbol}: фильтр рынка ({summary.regime})" for s in signals)
                for sig in signals:
                    self._shadow(sig, "regime", summary.regime)
                signals = []
        log.info("universe=%d, свечи получены для %d, сигналов %d", len(universe), len(candidates), len(signals))

        # 4. LLM
        account = self.account_state()
        result = self._ask_llm(now, account, candidates, signals, positions, features)
        summary.llm_status = result.status
        summary.llm_cost = result.cost_usd
        if result.ok:
            a = result.assessment
            summary.llm_report = a.report
            self.storage.set("watchlist", {"time": to_iso(now), "symbols": [w.symbol for w in a.watchlist]})
            self.storage.add_decision(
                now,
                "llm",
                {
                    "regime": a.market_regime,
                    "news_sentiment": a.news_sentiment,
                    "news_summary": a.news_summary,
                    "watchlist": [w.model_dump() for w in a.watchlist],
                    "verdicts": [v.model_dump() for v in a.signal_verdicts],
                    "advice": [p.model_dump() for p in a.position_advice],
                    "report": a.report,
                    "cost_usd": result.cost_usd,
                    "model": result.model,
                    "effort": result.effort,
                    "notes": result.notes,
                },
            )
            log.info("LLM: режим=%s, новости=%+.2f, watchlist=%s, effort=%s, $%.4f", a.market_regime,
                     a.news_sentiment, [w.symbol for w in a.watchlist], result.effort, result.cost_usd)
        else:
            log.info("LLM: %s — %s", result.status, result.detail)

        # 5. советы LLM по открытым позициям (только закрытие = снижение риска)
        if result.ok:
            by_symbol = {p.symbol: p for p in self.storage.open_positions()}
            for advice in result.assessment.position_advice:
                pos = by_symbol.get(advice.symbol)
                if pos and advice.action == "close" and advice.confidence >= self.s.close_advice_min_confidence:
                    if self.close_position(pos, "llm_advice", note=advice.reason):
                        summary.closed.append(f"{pos.symbol} (совет LLM)")

        # 6. входы
        approved: list[tuple[EntrySignal, float | None, float, str]] = []
        if result.ok:
            verdicts = {v.signal_id: v for v in result.assessment.signal_verdicts}
        else:
            # Claude в этом цикле не спрашивали (перезапуск в том же часе, сбой API, бюджет).
            # Если он уже оценивал этот же сигнал (та же свеча), его решение в силе.
            verdicts = self._prior_verdicts({sig.id for sig in signals}) if signals else {}
        earlier = "" if result.ok else ", решение Claude из прошлого вызова"
        for sig in signals:
            v = verdicts.get(sig.id)
            if result.ok or v is not None:
                if v and v.approve and v.confidence >= self.s.min_llm_confidence:
                    approved.append((sig, v.confidence, v.size_factor, v.reason + earlier))
                else:
                    why = v.reason if v else "нет оценки"
                    if v and v.approve:
                        why = f"уверенность {v.confidence:.0%} ниже порога {self.s.min_llm_confidence:.0%}"
                    summary.rejected.append(f"{sig.symbol}: LLM отклонил ({why}{earlier})")
                    self._shadow(sig, "llm", why, v.confidence if v else None)
            elif self.s.llm_fail_mode == "rules":
                approved.append((sig, None, 0.5, "LLM недоступен, вход по правилам с половинным размером"))
            else:
                summary.rejected.append(f"{sig.symbol}: LLM недоступен ({result.status}), режим hold")
                self._shadow(sig, "hold", f"LLM недоступен ({result.status})")

        approved.sort(key=lambda item: item[1] or 0.0, reverse=True)
        held = {p.symbol for p in self.storage.open_positions()}
        for sig, confidence, size_factor, why in approved:
            problem = self._open_position(sig, confidence, size_factor, why, held)
            if problem:
                summary.rejected.append(problem)
            else:
                summary.opened.append(sig.symbol)

        # 7. учёт и уведомления
        account = self.account_state()
        self.storage.add_equity_snapshot(now, account.equity, account.cash, account.exposure_pct)
        self.storage.add_decision(
            now,
            "cycle",
            {
                "universe": summary.universe,
                "signals": summary.signals,
                "opened": summary.opened,
                "closed": summary.closed,
                "rejected": summary.rejected,
                "llm_status": summary.llm_status,
                "equity": account.equity,
            },
        )
        for line in summary.rejected:
            log.info("отклонено: %s", line)
        if self.s.notify_every_cycle or summary.opened or summary.closed:
            self.notifier.send(self._cycle_text(summary, account))
        return summary

    def _cycle_text(self, summary: CycleSummary, account: AccountState) -> str:
        lines = [f"⏱ Цикл {summary.time:%H:%M} UTC: капитал {account.equity:,.2f} USDT ({account.day_pnl_pct:+.2f}% за день)"]
        if summary.signals:
            lines.append("Сигналы: " + ", ".join(s.split(" ")[0] for s in summary.signals))
        if summary.opened:
            lines.append("Куплено: " + ", ".join(summary.opened))
        if summary.closed:
            lines.append("Продано: " + ", ".join(summary.closed))
        if summary.rejected:
            lines.append("Отклонено: " + "; ".join(summary.rejected[:5]))
        if summary.llm_report:
            lines.append(f"LLM: {summary.llm_report}")
        elif summary.llm_status:
            lines.append(f"LLM: {summary.llm_status}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ отчёты
    def status(self) -> str:
        positions = self.storage.open_positions()
        prices: dict[str, float] = {}
        if positions:
            try:
                prices = self.broker.prices([p.symbol for p in positions])
            except Exception as exc:  # noqa: BLE001
                log.warning("цены недоступны: %s", exc)
        account = self.account_state(prices)
        budget = self.advisor.budget if self.advisor else None
        if budget is None:
            from .budget import BudgetGuard

            budget = BudgetGuard(self.storage, self.s.llm_monthly_budget_usd)
        return status_text(
            mode=self.s.mode,
            account=account,
            starting_capital=float(self.storage.get("starting_capital", 0.0) or 0.0),
            positions=positions,
            prices=prices,
            budget=budget,
            now=self.clock(),
            kill_switch=account.kill_switch,
        )

    def send_daily_report(self) -> str:
        text = daily_report_text(storage=self.storage, status=self.status(), now=self.clock())
        self.notifier.send(text)
        self.storage.set("last_report_date", self.clock().date().isoformat())
        return text

    def maybe_daily_report(self) -> None:
        now = self.clock()
        if now.hour >= self.s.daily_report_hour_utc and self.storage.get("last_report_date") != now.date().isoformat():
            self.send_daily_report()

    # ------------------------------------------------------------------ главный цикл
    def next_cycle_time(self, now: datetime) -> datetime:
        step = timeframe_minutes(self.s.timeframe) * 60
        epoch = int(now.timestamp())
        boundary = (epoch // step + 1) * step
        return datetime.fromtimestamp(boundary + self.s.cycle_delay_seconds, tz=timezone.utc)

    def stop(self, *_: Any) -> None:
        log.info("получен сигнал остановки, завершаю после текущего шага")
        self._stop = True

    def run_forever(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.startup()
        self.notifier.send(
            f"🚀 Бот запущен. Режим {self.s.mode}, модель {self.s.llm_model if self.s.llm_enabled else 'выкл'}, "
            f"бюджет LLM ${self.s.llm_monthly_budget_usd:.2f}/мес.\n{self.status()}"
        )
        next_cycle = self.clock()  # первый цикл сразу
        next_risk = next_cycle + timedelta(seconds=self.s.risk_check_interval_seconds)
        while not self._stop:
            now = self.clock()
            if now >= next_cycle:
                try:
                    self.run_hourly_cycle()
                    next_cycle = self.next_cycle_time(now)
                except Exception as exc:  # noqa: BLE001
                    log.exception("ошибка часового цикла")
                    self.notifier.error(f"Ошибка часового цикла: {exc}. Повтор через 5 минут.")
                    next_cycle = now + timedelta(minutes=5)
                next_risk = self.clock() + timedelta(seconds=self.s.risk_check_interval_seconds)
            elif now >= next_risk:
                try:
                    self.run_risk_check()
                except Exception as exc:  # noqa: BLE001
                    log.exception("ошибка проверки стопов")
                    self.notifier.error(f"Ошибка проверки стопов: {exc}")
                next_risk = self.clock() + timedelta(seconds=self.s.risk_check_interval_seconds)
            try:
                self.maybe_daily_report()
            except Exception as exc:  # noqa: BLE001
                log.exception("ошибка отчёта")
                self.notifier.error(f"Ошибка дневного отчёта: {exc}")
            wake = min(next_cycle, next_risk)
            while not self._stop and self.clock() < wake:
                time.sleep(min(5.0, max(0.5, (wake - self.clock()).total_seconds())))
        self.notifier.send("🛑 Бот остановлен. Открытые позиции остаются на бирже без присмотра стопов.")
