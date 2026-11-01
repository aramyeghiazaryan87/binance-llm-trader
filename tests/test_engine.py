"""Полный цикл бота офлайн: фейковая биржа + paper-исполнение + фейковый Claude."""
import logging
from dataclasses import replace
from datetime import timedelta

import pytest

from trader.broker import PaperBroker
from trader.budget import BudgetGuard
from trader.engine import TradingEngine
from trader.llm import LLMAdvisor
from trader.market_data import MarketData
from trader.storage import Storage

from .conftest import (
    SIGNAL_RE,
    Clock,
    FakeClaude,
    FakeExchange,
    FakeNews,
    FakeNotifier,
    approving_responder,
    make_candles,
    rejecting_responder,
)


def build(settings, now, *, responder=None, storage=None, exchange=None):
    exchange = exchange or FakeExchange(
        {
            "SOL/USDT": make_candles(300, breakout=True),
            "BTC/USDT": make_candles(300, start=60_000),
            "ETH/USDT": make_candles(300, start=3_000, drift=-0.001),
        },
        volumes={"BTC/USDT": 5e9, "ETH/USDT": 3e9, "SOL/USDT": 1e9},
    )
    storage = storage or Storage(settings.db_path)
    data = MarketData(exchange)
    broker = PaperBroker(storage=storage, price_source=data.prices, markets_source=lambda: data.markets,
                         start_usdt=settings.paper_start_usdt, fee_pct=0.1, slippage_pct=0.05)
    advisor = None
    client = None
    if responder is not None:
        client = FakeClaude(responder)
        advisor = LLMAdvisor(storage=storage, budget=BudgetGuard(storage, settings.llm_monthly_budget_usd),
                             model=settings.llm_model, effort=settings.llm_effort, max_tokens=settings.llm_max_tokens,
                             effort_signals=settings.llm_effort_signals,
                             max_watchlist=settings.max_watchlist, client=client)
    clock = Clock(now)
    notifier = FakeNotifier()
    engine = TradingEngine(settings=settings, storage=storage, broker=broker, data=data, advisor=advisor,
                           news=FakeNews(), notifier=notifier, clock=clock)
    return engine, exchange, clock, notifier, client


def test_full_cycle_open_then_stop_out(settings, now):
    engine, exchange, clock, notifier, client = build(settings, now, responder=approving_responder(size_factor=0.8))
    engine.startup()
    assert engine.storage.get("starting_capital") == 10_000

    summary = engine.run_hourly_cycle()
    assert summary.universe == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert [s.split(" ")[0] for s in summary.signals] == ["SOL/USDT"]
    assert summary.opened == ["SOL/USDT"]
    assert summary.llm_status == "ok" and summary.llm_cost > 0
    assert len(client.calls) == 1

    [pos] = engine.storage.open_positions()
    assert pos.symbol == "SOL/USDT"
    # 20% капитала (лимит позиции) x size_factor 0.8 = 1600 USDT
    assert pos.cost_usdt == pytest.approx(1600.0)
    assert pos.stop_ref_price < pos.entry_ref_price
    assert any("ПОКУПКА SOL/USDT" in m for m in notifier.messages)

    # повторный цикл через 10 минут: LLM не вызывается (защита от лишних трат при перезапусках)
    clock.advance(minutes=10)
    again = engine.run_hourly_cycle()
    assert again.llm_status == "skipped" and len(client.calls) == 1

    # цена падает на 10% -> стоп-лосс
    exchange.prices["SOL/USDT"] *= 0.9
    clock.advance(minutes=5)
    closed = engine.run_risk_check()
    assert closed and closed[0].startswith("SOL/USDT")
    assert engine.storage.open_positions() == []
    [done] = engine.storage.closed_positions()
    assert done.exit_reason == "stop_loss"
    assert done.pnl_usdt == pytest.approx(-1600 * 0.1, rel=0.05)
    assert "SOL/USDT" in engine._cooldowns()  # пауза после стопа

    # через час сигнал по SOL ещё есть, но пара на паузе
    exchange.prices["SOL/USDT"] /= 0.9
    clock.advance(hours=1)
    later = engine.run_hourly_cycle()
    assert later.opened == []
    assert all(not s.startswith("SOL/USDT") for s in later.signals)

    # состояние переживает перезапуск
    engine2, *_ = build(settings, clock.now, storage=engine.storage, exchange=exchange)
    engine2.startup()
    assert engine2.storage.get("starting_capital") == 10_000
    assert engine2.account_state().equity == pytest.approx(10_000 + done.pnl_usdt, abs=0.01)


def test_hold_mode_without_llm_opens_nothing(settings, now):
    engine, *_ = build(replace(settings, llm_enabled=False, llm_fail_mode="hold"), now)
    engine.startup()
    summary = engine.run_hourly_cycle()
    assert summary.signals and summary.opened == []
    assert summary.llm_status == "disabled"


def test_rules_mode_without_llm_trades_half_size(settings, now):
    engine, *_ = build(replace(settings, llm_enabled=False, llm_fail_mode="rules"), now)
    engine.startup()
    summary = engine.run_hourly_cycle()
    assert summary.opened == ["SOL/USDT"]
    [pos] = engine.storage.open_positions()
    assert pos.cost_usdt == pytest.approx(1000.0)  # 2000 x 0.5


def test_llm_rejection_blocks_entry(settings, now):
    engine, *_ = build(settings, now, responder=approving_responder(confidence=0.4))
    engine.startup()
    summary = engine.run_hourly_cycle()
    assert summary.opened == []
    assert any("ниже порога" in r for r in summary.rejected)


def test_llm_close_advice_and_kill_switch(settings, now):
    s = replace(settings, max_drawdown_pct=1.0, daily_loss_limit_pct=50.0)
    engine, exchange, clock, notifier, _ = build(s, now, responder=approving_responder())
    engine.startup()
    assert engine.run_hourly_cycle().opened == ["SOL/USDT"]

    # следующий час: LLM советует закрыть позицию
    engine.advisor.client = FakeClaude(approving_responder(close_positions=True))
    clock.advance(hours=1)
    summary = engine.run_hourly_cycle()
    assert "SOL/USDT (совет LLM)" in summary.closed

    # искусственно поднимаем пик капитала -> просадка > 1% -> аварийная остановка
    engine.storage.set("peak_equity", 20_000)
    clock.advance(minutes=5)
    engine.run_risk_check()
    assert engine.storage.get("kill_switch") is True
    assert any("АВАРИЙНАЯ ОСТАНОВКА" in m for m in notifier.messages)
    clock.advance(hours=1)
    blocked = engine.run_hourly_cycle()
    assert blocked.opened == [] and blocked.llm_status == "skipped"


def test_daily_report_text(settings, now):
    engine, *_ = build(settings, now, responder=approving_responder())
    engine.startup()
    engine.run_hourly_cycle()
    text = engine.send_daily_report()
    assert "Отчёт за 24 часа" in text and "SOL/USDT" in text and "LLM за сутки: 1 вызовов" in text


def regime_exchange(btc_drift: float):
    return FakeExchange(
        {
            "SOL/USDT": make_candles(300, breakout=True),
            "BTC/USDT": make_candles(300, start=60_000, drift=btc_drift),
        },
        volumes={"BTC/USDT": 5e9, "SOL/USDT": 1e9},
    )


def test_regime_filter_blocks_entries_even_if_llm_approves(settings, now, caplog):
    caplog.set_level(logging.INFO, logger="trader.engine")
    settings = replace(settings, regime_filter=True, regime_ema=50)
    engine, _, _, _, client = build(settings, now, responder=approving_responder(), exchange=regime_exchange(-0.002))
    engine.startup()
    summary = engine.run_hourly_cycle()

    assert [s.split(" ")[0] for s in summary.signals] == ["SOL/USDT"]
    assert summary.opened == [] and engine.storage.open_positions() == []
    assert "BTC ниже дневной EMA50" in summary.regime
    assert "фильтр рынка: BTC ниже дневной EMA50" in caplog.text  # видно и в логе постоянного режима
    assert any(r.startswith("SOL/USDT: фильтр рынка") for r in summary.rejected)
    # Claude не тратит токены на сигнал, который всё равно запрещён
    assert SIGNAL_RE.findall(client.calls[0]["messages"][0]["content"]) == []


def test_regime_filter_allows_entries_in_uptrend(settings, now):
    settings = replace(settings, regime_filter=True, regime_ema=50)
    engine, *_ = build(settings, now, responder=approving_responder(), exchange=regime_exchange(0.002))
    engine.startup()
    summary = engine.run_hourly_cycle()
    assert summary.opened == ["SOL/USDT"]
    assert "BTC выше дневной EMA50" in summary.regime


def test_regime_filter_fails_safe_without_data(settings, now):
    settings = replace(settings, regime_filter=True, regime_symbol="NOPE/USDT")
    engine, *_ = build(settings, now, responder=approving_responder())
    engine.startup()
    summary = engine.run_hourly_cycle()
    assert summary.opened == []
    assert "входы запрещены" in summary.regime


def test_rejected_signal_is_tracked_in_shadow(settings, now):
    engine, exchange, clock, _, _ = build(settings, now, responder=rejecting_responder())
    engine.startup()
    summary = engine.run_hourly_cycle()
    assert summary.opened == [] and engine.storage.open_positions() == []

    [shadow] = engine.storage.shadows(status="open")
    assert (shadow.symbol, shadow.source, shadow.confidence) == ("SOL/USDT", "llm", 0.7)
    assert shadow.reason == "RSI 71 extended, mixed trend"
    assert shadow.stop_price < shadow.entry_price

    # повтор цикла в том же часе (Claude пропущен) не создаёт вторую тень
    clock.advance(minutes=10)
    engine.run_hourly_cycle()
    assert len(engine.storage.shadows()) == 1

    # цена падает на 10%: тень закрывается по стопу, как закрылась бы настоящая позиция
    entry = shadow.entry_price
    exchange.prices["SOL/USDT"] = entry * 0.9
    clock.advance(minutes=5)
    engine.run_risk_check()
    [closed] = engine.storage.shadows(status="closed")
    assert closed.exit_reason == "stop_loss"
    assert closed.pnl_pct == pytest.approx(-10.0 - 0.3)  # минус комиссия и проскальзывание на входе и выходе
    assert engine.storage.open_positions() == []  # ордеров не было

    report = engine.send_daily_report()
    assert "Отклонённые сигналы" in report
    assert "за сутки отклонено: Claude 1" in report
    assert "• отклонил Claude: 1, в среднем -10.30%" in report


def test_shadow_trailing_stop_keeps_profit(settings, now):
    engine, exchange, clock, _, _ = build(settings, now, responder=rejecting_responder())
    engine.startup()
    engine.run_hourly_cycle()
    [shadow] = engine.storage.shadows(status="open")
    entry, atr = shadow.entry_price, shadow.atr

    exchange.prices["SOL/USDT"] = entry + 10 * atr  # сильный рост: трейлинг подтягивается
    clock.advance(minutes=5)
    engine.run_risk_check()
    assert engine.storage.shadows(status="closed") == []

    exchange.prices["SOL/USDT"] = entry + 6 * atr  # откат ниже трейлинга (максимум - 3 ATR)
    clock.advance(minutes=5)
    engine.run_risk_check()
    [closed] = engine.storage.shadows(status="closed")
    assert closed.exit_reason == "trailing_stop" and closed.pnl_pct > 0


def test_regime_and_hold_rejections_are_tracked(settings, now):
    s = replace(settings, regime_filter=True, regime_ema=50)
    engine, *_ = build(s, now, responder=approving_responder(), exchange=regime_exchange(-0.002))
    engine.startup()
    engine.run_hourly_cycle()
    [shadow] = engine.storage.shadows()
    assert shadow.source == "regime" and "BTC ниже дневной EMA50" in shadow.reason

    separate_db = settings.data_dir / "hold"
    hold, *_ = build(replace(settings, data_dir=separate_db, llm_enabled=False, llm_fail_mode="hold"), now)
    hold.startup()
    hold.run_hourly_cycle()
    [held] = hold.storage.shadows()
    assert held.source == "hold" and held.confidence is None


def test_shadow_list_text(settings, now):
    from trader.reports import shadow_list_text

    engine, exchange, clock, _, _ = build(settings, now, responder=rejecting_responder())
    engine.startup()
    assert shadow_list_text(engine.storage, now) == "Отклонённых сигналов пока не было"
    engine.run_hourly_cycle()
    text = shadow_list_text(engine.storage, now)
    assert "SOL/USDT, отклонил Claude" in text and "ещё не закрыта" in text and "уверенность Claude 70%" in text


def remember_verdict(engine, now, **verdict):
    """Как будто Claude оценил текущий сигнал по SOL 10 минут назад, а потом бот перезапустили."""
    from trader.strategy import entry_signal

    sig = entry_signal(engine._collect_features(["SOL/USDT"])["SOL/USDT"], engine.params)
    earlier = now - timedelta(minutes=10)
    engine.storage.add_llm_call(time=earlier, model="claude-sonnet-5-5", status="ok", input_tokens=3700,
                                output_tokens=300, cost_usd=0.01, detail="neutral")
    engine.storage.add_decision(earlier, "llm", {"regime": "neutral", "verdicts": [{"signal_id": sig.id, **verdict}]})
    return sig


def test_restart_in_same_hour_reuses_claude_approval(settings, now):
    engine, _, _, notifier, client = build(settings, now, responder=approving_responder())
    engine.startup()
    remember_verdict(engine, now, approve=True, confidence=0.8, size_factor=0.5, reason="clean breakout")

    summary = engine.run_hourly_cycle()
    assert summary.llm_status == "skipped" and client.calls == []  # Claude повторно не спрашивали
    assert summary.opened == ["SOL/USDT"]
    [pos] = engine.storage.open_positions()
    assert pos.cost_usdt == pytest.approx(2000 * 0.5) and pos.llm_confidence == 0.8
    assert any("решение Claude из прошлого вызова" in m for m in notifier.messages)


def test_restart_in_same_hour_reuses_claude_rejection(settings, now):
    engine, _, _, _, client = build(settings, now, responder=approving_responder())
    engine.startup()
    remember_verdict(engine, now, approve=False, confidence=0.7, size_factor=0.25, reason="RSI 71 extended")

    summary = engine.run_hourly_cycle()
    assert client.calls == [] and summary.opened == []
    assert summary.rejected == ["SOL/USDT: LLM отклонил (RSI 71 extended, решение Claude из прошлого вызова)"]
    [shadow] = engine.storage.shadows()
    assert (shadow.source, shadow.confidence, shadow.reason) == ("llm", 0.7, "RSI 71 extended")


def test_verdict_for_another_candle_is_not_reused(settings, now):
    engine, *_ = build(settings, now, responder=approving_responder())
    engine.startup()
    sig = remember_verdict(engine, now, approve=True, confidence=0.9, size_factor=1.0, reason="old")
    # то же решение, но для прошлой свечи: подменяем id в журнале
    engine.storage.conn.execute("UPDATE decisions SET payload = replace(payload, ?, ?)", (sig.id, "SOLUSDT-202001010000"))
    engine.storage.conn.commit()

    summary = engine.run_hourly_cycle()
    assert summary.opened == []
    assert summary.rejected == ["SOL/USDT: LLM недоступен (skipped), режим hold"]
    assert engine.storage.shadows()[0].source == "hold"
