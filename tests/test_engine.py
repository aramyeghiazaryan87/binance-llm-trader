"""Полный цикл бота офлайн: фейковая биржа + paper-исполнение + фейковый Claude."""
from dataclasses import replace

import pytest

from trader.broker import PaperBroker
from trader.budget import BudgetGuard
from trader.engine import TradingEngine
from trader.llm import LLMAdvisor
from trader.market_data import MarketData
from trader.storage import Storage

from .conftest import Clock, FakeClaude, FakeExchange, FakeNews, FakeNotifier, approving_responder, make_candles


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
