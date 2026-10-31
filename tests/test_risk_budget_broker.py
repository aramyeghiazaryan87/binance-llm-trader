from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trader.broker import PaperBroker, parse_order_fill
from trader.budget import BudgetGuard, tokens_cost, usage_cost
from trader.market_data import select_universe
from trader.risk import AccountState, RiskLimits, RiskManager
from trader.storage import Storage


def account(**kw):
    base = dict(equity=10_000.0, cash=10_000.0, positions_value=0.0, open_positions=0, day_pnl_pct=0.0,
                drawdown_pct=0.0, kill_switch=False, daily_blocked=False)
    base.update(kw)
    return AccountState(**base)


# ---------------- риск ----------------
def test_size_is_capped_by_max_position():
    rm = RiskManager(RiskLimits(risk_per_trade_pct=1.0, max_position_pct=20.0))
    d = rm.size_position(account(), price=100.0, stop_price=98.0, size_factor=1.0, min_cost=5.0, free_quote=1e9)
    # по риску 100 / 2% = 5000, но лимит позиции 20% = 2000
    assert d.ok and d.amount_usdt == pytest.approx(2000.0)


def test_size_by_risk_and_factor():
    rm = RiskManager(RiskLimits(risk_per_trade_pct=1.0, max_position_pct=50.0))
    d = rm.size_position(account(), price=100.0, stop_price=90.0, size_factor=0.5, min_cost=5.0, free_quote=1e9)
    # 100 / 10% = 1000, x0.5 = 500
    assert d.ok and d.amount_usdt == pytest.approx(500.0)


def test_size_respects_exposure_and_cash_and_minimum():
    rm = RiskManager(RiskLimits(max_total_exposure_pct=80.0, min_order_usdt=10.0))
    d = rm.size_position(account(positions_value=7_900.0, cash=2_100.0), price=100.0, stop_price=98.0,
                         size_factor=1.0, min_cost=5.0, free_quote=1e9)
    assert d.ok and d.amount_usdt == pytest.approx(100.0)  # осталось 8000 - 7900
    d = rm.size_position(account(cash=8.0), price=100.0, stop_price=98.0, size_factor=1.0, min_cost=5.0, free_quote=1e9)
    assert not d.ok
    d = rm.size_position(account(), price=100.0, stop_price=98.0, size_factor=1.0, min_cost=5.0, free_quote=50.0)
    assert d.ok and d.amount_usdt == pytest.approx(49.5)  # не больше реального баланса на бирже


def test_can_open_gates():
    rm = RiskManager(RiskLimits(max_open_positions=2))
    now = datetime.now(timezone.utc)
    assert rm.can_open(account(), "A/USDT", set(), {}, now).ok
    assert not rm.can_open(account(kill_switch=True), "A/USDT", set(), {}, now).ok
    assert not rm.can_open(account(daily_blocked=True), "A/USDT", set(), {}, now).ok
    assert not rm.can_open(account(open_positions=2), "A/USDT", set(), {}, now).ok
    assert not rm.can_open(account(), "A/USDT", {"A/USDT"}, {}, now).ok
    assert not rm.can_open(account(), "A/USDT", set(), {"A/USDT": now + timedelta(hours=1)}, now).ok


# ---------------- бюджет LLM ----------------
def test_usage_cost_simple_and_unknown_model():
    usage = SimpleNamespace(input_tokens=2_000, output_tokens=1_000, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0, iterations=None)
    cost, tin, tout = usage_cost(usage, "claude-sonnet-5-5")
    assert cost == pytest.approx(2_000 * 2 / 1e6 + 1_000 * 10 / 1e6)
    assert (tin, tout) == (2_000, 1_000)
    assert tokens_cost("some-new-model", 1_000_000, 0) == pytest.approx(10.0)  # консервативно


def test_usage_cost_counts_every_fallback_attempt():
    iterations = [
        SimpleNamespace(type="message", model="claude-sonnet-5-5", input_tokens=1000, output_tokens=0,
                        cache_creation_input_tokens=0, cache_read_input_tokens=0),
        SimpleNamespace(type="fallback_message", model="claude-opus-5-5", input_tokens=1000, output_tokens=500,
                        cache_creation_input_tokens=0, cache_read_input_tokens=0),
    ]
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0, iterations=iterations)
    cost, _, _ = usage_cost(usage, "claude-sonnet-5-5")
    assert cost == pytest.approx(1000 * 2 / 1e6 + 1000 * 4 / 1e6 + 500 * 20 / 1e6)


def test_budget_hard_cap_and_pacing(tmp_path):
    storage = Storage(tmp_path / "b.db")
    guard = BudgetGuard(storage, monthly_budget_usd=10.0)
    mid_month = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)  # ~половина месяца -> по графику ~$5.4
    assert guard.check(mid_month, 0.05, has_signals=False)[0]
    storage.add_llm_call(time=mid_month - timedelta(days=1), model="m", status="ok", input_tokens=1,
                         output_tokens=1, cost_usd=6.0)
    ok, why = guard.check(mid_month, 0.05, has_signals=False)
    assert not ok and "экономия" in why  # обгоняем график, сигналов нет
    assert guard.check(mid_month, 0.05, has_signals=True)[0]  # с сигналами можно
    storage.add_llm_call(time=mid_month, model="m", status="ok", input_tokens=1, output_tokens=1, cost_usd=3.97)
    ok, why = guard.check(mid_month, 0.05, has_signals=True)
    assert not ok and "исчерпан" in why  # 9.97 + 0.05 > 10
    # новый месяц — бюджет снова доступен
    assert guard.check(datetime(2026, 11, 1, 1, tzinfo=timezone.utc), 0.05, has_signals=False)[0]


# ---------------- биржа ----------------
def test_parse_buy_fill_with_fee_in_base():
    order = {"id": 1, "filled": 1.0, "cost": 100.0, "average": 100.0, "fees": [{"currency": "SOL", "cost": 0.001}]}
    fill = parse_order_fill(order, "buy", "SOL/USDT", "SOL", "USDT")
    assert fill.qty == pytest.approx(0.999)
    assert fill.fee_usdt == pytest.approx(0.1)
    assert fill.net_quote == pytest.approx(100.0)


def test_parse_sell_fill_with_fee_in_quote_and_bnb():
    order = {"id": 2, "filled": 0.5, "cost": 50.0, "average": 100.0, "fees": [{"currency": "USDT", "cost": 0.05}]}
    fill = parse_order_fill(order, "sell", "SOL/USDT", "SOL", "USDT")
    assert fill.net_quote == pytest.approx(49.95)
    order_bnb = {"id": 3, "filled": 0.5, "cost": 50.0, "average": 100.0, "fees": [{"currency": "BNB", "cost": 0.0001}]}
    fill = parse_order_fill(order_bnb, "sell", "SOL/USDT", "SOL", "USDT")
    assert fill.fee_usdt == pytest.approx(0.05)  # оценка 0.1% от суммы


def test_paper_broker_round_trip(tmp_path):
    storage = Storage(tmp_path / "p.db")
    markets = {"SOL/USDT": {"spot": True, "quote": "USDT", "precision": {"amount": 0.001}, "limits": {"cost": {"min": 5}}}}
    prices = {"SOL/USDT": 100.0}
    broker = PaperBroker(storage=storage, price_source=lambda syms: {s: prices[s] for s in syms},
                         markets_source=lambda: markets, start_usdt=1000.0, fee_pct=0.1, slippage_pct=0.0)
    buy = broker.market_buy("SOL/USDT", 500.0)
    assert buy.qty == pytest.approx(4.995)
    assert broker.free_balance("USDT") == pytest.approx(500.0)
    prices["SOL/USDT"] = 110.0
    sell = broker.market_sell("SOL/USDT", broker.round_qty("SOL/USDT", buy.qty))
    assert sell.net_quote == pytest.approx(4.995 * 110 * 0.999)
    assert broker.free_balance("SOL") == pytest.approx(0.0, abs=1e-9)


def test_select_universe_filters():
    markets = {
        s: {"spot": True, "active": True, "quote": "USDT", "base": s.split("/")[0]}
        for s in ["BTC/USDT", "ETH/USDT", "USDC/USDT", "BTCUP/USDT", "SOL/USDT", "LOW/USDT"]
    }
    markets["ETH/BTC"] = {"spot": True, "active": True, "quote": "BTC", "base": "ETH"}
    tickers = {s: {"quoteVolume": v} for s, v in {
        "BTC/USDT": 5e9, "ETH/USDT": 3e9, "USDC/USDT": 9e9, "BTCUP/USDT": 1e8, "SOL/USDT": 1e9,
        "LOW/USDT": 1e5, "ETH/BTC": 1e10}.items()}
    result = select_universe(tickers, markets, quote="USDT", limit=10, min_quote_volume=1e6, allowed=None)
    assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    result = select_universe(tickers, markets, quote="USDT", limit=10, min_quote_volume=1e6, allowed={"SOL/USDT", "ETH/USDT"})
    assert result == ["ETH/USDT", "SOL/USDT"]
