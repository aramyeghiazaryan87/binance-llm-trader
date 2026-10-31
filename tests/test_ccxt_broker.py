from dataclasses import replace
from datetime import datetime, timezone

import ccxt
import pytest

from trader.broker import BrokerError, CcxtBroker
from trader.engine import TradingEngine


class FakeCcxt:
    def __init__(self):
        self.markets = {"SOL/USDT": {"symbol": "SOL/USDT", "base": "SOL", "quote": "USDT", "spot": True, "active": True,
                                     "limits": {"cost": {"min": 5.0}}}}
        self.calls = []

    def market(self, symbol):
        return self.markets[symbol]

    def amount_to_precision(self, symbol, amount):
        if amount < 0.001:
            raise ccxt.InvalidOrder("too small")
        return f"{int(amount * 1000) / 1000:.3f}"

    def create_market_buy_order_with_cost(self, symbol, cost):
        self.calls.append(("buy", symbol, cost))
        return {"id": "11", "status": "closed"}  # без filled -> бот должен перечитать ордер

    def fetch_order(self, order_id, symbol):
        return {"id": order_id, "filled": 2.0, "cost": 300.0, "average": 150.0,
                "fees": [{"currency": "SOL", "cost": 0.002}]}

    def create_market_sell_order(self, symbol, amount):
        self.calls.append(("sell", symbol, amount))
        if float(amount) > 100:
            raise ccxt.InsufficientFunds("not enough")
        return {"id": "12", "filled": float(amount), "cost": float(amount) * 160, "average": 160.0,
                "fees": [{"currency": "USDT", "cost": 0.3}]}


def test_ccxt_broker_buy_sell_and_errors():
    ex = FakeCcxt()
    broker = CcxtBroker(ex, "testnet")
    fill = broker.market_buy("SOL/USDT", 300.009)
    assert ex.calls[0] == ("buy", "SOL/USDT", 300.0)  # округление вниз до центов
    assert fill.qty == pytest.approx(1.998) and fill.net_quote == pytest.approx(300.0)
    sell = broker.market_sell("SOL/USDT", 1.9987)
    assert ex.calls[1] == ("sell", "SOL/USDT", 1.998)  # округление вниз до шага
    assert sell.net_quote == pytest.approx(1.998 * 160 - 0.3)
    assert broker.min_order_cost("SOL/USDT") == 5.0
    assert broker.round_qty("SOL/USDT", 0.0001) == 0.0
    with pytest.raises(BrokerError):
        broker.market_sell("SOL/USDT", 500)
    with pytest.raises(BrokerError):
        broker.market_sell("SOL/USDT", 0.0001)


def test_next_cycle_time_aligns_to_hour(settings):
    engine = TradingEngine.__new__(TradingEngine)
    engine.s = replace(settings, timeframe="1h", cycle_delay_seconds=90)
    t = engine.next_cycle_time(datetime(2026, 10, 31, 14, 1, 30, tzinfo=timezone.utc))
    assert t == datetime(2026, 10, 31, 15, 1, 30, tzinfo=timezone.utc)
    t = engine.next_cycle_time(datetime(2026, 10, 31, 23, 59, 59, tzinfo=timezone.utc))
    assert t == datetime(2026, 11, 1, 0, 1, 30, tzinfo=timezone.utc)
