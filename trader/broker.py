"""Исполнение ордеров.

* CcxtBroker: реальные ордера через ccxt (testnet / demo / live).
* PaperBroker: симуляция на реальных ценах mainnet, без ключей и без ордеров на бирже.

Бот торгует только рыночными ордерами. Покупка задаётся суммой в USDT (quoteOrderQty),
продажа количеством монет.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import ccxt

from .storage import Storage

log = logging.getLogger(__name__)

# Если комиссия списана в третьей валюте (например, BNB), оцениваем её как 0.1% от суммы.
FALLBACK_FEE_RATE = 0.001


class BrokerError(Exception):
    pass


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str  # buy | sell
    qty: float  # buy: монет получено чистыми (после комиссии); sell: монет продано
    avg_price: float
    quote_amount: float  # оборот сделки в USDT (без учёта комиссии)
    fee_usdt: float
    net_quote: float  # buy: всего потрачено USDT; sell: получено USDT после комиссии
    order_id: str | None


class Broker(ABC):
    mode: str
    quote: str = "USDT"

    @abstractmethod
    def load_markets(self) -> None: ...

    @abstractmethod
    def symbols(self) -> set[str]: ...

    @abstractmethod
    def free_balance(self, asset: str) -> float: ...

    @abstractmethod
    def prices(self, symbols: list[str]) -> dict[str, float]: ...

    @abstractmethod
    def min_order_cost(self, symbol: str) -> float: ...

    @abstractmethod
    def round_qty(self, symbol: str, qty: float) -> float: ...

    @abstractmethod
    def market_buy(self, symbol: str, quote_amount: float) -> Fill: ...

    @abstractmethod
    def market_sell(self, symbol: str, qty: float) -> Fill: ...

    def price(self, symbol: str) -> float:
        return self.prices([symbol])[symbol]


def parse_order_fill(order: dict[str, Any], side: str, symbol: str, base: str, quote: str) -> Fill:
    """Ответ ccxt о рыночном ордере -> Fill с учётом комиссии."""
    filled = float(order.get("filled") or 0.0)
    cost = float(order.get("cost") or 0.0)
    average = order.get("average") or order.get("price")
    average = float(average) if average else (cost / filled if filled else 0.0)
    if filled <= 0 or cost <= 0 or average <= 0:
        raise BrokerError(f"ордер {order.get('id')} по {symbol} не исполнен: {order.get('status')}")

    fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
    fee_base = fee_quote = fee_other_usdt = 0.0
    for fee in fees:
        if not fee or fee.get("cost") in (None, ""):
            continue
        amount = float(fee["cost"])
        currency = fee.get("currency")
        if currency == base:
            fee_base += amount
        elif currency == quote:
            fee_quote += amount
        elif amount > 0:
            fee_other_usdt += cost * FALLBACK_FEE_RATE

    fee_usdt = fee_quote + fee_base * average + fee_other_usdt
    if side == "buy":
        qty = filled - fee_base
        net_quote = cost + fee_quote
    else:
        qty = filled
        net_quote = cost - fee_quote - fee_base * average
    return Fill(
        symbol=symbol,
        side=side,
        qty=qty,
        avg_price=average,
        quote_amount=cost,
        fee_usdt=fee_usdt,
        net_quote=net_quote,
        order_id=str(order.get("id")) if order.get("id") is not None else None,
    )


class CcxtBroker(Broker):
    def __init__(self, exchange: Any, mode: str, quote: str = "USDT"):
        self.exchange = exchange
        self.mode = mode
        self.quote = quote

    def load_markets(self) -> None:
        self.exchange.load_markets(reload=True)

    def symbols(self) -> set[str]:
        return {
            s
            for s, m in self.exchange.markets.items()
            if m.get("spot") and m.get("active") is not False and m.get("quote") == self.quote
        }

    def free_balance(self, asset: str) -> float:
        balance = self.exchange.fetch_balance()
        return float((balance.get("free") or {}).get(asset) or 0.0)

    def prices(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        result: dict[str, float] = {}
        try:
            tickers = self.exchange.fetch_tickers(symbols)
            for s, t in tickers.items():
                value = t.get("last") or t.get("close")
                if value:
                    result[s] = float(value)
        except Exception as exc:  # noqa: BLE001
            log.debug("fetch_tickers на бирже исполнения не удался: %s", exc)
        for symbol in symbols:
            if symbol not in result:
                ticker = self.exchange.fetch_ticker(symbol)
                result[symbol] = float(ticker.get("last") or ticker.get("close"))
        return result

    def min_order_cost(self, symbol: str) -> float:
        market = self.exchange.market(symbol)
        cost_min = ((market.get("limits") or {}).get("cost") or {}).get("min")
        return float(cost_min) if cost_min else 5.0

    def round_qty(self, symbol: str, qty: float) -> float:
        if qty <= 0:
            return 0.0
        try:
            return float(self.exchange.amount_to_precision(symbol, qty))
        except ccxt.InvalidOrder:
            return 0.0  # меньше минимального шага

    def market_buy(self, symbol: str, quote_amount: float) -> Fill:
        market = self.exchange.market(symbol)
        amount = math.floor(quote_amount * 100) / 100  # до центов, в меньшую сторону
        try:
            order = self.exchange.create_market_buy_order_with_cost(symbol, amount)
        except ccxt.BaseError as exc:
            raise BrokerError(f"покупка {symbol} на {amount} {self.quote} не удалась: {exc}") from exc
        order = self._ensure_filled(order, symbol)
        return parse_order_fill(order, "buy", symbol, market["base"], market["quote"])

    def market_sell(self, symbol: str, qty: float) -> Fill:
        market = self.exchange.market(symbol)
        amount = self.round_qty(symbol, qty)
        if amount <= 0:
            raise BrokerError(f"количество {qty} {market['base']} меньше минимального шага")
        try:
            order = self.exchange.create_market_sell_order(symbol, amount)
        except ccxt.BaseError as exc:
            raise BrokerError(f"продажа {amount} {symbol} не удалась: {exc}") from exc
        order = self._ensure_filled(order, symbol)
        return parse_order_fill(order, "sell", symbol, market["base"], market["quote"])

    def _ensure_filled(self, order: dict[str, Any], symbol: str) -> dict[str, Any]:
        if order.get("filled") and order.get("cost"):
            return order
        if order.get("id") is None:
            return order
        try:
            return self.exchange.fetch_order(order["id"], symbol)
        except ccxt.BaseError as exc:
            log.warning("не удалось перечитать ордер %s: %s", order.get("id"), exc)
            return order


class PaperBroker(Broker):
    """Симуляция исполнения по текущей цене mainnet с комиссией и проскальзыванием."""

    BALANCES_KEY = "paper_balances"

    def __init__(
        self,
        *,
        storage: Storage,
        price_source: Callable[[list[str]], dict[str, float]],
        markets_source: Callable[[], dict[str, dict[str, Any]]],
        start_usdt: float,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
        quote: str = "USDT",
    ):
        self.mode = "paper"
        self.quote = quote
        self.storage = storage
        self._prices = price_source
        self._markets = markets_source
        self.fee_rate = fee_pct / 100.0
        self.slippage = slippage_pct / 100.0
        if storage.get(self.BALANCES_KEY) is None:
            storage.set(self.BALANCES_KEY, {quote: float(start_usdt)})
        self._order_seq = int(storage.get("paper_order_seq", 0))

    def _balances(self) -> dict[str, float]:
        return dict(self.storage.get(self.BALANCES_KEY, {}))

    def _save(self, balances: dict[str, float]) -> None:
        self.storage.set(self.BALANCES_KEY, balances)

    def _next_order_id(self) -> str:
        self._order_seq += 1
        self.storage.set("paper_order_seq", self._order_seq)
        return f"paper-{self._order_seq}"

    def load_markets(self) -> None:
        self._markets()

    def symbols(self) -> set[str]:
        return {
            s
            for s, m in self._markets().items()
            if m.get("spot") and m.get("active") is not False and m.get("quote") == self.quote
        }

    def free_balance(self, asset: str) -> float:
        return float(self._balances().get(asset, 0.0))

    def prices(self, symbols: list[str]) -> dict[str, float]:
        return self._prices(symbols)

    def min_order_cost(self, symbol: str) -> float:
        market = self._markets().get(symbol, {})
        cost_min = ((market.get("limits") or {}).get("cost") or {}).get("min")
        return float(cost_min) if cost_min else 5.0

    def round_qty(self, symbol: str, qty: float) -> float:
        market = self._markets().get(symbol, {})
        step = ((market.get("precision") or {}).get("amount")) or 1e-8
        return math.floor(qty / step + 1e-9) * step if qty > 0 else 0.0

    def market_buy(self, symbol: str, quote_amount: float) -> Fill:
        base = symbol.split("/")[0]
        balances = self._balances()
        cash = balances.get(self.quote, 0.0)
        if quote_amount > cash + 1e-9:
            raise BrokerError(f"paper: недостаточно {self.quote}: нужно {quote_amount:.2f}, есть {cash:.2f}")
        price = self.price(symbol) * (1 + self.slippage)
        gross_qty = quote_amount / price
        fee_base = gross_qty * self.fee_rate
        qty = gross_qty - fee_base
        balances[self.quote] = cash - quote_amount
        balances[base] = balances.get(base, 0.0) + qty
        self._save(balances)
        return Fill(symbol, "buy", qty, price, quote_amount, fee_base * price, quote_amount, self._next_order_id())

    def market_sell(self, symbol: str, qty: float) -> Fill:
        base = symbol.split("/")[0]
        balances = self._balances()
        held = balances.get(base, 0.0)
        qty = min(qty, held)
        if qty <= 0:
            raise BrokerError(f"paper: нет {base} для продажи")
        price = self.price(symbol) * (1 - self.slippage)
        gross = qty * price
        fee = gross * self.fee_rate
        balances[base] = held - qty
        balances[self.quote] = balances.get(self.quote, 0.0) + gross - fee
        self._save(balances)
        return Fill(symbol, "sell", qty, price, gross, fee, gross - fee, self._next_order_id())
