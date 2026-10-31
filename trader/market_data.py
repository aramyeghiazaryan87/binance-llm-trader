"""Рыночные данные: свечи, тикеры, выбор торгуемых пар (universe)."""
from __future__ import annotations

import logging
import time
from typing import Any

import ccxt
import pandas as pd

from .strategy import timeframe_minutes

log = logging.getLogger(__name__)

# Стейблкоины, фиат и обёрнутые токены: торговать ими против USDT смысла нет.
EXCLUDED_BASES = frozenset(
    {
        "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USD1", "PYUSD", "RLUSD",
        "XUSD", "BFUSD", "AEUR", "EUR", "EURI", "GBP", "TRY", "BRL", "ARS", "JPY", "MXN",
        "PLN", "RON", "UAH", "ZAR", "IDRT", "BIDR", "UST", "USTC", "WBTC", "WBETH", "BETH",
        "PAXG", "XAUT",
    }
)
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def make_binance(
    *,
    api_key: str = "",
    secret: str = "",
    sandbox: bool = False,
    demo: bool = False,
    public_url: str = "",
) -> ccxt.binance:
    """Создаёт клиент ccxt для спота Binance (без ключей — только публичные данные).

    public_url — другой адрес для публичных данных, например https://data-api.binance.vision
    (официальный эндпоинт Binance только для рыночных данных).
    """
    exchange = ccxt.binance(
        {
            "apiKey": api_key or None,
            "secret": secret or None,
            "enableRateLimit": True,
            "timeout": 20_000,
            "options": {
                "defaultType": "spot",
                "fetchMarkets": {"types": ["spot"]},
                "fetchCurrencies": False,
                "adjustForTimeDifference": True,
                "recvWindow": 10_000,
            },
        }
    )
    if sandbox:
        exchange.set_sandbox_mode(True)
    if demo:
        exchange.enable_demo_trading(True)
    if public_url:
        api = dict(exchange.urls["api"])
        api["public"] = public_url.rstrip("/") + "/api/v3"
        exchange.urls = {**exchange.urls, "api": api}
    return exchange


def is_tradeable_base(base: str) -> bool:
    if base in EXCLUDED_BASES:
        return False
    return not any(base.endswith(suffix) and len(base) > len(suffix) for suffix in LEVERAGED_SUFFIXES)


def select_universe(
    tickers: dict[str, dict[str, Any]],
    markets: dict[str, dict[str, Any]],
    *,
    quote: str,
    limit: int,
    min_quote_volume: float,
    allowed: set[str] | None = None,
) -> list[str]:
    """Топ пар к `quote` по 24-часовому объёму, без стейблкоинов и плечевых токенов."""
    ranked: list[tuple[float, str]] = []
    for symbol, ticker in tickers.items():
        market = markets.get(symbol)
        if not market or not market.get("spot") or market.get("quote") != quote:
            continue
        if market.get("active") is False:
            continue
        if not is_tradeable_base(str(market.get("base", ""))):
            continue
        if allowed is not None and symbol not in allowed:
            continue
        volume = ticker.get("quoteVolume")
        if volume is None:
            base_volume, last = ticker.get("baseVolume"), ticker.get("last")
            volume = base_volume * last if base_volume and last else 0.0
        if volume < min_quote_volume:
            continue
        ranked.append((float(volume), symbol))
    ranked.sort(reverse=True)
    return [symbol for _, symbol in ranked[:limit]]


def ohlcv_to_frame(rows: list[list[float]], timeframe: str, now_ms: int | None = None) -> pd.DataFrame:
    """Свечи ccxt -> DataFrame. Незакрытая (текущая) свеча отбрасывается."""
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if frame.empty:
        return frame
    tf_ms = timeframe_minutes(timeframe) * 60_000
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    frame = frame[frame["ts"] + tf_ms <= now_ms]
    frame = frame.drop_duplicates("ts").sort_values("ts")
    frame.index = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    frame = frame.drop(columns=["ts"]).astype(float)
    return frame


class MarketData:
    """Обёртка над публичным API биржи (по умолчанию mainnet Binance, без ключей)."""

    def __init__(self, exchange: Any, name: str = "mainnet"):
        self.exchange = exchange
        self.name = name
        self._markets_loaded_at = 0.0

    def load_markets(self, force: bool = False) -> None:
        if force or not self.exchange.markets or time.time() - self._markets_loaded_at > 86_400:
            self.exchange.load_markets(reload=force)
            self._markets_loaded_at = time.time()

    @property
    def markets(self) -> dict[str, dict[str, Any]]:
        self.load_markets()
        return self.exchange.markets

    def symbols(self) -> set[str]:
        return {s for s, m in self.markets.items() if m.get("spot") and m.get("active") is not False}

    def top_symbols(
        self, *, quote: str, limit: int, min_quote_volume: float, allowed: set[str] | None = None
    ) -> list[str]:
        tickers = self.exchange.fetch_tickers()
        return select_universe(
            tickers, self.markets, quote=quote, limit=limit, min_quote_volume=min_quote_volume, allowed=allowed
        )

    def ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit + 1)
        return ohlcv_to_frame(rows, timeframe)

    def price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        value = ticker.get("last") or ticker.get("close")
        if not value:
            raise ValueError(f"нет цены для {symbol}")
        return float(value)

    def prices(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        try:
            tickers = self.exchange.fetch_tickers(symbols)
            result = {s: float(t.get("last") or t.get("close")) for s, t in tickers.items() if t.get("last") or t.get("close")}
            if all(s in result for s in symbols):
                return result
        except Exception as exc:  # noqa: BLE001 — переходим на запрос по одной паре
            log.debug("fetch_tickers(%s) не удался: %s", symbols, exc)
            result = {}
        for symbol in symbols:
            if symbol not in result:
                result[symbol] = self.price(symbol)
        return result
