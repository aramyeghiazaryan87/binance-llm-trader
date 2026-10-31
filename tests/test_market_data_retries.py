"""Повторы чтения рыночных данных при сетевых сбоях (без сети)."""
import ssl

import ccxt
import pytest

from trader import market_data
from trader.market_data import MarketData, with_retries

from .conftest import FakeExchange, make_candles


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(market_data.time, "sleep", calls.append)
    return calls


def ssl_failure() -> ccxt.NetworkError:
    """Как ccxt сообщает о подменённом сертификате: NetworkError с причиной SSL."""
    try:
        try:
            raise ssl.SSLCertVerificationError("certificate verify failed: self-signed certificate")
        except ssl.SSLCertVerificationError as cause:
            raise ccxt.NetworkError("binance GET https://api.binance.com/api/v3/ticker/24hr") from cause
    except ccxt.NetworkError as exc:
        return exc


def test_transient_network_error_is_retried(sleeps, caplog):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ssl_failure()
        return "ok"

    assert with_retries(flaky, what="mainnet: тикеры") == "ok"
    assert sleeps == [2.0, 4.0]
    assert "mainnet: тикеры: сбой сети (NetworkError: SSLCertVerificationError), повтор 1 из 2" in caplog.text


def test_gives_up_after_attempts_and_never_retries_other_errors(sleeps):
    def down():
        raise ccxt.RequestTimeout("timeout")

    with pytest.raises(ccxt.RequestTimeout):
        with_retries(down, attempts=3)
    assert sleeps == [2.0, 4.0]

    calls = []

    def bad_symbol():
        calls.append(1)
        raise ccxt.BadSymbol("нет такой пары")

    with pytest.raises(ccxt.BadSymbol):
        with_retries(bad_symbol)
    assert calls == [1]  # не сетевая ошибка: повтор бессмысленен


def test_market_data_survives_one_network_error(sleeps):
    exchange = FakeExchange({"SOL/USDT": make_candles(300)}, volumes={"SOL/USDT": 1e9})
    real_ohlcv, real_tickers = exchange.fetch_ohlcv, exchange.fetch_tickers
    failures = {"ohlcv": 1, "tickers": 1}

    def ohlcv(symbol, timeframe="1h", limit=300):
        if failures["ohlcv"]:
            failures["ohlcv"] -= 1
            raise ssl_failure()
        return real_ohlcv(symbol, timeframe, limit)

    def tickers(symbols=None):
        if failures["tickers"]:
            failures["tickers"] -= 1
            raise ccxt.ExchangeNotAvailable("502")
        return real_tickers(symbols)

    exchange.fetch_ohlcv, exchange.fetch_tickers = ohlcv, tickers
    data = MarketData(exchange)
    assert len(data.ohlcv("SOL/USDT", "1h", 300)) == 300
    assert data.top_symbols(quote="USDT", limit=5, min_quote_volume=0) == ["SOL/USDT"]
    assert sleeps == [2.0, 2.0]
