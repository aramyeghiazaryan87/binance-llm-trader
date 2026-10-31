"""Общие заглушки для офлайн-тестов: фейковая биржа, фейковый Claude, фейковые новости."""
from __future__ import annotations

import re
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trader.config import Settings
from trader.llm import MarketAssessment, PositionAdvice, SignalVerdict, WatchPick

HOUR_MS = 3_600_000


def make_candles(
    n: int = 300,
    *,
    start: float = 100.0,
    drift: float = 0.001,
    wiggle: float = 0.004,
    breakout: bool = False,
    now_ms: int | None = None,
) -> list[list[float]]:
    """Синтетические часовые свечи, последняя строка — незакрытая текущая свеча."""
    now_ms = now_ms or int(time.time() * 1000)
    current_open = now_ms // HOUR_MS * HOUR_MS
    first = current_open - n * HOUR_MS
    rows: list[list[float]] = []
    close = start
    for i in range(n):
        prev = close
        close = prev * (1 + drift + (wiggle if i % 2 == 0 else -wiggle))
        high = max(prev, close) * 1.001
        low = min(prev, close) * 0.999
        rows.append([first + i * HOUR_MS, prev, high, low, close, 100.0])
    if breakout:
        prior_high = max(r[2] for r in rows[-21:-1])
        prev = rows[-2][4]
        close = prior_high * 1.004
        rows[-1] = [rows[-1][0], prev, close * 1.001, prev * 0.999, close, 300.0]
    last_close = rows[-1][4]
    rows.append([current_open, last_close, last_close, last_close, last_close, 10.0])
    return rows


class FakeExchange:
    """Минимальный ccxt-подобный клиент."""

    def __init__(self, series: dict[str, list[list[float]]], volumes: dict[str, float] | None = None):
        self.series = series
        self.volumes = volumes or {}
        self.prices = {s: rows[-1][4] for s, rows in series.items()}
        self.markets: dict[str, dict] = {}
        for symbol in series:
            base, quote = symbol.split("/")
            self.markets[symbol] = {
                "symbol": symbol,
                "base": base,
                "quote": quote,
                "spot": True,
                "active": True,
                "limits": {"cost": {"min": 5.0}, "amount": {"min": 1e-5}},
                "precision": {"amount": 1e-5},
            }

    def load_markets(self, reload: bool = False):
        return self.markets

    def fetch_tickers(self, symbols=None):
        symbols = symbols or list(self.markets)
        return {s: {"symbol": s, "last": self.prices[s], "quoteVolume": self.volumes.get(s, 1e9)} for s in symbols}

    def fetch_ticker(self, symbol):
        return {"symbol": symbol, "last": self.prices[symbol]}

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=300):
        return self.series[symbol][-limit:]


class FakeNews:
    def __init__(self, headlines=None, fear_greed=None):
        self._headlines = headlines or []
        self._fg = fear_greed

    def headlines(self, now=None):
        return list(self._headlines)

    def fear_greed(self):
        return self._fg


class FakeNotifier:
    enabled = False

    def __init__(self):
        self.messages: list[str] = []

    def send(self, text: str) -> bool:
        self.messages.append(text)
        return True

    def error(self, text: str, min_interval_seconds: float = 0) -> None:
        self.messages.append("ERROR " + text)


SIGNAL_RE = re.compile(r"id=(\S+) symbol=(\S+)")
POSITION_RE = re.compile(r"^- (\S+/\S+) entry=", re.M)


def approving_responder(confidence: float = 0.8, size_factor: float = 0.8, close_positions: bool = False):
    """Фейковый Claude: одобряет все сигналы, кладёт их пары в watchlist."""

    def respond(kwargs):
        prompt = kwargs["messages"][0]["content"]
        signals = SIGNAL_RE.findall(prompt)
        positions = POSITION_RE.findall(prompt)
        assessment = MarketAssessment(
            market_regime="risk_on",
            news_sentiment=0.3,
            news_summary="Calm news flow.",
            watchlist=[WatchPick(symbol=sym, reason="strong trend") for _, sym in signals],
            signal_verdicts=[
                SignalVerdict(signal_id=sid, approve=True, confidence=confidence, size_factor=size_factor, reason="clean breakout")
                for sid, _ in signals
            ],
            position_advice=[
                PositionAdvice(symbol=sym, action="close" if close_positions else "hold", confidence=0.9, reason="test")
                for sym in positions
            ],
            report="Рынок спокойный, сигнал одобрен.",
        )
        return FakeResponse(assessment)

    return respond


def rejecting_responder(confidence: float = 0.7, reason: str = "RSI 71 extended, mixed trend"):
    """Фейковый Claude: отклоняет все сигналы."""

    def respond(kwargs):
        prompt = kwargs["messages"][0]["content"]
        assessment = MarketAssessment(
            market_regime="risk_off",
            news_sentiment=-0.3,
            news_summary="Exchange hack weighs on sentiment.",
            watchlist=[],
            signal_verdicts=[
                SignalVerdict(signal_id=sid, approve=False, confidence=confidence, size_factor=0.25, reason=reason)
                for sid, _ in SIGNAL_RE.findall(prompt)
            ],
            position_advice=[],
            report="Рынок осторожный, сигнал отклонён.",
        )
        return FakeResponse(assessment)

    return respond


class FakeUsage:
    def __init__(self, input_tokens=1500, output_tokens=700, iterations=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.iterations = iterations


class FakeResponse:
    def __init__(self, parsed, stop_reason="end_turn", usage=None, model="claude-sonnet-5-5"):
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = usage or FakeUsage()
        self.model = model


class FakeClaude:
    def __init__(self, responder):
        self.calls: list[dict] = []

        def parse(**kwargs):
            self.calls.append(kwargs)
            return responder(kwargs)

        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=parse))


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(
        Settings(),
        mode="paper",
        data_dir=tmp_path,
        universe_size=5,
        min_quote_volume=0.0,
        news_enabled=False,
        fear_greed_enabled=False,
        llm_monthly_budget_usd=10.0,
    )
