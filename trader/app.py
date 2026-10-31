"""Сборка компонентов бота из настроек."""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler

from .broker import Broker, CcxtBroker, PaperBroker
from .budget import BudgetGuard
from .config import Settings
from .engine import TradingEngine
from .llm import LLMAdvisor
from .market_data import MarketData, make_binance
from .news import NewsProvider
from .notifier import TELEGRAM_TOKEN_RE, Notifier
from .storage import Storage

log = logging.getLogger(__name__)


class SecretRedactor(logging.Filter):
    """Прячет ключи и токены в записях лога, в том числе в тексте исключений.

    Последняя линия защиты: ключи не должны попадать в лог, даже если их принесёт
    сообщение об ошибке сторонней библиотеки (например, адрес запроса к Telegram).
    """

    def __init__(self, secrets: Iterable[str | None]):
        super().__init__()
        # короткие строки не трогаем, чтобы не портить обычный текст
        self.secrets = sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "<скрыто>")
        return TELEGRAM_TOKEN_RE.sub("bot<скрыто>", text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            clean = self.redact(message)
            if clean != message:
                record.msg, record.args = clean, None
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = self.redact(record.exc_text)
        except Exception:  # noqa: BLE001 — фильтр не должен ронять логирование
            pass
        return True


def setup_logging(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    redactor = SecretRedactor(
        [
            settings.binance_api_key,
            settings.binance_api_secret,
            settings.telegram_bot_token,
            anthropic_api_key(),
            os.getenv("ANTHROPIC_AUTH_TOKEN"),
        ]
    )
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(redactor)
    root.addHandler(console)
    file_handler = RotatingFileHandler(settings.log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)
    for noisy in ("urllib3", "httpx", "httpx2", "anthropic", "ccxt"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_broker(settings: Settings, storage: Storage, data: MarketData) -> Broker:
    if settings.mode == "paper":
        return PaperBroker(
            storage=storage,
            price_source=data.prices,
            markets_source=lambda: data.markets,
            start_usdt=settings.paper_start_usdt,
            fee_pct=settings.paper_fee_pct,
            slippage_pct=settings.paper_slippage_pct,
            quote=settings.quote_asset,
        )
    exchange = make_binance(
        api_key=settings.binance_api_key,
        secret=settings.binance_api_secret,
        sandbox=settings.mode == "testnet",
        demo=settings.mode == "demo",
    )
    return CcxtBroker(exchange, settings.mode, settings.quote_asset)


def anthropic_api_key() -> str | None:
    """Ключ Anthropic API для бота.

    TRADER_ANTHROPIC_API_KEY — запасное имя: облачная среда Claude Code не передаёт
    ANTHROPIC_API_KEY в сессию (он зарезервирован под авторизацию самого Claude Code).
    """
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("TRADER_ANTHROPIC_API_KEY") or None


def build_advisor(settings: Settings, storage: Storage) -> LLMAdvisor | None:
    if not settings.llm_enabled:
        return None
    api_key = anthropic_api_key()
    if not api_key and not os.getenv("ANTHROPIC_AUTH_TOKEN"):
        log.warning(
            "ANTHROPIC_API_KEY (или TRADER_ANTHROPIC_API_KEY) не задан: LLM отключён, действует LLM_FAIL_MODE=%s",
            settings.llm_fail_mode,
        )
        return None
    return LLMAdvisor(
        api_key=api_key,
        storage=storage,
        budget=BudgetGuard(storage, settings.llm_monthly_budget_usd),
        model=settings.llm_model,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        max_watchlist=settings.max_watchlist,
        use_fallbacks=settings.llm_fallbacks,
    )


def build_engine(settings: Settings) -> TradingEngine:
    storage = Storage(settings.db_path)
    if settings.market_data_source == "mainnet":
        data = MarketData(make_binance(public_url=settings.market_data_url), name="mainnet")
    else:
        data = MarketData(
            make_binance(sandbox=settings.mode == "testnet", demo=settings.mode == "demo"),
            name=settings.mode,
        )
    broker = build_broker(settings, storage, data)
    news = NewsProvider(
        feeds=settings.news_feeds,
        lookback_hours=settings.news_lookback_hours,
        max_items=settings.news_max_items,
        news_enabled=settings.news_enabled,
        fear_greed_enabled=settings.fear_greed_enabled,
    )
    notifier = Notifier(settings.telegram_bot_token, settings.telegram_chat_id, prefix=f"[{settings.mode.upper()}]")
    return TradingEngine(
        settings=settings,
        storage=storage,
        broker=broker,
        data=data,
        advisor=build_advisor(settings, storage),
        news=news,
        notifier=notifier,
    )
