"""Сборка компонентов бота из настроек."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from .broker import Broker, CcxtBroker, PaperBroker
from .budget import BudgetGuard
from .config import Settings
from .engine import TradingEngine
from .llm import LLMAdvisor
from .market_data import MarketData, make_binance
from .news import NewsProvider
from .notifier import Notifier
from .storage import Storage

log = logging.getLogger(__name__)


def setup_logging(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = RotatingFileHandler(settings.log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
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
        trust_env=settings.http_trust_env,
    )
    return CcxtBroker(exchange, settings.mode, settings.quote_asset)


def build_advisor(settings: Settings, storage: Storage) -> LLMAdvisor | None:
    if not settings.llm_enabled:
        return None
    if not settings.anthropic_api_key and not os.getenv("ANTHROPIC_AUTH_TOKEN"):
        log.warning("ANTHROPIC_API_KEY (или TRADER_ANTHROPIC_API_KEY) не задан: LLM отключён, действует LLM_FAIL_MODE=%s", settings.llm_fail_mode)
        return None
    return LLMAdvisor(
        storage=storage,
        budget=BudgetGuard(storage, settings.llm_monthly_budget_usd),
        model=settings.llm_model,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        max_watchlist=settings.max_watchlist,
        use_fallbacks=settings.llm_fallbacks,
        api_key=settings.anthropic_api_key,
    )


def build_engine(settings: Settings) -> TradingEngine:
    storage = Storage(settings.db_path)
    if settings.market_data_source == "mainnet":
        data = MarketData(make_binance(public_url=settings.market_data_url, trust_env=settings.http_trust_env), name="mainnet")
    else:
        data = MarketData(
            make_binance(
                sandbox=settings.mode == "testnet",
                demo=settings.mode == "demo",
                trust_env=settings.http_trust_env,
            ),
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
