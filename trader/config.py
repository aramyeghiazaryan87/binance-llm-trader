"""Настройки бота. Всё читается из переменных окружения (файл .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

VALID_MODES = ("testnet", "demo", "paper", "live")
LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_THE_RISK"

DEFAULT_NEWS_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
)


class ConfigError(Exception):
    """Ошибка конфигурации, при которой бот не должен запускаться."""


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть числом, получено {raw!r}") from exc


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть целым числом, получено {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "да")


def _list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # --- режим и ключи ---
    mode: str = "testnet"
    live_confirm: str = ""
    binance_api_key: str = field(default="", repr=False)
    binance_api_secret: str = field(default="", repr=False)
    telegram_bot_token: str = field(default="", repr=False)
    telegram_chat_id: str = ""

    # --- рынок ---
    quote_asset: str = "USDT"
    timeframe: str = "1h"
    candles_limit: int = 300
    universe_size: int = 15
    min_quote_volume: float = 20_000_000.0
    max_watchlist: int = 5
    market_data_source: str = "mainnet"  # mainnet | exchange
    market_data_url: str = ""  # например https://data-api.binance.vision (только публичные данные)
    max_price_deviation_pct: float = 3.0

    # --- бюджет и риск ---
    budget_usdt: float = 0.0  # 0 = весь свободный USDT на старте
    paper_start_usdt: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 20.0
    max_open_positions: int = 4
    max_total_exposure_pct: float = 80.0
    daily_loss_limit_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    min_order_usdt: float = 10.0
    cooldown_hours: float = 6.0
    max_hold_hours: float = 96.0
    paper_fee_pct: float = 0.1
    paper_slippage_pct: float = 0.05

    # --- стратегия ---
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 3.0
    breakout_lookback: int = 20
    min_volume_ratio: float = 1.2
    # фильтр рынка: новые входы только когда REGIME_SYMBOL закрылся выше дневной EMA(REGIME_EMA)
    regime_filter: bool = True
    regime_symbol: str = "BTC/USDT"
    regime_ema: int = 200

    # --- LLM ---
    llm_enabled: bool = True
    llm_model: str = "claude-sonnet-5-5"
    llm_effort: str = "low"
    llm_effort_signals: str = "medium"  # effort, когда есть сигнал на вход или открытая позиция
    llm_max_tokens: int = 4000
    llm_monthly_budget_usd: float = 10.0
    llm_min_minutes_between_calls: float = 50.0
    llm_fallbacks: bool = True
    llm_fail_mode: str = "hold"  # hold | rules
    min_llm_confidence: float = 0.6
    close_advice_min_confidence: float = 0.7

    # --- новости ---
    news_enabled: bool = True
    news_feeds: tuple[str, ...] = DEFAULT_NEWS_FEEDS
    news_max_items: int = 15
    news_lookback_hours: float = 12.0
    fear_greed_enabled: bool = True

    # --- движок ---
    cycle_delay_seconds: int = 90
    risk_check_interval_seconds: int = 300
    daily_report_hour_utc: int = 18
    notify_every_cycle: bool = False
    data_dir: Path = Path("data")
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env_file: str | os.PathLike | None = ".env", validate: bool = True) -> "Settings":
        if env_file and Path(env_file).exists():
            load_dotenv(env_file, override=False)

        mode = _str("TRADING_MODE", "testnet").lower()
        if mode not in VALID_MODES:
            raise ConfigError(f"TRADING_MODE должен быть одним из {VALID_MODES}, получено {mode!r}")

        key_names = {
            "testnet": ("BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET"),
            "demo": ("BINANCE_DEMO_API_KEY", "BINANCE_DEMO_API_SECRET"),
            "live": ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
            "paper": ("", ""),
        }[mode]
        api_key = _str(key_names[0], "") if key_names[0] else ""
        api_secret = _str(key_names[1], "") if key_names[1] else ""

        settings = cls(
            mode=mode,
            live_confirm=_str("LIVE_TRADING_CONFIRM", ""),
            binance_api_key=api_key,
            binance_api_secret=api_secret,
            telegram_bot_token=_str("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=_str("TELEGRAM_CHAT_ID", ""),
            quote_asset=_str("QUOTE_ASSET", "USDT").upper(),
            timeframe=_str("TIMEFRAME", "1h"),
            candles_limit=_int("CANDLES_LIMIT", 300),
            universe_size=_int("UNIVERSE_SIZE", 15),
            min_quote_volume=_float("MIN_QUOTE_VOLUME", 20_000_000.0),
            max_watchlist=_int("MAX_WATCHLIST", 5),
            market_data_source=_str("MARKET_DATA_SOURCE", "mainnet").lower(),
            market_data_url=_str("MARKET_DATA_URL", ""),
            max_price_deviation_pct=_float("MAX_PRICE_DEVIATION_PCT", 3.0),
            budget_usdt=_float("BUDGET_USDT", 0.0),
            paper_start_usdt=_float("PAPER_START_USDT", 10_000.0),
            risk_per_trade_pct=_float("RISK_PER_TRADE_PCT", 1.0),
            max_position_pct=_float("MAX_POSITION_PCT", 20.0),
            max_open_positions=_int("MAX_OPEN_POSITIONS", 4),
            max_total_exposure_pct=_float("MAX_TOTAL_EXPOSURE_PCT", 80.0),
            daily_loss_limit_pct=_float("DAILY_LOSS_LIMIT_PCT", 3.0),
            max_drawdown_pct=_float("MAX_DRAWDOWN_PCT", 15.0),
            min_order_usdt=_float("MIN_ORDER_USDT", 10.0),
            cooldown_hours=_float("COOLDOWN_HOURS", 6.0),
            max_hold_hours=_float("MAX_HOLD_HOURS", 96.0),
            paper_fee_pct=_float("PAPER_FEE_PCT", 0.1),
            paper_slippage_pct=_float("PAPER_SLIPPAGE_PCT", 0.05),
            atr_stop_mult=_float("ATR_STOP_MULT", 2.0),
            atr_trail_mult=_float("ATR_TRAIL_MULT", 3.0),
            breakout_lookback=_int("BREAKOUT_LOOKBACK", 20),
            min_volume_ratio=_float("MIN_VOLUME_RATIO", 1.2),
            regime_filter=_bool("REGIME_FILTER", True),
            regime_symbol=_str("REGIME_SYMBOL", "BTC/USDT").upper(),
            regime_ema=_int("REGIME_EMA", 200),
            llm_enabled=_bool("LLM_ENABLED", True),
            llm_model=_str("LLM_MODEL", "claude-sonnet-5-5"),
            llm_effort=_str("LLM_EFFORT", "low").lower(),
            llm_effort_signals=_str("LLM_EFFORT_SIGNALS", "medium").lower(),
            llm_max_tokens=_int("LLM_MAX_TOKENS", 4000),
            llm_monthly_budget_usd=_float("LLM_MONTHLY_BUDGET_USD", 10.0),
            llm_min_minutes_between_calls=_float("LLM_MIN_MINUTES_BETWEEN_CALLS", 50.0),
            llm_fallbacks=_bool("LLM_FALLBACKS", True),
            llm_fail_mode=_str("LLM_FAIL_MODE", "hold").lower(),
            min_llm_confidence=_float("MIN_LLM_CONFIDENCE", 0.6),
            close_advice_min_confidence=_float("CLOSE_ADVICE_MIN_CONFIDENCE", 0.7),
            news_enabled=_bool("NEWS_ENABLED", True),
            news_feeds=_list("NEWS_FEEDS", DEFAULT_NEWS_FEEDS),
            news_max_items=_int("NEWS_MAX_ITEMS", 15),
            news_lookback_hours=_float("NEWS_LOOKBACK_HOURS", 12.0),
            fear_greed_enabled=_bool("FEAR_GREED_ENABLED", True),
            cycle_delay_seconds=_int("CYCLE_DELAY_SECONDS", 90),
            risk_check_interval_seconds=_int("RISK_CHECK_INTERVAL_SECONDS", 300),
            daily_report_hour_utc=_int("DAILY_REPORT_HOUR_UTC", 18),
            notify_every_cycle=_bool("NOTIFY_EVERY_CYCLE", False),
            data_dir=Path(_str("DATA_DIR", "data")),
            log_level=_str("LOG_LEVEL", "INFO").upper(),
        )
        problems = settings.problems() if validate else []
        if problems:
            raise ConfigError("Ошибки конфигурации:\n- " + "\n- ".join(problems))
        return settings

    def problems(self) -> list[str]:
        """Проверка настроек. Пустой список = всё в порядке."""
        p: list[str] = []
        if self.mode not in VALID_MODES:
            p.append(f"неизвестный режим {self.mode!r}")
        if self.mode == "live" and self.live_confirm != LIVE_CONFIRM_PHRASE:
            p.append(
                "режим live торгует РЕАЛЬНЫМИ деньгами. Чтобы включить его, задайте "
                f"LIVE_TRADING_CONFIRM={LIVE_CONFIRM_PHRASE}"
            )
        if self.mode in ("testnet", "demo", "live") and not (self.binance_api_key and self.binance_api_secret):
            names = {
                "testnet": "BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET",
                "demo": "BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET",
                "live": "BINANCE_API_KEY / BINANCE_API_SECRET",
            }[self.mode]
            p.append(f"для режима {self.mode} нужны ключи {names}")
        if self.market_data_source not in ("mainnet", "exchange"):
            p.append("MARKET_DATA_SOURCE должен быть mainnet или exchange")
        if self.market_data_url and not self.market_data_url.startswith("https://"):
            p.append("MARKET_DATA_URL должен начинаться с https://")
        if self.mode == "paper" and self.market_data_source != "mainnet":
            p.append("в режиме paper данные берутся только с mainnet (MARKET_DATA_SOURCE=mainnet)")
        if self.llm_effort_signals not in ("low", "medium", "high", "xhigh", "max"):
            p.append("LLM_EFFORT_SIGNALS должен быть low, medium, high, xhigh или max")
        if self.llm_effort not in ("low", "medium", "high", "xhigh", "max"):
            p.append("LLM_EFFORT должен быть low, medium, high, xhigh или max")
        if self.llm_fail_mode not in ("hold", "rules"):
            p.append("LLM_FAIL_MODE должен быть hold или rules")
        if self.llm_monthly_budget_usd < 0:
            p.append("LLM_MONTHLY_BUDGET_USD не может быть отрицательным")
        if not 0 < self.risk_per_trade_pct <= 5:
            p.append("RISK_PER_TRADE_PCT должен быть в диапазоне (0, 5]")
        if not 0 < self.max_position_pct <= 100:
            p.append("MAX_POSITION_PCT должен быть в диапазоне (0, 100]")
        if not 0 < self.max_total_exposure_pct <= 100:
            p.append("MAX_TOTAL_EXPOSURE_PCT должен быть в диапазоне (0, 100]")
        if self.max_open_positions < 1:
            p.append("MAX_OPEN_POSITIONS должен быть >= 1")
        if self.regime_ema < 2:
            p.append("REGIME_EMA должен быть >= 2")
        if self.candles_limit < 230:
            p.append("CANDLES_LIMIT должен быть >= 230 (нужно для EMA200)")
        if not 0 <= self.daily_report_hour_utc <= 23:
            p.append("DAILY_REPORT_HOUR_UTC должен быть от 0 до 23")
        return p

    @property
    def db_path(self) -> Path:
        return self.data_dir / f"trader_{self.mode}.db"

    @property
    def log_path(self) -> Path:
        return self.data_dir / f"trader_{self.mode}.log"
