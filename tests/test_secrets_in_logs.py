"""Ключи и токены не должны попадать в лог ни при каких ошибках."""
import logging
from dataclasses import replace

import pytest
import requests

from trader.app import setup_logging
from trader.notifier import Notifier

TOKEN = "123456789:AAH-fake_token_value_for_tests_0123456"
ANTHROPIC_KEY = "sk-ant-api03-secret-value-for-tests-123"


def test_telegram_error_does_not_leak_token(monkeypatch, caplog):
    def unreachable(url, **kwargs):
        # так requests описывает сетевую ошибку: адрес запроса вместе с токеном
        path = url.split("api.telegram.org")[1]
        raise requests.ConnectionError(f"HTTPSConnectionPool(host='api.telegram.org'): Max retries exceeded with url: {path}")

    monkeypatch.setattr("trader.notifier.requests.post", unreachable)
    with caplog.at_level(logging.WARNING, logger="trader.notifier"):
        assert Notifier(TOKEN, "42").send("привет") is False
    assert "Telegram недоступен" in caplog.text
    assert TOKEN not in caplog.text
    assert "bot<скрыто>" in caplog.text


@pytest.fixture
def restore_root_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_log_file_never_contains_secrets(settings, monkeypatch, restore_root_logging):
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    s = replace(
        settings,
        binance_api_key="binance-key-AAAA1111",
        binance_api_secret="binance-secret-BBBB2222",
        telegram_bot_token=TOKEN,
    )
    setup_logging(s)
    log = logging.getLogger("trader.test")
    log.warning("ключ %s и секрет %s", s.binance_api_key, s.binance_api_secret)
    try:
        raise RuntimeError(f"запрос к https://api.telegram.org/bot{TOKEN}/sendMessage, ключ {ANTHROPIC_KEY}")
    except RuntimeError:
        log.exception("ошибка цикла")
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = s.log_path.read_text(encoding="utf-8")
    assert "ошибка цикла" in text and "RuntimeError" in text
    for secret in (s.binance_api_key, s.binance_api_secret, TOKEN, ANTHROPIC_KEY):
        assert secret not in text
    assert text.count("<скрыто>") >= 4
