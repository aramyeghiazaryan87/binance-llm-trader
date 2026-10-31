"""Уведомления в Telegram (необязательно). Без токена сообщения только пишутся в лог."""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4000


class Notifier:
    def __init__(self, token: str = "", chat_id: str = "", prefix: str = ""):
        self.token = token
        self.chat_id = chat_id
        self.prefix = prefix
        self._last_error_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        message = f"{self.prefix} {text}".strip() if self.prefix else text
        log.info("уведомление:\n%s", message)
        if not self.enabled:
            return False
        ok = True
        for start in range(0, len(message), TELEGRAM_LIMIT):
            chunk = message[start : start + TELEGRAM_LIMIT]
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": chunk, "disable_web_page_preview": True},
                    timeout=15,
                )
                if response.status_code != 200:
                    ok = False
                    log.warning("Telegram ответил %s: %s", response.status_code, response.text[:200])
            except requests.RequestException as exc:
                ok = False
                log.warning("Telegram недоступен: %s", exc)
        return ok

    def error(self, text: str, min_interval_seconds: float = 1800.0) -> None:
        """Ошибки отправляем не чаще раза в 30 минут, чтобы не спамить."""
        now = time.monotonic()
        if now - self._last_error_at < min_interval_seconds:
            log.error("%s", text)
            return
        self._last_error_at = now
        self.send(f"⚠️ {text}")
