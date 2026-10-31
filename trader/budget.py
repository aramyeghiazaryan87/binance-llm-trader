"""Учёт расходов на Claude API и жёсткий месячный лимит."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .storage import Storage

# Цены за 1 млн токенов, USD: (вход, выход). Источник: прайс Anthropic на октябрь 2026.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
}
# Неизвестная модель: считаем по самому дорогому тарифу, чтобы лимит не был превышен.
CONSERVATIVE_PRICE = (10.0, 50.0)
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.1


def price_for(model: str | None) -> tuple[float, float]:
    return PRICES_PER_MTOK.get(str(model or ""), CONSERVATIVE_PRICE)


def tokens_cost(model: str | None, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    p_in, p_out = price_for(model)
    return (
        input_tokens * p_in
        + output_tokens * p_out
        + cache_write * p_in * CACHE_WRITE_MULT
        + cache_read * p_in * CACHE_READ_MULT
    ) / 1_000_000


def usage_cost(usage: Any, requested_model: str) -> tuple[float, int, int]:
    """Стоимость ответа по `usage`. Учитывает все попытки (fallback) из usage.iterations."""
    entries: list[tuple[str, int, int, int, int]] = []
    for item in getattr(usage, "iterations", None) or []:
        if getattr(item, "type", "") in ("message", "fallback_message"):
            entries.append(
                (
                    getattr(item, "model", None) or requested_model,
                    int(getattr(item, "input_tokens", 0) or 0),
                    int(getattr(item, "output_tokens", 0) or 0),
                    int(getattr(item, "cache_creation_input_tokens", 0) or 0),
                    int(getattr(item, "cache_read_input_tokens", 0) or 0),
                )
            )
    if not entries:
        entries.append(
            (
                requested_model,
                int(getattr(usage, "input_tokens", 0) or 0),
                int(getattr(usage, "output_tokens", 0) or 0),
                int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
                int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            )
        )
    cost = sum(tokens_cost(m, i, o, cw, cr) for m, i, o, cw, cr in entries)
    total_in = sum(e[1] + e[3] + e[4] for e in entries)
    total_out = sum(e[2] for e in entries)
    return cost, total_in, total_out


def estimate_input_tokens(*texts: str) -> int:
    """Грубая верхняя оценка: ~3 символа на токен (для английского текста и чисел обычно больше)."""
    return math.ceil(sum(len(t) for t in texts) / 3) + 50


def worst_case_cost(model: str, input_tokens: int, max_tokens: int) -> float:
    return tokens_cost(model, input_tokens, max_tokens)


def month_bounds(now: datetime) -> tuple[datetime, datetime]:
    now = now.astimezone(timezone.utc)
    start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc) if now.month == 12 else datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
    return start, end


class BudgetGuard:
    """Два правила:

    1. Жёсткий лимит: потрачено за месяц + худший случай для этого вызова <= бюджет.
    2. Темп: если потрачено больше, чем положено к этому дню месяца (+5% запаса),
       вызываем LLM только когда есть сигналы на вход. Позиции при этом охраняют
       стопы в коде.
    """

    def __init__(self, storage: Storage, monthly_budget_usd: float, pace_buffer_pct: float = 5.0):
        self.storage = storage
        self.monthly_budget = float(monthly_budget_usd)
        self.pace_buffer = pace_buffer_pct / 100.0

    def spent_this_month(self, now: datetime) -> float:
        start, end = month_bounds(now)
        return self.storage.llm_spent(start, end)

    def pace_limit(self, now: datetime) -> float:
        start, end = month_bounds(now)
        elapsed = (now - start).total_seconds() / (end - start).total_seconds()
        return self.monthly_budget * (elapsed + self.pace_buffer)

    def check(self, now: datetime, worst_case: float, has_signals: bool) -> tuple[bool, str]:
        spent = self.spent_this_month(now)
        if spent + worst_case > self.monthly_budget:
            return False, (
                f"месячный бюджет LLM исчерпан: потрачено ${spent:.2f} из ${self.monthly_budget:.2f} "
                f"(вызов может стоить до ${worst_case:.3f})"
            )
        if spent > self.pace_limit(now) and not has_signals:
            return False, (
                f"экономия бюджета: потрачено ${spent:.2f}, по графику ${self.pace_limit(now):.2f}; "
                "сигналов нет, вызов пропущен"
            )
        return True, ""
