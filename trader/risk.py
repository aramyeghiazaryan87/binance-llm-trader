"""Риск-менеджер: жёсткие лимиты, которые LLM обойти не может."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade_pct: float = 1.0  # сколько % капитала теряем, если сработал стоп
    max_position_pct: float = 20.0  # максимальный размер одной позиции, % капитала
    max_open_positions: int = 4
    max_total_exposure_pct: float = 80.0  # сколько % капитала может быть в монетах
    daily_loss_limit_pct: float = 3.0  # после такого убытка за день новые входы запрещены
    max_drawdown_pct: float = 15.0  # просадка от пика -> аварийная остановка
    min_order_usdt: float = 10.0


@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float  # свободные USDT бюджета бота
    positions_value: float
    open_positions: int
    day_pnl_pct: float
    drawdown_pct: float
    kill_switch: bool
    daily_blocked: bool

    @property
    def exposure_pct(self) -> float:
        return self.positions_value / self.equity * 100.0 if self.equity > 0 else 0.0


@dataclass(frozen=True)
class Decision:
    ok: bool
    reason: str = ""
    amount_usdt: float = 0.0


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def can_open(
        self,
        account: AccountState,
        symbol: str,
        held_symbols: set[str],
        cooldowns: dict[str, datetime],
        now: datetime,
    ) -> Decision:
        if account.kill_switch:
            return Decision(False, "аварийная остановка включена")
        if account.daily_blocked:
            return Decision(False, f"достигнут дневной лимит убытка {self.limits.daily_loss_limit_pct}%")
        if account.open_positions >= self.limits.max_open_positions:
            return Decision(False, f"уже открыто {account.open_positions} позиций (максимум {self.limits.max_open_positions})")
        if symbol in held_symbols:
            return Decision(False, "по паре уже есть позиция")
        until = cooldowns.get(symbol)
        if until and until > now:
            return Decision(False, f"пауза после стопа до {until:%Y-%m-%d %H:%M} UTC")
        return Decision(True)

    def size_position(
        self,
        account: AccountState,
        *,
        price: float,
        stop_price: float,
        size_factor: float,
        min_cost: float,
        free_quote: float,
    ) -> Decision:
        """Размер позиции от риска: потеря при срабатывании стопа = risk_per_trade_pct капитала."""
        if account.equity <= 0:
            return Decision(False, "капитал <= 0")
        stop_distance = (price - stop_price) / price
        if stop_distance <= 0:
            return Decision(False, "стоп выше цены входа")
        factor = min(max(size_factor, 0.25), 1.0)
        by_risk = account.equity * self.limits.risk_per_trade_pct / 100.0 / stop_distance
        by_cap = account.equity * self.limits.max_position_pct / 100.0
        exposure_room = account.equity * self.limits.max_total_exposure_pct / 100.0 - account.positions_value
        # 1% запаса на комиссию и округление
        cash_room = min(account.cash, free_quote) * 0.99
        amount = min(by_risk, by_cap) * factor
        amount = min(amount, exposure_room, cash_room)
        floor = max(self.limits.min_order_usdt, min_cost * 1.1)
        if amount < floor:
            return Decision(
                False,
                f"размер {amount:.2f} USDT меньше минимума {floor:.2f} "
                f"(свободно {cash_room:.2f}, лимит экспозиции {max(exposure_room, 0):.2f})",
            )
        return Decision(True, "", round(amount, 2))

    def daily_loss_hit(self, day_pnl_pct: float) -> bool:
        return day_pnl_pct <= -abs(self.limits.daily_loss_limit_pct)

    def drawdown_hit(self, drawdown_pct: float) -> bool:
        return drawdown_pct >= abs(self.limits.max_drawdown_pct)
