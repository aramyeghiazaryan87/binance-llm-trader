"""Текстовые отчёты: статус и дневной отчёт."""
from __future__ import annotations

from datetime import datetime, timedelta

from .budget import BudgetGuard
from .risk import AccountState
from .storage import Position, Storage

EXIT_REASONS = {
    "stop_loss": "стоп-лосс",
    "trailing_stop": "трейлинг-стоп",
    "time_stop": "стоп по времени",
    "llm_advice": "совет LLM",
    "kill_switch": "аварийная остановка",
    "manual": "вручную",
    "reconcile": "сверка с биржей",
}


def fmt_money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def position_line(pos: Position, price: float | None) -> str:
    if price is None:
        return f"• {pos.symbol}: {pos.qty:.6g} шт., вход {pos.entry_price:.6g}"
    value = pos.qty * price
    pnl = value - pos.cost_usdt
    pnl_pct = pnl / pos.cost_usdt * 100 if pos.cost_usdt else 0.0
    return (
        f"• {pos.symbol}: вход {pos.entry_price:.6g} → {price:.6g}, "
        f"{fmt_money(value)} USDT, PnL {pnl:+.2f} ({pnl_pct:+.2f}%), стоп {pos.stop_ref_price:.6g}"
    )


def status_text(
    *,
    mode: str,
    account: AccountState,
    starting_capital: float,
    positions: list[Position],
    prices: dict[str, float],
    budget: BudgetGuard,
    now: datetime,
    kill_switch: bool,
) -> str:
    total_pnl = account.equity - starting_capital
    total_pct = total_pnl / starting_capital * 100 if starting_capital else 0.0
    lines = [
        f"Режим: {mode}",
        f"Капитал: {fmt_money(account.equity)} USDT (старт {fmt_money(starting_capital)}, "
        f"итог {total_pnl:+.2f} / {total_pct:+.2f}%)",
        f"Свободно: {fmt_money(account.cash)} USDT, в позициях {account.exposure_pct:.1f}%",
        f"За день: {account.day_pnl_pct:+.2f}%, просадка от пика {account.drawdown_pct:.2f}%",
        f"LLM за месяц: ${budget.spent_this_month(now):.2f} из ${budget.monthly_budget:.2f}",
    ]
    if kill_switch:
        lines.append("⛔ Аварийная остановка ВКЛЮЧЕНА: новые сделки запрещены (сброс: reset-kill-switch)")
    elif account.daily_blocked:
        lines.append("⏸ Дневной лимит убытка достигнут: новые входы только завтра")
    lines.append(f"Открытые позиции ({len(positions)}):" if positions else "Открытых позиций нет")
    lines.extend(position_line(p, prices.get(p.symbol)) for p in positions)
    return "\n".join(lines)


def daily_report_text(
    *,
    storage: Storage,
    status: str,
    now: datetime,
) -> str:
    since = now - timedelta(hours=24)
    closed = storage.closed_positions(since)
    all_closed = storage.closed_positions()
    trades = storage.trades_since(since)
    llm_calls = storage.llm_calls_between(since, now + timedelta(seconds=1))
    llm_cost = sum(c["cost_usd"] for c in llm_calls)
    llm_ok = sum(1 for c in llm_calls if c["status"] == "ok")

    lines = [f"📊 Отчёт за 24 часа ({now:%Y-%m-%d %H:%M} UTC)", "", status, ""]
    if closed:
        day_pnl = sum(p.pnl_usdt or 0 for p in closed)
        lines.append(f"Закрыто сделок: {len(closed)}, результат {day_pnl:+.2f} USDT")
        for p in closed:
            lines.append(
                f"• {p.symbol}: {p.pnl_usdt:+.2f} USDT ({p.pnl_pct:+.2f}%), "
                f"{EXIT_REASONS.get(p.exit_reason or '', p.exit_reason)}"
            )
    else:
        lines.append("Закрытых сделок за сутки нет")
    lines.append(f"Ордеров за сутки: {len(trades)}")
    if all_closed:
        wins = sum(1 for p in all_closed if (p.pnl_usdt or 0) > 0)
        lines.append(f"Всего закрыто: {len(all_closed)}, прибыльных {wins / len(all_closed) * 100:.0f}%")
    lines.append(f"LLM за сутки: {len(llm_calls)} вызовов ({llm_ok} успешных), ${llm_cost:.3f}")

    last = next(storage.decisions(kind="llm", limit=1), None)
    if last:
        payload = last["payload"]
        lines.extend(["", f"Последний взгляд LLM ({payload.get('regime', '?')}):", payload.get("report", "")])
    return "\n".join(lines)
