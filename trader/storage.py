"""Хранилище состояния в SQLite: позиции, сделки, вызовы LLM, решения, снимки капитала."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    cost_usdt REAL NOT NULL,
    entry_ref_price REAL NOT NULL,
    stop_ref_price REAL NOT NULL,
    highest_ref_price REAL NOT NULL,
    atr_at_entry REAL NOT NULL,
    entry_time TEXT NOT NULL,
    signal_id TEXT,
    llm_confidence REAL,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    proceeds_usdt REAL,
    pnl_usdt REAL,
    pnl_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    position_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    quote_amount REAL NOT NULL,
    fee_usdt REAL NOT NULL,
    order_id TEXT,
    mode TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_time ON llm_calls(time);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    confidence REAL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    highest_price REAL NOT NULL,
    atr REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    exit_time TEXT,
    exit_price REAL,
    exit_reason TEXT,
    pnl_pct REAL
);
CREATE TABLE IF NOT EXISTS equity (
    time TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    exposure_pct REAL NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float  # фактическая цена исполнения с учётом комиссии
    cost_usdt: float  # потрачено USDT всего (с комиссией)
    entry_ref_price: float  # цена по источнику данных в момент входа
    stop_ref_price: float
    highest_ref_price: float
    atr_at_entry: float
    entry_time: datetime
    signal_id: str | None = None
    llm_confidence: float | None = None
    status: str = "open"
    id: int | None = None
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None
    proceeds_usdt: float | None = None
    pnl_usdt: float | None = None
    pnl_pct: float | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Position":
        data = dict(row)
        data["entry_time"] = from_iso(data["entry_time"])
        data["exit_time"] = from_iso(data.get("exit_time"))
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class Shadow:
    """Отклонённый сигнал: чем закончилась бы сделка, если бы бот вошёл. Ордеров нет.

    source: llm — отклонил Claude; regime — фильтр рынка; hold — Claude недоступен или пропущен.
    """

    signal_id: str
    symbol: str
    source: str
    entry_time: datetime
    entry_price: float
    stop_price: float
    highest_price: float
    atr: float
    reason: str = ""
    confidence: float | None = None
    status: str = "open"
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_pct: float | None = None  # за вычетом комиссии и проскальзывания
    id: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Shadow":
        data = dict(row)
        data["entry_time"] = from_iso(data["entry_time"])
        data["exit_time"] = from_iso(data.get("exit_time"))
        return cls(**data)


class Storage:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- key/value состояние ----------
    def get(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, ensure_ascii=False, default=str)),
        )
        self.conn.commit()

    def delete(self, key: str) -> None:
        self.conn.execute("DELETE FROM state WHERE key = ?", (key,))
        self.conn.commit()

    # ---------- позиции ----------
    def add_position(self, pos: Position) -> Position:
        data = asdict(pos)
        data.pop("id")
        data["entry_time"] = to_iso(pos.entry_time)
        data["exit_time"] = to_iso(pos.exit_time) if pos.exit_time else None
        columns = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        cur = self.conn.execute(f"INSERT INTO positions({columns}) VALUES({marks})", tuple(data.values()))
        self.conn.commit()
        pos.id = cur.lastrowid
        return pos

    def update_position(self, pos: Position) -> None:
        if pos.id is None:
            raise ValueError("позиция без id")
        data = asdict(pos)
        pos_id = data.pop("id")
        data["entry_time"] = to_iso(pos.entry_time)
        data["exit_time"] = to_iso(pos.exit_time) if pos.exit_time else None
        assignments = ", ".join(f"{k} = ?" for k in data)
        self.conn.execute(f"UPDATE positions SET {assignments} WHERE id = ?", (*data.values(), pos_id))
        self.conn.commit()

    def open_positions(self) -> list[Position]:
        rows = self.conn.execute("SELECT * FROM positions WHERE status = 'open' ORDER BY id").fetchall()
        return [Position.from_row(r) for r in rows]

    def closed_positions(self, since: datetime | None = None) -> list[Position]:
        if since is None:
            rows = self.conn.execute("SELECT * FROM positions WHERE status = 'closed' ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM positions WHERE status = 'closed' AND exit_time >= ? ORDER BY id", (to_iso(since),)
            ).fetchall()
        return [Position.from_row(r) for r in rows]

    def realized_pnl(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(pnl_usdt), 0) AS s FROM positions WHERE status = 'closed'").fetchone()
        return float(row["s"])

    # ---------- тень: отклонённые сигналы ----------
    def add_shadow(self, sh: Shadow) -> bool:
        """False, если этот сигнал уже записан (повтор цикла в том же часе)."""
        data = asdict(sh)
        data.pop("id")
        data["entry_time"] = to_iso(sh.entry_time)
        data["exit_time"] = to_iso(sh.exit_time) if sh.exit_time else None
        columns = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        cur = self.conn.execute(
            f"INSERT OR IGNORE INTO shadow_positions({columns}) VALUES({marks})", tuple(data.values())
        )
        self.conn.commit()
        if cur.rowcount:
            sh.id = cur.lastrowid
        return bool(cur.rowcount)

    def update_shadow(self, sh: Shadow) -> None:
        if sh.id is None:
            raise ValueError("тень без id")
        data = asdict(sh)
        sh_id = data.pop("id")
        data["entry_time"] = to_iso(sh.entry_time)
        data["exit_time"] = to_iso(sh.exit_time) if sh.exit_time else None
        assignments = ", ".join(f"{k} = ?" for k in data)
        self.conn.execute(f"UPDATE shadow_positions SET {assignments} WHERE id = ?", (*data.values(), sh_id))
        self.conn.commit()

    def shadows(self, *, status: str | None = None, since: datetime | None = None) -> list[Shadow]:
        """status=open/closed; since — по времени входа (для открытых) или выхода (для закрытых)."""
        query, args = "SELECT * FROM shadow_positions WHERE 1=1", []
        if status:
            query += " AND status = ?"
            args.append(status)
        if since is not None:
            query += " AND COALESCE(exit_time, entry_time) >= ?" if status == "closed" else " AND entry_time >= ?"
            args.append(to_iso(since))
        rows = self.conn.execute(query + " ORDER BY id", args).fetchall()
        return [Shadow.from_row(r) for r in rows]

    # ---------- сделки ----------
    def add_trade(
        self,
        *,
        time: datetime,
        position_id: int | None,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        quote_amount: float,
        fee_usdt: float,
        order_id: str | None,
        mode: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO trades(time, position_id, symbol, side, qty, price, quote_amount, fee_usdt, order_id, mode) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (to_iso(time), position_id, symbol, side, qty, price, quote_amount, fee_usdt, order_id, mode),
        )
        self.conn.commit()

    def trades_since(self, since: datetime) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM trades WHERE time >= ? ORDER BY id", (to_iso(since),)).fetchall()
        return [dict(r) for r in rows]

    # ---------- LLM ----------
    def add_llm_call(
        self,
        *,
        time: datetime,
        model: str,
        status: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        detail: str = "",
    ) -> None:
        self.conn.execute(
            "INSERT INTO llm_calls(time, model, status, input_tokens, output_tokens, cost_usd, detail) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (to_iso(time), model, status, int(input_tokens), int(output_tokens), float(cost_usd), detail[:2000]),
        )
        self.conn.commit()

    def llm_spent(self, start: datetime, end: datetime) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_calls WHERE time >= ? AND time < ?",
            (to_iso(start), to_iso(end)),
        ).fetchone()
        return float(row["s"])

    def llm_calls_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM llm_calls WHERE time >= ? AND time < ? ORDER BY id", (to_iso(start), to_iso(end))
        ).fetchall()
        return [dict(r) for r in rows]

    def last_llm_call(self, status: str | None = None) -> dict[str, Any] | None:
        if status:
            row = self.conn.execute(
                "SELECT * FROM llm_calls WHERE status = ? ORDER BY id DESC LIMIT 1", (status,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT * FROM llm_calls ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    # ---------- журнал решений и капитал ----------
    def add_decision(self, time: datetime, kind: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO decisions(time, kind, payload) VALUES(?, ?, ?)",
            (to_iso(time), kind, json.dumps(payload, ensure_ascii=False, default=str)),
        )
        self.conn.commit()

    def decisions(self, kind: str | None = None, limit: int = 50) -> Iterator[dict[str, Any]]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM decisions WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item["payload"])
            yield item

    def add_equity_snapshot(self, time: datetime, equity: float, cash: float, exposure_pct: float) -> None:
        self.conn.execute(
            "INSERT INTO equity(time, equity, cash, exposure_pct) VALUES(?, ?, ?, ?)",
            (to_iso(time), equity, cash, exposure_pct),
        )
        self.conn.commit()
