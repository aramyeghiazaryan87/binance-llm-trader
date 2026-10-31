"""Новости (заголовки из RSS) и индекс страха и жадности. Бесплатно, без API-ключей."""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import feedparser
import requests

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; binance-llm-trader/0.1)"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Headline:
    source: str
    title: str
    published: datetime


@dataclass(frozen=True)
class FearGreed:
    value: int
    label: str


def clean_text(text: str, max_len: int = 160) -> str:
    """Убирает HTML и угловые скобки (чтобы заголовок не мог закрыть тег <news> в промпте)."""
    text = html.unescape(_TAG_RE.sub(" ", text or ""))
    text = text.replace("<", " ").replace(">", " ")
    text = _SPACE_RE.sub(" ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def parse_feed(content: bytes | str, *, source_hint: str, cutoff: datetime) -> list[Headline]:
    parsed = feedparser.parse(content)
    source = clean_text(parsed.feed.get("title", "") if parsed.feed else "", 40) or source_hint
    items: list[Headline] = []
    for entry in parsed.entries:
        stamp = entry.get("published_parsed") or entry.get("updated_parsed")
        if not stamp:
            continue
        published = datetime(*stamp[:6], tzinfo=timezone.utc)
        if published < cutoff:
            continue
        title = clean_text(entry.get("title", ""))
        if title:
            items.append(Headline(source=source, title=title, published=published))
    return items


def fetch_headlines(
    feeds: tuple[str, ...] | list[str],
    *,
    lookback_hours: float,
    max_items: int,
    now: datetime | None = None,
    timeout: float = 10.0,
) -> list[Headline]:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    collected: list[Headline] = []
    for url in feeds:
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            collected.extend(parse_feed(response.content, source_hint=urlparse(url).netloc, cutoff=cutoff))
        except Exception as exc:  # noqa: BLE001 — новости не должны ронять бота
            log.warning("лента %s недоступна: %s", url, exc)
    seen: set[str] = set()
    unique: list[Headline] = []
    for item in sorted(collected, key=lambda h: h.published, reverse=True):
        key = re.sub(r"[^a-z0-9]", "", item.title.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:max_items]


def fetch_fear_greed(timeout: float = 10.0) -> FearGreed | None:
    try:
        response = requests.get(FEAR_GREED_URL, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        data = response.json()["data"][0]
        return FearGreed(value=int(data["value"]), label=str(data["value_classification"]))
    except Exception as exc:  # noqa: BLE001
        log.warning("индекс страха и жадности недоступен: %s", exc)
        return None


class NewsProvider:
    def __init__(self, *, feeds, lookback_hours: float, max_items: int, news_enabled: bool, fear_greed_enabled: bool):
        self.feeds = tuple(feeds)
        self.lookback_hours = lookback_hours
        self.max_items = max_items
        self.news_enabled = news_enabled
        self.fear_greed_enabled = fear_greed_enabled

    def headlines(self, now: datetime | None = None) -> list[Headline]:
        if not self.news_enabled or not self.feeds:
            return []
        return fetch_headlines(self.feeds, lookback_hours=self.lookback_hours, max_items=self.max_items, now=now)

    def fear_greed(self) -> FearGreed | None:
        return fetch_fear_greed() if self.fear_greed_enabled else None
