# binance-llm-trader: заметки для Claude

Binance spot бот: стратегия трендового пробоя (`trader/strategy.py`) + Claude как фильтр сигналов, аналитик новостей и автор отчётов (`trader/llm.py`). Пользователь общается на русском: отвечать по-русски, сообщения бота и README тоже на русском.

## Команды
- Тесты (офлайн, без ключей): `python -m pytest -q`
- Проверка подключений: `python -m trader check`
- Один цикл: `python -m trader once`; постоянно: `python -m trader run`
- Зависимости: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`

## Облачная сессия Claude Code
- Binance отвечает `451 restricted location` на IP облака Claude, **включая testnet**. Ордера на testnet отсюда невозможны, обходить блокировку (VPN/прокси) нельзя.
- Здесь доступен только paper-режим на публичных данных:
  `TRADING_MODE=paper MARKET_DATA_URL=https://data-api.binance.vision python -m trader once`
  (домен `data-api.binance.vision` должен быть в Allowed domains среды; если он тоже отвечает 451, тестировать можно только офлайн).
- Исходящий HTTPS идёт через прокси среды со своим CA (`REQUESTS_CA_BUNDLE`). ccxt читает окружение благодаря `requests_trust_env` в `make_binance`; не убирать, иначе будет `CERTIFICATE_VERIFY_FAILED`. Отключать проверку TLS нельзя.
- Ключи (`BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`, `TRADER_ANTHROPIC_API_KEY`, `TELEGRAM_*`) задаются переменными окружения среды; не просить их в чате и не выводить в лог. `ANTHROPIC_API_KEY` среда в сессию не передаёт (он для самого Claude Code), поэтому бот читает запасное имя `TRADER_ANTHROPIC_API_KEY`.
- Постоянная работа бота: компьютер пользователя или VPS в разрешённой стране (не США).

## Правила проекта
- LLM только советует: открытие сделки без сигнала стратегии, рост размера сверх лимитов и отмена стопов невозможны по коду (`risk.py`, `sanitize_assessment`).
- Модель по умолчанию `claude-sonnet-5-5` (выбор пользователя), effort `low`, лимит $10/мес (`budget.py`).
- Новые функции сопровождать тестами в `tests/`; сеть в тестах не использовать.
