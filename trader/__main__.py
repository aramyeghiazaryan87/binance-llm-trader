"""Командная строка.

    python -m trader check              проверить подключения и настройки (ордера не ставятся)
    python -m trader once               один торговый цикл и выход
    python -m trader run                работать постоянно
    python -m trader status             капитал, позиции, расходы на LLM
    python -m trader report             отправить дневной отчёт сейчас
    python -m trader close-all          продать все позиции бота
    python -m trader reset-kill-switch  снять аварийную остановку
    python -m trader reset --yes        удалить базу состояния (только без открытых позиций)
    python -m trader shadow             отклонённые сигналы: чем закончились бы сделки
    python -m trader backtest           проверить стратегию на истории (без LLM и без ключей)
        --days 90                       сколько дней истории
        --variant "имя:UNIVERSE_SIZE=25,MIN_VOLUME_RATIO=1.0"   сравнить с другими настройками
        --universe today                пары по сегодняшнему объёму (быстрее, но оптимистично)
        --kill-reset-days 3             снимать аварийную остановку через 3 дня (0 = никогда)
"""
from __future__ import annotations

import argparse
import logging
import sys

from .app import build_engine, setup_logging
from .budget import estimate_input_tokens, month_bounds, tokens_cost
from .config import ConfigError, Settings
from .llm import DecisionContext, PositionView, build_user_prompt
from .strategy import entry_signal

log = logging.getLogger("trader")


def cmd_check(engine) -> int:
    s = engine.s
    ok = True
    print(f"Режим: {s.mode} | данные: {s.market_data_source} | модель: {s.llm_model if s.llm_enabled else 'LLM выключен'}")
    print(f"Бюджет LLM: ${s.llm_monthly_budget_usd:.2f}/мес, effort={s.llm_effort}, max_tokens={s.llm_max_tokens}")

    print("\n[1] Рыночные данные")
    try:
        engine.data.load_markets()
        print(f"  OK: {engine.data.name}, спотовых пар: {len(engine.data.symbols())}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ОШИБКА: {exc}")
        print("  Подсказка: HTTP 451 / restricted location = Binance не обслуживает страну, где стоит этот компьютер")
        print("  или сервер (например, США). Запускайте бота на компьютере или VPS в разрешённой стране.")
        return 1

    print("\n[2] Биржа исполнения")
    try:
        engine.broker.load_markets()
        allowed = engine.broker.symbols()
        free = engine.broker.free_balance(s.quote_asset)
        print(f"  OK: {s.mode}, пар к {s.quote_asset}: {len(allowed)}, свободно {free:,.2f} {s.quote_asset}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        allowed = None
        print(f"  ОШИБКА: {exc}")
        print("  Проверьте ключи API, права (Enable Spot Trading) и привязку к IP.")

    print("\n[3] Универс пар")
    universe = engine.data.top_symbols(
        quote=s.quote_asset, limit=s.universe_size, min_quote_volume=s.min_quote_volume, allowed=allowed
    )
    print(f"  {len(universe)}: {', '.join(universe) or '—'}")
    features = engine._collect_features(universe)
    signals = [sig for f in features.values() if (sig := entry_signal(f, engine.params))]
    print(f"  свечи получены: {len(features)}, сигналов на вход прямо сейчас: {len(signals)}")
    for sig in signals:
        print(f"   • {sig.symbol}: {sig.reason}")

    print("\n[4] Новости")
    headlines = engine.news.headlines()
    fear_greed = engine.news.fear_greed()
    print(f"  заголовков: {len(headlines)}; индекс страха и жадности: "
          f"{f'{fear_greed.value} ({fear_greed.label})' if fear_greed else 'недоступен'}")

    print("\n[5] Claude API")
    if engine.advisor is None:
        print("  LLM выключен или не задан ANTHROPIC_API_KEY / TRADER_ANTHROPIC_API_KEY " f"(при сбое LLM действует режим {s.llm_fail_mode})")
    else:
        now = engine.clock()
        account = engine.account_state({}) if engine.storage.get("starting_capital") else None
        if account is None:
            from .risk import AccountState

            account = AccountState(10_000, 10_000, 0, 0, 0, 0, False, False)
        positions = [PositionView.from_position(p, p.entry_ref_price, now) for p in engine.storage.open_positions()]
        ctx = DecisionContext(
            now=now, mode=s.mode, account=account, max_positions=s.max_open_positions,
            candidates=list(features.values()), signals=signals, positions=positions,
            headlines=headlines, fear_greed=fear_greed, news_lookback_hours=s.news_lookback_hours,
        )
        try:
            tokens = engine.advisor.count_prompt_tokens(ctx)
            print(f"  OK: ключ принят, промпт ~{tokens} входных токенов (подсчёт бесплатный)")
        except Exception as exc:  # noqa: BLE001
            ok = False
            tokens = estimate_input_tokens(engine.advisor.system_prompt, build_user_prompt(ctx))
            print(f"  ОШИБКА: {exc}")
        per_call_typ = tokens_cost(s.llm_model, tokens, 1200)
        per_call_max = tokens_cost(s.llm_model, tokens, s.llm_max_tokens)
        print(f"  стоимость вызова: обычно ~${per_call_typ:.4f}, максимум ${per_call_max:.4f}; "
              f"раз в час ~${per_call_typ * 24 * 30:.2f}/мес (лимит ${s.llm_monthly_budget_usd:.2f} соблюдается кодом)")
        start, end = month_bounds(now)
        print(f"  потрачено в этом месяце: ${engine.storage.llm_spent(start, end):.4f}")

    print("\n[6] Telegram")
    if engine.notifier.enabled:
        sent = engine.notifier.send("✅ Проверка связи: бот видит этот чат.")
        print("  OK: тестовое сообщение отправлено" if sent else "  ОШИБКА: сообщение не отправлено (см. лог)")
        ok = ok and sent
    else:
        print("  не настроен (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — уведомления будут только в логе")

    print("\nИтог:", "всё готово к запуску" if ok else "есть ошибки, см. выше")
    return 0 if ok else 1


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    import time
    from datetime import datetime, timedelta, timezone

    from dataclasses import replace

    from .backtest import (
        BacktestConfig,
        CandleCache,
        fetch_many,
        format_breakdown,
        format_comparison,
        format_result,
        monthly_breakdown,
        parse_variant,
        run_backtest,
        select_pool,
    )
    from .market_data import MarketData, is_tradeable_base, make_binance
    from .strategy import timeframe_minutes

    dynamic = args.universe == "dynamic"
    base = replace(
        BacktestConfig.from_settings(settings),
        dynamic_universe=dynamic,
        kill_switch_reset_hours=max(args.kill_reset_days, 0.0) * 24.0,
    )
    try:
        variants = [("текущие настройки", base)]
        for spec in args.variant or []:
            name, overrides = parse_variant(spec)
            variants.append((name, base.with_overrides(overrides)))
    except ValueError as exc:
        print(f"Ошибка в --variant: {exc}", file=sys.stderr)
        return 2
    if args.days <= 0:
        print("--days должно быть больше 0", file=sys.stderr)
        return 2

    def make_exchange():
        return make_binance(public_url=settings.market_data_url)

    data = MarketData(make_exchange(), name="mainnet")
    cache = CandleCache(None if args.no_cache else settings.data_dir / "candles")

    def progress(every: int):
        def report(done: int, total: int) -> None:
            if done % every == 0 or done == total:
                print(f"  {done}/{total}", flush=True)
        return report
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=args.days)).replace(minute=0, second=0, microsecond=0)
    warmup = timedelta(minutes=timeframe_minutes(settings.timeframe) * (max(c.candles_window for _, c in variants) + 5))
    since_ms = int((start - warmup).timestamp() * 1000)
    need = max(cfg.universe_size for _, cfg in variants)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif not dynamic:
        symbols = data.top_symbols(quote=settings.quote_asset, limit=need, min_quote_volume=settings.min_quote_volume)
    else:
        # все пары к USDT, включая делистнутые: иначе в тест попадут только сегодняшние «выжившие»
        candidates = sorted(
            s for s, m in data.markets.items()
            if m.get("spot") and m.get("quote") == settings.quote_asset and is_tradeable_base(str(m.get("base", "")))
        )
        print(f"Дневные свечи по {len(candidates)} парам к {settings.quote_asset} (включая делистнутые)...")
        day_since = int((start - timedelta(days=2)).timestamp() * 1000)
        daily, errors = fetch_many(
            make_exchange, candidates, "1d", day_since, cache=cache, workers=args.workers, progress=progress(100)
        )
        daily = {s: f for s, f in daily.items() if not f.empty}
        if errors:
            log.debug("дневные свечи недоступны: %s", errors)
        min_volume = min(cfg.min_quote_volume for _, cfg in variants)
        symbols = select_pool(daily, start, top_n=need * args.pool_margin, min_quote_volume=min_volume)
        delisted = [s for s in symbols if data.markets[s].get("active") is False]
        print(f"Пул: {len(symbols)} пар хотя бы день были в топ-{need * args.pool_margin} по объёму"
              + (f", из них сейчас делистнуты: {', '.join(delisted)}" if delisted else ""))

    regime_daily = None
    regime_emas = [cfg.regime_ema for _, cfg in variants if cfg.regime_filter]
    if regime_emas:
        regime_since = int((start - timedelta(days=max(regime_emas) * 2 + 10)).timestamp() * 1000)
        try:
            regime_daily = cache.history(data.exchange, settings.regime_symbol, "1d", regime_since)
        except Exception as exc:  # noqa: BLE001
            print(f"Дневные свечи {settings.regime_symbol} для фильтра рынка недоступны: {exc}", file=sys.stderr)
            return 1

    print(f"Загружаю свечи {settings.timeframe} за {args.days} дн. по {len(symbols)} парам...")
    t0 = time.time()
    frames, errors = fetch_many(
        make_exchange, symbols, settings.timeframe, since_ms, cache=cache, workers=args.workers, progress=progress(50)
    )
    for symbol, error in errors.items():
        print(f"  {symbol}: свечи недоступны ({error}), пропускаю")
    print(f"Загружено за {time.time() - t0:.0f} с" + ("" if args.no_cache else f" (кэш: {cache.dir})"))
    if dynamic:
        print(f"Период: {start:%Y-%m-%d} — {now:%Y-%m-%d} UTC; список пар пересчитывается каждый час: "
              f"топ по объёму за 24 часа на тот момент")
    else:
        print(f"Период: {start:%Y-%m-%d} — {now:%Y-%m-%d} UTC; пары по сегодняшнему объёму: {', '.join(frames)}")
    print("Claude не участвует: каждый сигнал считается одобренным. Комиссия "
          f"{base.fee_pct}% + проскальзывание {base.slippage_pct}% на каждую сторону.")
    if base.kill_switch_reset_hours > 0:
        print(f"Аварийная остановка снимается через {args.kill_reset_days:g} дн. (живой бот ждёт ручного reset-kill-switch).")
    print()

    results = []
    for name, cfg in variants:
        t0 = time.time()
        result = run_backtest(frames, cfg, start, regime_daily=regime_daily)
        results.append((name, result))
        print(format_result(result, name))
        log.debug("вариант %s посчитан за %.1f с", name, time.time() - t0)
        print()
    if len(results) > 1:
        print(format_comparison(results))
        print()
    breakdown = monthly_breakdown(results[0][1])
    if len(breakdown) > 1:
        print(f"По месяцам ({results[0][0]}):")
        print(format_breakdown(breakdown))
        print()
    bench = results[0][1].benchmark_pct
    if bench is not None:
        print(f"Для сравнения: BTC купить и держать за период {bench:+.2f}%")
    if dynamic:
        hours = results[0][1].universe_hours
        total = max(len(results[0][1].equity), 1)
        top = sorted(hours.items(), key=lambda x: -x[1])[:20]
        print("Чаще всего в списке: " + ", ".join(f"{s.split('/')[0]} {h / total:.0%}" for s, h in top))
    else:
        print("Пары выбраны по сегодняшнему объёму, поэтому результат оптимистичен: "
              "в этот список чаще попадают монеты, которые уже выросли.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trader", description="Binance spot бот с Claude")
    parser.add_argument(
        "command",
        choices=[
            "check", "once", "run", "status", "report", "close-all", "reset-kill-switch", "reset", "backtest", "shadow",
        ],
    )
    parser.add_argument("--env", default=".env", help="путь к .env (по умолчанию ./.env)")
    parser.add_argument("--yes", action="store_true", help="подтверждение для reset")
    parser.add_argument("--days", type=int, default=90, help="backtest: дней истории (по умолчанию 90)")
    parser.add_argument("--symbols", default="", help="backtest: свои пары через запятую вместо топа по объёму")
    parser.add_argument(
        "--variant", action="append", help='backtest: "имя:KEY=VALUE,KEY=VALUE" — сравнить с другими настройками'
    )
    parser.add_argument(
        "--universe", choices=["dynamic", "today"], default="dynamic",
        help="backtest: dynamic — топ по объёму на каждый час истории (по умолчанию); today — по сегодняшнему объёму",
    )
    parser.add_argument("--pool-margin", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument(
        "--kill-reset-days", type=float, default=3.0,
        help="backtest: через сколько дней снимать аварийную остановку (0 = никогда, как у живого бота)",
    )
    parser.add_argument("--workers", type=int, default=6, help="backtest: потоков загрузки (по умолчанию 6)")
    parser.add_argument("--no-cache", action="store_true", help="backtest: не сохранять свечи в data/candles")
    args = parser.parse_args(argv)

    try:
        # бэктесту ключи биржи не нужны: проверку настроек режима пропускаем
        settings = Settings.from_env(args.env, validate=args.command != "backtest")
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.command == "backtest":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        return cmd_backtest(settings, args)
    setup_logging(settings)

    if args.command == "shadow":
        from .reports import shadow_list_text
        from .storage import Storage, utcnow

        print(shadow_list_text(Storage(settings.db_path), utcnow()))
        return 0

    if args.command == "reset":
        if not args.yes:
            print("Удалит всю историю бота в", settings.db_path, "— добавьте --yes для подтверждения")
            return 1
        engine = build_engine(settings)
        if engine.storage.open_positions():
            print("Есть открытые позиции. Сначала закройте их: python -m trader close-all")
            return 1
        engine.storage.close()
        settings.db_path.unlink(missing_ok=True)
        print("База состояния удалена:", settings.db_path)
        return 0

    engine = build_engine(settings)
    if args.command == "check":
        return cmd_check(engine)
    if args.command == "run":
        engine.run_forever()
        return 0

    engine.startup()
    if args.command == "once":
        summary = engine.run_hourly_cycle()
        print(engine.status())
        print(f"\nУниверс: {', '.join(summary.universe)}")
        print(f"Сигналы: {', '.join(summary.signals) or 'нет'}")
        if summary.regime:
            print(f"Фильтр рынка: {summary.regime}")
        print(f"Куплено: {', '.join(summary.opened) or 'нет'}; продано: {', '.join(summary.closed) or 'нет'}")
        for line in summary.rejected:
            print(f"Отклонено: {line}")
        print(f"LLM: {summary.llm_status} (${summary.llm_cost:.4f})")
        if summary.llm_report:
            print(f"Отчёт LLM: {summary.llm_report}")
    elif args.command == "status":
        print(engine.status())
    elif args.command == "report":
        print(engine.send_daily_report())
    elif args.command == "close-all":
        print(f"Закрыто позиций: {engine.close_all('manual')}")
    elif args.command == "reset-kill-switch":
        engine.storage.set("kill_switch", False)
        account = engine.account_state()
        engine.storage.set("peak_equity", account.equity)
        print(f"Аварийная остановка снята, пик капитала сброшен на {account.equity:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
