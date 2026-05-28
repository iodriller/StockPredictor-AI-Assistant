"""Calendar awareness: market session, earnings, and configured macro events.

This module produces the CalendarContext that surfaces hard "skip this trade"
flags before the user pulls the trigger:

- Is the market actually open right now?
- Is earnings inside 24h? (yfinance Ticker.calendar)
- Is there a macro event (CPI, FOMC, NFP) inside 24h, per the configured list?
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Iterable

from .config import Settings
from .contracts import CalendarContext


LOGGER = logging.getLogger(__name__)


def build_calendar_context(symbol: str, settings: Settings) -> CalendarContext:
    market_tz = str(settings.app.get("market_timezone") or settings.app.get("timezone") or "America/New_York")
    now_market = _now_in_tz(market_tz)
    session, mins_to_open, mins_to_close = _classify_session(now_market)
    context = CalendarContext(
        symbol=symbol.upper(),
        as_of=now_market.isoformat(),
        market_session=session,
        minutes_to_open=mins_to_open,
        minutes_to_close=mins_to_close,
    )

    cfg = settings.raw.get("calendar", {}) or {}
    if cfg.get("earnings_enabled", True):
        earnings_date = _next_earnings_date(symbol)
        if earnings_date is not None:
            context.next_earnings_date = earnings_date.isoformat()
            delta_hours = (earnings_date - now_market).total_seconds() / 3600
            context.hours_to_earnings = round(delta_hours, 2)
            context.earnings_within_24h = 0 <= delta_hours <= 24

    macro_events = cfg.get("macro_events", []) or []
    today_iso = now_market.date().isoformat()
    for event in macro_events:
        event_date = str(event.get("date", ""))
        if not event_date:
            continue
        record = {
            "label": str(event.get("label", "macro event")),
            "date": event_date,
            "time": str(event.get("time", "")),
            "impact": str(event.get("impact", "medium")),
        }
        if event_date == today_iso:
            context.macro_events_today.append(record)
        try:
            event_dt = _parse_event_datetime(event_date, record["time"], market_tz)
        except ValueError:
            continue
        delta_hours = (event_dt - now_market).total_seconds() / 3600
        if 0 <= delta_hours <= 24:
            context.macro_events_next_24h.append({**record, "hours_away": round(delta_hours, 2)})

    context.no_trade_flags = list(calendar_no_trade_flags(context))
    return context


def calendar_no_trade_flags(context: CalendarContext) -> Iterable[str]:
    if context.market_session in {"closed_weekend", "closed_overnight"}:
        yield "market is currently closed"
    if context.earnings_within_24h:
        yield f"earnings inside 24h ({context.next_earnings_date})"
    for event in context.macro_events_next_24h:
        if event.get("impact") == "high":
            yield f"high-impact macro event in {event['hours_away']}h: {event['label']}"


def _now_in_tz(tz_name: str) -> datetime:
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            return datetime.now(ZoneInfo(tz_name))
        except ZoneInfoNotFoundError:
            return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(timezone.utc)


def _parse_event_datetime(date_str: str, time_str: str, tz_name: str) -> datetime:
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    parts = date_str.split("-")
    if len(parts) != 3:
        raise ValueError(f"invalid event date {date_str!r}")
    year, month, day = (int(part) for part in parts)
    hour, minute = 8, 30  # Default to typical 8:30 ET economic release
    if time_str:
        time_parts = time_str.split(":")
        if len(time_parts) >= 2:
            hour = int(time_parts[0])
            minute = int(time_parts[1])
    return datetime(year, month, day, hour, minute, tzinfo=tz)


def _classify_session(now_market: datetime) -> tuple[str, int | None, int | None]:
    weekday = now_market.weekday()
    if weekday >= 5:
        return ("closed_weekend", None, None)
    current = now_market.time()
    regular_open = time(9, 30)
    regular_close = time(16, 0)
    premarket_open = time(4, 0)
    afterhours_close = time(20, 0)
    if current < premarket_open:
        return ("closed_overnight", _minutes_until(now_market, regular_open), None)
    if current < regular_open:
        return ("premarket", _minutes_until(now_market, regular_open), None)
    if current < regular_close:
        mins_close = _minutes_until(now_market, regular_close)
        return ("regular", None, mins_close)
    if current < afterhours_close:
        return ("after_hours", None, None)
    return ("closed_overnight", None, None)


def _minutes_until(now_market: datetime, target: time) -> int:
    target_dt = now_market.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    if target_dt < now_market:
        target_dt = target_dt + timedelta(days=1)
    return int((target_dt - now_market).total_seconds() // 60)


def _next_earnings_date(symbol: str) -> datetime | None:
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        # yfinance has changed this API several times; try the common shapes.
        calendar = getattr(ticker, "calendar", None)
        candidate = None
        if hasattr(calendar, "loc"):
            try:
                value = calendar.loc["Earnings Date"]
                candidate = value.iloc[0] if hasattr(value, "iloc") else value
            except Exception:
                candidate = None
        if candidate is None and isinstance(calendar, dict):
            value = calendar.get("Earnings Date") or calendar.get("Earnings Average")
            if isinstance(value, (list, tuple)) and value:
                candidate = value[0]
            else:
                candidate = value
        if candidate is None:
            try:
                earnings_dates = ticker.get_earnings_dates(limit=4)
                if earnings_dates is not None and not earnings_dates.empty:
                    candidate = earnings_dates.index[0]
            except Exception:
                candidate = None
        if candidate is None:
            return None
        if hasattr(candidate, "to_pydatetime"):
            candidate = candidate.to_pydatetime()
        if isinstance(candidate, datetime):
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=timezone.utc)
            return candidate
        return None
    except Exception as exc:
        LOGGER.info("Earnings calendar fetch for %s failed: %s", symbol, exc)
        return None
