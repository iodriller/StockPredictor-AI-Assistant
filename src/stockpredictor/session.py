"""Session-anchored intraday context.

Closes the gap where the platform decided on yesterday's close: this module
computes today's session VWAP, premarket levels, opening range, time-of-day
RVOL, and a current live price from intraday bars. It also classifies the
current market session (premarket / regular / lunch / close / after-hours /
closed) so downstream code can guard decisions appropriately.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Settings
from .contracts import SessionContext
from .utils import to_float


LOGGER = logging.getLogger(__name__)

# Regular US equity session boundaries in market time (default America/New_York).
_REGULAR_OPEN = time(9, 30)
_LUNCH_START = time(11, 30)
_LUNCH_END = time(13, 30)
_REGULAR_CLOSE = time(16, 0)
_PREMARKET_OPEN = time(4, 0)
_AFTERHOURS_CLOSE = time(20, 0)


def build_session_context(
    symbol: str,
    intraday_frame: pd.DataFrame | None,
    settings: Settings,
) -> SessionContext:
    market_tz = _market_timezone(settings)
    now_market = _now_market(market_tz)
    market_session, mins_open, mins_close = classify_market_session(now_market)
    context = SessionContext(
        symbol=symbol.upper(),
        as_of=now_market.isoformat(),
        market_session=market_session,
        minutes_since_open=mins_open,
        minutes_to_close=mins_close,
        interval=str(settings.data.get("intraday", {}).get("interval", "1m")),
    )
    if intraday_frame is None or intraday_frame.empty:
        context.notes.append("intraday bars unavailable; falling back to daily-bar analysis")
        return context

    frame = _ensure_market_time_index(intraday_frame, market_tz)
    today = _today_in_market(now_market)
    todays_bars = frame[frame.index.date == today]
    is_current_session = not todays_bars.empty
    if todays_bars.empty:
        last_session_date = frame.index.date.max() if len(frame.index) else None
        if last_session_date is None:
            context.notes.append("intraday frame contained no usable bars")
            return context
        todays_bars = frame[frame.index.date == last_session_date]
        context.notes.append(
            f"no bars for current trading day; using most recent available session ({last_session_date.isoformat()})"
        )

    regular_bars = todays_bars[(todays_bars.index.time >= _REGULAR_OPEN) & (todays_bars.index.time <= _REGULAR_CLOSE)]
    premarket_bars = todays_bars[todays_bars.index.time < _REGULAR_OPEN]

    context.bars_loaded = int(len(todays_bars))
    context.data_available = bool(len(todays_bars))
    context.session_date = todays_bars.index.date.max().isoformat()
    context.is_live = is_current_session
    context.reference_price = to_float(todays_bars["Close"].iloc[-1], None)
    if is_current_session:
        context.live_price = context.reference_price

    if not regular_bars.empty:
        context.session_open = to_float(regular_bars["Open"].iloc[0], None)
        context.session_high = to_float(regular_bars["High"].max(), None)
        context.session_low = to_float(regular_bars["Low"].min(), None)
        context.session_volume = to_float(regular_bars["Volume"].sum(), None)
        context.session_vwap = to_float(_session_vwap(regular_bars), None)
        opening_range_minutes = int(settings.features.get("opening_range_minutes", 30))
        context.opening_range_high, context.opening_range_low, context.opening_range_status = _opening_range(
            regular_bars, opening_range_minutes
        )

    if not premarket_bars.empty:
        context.premarket_high = to_float(premarket_bars["High"].max(), None)
        context.premarket_low = to_float(premarket_bars["Low"].min(), None)
        context.premarket_volume = to_float(premarket_bars["Volume"].sum(), None)

    if is_current_session:
        context.time_of_day_rvol = _time_of_day_rvol(frame, todays_bars, now_market)
    return context


def classify_market_session(now_market: datetime) -> tuple[str, int | None, int | None]:
    """Return (session_label, minutes_since_open, minutes_to_close)."""
    weekday = now_market.weekday()
    if weekday >= 5:
        return ("closed_weekend", None, None)
    current = now_market.time()
    if current < _PREMARKET_OPEN:
        minutes_to_open = int((_combine(now_market, _REGULAR_OPEN) - now_market).total_seconds() // 60)
        return ("closed_overnight", None, minutes_to_open)
    if current < _REGULAR_OPEN:
        minutes_to_open = int((_combine(now_market, _REGULAR_OPEN) - now_market).total_seconds() // 60)
        return ("premarket", None, minutes_to_open)
    if current < _LUNCH_START:
        mins_open = int((now_market - _combine(now_market, _REGULAR_OPEN)).total_seconds() // 60)
        mins_close = int((_combine(now_market, _REGULAR_CLOSE) - now_market).total_seconds() // 60)
        return ("regular_morning", mins_open, mins_close)
    if current < _LUNCH_END:
        mins_open = int((now_market - _combine(now_market, _REGULAR_OPEN)).total_seconds() // 60)
        mins_close = int((_combine(now_market, _REGULAR_CLOSE) - now_market).total_seconds() // 60)
        return ("regular_lunch", mins_open, mins_close)
    if current < _REGULAR_CLOSE:
        mins_open = int((now_market - _combine(now_market, _REGULAR_OPEN)).total_seconds() // 60)
        mins_close = int((_combine(now_market, _REGULAR_CLOSE) - now_market).total_seconds() // 60)
        # Last 30 minutes get a distinct label so the dashboard can highlight close-window risk.
        if mins_close <= 30:
            return ("regular_close", mins_open, mins_close)
        return ("regular_afternoon", mins_open, mins_close)
    if current < _AFTERHOURS_CLOSE:
        return ("after_hours", None, None)
    return ("closed_overnight", None, None)


def _market_timezone(settings: Settings) -> str:
    return str(settings.app.get("market_timezone") or settings.app.get("timezone") or "America/New_York")


def _now_market(tz_name: str) -> datetime:
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            return datetime.now(ZoneInfo(tz_name))
        except ZoneInfoNotFoundError:
            return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def _today_in_market(now_market: datetime):
    return now_market.date()


def _combine(now_market: datetime, when: time) -> datetime:
    return now_market.replace(hour=when.hour, minute=when.minute, second=0, microsecond=0)


def _ensure_market_time_index(frame: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    if frame.index.tz is None:
        try:
            localized = frame.tz_localize("UTC")
        except Exception:
            return frame
    else:
        localized = frame
    try:
        return localized.tz_convert(tz_name)
    except Exception:
        return localized


def _session_vwap(regular_bars: pd.DataFrame) -> float | None:
    typical = (regular_bars["High"] + regular_bars["Low"] + regular_bars["Close"]) / 3
    volume = regular_bars["Volume"].fillna(0)
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return None
    return float((typical * volume).sum() / total_volume)


def _opening_range(regular_bars: pd.DataFrame, minutes: int) -> tuple[float | None, float | None, str]:
    if regular_bars.empty:
        return None, None, "unavailable"
    start = regular_bars.index[0]
    cutoff = start + timedelta(minutes=max(1, minutes))
    window = regular_bars[regular_bars.index <= cutoff]
    if window.empty:
        return None, None, "unavailable"
    return float(window["High"].max()), float(window["Low"].min()), "available"


def _time_of_day_rvol(frame: pd.DataFrame, todays_bars: pd.DataFrame, now_market: datetime) -> float | None:
    """Volume so far today vs. average volume through the same wall-clock minute on prior days."""
    if todays_bars.empty:
        return None
    cutoff = now_market.time()
    todays_so_far = todays_bars[todays_bars.index.time <= cutoff]["Volume"].sum()
    if not todays_so_far:
        return None
    today = _today_in_market(now_market)
    by_day = frame[frame.index.date != today]
    if by_day.empty:
        return None
    grouped = by_day.groupby(by_day.index.date)
    prior_day_volumes: list[float] = []
    for _, day_bars in grouped:
        cumulative = day_bars[day_bars.index.time <= cutoff]["Volume"].sum()
        if cumulative > 0:
            prior_day_volumes.append(float(cumulative))
    if not prior_day_volumes:
        return None
    average = float(np.mean(prior_day_volumes))
    if average <= 0:
        return None
    return float(todays_so_far) / average


def session_no_trade_flags(context: SessionContext) -> Iterable[str]:
    """Suggest no-trade flags based on session state."""
    if context.market_session in {"closed_overnight", "closed_weekend"}:
        yield "market is closed"
    if context.market_session == "premarket":
        yield "premarket session — regular-hours risk plan may not apply yet"
    if context.market_session == "regular_lunch":
        yield "lunch chop — volume tends to fade"
    if context.market_session == "regular_close":
        yield "final 30 minutes — close-window volatility"
    if context.market_session == "after_hours":
        yield "after-hours session — wide spreads, low liquidity"
