from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .contracts import FeatureSet, SessionContext
from .utils import clamp, to_float


def build_feature_set(symbol: str, frame: pd.DataFrame, settings: Settings) -> FeatureSet:
    indicators = calculate_indicators(frame, settings)
    latest = frame.iloc[-1]
    latest_price = to_float(latest["Close"])
    as_of = frame.index[-1].isoformat()
    levels = {
        "support": indicators.get("support"),
        "resistance": indicators.get("resistance"),
        "vwap": indicators.get("vwap"),
        "sma_20": indicators.get("sma_20"),
        "sma_50": indicators.get("sma_50"),
        "prior_high": indicators.get("prior_high"),
        "prior_low": indicators.get("prior_low"),
        "session_open": indicators.get("session_open"),
        "opening_range_high": indicators.get("opening_range_high"),
        "opening_range_low": indicators.get("opening_range_low"),
    }
    regime = str(indicators.get("market_regime") or "unknown")
    score, reasons = technical_score(latest_price, indicators)
    return FeatureSet(
        symbol=symbol.upper(),
        as_of=as_of,
        latest_price=latest_price,
        indicators=indicators,
        levels=levels,
        regime=regime,
        technical_score=score,
        reasons=reasons,
    )


def calculate_indicators(frame: pd.DataFrame, settings: Settings) -> dict[str, float | str | None]:
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    open_ = frame["Open"]
    volume = frame["Volume"]
    features_cfg = settings.features
    enabled = set(features_cfg.get("enabled", []))
    latest_values: dict[str, float | str | None] = {}
    volume_window = int(features_cfg.get("volume_window", 20))

    previous_close = close.shift(1)
    latest_values["price_change_pct"] = to_float(close.pct_change().iloc[-1], 0.0)
    latest_values["range_pct"] = to_float((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1], 0.0)

    if "vwap" in enabled:
        typical_price = (high + low + close) / 3
        vwap_window = int(features_cfg.get("vwap_window", volume_window))
        rolling_volume = volume.rolling(vwap_window).sum().replace(0, np.nan)
        vwap = (typical_price * volume).rolling(vwap_window).sum() / rolling_volume
        latest_values["vwap"] = to_float(vwap.iloc[-1], None)

    if "moving_averages" in enabled:
        ma_windows = [int(window) for window in features_cfg.get("ma_windows", [9, 20, 50])]
        moving_averages = {f"sma_{window}": close.rolling(window).mean() for window in ma_windows}
        for name, series in moving_averages.items():
            latest_values[name] = to_float(series.iloc[-1], None)

    if "rsi" in enabled:
        rsi = _rsi(close)
        latest_values["rsi"] = to_float(rsi.iloc[-1], None)

    if "macd" in enabled:
        macd_line, macd_signal, macd_hist = _macd(close)
        latest_values["macd"] = to_float(macd_line.iloc[-1], None)
        latest_values["macd_signal"] = to_float(macd_signal.iloc[-1], None)
        latest_values["macd_hist"] = to_float(macd_hist.iloc[-1], None)

    if "atr" in enabled:
        atr = _atr(high, low, close)
        latest_values["atr"] = to_float(atr.iloc[-1], None)
        latest_values["atr_pct"] = to_float(atr.iloc[-1] / close.iloc[-1], None)

    avg_volume = volume.rolling(volume_window).mean()
    latest_values["avg_volume"] = to_float(avg_volume.iloc[-1], None)
    if "volume_anomaly" in enabled:
        volume_anomaly = volume / avg_volume.replace(0, np.nan)
        latest_values["volume_anomaly"] = to_float(volume_anomaly.iloc[-1], None)

    if "volatility" in enabled:
        volatility_window = int(features_cfg.get("volatility_window", 20))
        volatility = close.pct_change().rolling(volatility_window).std()
        latest_values["volatility"] = to_float(volatility.iloc[-1], None)

    if "support_resistance" in enabled:
        sr_window = int(features_cfg.get("support_resistance_window", 30))
        support = low.rolling(sr_window).min()
        resistance = high.rolling(sr_window).max()
        latest_values["support"] = to_float(support.iloc[-1], None)
        latest_values["resistance"] = to_float(resistance.iloc[-1], None)

    if "gap" in enabled:
        gap_pct = (open_ - previous_close) / previous_close
        latest_values["gap_pct"] = to_float(gap_pct.iloc[-1], 0.0)

    if "session_levels" in enabled:
        latest_values["session_open"] = to_float(open_.iloc[-1], None)
        latest_values["session_high"] = to_float(high.iloc[-1], None)
        latest_values["session_low"] = to_float(low.iloc[-1], None)
        if len(frame) >= 2:
            latest_values["prior_high"] = to_float(high.iloc[-2], None)
            latest_values["prior_low"] = to_float(low.iloc[-2], None)
            latest_values["prior_close"] = to_float(close.iloc[-2], None)

    if "opening_range" in enabled:
        opening_high, opening_low, opening_status = _opening_range_levels(
            frame,
            minutes=int(features_cfg.get("opening_range_minutes", 30)),
        )
        latest_values["opening_range_high"] = opening_high
        latest_values["opening_range_low"] = opening_low
        latest_values["opening_range_status"] = opening_status

    if "trend" in enabled:
        latest_values["trend"] = _trend_label(close.iloc[-1], latest_values)
    if "market_regime" in enabled:
        latest_values["market_regime"] = _market_regime(latest_values)
    return latest_values


def technical_score(latest_price: float, indicators: dict[str, float | str | None]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    sma_9 = to_float(indicators.get("sma_9"))
    sma_20 = to_float(indicators.get("sma_20"))
    vwap = to_float(indicators.get("vwap"))
    rsi = to_float(indicators.get("rsi"))
    macd_hist = to_float(indicators.get("macd_hist"))
    volume_anomaly = to_float(indicators.get("volume_anomaly"), 1.0)
    trend = str(indicators.get("trend") or "mixed")

    if trend == "uptrend":
        score += 0.22
        reasons.append("trend is up")
    elif trend == "downtrend":
        score -= 0.22
        reasons.append("trend is down")

    if vwap and latest_price > vwap:
        score += 0.15
        reasons.append("price is above the rolling volume-weighted average")
    elif vwap and latest_price < vwap:
        score -= 0.15
        reasons.append("price is below the rolling volume-weighted average")

    if sma_20 and latest_price > sma_20:
        score += 0.12
        reasons.append("price is above 20-period average")
    elif sma_20 and latest_price < sma_20:
        score -= 0.12
        reasons.append("price is below 20-period average")

    if sma_9 and sma_20 and sma_9 > sma_20:
        score += 0.10
    elif sma_9 and sma_20 and sma_9 < sma_20:
        score -= 0.10

    if macd_hist > 0:
        score += 0.12
        reasons.append("MACD momentum is positive")
    elif macd_hist < 0:
        score -= 0.12
        reasons.append("MACD momentum is negative")

    if rsi >= 70:
        score -= 0.10
        reasons.append("RSI is extended")
    elif rsi >= 55:
        score += 0.08
        reasons.append("RSI supports upside momentum")
    elif rsi <= 30:
        score += 0.10
        reasons.append("RSI is oversold")
    elif rsi <= 45:
        score -= 0.08
        reasons.append("RSI is weak")

    if volume_anomaly >= 1.5:
        score += 0.06 if score >= 0 else -0.06
        reasons.append("volume is above average")

    return clamp(score, -1.0, 1.0), reasons or ["technical signal is neutral"]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal, macd_line - signal


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def _trend_label(latest_close: float, indicators: dict[str, float | str | None]) -> str:
    sma_9 = to_float(indicators.get("sma_9"))
    sma_20 = to_float(indicators.get("sma_20"))
    sma_50 = to_float(indicators.get("sma_50"))
    if latest_close > sma_9 > sma_20 and (not sma_50 or sma_20 > sma_50):
        return "uptrend"
    if latest_close < sma_9 < sma_20 and (not sma_50 or sma_20 < sma_50):
        return "downtrend"
    return "mixed"


def build_intraday_features(
    symbol: str,
    intraday_frame: pd.DataFrame,
    session: SessionContext | None,
    settings: Settings,
) -> dict[str, float | str | None]:
    """Compute a small set of intraday indicators on minute/5-minute bars.

    Returns a flat dict that lives next to (but separate from) the daily indicators.
    Anchored values prefer the SessionContext, which is already today-only.
    """
    if intraday_frame is None or intraday_frame.empty:
        return {}
    features: dict[str, float | str | None] = {"intraday_bars": float(len(intraday_frame))}
    close = intraday_frame["Close"]
    high = intraday_frame["High"]
    low = intraday_frame["Low"]
    rsi_window = int(settings.features.get("intraday_rsi_window", 14))
    if len(close) >= rsi_window + 1:
        features["intraday_rsi"] = to_float(_rsi(close, period=rsi_window).iloc[-1], None)
    if len(close) >= 26:
        macd_line, _, macd_hist = _macd(close)
        features["intraday_macd"] = to_float(macd_line.iloc[-1], None)
        features["intraday_macd_hist"] = to_float(macd_hist.iloc[-1], None)
    atr_window = int(settings.features.get("intraday_atr_window", 14))
    if len(close) >= atr_window + 1:
        atr_series = _atr(high, low, close, period=atr_window)
        features["intraday_atr"] = to_float(atr_series.iloc[-1], None)
        last_close = float(close.iloc[-1])
        features["intraday_atr_pct"] = to_float(atr_series.iloc[-1] / last_close if last_close else None, None)
    # Last 5 / 20 bar momentum gives the dashboard a "what just happened" view.
    if len(close) >= 6:
        features["intraday_return_5bar"] = to_float((close.iloc[-1] / close.iloc[-6]) - 1, None)
    if len(close) >= 21:
        features["intraday_return_20bar"] = to_float((close.iloc[-1] / close.iloc[-21]) - 1, None)
    if session is not None:
        if session.live_price is not None and session.session_vwap:
            features["intraday_vwap_distance_pct"] = (session.live_price / session.session_vwap) - 1
        if session.live_price is not None and session.session_open:
            features["intraday_open_distance_pct"] = (session.live_price / session.session_open) - 1
        if session.time_of_day_rvol is not None:
            features["intraday_rvol_tod"] = session.time_of_day_rvol
        if session.opening_range_high is not None and session.opening_range_low is not None and session.live_price is not None:
            if session.live_price > session.opening_range_high:
                features["opening_range_break"] = "above"
            elif session.live_price < session.opening_range_low:
                features["opening_range_break"] = "below"
            else:
                features["opening_range_break"] = "inside"
        if session.premarket_high is not None and session.live_price is not None:
            features["above_premarket_high"] = bool(session.live_price > session.premarket_high)
        if session.premarket_low is not None and session.live_price is not None:
            features["below_premarket_low"] = bool(session.live_price < session.premarket_low)
    return features


def intraday_technical_score(features: dict[str, float | str | None]) -> tuple[float, list[str]]:
    """Translate intraday features into a directional score in [-1, 1] with human-readable reasons."""
    score = 0.0
    reasons: list[str] = []
    if not features:
        return 0.0, []
    vwap_distance = to_float(features.get("intraday_vwap_distance_pct"), 0.0)
    if vwap_distance > 0.0015:
        score += 0.18
        reasons.append("price is holding above today's VWAP")
    elif vwap_distance < -0.0015:
        score -= 0.18
        reasons.append("price is below today's VWAP")
    rsi = to_float(features.get("intraday_rsi"), 50.0)
    if rsi >= 70:
        score -= 0.10
        reasons.append("intraday RSI is extended")
    elif rsi <= 30:
        score += 0.10
        reasons.append("intraday RSI is oversold")
    macd_hist = to_float(features.get("intraday_macd_hist"), 0.0)
    if macd_hist > 0:
        score += 0.10
        reasons.append("intraday MACD momentum is positive")
    elif macd_hist < 0:
        score -= 0.10
        reasons.append("intraday MACD momentum is negative")
    rvol = to_float(features.get("intraday_rvol_tod"), 1.0)
    if rvol >= 1.5:
        # Volume confirms direction rather than driving it.
        score += 0.06 if score >= 0 else -0.06
        reasons.append("intraday volume is above the typical pace for this time of day")
    elif rvol < 0.7:
        reasons.append("intraday volume is below typical pace — confirmation is weak")
    orb = str(features.get("opening_range_break", ""))
    if orb == "above":
        score += 0.12
        reasons.append("price broke above the opening range")
    elif orb == "below":
        score -= 0.12
        reasons.append("price broke below the opening range")
    return clamp(score, -1.0, 1.0), reasons


def _market_regime(indicators: dict[str, float | str | None]) -> str:
    volatility = to_float(indicators.get("volatility"))
    trend = str(indicators.get("trend") or "mixed")
    high_vol = volatility > 0.03
    if trend in {"uptrend", "downtrend"} and high_vol:
        return "trending_high_volatility"
    if trend in {"uptrend", "downtrend"}:
        return "trending"
    if high_vol:
        return "choppy_high_volatility"
    return "choppy"


def _opening_range_levels(frame: pd.DataFrame, minutes: int = 30) -> tuple[float | None, float | None, str]:
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < 2:
        return None, None, "unavailable"

    latest_day = frame.index[-1].date()
    session = frame[frame.index.date == latest_day]
    if len(session) < 2:
        return None, None, "requires_intraday_data"

    start = session.index[0]
    cutoff = start + pd.Timedelta(minutes=max(1, minutes))
    opening = session[session.index <= cutoff]
    if opening.empty:
        return None, None, "unavailable"
    return to_float(opening["High"].max(), None), to_float(opening["Low"].min(), None), "available"
