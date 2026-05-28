from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .config import Settings
from .contracts import FeatureSet, RiskPlan, SessionContext, SignalDecision
from .utils import to_float


def apply_risk_controls(
    decision: SignalDecision,
    features: FeatureSet,
    frame: pd.DataFrame,
    settings: Settings,
    session: SessionContext | None = None,
    horizon: str | None = None,
    intraday_features: dict | None = None,
) -> tuple[SignalDecision, RiskPlan]:
    risk_cfg = settings.risk
    horizon_profile = settings.horizon_profile(horizon)
    horizon_name = str(horizon_profile.get("name", "swing"))

    # Anchor the plan on the live intraday price when available — this is the core
    # decision-tool fix: stop pricing trades against yesterday's close.
    live_price = session.live_price if session is not None else None
    latest_price = to_float(live_price if live_price is not None else frame["Close"].iloc[-1])

    # ATR source depends on horizon: intraday uses minute-bar ATR if we have it.
    intraday_atr = None
    intraday_atr_pct = None
    if intraday_features:
        intraday_atr = intraday_features.get("intraday_atr")
        intraday_atr_pct = intraday_features.get("intraday_atr_pct")
    if horizon_name == "intraday" and intraday_atr:
        atr = to_float(intraday_atr, latest_price * 0.005)
        atr_pct = to_float(intraday_atr_pct, atr / latest_price if latest_price else 0.0)
    else:
        atr = to_float(features.indicators.get("atr"), latest_price * 0.02)
        atr_pct = to_float(features.indicators.get("atr_pct"), atr / latest_price if latest_price else 0.0)

    volume_window = int(settings.features.get("volume_window", 20))
    avg_volume = to_float(features.indicators.get("avg_volume"), frame["Volume"].tail(volume_window).mean())

    # Prefer the session-anchored VWAP/support/resistance when we have today's data.
    daily_vwap = to_float(features.indicators.get("vwap"), 0.0)
    daily_support = to_float(features.indicators.get("support"), 0.0)
    daily_resistance = to_float(features.indicators.get("resistance"), 0.0)
    if session is not None and horizon_name == "intraday":
        vwap = to_float(session.session_vwap, daily_vwap)
        support = to_float(session.session_low or session.opening_range_low or daily_support, daily_support)
        resistance = to_float(session.session_high or session.opening_range_high or daily_resistance, daily_resistance)
    else:
        vwap = daily_vwap
        support = daily_support
        resistance = daily_resistance

    notes: list[str] = [f"Horizon profile: {horizon_name}."]
    if live_price is not None:
        notes.append(f"Anchored on live price {live_price:.2f}.")
    session_checks = _session_checks(settings, avg_volume)

    if decision.action not in {"long", "short"}:
        return decision, RiskPlan(
            symbol=decision.symbol,
            action=decision.action,
            entry=None,
            stop_loss=None,
            invalidation="No trade plan because the fused signal is not actionable.",
            notes=list(decision.reasons),
            entry_zone=_entry_zone(latest_price, atr, decision.action),
            liquidity_ok=avg_volume >= float(risk_cfg.get("min_avg_volume", 0)),
            setup_quality="not_actionable",
            session_checks=session_checks,
            no_trade_reasons=_no_trade_reasons(["fused signal is not actionable"]),
        )

    if decision.confidence < float(risk_cfg.get("min_confidence_for_trade", 0.45)):
        adjusted = replace(decision, action="low_confidence", reasons=decision.reasons + ["risk layer blocked trade: confidence too low"])
        return adjusted, RiskPlan(
            symbol=decision.symbol,
            action=adjusted.action,
            entry=None,
            stop_loss=None,
            invalidation="Confidence is below configured trade threshold.",
            notes=adjusted.reasons,
            entry_zone=_entry_zone(latest_price, atr, adjusted.action),
            liquidity_ok=avg_volume >= float(risk_cfg.get("min_avg_volume", 0)),
            setup_quality="low_confidence",
            session_checks=session_checks,
            no_trade_reasons=_no_trade_reasons(["confidence below trade threshold"]),
        )

    min_avg_volume = float(risk_cfg.get("min_avg_volume", 0))
    if avg_volume < min_avg_volume:
        adjusted = replace(decision, action="no_trade", reasons=decision.reasons + ["risk layer blocked trade: liquidity below configured minimum"])
        return adjusted, RiskPlan(
            symbol=decision.symbol,
            action=adjusted.action,
            entry=None,
            stop_loss=None,
            invalidation="Average volume is below configured liquidity minimum.",
            notes=adjusted.reasons,
            entry_zone=_entry_zone(latest_price, atr, adjusted.action),
            liquidity_ok=False,
            setup_quality="low_liquidity",
            session_checks=session_checks,
            no_trade_reasons=_no_trade_reasons(["liquidity below configured minimum"]),
        )

    if atr_pct > float(risk_cfg.get("skip_if_atr_pct_above", 0.12)):
        adjusted = replace(decision, action="no_trade", reasons=decision.reasons + ["risk layer blocked trade: ATR percentage too high"])
        return adjusted, RiskPlan(
            symbol=decision.symbol,
            action=adjusted.action,
            entry=None,
            stop_loss=None,
            invalidation="Volatility is above configured maximum.",
            notes=adjusted.reasons,
            entry_zone=_entry_zone(latest_price, atr, adjusted.action),
            liquidity_ok=True,
            setup_quality="too_volatile",
            session_checks=session_checks,
            no_trade_reasons=_no_trade_reasons(["volatility above configured maximum"]),
        )

    max_vwap_distance = float(risk_cfg.get("max_entry_distance_from_vwap_pct", 1.0))
    if vwap and abs(latest_price / vwap - 1) > max_vwap_distance:
        adjusted = replace(decision, action="no_trade", reasons=decision.reasons + ["risk layer blocked trade: price too extended from VWAP"])
        return adjusted, RiskPlan(
            symbol=decision.symbol,
            action=adjusted.action,
            entry=latest_price,
            stop_loss=None,
            invalidation="Price is too far from VWAP for a fresh entry.",
            notes=adjusted.reasons,
            entry_zone=_entry_zone(latest_price, atr, adjusted.action),
            liquidity_ok=True,
            setup_quality="extended",
            session_checks=session_checks,
            no_trade_reasons=_no_trade_reasons(["price too extended from VWAP"]),
        )

    entry = latest_price
    atr_stop_multiple = float(horizon_profile.get("atr_stop_multiple", risk_cfg.get("atr_stop_multiple", 1.5)))
    entry_cushion_atr = float(horizon_profile.get("entry_cushion_atr", 0.25))
    entry_cushion_pct = float(horizon_profile.get("entry_cushion_pct", 0.002))
    entry_zone = _entry_zone(latest_price, atr, decision.action, entry_cushion_atr, entry_cushion_pct)
    stop_distance = max(atr * atr_stop_multiple, entry * 0.003)
    if decision.action == "long":
        fallback_stop = max(entry - stop_distance, entry * 0.01)
        structural_stop, stop_source = _best_long_stop(
            {"support": support, "vwap": vwap, "atr_fallback": fallback_stop},
            entry,
            fallback_stop,
        )
        stop_loss = min(entry - entry * 0.003, structural_stop - atr * 0.20)
        raw_target = entry + abs(entry - stop_loss) * _primary_target_multiple(risk_cfg, horizon_profile)
        targets, target_source = _merge_structural_target(raw_target, resistance, long=True)
        invalidation = "Long idea is invalid below stop loss, nearby support, or sustained loss of VWAP."
    else:
        fallback_stop = entry + stop_distance
        structural_stop, stop_source = _best_short_stop(
            {"resistance": resistance, "vwap": vwap, "atr_fallback": fallback_stop},
            entry,
            fallback_stop,
        )
        stop_loss = max(entry + entry * 0.003, structural_stop + atr * 0.20)
        raw_target = entry - abs(entry - stop_loss) * _primary_target_multiple(risk_cfg, horizon_profile)
        targets, target_source = _merge_structural_target(raw_target, support, long=False)
        invalidation = "Short idea is invalid above stop loss, nearby resistance, or sustained reclaim of VWAP."

    risk_reward = abs(targets[0] - entry) / abs(entry - stop_loss)
    risk_per_share = abs(entry - stop_loss)
    notes.append(f"Stop anchored on {stop_source}; target anchored on {target_source}.")
    if risk_reward < float(risk_cfg.get("min_risk_reward", 1.5)):
        adjusted = replace(decision, action="no_trade", reasons=decision.reasons + ["risk layer blocked trade: risk/reward too low"])
        return adjusted, RiskPlan(
            symbol=decision.symbol,
            action=adjusted.action,
            entry=entry,
            stop_loss=stop_loss,
            targets=targets,
            risk_reward=risk_reward,
            invalidation="Risk/reward is below configured minimum.",
            notes=adjusted.reasons,
            entry_zone=entry_zone,
            liquidity_ok=True,
            setup_quality="poor_risk_reward",
            risk_per_share=risk_per_share,
            session_checks=session_checks,
            no_trade_reasons=_no_trade_reasons(
                [f"risk/reward below configured minimum (stop anchored on {stop_source})"]
            ),
            stop_source=stop_source,
            target_source=target_source,
        )

    account_size = float(risk_cfg.get("account_size", 100000))
    max_position_risk = account_size * float(risk_cfg.get("max_risk_per_trade_pct", 0.01))
    shares_by_risk = int(max_position_risk // risk_per_share)
    shares_by_value = int((account_size * float(risk_cfg.get("max_position_fraction", 0.20))) // entry)
    position_size = max(0, min(shares_by_risk, shares_by_value))
    if position_size < 1:
        adjusted = replace(decision, action="no_trade", reasons=decision.reasons + ["risk layer blocked trade: position size below one share"])
        return adjusted, RiskPlan(
            symbol=decision.symbol,
            action=adjusted.action,
            entry=entry,
            stop_loss=stop_loss,
            targets=targets,
            risk_reward=risk_reward,
            max_position_risk=max_position_risk,
            position_size=0,
            invalidation="Configured account/risk limits do not allow a valid position size.",
            notes=adjusted.reasons,
            entry_zone=entry_zone,
            liquidity_ok=True,
            setup_quality="invalid_position_size",
            risk_per_share=risk_per_share,
            session_checks=session_checks,
            no_trade_reasons=_no_trade_reasons(["position size below one share"]),
            stop_source=stop_source,
            target_source=target_source,
        )

    planned_risk = position_size * risk_per_share
    planned_position_value = position_size * entry
    notes.append(f"Max planned account risk is {max_position_risk:.2f}.")
    notes.append(f"Position size is capped at {position_size} shares by risk and exposure limits.")
    notes.append(f"Entry zone is {entry_zone[0]:.2f} to {entry_zone[1]:.2f}.")
    notes.append(_session_note(session_checks))
    return decision, RiskPlan(
        symbol=decision.symbol,
        action=decision.action,
        entry=entry,
        stop_loss=stop_loss,
        targets=targets,
        risk_reward=risk_reward,
        max_position_risk=max_position_risk,
        position_size=position_size,
        invalidation=invalidation,
        notes=notes,
        entry_zone=entry_zone,
        liquidity_ok=True,
        setup_quality="actionable",
        risk_per_share=risk_per_share,
        planned_risk=planned_risk,
        planned_position_value=planned_position_value,
        session_checks=session_checks,
        no_trade_reasons=[],
        stop_source=stop_source,
        target_source=target_source,
    )


def _entry_zone(
    price: float,
    atr: float,
    action: str,
    atr_cushion: float = 0.25,
    pct_cushion: float = 0.002,
) -> tuple[float, float] | None:
    if action not in {"long", "short"} or price <= 0:
        return None
    cushion = max(atr * atr_cushion, price * pct_cushion)
    if action == "long":
        return (price - cushion, price + cushion * 0.5)
    return (price - cushion * 0.5, price + cushion)


def _merge_structural_target(raw_target: float, structural_level: float, long: bool) -> tuple[list[float], str]:
    if not structural_level:
        return [raw_target], "r_multiple"
    if long and 0 < structural_level < raw_target:
        return [structural_level], "structural_resistance"
    if not long and structural_level > raw_target > 0:
        return [structural_level], "structural_support"
    return [raw_target], "r_multiple"


def _primary_target_multiple(risk_cfg: dict, horizon_profile: dict | None = None) -> float:
    if horizon_profile and "target_r_multiple" in horizon_profile:
        return float(horizon_profile["target_r_multiple"])
    if "target_r_multiple" in risk_cfg:
        return float(risk_cfg["target_r_multiple"])
    configured = risk_cfg.get("target_r_multiples", [1.5])
    if not configured:
        return 1.5
    return float(configured[0])


def _best_long_stop(candidates: dict[str, float], entry: float, fallback: float) -> tuple[float, str]:
    valid = [(name, value) for name, value in candidates.items() if value > 0 and value < entry]
    if not valid:
        return fallback, "atr_fallback"
    name, value = max(valid, key=lambda item: item[1])
    return value, name


def _best_short_stop(candidates: dict[str, float], entry: float, fallback: float) -> tuple[float, str]:
    valid = [(name, value) for name, value in candidates.items() if value > entry]
    if not valid:
        return fallback, "atr_fallback"
    name, value = min(valid, key=lambda item: item[1])
    return value, name


def _session_checks(settings: Settings, avg_volume: float) -> dict[str, float | int | bool | str]:
    risk_cfg = settings.risk
    account_size = float(risk_cfg.get("account_size", 100000))
    max_daily_loss_pct = float(risk_cfg.get("max_daily_loss_pct", 0.03))
    min_avg_volume = float(risk_cfg.get("min_avg_volume", 0))
    return {
        "account_size": account_size,
        "max_daily_loss": account_size * max_daily_loss_pct,
        "max_daily_loss_pct": max_daily_loss_pct,
        "max_trades_per_day": int(risk_cfg.get("max_trades_per_day", 5)),
        "stop_after_consecutive_losses": int(risk_cfg.get("stop_after_consecutive_losses", 3)),
        "pdt_min_equity": float(risk_cfg.get("pdt_min_equity", 25000)),
        "pdt_warning": bool(risk_cfg.get("pdt_warning_enabled", True) and account_size < float(risk_cfg.get("pdt_min_equity", 25000))),
        "liquidity_min_avg_volume": min_avg_volume,
        "liquidity_ok": avg_volume >= min_avg_volume,
    }


def _session_note(session_checks: dict[str, float | int | bool | str]) -> str:
    return (
        "Session guardrails: max daily loss "
        f"{float(session_checks['max_daily_loss']):.2f}, max trades "
        f"{int(session_checks['max_trades_per_day'])}, stop after "
        f"{int(session_checks['stop_after_consecutive_losses'])} consecutive losses."
    )


def _no_trade_reasons(reasons: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        clean = reason.strip()
        if clean and clean not in seen:
            seen.add(clean)
            output.append(clean)
    return output
