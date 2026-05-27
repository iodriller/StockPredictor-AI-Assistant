from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .config import Settings
from .contracts import FeatureSet, RiskPlan, SignalDecision
from .utils import to_float


def apply_risk_controls(
    decision: SignalDecision,
    features: FeatureSet,
    frame: pd.DataFrame,
    settings: Settings,
) -> tuple[SignalDecision, RiskPlan]:
    risk_cfg = settings.risk
    latest_price = to_float(frame["Close"].iloc[-1])
    atr = to_float(features.indicators.get("atr"), latest_price * 0.02)
    atr_pct = to_float(features.indicators.get("atr_pct"), atr / latest_price if latest_price else 0.0)
    avg_volume = to_float(features.indicators.get("avg_volume"), frame["Volume"].tail(20).mean())
    vwap = to_float(features.indicators.get("vwap"), 0.0)
    support = to_float(features.indicators.get("support"), 0.0)
    resistance = to_float(features.indicators.get("resistance"), 0.0)
    notes: list[str] = []
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
            no_trade_reasons=_no_trade_reasons(decision, ["fused signal is not actionable"]),
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
            no_trade_reasons=_no_trade_reasons(adjusted, ["confidence below trade threshold"]),
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
            no_trade_reasons=_no_trade_reasons(adjusted, ["liquidity below configured minimum"]),
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
            no_trade_reasons=_no_trade_reasons(adjusted, ["volatility above configured maximum"]),
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
            no_trade_reasons=_no_trade_reasons(adjusted, ["price too extended from VWAP"]),
        )

    entry = latest_price
    entry_zone = _entry_zone(latest_price, atr, decision.action)
    stop_distance = max(atr * float(risk_cfg.get("atr_stop_multiple", 1.5)), entry * 0.003)
    if decision.action == "long":
        structural_stop = max(value for value in [support, vwap, entry - stop_distance] if value > 0 and value < entry)
        stop_loss = min(entry - entry * 0.003, structural_stop - atr * 0.20)
        raw_targets = [entry + abs(entry - stop_loss) * float(mult) for mult in risk_cfg.get("target_r_multiples", [1.5, 2.0, 3.0])]
        targets = _merge_structural_target(raw_targets, resistance, long=True)
        invalidation = "Long idea is invalid below stop loss, nearby support, or sustained loss of VWAP."
    else:
        structural_stop = min(value for value in [resistance, vwap, entry + stop_distance] if value > entry)
        stop_loss = max(entry + entry * 0.003, structural_stop + atr * 0.20)
        raw_targets = [entry - abs(entry - stop_loss) * float(mult) for mult in risk_cfg.get("target_r_multiples", [1.5, 2.0, 3.0])]
        targets = _merge_structural_target(raw_targets, support, long=False)
        invalidation = "Short idea is invalid above stop loss, nearby resistance, or sustained reclaim of VWAP."

    risk_reward = abs(targets[0] - entry) / abs(entry - stop_loss)
    risk_per_share = abs(entry - stop_loss)
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
            no_trade_reasons=_no_trade_reasons(adjusted, ["risk/reward below configured minimum"]),
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
            no_trade_reasons=_no_trade_reasons(adjusted, ["position size below one share"]),
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
    )


def _entry_zone(price: float, atr: float, action: str) -> tuple[float, float] | None:
    if action not in {"long", "short"} or price <= 0:
        return None
    cushion = max(atr * 0.25, price * 0.002)
    if action == "long":
        return (price - cushion, price + cushion * 0.5)
    return (price - cushion * 0.5, price + cushion)


def _merge_structural_target(raw_targets: list[float], structural_level: float, long: bool) -> list[float]:
    if not structural_level:
        return raw_targets
    first = raw_targets[0]
    if long and structural_level > first:
        raw_targets[0] = structural_level
    if not long and 0 < structural_level < first:
        raw_targets[0] = structural_level
    return raw_targets


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


def _no_trade_reasons(decision: SignalDecision, extra: list[str]) -> list[str]:
    reasons = list(extra)
    reasons.extend(reason for reason in decision.reasons if "blocked" in reason or "no " in reason.lower() or "too " in reason.lower())
    output: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        clean = reason.strip()
        if clean and clean not in seen:
            seen.add(clean)
            output.append(clean)
    return output
