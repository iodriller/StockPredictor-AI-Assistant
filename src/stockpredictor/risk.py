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
    notes: list[str] = []

    if decision.action not in {"long", "short"}:
        return decision, RiskPlan(
            symbol=decision.symbol,
            action=decision.action,
            entry=None,
            stop_loss=None,
            invalidation="No trade plan because the fused signal is not actionable.",
            notes=list(decision.reasons),
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
        )

    entry = latest_price
    stop_distance = max(atr * float(risk_cfg.get("atr_stop_multiple", 1.5)), entry * 0.003)
    if decision.action == "long":
        stop_loss = entry - stop_distance
        targets = [entry + stop_distance * float(mult) for mult in risk_cfg.get("target_r_multiples", [1.5, 2.0, 3.0])]
        invalidation = "Long idea is invalid below stop loss or sustained loss of VWAP/support."
    else:
        stop_loss = entry + stop_distance
        targets = [entry - stop_distance * float(mult) for mult in risk_cfg.get("target_r_multiples", [1.5, 2.0, 3.0])]
        invalidation = "Short idea is invalid above stop loss or sustained reclaim of VWAP/resistance."

    risk_reward = abs(targets[0] - entry) / abs(entry - stop_loss)
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
        )

    account_size = float(risk_cfg.get("account_size", 100000))
    max_position_risk = account_size * float(risk_cfg.get("max_risk_per_trade_pct", 0.01))
    shares_by_risk = int(max_position_risk // abs(entry - stop_loss))
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
        )

    notes.append(f"Max planned account risk is {max_position_risk:.2f}.")
    notes.append(f"Position size is capped at {position_size} shares by risk and exposure limits.")
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
    )

