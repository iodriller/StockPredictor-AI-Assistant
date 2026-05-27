from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .contracts import BacktestReport
from .data import fetch_market_data, get_market_data_provider
from .pipeline import analyze_symbol

if TYPE_CHECKING:
    from .data import MarketDataProvider


LOGGER = logging.getLogger(__name__)


def run_backtest(
    settings: Settings | None = None,
    symbols: list[str] | None = None,
    provider: MarketDataProvider | None = None,
) -> BacktestReport:
    settings = settings or load_settings()
    provider = provider or get_market_data_provider(settings)
    symbols = symbols or settings.watchlist()
    cfg = settings.backtest
    initial_capital = float(cfg.get("initial_capital", 100000))
    equity = initial_capital
    equity_curve: list[dict[str, float | str]] = []
    returns: list[float] = []
    trade_log: list[dict[str, float | str | int]] = []
    no_trades = 0
    evaluations = 0
    start = ""
    end = ""
    model_subset = [str(name) for name in cfg.get("model_subset", ["baseline"])]
    lookback = int(cfg.get("lookback_rows", 90))
    holding = int(cfg.get("holding_period_days", 5))
    step = int(cfg.get("evaluation_step_days", 5))
    slippage = float(cfg.get("slippage_bps", 0)) / 10000
    commission = float(cfg.get("commission_per_trade", 0))

    session_guard_cfg = settings.risk
    max_trades_per_day = int(session_guard_cfg.get("max_trades_per_day", 0)) or None
    max_daily_loss_pct = float(session_guard_cfg.get("max_daily_loss_pct", 0)) or None
    consecutive_loss_limit = int(session_guard_cfg.get("stop_after_consecutive_losses", 0)) or None
    use_planned_size = bool(settings.backtest.get("use_planned_position_size", True))

    daily_trade_counter: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}
    daily_equity_open: dict[str, float] = {}
    consecutive_losses = 0
    session_blocked: set[str] = set()
    session_skip_counts = {"max_trades_per_day": 0, "max_daily_loss_pct": 0, "stop_after_consecutive_losses": 0}

    for symbol in symbols:
        frame = fetch_market_data(symbol, settings, provider)
        start = start or frame.index[0].isoformat()
        end = frame.index[-1].isoformat()
        for index in range(lookback, len(frame) - holding, step):
            bar_timestamp = frame.index[index]
            day_key = bar_timestamp.date().isoformat() if hasattr(bar_timestamp, "date") else str(bar_timestamp)
            daily_equity_open.setdefault(day_key, equity)
            history = frame.iloc[:index].copy()
            future_close = float(frame["Close"].iloc[index + holding])
            current_close = float(frame["Close"].iloc[index])
            result = analyze_symbol(
                symbol,
                settings=settings,
                provider=provider,
                model_names=model_subset,
                data_frame=history,
                include_context=False,
            )
            evaluations += 1
            plan = result.risk_plan
            skip_session_reason = _session_guard_skip(
                day_key,
                session_blocked,
                daily_trade_counter,
                daily_pnl,
                daily_equity_open,
                max_trades_per_day,
                max_daily_loss_pct,
                consecutive_losses,
                consecutive_loss_limit,
            )
            if skip_session_reason:
                no_trades += 1
                session_skip_counts[skip_session_reason] = session_skip_counts.get(skip_session_reason, 0) + 1
                trade_log.append(
                    {
                        "symbol": symbol.upper(),
                        "date": bar_timestamp.isoformat(),
                        "action": result.decision.action,
                        "exit_reason": "session_blocked",
                        "return": 0.0,
                        "confidence": result.decision.confidence,
                        "score": result.decision.score,
                        "setup_quality": plan.setup_quality,
                        "top_reason": result.decision.top_reason,
                        "skip_reasons": f"session guardrail: {skip_session_reason}",
                        "equity": equity,
                    }
                )
                continue
            if result.decision.action not in {"long", "short"} or plan.entry is None or plan.stop_loss is None or not plan.targets:
                no_trades += 1
                trade_log.append(
                    {
                        "symbol": symbol.upper(),
                        "date": bar_timestamp.isoformat(),
                        "action": result.decision.action,
                        "exit_reason": "no_trade",
                        "return": 0.0,
                        "confidence": result.decision.confidence,
                        "score": result.decision.score,
                        "setup_quality": plan.setup_quality,
                        "top_reason": result.decision.top_reason,
                        "skip_reasons": "; ".join(plan.no_trade_reasons),
                        "equity": equity,
                    }
                )
                continue
            future_window = frame.iloc[index + 1 : index + holding + 1]
            exit_price, exit_reason, exit_date = _simulate_exit(result.decision.action, future_window, plan.stop_loss, plan.targets[0])
            if exit_price is None:
                exit_price = future_close
                exit_reason = "time_exit"
                exit_date = frame.index[index + holding].isoformat()
            entry = current_close * (1 + slippage if result.decision.action == "long" else 1 - slippage)
            exit_price = exit_price * (1 - slippage if result.decision.action == "long" else 1 + slippage)
            mae, mfe = _excursions(result.decision.action, future_window, entry)
            if result.decision.action == "long":
                trade_return = (exit_price / entry) - 1
            else:
                trade_return = (entry / exit_price) - 1
            planned_stop_exit = plan.stop_loss * (1 - slippage if result.decision.action == "long" else 1 + slippage)
            risk_per_share = abs(entry - planned_stop_exit)
            r_multiple = ((exit_price - entry) if result.decision.action == "long" else (entry - exit_price)) / risk_per_share if risk_per_share else 0.0
            shares, exposure_basis = _backtest_position_size(plan, entry, equity, settings.risk, use_planned_size)
            position_value = shares * entry
            portfolio_return = (trade_return * position_value - commission) / equity if equity else 0.0
            equity *= 1 + portfolio_return
            returns.append(portfolio_return)
            if portfolio_return < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            daily_trade_counter[day_key] = daily_trade_counter.get(day_key, 0) + 1
            daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + portfolio_return * daily_equity_open[day_key]
            equity_curve.append({"date": bar_timestamp.isoformat(), "equity": equity, "symbol": symbol.upper()})
            trade_log.append(
                {
                    "symbol": symbol.upper(),
                    "date": bar_timestamp.isoformat(),
                    "action": result.decision.action,
                    "entry": entry,
                    "stop_loss": plan.stop_loss,
                    "target": plan.targets[0],
                    "exit_price": exit_price,
                    "exit_date": exit_date,
                    "exit_reason": exit_reason,
                    "return": portfolio_return,
                    "trade_return": trade_return,
                    "r_multiple": r_multiple,
                    "max_adverse_excursion": mae,
                    "max_favorable_excursion": mfe,
                    "confidence": result.decision.confidence,
                    "score": result.decision.score,
                    "risk_reward": plan.risk_reward or 0.0,
                    "planned_risk": plan.planned_risk or 0.0,
                    "position_size": shares,
                    "exposure_basis": exposure_basis,
                    "setup_quality": plan.setup_quality,
                    "stop_source": plan.stop_source,
                    "target_source": plan.target_source,
                    "top_reason": result.decision.top_reason,
                    "equity": equity,
                }
            )

    trades = len(returns)
    win_rate = sum(1 for value in returns if value > 0) / trades if trades else 0.0
    average_return = float(np.mean(returns)) if returns else 0.0
    max_drawdown = _max_drawdown([initial_capital] + [float(point["equity"]) for point in equity_curve])
    sharpe_like = _sharpe_like(returns)
    no_trade_rate = no_trades / evaluations if evaluations else 0.0
    if any(session_skip_counts.values()):
        LOGGER.info("Session guard skips: %s", session_skip_counts)
    return BacktestReport(
        strategy="configured_fused_signal",
        symbols=[symbol.upper() for symbol in symbols],
        start=start,
        end=end,
        trades=trades,
        win_rate=win_rate,
        average_return=average_return,
        max_drawdown=max_drawdown,
        sharpe_like=sharpe_like,
        no_trade_rate=no_trade_rate,
        evaluations=evaluations,
        no_trades=no_trades,
        equity_curve=equity_curve,
        trade_log=trade_log,
    )


def _max_drawdown(equity_values: list[float]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, (value - peak) / peak)
    return abs(max_dd)


def _sharpe_like(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(252))


def _simulate_exit(action: str, window: pd.DataFrame, stop_loss: float, target: float) -> tuple[float | None, str, str]:
    for date, row in window.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        date_text = date.isoformat() if hasattr(date, "isoformat") else str(date)
        if action == "long":
            if low <= stop_loss:
                return stop_loss, "stop_hit", date_text
            if high >= target:
                return target, "target_hit", date_text
        if action == "short":
            if high >= stop_loss:
                return stop_loss, "stop_hit", date_text
            if low <= target:
                return target, "target_hit", date_text
    return None, "time_exit", ""


def _excursions(action: str, window: pd.DataFrame, entry: float) -> tuple[float, float]:
    if window.empty or entry <= 0:
        return 0.0, 0.0
    if action == "short":
        adverse = ((window["High"] - entry) / entry).max()
        favorable = ((entry - window["Low"]) / entry).max()
    else:
        adverse = ((entry - window["Low"]) / entry).max()
        favorable = ((window["High"] - entry) / entry).max()
    return float(max(adverse, 0.0)), float(max(favorable, 0.0))


def _backtest_position_size(plan, entry: float, equity: float, risk_cfg: dict, use_planned: bool) -> tuple[int, str]:
    if use_planned and plan.position_size:
        return int(plan.position_size), "planned"
    position_fraction = float(risk_cfg.get("max_position_fraction", 0.20))
    if entry <= 0:
        return 0, "fraction"
    shares = int((equity * position_fraction) // entry)
    return max(0, shares), "fraction"


def _session_guard_skip(
    day_key: str,
    session_blocked: set,
    daily_trade_counter: dict,
    daily_pnl: dict,
    daily_equity_open: dict,
    max_trades_per_day: int | None,
    max_daily_loss_pct: float | None,
    consecutive_losses: int,
    consecutive_loss_limit: int | None,
) -> str:
    if day_key in session_blocked:
        return "max_daily_loss_pct"
    if consecutive_loss_limit and consecutive_losses >= consecutive_loss_limit:
        return "stop_after_consecutive_losses"
    if max_trades_per_day and daily_trade_counter.get(day_key, 0) >= max_trades_per_day:
        return "max_trades_per_day"
    if max_daily_loss_pct:
        opening_equity = daily_equity_open.get(day_key)
        loss = -daily_pnl.get(day_key, 0.0)
        if opening_equity and (loss / opening_equity) >= max_daily_loss_pct:
            session_blocked.add(day_key)
            return "max_daily_loss_pct"
    return ""
