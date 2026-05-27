from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .contracts import BacktestReport
from .data import MarketDataProvider, fetch_market_data, get_market_data_provider
from .pipeline import analyze_symbol


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

    for symbol in symbols:
        frame = fetch_market_data(symbol, settings, provider)
        start = start or frame.index[0].isoformat()
        end = frame.index[-1].isoformat()
        for index in range(lookback, len(frame) - holding, step):
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
            if result.decision.action not in {"long", "short"} or plan.entry is None or plan.stop_loss is None or not plan.targets:
                no_trades += 1
                trade_log.append(
                    {
                        "symbol": symbol.upper(),
                        "date": frame.index[index].isoformat(),
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
            mae, mfe = _excursions(result.decision.action, future_window, current_close)
            if exit_price is None:
                exit_price = future_close
                exit_reason = "time_exit"
                exit_date = frame.index[index + holding].isoformat()
            entry = current_close * (1 + slippage if result.decision.action == "long" else 1 - slippage)
            exit_price = exit_price * (1 - slippage if result.decision.action == "long" else 1 + slippage)
            if result.decision.action == "long":
                trade_return = (exit_price / entry) - 1
            else:
                trade_return = (entry / exit_price) - 1
            risk_per_share = plan.risk_per_share or abs(entry - plan.stop_loss)
            r_multiple = ((exit_price - entry) if result.decision.action == "long" else (entry - exit_price)) / risk_per_share if risk_per_share else 0.0
            position_fraction = float(settings.risk.get("max_position_fraction", 0.20))
            position_value = equity * position_fraction
            commission_return = commission / position_value if position_value else 0.0
            portfolio_return = trade_return * position_fraction - commission_return
            equity *= 1 + portfolio_return
            returns.append(portfolio_return)
            equity_curve.append({"date": frame.index[index].isoformat(), "equity": equity, "symbol": symbol.upper()})
            trade_log.append(
                {
                    "symbol": symbol.upper(),
                    "date": frame.index[index].isoformat(),
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
                    "position_size": plan.position_size or 0,
                    "setup_quality": plan.setup_quality,
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
        adverse = ((window["High"] / entry) - 1).max()
        favorable = ((entry / window["Low"]) - 1).max()
    else:
        adverse = ((entry / window["Low"]) - 1).max()
        favorable = ((window["High"] / entry) - 1).max()
    return float(max(adverse, 0.0)), float(max(favorable, 0.0))
