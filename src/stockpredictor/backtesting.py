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
    no_trades = 0
    evaluations = 0
    start = ""
    end = ""
    model_subset = [str(name) for name in cfg.get("model_subset", ["baseline"])]
    lookback = int(cfg.get("lookback_rows", 90))
    holding = int(cfg.get("holding_period_days", 5))
    step = int(cfg.get("evaluation_step_days", 5))

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
            if result.decision.action == "long":
                trade_return = (future_close / current_close) - 1
            elif result.decision.action == "short":
                trade_return = (current_close / future_close) - 1
            else:
                no_trades += 1
                continue
            position_fraction = float(settings.risk.get("max_position_fraction", 0.20))
            portfolio_return = trade_return * position_fraction
            equity *= 1 + portfolio_return
            returns.append(portfolio_return)
            equity_curve.append({"date": frame.index[index].isoformat(), "equity": equity, "symbol": symbol.upper()})

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
        equity_curve=equity_curve,
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
