from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = (
    "app",
    "data",
    "watchlists",
    "features",
    "models",
    "signal_fusion",
    "risk",
    "context_agent",
    "backtest",
    "dashboard",
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    path: Path

    @property
    def app(self) -> dict[str, Any]:
        return self.raw["app"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def signal_fusion(self) -> dict[str, Any]:
        return self.raw["signal_fusion"]

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw["risk"]

    @property
    def context_agent(self) -> dict[str, Any]:
        return self.raw["context_agent"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]

    @property
    def dashboard(self) -> dict[str, Any]:
        return self.raw["dashboard"]

    def watchlist(self, name: str | None = None) -> list[str]:
        watchlists = self.raw["watchlists"]
        selected = name or self.dashboard.get("default_watchlist") or "default"
        symbols = watchlists.get(selected)
        if not symbols:
            symbols = watchlists.get("default", [])
        return [str(symbol).upper() for symbol in symbols]

    def enabled_models(self) -> list[str]:
        configured = self.models.get("enabled", [])
        enabled: list[str] = []
        for model_name in configured:
            model_cfg = self.models.get(model_name, {})
            if model_cfg.get("enabled", True):
                enabled.append(str(model_name))
        return enabled


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    path = Path(config_path or os.environ.get("STOCKPREDICTOR_CONFIG") or DEFAULT_CONFIG_PATH)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a mapping: {path}")

    missing = [section for section in REQUIRED_SECTIONS if section not in raw]
    if missing:
        raise ConfigError(f"Config missing required section(s): {', '.join(missing)}")

    _validate(raw)
    return Settings(raw=raw, path=path)


def _validate(raw: dict[str, Any]) -> None:
    weights = raw["signal_fusion"].get("weights", {})
    if not weights:
        raise ConfigError("signal_fusion.weights must not be empty")
    if sum(float(value) for value in weights.values()) <= 0:
        raise ConfigError("signal_fusion.weights must sum to a positive value")

    risk = raw["risk"]
    if float(risk.get("max_risk_per_trade_pct", 0)) <= 0:
        raise ConfigError("risk.max_risk_per_trade_pct must be positive")
    if float(risk.get("account_size", 0)) <= 0:
        raise ConfigError("risk.account_size must be positive")

    if int(raw["data"].get("min_rows", 1)) <= 0:
        raise ConfigError("data.min_rows must be positive")

