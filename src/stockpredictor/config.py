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

DEFAULT_HORIZONS: dict[str, Any] = {
    "default": "swing",
    "profiles": {
        "intraday": {
            "horizon_days": 1,
            "lookback_rows": 30,
            "atr_stop_multiple": 0.8,
            "target_r_multiple": 1.5,
            "entry_cushion_atr": 0.15,
            "entry_cushion_pct": 0.0015,
            "weights": {"models": 0.15, "technicals": 0.20, "intraday": 0.45, "context": 0.15, "sentiment": 0.05},
        },
        "swing": {
            "horizon_days": 5,
            "lookback_rows": 180,
            "atr_stop_multiple": 1.5,
            "target_r_multiple": 1.5,
            "entry_cushion_atr": 0.25,
            "entry_cushion_pct": 0.002,
            "max_entry_distance_from_vwap_pct": 0.60,
            "weights": {"models": 0.35, "technicals": 0.30, "intraday": 0.10, "context": 0.20, "sentiment": 0.05},
        },
        "position": {
            "horizon_days": 20,
            "lookback_rows": 252,
            "atr_stop_multiple": 2.5,
            "target_r_multiple": 2.5,
            "entry_cushion_atr": 0.40,
            "entry_cushion_pct": 0.004,
            "max_entry_distance_from_vwap_pct": 1.00,
            "weights": {"models": 0.45, "technicals": 0.30, "intraday": 0.0, "context": 0.20, "sentiment": 0.05},
        },
    },
}


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
        return [str(model_name) for model_name in self.models.get("enabled", [])]

    @property
    def horizons(self) -> dict[str, Any]:
        configured = self.raw.get("horizons", {})
        merged = {
            "default": DEFAULT_HORIZONS["default"],
            "profiles": {name: dict(profile) for name, profile in DEFAULT_HORIZONS["profiles"].items()},
        }
        if not isinstance(configured, dict):
            return merged
        if configured.get("default"):
            merged["default"] = str(configured["default"]).lower()
        configured_profiles = configured.get("profiles", {})
        if isinstance(configured_profiles, dict):
            for name, profile in configured_profiles.items():
                if not isinstance(profile, dict):
                    continue
                key = str(name).lower()
                base = dict(merged["profiles"].get(key, {}))
                base.update(profile)
                merged["profiles"][key] = base
        return merged

    def horizon_profile(self, horizon: str | None = None) -> dict[str, Any]:
        """Return the configured horizon profile, with safe fallbacks if the section is absent."""
        profiles = self.horizons.get("profiles", {})
        default_name = str(self.horizons.get("default", "swing"))
        name = (horizon or default_name).lower()
        profile = profiles.get(name) or profiles.get(default_name) or {}
        # Always carry the resolved name back to the caller so logging/UIs can show it.
        profile = dict(profile)
        profile.setdefault("name", name)
        return profile


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
