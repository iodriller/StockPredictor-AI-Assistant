from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings


LOGGER = logging.getLogger(__name__)
DEFAULT_CACHE_PATH = Path("data") / "dashboard_state.local.pkl"
CACHE_KEYS = {
    "latest_scan_results",
    "latest_scan_symbols",
    "latest_analysis",
    "latest_news_feed",
    "latest_backtest_report",
    "latest_backtest_symbols",
}


def load_dashboard_cache(settings: Settings) -> dict[str, Any]:
    cfg = settings.dashboard.get("result_cache", {})
    if not cfg.get("enabled", True):
        return {}
    path = dashboard_cache_path(settings)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        saved_at = datetime.fromisoformat(str(payload["saved_at"]))
        max_age_minutes = float(cfg.get("max_age_minutes", 240))
        age_minutes = (datetime.now(timezone.utc) - saved_at.astimezone(timezone.utc)).total_seconds() / 60
        if age_minutes > max_age_minutes:
            return {}
        state = payload.get("state", {})
        return {key: value for key, value in state.items() if key in CACHE_KEYS}
    except Exception as exc:
        LOGGER.warning("Ignoring invalid local dashboard cache at %s: %s", path, exc)
        return {}


def save_dashboard_cache(settings: Settings, state: dict[str, Any]) -> None:
    cfg = settings.dashboard.get("result_cache", {})
    if not cfg.get("enabled", True):
        return
    path = dashboard_cache_path(settings)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state": {key: value for key, value in state.items() if key in CACHE_KEYS},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except Exception as exc:
        LOGGER.warning("Could not save local dashboard cache at %s: %s", path, exc)


def dashboard_cache_path(settings: Settings) -> Path:
    configured = settings.dashboard.get("result_cache", {}).get("path", str(DEFAULT_CACHE_PATH))
    path = Path(str(configured))
    if not path.is_absolute():
        path = settings.path.parent.parent / path
    return path.resolve()
