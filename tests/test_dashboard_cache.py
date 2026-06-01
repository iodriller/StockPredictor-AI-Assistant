from __future__ import annotations

from pathlib import Path

from stockpredictor.config import Settings
from stockpredictor.dashboard_cache import load_dashboard_cache, save_dashboard_cache


def test_dashboard_cache_persists_allowed_results(tmp_path: Path) -> None:
    settings = Settings(
        raw={
            "dashboard": {
                "result_cache": {
                    "enabled": True,
                    "path": str(tmp_path / "dashboard.local.pkl"),
                    "max_age_minutes": 60,
                }
            }
        },
        path=tmp_path / "config.yaml",
    )

    save_dashboard_cache(settings, {"latest_scan_symbols": ["AMD"], "latest_backtest_depth": "Standard review", "ignored": "value"})

    assert load_dashboard_cache(settings) == {"latest_scan_symbols": ["AMD"], "latest_backtest_depth": "Standard review"}
