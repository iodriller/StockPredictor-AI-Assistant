from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

import stockpredictor.api as api_module
from stockpredictor.api import create_app


def test_api_module_does_not_create_app_at_import() -> None:
    assert not hasattr(api_module, "app")


def test_api_health_config_and_analyze(tmp_path: Path, monkeypatch) -> None:
    config_path = _api_config(tmp_path)
    app = create_app(str(config_path))
    client = TestClient(app)
    monkeypatch.setattr("stockpredictor.api.search_symbols", lambda q, limit=25: [{"symbol": "TEST", "name": "Test Corp", "source": "fixture"}])
    monkeypatch.setattr(
        "stockpredictor.api.build_news_feed",
        lambda symbols, settings, limit=25: {
            "symbols": symbols,
            "headline_count": 1,
            "source_count": 1,
            "summaries": [{"symbol": symbols[0], "grand_summary": "Test summary", "source_count": 1, "sources": []}],
            "headlines": [{"symbol": symbols[0], "title": "Test raises guidance", "sentiment": "bullish"}],
            "analysis_provider": "fixture",
        },
    )

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    config = client.get("/config")
    assert config.status_code == 200
    assert config.json()["data"]["provider"] == "synthetic"

    symbols = client.get("/symbols/search?q=test")
    assert symbols.status_code == 200
    assert symbols.json()[0]["symbol"] == "TEST"

    news = client.get("/news?symbols=TEST")
    assert news.status_code == 200
    assert news.json()["summaries"][0]["grand_summary"] == "Test summary"

    analysis_get = client.get("/analyze/TEST")
    assert analysis_get.status_code == 200
    assert analysis_get.json()["snapshot"]["symbol"] == "TEST"

    analysis = client.post("/analyze/TEST")
    assert analysis.status_code == 200
    assert analysis.json()["snapshot"]["symbol"] == "TEST"
    assert "scanner_row" in analysis.json()
    assert "setup_quality" in analysis.json()["risk_plan"]

    latest = client.get("/signals/latest")
    assert latest.status_code == 200
    assert latest.json()[0]["snapshot"]["symbol"] == "TEST"

    scan = client.post("/scan", json={"symbols": ["TEST"]})
    assert scan.status_code == 200
    assert scan.json()[0]["scanner_row"]["symbol"] == "TEST"

    scan_get = client.get("/scan?symbols=TEST")
    assert scan_get.status_code == 200
    assert scan_get.json()[0]["scanner_row"]["symbol"] == "TEST"

    session_analysis = client.post("/analyze/TEST", json={"session_id": "alpha"})
    assert session_analysis.status_code == 200
    alpha_latest = client.get("/signals/latest?session_id=alpha")
    assert alpha_latest.status_code == 200
    assert alpha_latest.json()[0]["snapshot"]["symbol"] == "TEST"
    missing_latest = client.get("/signals/latest?session_id=missing")
    assert missing_latest.status_code == 200
    assert missing_latest.json() == []

    backtest = client.post("/backtest", json={"symbols": ["TEST"]})
    assert backtest.status_code == 200
    assert "trade_log" in backtest.json()
    assert "evaluations" in backtest.json()

    journal = client.post("/journal", json={"symbol": "TEST", "action": "long", "setup_type": "vwap_reclaim"})
    assert journal.status_code == 200
    assert journal.json()["symbol"] == "TEST"

    journal_rows = client.get("/journal")
    assert journal_rows.status_code == 200
    assert journal_rows.json()[-1]["setup_type"] == "vwap_reclaim"


def _api_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/default.example.yaml").read_text(encoding="utf-8"))
    raw["data"]["provider"] = "synthetic"
    raw["data"]["min_rows"] = 60
    raw["models"]["enabled"] = ["baseline"]
    raw["context_agent"]["enabled"] = False
    raw["journal"] = {"enabled": True, "path": str(tmp_path / "journal.local.jsonl")}
    raw["watchlists"]["default"] = ["TEST"]
    config_path = tmp_path / "api_config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path
