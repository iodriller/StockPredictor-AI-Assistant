from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from stockpredictor.api import create_app


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

    backtest = client.post("/backtest", json={"symbols": ["TEST"]})
    assert backtest.status_code == 200
    assert "trade_log" in backtest.json()
    assert "evaluations" in backtest.json()


def _api_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    raw["data"]["provider"] = "synthetic"
    raw["data"]["min_rows"] = 60
    raw["models"]["enabled"] = ["baseline"]
    raw["context_agent"]["enabled"] = False
    raw["watchlists"]["default"] = ["TEST"]
    config_path = tmp_path / "api_config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path
