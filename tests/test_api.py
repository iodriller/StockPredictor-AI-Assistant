from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from stockpredictor.api import create_app


def test_api_health_config_and_analyze(tmp_path: Path) -> None:
    config_path = _api_config(tmp_path)
    app = create_app(str(config_path))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    config = client.get("/config")
    assert config.status_code == 200
    assert config.json()["data"]["provider"] == "synthetic"

    analysis = client.post("/analyze/TEST")
    assert analysis.status_code == 200
    assert analysis.json()["snapshot"]["symbol"] == "TEST"

    latest = client.get("/signals/latest")
    assert latest.status_code == 200
    assert latest.json()[0]["snapshot"]["symbol"] == "TEST"


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
