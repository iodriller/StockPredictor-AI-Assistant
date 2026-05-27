from __future__ import annotations

from pathlib import Path

import yaml

from stockpredictor.config import load_settings
from stockpredictor.news import build_news_feed


def test_news_feed_builds_grand_summary_with_sources(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50: [
            {
                "symbol": "TEST",
                "title": "TEST raises guidance after earnings beat",
                "provider": "Fixture News",
                "published": "2026-01-01T09:30:00Z",
                "url": "https://example.com/test",
                "impact": 0.5,
                "sentiment": "bullish",
            }
        ],
    )

    feed = build_news_feed(["TEST"], settings, limit=10)

    assert feed["summaries"][0]["symbol"] == "TEST"
    assert feed["summaries"][0]["source_count"] == 1
    assert "guidance" in feed["summaries"][0]["grand_summary"].lower()
    assert feed["summaries"][0]["day_trader_focus"]["catalyst"]
    assert feed["headlines"][0]["category"] == "earnings_guidance"


def test_news_feed_can_use_localdeploy_llm(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["news_analysis"]["llm"]["enabled"] = True
    raw["context_agent"]["news_analysis"]["llm"]["provider"] = "localdeploy"
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50: [
            {
                "symbol": "TEST",
                "title": "TEST shares jump after AI contract",
                "provider": "Fixture News",
                "published": "2026-01-01T09:30:00Z",
                "url": "https://example.com/test",
                "impact": 0.5,
                "sentiment": "bullish",
            }
        ],
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"grand_summary":"LLM summary","dominant_category":"product_business","day_trader_focus":{"catalyst":"AI contract","risk":"headline may fade","tradeability":"confirm with RVOL and VWAP","no_trade_flags":["late extension"]},"notes":["fixture"]}'
                        }
                    }
                ]
            }

    monkeypatch.setattr("stockpredictor.news.httpx.post", lambda *args, **kwargs: FakeResponse())

    feed = build_news_feed(["TEST"], settings, limit=10)

    assert feed["summaries"][0]["analysis_provider"] == "localdeploy"
    assert feed["summaries"][0]["grand_summary"] == "LLM summary"
    assert feed["summaries"][0]["day_trader_focus"]["catalyst"] == "AI contract"


def _settings(tmp_path: Path):
    raw = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    raw["data"]["provider"] = "synthetic"
    raw["context_agent"]["news_analysis"]["llm"]["enabled"] = False
    config_path = tmp_path / "news_config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_settings(config_path)
