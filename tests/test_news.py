from __future__ import annotations

from pathlib import Path

import yaml

from stockpredictor.config import load_settings
from stockpredictor.context import fetch_news_items
import pytest

from stockpredictor.news import NewsAnalysisError, _call_openai_compatible_chat, _extract_article_text, _strip_json_fence, build_news_feed


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
    raw["context_agent"]["news_analysis"]["llm"]["fallback_to_heuristic"] = True
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
    assert feed["analysis_provider"] == "localdeploy"
    assert feed["summaries"][0]["grand_summary"] == "LLM summary"
    assert feed["summaries"][0]["day_trader_focus"]["catalyst"] == "AI contract"


def test_news_feed_reports_heuristic_fallback_and_progress(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["news_analysis"]["llm"]["enabled"] = True
    raw["context_agent"]["news_analysis"]["llm"]["provider"] = "localdeploy"
    raw["context_agent"]["news_analysis"]["llm"]["fallback_to_heuristic"] = True
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50: [
            {
                "symbol": "TEST",
                "title": "TEST shares move on product launch",
                "provider": "Fixture News",
                "published": "2026-01-01T09:30:00Z",
                "url": "https://example.com/test",
                "impact": 0.2,
                "sentiment": "bullish",
            }
        ],
    )
    monkeypatch.setattr(
        "stockpredictor.news._call_openai_compatible_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    progress = []

    feed = build_news_feed(["TEST"], settings, limit=10, progress_callback=lambda value, message: progress.append((value, message)))

    assert feed["analysis_provider"] == "heuristic_fallback"
    assert feed["summaries"][0]["analysis_provider"] == "heuristic_fallback"
    assert "heuristic fallback" in feed["summaries"][0]["grand_summary"].lower()
    assert "llm_error" in feed["summaries"][0]
    assert feed["coverage"]["llm_enabled"] is True
    assert feed["coverage"]["article_body_scraping"] is False
    assert progress[0][0] == 0.05
    assert progress[-1][0] == 0.95


def test_news_feed_can_disable_heuristic_fallback(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["news_analysis"]["llm"]["enabled"] = True
    raw["context_agent"]["news_analysis"]["llm"]["provider"] = "localdeploy"
    raw["context_agent"]["news_analysis"]["llm"]["fallback_to_heuristic"] = False
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50: [{"symbol": "TEST", "title": "TEST wins contract", "url": "https://example.com/test"}],
    )
    monkeypatch.setattr(
        "stockpredictor.news._call_openai_compatible_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    with pytest.raises(NewsAnalysisError):
        build_news_feed(["TEST"], settings, limit=10)


def test_news_feed_can_attach_article_excerpts(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["news_analysis"]["article_scraping"]["enabled"] = True
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50: [
            {
                "symbol": "TEST",
                "title": "TEST raises guidance",
                "provider": "Fixture News",
                "published": "2026-01-01T09:30:00Z",
                "url": "https://example.com/test",
                "impact": 0.5,
                "sentiment": "bullish",
            }
        ],
    )

    class FakeResponse:
        headers = {"content-type": "text/html"}
        text = "<html><head><title>TEST news</title><meta name='description' content='Company raises full-year outlook.'></head><body><p>Management raised guidance after stronger demand from AI customers.</p></body></html>"

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr("stockpredictor.news.httpx.get", lambda *args, **kwargs: FakeResponse())

    feed = build_news_feed(["TEST"], settings, limit=10)

    assert feed["coverage"]["article_body_scraping"] is True
    assert feed["coverage"]["article_excerpt_count"] == 1
    assert "raised guidance" in feed["headlines"][0]["article_excerpt"].lower()


def test_extract_article_text_uses_meta_and_paragraphs() -> None:
    text = _extract_article_text(
        "<html><head><meta property='og:description' content='Key catalyst summary.'></head>"
        "<body><script>ignore me</script><h1>Headline here</h1><p>Paragraph with enough useful detail for extraction.</p></body></html>"
    )

    assert "Key catalyst summary" in text
    assert "Paragraph with enough useful detail" in text
    assert "ignore me" not in text


def test_fetch_news_items_passes_limit_to_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "stockpredictor.context._yfinance_news",
        lambda symbol, limit=25: [{"symbol": symbol, "title": str(index)} for index in range(limit)],
    )

    assert len(fetch_news_items(["TEST"], limit=8)) == 8


def test_strip_json_fence_handles_single_line_fence() -> None:
    assert _strip_json_fence('```json{"ok": true}```') == '{"ok": true}'
    assert _strip_json_fence('```\n{"ok": true}\n```') == '{"ok": true}'


def test_openai_compatible_body_omits_localdeploy_extensions(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"grand_summary":"ok","day_trader_focus":{}}'}}]}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr("stockpredictor.news.httpx.post", fake_post)

    _call_openai_compatible_chat(
        "TEST",
        [{"title": "TEST wins contract"}],
        {
            "provider": "openai_compatible",
            "base_url": "http://example.test/v1/chat/completions",
            "model": "fixture",
            "timeout_seconds": 5,
        },
    )

    assert "safe_mode" not in captured
    assert "timeout_seconds" not in captured


def _settings(tmp_path: Path):
    raw = yaml.safe_load(Path("configs/default.example.yaml").read_text(encoding="utf-8"))
    raw["data"]["provider"] = "synthetic"
    raw["context_agent"]["news_analysis"]["llm"]["enabled"] = False
    raw["context_agent"]["news_analysis"]["article_scraping"]["enabled"] = False
    config_path = tmp_path / "news_config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_settings(config_path)
