from __future__ import annotations

from pathlib import Path
import threading
import time

import yaml

from stockpredictor.config import load_settings
from stockpredictor.context import fetch_news_items
import pytest

from stockpredictor.news import NewsAnalysisError, _attach_article_excerpts, _call_openai_compatible_chat, _classify_headlines, _extract_article_text, _normalize_llm_summary, _strip_json_fence, analyze_symbol_news, build_news_feed


def test_news_feed_builds_grand_summary_with_sources(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50, **kwargs: [
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


def test_analyze_symbol_news_returns_summary_and_evidence(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50, **kwargs: [
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

    result = analyze_symbol_news("test", settings, limit=10)

    assert result["symbol"] == "TEST"
    assert result["summary"]["symbol"] == "TEST"
    assert result["headlines"]
    assert result["headlines"][0]["category"] == "earnings_guidance"
    assert result["summary"]["day_trader_focus"]["catalyst"]


def test_analyze_symbol_news_limit_overrides_config_cap(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)  # default max_headlines_per_symbol is 50
    captured = {}

    def fake_fetch(symbols, limit=50, **kwargs):
        captured["limit"] = limit
        return [
            {"symbol": "TEST", "title": f"TEST headline {index}", "url": f"https://example.com/{index}", "impact": 0.0, "sentiment": "neutral"}
            for index in range(limit)
        ]

    monkeypatch.setattr("stockpredictor.news.fetch_news_items", fake_fetch)

    result = analyze_symbol_news("TEST", settings, limit=120)

    assert captured["limit"] == 120  # explicit actuator value, above the config cap of 50
    assert len(result["headlines"]) == 120


def test_normalize_llm_summary_extracts_stance() -> None:
    bullish = _normalize_llm_summary(
        {"grand_summary": "x", "stance": {"direction": "bullish", "conviction": 0.8}, "day_trader_focus": {}},
        "localdeploy",
    )
    bearish = _normalize_llm_summary(
        {"grand_summary": "x", "stance": {"direction": "bearish", "conviction": 0.5}, "day_trader_focus": {}},
        "localdeploy",
    )
    junk = _normalize_llm_summary(
        {"grand_summary": "x", "stance": "not a dict", "day_trader_focus": {}},
        "localdeploy",
    )

    assert bullish["stance"]["direction"] == "bullish"
    assert bullish["stance_score"] == pytest.approx(0.8)
    assert bearish["stance_score"] == pytest.approx(-0.5)
    assert junk["stance"]["direction"] == "neutral"
    assert junk["stance_score"] == 0.0


def test_news_feed_uses_requested_limit_for_symbol_summary(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50, **kwargs: [
            {
                "symbol": "TEST",
                "title": f"TEST headline {index}",
                "provider": "Fixture News",
                "published": "2026-01-01T09:30:00Z",
                "url": f"https://example.com/test/{index}",
                "impact": 0.0,
                "sentiment": "neutral",
            }
            for index in range(limit)
        ],
    )

    feed = build_news_feed(["TEST"], settings, limit=25)

    assert feed["requested_headline_limit"] == 25
    assert feed["returned_headline_count"] == 25
    assert feed["summary_headline_limit_per_symbol"] == 25
    assert feed["summaries"][0]["headline_count"] == 25


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
        lambda symbols, limit=50, **kwargs: [
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
        lambda symbols, limit=50, **kwargs: [
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
        lambda symbols, limit=50, **kwargs: [{"symbol": "TEST", "title": "TEST wins contract", "url": "https://example.com/test"}],
    )
    monkeypatch.setattr(
        "stockpredictor.news._call_openai_compatible_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    with pytest.raises(NewsAnalysisError):
        build_news_feed(["TEST"], settings, limit=10)


def test_news_feed_keeps_multi_symbol_results_when_one_llm_summary_fails(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["news_analysis"]["llm"]["enabled"] = True
    raw["context_agent"]["news_analysis"]["llm"]["provider"] = "localdeploy"
    raw["context_agent"]["news_analysis"]["llm"]["fallback_to_heuristic"] = False
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50, **kwargs: [
            {"symbol": "GOOD", "title": "GOOD wins contract", "url": "https://example.com/good", "impact": 0.4, "sentiment": "bullish"},
            {"symbol": "BAD", "title": "BAD reports delay", "url": "https://example.com/bad", "impact": -0.4, "sentiment": "bearish"},
        ],
    )

    def fake_call(symbol, payload_items, news_cfg):
        if symbol == "BAD":
            raise RuntimeError("bad LLM response")
        return {
            "grand_summary": "GOOD summary",
            "dominant_category": "product_business",
            "day_trader_focus": {"catalyst": "contract", "risk": "none", "tradeability": "confirm", "no_trade_flags": []},
        }

    monkeypatch.setattr("stockpredictor.news._call_openai_compatible_chat", fake_call)

    feed = build_news_feed(["GOOD", "BAD"], settings, limit=10)

    providers = {summary["symbol"]: summary["analysis_provider"] for summary in feed["summaries"]}
    assert providers == {"GOOD": "localdeploy", "BAD": "llm_error"}


def test_news_feed_summarizes_symbols_with_bounded_workers(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50, **kwargs: [{"symbol": symbol, "title": f"{symbol} headline"} for symbol in symbols],
    )
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_summary(symbol, items, settings):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"symbol": symbol, "analysis_provider": "fixture", "sources": items}

    monkeypatch.setattr("stockpredictor.news._summarize_symbol_news", fake_summary)

    build_news_feed(["AAA", "BBB", "CCC"], settings, limit=6)

    assert max_active == 2


def test_news_feed_can_attach_article_excerpts(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["news_analysis"]["article_scraping"]["enabled"] = True
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50, **kwargs: [
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


def test_article_excerpt_limit_caps_failed_attempts(monkeypatch) -> None:
    calls = []

    def fail_fetch(url, **kwargs):
        calls.append(url)
        raise RuntimeError("blocked")

    monkeypatch.setattr("stockpredictor.news._fetch_article_excerpt", fail_fetch)
    items = [{"symbol": "TEST", "title": f"Headline {index}", "url": f"https://example.com/{index}"} for index in range(8)]

    enriched = _attach_article_excerpts(
        items,
        {"article_scraping": {"enabled": True, "max_articles_per_symbol": 3}},
    )

    assert len(calls) == 3
    assert len(enriched) == 8


def test_headline_classifier_records_llm_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        "stockpredictor.news._llm_headline_classifications",
        lambda symbol, items, config: ([{"category": "analyst_action", "sentiment": "bullish", "impact": 0.8}], "localdeploy"),
    )

    classified = _classify_headlines(
        [{"symbol": "TEST", "title": "Opaque headline", "category": "other", "sentiment": "neutral", "impact": 0.0, "freshness": 1.0}],
        {"headline_classifier": {"enabled": True}},
    )

    assert classified[0]["category"] == "analyst_action"
    assert classified[0]["sentiment"] == "bullish"
    assert classified[0]["impact"] == 0.8
    assert classified[0]["classification_provider"] == "localdeploy"


def test_headline_classifier_falls_back_to_keyword_with_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        "stockpredictor.news._llm_headline_classifications",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    classified = _classify_headlines(
        [{"symbol": "TEST", "title": "Opaque headline", "category": "other", "sentiment": "neutral", "impact": 0.0, "freshness": 1.0, "classification_provider": "keyword"}],
        {"headline_classifier": {"enabled": True, "fallback_to_keyword": True}},
    )

    assert classified[0]["classification_provider"] == "keyword"


def test_news_feed_scrapes_ranked_headlines_first(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["news_analysis"]["article_scraping"]["enabled"] = True
    raw["context_agent"]["news_analysis"]["article_scraping"]["max_articles_per_symbol"] = 1
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50, **kwargs: [
            {"symbol": "TEST", "title": "Old recap", "url": "https://example.com/old", "published": "2026-01-01T09:30:00Z", "impact": 0.0},
            {"symbol": "TEST", "title": "TEST raises guidance after earnings beat", "url": "https://example.com/fresh", "published": "2099-01-01T09:30:00Z", "impact": 0.8},
        ],
    )
    calls = []
    monkeypatch.setattr("stockpredictor.news._fetch_article_excerpt", lambda url, **kwargs: calls.append(url) or "excerpt")

    build_news_feed(["TEST"], settings, limit=10)

    assert calls == ["https://example.com/fresh"]


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

    assert len(fetch_news_items(["TEST"], limit=8, sources=["yfinance_news"])) == 8


def test_fetch_news_items_interleaves_configured_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        "stockpredictor.context._yfinance_news",
        lambda symbol, limit=25: [{"symbol": symbol, "source": "yfinance_news", "title": f"yf-{index}"} for index in range(limit)],
    )
    monkeypatch.setattr(
        "stockpredictor.context._yahoo_search_news",
        lambda symbol, limit=25: [{"symbol": symbol, "source": "yahoo_search_news", "title": f"ys-{index}"} for index in range(limit)],
    )
    monkeypatch.setattr(
        "stockpredictor.context._google_news_rss",
        lambda symbol, limit=25, source_config=None: [{"symbol": symbol, "source": "google_news_rss", "title": f"gn-{index}"} for index in range(limit)],
    )

    items = fetch_news_items(["TEST"], limit=6, sources=["yfinance_news", "yahoo_search_news", "google_news_rss"])

    assert [item["source"] for item in items] == [
        "yfinance_news",
        "yahoo_search_news",
        "google_news_rss",
        "yfinance_news",
        "yahoo_search_news",
        "google_news_rss",
    ]


def test_fetch_news_items_distributes_limit_across_symbols(monkeypatch) -> None:
    monkeypatch.setattr(
        "stockpredictor.context._yfinance_news",
        lambda symbol, limit=25: [{"symbol": symbol, "source": "yfinance_news", "title": f"{symbol}-yf-{index}"} for index in range(limit)],
    )
    monkeypatch.setattr(
        "stockpredictor.context._yahoo_search_news",
        lambda symbol, limit=25: [{"symbol": symbol, "source": "yahoo_search_news", "title": f"{symbol}-ys-{index}"} for index in range(limit)],
    )
    monkeypatch.setattr(
        "stockpredictor.context._google_news_rss",
        lambda symbol, limit=25, source_config=None: [{"symbol": symbol, "source": "google_news_rss", "title": f"{symbol}-gn-{index}"} for index in range(limit)],
    )

    items = fetch_news_items(["AAA", "BBB"], limit=6, sources=["yfinance_news", "yahoo_search_news", "google_news_rss"])

    assert [item["symbol"] for item in items] == ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"]


def test_fetch_news_items_dedupes_syndicated_headlines_with_different_links(monkeypatch) -> None:
    monkeypatch.setattr(
        "stockpredictor.context._yfinance_news",
        lambda symbol, limit=25: [{"symbol": symbol, "source": "yfinance_news", "title": "Same syndicated story", "url": "https://example.com/one"}],
    )
    monkeypatch.setattr(
        "stockpredictor.context._yahoo_search_news",
        lambda symbol, limit=25: [{"symbol": symbol, "source": "yahoo_search_news", "title": "  SAME syndicated   story ", "url": "https://example.com/two"}],
    )

    items = fetch_news_items(["TEST"], limit=5, sources=["yfinance_news", "yahoo_search_news"])

    assert len(items) == 1


def test_fetch_news_items_uses_bounded_parallel_sources(monkeypatch) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_fetch(symbol, limit=25, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return [{"symbol": symbol, "source": "fixture", "title": f"{symbol} headline"}]

    monkeypatch.setattr("stockpredictor.context._yfinance_news", fake_fetch)
    monkeypatch.setattr("stockpredictor.context._yahoo_search_news", fake_fetch)
    monkeypatch.setattr("stockpredictor.context._google_news_rss", fake_fetch)

    fetch_news_items(
        ["AAA", "BBB"],
        limit=6,
        sources=["yfinance_news", "yahoo_search_news", "google_news_rss"],
        source_config={"fetch_workers": 3},
    )

    assert 1 < max_active <= 3


def test_normalize_llm_summary_handles_non_list_flags() -> None:
    summary = _normalize_llm_summary(
        {
            "grand_summary": "ok",
            "day_trader_focus": {"catalyst": "contract", "risk": "none", "tradeability": "confirm", "no_trade_flags": False},
            "notes": False,
        },
        "localdeploy",
    )

    assert summary["day_trader_focus"]["no_trade_flags"] == []
    assert summary["llm_notes"] == []


def test_normalize_llm_summary_discards_reassuring_no_trade_flag_text() -> None:
    summary = _normalize_llm_summary(
        {
            "grand_summary": "bullish read",
            "day_trader_focus": {
                "catalyst": "AI product launch",
                "risk": "none",
                "tradeability": "confirm",
                "no_trade_flags": [
                    "No significant red flags; no earnings or macro catalysts are imminent.",
                    "late extension from VWAP",
                ],
            },
        },
        "localdeploy",
    )

    assert summary["day_trader_focus"]["no_trade_flags"] == ["late extension from VWAP"]


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
    raw["context_agent"]["news_analysis"]["headline_classifier"]["enabled"] = False
    raw["context_agent"]["news_analysis"]["article_scraping"]["enabled"] = False
    config_path = tmp_path / "news_config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_settings(config_path)
