from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from .config import Settings
from .context import fetch_news_items
from .utils import clamp, clean_symbol_list

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[float, str], None]


class NewsAnalysisError(RuntimeError):
    pass


CATEGORY_KEYWORDS = {
    "earnings_guidance": {"earnings", "revenue", "profit", "eps", "guidance", "forecast", "quarter", "results"},
    "analyst_action": {"upgrade", "downgrade", "price target", "initiates", "analyst", "rating"},
    "sec_filing": {"sec", "filing", "8-k", "10-k", "10-q", "s-1", "13d", "13f"},
    "macro": {"fed", "fomc", "rates", "inflation", "cpi", "jobs", "tariff", "oil", "dollar", "yield"},
    "legal_regulatory": {"lawsuit", "probe", "investigation", "regulator", "recall", "ban", "antitrust"},
    "m_and_a": {"acquire", "acquisition", "merger", "buyout", "stake", "deal"},
    "product_business": {"launch", "product", "partnership", "contract", "customer", "ai", "chip", "factory"},
    "market_sentiment": {"stock", "shares", "rally", "slump", "falls", "jumps", "market", "investors"},
}


def build_news_feed(symbols: list[str], settings: Settings, limit: int = 50, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    cfg = settings.context_agent.get("news_analysis", {})
    max_per_symbol = int(cfg.get("max_headlines_per_symbol", 8))
    clean_symbols = clean_symbol_list(symbols)
    _notify_progress(progress_callback, 0.05, f"Preparing news request for {len(clean_symbols)} symbol(s)")
    _notify_progress(progress_callback, 0.15, "Fetching configured headline sources")
    all_items = fetch_news_items(clean_symbols, limit=max(limit, len(clean_symbols) * max_per_symbol))
    _notify_progress(progress_callback, 0.40, f"Fetched {len(all_items)} headline item(s)")
    enriched = [_enrich_item(item) for item in all_items]
    _notify_progress(progress_callback, 0.55, "Classified headline sentiment, category, impact, and freshness")
    enriched = _attach_article_excerpts(enriched, cfg, progress_callback)
    # Prioritize fresh, relevant items so the dashboard's top row is the most
    # actionable headline, not just the first one the provider returned.
    enriched.sort(
        key=lambda item: (
            float(item.get("freshness", 0.0)) * float(item.get("day_trader_relevance", 0.0)),
            float(item.get("freshness", 0.0)),
            abs(float(item.get("impact", 0.0))),
        ),
        reverse=True,
    )
    fresh_threshold = float(cfg.get("fresh_window_minutes", 60))
    fresh_catalyst_count = sum(
        1 for item in enriched if (item.get("age_minutes") is not None and float(item["age_minutes"]) <= fresh_threshold and abs(float(item.get("impact", 0.0))) >= 0.15)
    )
    grouped = {symbol: [item for item in enriched if item.get("symbol") == symbol] for symbol in clean_symbols}
    summaries = []
    total_symbols = max(len(grouped), 1)
    for index, (symbol, items) in enumerate(grouped.items(), start=1):
        _notify_progress(progress_callback, 0.78 + 0.15 * ((index - 1) / total_symbols), f"Summarizing {symbol}")
        clipped = items[:max_per_symbol]
        summaries.append(_summarize_symbol_news(symbol, clipped, settings))
    _notify_progress(progress_callback, 0.95, "Finalizing news feed")
    return {
        "symbols": clean_symbols,
        "headline_count": len(enriched),
        "source_count": len([item for item in enriched if item.get("url")]),
        "fresh_catalyst_count": fresh_catalyst_count,
        "fresh_window_minutes": fresh_threshold,
        "article_excerpt_count": sum(1 for item in enriched if item.get("article_excerpt")),
        "summaries": summaries,
        "headlines": enriched[:limit],
        "analysis_provider": _actual_analysis_provider(summaries),
        "coverage": _coverage_metadata(settings, enriched),
    }


def _summarize_symbol_news(symbol: str, items: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    heuristic = _heuristic_symbol_summary(symbol, items)
    llm_summary = _llm_symbol_summary(symbol, items, settings)
    if llm_summary:
        if llm_summary.get("analysis_provider") == "heuristic_fallback":
            heuristic["analysis_provider"] = "heuristic_fallback"
            heuristic["llm_error"] = llm_summary.get("llm_error", "LLM summary unavailable")
            heuristic["grand_summary"] = f"Heuristic fallback used because the configured LLM was unavailable. {heuristic['grand_summary']}"
            return heuristic
        merged = {**heuristic, **llm_summary}
        merged["analysis_provider"] = llm_summary.get("analysis_provider", "llm")
        merged["sources"] = heuristic["sources"]
        merged["source_count"] = heuristic["source_count"]
        return merged
    return heuristic


def _heuristic_symbol_summary(symbol: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    bullish = [item for item in items if item.get("sentiment") == "bullish"]
    bearish = [item for item in items if item.get("sentiment") == "bearish"]
    neutral = [item for item in items if item.get("sentiment") == "neutral"]
    categories = _category_counts(items)
    catalysts = [item["title"] for item in items if float(item.get("impact", 0)) > 0.15][:3]
    risks = [item["title"] for item in items if float(item.get("impact", 0)) < -0.15][:3]
    top_category = max(categories, key=categories.get) if categories else "other"
    if not items:
        grand_summary = f"No recent configured headlines were returned for {symbol}."
    elif catalysts or risks:
        parts = []
        if catalysts:
            parts.append("positive catalysts include " + "; ".join(catalysts[:2]))
        if risks:
            parts.append("risks include " + "; ".join(risks[:2]))
        grand_summary = f"{symbol} news is driven by {top_category.replace('_', ' ')}; " + ". ".join(parts) + "."
    else:
        grand_summary = f"{symbol} has recent headlines, but no strong directional catalyst was detected by the heuristic classifier."
    return {
        "symbol": symbol,
        "grand_summary": grand_summary,
        "source_count": len([item for item in items if item.get("url")]),
        "headline_count": len(items),
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "dominant_category": top_category,
        "categories": categories,
        "day_trader_focus": {
            "catalyst": catalysts[0] if catalysts else "No strong fresh catalyst detected.",
            "risk": risks[0] if risks else "No obvious headline risk detected.",
            "tradeability": _tradeability_note(items),
            "no_trade_flags": _no_trade_flags(items),
        },
        "sources": items,
        "analysis_provider": "heuristic",
    }


def _llm_symbol_summary(symbol: str, items: list[dict[str, Any]], settings: Settings) -> dict[str, Any] | None:
    news_cfg = settings.context_agent.get("news_analysis", {}).get("llm", {})
    if not news_cfg.get("enabled", False):
        return None
    provider = str(news_cfg.get("provider", "")).lower()
    payload_items = [
        {
            "title": item.get("title", ""),
            "provider": item.get("provider", ""),
            "published": item.get("published", ""),
            "sentiment": item.get("sentiment", ""),
            "impact": item.get("impact", 0),
            "category": item.get("category", "other"),
            "article_excerpt": item.get("article_excerpt", ""),
        }
        for item in items
    ]
    try:
        if provider == "openai":
            api_key = os.environ.get(str(news_cfg.get("api_key_env", "OPENAI_API_KEY")))
            if not api_key:
                LOGGER.info("OpenAI news summary skipped because %s is not set", news_cfg.get("api_key_env", "OPENAI_API_KEY"))
                return None
            parsed = _call_openai_responses(symbol, payload_items, news_cfg, api_key)
            return _normalize_llm_summary(parsed, "openai")
        if provider in {"localdeploy", "openai_compatible", "local_openai"}:
            parsed = _call_openai_compatible_chat(symbol, payload_items, news_cfg)
            return _normalize_llm_summary(parsed, "localdeploy")
        return None
    except Exception as exc:
        LOGGER.warning("News LLM summary failed for %s with provider %s: %s", symbol, provider or "unknown", exc)
        if news_cfg.get("fallback_to_heuristic", True):
            return {"analysis_provider": "heuristic_fallback", "llm_error": str(exc)}
        raise NewsAnalysisError(f"LLM news summary failed for {symbol} and heuristic fallback is disabled: {exc}") from exc


def _news_llm_instructions() -> str:
    return (
        "You are a day-trading news classifier. Return only valid JSON. "
        "Summarize what matters for an active trader: catalyst, freshness, risk, "
        "tradeability, no-trade flags, and why the headlines may or may not matter. "
        "Use article_excerpt fields when present; otherwise rely only on the provided headline metadata. "
        "Do not give financial advice or tell the user to buy or sell. "
        "Return keys: grand_summary, dominant_category, day_trader_focus, notes. "
        "day_trader_focus must contain catalyst, risk, tradeability, no_trade_flags."
    )


def _call_openai_responses(symbol: str, payload_items: list[dict[str, Any]], news_cfg: dict[str, Any], api_key: str) -> dict[str, Any]:
    body = {
        "model": str(news_cfg.get("model", "gpt-5")),
        "instructions": _news_llm_instructions(),
        "input": json.dumps({"symbol": symbol, "headlines": payload_items}, ensure_ascii=False),
        "max_output_tokens": 900,
    }
    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=float(news_cfg.get("timeout_seconds", 30)),
    )
    response.raise_for_status()
    return json.loads(_strip_json_fence(_extract_response_text(response.json())))


def _call_openai_compatible_chat(symbol: str, payload_items: list[dict[str, Any]], news_cfg: dict[str, Any]) -> dict[str, Any]:
    base_url = os.environ.get("LOCALDEPLOY_BASE_URL") or str(news_cfg.get("base_url", "http://127.0.0.1:8100/v1/chat/completions"))
    model = os.environ.get("LOCALDEPLOY_NEWS_MODEL") or str(news_cfg.get("model", "qwen3vl_8b_ollama"))
    provider = str(news_cfg.get("provider", "")).lower()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _news_llm_instructions()},
            {"role": "user", "content": json.dumps({"symbol": symbol, "headlines": payload_items}, ensure_ascii=False)},
        ],
        "temperature": float(news_cfg.get("temperature", 0.1)),
        "max_tokens": int(news_cfg.get("max_output_tokens", 700)),
        "response_format": {"type": "json_object"},
    }
    if provider == "localdeploy":
        body["safe_mode"] = True
    response = httpx.post(base_url, json=body, timeout=float(news_cfg.get("timeout_seconds", 30)) + 5)
    response.raise_for_status()
    payload = response.json()
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    return json.loads(_strip_json_fence(str(content)))


def _normalize_llm_summary(parsed: dict[str, Any], provider: str) -> dict[str, Any]:
    focus = parsed.get("day_trader_focus", {})
    if not isinstance(focus, dict):
        focus = {}
    no_trade_flags = focus.get("no_trade_flags", [])
    if isinstance(no_trade_flags, str):
        no_trade_flags = [no_trade_flags]
    focus = {
        "catalyst": str(focus.get("catalyst", "No clear catalyst identified.")),
        "risk": str(focus.get("risk", "No clear headline risk identified.")),
        "tradeability": str(focus.get("tradeability", "Confirm with price, volume, VWAP, spread, and levels.")),
        "no_trade_flags": [str(flag) for flag in no_trade_flags],
    }
    notes = parsed.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    return {
        "grand_summary": str(parsed.get("grand_summary", "")),
        "dominant_category": str(parsed.get("dominant_category", "other")),
        "day_trader_focus": focus,
        "llm_notes": [str(note) for note in notes],
        "analysis_provider": provider,
    }


def _enrich_item(item: dict[str, Any]) -> dict[str, Any]:
    title = str(item.get("title", ""))
    category = _classify_category(title)
    impact = clamp(float(item.get("impact", 0.0) or 0.0), -1.0, 1.0)
    age_minutes = _published_age_minutes(item.get("published"))
    freshness = _freshness_score(age_minutes)
    return {
        **item,
        "symbol": str(item.get("symbol", "")).upper(),
        "category": category,
        "impact": impact,
        "age_minutes": age_minutes,
        "freshness": freshness,
        "day_trader_relevance": _relevance_score(title, impact, category, freshness),
    }


def _attach_article_excerpts(items: list[dict[str, Any]], cfg: dict[str, Any], progress_callback: ProgressCallback | None = None) -> list[dict[str, Any]]:
    scrape_cfg = cfg.get("article_scraping", {})
    if not scrape_cfg.get("enabled", False):
        return items
    max_articles_per_symbol = int(scrape_cfg.get("max_articles_per_symbol", 3))
    timeout_seconds = float(scrape_cfg.get("timeout_seconds", 8))
    max_chars = int(scrape_cfg.get("max_chars_per_article", 1400))
    user_agent = str(scrape_cfg.get("user_agent", "StockPredictor research crawler/0.1"))
    counts: dict[str, int] = {}
    output = []
    candidates = [item for item in items if item.get("url")]
    _notify_progress(progress_callback, 0.60, f"Fetching up to {max_articles_per_symbol} article page(s) per symbol")
    for index, item in enumerate(items):
        enriched_item = dict(item)
        symbol = str(item.get("symbol", "")).upper()
        url = str(item.get("url", "") or "")
        if url and counts.get(symbol, 0) < max_articles_per_symbol:
            try:
                excerpt = _fetch_article_excerpt(url, timeout_seconds=timeout_seconds, max_chars=max_chars, user_agent=user_agent)
                if excerpt:
                    enriched_item["article_excerpt"] = excerpt
                    enriched_item["article_fetched"] = True
                    counts[symbol] = counts.get(symbol, 0) + 1
                else:
                    enriched_item["article_fetched"] = False
                    enriched_item["article_error"] = "no extractable article text"
            except Exception as exc:
                LOGGER.info("Article excerpt fetch failed for %s: %s", url, exc)
                enriched_item["article_fetched"] = False
                enriched_item["article_error"] = str(exc)
        output.append(enriched_item)
        if candidates:
            _notify_progress(progress_callback, 0.60 + 0.15 * ((index + 1) / max(len(items), 1)), "Fetching article excerpts")
    return output


def _fetch_article_excerpt(url: str, timeout_seconds: float, max_chars: int, user_agent: str) -> str:
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=timeout_seconds,
        headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    prefix = response.text.lstrip()[:200].lower()
    if "html" not in content_type.lower() and "<html" not in prefix and "<!doctype html" not in prefix:
        return ""
    return _extract_article_text(response.text, max_chars=max_chars)


def _extract_article_text(html_text: str, max_chars: int = 1400) -> str:
    parser = _ArticleTextParser()
    parser.feed(html_text[:500_000])
    chunks = parser.text_chunks()
    text = " ".join(chunks)
    text = unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:max_chars].strip()


class _ArticleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture_depth = 0
        self._skip_depth = 0
        self._chunks: list[str] = []
        self._meta_description = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if lowered == "meta":
            data = {name.lower(): (value or "") for name, value in attrs}
            name = data.get("name", "").lower()
            prop = data.get("property", "").lower()
            if name == "description" or prop == "og:description":
                self._meta_description = data.get("content", "")
        if lowered in {"title", "h1", "h2", "p", "li"}:
            self._capture_depth += 1

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if lowered in {"title", "h1", "h2", "p", "li"} and self._capture_depth:
            self._capture_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._capture_depth:
            return
        text = data.strip()
        if len(text) >= 20:
            self._chunks.append(text)

    def text_chunks(self) -> list[str]:
        chunks = []
        if self._meta_description:
            chunks.append(self._meta_description)
        chunks.extend(self._chunks)
        return chunks


def _classify_category(title: str) -> str:
    lowered = title.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"


def _relevance_score(title: str, impact: float, category: str, freshness: float = 0.0) -> float:
    category_boost = 0.25 if category in {"earnings_guidance", "analyst_action", "sec_filing", "macro", "legal_regulatory", "m_and_a"} else 0.05
    title_boost = min(len(title.split()) / 80, 0.15)
    # Freshness compounds the relevance of high-impact items: a 60-minute-old
    # earnings beat matters more than a 12-hour-old one.
    freshness_boost = freshness * 0.20
    return clamp(abs(impact) + category_boost + title_boost + freshness_boost, 0.0, 1.0)


def _published_age_minutes(published: object) -> float | None:
    if published is None:
        return None
    raw = str(published)
    if not raw:
        return None
    parsed: datetime | None = None
    # yfinance sometimes returns epoch seconds, sometimes ISO strings.
    if raw.isdigit():
        try:
            parsed = datetime.fromtimestamp(int(raw), tz=timezone.utc)
        except (OverflowError, ValueError):
            parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - parsed).total_seconds() / 60
    return max(0.0, age)


def _freshness_score(age_minutes: float | None) -> float:
    """Decay-based score: 1.0 right now, 0.5 at ~3h, ~0.1 by ~24h."""
    if age_minutes is None:
        return 0.0
    if age_minutes <= 5:
        return 1.0
    if age_minutes <= 180:
        return clamp(1.0 - (age_minutes - 5) / 350, 0.5, 1.0)
    if age_minutes <= 24 * 60:
        return clamp(0.5 - (age_minutes - 180) / 2880, 0.1, 0.5)
    return 0.0


def _category_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        category = str(item.get("category", "other"))
        counts[category] = counts.get(category, 0) + 1
    return counts


def _tradeability_note(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No headline evidence; rely on scanner, liquidity, VWAP, and risk controls."
    high_relevance = [item for item in items if float(item.get("day_trader_relevance", 0)) >= 0.45]
    if high_relevance:
        return "Headline flow has trader-relevant catalysts; confirm with volume, VWAP, spread, and levels."
    return "Headline flow is low impact; avoid forcing a trade without price/volume confirmation."


def _no_trade_flags(items: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    if not items:
        return ["no recent headline source"]
    if not any(float(item.get("day_trader_relevance", 0)) >= 0.35 for item in items):
        flags.append("no high-relevance catalyst")
    if any(item.get("category") == "legal_regulatory" for item in items):
        flags.append("headline risk from legal/regulatory item")
    if not any(item.get("url") for item in items):
        flags.append("sources have no clickable links")
    return flags

def _actual_analysis_provider(summaries: list[dict[str, Any]]) -> str:
    providers = sorted({str(summary.get("analysis_provider", "heuristic")) for summary in summaries if summary})
    return ", ".join(providers) if providers else "heuristic"


def _coverage_metadata(settings: Settings, items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    llm_cfg = settings.context_agent.get("news_analysis", {}).get("llm", {})
    news_cfg = settings.context_agent.get("news_analysis", {})
    scrape_cfg = news_cfg.get("article_scraping", {})
    configured_sources = [str(source) for source in settings.context_agent.get("sources", [])]
    items = items or []
    article_excerpt_count = sum(1 for item in items if item.get("article_excerpt"))
    return {
        "configured_sources": configured_sources,
        "headline_provider": "yfinance_news" if "yfinance_news" in configured_sources else "configured/manual",
        "llm_enabled": bool(llm_cfg.get("enabled", False)),
        "llm_provider": str(llm_cfg.get("provider", "heuristic")),
        "fallback_to_heuristic": bool(llm_cfg.get("fallback_to_heuristic", True)),
        "article_body_scraping": bool(scrape_cfg.get("enabled", False)),
        "article_excerpt_count": article_excerpt_count,
        "source_mode": "headline metadata, provider links, and article excerpts" if scrape_cfg.get("enabled", False) else "headline metadata and provider links",
    }


def _analysis_provider(settings: Settings) -> str:
    llm_cfg = settings.context_agent.get("news_analysis", {}).get("llm", {})
    provider = str(llm_cfg.get("provider", "heuristic")).lower()
    if llm_cfg.get("enabled", False) and provider in {"localdeploy", "openai_compatible", "local_openai"}:
        return "localdeploy"
    if llm_cfg.get("enabled", False) and provider == "openai" and os.environ.get(str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))):
        return "openai"
    if llm_cfg.get("enabled", False):
        return "heuristic_fallback"
    return "heuristic"


def _extract_response_text(payload: dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(str(content["text"]))
    return "\n".join(parts)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:].strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _notify_progress(callback: ProgressCallback | None, progress: float, message: str) -> None:
    if callback is not None:
        callback(clamp(progress, 0.0, 1.0), message)
