from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import logging
import xml.etree.ElementTree as ET
import math

import httpx

from .config import Settings
from .contracts import ContextSummary
from .utils import TTLCache, clamp, dedupe_preserve_order


LOGGER = logging.getLogger(__name__)
_NEWS_CACHE = TTLCache(ttl_seconds=120)
DEFAULT_NEWS_SOURCES = ("yfinance_news", "yahoo_search_news", "google_news_rss")


POSITIVE_WORDS = {"beat", "beats", "raise", "raises", "upgrade", "growth", "approval", "strong", "record", "deal", "guidance"}
NEGATIVE_WORDS = {"miss", "misses", "cut", "downgrade", "probe", "lawsuit", "weak", "warning", "delay", "risk", "recall"}


def build_context_summary(
    symbol: str,
    settings: Settings,
    include_live_sources: bool = True,
    news_analysis: dict[str, Any] | None = None,
) -> ContextSummary:
    """Build the catalyst/context summary that feeds the signal fusion.

    When ``news_analysis`` is supplied (the deep-dive path; see
    ``news.analyze_symbol_news``), its enriched headlines drive the catalyst score
    and the rich LLM summary is attached for white-box display, so the trade
    decision both consumes and exposes the gathered news. When it is ``None`` (the
    scanner path), the lightweight live-fetch behavior is used as before.
    """
    cfg = settings.context_agent
    mind_file = _resolve_mind_file(settings, str(cfg.get("mind_file", "traders.mind.md")))
    checklist = _load_trader_checklist(mind_file)
    if not cfg.get("enabled", False):
        return ContextSummary(
            symbol=symbol.upper(),
            enabled=False,
            score=0.0,
            sentiment="neutral",
            raw_summary="Context agent disabled by configuration.",
            features={"context_confidence": 0.0, "checklist_items": float(len(checklist))},
            reasons_to_skip=["context agent disabled"],
        )

    items: list[dict[str, Any]] = []
    sources = [str(source) for source in cfg.get("sources", [])]
    if "manual" in sources:
        items.extend(cfg.get("manual_items", []))
    news_headlines = [dict(item) for item in (news_analysis or {}).get("headlines", [])]
    if news_headlines:
        # Reuse the same scoring loop with the already-enriched, LLM-aware headlines.
        items.extend(news_headlines)
    elif include_live_sources and any(source in sources for source in DEFAULT_NEWS_SOURCES):
        items.extend(fetch_news_items([symbol], limit=int(cfg.get("news_limit", 8)), sources=sources, source_config=cfg.get("news_analysis", {})))

    catalysts: list[str] = []
    risks: list[str] = []
    reasons_to_trade: list[str] = []
    reasons_to_skip: list[str] = []
    score_parts: list[float] = []
    freshness_parts: list[float] = []
    market_alignment_parts: list[float] = []
    sector_alignment_parts: list[float] = []
    used_sources: list[str] = []
    for item in items:
        title = str(item.get("title") or item.get("headline") or "").strip()
        if not title:
            continue
        impact = _score_item(title, item)
        score_parts.append(impact)
        freshness_parts.append(_bounded_item_float(item, "freshness", default=0.5))
        market_alignment_parts.append(_bounded_item_float(item, "market_alignment", default=0.0))
        sector_alignment_parts.append(_bounded_item_float(item, "sector_alignment", default=0.0))
        used_sources.append(str(item.get("source", "configured")))
        if impact >= 0.15:
            catalysts.append(title)
            reasons_to_trade.append(f"positive catalyst: {title}")
        elif impact <= -0.15:
            risks.append(title)
            reasons_to_skip.append(f"context risk: {title}")

    catalyst_score = clamp(sum(score_parts) / len(score_parts), -1.0, 1.0) if score_parts else 0.0
    # Merge the LLM's net directional stance into the catalyst score so its read of
    # the news actually moves (and visibly shapes) the decision, instead of only the
    # headline keyword impacts. Heuristic stances are not blended in — they already
    # equal the keyword view, so blending would double-count.
    news_payload = _news_analysis_payload(news_analysis)
    news_stance_score = _llm_stance_score(news_payload)
    keyword_catalyst_score = catalyst_score
    if news_stance_score is not None:
        stance_weight = float(cfg.get("news_analysis", {}).get("llm_stance_weight", 0.6))
        catalyst_score = clamp(stance_weight * news_stance_score + (1.0 - stance_weight) * catalyst_score, -1.0, 1.0)
    catalyst_freshness = clamp(sum(freshness_parts) / len(freshness_parts), 0.0, 1.0) if freshness_parts else 0.0
    market_alignment_present = any(part != 0 for part in market_alignment_parts)
    sector_alignment_present = any(part != 0 for part in sector_alignment_parts)
    market_alignment = clamp(sum(market_alignment_parts) / len(market_alignment_parts), -1.0, 1.0) if market_alignment_parts else 0.0
    sector_alignment = clamp(sum(sector_alignment_parts) / len(sector_alignment_parts), -1.0, 1.0) if sector_alignment_parts else 0.0
    # Re-normalize: when an alignment input is absent (provider didn't supply it),
    # roll its weight back into catalyst_score so the context score isn't silently
    # diluted by inert weights.
    weights = {"catalyst": 0.65, "market": 0.20 if market_alignment_present else 0.0, "sector": 0.15 if sector_alignment_present else 0.0}
    total_weight = sum(weights.values()) or 1.0
    score = clamp(
        ((catalyst_score * weights["catalyst"]) + (market_alignment * weights["market"]) + (sector_alignment * weights["sector"])) / total_weight,
        -1.0,
        1.0,
    )
    sentiment = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
    if not catalysts:
        reasons_to_skip.append("no strong configured catalyst")
    if score < 0:
        reasons_to_skip.append("context score is negative")
    if score > 0 and catalysts:
        reasons_to_trade.append("context supports the setup")

    # Fold the rich news analysis (LLM or heuristic) into the context: surface its
    # no-trade flags as reasons-to-skip and keep its summary as white-box evidence.
    news_focus = news_payload.get("day_trader_focus", {}) if news_payload else {}
    news_no_trade_flags = [str(flag) for flag in news_focus.get("no_trade_flags", []) if str(flag).strip()]
    for flag in news_no_trade_flags:
        reasons_to_skip.append(f"news no-trade flag: {flag}")

    raw_summary = _summarize_context(symbol, catalysts, risks, mind_file, checklist)
    return ContextSummary(
        symbol=symbol.upper(),
        enabled=True,
        score=score,
        sentiment=sentiment,
        catalysts=catalysts[:8],
        risks=risks[:8],
        sources=sorted(set(used_sources)),
        raw_summary=raw_summary,
        features={
            "context_confidence": min(1.0, len(score_parts) / 4) if score_parts else 0.0,
            "catalyst_count": float(len(catalysts)),
            "risk_count": float(len(risks)),
            "catalyst_score": catalyst_score,
            "catalyst_freshness": catalyst_freshness,
            "market_alignment": market_alignment,
            "sector_alignment": sector_alignment,
            "checklist_items": float(len(checklist)),
            "news_llm_enabled": bool(cfg.get("news_analysis", {}).get("llm", {}).get("enabled", False)),
            "news_no_trade_flag_count": float(len(news_no_trade_flags)),
            "news_stance_score": float(news_stance_score) if news_stance_score is not None else 0.0,
            "keyword_catalyst_score": keyword_catalyst_score,
        },
        reasons_to_trade=dedupe_preserve_order(reasons_to_trade)[:8],
        reasons_to_skip=dedupe_preserve_order(reasons_to_skip)[:8],
        news_analysis=news_payload,
        evidence=news_headlines,
    )


def _news_analysis_payload(news_analysis: dict[str, Any] | None) -> dict[str, Any]:
    """Compact, display-ready slice of a per-symbol news summary for white-box UI.

    Drops the heavy ``sources`` list (the headlines are carried separately as
    ``evidence``) and keeps only the fields the decision panel renders.
    """
    if not news_analysis:
        return {}
    summary = news_analysis.get("summary", {}) or {}
    if not summary:
        return {}
    keys = (
        "grand_summary",
        "dominant_category",
        "day_trader_focus",
        "stance",
        "stance_score",
        "analysis_provider",
        "llm_notes",
        "llm_error",
        "bullish_count",
        "bearish_count",
        "neutral_count",
        "headline_count",
        "source_count",
    )
    return {key: summary[key] for key in keys if key in summary}


def _llm_stance_score(news_payload: dict[str, Any]) -> float | None:
    """Signed stance score from the news summary, only when it came from a real LLM.

    Heuristic stances mirror the keyword catalyst score, so they are intentionally
    excluded here to avoid double-counting the same signal.
    """
    if not news_payload:
        return None
    if str(news_payload.get("analysis_provider", "")) not in {"localdeploy", "openai"}:
        return None
    raw = news_payload.get("stance_score")
    if raw is None:
        return None
    try:
        return clamp(float(raw), -1.0, 1.0)
    except (TypeError, ValueError):
        return None


def fetch_news_items(
    symbols: list[str],
    limit: int = 25,
    sources: list[str] | None = None,
    source_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    active_sources = _active_news_sources(sources)
    source_config = source_config or {}
    per_symbol_limit = max(1, math.ceil(limit / max(len(symbols), 1)))
    items: list[dict[str, Any]] = []
    for symbol in symbols:
        source_batches: list[list[dict[str, Any]]] = []
        if "yfinance_news" in active_sources:
            source_batches.append(_yfinance_news(symbol, limit=per_symbol_limit))
        if "yahoo_search_news" in active_sources:
            source_batches.append(_yahoo_search_news(symbol, limit=per_symbol_limit))
        if "google_news_rss" in active_sources:
            source_batches.append(_google_news_rss(symbol, limit=per_symbol_limit, source_config=source_config.get("google_news_rss", {})))
        symbol_items = _interleave_news_batches(source_batches)
        items.extend(_dedupe_news_items(symbol_items)[:per_symbol_limit])
    return items[:limit]


def _active_news_sources(sources: list[str] | None) -> list[str]:
    requested = [str(source).lower() for source in (sources or DEFAULT_NEWS_SOURCES)]
    active = [source for source in requested if source in DEFAULT_NEWS_SOURCES]
    return active or ["yfinance_news"]


def _yfinance_news(symbol: str, limit: int = 25) -> list[dict[str, Any]]:
    cache_key = ("yfinance_news", symbol.upper(), limit)
    cached = _NEWS_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    try:
        import yfinance as yf

        news = yf.Ticker(symbol).news or []
    except Exception as exc:
        LOGGER.warning("yfinance news fetch failed for %s: %s", symbol, exc)
        return []
    items = []
    for entry in news[:limit]:
        content = entry.get("content", entry) if isinstance(entry, dict) else {}
        title = content.get("title") or entry.get("title") if isinstance(entry, dict) else ""
        link = ""
        canonical = content.get("canonicalUrl") if isinstance(content, dict) else None
        if isinstance(canonical, dict):
            link = str(canonical.get("url") or "")
        elif isinstance(entry, dict):
            link = str(entry.get("link") or "")
        provider = ""
        content_provider = content.get("provider") if isinstance(content, dict) else None
        if isinstance(content_provider, dict):
            provider = str(content_provider.get("displayName") or "")
        published = content.get("pubDate") or entry.get("providerPublishTime", "") if isinstance(entry, dict) else ""
        if title:
            impact = _score_item(title, {})
            items.append(
                {
                    "symbol": symbol.upper(),
                    "source": "yfinance_news",
                    "provider": provider or "Yahoo Finance",
                    "title": title,
                    "url": link,
                    "published": str(published),
                    "impact": impact,
                    "sentiment": "bullish" if impact > 0.15 else "bearish" if impact < -0.15 else "neutral",
                }
            )
    _NEWS_CACHE.set(cache_key, list(items))
    return items


def _yahoo_search_news(symbol: str, limit: int = 25) -> list[dict[str, Any]]:
    cache_key = ("yahoo_search_news", symbol.upper(), limit)
    cached = _NEWS_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    try:
        response = httpx.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": symbol.upper(), "quotesCount": 0, "newsCount": limit},
            headers={"User-Agent": "Mozilla/5.0 StockPredictor/0.1"},
            timeout=10,
        )
        response.raise_for_status()
        news = response.json().get("news", [])
    except Exception as exc:
        LOGGER.warning("Yahoo search news fetch failed for %s: %s", symbol, exc)
        return []

    items = []
    for entry in news[:limit]:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        published = entry.get("providerPublishTime", "")
        if isinstance(published, (int, float)):
            published = datetime.fromtimestamp(int(published), tz=timezone.utc).isoformat()
        impact = _score_item(title, {})
        items.append(
            {
                "symbol": symbol.upper(),
                "source": "yahoo_search_news",
                "provider": str(entry.get("publisher") or "Yahoo Finance Search"),
                "title": title,
                "url": str(entry.get("link") or ""),
                "published": str(published),
                "impact": impact,
                "sentiment": "bullish" if impact > 0.15 else "bearish" if impact < -0.15 else "neutral",
            }
        )
    _NEWS_CACHE.set(cache_key, list(items))
    return items


def _google_news_rss(symbol: str, limit: int = 25, source_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    source_config = source_config or {}
    query_template = str(source_config.get("query_template", "{symbol} stock"))
    query = query_template.format(symbol=symbol.upper())
    params = {
        "q": query,
        "hl": str(source_config.get("hl", "en-US")),
        "gl": str(source_config.get("gl", "US")),
        "ceid": str(source_config.get("ceid", "US:en")),
    }
    cache_key = ("google_news_rss", symbol.upper(), limit, tuple(sorted(params.items())))
    cached = _NEWS_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    try:
        response = httpx.get(
            "https://news.google.com/rss/search",
            params=params,
            headers={"User-Agent": "Mozilla/5.0 StockPredictor/0.1"},
            timeout=float(source_config.get("timeout_seconds", 10)),
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception as exc:
        LOGGER.warning("Google News RSS fetch failed for %s: %s", symbol, exc)
        return []

    items = []
    for entry in root.findall("./channel/item")[:limit]:
        title = (entry.findtext("title") or "").strip()
        if not title:
            continue
        provider = (entry.findtext("source") or "").strip()
        published = _rss_date_to_iso(entry.findtext("pubDate"))
        impact = _score_item(title, {})
        items.append(
            {
                "symbol": symbol.upper(),
                "source": "google_news_rss",
                "provider": provider or "Google News",
                "title": title,
                "url": str(entry.findtext("link") or ""),
                "published": published,
                "impact": impact,
                "sentiment": "bullish" if impact > 0.15 else "bearish" if impact < -0.15 else "neutral",
            }
        )
    _NEWS_CACHE.set(cache_key, list(items))
    return items


def _rss_date_to_iso(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _dedupe_news_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped = []
    for item in items:
        key = (str(item.get("symbol") or ""), str(item.get("url") or item.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _interleave_news_batches(batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    max_len = max((len(batch) for batch in batches), default=0)
    for index in range(max_len):
        for batch in batches:
            if index < len(batch):
                merged.append(batch[index])
    return merged


def _score_item(title: str, item: dict[str, Any]) -> float:
    if "impact" in item:
        try:
            return clamp(float(item["impact"]), -1.0, 1.0)
        except (TypeError, ValueError):
            pass
    sentiment = str(item.get("sentiment", "")).lower()
    if sentiment == "bullish":
        return 0.35
    if sentiment == "bearish":
        return -0.35
    words = {word.strip(".,:;!?()[]{}").lower() for word in title.split()}
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    return clamp((positive - negative) * 0.20, -0.8, 0.8)


def _bounded_item_float(item: dict[str, Any], key: str, default: float) -> float:
    try:
        return clamp(float(item.get(key, default)), -1.0, 1.0)
    except (TypeError, ValueError):
        return default


def _resolve_mind_file(settings: Settings, mind_file: str) -> Path:
    path = Path(mind_file)
    if path.is_absolute():
        return path
    return (settings.path.parent.parent / path).resolve()


def _load_trader_checklist(mind_file: Path) -> list[str]:
    if not mind_file.exists():
        return []
    items: list[str] = []
    for line in mind_file.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        marker = stripped[:2]
        if marker in {"- ", "* ", "+ "}:
            content = stripped[2:].strip()
            if content:
                items.append(content)
    return items


def _summarize_context(symbol: str, catalysts: list[str], risks: list[str], mind_file: Path, checklist: list[str]) -> str:
    checklist_summary = f"trader checklist loaded with {len(checklist)} items" if checklist else "trader checklist unavailable"
    if catalysts or risks:
        parts = [f"{symbol.upper()} context summary ({checklist_summary})."]
        if catalysts:
            parts.append("Catalysts: " + "; ".join(catalysts[:3]))
        if risks:
            parts.append("Risks: " + "; ".join(risks[:3]))
        return " ".join(parts)
    return f"{symbol.upper()} has no strong configured catalyst. {checklist_summary}; default to price, volume, levels, and risk controls."
