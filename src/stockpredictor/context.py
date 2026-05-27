from __future__ import annotations

from pathlib import Path
from typing import Any

import logging

from .config import Settings
from .contracts import ContextSummary
from .utils import TTLCache, clamp, dedupe_preserve_order


LOGGER = logging.getLogger(__name__)
_NEWS_CACHE = TTLCache(ttl_seconds=120)


POSITIVE_WORDS = {"beat", "beats", "raise", "raises", "upgrade", "growth", "approval", "strong", "record", "deal", "guidance"}
NEGATIVE_WORDS = {"miss", "misses", "cut", "downgrade", "probe", "lawsuit", "weak", "warning", "delay", "risk", "recall"}


def build_context_summary(symbol: str, settings: Settings, include_live_sources: bool = True) -> ContextSummary:
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
    if include_live_sources and "yfinance_news" in sources:
        items.extend(fetch_news_items([symbol], limit=int(cfg.get("news_limit", 8))))

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
        },
        reasons_to_trade=dedupe_preserve_order(reasons_to_trade)[:8],
        reasons_to_skip=dedupe_preserve_order(reasons_to_skip)[:8],
    )


def fetch_news_items(symbols: list[str], limit: int = 25) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for symbol in symbols:
        items.extend(_yfinance_news(symbol, limit=limit))
    return items[:limit]


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
