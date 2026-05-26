from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .contracts import ContextSummary
from .utils import clamp


POSITIVE_WORDS = {"beat", "beats", "raise", "raises", "upgrade", "growth", "approval", "strong", "record", "deal"}
NEGATIVE_WORDS = {"miss", "misses", "cut", "downgrade", "probe", "lawsuit", "weak", "warning", "delay", "risk"}


def build_context_summary(symbol: str, settings: Settings, include_live_sources: bool = True) -> ContextSummary:
    cfg = settings.context_agent
    if not cfg.get("enabled", False):
        return ContextSummary(
            symbol=symbol.upper(),
            enabled=False,
            score=0.0,
            sentiment="neutral",
            raw_summary="Context agent disabled by configuration.",
            features={"context_confidence": 0.0},
        )

    items: list[dict[str, Any]] = []
    sources = [str(source) for source in cfg.get("sources", [])]
    if "manual" in sources:
        items.extend(cfg.get("manual_items", []))
    if include_live_sources and "yfinance_news" in sources:
        items.extend(_yfinance_news(symbol))

    catalysts: list[str] = []
    risks: list[str] = []
    score_parts: list[float] = []
    used_sources: list[str] = []
    for item in items:
        title = str(item.get("title") or item.get("headline") or "").strip()
        if not title:
            continue
        impact = _score_item(title, item)
        score_parts.append(impact)
        used_sources.append(str(item.get("source", "configured")))
        if impact >= 0.15:
            catalysts.append(title)
        elif impact <= -0.15:
            risks.append(title)

    score = clamp(sum(score_parts) / len(score_parts), -1.0, 1.0) if score_parts else 0.0
    sentiment = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
    mind_file = _resolve_mind_file(settings, str(cfg.get("mind_file", "traders.mind.md")))
    raw_summary = _summarize_context(symbol, catalysts, risks, mind_file)
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
            "llm_enabled": bool(cfg.get("llm_enabled", False)),
        },
    )


def _yfinance_news(symbol: str) -> list[dict[str, Any]]:
    try:
        import yfinance as yf

        news = yf.Ticker(symbol).news or []
    except Exception:
        return []
    items = []
    for entry in news[:5]:
        content = entry.get("content", entry) if isinstance(entry, dict) else {}
        title = content.get("title") or entry.get("title") if isinstance(entry, dict) else ""
        if title:
            items.append({"source": "yfinance_news", "title": title})
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


def _resolve_mind_file(settings: Settings, mind_file: str) -> Path:
    path = Path(mind_file)
    if path.is_absolute():
        return path
    return (settings.path.parent.parent / path).resolve()


def _summarize_context(symbol: str, catalysts: list[str], risks: list[str], mind_file: Path) -> str:
    checklist = "trader checklist unavailable"
    if mind_file.exists():
        checklist = "trader checklist loaded"
    if catalysts or risks:
        parts = [f"{symbol.upper()} context summary ({checklist})."]
        if catalysts:
            parts.append("Catalysts: " + "; ".join(catalysts[:3]))
        if risks:
            parts.append("Risks: " + "; ".join(risks[:3]))
        return " ".join(parts)
    return f"{symbol.upper()} has no strong configured catalyst. {checklist}; default to price, volume, levels, and risk controls."
