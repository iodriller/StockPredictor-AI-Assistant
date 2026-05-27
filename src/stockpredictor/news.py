from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .config import Settings
from .context import fetch_news_items
from .utils import clamp, clean_symbol_list


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


def build_news_feed(symbols: list[str], settings: Settings, limit: int = 50) -> dict[str, Any]:
    cfg = settings.context_agent.get("news_analysis", {})
    max_per_symbol = int(cfg.get("max_headlines_per_symbol", 8))
    clean_symbols = clean_symbol_list(symbols)
    all_items = fetch_news_items(clean_symbols, limit=max(limit, len(clean_symbols) * max_per_symbol))
    enriched = [_enrich_item(item) for item in all_items]
    grouped = {symbol: [item for item in enriched if item.get("symbol") == symbol] for symbol in clean_symbols}
    summaries = []
    for symbol, items in grouped.items():
        clipped = items[:max_per_symbol]
        summaries.append(_summarize_symbol_news(symbol, clipped, settings))
    return {
        "symbols": clean_symbols,
        "headline_count": len(enriched),
        "source_count": len([item for item in enriched if item.get("url")]),
        "summaries": summaries,
        "headlines": enriched[:limit],
        "analysis_provider": _analysis_provider(settings),
    }


def _summarize_symbol_news(symbol: str, items: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    heuristic = _heuristic_symbol_summary(symbol, items)
    llm_summary = _llm_symbol_summary(symbol, items, settings)
    if llm_summary:
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
        }
        for item in items
    ]
    try:
        if provider == "openai":
            api_key = os.environ.get(str(news_cfg.get("api_key_env", "OPENAI_API_KEY")))
            if not api_key:
                return None
            parsed = _call_openai_responses(symbol, payload_items, news_cfg, api_key)
            return _normalize_llm_summary(parsed, "openai")
        if provider in {"localdeploy", "openai_compatible", "local_openai"}:
            parsed = _call_openai_compatible_chat(symbol, payload_items, news_cfg)
            return _normalize_llm_summary(parsed, "localdeploy")
        return None
    except Exception:
        if news_cfg.get("fallback_to_heuristic", True):
            return None
        raise


def _news_llm_instructions() -> str:
    return (
        "You are a day-trading news classifier. Return only valid JSON. "
        "Summarize what matters for an active trader: catalyst, freshness, risk, "
        "tradeability, no-trade flags, and why the headlines may or may not matter. "
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
    return {
        **item,
        "symbol": str(item.get("symbol", "")).upper(),
        "category": category,
        "impact": impact,
        "day_trader_relevance": _relevance_score(title, impact, category),
    }


def _classify_category(title: str) -> str:
    lowered = title.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"


def _relevance_score(title: str, impact: float, category: str) -> float:
    category_boost = 0.25 if category in {"earnings_guidance", "analyst_action", "sec_filing", "macro", "legal_regulatory", "m_and_a"} else 0.05
    title_boost = min(len(title.split()) / 80, 0.15)
    return clamp(abs(impact) + category_boost + title_boost, 0.0, 1.0)


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
