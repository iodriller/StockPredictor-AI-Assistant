from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

FALLBACK_SYMBOLS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "source": "fallback"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "source": "fallback"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "source": "fallback"},
    {"symbol": "TSLA", "name": "Tesla, Inc.", "source": "fallback"},
    {"symbol": "AMD", "name": "Advanced Micro Devices, Inc.", "source": "fallback"},
    {"symbol": "META", "name": "Meta Platforms, Inc.", "source": "fallback"},
    {"symbol": "AMZN", "name": "Amazon.com, Inc.", "source": "fallback"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "source": "fallback"},
    {"symbol": "GOOG", "name": "Alphabet Inc.", "source": "fallback"},
    {"symbol": "NFLX", "name": "Netflix, Inc.", "source": "fallback"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "source": "fallback"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "source": "fallback"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "source": "fallback"},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "source": "fallback"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "source": "fallback"},
    {"symbol": "SOFI", "name": "SoFi Technologies, Inc.", "source": "fallback"},
    {"symbol": "PLTR", "name": "Palantir Technologies Inc.", "source": "fallback"},
    {"symbol": "COIN", "name": "Coinbase Global, Inc.", "source": "fallback"},
    {"symbol": "RIVN", "name": "Rivian Automotive, Inc.", "source": "fallback"},
    {"symbol": "SMCI", "name": "Super Micro Computer, Inc.", "source": "fallback"},
    {"symbol": "MARA", "name": "MARA Holdings, Inc.", "source": "fallback"},
    {"symbol": "RIOT", "name": "Riot Platforms, Inc.", "source": "fallback"},
]


def search_symbols(query: str, limit: int = 25, cache_dir: str | Path = ".cache") -> list[dict[str, Any]]:
    query = normalize_symbol(query)
    if not query:
        return []
    universe = load_symbol_universe(cache_dir=cache_dir)
    scored = []
    for item in universe:
        symbol = str(item["symbol"]).upper()
        name = str(item["name"]).upper()
        score = _score_match(query, symbol, name)
        if score > 0:
            scored.append((score, item))
    if not scored and _looks_like_ticker(query):
        scored.append((10, {"symbol": query, "name": f"Direct ticker: {query}", "source": "direct"}))
    scored.sort(key=lambda row: (-row[0], row[1]["symbol"]))
    results = [item for _, item in scored[:limit]]
    return results[:limit]


def load_symbol_universe(cache_dir: str | Path = ".cache") -> list[dict[str, Any]]:
    cache_path = Path(cache_dir) / "sec_company_tickers.json"
    if cache_path.exists():
        try:
            return _parse_sec_company_tickers(json.loads(cache_path.read_text(encoding="utf-8")), source="sec_cache")
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            pass
    try:
        import httpx

        response = httpx.get(
            SEC_COMPANY_TICKERS_URL,
            headers={"User-Agent": "StockPredictor research app contact@example.com"},
            timeout=10,
            follow_redirects=True,
        )
        response.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(response.text, encoding="utf-8")
        return _parse_sec_company_tickers(response.json(), source="sec")
    except Exception:
        return FALLBACK_SYMBOLS


def _parse_sec_company_tickers(payload: dict[str, Any], source: str) -> list[dict[str, Any]]:
    symbols = []
    for row in payload.values():
        ticker = str(row.get("ticker", "")).upper().strip()
        name = str(row.get("title", "")).strip()
        if ticker and name:
            symbols.append({"symbol": ticker, "name": name, "source": source})
    symbols.extend(FALLBACK_SYMBOLS)
    deduped: dict[str, dict[str, Any]] = {}
    for item in symbols:
        deduped.setdefault(str(item["symbol"]), item)
    return list(deduped.values())


def _score_match(query: str, symbol: str, name: str) -> int:
    if symbol == query:
        return 100
    if symbol.startswith(query):
        return 80
    if name.startswith(query):
        return 60
    if query in symbol:
        return 40
    if query in name:
        return 20
    return 0


def normalize_symbol(value: str) -> str:
    return value.strip().upper().replace(".", "-").replace("/", "-")


def _looks_like_ticker(value: str) -> bool:
    if not value or len(value) > 8:
        return False
    return all(char.isalnum() or char == "-" for char in value)
