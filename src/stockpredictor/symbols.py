from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SYMBOL_UNIVERSE_CACHE = "symbol_universe.json"
SYMBOL_UNIVERSE_TTL_SECONDS = 24 * 60 * 60
LOGGER = logging.getLogger(__name__)

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
    cache_root = Path(cache_dir)
    universe_cache = cache_root / SYMBOL_UNIVERSE_CACHE
    cached_universe = _read_cached_universe(universe_cache)
    if cached_universe is not None and _cache_is_fresh(universe_cache):
        return cached_universe

    symbols: list[dict[str, Any]] = []
    sec_cache = cache_root / "sec_company_tickers.json"
    if sec_cache.exists():
        try:
            symbols.extend(_parse_sec_company_tickers(json.loads(sec_cache.read_text(encoding="utf-8")), source="sec_cache"))
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
            LOGGER.warning("Ignoring invalid cached SEC symbol universe at %s: %s", sec_cache, exc)
    else:
        try:
            import httpx

            response = httpx.get(
                SEC_COMPANY_TICKERS_URL,
                headers={"User-Agent": "StockPredictor research app contact@example.com"},
                timeout=10,
                follow_redirects=True,
            )
            response.raise_for_status()
            cache_root.mkdir(parents=True, exist_ok=True)
            sec_cache.write_text(response.text, encoding="utf-8")
            symbols.extend(_parse_sec_company_tickers(response.json(), source="sec"))
        except Exception as exc:
            LOGGER.warning("SEC symbol universe fetch failed: %s", exc)

    symbols.extend(_load_exchange_symbols(cache_root, "nasdaqlisted.txt", NASDAQ_LISTED_URL, _parse_nasdaq_listed))
    symbols.extend(_load_exchange_symbols(cache_root, "otherlisted.txt", OTHER_LISTED_URL, _parse_other_listed))
    if not symbols and cached_universe is not None:
        return cached_universe
    universe = _dedupe_symbols([*symbols, *FALLBACK_SYMBOLS])
    if universe:
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            universe_cache.write_text(json.dumps(universe, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Could not cache symbol universe at %s: %s", universe_cache, exc)
        return universe
    LOGGER.warning("All symbol universe sources failed; using the small fallback list")
    return FALLBACK_SYMBOLS


def _parse_sec_company_tickers(payload: dict[str, Any], source: str) -> list[dict[str, Any]]:
    symbols = []
    for row in payload.values():
        ticker = str(row.get("ticker", "")).upper().strip()
        name = str(row.get("title", "")).strip()
        if ticker and name:
            symbols.append({"symbol": ticker, "name": name, "source": source})
    return symbols


def _load_exchange_symbols(cache_root: Path, cache_name: str, url: str, parser) -> list[dict[str, Any]]:
    cache_path = cache_root / cache_name
    text = ""
    if cache_path.exists():
        try:
            text = cache_path.read_text(encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Could not read cached symbol list %s: %s", cache_path, exc)
    if not text or not _cache_is_fresh(cache_path):
        try:
            import httpx

            response = httpx.get(url, headers={"User-Agent": "StockPredictor research app"}, timeout=10, follow_redirects=True)
            response.raise_for_status()
            text = response.text
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Exchange symbol list fetch failed for %s: %s", url, exc)
    return parser(text) if text else []


def _parse_nasdaq_listed(text: str) -> list[dict[str, Any]]:
    return _parse_pipe_list(text, symbol_key="Symbol", name_key="Security Name", exchange="NASDAQ")


def _parse_other_listed(text: str) -> list[dict[str, Any]]:
    return _parse_pipe_list(text, symbol_key="ACT Symbol", name_key="Security Name", exchange_key="Exchange")


def _parse_pipe_list(
    text: str,
    symbol_key: str,
    name_key: str,
    exchange: str = "",
    exchange_key: str = "",
) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    headers = lines[0].split("|")
    symbols = []
    for line in lines[1:]:
        values = line.split("|")
        if len(values) != len(headers):
            continue
        row = dict(zip(headers, values))
        ticker = normalize_symbol(str(row.get(symbol_key, "")))
        name = str(row.get(name_key, "")).strip()
        if not ticker or not name or ticker.startswith("FILE CREATION TIME") or str(row.get("Test Issue", "N")).upper() == "Y":
            continue
        symbols.append(
            {
                "symbol": ticker,
                "name": name,
                "source": "nasdaq_trader",
                "exchange": exchange or str(row.get(exchange_key, "")).strip(),
                "asset_type": "etf" if str(row.get("ETF", "N")).upper() == "Y" else "security",
            }
        )
    return symbols


def _read_cached_universe(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        LOGGER.warning("Ignoring invalid cached symbol universe at %s: %s", path, exc)
        return None
    return _dedupe_symbols(payload) if isinstance(payload, list) else None


def _cache_is_fresh(path: Path) -> bool:
    try:
        age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    except OSError:
        return False
    return age_seconds < SYMBOL_UNIVERSE_TTL_SECONDS


def _dedupe_symbols(symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in symbols:
        if not isinstance(item, dict):
            continue
        symbol = normalize_symbol(str(item.get("symbol", "")))
        name = str(item.get("name", "")).strip()
        if symbol and name:
            deduped.setdefault(symbol, {**item, "symbol": symbol, "name": name})
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
