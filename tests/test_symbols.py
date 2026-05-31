from __future__ import annotations

import json
from pathlib import Path

from stockpredictor.symbols import search_symbols


def test_symbol_search_uses_cached_universe(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    payload = {
        "0": {"ticker": "PLTR", "title": "Palantir Technologies Inc."},
        "1": {"ticker": "SHOP", "title": "Shopify Inc."},
    }
    (cache_dir / "sec_company_tickers.json").write_text(json.dumps(payload), encoding="utf-8")

    results = search_symbols("palantir", cache_dir=cache_dir)

    assert results[0]["symbol"] == "PLTR"
    assert "Palantir" in results[0]["name"]


def test_symbol_search_only_offers_ticker_shaped_direct_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    assert search_symbols("ABCD", cache_dir=tmp_path)[0]["source"] == "direct"
    assert search_symbols("not a ticker", cache_dir=tmp_path) == []
