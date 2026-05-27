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
