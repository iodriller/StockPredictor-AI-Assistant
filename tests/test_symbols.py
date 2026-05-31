from __future__ import annotations

import json
from pathlib import Path

from stockpredictor.symbols import _parse_nasdaq_listed, _parse_other_listed, search_symbols


def test_symbol_search_uses_cached_universe(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    payload = {
        "0": {"ticker": "PLTR", "title": "Palantir Technologies Inc."},
        "1": {"ticker": "SHOP", "title": "Shopify Inc."},
    }
    (cache_dir / "sec_company_tickers.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    results = search_symbols("palantir", cache_dir=cache_dir)

    assert results[0]["symbol"] == "PLTR"
    assert "Palantir" in results[0]["name"]


def test_exchange_symbol_lists_include_etfs_and_other_listed_securities() -> None:
    nasdaq = _parse_nasdaq_listed(
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "QQQ|Invesco QQQ Trust|G|N|N|100|Y|N\n"
    )
    other = _parse_other_listed(
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "VTI|Vanguard Total Stock Market ETF|P|VTI|Y|100|N|VTI\n"
    )

    assert nasdaq[0]["symbol"] == "QQQ"
    assert nasdaq[0]["asset_type"] == "etf"
    assert other[0]["symbol"] == "VTI"
    assert other[0]["exchange"] == "P"


def test_symbol_search_only_offers_ticker_shaped_direct_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    assert search_symbols("ABCD", cache_dir=tmp_path)[0]["source"] == "direct"
    assert search_symbols("not a ticker", cache_dir=tmp_path) == []
