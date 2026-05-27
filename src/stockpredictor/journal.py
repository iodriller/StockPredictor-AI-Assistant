from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .config import Settings
from .utils import now_in_timezone_iso


DEFAULT_JOURNAL_PATH = Path("data") / "trade_journal.local.jsonl"


def append_journal_entry(settings: Settings, entry: dict[str, Any]) -> dict[str, Any]:
    record = _normalize_entry(entry, tz_name=str(settings.app.get("timezone", "UTC")))
    path = journal_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def load_journal_entries(settings: Settings, limit: int = 100) -> list[dict[str, Any]]:
    rows = _read_all(settings)
    return rows[-limit:]


def update_journal_entry(settings: Settings, entry_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    rows = _read_all(settings)
    tz_name = str(settings.app.get("timezone", "UTC"))
    for row in rows:
        if row.get("id") == entry_id:
            merged = {**row, **updates}
            normalized = _normalize_entry(merged, existing_id=entry_id, existing_timestamp=row.get("timestamp"), tz_name=tz_name)
            row.clear()
            row.update(normalized)
            _write_all(settings, rows)
            return normalized
    return None


def delete_journal_entry(settings: Settings, entry_id: str) -> bool:
    rows = _read_all(settings)
    filtered = [row for row in rows if row.get("id") != entry_id]
    if len(filtered) == len(rows):
        return False
    _write_all(settings, filtered)
    return True


def journal_path(settings: Settings) -> Path:
    configured = settings.raw.get("journal", {}).get("path", str(DEFAULT_JOURNAL_PATH))
    path = Path(str(configured))
    if not path.is_absolute():
        path = settings.path.parent.parent / path
    return path.resolve()


def _read_all(settings: Settings) -> list[dict[str, Any]]:
    path = journal_path(settings)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_all(settings: Settings, rows: list[dict[str, Any]]) -> None:
    path = journal_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _normalize_entry(
    entry: dict[str, Any],
    existing_id: str | None = None,
    existing_timestamp: str | None = None,
    tz_name: str = "UTC",
) -> dict[str, Any]:
    symbol = str(entry.get("symbol", "")).strip().upper()
    if not symbol:
        raise ValueError("Journal entry requires a symbol.")
    timestamp = str(entry.get("timestamp") or existing_timestamp or now_in_timezone_iso(tz_name))
    return {
        "id": existing_id or str(entry.get("id") or uuid.uuid4()),
        "timestamp": timestamp,
        "symbol": symbol,
        "action": str(entry.get("action", "watch")),
        "setup_type": str(entry.get("setup_type", "unclassified")),
        "followed_plan": bool(entry.get("followed_plan", False)),
        "emotional_state": str(entry.get("emotional_state", "neutral")),
        "entry_quality": int(entry.get("entry_quality", 3)),
        "exit_quality": int(entry.get("exit_quality", 3)),
        "risk_respected": bool(entry.get("risk_respected", False)),
        "outcome": str(entry.get("outcome", "open")),
        "notes": str(entry.get("notes", "")),
        "decision_score": _optional_float(entry.get("decision_score")),
        "confidence": _optional_float(entry.get("confidence")),
        "risk_reward": _optional_float(entry.get("risk_reward")),
    }


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)
