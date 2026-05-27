from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings


DEFAULT_JOURNAL_PATH = Path("data") / "trade_journal.local.jsonl"


def append_journal_entry(settings: Settings, entry: dict[str, Any]) -> dict[str, Any]:
    record = _normalize_entry(entry)
    path = journal_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def load_journal_entries(settings: Settings, limit: int = 100) -> list[dict[str, Any]]:
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
    return rows[-limit:]


def journal_path(settings: Settings) -> Path:
    configured = settings.raw.get("journal", {}).get("path", str(DEFAULT_JOURNAL_PATH))
    path = Path(str(configured))
    if not path.is_absolute():
        path = settings.path.parent.parent / path
    return path.resolve()


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    symbol = str(entry.get("symbol", "")).strip().upper()
    if not symbol:
        raise ValueError("Journal entry requires a symbol.")
    timestamp = str(entry.get("timestamp") or datetime.now(timezone.utc).isoformat())
    return {
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
