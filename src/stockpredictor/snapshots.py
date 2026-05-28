"""Persist compact records of each analyze run so the dashboard can show
"what changed since I last looked?". Stored as JSONL under
`data/analysis_snapshots.local.jsonl` by default. Local file, gitignored."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .config import Settings
from .contracts import AnalysisResult, AnalysisSnapshot
from .utils import now_in_timezone_iso


DEFAULT_SNAPSHOTS_PATH = Path("data") / "analysis_snapshots.local.jsonl"


def record_snapshot(settings: Settings, result: AnalysisResult, horizon: str | None = None) -> AnalysisSnapshot:
    cfg = settings.raw.get("snapshots", {}) or {}
    if not cfg.get("enabled", True):
        # Build a transient record but don't write it.
        return _build_record(result, horizon)
    record = _build_record(result, horizon)
    path = snapshots_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record_to_dict(record), ensure_ascii=False, sort_keys=True) + "\n")
    return record


def load_snapshots(settings: Settings, symbol: str | None = None, limit: int = 25) -> list[AnalysisSnapshot]:
    path = snapshots_path(settings)
    if not path.exists():
        return []
    matching: list[AnalysisSnapshot] = []
    target = symbol.upper() if symbol else None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if target is not None and str(row.get("symbol", "")).upper() != target:
                continue
            matching.append(_dict_to_record(row))
    return matching[-limit:]


def snapshots_path(settings: Settings) -> Path:
    cfg = settings.raw.get("snapshots", {}) or {}
    configured = cfg.get("path", str(DEFAULT_SNAPSHOTS_PATH))
    path = Path(str(configured))
    if not path.is_absolute():
        path = settings.path.parent.parent / path
    return path.resolve()


def diff_snapshots(current: AnalysisSnapshot, previous: AnalysisSnapshot) -> dict[str, Any]:
    """Compact delta between two snapshots so the UI can display change-since-last-check."""
    return {
        "score_delta": _safe_diff(current.score, previous.score),
        "confidence_delta": _safe_diff(current.confidence, previous.confidence),
        "live_price_delta": _safe_diff(current.live_price, previous.live_price),
        "action_changed": current.action != previous.action,
        "previous_action": previous.action,
        "previous_timestamp": previous.timestamp,
    }


def _build_record(result: AnalysisResult, horizon: str | None) -> AnalysisSnapshot:
    live_price = None
    if result.session is not None and result.session.live_price is not None:
        live_price = float(result.session.live_price)
    target_price = result.risk_plan.targets[0] if result.risk_plan.targets else None
    market_session = result.calendar.market_session if result.calendar is not None else "unknown"
    return AnalysisSnapshot(
        snapshot_id=str(uuid.uuid4()),
        symbol=result.snapshot.symbol,
        timestamp=now_in_timezone_iso("UTC"),
        horizon=horizon or result.horizon or "swing",
        action=result.decision.action,
        score=float(result.decision.score),
        confidence=float(result.decision.confidence),
        live_price=live_price,
        entry=result.risk_plan.entry,
        stop_loss=result.risk_plan.stop_loss,
        target=target_price,
        risk_reward=result.risk_plan.risk_reward,
        market_session=market_session,
        top_reason=result.decision.top_reason,
        no_trade_flags=list(result.risk_plan.no_trade_reasons),
    )


def _record_to_dict(record: AnalysisSnapshot) -> dict[str, Any]:
    return {
        "snapshot_id": record.snapshot_id,
        "symbol": record.symbol,
        "timestamp": record.timestamp,
        "horizon": record.horizon,
        "action": record.action,
        "score": record.score,
        "confidence": record.confidence,
        "live_price": record.live_price,
        "entry": record.entry,
        "stop_loss": record.stop_loss,
        "target": record.target,
        "risk_reward": record.risk_reward,
        "market_session": record.market_session,
        "top_reason": record.top_reason,
        "no_trade_flags": record.no_trade_flags,
    }


def _dict_to_record(row: dict[str, Any]) -> AnalysisSnapshot:
    return AnalysisSnapshot(
        snapshot_id=str(row.get("snapshot_id") or uuid.uuid4()),
        symbol=str(row.get("symbol", "")).upper(),
        timestamp=str(row.get("timestamp", "")),
        horizon=str(row.get("horizon", "swing")),
        action=str(row.get("action", "")),
        score=float(row.get("score", 0.0) or 0.0),
        confidence=float(row.get("confidence", 0.0) or 0.0),
        live_price=_optional_float(row.get("live_price")),
        entry=_optional_float(row.get("entry")),
        stop_loss=_optional_float(row.get("stop_loss")),
        target=_optional_float(row.get("target")),
        risk_reward=_optional_float(row.get("risk_reward")),
        market_session=str(row.get("market_session", "unknown")),
        top_reason=str(row.get("top_reason", "")),
        no_trade_flags=list(row.get("no_trade_flags") or []),
    )


def _safe_diff(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return float(current) - float(previous)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
