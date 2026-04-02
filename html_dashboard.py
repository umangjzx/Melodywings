"""
html_dashboard.py — Flask HTML Safety Dashboard
MelodyWings Guard | Real-Time Content Safety System

Run with:
    python html_dashboard.py

Then open:
    http://localhost:8502
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, url_for

from database import get_db

app = Flask(__name__, template_folder="templates", static_folder="static")

MODULE_PAGES: dict[str, dict[str, Any]] = {
    "overview": {
        "slug": "overview",
        "title": "Operational Safety Console",
        "subtitle": "Unified safety telemetry across chat, video, and audio analyzers",
        "description": "Monitor all signals in one place and drill down into risky events quickly.",
        "sources": [],
        "accent": "overview",
    },
    "chat": {
        "slug": "chat",
        "title": "Chat Analysis",
        "subtitle": "Toxicity, profanity, PII, and sentiment monitoring",
        "description": "Focused monitoring for text and transcript-driven risk patterns.",
        "sources": ["chat", "transcript"],
        "accent": "chat",
    },
    "video": {
        "slug": "video",
        "title": "Video Analysis",
        "subtitle": "Frame-level NSFW and emotion safety intelligence",
        "description": "Track visual content risk over time with frame-level confidence.",
        "sources": ["video_frame"],
        "accent": "video",
    },
    "audio": {
        "slug": "audio",
        "title": "Audio Analysis",
        "subtitle": "Acoustic behavior, silence, and noise pattern monitoring",
        "description": "Inspect loudness spikes, silence gaps, and other voice risk markers.",
        "sources": ["audio"],
        "accent": "audio",
    },
}

TOXICITY_THRESHOLD = 0.75
STRONG_NEGATIVE_THRESHOLD = 0.98
NSFW_THRESHOLD = 0.70
FLAGGED_VIDEO_EMOTIONS = {"disgust", "angry"}

DEFAULT_LIMIT = 1000
MAX_LIMIT = 10000
CACHE_TTL_SECONDS = 2.0

_cache_lock = threading.Lock()
_cached_rows: list[dict[str, Any]] = []
_cached_options: dict[str, list[str]] = {}
_cache_expiry = 0.0


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Safely parse float values."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely parse integer values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_timestamp(timestamp: str | None) -> datetime | None:
    """Parse ISO timestamp and normalize to UTC."""
    if not timestamp:
        return None
    try:
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _parse_date(value: str | None) -> datetime | None:
    """Parse date input from query string."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_audio_reasons(message: str) -> list[str]:
    """Extract comma-separated audio reason tokens from message text."""
    if not message:
        return []
    return [part.strip() for part in message.split(",") if part.strip()]


def _derive_reason_tags(row: dict[str, Any]) -> list[str]:
    """Build standardized reason tags from joined detail fields."""
    source = row.get("source", "")
    reasons: list[str] = []

    if source in ("chat", "transcript"):
        if row.get("has_profanity"):
            reasons.append("profanity")

        pii_types = row.get("pii_types") or []
        if isinstance(pii_types, list):
            for pii_type in pii_types:
                if pii_type:
                    reasons.append(f"pii:{pii_type}")

        toxicity_score = _safe_float(row.get("toxicity_score"))
        if toxicity_score is not None and toxicity_score >= TOXICITY_THRESHOLD:
            reasons.append("toxicity:toxic")

        sentiment = (row.get("sentiment") or "").lower()
        sentiment_score = _safe_float(row.get("sentiment_score"))
        if (
            sentiment == "negative"
            and sentiment_score is not None
            and sentiment_score > STRONG_NEGATIVE_THRESHOLD
        ):
            reasons.append("strong_negative_sentiment")

    elif source == "video_frame":
        nsfw_label = (row.get("nsfw_label") or "").lower()
        nsfw_score = _safe_float(row.get("nsfw_score"))
        if nsfw_label == "nsfw" and nsfw_score is not None and nsfw_score >= NSFW_THRESHOLD:
            reasons.append("nsfw:nsfw")

        emotion = (row.get("emotion") or "").lower()
        if emotion in FLAGGED_VIDEO_EMOTIONS:
            reasons.append(f"emotion:{emotion}")

    elif source == "audio":
        reasons.extend(_parse_audio_reasons(str(row.get("message") or "")))

    if row.get("flagged") and not reasons:
        reasons.append("flagged_unspecified")

    return sorted(dict.fromkeys(reasons))


def _content_text(row: dict[str, Any]) -> str:
    """Return the best available message text for UI display."""
    for candidate in (
        row.get("chat_text"),
        row.get("transcript_segment_text"),
        row.get("message"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _normalize_entities(value: Any) -> list[Any]:
    """Ensure entities are always a list."""
    if isinstance(value, list):
        return value
    return []


def _normalize_list(value: Any) -> list[str]:
    """Ensure pii types and similar values are always string lists."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _get_args_list(args: Any, key: str) -> list[str]:
    """Read repeated query values from Flask MultiDict or plain dict."""
    if hasattr(args, "getlist"):
        values = args.getlist(key)
    else:
        value = args.get(key)
        if isinstance(value, list):
            values = value
        elif value is None or value == "":
            values = []
        else:
            values = [value]
    return [str(v) for v in values if str(v).strip()]


def _primary_reason_key(reason_tags: list[str]) -> str | None:
    """Map reason tag into confidence_by_reason key."""
    if not reason_tags:
        return None

    first = str(reason_tags[0]).lower()
    if first.startswith("toxicity:"):
        return "toxicity"
    if first.startswith("pii:"):
        return "pii"
    if first == "profanity":
        return "profanity"
    if "sentiment" in first:
        return "sentiment"
    if first.startswith("emotion:"):
        return "emotion"
    if first.startswith("nsfw:"):
        return "nsfw"
    return None


def _display_confidence(raw_confidence: Any, confidence_by_reason: Any, reason_tags: list[str]) -> float:
    """Derive displayed confidence based on primary reason."""
    if isinstance(confidence_by_reason, dict):
        reason_key = _primary_reason_key(reason_tags)
        if reason_key and reason_key in confidence_by_reason:
            return round(_safe_float(confidence_by_reason.get(reason_key), 0.0) or 0.0, 4)

    return round(_safe_float(raw_confidence, 0.0) or 0.0, 4)


def _normalize_row(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one joined DB row and derive dashboard fields."""
    ts_dt = _parse_timestamp(record.get("timestamp"))

    reason_tags = _derive_reason_tags(record)
    content_text = _content_text(record)
    preview = content_text[:140] + ("..." if len(content_text) > 140 else "")
    confidence_by_reason = record.get("confidence_by_reason")
    if not isinstance(confidence_by_reason, dict):
        confidence_by_reason = {}

    data_blob = record.get("data")
    health_status: dict[str, Any] = {}
    if isinstance(data_blob, dict):
        maybe_health = data_blob.get("health_status")
        if isinstance(maybe_health, dict):
            health_status = maybe_health

    normalized = {
        "id": _safe_int(record.get("id")),
        "run_id": str(record.get("run_id") or ""),
        "source": str(record.get("source") or "unknown"),
        "timestamp": str(record.get("timestamp") or ""),
        "timestamp_display": ts_dt.strftime("%Y-%m-%d %H:%M:%S UTC") if ts_dt else str(record.get("timestamp") or ""),
        "flagged": bool(record.get("flagged")),
        "severity": str(record.get("severity") or "low"),
        "category": str(record.get("category") or ""),
        "confidence": _display_confidence(record.get("confidence"), confidence_by_reason, reason_tags),
        "confidence_by_reason": confidence_by_reason,
        "reason_tags": reason_tags,
        "reason_text": ", ".join(reason_tags) if reason_tags else "safe",
        "content_text": content_text,
        "preview": preview,
        "health_status": health_status,
        "sentiment": str(record.get("sentiment") or "").lower(),
        "sentiment_score": round(_safe_float(record.get("sentiment_score"), 0.0) or 0.0, 4),
        "emotion": str(record.get("emotion") or "none").lower(),
        "frame_number": _safe_int(record.get("frame_number"), 0),
        "timestamp_sec": round(_safe_float(record.get("timestamp_sec"), 0.0) or 0.0, 3),
        "nsfw_label": str(record.get("nsfw_label") or ""),
        "nsfw_score": round(_safe_float(record.get("nsfw_score"), 0.0) or 0.0, 4),
        "has_profanity": bool(record.get("has_profanity")) if record.get("has_profanity") is not None else False,
        "has_pii": bool(record.get("has_pii")) if record.get("has_pii") is not None else False,
        "pii_types": _normalize_list(record.get("pii_types")),
        "entities": _normalize_entities(record.get("entities")),
        "max_volume_db": _safe_float(record.get("max_volume_db")),
        "mean_volume_db": _safe_float(record.get("mean_volume_db")),
        "silence_count": _safe_int(record.get("silence_count"), 0),
        "speech_rate_wpm": _safe_float(record.get("speech_rate_wpm")),
        "background_noise_db": _safe_float(record.get("background_noise_db")),
        "speaker_count": _safe_int(record.get("speaker_count"), 0),
        "transcript_segment_text": str(record.get("transcript_segment_text") or ""),
        "segment_confidence": _safe_float(record.get("segment_confidence")),
        "segment_start_time": _safe_float(record.get("segment_start_time")),
        "segment_end_time": _safe_float(record.get("segment_end_time")),
        "_timestamp_dt": ts_dt,
    }

    seg_start = normalized.get("segment_start_time")
    seg_end = normalized.get("segment_end_time")
    if isinstance(seg_start, float) and isinstance(seg_end, float) and seg_end >= seg_start:
        normalized["segment_duration_sec"] = round(seg_end - seg_start, 3)
    else:
        normalized["segment_duration_sec"] = None

    normalized["_search_blob"] = " ".join(
        [
            normalized["run_id"],
            normalized["source"],
            normalized["severity"],
            normalized["category"],
            normalized["reason_text"],
            normalized["content_text"],
        ]
    ).lower()

    return normalized


def _build_enriched_rows() -> list[dict[str, Any]]:
    """Load joined records and compute derived dashboard fields."""
    db = get_db()
    raw_rows = db.get_alerts_detailed()

    rows = [_normalize_row(dict(row)) for row in raw_rows]

    rows.sort(
        key=lambda item: item.get("_timestamp_dt") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return rows


def _get_cached_rows_and_options() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Serve enriched rows/options from short-lived in-memory cache."""
    global _cached_rows, _cached_options, _cache_expiry

    now = time.monotonic()
    with _cache_lock:
        if _cached_rows and now < _cache_expiry:
            return _cached_rows, _cached_options

    rows = _build_enriched_rows()
    options = _collect_filter_options(rows)

    with _cache_lock:
        _cached_rows = rows
        _cached_options = options
        _cache_expiry = now + CACHE_TTL_SECONDS

    return rows, options


def _collect_filter_options(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Collect unique filter option values."""
    reason_set = set()
    run_ids: list[str] = []
    seen_run_ids: set[str] = set()

    for row in rows:
        reason_set.update(row.get("reason_tags", []))
        run_id = str(row.get("run_id") or "")
        if run_id and run_id not in seen_run_ids:
            seen_run_ids.add(run_id)
            run_ids.append(run_id)

    return {
        "run_ids": run_ids,
        "sources": sorted({row.get("source", "") for row in rows if row.get("source")}),
        "severities": sorted({row.get("severity", "") for row in rows if row.get("severity")}),
        "categories": sorted({row.get("category", "") for row in rows if row.get("category")}),
        "reasons": sorted(reason_set),
        "sentiments": sorted({row.get("sentiment", "") for row in rows if row.get("sentiment")}),
        "emotions": sorted({row.get("emotion", "") for row in rows if row.get("emotion") and row.get("emotion") != "none"}),
    }


def _build_source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute per-source totals and flagged rates for source health strip."""
    totals: Counter[str] = Counter()
    flagged_totals: Counter[str] = Counter()

    for row in rows:
        source = str(row.get("source") or "unknown")
        totals[source] += 1
        if bool(row.get("flagged")):
            flagged_totals[source] += 1

    ordered_sources = sorted(totals.keys(), key=lambda src: totals[src], reverse=True)
    summary: list[dict[str, Any]] = []
    for source in ordered_sources:
        total = int(totals[source])
        flagged = int(flagged_totals[source])
        flag_rate = round((flagged / total * 100.0) if total else 0.0, 2)
        summary.append(
            {
                "source": source,
                "total": total,
                "flagged": flagged,
                "safe": max(0, total - flagged),
                "flag_rate": flag_rate,
            }
        )
    return summary


def _filter_rows(rows: list[dict[str, Any]], args: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply query-string filters to enriched rows."""
    run_ids = set(_get_args_list(args, "run_id"))
    sources = set(_get_args_list(args, "source"))
    severities = set(_get_args_list(args, "severity"))
    categories = set(_get_args_list(args, "category"))
    reasons = set(_get_args_list(args, "reason"))
    sentiments = set(_get_args_list(args, "sentiment"))
    emotions = set(_get_args_list(args, "emotion"))

    status = str(args.get("status") or "all").lower()
    search = str(args.get("search") or "").strip().lower()

    confidence_min = _safe_float(args.get("confidence_min"), 0.0) or 0.0
    confidence_max = _safe_float(args.get("confidence_max"), 1.0) or 1.0
    if confidence_max < confidence_min:
        confidence_min, confidence_max = confidence_max, confidence_min

    start_date = _parse_date(args.get("start_date"))
    end_date = _parse_date(args.get("end_date"))
    end_date_exclusive = end_date + timedelta(days=1) if end_date else None

    filtered: list[dict[str, Any]] = []
    for row in rows:
        if run_ids and row.get("run_id") not in run_ids:
            continue

        if sources and row.get("source") not in sources:
            continue

        flagged = bool(row.get("flagged"))
        if status == "flagged" and not flagged:
            continue
        if status == "safe" and flagged:
            continue

        if severities and row.get("severity") not in severities:
            continue
        if categories and row.get("category") not in categories:
            continue

        confidence = _safe_float(row.get("confidence"), 0.0) or 0.0
        if confidence < confidence_min or confidence > confidence_max:
            continue

        row_reasons = set(row.get("reason_tags") or [])
        if reasons and not row_reasons.intersection(reasons):
            continue

        if sentiments and row.get("sentiment") not in sentiments:
            continue
        if emotions and row.get("emotion") not in emotions:
            continue

        ts_dt = row.get("_timestamp_dt")
        if start_date and ts_dt and ts_dt < start_date:
            continue
        if end_date_exclusive and ts_dt and ts_dt >= end_date_exclusive:
            continue

        if search and search not in row.get("_search_blob", ""):
            continue

        filtered.append(row)

    limit = _safe_int(args.get("limit"), DEFAULT_LIMIT)
    limit = max(10, min(limit, MAX_LIMIT))
    return filtered[:limit]


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Compute top-level KPI metrics."""
    total = len(rows)
    flagged_rows = [row for row in rows if row.get("flagged")]
    flagged = len(flagged_rows)
    safe = total - flagged
    high_critical = sum(1 for row in rows if row.get("severity") in ("high", "critical"))

    avg_conf_flagged = 0.0
    if flagged_rows:
        avg_conf_flagged = sum(_safe_float(row.get("confidence"), 0.0) or 0.0 for row in flagged_rows) / flagged

    return {
        "total": total,
        "flagged": flagged,
        "safe": safe,
        "flag_rate": round((flagged / total * 100.0) if total else 0.0, 2),
        "high_critical": high_critical,
        "avg_confidence_flagged": round(avg_conf_flagged, 4),
    }


def _build_chart_data(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build chart payloads for frontend visualizations."""
    source_counts = Counter(row.get("source", "unknown") for row in rows)
    severity_counts = Counter(row.get("severity", "low") for row in rows)

    reason_counts = Counter()
    for row in rows:
        tags = row.get("reason_tags") or []
        if tags:
            reason_counts.update(tags)
        else:
            reason_counts.update(["safe"])

    trend_map: dict[str, dict[str, int]] = defaultdict(lambda: {"flagged": 0, "safe": 0})
    for row in rows:
        ts_dt = row.get("_timestamp_dt")
        if not ts_dt:
            continue
        key = ts_dt.replace(second=0, microsecond=0).isoformat()
        if row.get("flagged"):
            trend_map[key]["flagged"] += 1
        else:
            trend_map[key]["safe"] += 1

    trend_points = [
        {
            "minute": minute,
            "flagged": counts["flagged"],
            "safe": counts["safe"],
        }
        for minute, counts in sorted(trend_map.items())
    ]

    return {
        "sources": [
            {"label": label, "value": value}
            for label, value in sorted(source_counts.items(), key=lambda item: item[1], reverse=True)
        ],
        "severities": [
            {"label": label, "value": value}
            for label, value in sorted(severity_counts.items(), key=lambda item: item[1], reverse=True)
        ],
        "reasons": [
            {"label": label, "value": value}
            for label, value in reason_counts.most_common(12)
        ],
        "trend": trend_points,
    }


def _sanitize_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove private helper keys before returning API response."""
    return [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in rows
    ]


def _get_page_context(page_slug: str) -> dict[str, Any]:
    """Return safe page context for dashboard rendering."""
    if page_slug not in MODULE_PAGES:
        page_slug = "overview"

    page_context = dict(MODULE_PAGES[page_slug])
    page_context["source_label"] = ", ".join(page_context.get("sources") or []) or "all"
    return page_context


@app.get("/")
def index() -> Any:
    """Redirect root path to dashboard overview page."""
    return redirect(url_for("dashboard_page", page_slug="overview"))


@app.get("/dashboard")
def dashboard_root() -> Any:
    """Redirect short dashboard path to overview page."""
    return redirect(url_for("dashboard_page", page_slug="overview"))


@app.get("/dashboard/<page_slug>")
def dashboard_page(page_slug: str) -> str:
    """Serve module-specific HTML dashboard page."""
    page_context = _get_page_context(page_slug)
    nav_pages = [MODULE_PAGES[key] for key in ("overview", "chat", "video", "audio")]
    return render_template(
        "dashboard.html",
        page_context=page_context,
        nav_pages=nav_pages,
    )


@app.get("/api/dashboard-data")
def dashboard_data() -> Any:
    """Return filtered dashboard data with metrics, options, and chart payloads."""
    rows, options = _get_cached_rows_and_options()
    filtered = _filter_rows(rows, request.args)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(rows),
        "filtered_records": len(filtered),
        "metrics": _compute_metrics(filtered),
        "source_summary": _build_source_summary(filtered),
        "options": options,
        "chart_data": _build_chart_data(filtered),
        "records": _sanitize_records(filtered),
    }
    return jsonify(payload)


@app.get("/api/alerts/<int:alert_id>")
def get_alert(alert_id: int) -> Any:
    """Return a single alert record by id."""
    db = get_db()
    raw_row = db.get_alert_detailed_by_id(alert_id)
    match = _normalize_row(raw_row) if raw_row is not None else None
    if match is None:
        return jsonify({"error": "Alert not found"}), 404
    return jsonify({"record": _sanitize_records([match])[0]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502, debug=False)
