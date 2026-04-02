"""
alert_engine.py — Part 3: Unified Alert Logger
MelodyWings Guard | Real-Time Content Safety System

Accepts structured alert dicts from chat_analyzer and video_analyzer,
tags them with a "source" field, persists them to SQLite database,
and exposes retrieval and statistics functions.
"""

import logging
import os
from typing import Any, Literal, Optional
from database import get_db

logger = logging.getLogger(__name__)

# Alert source types
SourceType = Literal["chat", "video_frame", "transcript", "audio"]


def _read_env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read integer env var safely with lower-bound clamping."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


_DB_MESSAGE_MAX_CHARS = _read_env_int("MWG_DB_MESSAGE_MAX_CHARS", 0, minimum=0)
_ALERT_PREVIEW_CHARS = _read_env_int("MWG_ALERT_PREVIEW_CHARS", 0, minimum=0)


def _message_for_storage(message: str) -> str:
    """Preserve full message by default; optional cap via env var."""
    if _DB_MESSAGE_MAX_CHARS > 0 and len(message) > _DB_MESSAGE_MAX_CHARS:
        return message[:_DB_MESSAGE_MAX_CHARS]
    return message


def _message_for_console(message: str) -> str:
    """Format message for console output with optional preview cap."""
    normalized = " ".join(str(message or "").split())
    if _ALERT_PREVIEW_CHARS > 0 and len(normalized) > _ALERT_PREVIEW_CHARS:
        return normalized[:_ALERT_PREVIEW_CHARS].rstrip() + "..."
    return normalized


def _build_confidence_by_reason(alert: dict) -> dict[str, float]:
    """Build confidence map keyed by reason type for dashboard and storage."""
    reasons = [str(r) for r in (alert.get("reasons") or [])]
    confidence_map: dict[str, float] = {}

    toxicity_score = float(alert.get("confidence") or 0.0)
    if any(reason.startswith("toxicity:") for reason in reasons):
        confidence_map["toxicity"] = toxicity_score

    if any(reason.startswith("pii:") for reason in reasons):
        confidence_map["pii"] = 1.0

    if any(reason == "profanity" for reason in reasons):
        confidence_map["profanity"] = 1.0

    sentiment_score = float(alert.get("sentiment_score") or 0.0)
    if any("sentiment" in reason for reason in reasons):
        confidence_map["sentiment"] = sentiment_score or 1.0

    if any(reason.startswith("emotion:") for reason in reasons):
        confidence_map["emotion"] = 1.0

    if any(reason.startswith("nsfw:") for reason in reasons):
        confidence_map["nsfw"] = float(alert.get("nsfw_score") or 0.0)

    return confidence_map


def _normalize_entities_for_storage(entities: list[Any] | None) -> list[dict[str, str]]:
    """Normalize entity payload to [{'text': ..., 'label': ...}] shape."""
    normalized: list[dict[str, str]] = []
    for item in entities or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            normalized.append(
                {
                    "text": text,
                    "label": str(item.get("label") or "ENTITY"),
                }
            )
            continue

        text = str(item).strip()
        if text:
            normalized.append({"text": text, "label": "ENTITY"})

    return normalized


def _build_alert_data(alert: dict) -> dict[str, Any]:
    """Build optional metadata blob persisted with generic alert row."""
    data: dict[str, Any] = {}
    health_status = alert.get("health_status")
    if isinstance(health_status, dict) and health_status:
        data["health_status"] = health_status
    return data


# ─────────────────────────────────────────────
# ALERT LOGGING
# ─────────────────────────────────────────────

def log_chat_alert(
    alert: dict,
    print_summary: bool = True,
    commit: bool = True,
    run_id: Optional[str] = None,
) -> Optional[int]:
    """
    Logs a chat analysis alert to the database.

    Args:
        alert: Alert dict from chat_analyzer.analyze_message()
        print_summary: If True, prints alert to stdout

    Returns:
        The inserted alert ID, or None on failure.
    """
    db = get_db()
    try:
        message = alert.get("message", "")
        flagged = alert.get("flagged", False)
        confidence = alert.get("confidence")
        reason_list = alert.get("reasons", [])
        reasons = ", ".join(reason_list)
        severity = _determine_severity(reason_list)
        confidence_by_reason = _build_confidence_by_reason(alert)

        # Insert main alert
        alert_id = db.insert_alert(
            run_id=run_id,
            source="chat",
            message=_message_for_storage(message),
            flagged=flagged,
            severity=severity,
            category="chat_safety",
            confidence=confidence or 0.0,
            confidence_by_reason=confidence_by_reason,
            reasons=reason_list,
            data=_build_alert_data(alert),
            commit=False,
        )

        # Insert chat-specific details
        db.insert_chat_alert(
            run_id=run_id,
            alert_id=alert_id,
            text=message,
            has_profanity="profanity" in reason_list,
            has_pii=any("pii:" in r for r in reason_list),
            pii_types=[r.split(":")[1] for r in reason_list if r.startswith("pii:")],
            toxicity_score=confidence or 0.0,
            toxicity_label="toxic" if ("toxicity:" in reasons) else "non_toxic",
            sentiment=alert.get("sentiment") or "",
            sentiment_score=float(alert.get("sentiment_score") or 0.0),
            entities=_normalize_entities_for_storage(alert.get("entities")),
            commit=commit,
        )

        if print_summary:
            _print_chat_alert(alert)

        return alert_id
    except Exception as exc:
        logger.error(f"Failed to log chat alert: {exc}")
        return None


def log_video_alert(
    alert: dict,
    print_summary: bool = True,
    commit: bool = True,
    run_id: Optional[str] = None,
) -> Optional[int]:
    """
    Logs a video frame analysis alert to the database.

    Args:
        alert: Alert dict from video_analyzer.analyze_frame()
        print_summary: If True, prints alert to stdout

    Returns:
        The inserted alert ID, or None on failure.
    """
    db = get_db()
    try:
        frame_number = alert.get("frame_number", 0)
        timestamp_sec = alert.get("timestamp_sec", 0.0)
        nsfw_label = alert.get("nsfw_label", "unknown")
        nsfw_score = alert.get("nsfw_score", 0.0)
        emotion = alert.get("emotion", "none")
        flagged = alert.get("flagged", False)
        reasons = ", ".join(alert.get("reasons", []))

        alert_id = db.insert_alert(
            run_id=run_id,
            source="video_frame",
            message=f"Frame {frame_number} @ {timestamp_sec:.1f}s",
            flagged=flagged,
            severity=_determine_severity(alert.get("reasons", [])),
            category="video_safety",
            confidence=nsfw_score,
            confidence_by_reason=_build_confidence_by_reason(alert),
            reasons=alert.get("reasons", []),
            data=_build_alert_data(alert),
            commit=False,
        )

        db.insert_video_alert(
            run_id=run_id,
            alert_id=alert_id,
            frame_number=frame_number,
            timestamp_sec=timestamp_sec,
            nsfw_label=nsfw_label,
            nsfw_score=nsfw_score,
            emotion=emotion,
            commit=commit,
        )

        if print_summary:
            _print_video_alert(alert)

        return alert_id
    except Exception as exc:
        logger.error(f"Failed to log video alert: {exc}")
        return None


def log_audio_alert(
    alert: dict,
    print_summary: bool = True,
    commit: bool = True,
    run_id: Optional[str] = None,
) -> Optional[int]:
    """
    Logs an audio analysis alert to the database.

    Args:
        alert: Alert dict from audio_analyzer.analyze_audio_features()
        print_summary: If True, prints alert to stdout

    Returns:
        The inserted alert ID, or None on failure.
    """
    db = get_db()
    try:
        flagged = alert.get("flagged", False)
        reasons = alert.get("reasons", [])
        severity = _determine_severity(reasons)

        alert_id = db.insert_alert(
            run_id=run_id,
            source="audio",
            message=", ".join(reasons) if reasons else "No issues detected",
            flagged=flagged,
            severity=severity,
            category="audio_safety",
            confidence_by_reason=_build_confidence_by_reason(alert),
            reasons=reasons,
            data=_build_alert_data(alert),
            commit=False,
        )

        volume_stats = alert.get("volume_stats", {})
        db.insert_audio_alert(
            run_id=run_id,
            alert_id=alert_id,
            max_volume_db=volume_stats.get("max_db", 0.0),
            mean_volume_db=volume_stats.get("mean_db", 0.0),
            silence_count=len(alert.get("silence_periods", [])),
            speech_rate_wpm=alert.get("speech_rate_wpm", 0.0),
            background_noise_db=alert.get("background_noise_db", 0.0),
            speaker_count=alert.get("estimated_speakers", 1),
            commit=commit,
        )

        if print_summary:
            _print_audio_alert(alert)

        return alert_id
    except Exception as exc:
        logger.error(f"Failed to log audio alert: {exc}")
        return None


def log_transcript_alert(
    message: str,
    alert: dict,
    print_summary: bool = True,
    commit: bool = True,
    run_id: Optional[str] = None,
) -> Optional[int]:
    """
    Logs a transcript (from video audio) analysis alert to the database.

    Args:
        message: The transcript text
        alert: Alert dict from chat_analyzer.analyze_message()
        print_summary: If True, prints alert to stdout

    Returns:
        The inserted alert ID, or None on failure.
    """
    db = get_db()
    try:
        flagged = alert.get("flagged", False)
        confidence = alert.get("confidence")
        reason_list = alert.get("reasons", [])
        reasons = ", ".join(reason_list)
        severity = _determine_severity(reason_list)
        confidence_by_reason = _build_confidence_by_reason(alert)

        alert_id = db.insert_alert(
            run_id=run_id,
            source="transcript",
            message=_message_for_storage(message),
            flagged=flagged,
            severity=severity,
            category="transcript_safety",
            confidence=confidence or 0.0,
            confidence_by_reason=confidence_by_reason,
            reasons=reason_list,
            data=_build_alert_data(alert),
            commit=False,
        )

        # Insert chat-specific details (transcript is analyzed like chat)
        db.insert_chat_alert(
            run_id=run_id,
            alert_id=alert_id,
            text=message,
            has_profanity="profanity" in reason_list,
            has_pii=any("pii:" in r for r in reason_list),
            pii_types=[r.split(":")[1] for r in reason_list if r.startswith("pii:")],
            toxicity_score=confidence or 0.0,
            toxicity_label="toxic" if ("toxicity:" in reasons) else "non_toxic",
            sentiment=alert.get("sentiment") or "",
            sentiment_score=float(alert.get("sentiment_score") or 0.0),
            entities=_normalize_entities_for_storage(alert.get("entities")),
            commit=commit,
        )

        db.insert_transcript_segment(
            run_id=run_id,
            alert_id=alert_id,
            segment_text=message,
            confidence=float(alert.get("segment_confidence") or confidence or 0.0),
            start_time=alert.get("segment_start_time"),
            end_time=alert.get("segment_end_time"),
            commit=commit,
        )

        if print_summary:
            _print_transcript_alert(alert)

        return alert_id
    except Exception as exc:
        logger.error(f"Failed to log transcript alert: {exc}")
        return None


# ─────────────────────────────────────────────
# BATCH LOGGING (backward compatibility)
# ─────────────────────────────────────────────

def log_alerts(
    alerts: list[dict],
    source: SourceType,
    print_summary: bool = True,
    run_id: Optional[str] = None,
) -> list[int]:
    """
    Tags a batch of alert dicts with `source` and persists them to database.

    Args:
        alerts: List of raw alert dicts from analyzer.
        source: "chat", "video_frame", "transcript", or "audio".
        print_summary: If True, prints each alert to stdout.

    Returns:
        List of inserted alert IDs.
    """
    if not alerts:
        logger.info(f"No alerts to log for source={source!r}.")
        return []

    inserted_ids = []
    for alert in alerts:
        alert_id = None
        if source == "chat":
            alert_id = log_chat_alert(alert, print_summary=print_summary, commit=False, run_id=run_id)
        elif source == "video_frame":
            alert_id = log_video_alert(alert, print_summary=print_summary, commit=False, run_id=run_id)
        elif source == "audio":
            alert_id = log_audio_alert(alert, print_summary=print_summary, commit=False, run_id=run_id)
        elif source == "transcript":
            alert_id = log_transcript_alert(
                alert.get("message", ""),
                alert,
                print_summary=print_summary,
                commit=False,
                run_id=run_id,
            )

        if alert_id:
            inserted_ids.append(alert_id)

    # Single commit for the whole batch significantly reduces write overhead.
    db = get_db()
    db.commit()

    logger.info(
        f"Logged {len(inserted_ids)} alert(s) from source={source!r} to database"
    )

    return inserted_ids


# ─────────────────────────────────────────────
# CONSOLE PRINTING FUNCTIONS
# ─────────────────────────────────────────────

def _determine_severity(reasons: list[str]) -> str:
    """Determine severity based on alert reasons."""
    if not reasons:
        return "low"
    if any("critical" in r.lower() for r in reasons):
        return "critical"
    if any("strong_" in r or "shouting" in r for r in reasons):
        return "high"
    if any("pii:" in r for r in reasons):
        return "high"
    return "medium"


def _print_chat_alert(alert: dict) -> None:
    """Prints formatted chat alert."""
    reasons_str = ", ".join(alert.get("reasons", []))
    msg = _message_for_console(str(alert.get("message", "")))
    sentiment = alert.get("sentiment", "N/A")
    print(
        f"[CHAT ALERT] | {msg} | Sentiment: {sentiment} | Reasons: {reasons_str}"
    )


def _print_video_alert(alert: dict) -> None:
    """Prints formatted video alert."""
    frame = alert.get("frame_number", "?")
    ts = alert.get("timestamp_sec", "?")
    nsfw = alert.get("nsfw_label", "?")
    emotion = alert.get("emotion", "?")
    reasons = ", ".join(alert.get("reasons", []))
    print(
        f"[VIDEO ALERT] | Frame {frame} @ {ts:.1f}s | NSFW: {nsfw} | "
        f"Emotion: {emotion} | {reasons}"
    )


def _print_audio_alert(alert: dict) -> None:
    """Prints formatted audio alert."""
    reasons = ", ".join(alert.get("reasons", []))
    volume = alert.get("volume_stats", {}).get("max_db", 0)
    speakers = alert.get("estimated_speakers", 1)
    print(
        f"[AUDIO ALERT] | Max Volume: {volume:.1f}dB | Speakers: {speakers} | "
        f"{reasons}"
    )


def _print_transcript_alert(alert: dict) -> None:
    """Prints formatted transcript alert."""
    msg = _message_for_console(str(alert.get("message", "")))
    reasons = ", ".join(alert.get("reasons", []))
    print(f"[TRANSCRIPT ALERT] | {msg} | {reasons}")


# ─────────────────────────────────────────────
# RETRIEVAL & STATISTICS
# ─────────────────────────────────────────────

def get_all_alerts(run_id: Optional[str] = None) -> list[dict]:
    """
    Returns all alerts from the database.

    Returns:
        List of alert dicts sorted by timestamp (descending).
    """
    db = get_db()
    return db.get_all_alerts(run_id=run_id)


def get_alert_stats(run_id: Optional[str] = None) -> dict:
    """
    Computes summary statistics over all persisted alerts.

    Returns:
        A dict with keys:
          - total_alerts (int)
          - flagged_alerts (int)
          - by_source (dict[source → count])
          - by_severity (dict[severity → count])
    """
    db = get_db()
    return db.get_alert_stats(run_id=run_id)


def clear_alerts() -> None:
    """
    Wipes all alerts from the database.
    """
    db = get_db()
    logger.warning("Clearing all alerts from database...")
    db.clear_all_alerts()
