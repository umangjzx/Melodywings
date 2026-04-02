"""
alert_engine.py — Part 3: Unified Alert Logger
MelodyWings Guard | Real-Time Content Safety System

Accepts structured alert dicts from chat_analyzer and video_analyzer,
tags them with a "source" field, persists them to SQLite database,
and exposes retrieval and statistics functions.
"""

import logging
from typing import Literal, Optional
from database import get_db

logger = logging.getLogger(__name__)

# Alert source types
SourceType = Literal["chat", "video_frame", "transcript", "audio"]


# ─────────────────────────────────────────────
# ALERT LOGGING
# ─────────────────────────────────────────────

def log_chat_alert(alert: dict, print_summary: bool = True) -> Optional[int]:
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
        reasons = ", ".join(alert.get("reasons", []))
        severity = _determine_severity(alert.get("reasons", []))

        # Insert main alert
        alert_id = db.insert_alert(
            source="chat",
            message=message[:500],
            flagged=flagged,
            severity=severity,
            category="chat_safety",
            confidence=confidence or 0.0,
        )

        # Insert chat-specific details
        db.insert_chat_alert(
            alert_id=alert_id,
            text=message,
            has_profanity="profanity" in alert.get("reasons", []),
            has_pii=any("pii:" in r for r in alert.get("reasons", [])),
            pii_types=[r.split(":")[1] for r in alert.get("reasons", []) if r.startswith("pii:")],
            toxicity_score=confidence or 0.0,
            toxicity_label="toxic" if ("toxicity:" in reasons) else "non_toxic",
            sentiment=alert.get("sentiment") or "",
            sentiment_score=float(alert.get("sentiment_score") or 0.0),
            entities=[e.get("text") for e in alert.get("entities", [])],
        )

        if print_summary:
            _print_chat_alert(alert)

        return alert_id
    except Exception as exc:
        logger.error(f"Failed to log chat alert: {exc}")
        return None


def log_video_alert(alert: dict, print_summary: bool = True) -> Optional[int]:
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
            source="video_frame",
            message=f"Frame {frame_number} @ {timestamp_sec:.1f}s",
            flagged=flagged,
            severity=_determine_severity(alert.get("reasons", [])),
            category="video_safety",
            confidence=nsfw_score,
        )

        db.insert_video_alert(
            alert_id=alert_id,
            frame_number=frame_number,
            timestamp_sec=timestamp_sec,
            nsfw_label=nsfw_label,
            nsfw_score=nsfw_score,
            emotion=emotion,
        )

        if print_summary:
            _print_video_alert(alert)

        return alert_id
    except Exception as exc:
        logger.error(f"Failed to log video alert: {exc}")
        return None


def log_audio_alert(alert: dict, print_summary: bool = True) -> Optional[int]:
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
            source="audio",
            message=", ".join(reasons) if reasons else "No issues detected",
            flagged=flagged,
            severity=severity,
            category="audio_safety",
        )

        volume_stats = alert.get("volume_stats", {})
        db.insert_audio_alert(
            alert_id=alert_id,
            max_volume_db=volume_stats.get("max_db", 0.0),
            mean_volume_db=volume_stats.get("mean_db", 0.0),
            silence_count=len(alert.get("silence_periods", [])),
            speech_rate_wpm=alert.get("speech_rate_wpm", 0.0),
            background_noise_db=alert.get("background_noise_db", 0.0),
            speaker_count=alert.get("estimated_speakers", 1),
        )

        if print_summary:
            _print_audio_alert(alert)

        return alert_id
    except Exception as exc:
        logger.error(f"Failed to log audio alert: {exc}")
        return None


def log_transcript_alert(message: str, alert: dict, print_summary: bool = True) -> Optional[int]:
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
        reasons = ", ".join(alert.get("reasons", []))
        severity = _determine_severity(alert.get("reasons", []))

        alert_id = db.insert_alert(
            source="transcript",
            message=message[:500],
            flagged=flagged,
            severity=severity,
            category="transcript_safety",
            confidence=confidence or 0.0,
        )

        # Insert chat-specific details (transcript is analyzed like chat)
        db.insert_chat_alert(
            alert_id=alert_id,
            text=message,
            has_profanity="profanity" in alert.get("reasons", []),
            has_pii=any("pii:" in r for r in alert.get("reasons", [])),
            toxicity_score=confidence or 0.0,
            sentiment=alert.get("sentiment") or "",
            sentiment_score=float(alert.get("sentiment_score") or 0.0),
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
            alert_id = log_chat_alert(alert, print_summary=print_summary)
        elif source == "video_frame":
            alert_id = log_video_alert(alert, print_summary=print_summary)
        elif source == "audio":
            alert_id = log_audio_alert(alert, print_summary=print_summary)
        elif source == "transcript":
            alert_id = log_transcript_alert(
                alert.get("message", ""),
                alert,
                print_summary=print_summary,
            )

        if alert_id:
            inserted_ids.append(alert_id)

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
    msg = alert.get("message", "")[:80].replace("\n", " ")
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
    msg = alert.get("message", "")[:80].replace("\n", " ")
    reasons = ", ".join(alert.get("reasons", []))
    print(f"[TRANSCRIPT ALERT] | {msg} | {reasons}")


# ─────────────────────────────────────────────
# RETRIEVAL & STATISTICS
# ─────────────────────────────────────────────

def get_all_alerts() -> list[dict]:
    """
    Returns all alerts from the database.

    Returns:
        List of alert dicts sorted by timestamp (descending).
    """
    db = get_db()
    return db.get_all_alerts()


def get_alert_stats() -> dict:
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
    return db.get_alert_stats()


def clear_alerts() -> None:
    """
    Wipes all alerts from the database.
    """
    db = get_db()
    logger.warning("Clearing all alerts from database...")
    db.clear_all_alerts()
