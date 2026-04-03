"""
upload_module.py — Upload + Validation Routes
MelodyWings Guard | HTML Safety Dashboard
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, send_file, url_for
from werkzeug.utils import secure_filename

from alert_engine import log_alerts, log_audio_alert
from audio_analyzer import analyze_audio_features
from database import get_db
from video_analyzer import analyze_video, extract_audio, transcribe_audio_with_fallback

upload_bp = Blueprint("upload", __name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
ALLOWED_VIDEO_MIME_TYPES = {
    "application/octet-stream",
    "video/avi",
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/webm",
    "video/x-flv",
    "video/x-matroska",
    "video/x-ms-asf",
    "video/x-msvideo",
}


def _read_env_int(name: str, default: int, minimum: int = 0) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


MAX_UPLOAD_MB = _read_env_int("MWG_MAX_UPLOAD_MB", 512, minimum=1)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

RUN_STATUS: dict[str, dict[str, Any]] = {}
RUN_STATUS_LOCK = threading.Lock()

TOXICITY_THRESHOLD = 0.75
STRONG_NEGATIVE_THRESHOLD = 0.98
NSFW_THRESHOLD = 0.70
FLAGGED_VIDEO_EMOTIONS = {"disgust", "angry"}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_timestamp(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _parse_audio_reasons(message: str) -> list[str]:
    if not message:
        return []
    return [part.strip() for part in message.split(",") if part.strip()]


def _derive_reason_tags(row: dict[str, Any]) -> list[str]:
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
    for candidate in (
        row.get("chat_text"),
        row.get("transcript_segment_text"),
        row.get("message"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _get_args_list(args: Any, key: str) -> list[str]:
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


def _allowed_video_file(filename: str) -> bool:
    ext = Path(filename or "").suffix.lower()
    return ext in ALLOWED_VIDEO_EXTS


def _estimate_upload_size_bytes(file_storage: Any) -> int | None:
    """Estimate uploaded file size from in-memory stream when available."""
    stream = getattr(file_storage, "stream", None)
    if stream is None:
        return None

    try:
        current_pos = stream.tell()
        stream.seek(0, os.SEEK_END)
        size = int(stream.tell())
        stream.seek(current_pos)
        return max(0, size)
    except Exception:
        return None


def _init_run_status(run_id: str, video_path: str) -> None:
    with RUN_STATUS_LOCK:
        RUN_STATUS[run_id] = {
            "run_id": run_id,
            "video_path": video_path,
            "status": "queued",
            "stage": "upload",
            "total_frames": 0,
            "processed_frames": 0,
            "progress": 0.0,
            "current_frame": None,
            "message": "queued",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def _update_run_status(run_id: str, **updates: Any) -> None:
    with RUN_STATUS_LOCK:
        payload = RUN_STATUS.get(run_id, {"run_id": run_id})
        payload.update(updates)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        RUN_STATUS[run_id] = payload


def _get_run_status(run_id: str) -> dict[str, Any] | None:
    with RUN_STATUS_LOCK:
        payload = RUN_STATUS.get(run_id)
        return dict(payload) if payload else None


def _progress_callback_factory(run_id: str):
    def _callback(payload: dict[str, Any]) -> None:
        total_frames = int(payload.get("total_frames") or 0)
        processed_frames = int(payload.get("processed_frames") or 0)
        progress = (processed_frames / total_frames * 100.0) if total_frames else 0.0
        _update_run_status(
            run_id,
            status="running",
            stage=str(payload.get("stage") or "frames"),
            total_frames=total_frames,
            processed_frames=processed_frames,
            progress=round(progress, 2),
            current_frame=payload.get("current_frame"),
            message=payload.get("message") or "running",
        )

    return _callback


def _process_video_run(run_id: str, video_path: str) -> None:
    _update_run_status(run_id, status="running", stage="init", message="starting")

    try:
        shared_audio: dict[str, Any] | None = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = os.path.join(tmp_dir, "extracted_audio.wav")

            if extract_audio(video_path, audio_path):
                transcript, word_data, confidence, engine = transcribe_audio_with_fallback(audio_path)
                shared_audio = {
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "word_data": word_data,
                    "transcription_confidence": confidence,
                    "transcription_engine": engine,
                }
            else:
                shared_audio = None

            frame_results, transcript_results = analyze_video(
                video_path,
                shared_audio=shared_audio,
                progress_callback=_progress_callback_factory(run_id),
                run_id=run_id,
                save_flagged_frames=True,
            )

            if frame_results:
                log_alerts(frame_results, source="video_frame", print_summary=False, run_id=run_id)
            if transcript_results:
                log_alerts(transcript_results, source="transcript", print_summary=False, run_id=run_id)

            if shared_audio and shared_audio.get("audio_path") and os.path.exists(shared_audio["audio_path"]):
                _update_run_status(run_id, stage="audio", message="analyzing audio")
                audio_result = analyze_audio_features(shared_audio["audio_path"]) or {}
                if audio_result:
                    log_audio_alert(audio_result, print_summary=False, run_id=run_id)

        if frame_results:
            db = get_db()
            for result in frame_results:
                if not result.get("flagged"):
                    continue
                frame_path = result.get("frame_path")
                if not frame_path:
                    continue
                db.insert_flagged_frame(
                    run_id=run_id,
                    frame_path=str(frame_path),
                    timestamp_sec=float(result.get("timestamp_sec") or 0.0),
                    label=str(result.get("flag_label") or "flagged"),
                    confidence=float(result.get("flag_confidence") or 0.0),
                )

        status_payload = _get_run_status(run_id) or {}
        total_frames = int(status_payload.get("total_frames") or 0)
        processed_frames = len(frame_results)
        progress = (processed_frames / total_frames * 100.0) if total_frames else 100.0
        _update_run_status(
            run_id,
            status="complete",
            stage="complete",
            processed_frames=processed_frames,
            progress=round(progress, 2),
            message="complete",
        )
    except Exception as exc:
        _update_run_status(run_id, status="error", stage="error", message=str(exc))


@upload_bp.post("/upload")
def upload_video() -> Any:
    request_size = int(request.content_length or 0)
    if request_size > MAX_UPLOAD_BYTES:
        return jsonify({"error": "file_too_large", "max_upload_mb": MAX_UPLOAD_MB}), 413

    if "video" not in request.files:
        return jsonify({"error": "missing_video_file"}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"error": "empty_filename"}), 400

    estimated_size = _estimate_upload_size_bytes(file)
    if estimated_size is not None and estimated_size > MAX_UPLOAD_BYTES:
        return jsonify({"error": "file_too_large", "max_upload_mb": MAX_UPLOAD_MB}), 413

    if not _allowed_video_file(file.filename):
        return jsonify({"error": "unsupported_file_type"}), 400

    mime_type = str(file.mimetype or "").lower().strip()
    if mime_type and mime_type not in ALLOWED_VIDEO_MIME_TYPES:
        return jsonify({"error": "unsupported_mime_type", "mime_type": mime_type}), 400

    run_id = uuid.uuid4().hex
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "invalid_filename"}), 400

    save_path = UPLOAD_DIR / f"{run_id}_{filename}"
    file.save(save_path)

    _init_run_status(run_id, str(save_path))

    worker = threading.Thread(
        target=_process_video_run,
        args=(run_id, str(save_path)),
        daemon=True,
    )
    worker.start()

    return jsonify({"run_id": run_id, "filename": filename, "status": "started"})


@upload_bp.get("/status/<run_id>")
def run_status(run_id: str) -> Any:
    payload = _get_run_status(run_id)
    if not payload:
        return jsonify({"error": "run_not_found"}), 404
    return jsonify(payload)


@upload_bp.get("/video/<run_id>")
def stream_video(run_id: str) -> Any:
    payload = _get_run_status(run_id)
    if not payload:
        return jsonify({"error": "run_not_found"}), 404

    video_path = payload.get("video_path")
    if not video_path:
        return jsonify({"error": "video_path_missing"}), 404

    resolved = Path(video_path).resolve()
    uploads_root = UPLOAD_DIR.resolve()
    try:
        resolved.relative_to(uploads_root)
    except ValueError:
        return jsonify({"error": "invalid_video_path"}), 403

    if not resolved.exists():
        return jsonify({"error": "video_not_found"}), 404

    return send_file(resolved, as_attachment=False)


@upload_bp.get("/api/transcript")
def transcript_items() -> Any:
    run_id = str(request.args.get("run_id") or "").strip() or None
    search = str(request.args.get("search") or "").strip().lower()
    flagged_only = str(request.args.get("flagged") or "").strip().lower() in {"1", "true", "yes"}
    limit = int(request.args.get("limit") or 200)
    limit = max(10, min(limit, 1000))

    db = get_db()
    rows = db.get_alerts_detailed(source="transcript", flagged_only=flagged_only, run_id=run_id)

    items: list[dict[str, Any]] = []
    for row in rows:
        segment_text = str(row.get("transcript_segment_text") or row.get("message") or "").strip()
        if not segment_text:
            continue
        if search and search not in segment_text.lower():
            continue

        reasons = _derive_reason_tags(row)
        items.append(
            {
                "id": row.get("id"),
                "run_id": row.get("run_id"),
                "segment_text": segment_text,
                "segment_start_time": row.get("segment_start_time"),
                "segment_end_time": row.get("segment_end_time"),
                "segment_confidence": row.get("segment_confidence"),
                "flagged": bool(row.get("flagged")),
                "reason_text": ", ".join(reasons) if reasons else "safe",
                "sentiment": row.get("sentiment"),
                "sentiment_score": row.get("sentiment_score"),
                "toxicity_score": row.get("toxicity_score"),
                "timestamp": row.get("timestamp"),
            }
        )

        if len(items) >= limit:
            break

    items.sort(key=lambda item: float(item.get("segment_start_time") or 0.0))
    return jsonify({"items": items})


@upload_bp.get("/api/flagged-items")
def flagged_items() -> Any:
    run_id = str(request.args.get("run_id") or "").strip() or None
    types = {
        *(str(t).strip().lower() for t in _get_args_list(request.args, "type")),
        *(str(t).strip().lower() for t in _get_args_list(request.args, "types")),
    }

    include_nsfw = not types or "nsfw" in types
    include_emotion = not types or "emotion" in types
    include_audio = not types or "audio" in types
    include_text = not types or "text" in types

    db = get_db()
    items: list[dict[str, Any]] = []

    if include_nsfw or include_emotion:
        frames = db.get_flagged_frames(run_id=run_id)
        frame_ids = [int(frame.get("id", 0)) for frame in frames]
        validation_summary = db.get_validation_summary(frame_ids)

        for frame in frames:
            label = str(frame.get("label") or "")
            is_nsfw = label == "nsfw"
            is_emotion = not is_nsfw
            if (is_nsfw and not include_nsfw) or (is_emotion and not include_emotion):
                continue

            frame_id = int(frame.get("id", 0))
            frame_path = str(frame.get("frame_path") or "")
            items.append(
                {
                    "id": frame_id,
                    "item_type": "video_frame",
                    "run_id": frame.get("run_id"),
                    "frame_path": frame_path,
                    "frame_url": url_for("static", filename=frame_path) if frame_path else "",
                    "timestamp_sec": float(frame.get("timestamp_sec") or 0.0),
                    "label": label or "flagged",
                    "confidence": float(frame.get("confidence") or 0.0),
                    "validation": validation_summary.get(frame_id, {"correct": 0, "incorrect": 0}),
                    "_sort_key": float(frame.get("timestamp_sec") or 0.0),
                }
            )

    if include_audio or include_text:
        rows = db.get_alerts_detailed(flagged_only=True, run_id=run_id)
        for row in rows:
            source = str(row.get("source") or "")
            if source == "audio" and not include_audio:
                continue
            if source in {"chat", "transcript"} and not include_text:
                continue
            if source not in {"audio", "chat", "transcript"}:
                continue

            reason_tags = _derive_reason_tags(row)
            reason_text = ", ".join(reason_tags) if reason_tags else "flagged"
            content_text = _content_text(row)
            preview = content_text[:160] + ("..." if len(content_text) > 160 else "")
            timestamp_dt = _parse_timestamp(str(row.get("timestamp") or ""))
            sort_key = timestamp_dt.timestamp() if timestamp_dt else 0.0

            items.append(
                {
                    "id": row.get("id"),
                    "item_type": "audio" if source == "audio" else "text",
                    "run_id": row.get("run_id"),
                    "frame_path": "",
                    "frame_url": "",
                    "timestamp_display": str(row.get("timestamp") or ""),
                    "label": "audio" if source == "audio" else "text",
                    "confidence": float(row.get("confidence") or 0.0),
                    "message": preview,
                    "reason_text": reason_text,
                    "_sort_key": sort_key,
                }
            )

    items.sort(key=lambda item: float(item.get("_sort_key") or 0.0))
    for item in items:
        item.pop("_sort_key", None)

    return jsonify({"items": items})


@upload_bp.post("/validate")
def validate_frame() -> Any:
    payload = request.get_json(silent=True) or request.form
    frame_id = payload.get("frame_id") if payload else None
    feedback = str(payload.get("user_feedback") or payload.get("feedback") or "").strip().lower()

    if not frame_id:
        return jsonify({"error": "missing_frame_id"}), 400
    if feedback not in {"correct", "incorrect"}:
        return jsonify({"error": "invalid_feedback"}), 400

    db = get_db()
    log_id = db.insert_validation_log(int(frame_id), feedback)
    summary = db.get_validation_summary([int(frame_id)])
    return jsonify({"status": "ok", "validation_id": log_id, "summary": summary.get(int(frame_id), {})})
