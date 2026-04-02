"""
video_analyzer.py — Part 2: Video Frame + Transcript Safety Analysis
MelodyWings Guard | Real-Time Content Safety System

Analyzes video content for:
  1. NSFW frames  (HuggingFace: Falconsai/nsfw_image_detection)
  2. Emotion detection per frame  (DeepFace)
    3. Transcript toxicity via Whisper (long-form) + fallback SpeechRecognition + chat_analyzer pipeline
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LAZY MODEL LOADERS
# ─────────────────────────────────────────────
_nsfw_pipeline = None
_whisper_asr_pipeline = None
_face_detector = None
_last_video_analysis_metrics: dict[str, Any] = {}


def _read_env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read integer environment variable safely with lower-bound clamping."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


def _read_env_float(
    name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Read float environment variable safely with optional clamping."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        parsed = float(raw_value)
    except ValueError:
        parsed = float(default)

    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _read_env_bool(name: str, default: bool = False) -> bool:
    """Read boolean environment variable safely."""
    fallback = "true" if default else "false"
    raw_value = os.getenv(name, fallback).strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator while avoiding division-by-zero errors."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def get_last_video_analysis_metrics() -> dict[str, Any]:
    """Return metrics from the most recent analyze_video run."""
    return dict(_last_video_analysis_metrics)


def _set_last_video_analysis_metrics(metrics: dict[str, Any]) -> None:
    """Persist latest video-analysis metrics for external reporting."""
    global _last_video_analysis_metrics
    _last_video_analysis_metrics = dict(metrics)


def _get_nsfw_pipeline():
    """
    Lazily loads and caches the NSFW image classifier pipeline.
    Returns the pipeline or None if loading fails.
    """
    global _nsfw_pipeline
    if _nsfw_pipeline is not None:
        return _nsfw_pipeline
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        device_name = "cuda" if device == 0 else "cpu"

        logger.info(f"Loading NSFW model: Falconsai/nsfw_image_detection (device={device_name}) ...")
        _nsfw_pipeline = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device,
        )
        logger.info("NSFW model loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load NSFW model: {exc}")
        _nsfw_pipeline = None
    return _nsfw_pipeline


def _normalize_nsfw_label(label: str) -> str:
    """Normalize model-specific labels to safe/nsfw/unknown categories."""
    normalized = str(label or "").strip().lower().replace("_", " ")
    if not normalized:
        return "unknown"

    if "not safe" in normalized or "nsfw" in normalized or "unsafe" in normalized:
        return "nsfw"

    safe_markers = ("safe", "sfw", "normal", "neutral")
    if any(marker in normalized for marker in safe_markers):
        return "safe"

    return "unknown"


def _extract_nsfw_probability(raw_prediction: object) -> tuple[str, float]:
    """Convert HF image-classification output into (label, nsfw_probability)."""
    candidates: list[dict[str, object]] = []

    if isinstance(raw_prediction, dict):
        candidates = [raw_prediction]
    elif isinstance(raw_prediction, list):
        candidates = [item for item in raw_prediction if isinstance(item, dict)]

    if not candidates:
        return "unknown", 0.0

    nsfw_scores: list[float] = []
    safe_scores: list[float] = []
    scored_candidates: list[tuple[str, float]] = []

    for candidate in candidates:
        label = _normalize_nsfw_label(str(candidate.get("label", "")))
        try:
            score = float(candidate.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        score = max(0.0, min(1.0, score))
        scored_candidates.append((label, score))

        if label == "nsfw":
            nsfw_scores.append(score)
        elif label == "safe":
            safe_scores.append(score)

    if nsfw_scores:
        nsfw_probability = max(nsfw_scores)
    elif safe_scores:
        nsfw_probability = max(0.0, 1.0 - max(safe_scores))
    else:
        top_label, top_score = max(scored_candidates, key=lambda item: item[1])
        nsfw_probability = top_score if top_label == "nsfw" else 0.0

    nsfw_probability = round(max(0.0, min(1.0, nsfw_probability)), 4)
    return ("nsfw" if nsfw_probability >= 0.5 else "safe"), nsfw_probability


def _get_whisper_asr_pipeline():
    """
    Lazily loads and caches a Whisper ASR pipeline for local transcription.

    Returns:
        HuggingFace ASR pipeline or None if loading fails.
    """
    global _whisper_asr_pipeline
    if _whisper_asr_pipeline is not None:
        return _whisper_asr_pipeline

    model_id = os.getenv("MWG_WHISPER_MODEL", "openai/whisper-base")
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        device_name = "cuda" if device == 0 else "cpu"

        logger.info(f"Loading Whisper ASR model: {model_id} (device={device_name}) ...")
        _whisper_asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
        )
        logger.info("Whisper ASR model loaded successfully.")
    except Exception as exc:
        logger.warning(f"Unable to initialize Whisper ASR model '{model_id}': {exc}")
        _whisper_asr_pipeline = None

    return _whisper_asr_pipeline


# ─────────────────────────────────────────────
# AUDIO EXTRACTION (for audio analysis)
# ─────────────────────────────────────────────

def extract_audio(video_path: str, audio_path: str) -> bool:
    """
    Extracts audio from a video file and saves it as WAV format.
    Used by audio_analyzer for audio feature extraction.

    Args:
        video_path: Path to the video file.
        audio_path: Path where the extracted WAV audio will be saved.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from moviepy.editor import VideoFileClip
        
        logger.info("Extracting audio from video using moviepy...")
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            logger.warning("Video has no audio track")
            video.close()
            return False
        
        # Write audio as WAV file
        video.audio.write_audiofile(
            audio_path,
            verbose=False,
            logger=None,
            write_logfile=False
        )
        video.close()
        logger.info(f"Audio extracted successfully to: {audio_path}")
        return True
        
    except ImportError:
        logger.error("moviepy not installed. Install with: pip install moviepy")
        return False
    except Exception as exc:
        logger.error(f"Audio extraction failed: {exc}")
        return False


# ─────────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────────

def extract_frames(video_path: str, fps: float = 1.0, max_frames: Optional[int] = None):
    """
    Extracts video frames at the given rate using OpenCV.

    Args:
        video_path: Absolute or relative path to the video file.
        fps:        Target frames per second to sample (default: 1).
        max_frames: Optional maximum number of sampled frames.

    Yields:
        Tuples of (frame_number: int, timestamp_sec: float, frame: np.ndarray).
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) not installed. Cannot extract frames.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = max(0.1, float(fps))
    interval = max(1, int(round(video_fps / target_fps)))

    frame_idx = 0
    sampled_idx = 0
    max_samples = max(0, int(max_frames)) if max_frames is not None else 0

    while True:
        if frame_idx % interval != 0:
            # grab() advances frame pointer without full decode.
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = round(frame_idx / video_fps, 3)
        yield sampled_idx, timestamp_sec, frame
        sampled_idx += 1
        frame_idx += 1

        if max_samples > 0 and sampled_idx >= max_samples:
            logger.info(f"Frame extraction reached max_samples={max_samples}; stopping early.")
            break

    cap.release()
    logger.info(f"Frame extraction complete. Total sampled frames: {sampled_idx}")


# ─────────────────────────────────────────────
# NSFW CLASSIFICATION
# ─────────────────────────────────────────────

def classify_nsfw(frame) -> tuple[str, float]:
    """
    Runs NSFW classification on a single video frame.

    Args:
        frame: A numpy array representing the BGR video frame (from OpenCV).

    Returns:
        A (label, score) tuple.
        label is "safe" or "nsfw"; score is a float in [0, 1].
        Returns ("unknown", 0.0) on failure.
    """
    pipe = _get_nsfw_pipeline()
    if pipe is None:
        return "unknown", 0.0
    try:
        from PIL import Image
        import cv2
        # Convert BGR (OpenCV) → RGB → PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        results = pipe(pil_image)
        return _extract_nsfw_probability(results)
    except Exception as exc:
        logger.error(f"NSFW classification failed: {exc}")
        return "unknown", 0.0


def classify_nsfw_batch(frames: list) -> list[tuple[str, float]]:
    """Runs NSFW classification on a frame batch to reduce model call overhead."""
    if not frames:
        return []

    pipe = _get_nsfw_pipeline()
    if pipe is None:
        return [("unknown", 0.0) for _ in frames]

    try:
        from PIL import Image
        import cv2

        batch_size = _read_env_int("MWG_NSFW_PIPELINE_BATCH_SIZE", 8, minimum=1)

        pil_images = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb_frame))

        batch_results = pipe(pil_images, batch_size=batch_size)
        if isinstance(batch_results, dict):
            batch_results = [batch_results]

        parsed: list[tuple[str, float]] = []
        for item in batch_results:
            parsed.append(_extract_nsfw_probability(item))

        if len(parsed) < len(frames):
            parsed.extend([("unknown", 0.0)] * (len(frames) - len(parsed)))
        return parsed[: len(frames)]
    except Exception as exc:
        logger.error(f"Batch NSFW classification failed, falling back to single-frame mode: {exc}")
        return [classify_nsfw(frame) for frame in frames]


# ─────────────────────────────────────────────
# EMOTION DETECTION
# ─────────────────────────────────────────────

FLAGGED_EMOTIONS = {"disgust", "angry"}
LOW_RISK_EMOTIONS = {"fear", "sad"}
EMOTION_CONSECUTIVE_FRAMES = int(os.getenv("MWG_EMOTION_CONSEC_FRAMES", "3"))
LOW_RISK_EMOTION_NSFW_THRESHOLD = 0.70
NSFW_TEMPORAL_ALPHA = _read_env_float("MWG_NSFW_TEMPORAL_ALPHA", 0.45, minimum=0.05, maximum=1.0)
NSFW_MIN_CONSEC_FRAMES = _read_env_int("MWG_NSFW_CONSEC_FRAMES", 2, minimum=1)
NSFW_HIGH_CONFIDENCE_BYPASS = _read_env_float(
    "MWG_NSFW_HIGH_CONFIDENCE_BYPASS", 0.95, minimum=0.0, maximum=1.0
)
FRAME_RESIZE_WIDTH = _read_env_int("MWG_FRAME_RESIZE_WIDTH", 960, minimum=0)
FRAME_ENABLE_CLAHE = _read_env_bool("MWG_FRAME_ENABLE_CLAHE", False)
FRAME_MAX_SAMPLES = _read_env_int("MWG_MAX_FRAMES", 0, minimum=0)
TRANSCRIPT_PRINT_MAX_CHARS = _read_env_int("MWG_TRANSCRIPT_PRINT_MAX_CHARS", 2000, minimum=0)
EMOTION_MIN_BLUR_VARIANCE = _read_env_float("MWG_EMOTION_MIN_BLUR_VARIANCE", 12.0, minimum=0.0)
EMOTION_MIN_BRIGHTNESS = _read_env_float("MWG_EMOTION_MIN_BRIGHTNESS", 8.0, minimum=0.0)
EMOTION_REQUIRE_FACE = _read_env_bool("MWG_EMOTION_REQUIRE_FACE", False)
EMOTION_MIN_FACE_AREA_RATIO = _read_env_float(
    "MWG_EMOTION_MIN_FACE_AREA_RATIO", 0.004, minimum=0.0, maximum=1.0
)
EMOTION_MIN_CONFIDENCE = _read_env_float("MWG_EMOTION_MIN_CONFIDENCE", 0.18, minimum=0.0, maximum=1.0)
EMOTION_FALLBACK_FULL_FRAME = _read_env_bool("MWG_EMOTION_FALLBACK_FULL_FRAME", True)
EMOTION_HOLD_FRAMES = _read_env_int("MWG_EMOTION_HOLD_FRAMES", 3, minimum=0)
EMOTION_RECHECK_INTERVAL = _read_env_int("MWG_EMOTION_RECHECK_INTERVAL", 4, minimum=1)


def _frame_quality_metrics(frame) -> dict[str, float]:
    """Compute lightweight quality metrics used to gate expensive inference."""
    try:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        return {
            "blur_variance": round(blur_variance, 3),
            "brightness": round(brightness, 3),
        }
    except Exception:
        return {"blur_variance": 0.0, "brightness": 0.0}


def _preprocess_frame(frame):
    """Apply lightweight normalization before model inference."""
    try:
        import cv2

        processed = frame
        if FRAME_RESIZE_WIDTH > 0 and int(processed.shape[1]) > FRAME_RESIZE_WIDTH:
            scale = FRAME_RESIZE_WIDTH / float(processed.shape[1])
            target_height = max(1, int(processed.shape[0] * scale))
            processed = cv2.resize(
                processed,
                (FRAME_RESIZE_WIDTH, target_height),
                interpolation=cv2.INTER_AREA,
            )

        if FRAME_ENABLE_CLAHE:
            ycrcb = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)
            y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y_channel = clahe.apply(y_channel)
            processed = cv2.cvtColor(
                cv2.merge((y_channel, cr_channel, cb_channel)),
                cv2.COLOR_YCrCb2BGR,
            )

        return processed
    except Exception as exc:
        logger.debug(f"Frame preprocessing skipped due to error: {exc}")
        return frame


def _get_face_detector():
    """Lazily load and cache OpenCV Haar cascade face detector."""
    global _face_detector
    if _face_detector is not None:
        return _face_detector

    try:
        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise RuntimeError("OpenCV Haar cascade failed to initialize")
        _face_detector = detector
    except Exception as exc:
        logger.warning(f"Face detector unavailable. Emotion detection may be degraded: {exc}")
        _face_detector = None

    return _face_detector


def _extract_largest_face_roi(frame):
    """Extract largest detected face region to reduce DeepFace overhead/noise."""
    detector = _get_face_detector()
    if detector is None:
        return None

    try:
        import cv2

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_size = max(36, int(min(height, width) * 0.08))
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        face_ratio = (float(w) * float(h)) / max(1.0, float(width * height))
        if face_ratio < EMOTION_MIN_FACE_AREA_RATIO:
            return None

        pad_w = int(0.12 * w)
        pad_h = int(0.12 * h)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(width, x + w + pad_w)
        y2 = min(height, y + h + pad_h)
        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]
    except Exception as exc:
        logger.debug(f"Face ROI extraction failed: {exc}")
        return None


def _normalize_emotion_probability(raw_value: object) -> float:
    """Normalize model probability from either [0,1] or [0,100] scale."""
    try:
        score = float(raw_value)
    except (TypeError, ValueError):
        return 0.0

    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _extract_emotion_prediction(analysis: object) -> tuple[Optional[str], float]:
    """Extract dominant emotion and confidence from DeepFace output."""
    records: list[dict[str, Any]] = []
    if isinstance(analysis, dict):
        records = [analysis]
    elif isinstance(analysis, list):
        records = [item for item in analysis if isinstance(item, dict)]

    best_emotion: Optional[str] = None
    best_score = 0.0

    for record in records:
        emotion_map = record.get("emotion")
        dominant_raw = record.get("dominant_emotion")

        candidate_label: Optional[str] = None
        candidate_score = 0.0

        if isinstance(emotion_map, dict) and emotion_map:
            normalized = {
                str(label).strip().lower(): _normalize_emotion_probability(value)
                for label, value in emotion_map.items()
            }
            normalized = {
                label: score
                for label, score in normalized.items()
                if label
            }

            if normalized:
                if dominant_raw:
                    candidate_label = str(dominant_raw).strip().lower()
                    candidate_score = normalized.get(candidate_label, 0.0)

                if not candidate_label or candidate_score <= 0.0:
                    candidate_label, candidate_score = max(
                        normalized.items(), key=lambda item: item[1]
                    )

        if not candidate_label and dominant_raw:
            candidate_label = str(dominant_raw).strip().lower()
            candidate_score = 1.0

        if candidate_label and candidate_score > best_score:
            best_emotion = candidate_label
            best_score = candidate_score

    return best_emotion, round(best_score, 4)


def detect_emotion(frame) -> Optional[str]:
    """
    Detects the dominant facial emotion in a video frame using DeepFace.

    Args:
        frame: A numpy array representing the BGR video frame.

    Returns:
        The dominant emotion string (e.g., "happy", "fear"), or None if
        no face is detected or an error occurs.
    """
    try:
        from deepface import DeepFace

        inputs: list[object] = []
        face_roi = _extract_largest_face_roi(frame)
        if face_roi is not None:
            inputs.append(face_roi)

        if not EMOTION_REQUIRE_FACE and EMOTION_FALLBACK_FULL_FRAME:
            inputs.append(frame)

        if EMOTION_REQUIRE_FACE and face_roi is None:
            return None

        if not inputs:
            inputs.append(frame)

        for emotion_input in inputs:
            try:
                analysis = DeepFace.analyze(
                    emotion_input,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True,
                )
            except Exception as exc:
                logger.debug(f"Emotion inference attempt failed: {exc}")
                continue

            emotion, confidence = _extract_emotion_prediction(analysis)
            if emotion and confidence >= EMOTION_MIN_CONFIDENCE:
                return emotion

        return None
    except Exception as exc:
        logger.debug(f"Emotion detection skipped or failed: {exc}")
        return None


# ─────────────────────────────────────────────
# PER-FRAME ANALYSIS
# ─────────────────────────────────────────────

NSFW_THRESHOLD = 0.70  # confidence cutoff for NSFW flagging
PRINT_FRAME_STATUS = _read_env_bool("MWG_PRINT_FRAME_STATUS", False)


def _emotion_to_sentiment(emotion: Optional[str]) -> str:
    """Map face emotion labels into coarse person sentiment categories."""
    if not emotion:
        return "unknown"

    normalized = str(emotion).strip().lower()
    if normalized in {"happy", "surprise"}:
        return "positive"
    if normalized in {"angry", "disgust", "fear", "sad"}:
        return "negative"
    if normalized in {"neutral", "calm"}:
        return "neutral"
    return "unknown"


def _build_frame_result(
    frame_number: int,
    timestamp_sec: float,
    nsfw_label: str,
    nsfw_score: float,
    emotion: Optional[str],
    emotion_persistence: int,
    nsfw_score_raw: Optional[float] = None,
    frame_quality: Optional[dict[str, float]] = None,
    nsfw_consecutive_hits: int = 0,
) -> dict:
    """Compose one per-frame analysis result and apply rule-based flagging."""
    reasons: list[str] = []

    if nsfw_label == "nsfw" and nsfw_score >= NSFW_THRESHOLD:
        reasons.append(f"nsfw:{nsfw_label}(score={nsfw_score})")

    if (
        emotion
        and emotion in FLAGGED_EMOTIONS
        and emotion_persistence >= EMOTION_CONSECUTIVE_FRAMES
    ):
        reasons.append(f"emotion:{emotion}")

    if (
        emotion
        and emotion in LOW_RISK_EMOTIONS
        and emotion_persistence >= EMOTION_CONSECUTIVE_FRAMES
        and nsfw_label == "nsfw"
        and nsfw_score >= LOW_RISK_EMOTION_NSFW_THRESHOLD
    ):
        reasons.append(f"emotion:{emotion}:corroborated")

    flagged = len(reasons) > 0
    result = {
        "frame_number": frame_number,
        "timestamp_sec": timestamp_sec,
        "nsfw_label": nsfw_label,
        "nsfw_score": nsfw_score,
        "nsfw_score_raw": round(float(nsfw_score_raw if nsfw_score_raw is not None else nsfw_score), 4),
        "nsfw_score_smoothed": round(float(nsfw_score), 4),
        "nsfw_consecutive_hits": int(max(0, nsfw_consecutive_hits)),
        "emotion": emotion or "none",
        "person_sentiment": _emotion_to_sentiment(emotion),
        "flagged": flagged,
        "reasons": reasons,
        "frame_quality": frame_quality or {},
    }

    if flagged:
        _print_frame_alert(result)
    elif PRINT_FRAME_STATUS:
        print(
            f"[FRAME] | Frame: {result['frame_number']:04d} | "
            f"Time: {result['timestamp_sec']:.1f}s | Status: Frame OK"
        )

    return result


def analyze_frame(frame_number: int, timestamp_sec: float, frame) -> dict:
    """
    Runs all safety checks on a single extracted video frame.

    Args:
        frame_number:  Sequential index of the sampled frame.
        timestamp_sec: Time position in the video (seconds).
        frame:         Numpy array (BGR) from OpenCV.

    Returns:
        A dict with keys: frame_number, timestamp_sec, nsfw_label,
        nsfw_score, emotion, flagged, reasons.
    """
    nsfw_label, nsfw_score = classify_nsfw(frame)
    emotion = detect_emotion(frame)
    return _build_frame_result(frame_number, timestamp_sec, nsfw_label, nsfw_score, emotion, 1)


def _print_frame_alert(result: dict) -> None:
    """
    Prints a formatted alert line to stdout for flagged video frames.

    Args:
        result: The frame analysis result dict.
    """
    reasons_str = ", ".join(result["reasons"])
    print(
        f"[ALERT] | Frame: {result['frame_number']:04d} | "
        f"Time: {result['timestamp_sec']:.1f}s | Type: {reasons_str}"
    )


# ─────────────────────────────────────────────
# TRANSCRIPT EXTRACTION
# ─────────────────────────────────────────────

def _merge_chunk_text(existing: str, incoming: str, max_overlap_words: int = 12) -> str:
    """
    Merges two transcript chunks and removes duplicated overlap at the boundary.
    """
    existing = " ".join(existing.split())
    incoming = " ".join(incoming.split())

    if not existing:
        return incoming
    if not incoming:
        return existing

    existing_words = existing.split()
    incoming_words = incoming.split()
    overlap_cap = min(max_overlap_words, len(existing_words), len(incoming_words))

    overlap_size = 0
    for size in range(overlap_cap, 2, -1):
        left = [w.lower() for w in existing_words[-size:]]
        right = [w.lower() for w in incoming_words[:size]]
        if left == right:
            overlap_size = size
            break

    merged = existing_words + incoming_words[overlap_size:]
    return " ".join(merged)


def _build_word_data(text: str, confidence: float) -> list[dict]:
    """
    Converts transcript text into word-level confidence records.
    """
    return [{"word": word, "confidence": confidence} for word in text.split()]


def transcribe_with_whisper(audio_path: str) -> tuple[Optional[str], Optional[list], float]:
    """
    Transcribes full-length audio with local Whisper in long-form mode.

    Audio is loaded as waveform to avoid ffmpeg path issues.

    Args:
        audio_path: Path to WAV audio file.

    Returns:
        (transcript, word_data, confidence_estimate)
    """
    pipe = _get_whisper_asr_pipeline()
    if pipe is None:
        return None, None, 0.0

    language = os.getenv("MWG_TRANSCRIBE_LANGUAGE", "").strip()

    try:
        import librosa

        audio_samples, sample_rate = librosa.load(audio_path, sr=16_000, mono=True)
        if audio_samples is None or len(audio_samples) == 0:
            return None, None, 0.0

        logger.info(
            "Transcribing with Whisper (long-form): "
            f"duration={len(audio_samples) / sample_rate:.1f}s"
        )

        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        try:
            result = pipe(
                audio_samples,
                return_timestamps="word",
                generate_kwargs=generate_kwargs,
            )
        except Exception:
            # Retry with segment timestamps if word timestamps are unavailable.
            result = pipe(
                audio_samples,
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )

        transcript = ""
        chunks = []
        if isinstance(result, dict):
            transcript = str(result.get("text", "")).strip()
            chunks = result.get("chunks") or []
        elif isinstance(result, str):
            transcript = result.strip()
        else:
            transcript = str(result).strip()

        if not transcript:
            return None, None, 0.0

        # Whisper pipeline does not always expose token confidence; use strong baseline.
        confidence_estimate = 0.92
        if chunks:
            word_data: list[dict] = []
            for chunk in chunks:
                chunk_text = str(chunk.get("text", "")).strip()
                if not chunk_text:
                    continue

                words = chunk_text.split()
                timestamp_pair = chunk.get("timestamp")
                start_time = None
                end_time = None
                if isinstance(timestamp_pair, (list, tuple)) and len(timestamp_pair) == 2:
                    start_time = timestamp_pair[0]
                    end_time = timestamp_pair[1]

                if (
                    start_time is not None
                    and end_time is not None
                    and isinstance(start_time, (int, float))
                    and isinstance(end_time, (int, float))
                    and end_time > start_time
                    and words
                ):
                    step = (float(end_time) - float(start_time)) / len(words)
                    for index, word in enumerate(words):
                        w_start = float(start_time) + index * step
                        w_end = float(start_time) + (index + 1) * step
                        word_data.append(
                            {
                                "word": word,
                                "confidence": confidence_estimate,
                                "start_time": round(w_start, 3),
                                "end_time": round(w_end, 3),
                            }
                        )
                else:
                    for word in words:
                        word_data.append({"word": word, "confidence": confidence_estimate})

            if not word_data:
                word_data = _build_word_data(transcript, confidence_estimate)
        else:
            word_data = _build_word_data(transcript, confidence_estimate)

        return transcript, word_data, confidence_estimate
    except Exception as exc:
        logger.warning(f"Whisper transcription failed: {exc}")
        return None, None, 0.0


def extract_audio_and_transcribe(video_path: str) -> tuple[Optional[str], Optional[list]]:
    """
    Extracts audio from video and transcribes full duration using robust fallback.

    Primary engine: local Whisper (chunked).
    Fallback engine: chunked SpeechRecognition.

    Args:
        video_path: Path to the source video file.

    Returns:
        A tuple of (transcript: str, word_data: list[dict]) where word_data contains
        confidence scores. Returns (None, None) if transcription fails.
    """
    try:
        from moviepy.editor import VideoFileClip
        
        logger.info("Extracting audio from video ...")
        
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        if video.audio is None:
            logger.warning("Video has no audio track")
            video.close()
            return None, None
        
        # Create temporary audio file (WAV format for speech recognition)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            audio_path = tmp_audio.name
        
        logger.info(f"Writing audio to: {audio_path}")
        video.audio.write_audiofile(
            audio_path,
            verbose=False,
            logger=None,
            write_logfile=False
        )
        video.close()
        
        # Transcribe with Whisper first, then fallback to chunked SpeechRecognition
        logger.info("Transcribing extracted audio ...")
        transcript, word_data, confidence, engine = transcribe_audio_with_fallback(audio_path)
        
        # Clean up temp audio
        try:
            os.remove(audio_path)
        except Exception:
            pass
        
        if transcript:
            logger.info(
                f"Transcription complete via {engine}. "
                f"Length: {len(transcript)} chars | Confidence: {confidence:.2%}"
            )
            return transcript, word_data
        else:
            logger.warning("No speech detected in audio")
            return None, None
            
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install moviepy SpeechRecognition pydub")
        return None, None
    except Exception as exc:
        logger.error(f"Audio extraction failed: {exc}")
        return None, None


def transcribe_audio_with_fallback(
    audio_path: str,
) -> tuple[Optional[str], Optional[list], float, str]:
    """
    Best-effort transcription wrapper.

    Returns:
        (transcript, word_data, confidence, engine_name)
    """
    transcript, word_data, confidence = transcribe_with_whisper(audio_path)
    if transcript:
        return transcript, word_data, confidence, "whisper"

    logger.warning("Whisper unavailable/failed. Falling back to chunked SpeechRecognition.")
    transcript, word_data, confidence = transcribe_with_speech_recognition(audio_path)
    if transcript:
        return transcript, word_data, confidence, "speech_recognition"

    return None, None, 0.0, "none"


def transcribe_with_speech_recognition(audio_path: str) -> tuple[Optional[str], Optional[list], float]:
    """
    Transcribes audio using SpeechRecognition in fixed chunks.

    Chunking avoids partial transcripts caused by single giant recognition calls.
    Uses Google recognizer first, then pocketsphinx fallback per chunk.

    Args:
        audio_path: Path to the WAV audio file.

    Returns:
        (transcript, word_data, confidence_estimate)
    """
    try:
        import speech_recognition as sr
        from pydub import AudioSegment
        
        logger.info("Loading speech recognizer ...")
        recognizer = sr.Recognizer()

        chunk_ms = max(5_000, int(os.getenv("MWG_SR_CHUNK_MS", "30000")))
        overlap_ms = max(0, int(os.getenv("MWG_SR_OVERLAP_MS", "1200")))

        audio = AudioSegment.from_file(audio_path)
        total_ms = len(audio)
        if total_ms <= 0:
            return None, None, 0.0

        logger.info(
            "SpeechRecognition fallback chunking: "
            f"duration={total_ms/1000:.1f}s chunk={chunk_ms/1000:.1f}s overlap={overlap_ms/1000:.1f}s"
        )

        merged_transcript = ""
        chunk_confidences = []
        word_data: list[dict] = []

        chunk_index = 0
        start_ms = 0
        while start_ms < total_ms:
            end_ms = min(start_ms + chunk_ms, total_ms)
            read_start = max(0, start_ms - overlap_ms if start_ms > 0 else 0)
            chunk = audio[read_start:end_ms]

            # Skip silent chunks quickly.
            if chunk.rms == 0:
                start_ms += chunk_ms
                chunk_index += 1
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
                chunk_path = tmp_chunk.name
            chunk.export(chunk_path, format="wav")

            try:
                with sr.AudioFile(chunk_path) as source:
                    audio_data = recognizer.record(source)

                chunk_text = None
                chunk_conf = 0.0

                try:
                    chunk_text = recognizer.recognize_google(audio_data)
                    chunk_conf = 0.82
                except sr.RequestError:
                    # Fallback to Sphinx (offline recognition).
                    try:
                        chunk_text = recognizer.recognize_sphinx(audio_data)
                        chunk_conf = 0.65
                    except Exception:
                        chunk_text = None
                        chunk_conf = 0.0
                except sr.UnknownValueError:
                    chunk_text = None
                    chunk_conf = 0.0

                if chunk_text:
                    merged_transcript = _merge_chunk_text(merged_transcript, chunk_text)
                    if chunk_conf > 0:
                        chunk_confidences.append(chunk_conf)

                    chunk_start_sec = read_start / 1000.0
                    chunk_end_sec = end_ms / 1000.0
                    chunk_words = chunk_text.split()
                    if chunk_words and chunk_end_sec > chunk_start_sec:
                        step = (chunk_end_sec - chunk_start_sec) / len(chunk_words)
                        for index, word in enumerate(chunk_words):
                            w_start = chunk_start_sec + index * step
                            w_end = chunk_start_sec + (index + 1) * step
                            word_data.append(
                                {
                                    "word": word,
                                    "confidence": chunk_conf,
                                    "start_time": round(w_start, 3),
                                    "end_time": round(w_end, 3),
                                }
                            )
            finally:
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass

            chunk_index += 1
            if chunk_index % 5 == 0:
                logger.info(f"SpeechRecognition fallback progress: {chunk_index} chunk(s) processed")

            start_ms += chunk_ms

        merged_transcript = " ".join(merged_transcript.split())
        if not merged_transcript:
            return None, None, 0.0

        avg_conf = (
            sum(chunk_confidences) / len(chunk_confidences)
            if chunk_confidences
            else 0.70
        )
        if not word_data:
            word_data = _build_word_data(merged_transcript, avg_conf)
        return merged_transcript, word_data, avg_conf

    except ImportError as exc:
        logger.error(f"SpeechRecognition fallback dependencies missing: {exc}")
        logger.error("Install with: pip install SpeechRecognition pydub pocketsphinx")
        return None, None, 0.0
    except Exception as exc:
        logger.error(f"Speech recognition fallback failed: {exc}")
        return None, None, 0.0


def transcribe_audio(audio_path: str) -> tuple[Optional[str], Optional[list]]:
    """
    Legacy wrapper function for backward compatibility.

    Args:
        audio_path: Path to the WAV audio file.

    Returns:
        A tuple of (transcript: str, segments: list[dict])
    """
    transcript, word_data, _, _ = transcribe_audio_with_fallback(audio_path)
    return transcript, word_data


# ─────────────────────────────────────────────
# FULL VIDEO ANALYSIS PIPELINE
# ─────────────────────────────────────────────

def analyze_video(video_path: str, shared_audio: Optional[dict] = None) -> tuple[list[dict], list[dict]]:
    """
    Runs the complete video safety analysis pipeline.

    Performs:
      1. Frame-by-frame NSFW + emotion analysis
            2. Audio extraction + robust full-length transcription + chat safety analysis

    Args:
        video_path: Absolute or relative path to the video file.
        shared_audio: Optional precomputed payload with keys: transcript, word_data.

    Returns:
        A tuple of (frame_results, transcript_results):
          - frame_results:      list of per-frame analysis dicts
          - transcript_results: list of chat_analyzer result dicts for transcript
    """
    from chat_analyzer import analyze_messages

    analysis_start = time.perf_counter()
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}. Skipping video analysis.")
        _set_last_video_analysis_metrics(
            {
                "video_path": video_path,
                "status": "missing_video",
                "frames_processed": 0,
                "frames_flagged": 0,
                "total_seconds": 0.0,
            }
        )
        return [], []

    logger.info(f"Starting video analysis: {video_path}")

    frame_results: list[dict] = []
    transcript_results: list[dict] = []

    frame_batch_size = _read_env_int("MWG_FRAME_BATCH_SIZE", 8, minimum=1)
    emotion_stride = _read_env_int("MWG_EMOTION_STRIDE", 1, minimum=1)
    sample_fps = _read_env_float("MWG_VIDEO_SAMPLE_FPS", 1.0, minimum=0.1)

    frame_stage_start = time.perf_counter()
    preprocess_seconds = 0.0
    nsfw_inference_seconds = 0.0
    emotion_inference_seconds = 0.0
    emotion_calls = 0
    emotion_skipped_low_quality = 0
    emotion_detected_count = 0
    emotion_reused_count = 0
    emotion_none_count = 0
    quality_blur_sum = 0.0
    quality_brightness_sum = 0.0

    frame_batch: list[tuple[int, float, object, object, dict[str, float]]] = []
    last_emotion: Optional[str] = None
    emotion_hold_remaining = 0
    emotion_streak_label: Optional[str] = None
    emotion_streak_count = 0
    smoothed_nsfw_score = 0.0
    nsfw_consecutive_hits = 0

    def _should_run_emotion_inference(quality: dict[str, float]) -> bool:
        return (
            float(quality.get("blur_variance") or 0.0) >= EMOTION_MIN_BLUR_VARIANCE
            and float(quality.get("brightness") or 0.0) >= EMOTION_MIN_BRIGHTNESS
        )

    def _process_frame_batch(batch: list[tuple[int, float, object, object, dict[str, float]]]) -> None:
        nonlocal emotion_calls
        nonlocal emotion_detected_count
        nonlocal emotion_inference_seconds
        nonlocal emotion_none_count
        nonlocal emotion_reused_count
        nonlocal emotion_skipped_low_quality
        nonlocal emotion_hold_remaining
        nonlocal emotion_streak_count
        nonlocal emotion_streak_label
        nonlocal last_emotion
        nonlocal nsfw_consecutive_hits
        nonlocal nsfw_inference_seconds
        nonlocal smoothed_nsfw_score
        if not batch:
            return

        nsfw_inputs = [item[3] for item in batch]
        nsfw_start = time.perf_counter()
        nsfw_predictions = classify_nsfw_batch(nsfw_inputs)
        nsfw_inference_seconds += time.perf_counter() - nsfw_start

        for idx, (frame_number, timestamp_sec, _raw_frame, preprocessed_frame, quality) in enumerate(batch):
            raw_nsfw_label, raw_nsfw_score = nsfw_predictions[idx]
            raw_nsfw_score = float(raw_nsfw_score or 0.0)
            smoothed_nsfw_score = (
                NSFW_TEMPORAL_ALPHA * raw_nsfw_score
                + (1.0 - NSFW_TEMPORAL_ALPHA) * smoothed_nsfw_score
            )
            smoothed_nsfw_score = max(0.0, min(1.0, smoothed_nsfw_score))

            if raw_nsfw_label == "nsfw" and smoothed_nsfw_score >= NSFW_THRESHOLD:
                nsfw_consecutive_hits += 1
            elif raw_nsfw_label == "nsfw":
                nsfw_consecutive_hits = max(1, nsfw_consecutive_hits)
            else:
                nsfw_consecutive_hits = 0

            nsfw_confirmed = (
                raw_nsfw_score >= NSFW_HIGH_CONFIDENCE_BYPASS
                or nsfw_consecutive_hits >= NSFW_MIN_CONSEC_FRAMES
            )
            effective_nsfw_label = "nsfw" if nsfw_confirmed else "safe"

            emotion: Optional[str] = None
            should_detect = frame_number % emotion_stride == 0
            if should_detect:
                quality_ok = _should_run_emotion_inference(quality)
                periodic_recheck = frame_number % EMOTION_RECHECK_INTERVAL == 0

                if quality_ok or periodic_recheck:
                    emotion_calls += 1
                    emotion_start = time.perf_counter()
                    emotion = detect_emotion(preprocessed_frame)
                    emotion_inference_seconds += time.perf_counter() - emotion_start

                    if emotion:
                        emotion_detected_count += 1
                        last_emotion = emotion
                        emotion_hold_remaining = EMOTION_HOLD_FRAMES
                    elif last_emotion and emotion_hold_remaining > 0:
                        emotion = last_emotion
                        emotion_hold_remaining -= 1
                        emotion_reused_count += 1
                else:
                    emotion_skipped_low_quality += 1
                    if last_emotion and emotion_hold_remaining > 0:
                        emotion = last_emotion
                        emotion_hold_remaining -= 1
                        emotion_reused_count += 1
            elif emotion_stride > 1:
                if last_emotion and emotion_hold_remaining > 0:
                    emotion = last_emotion
                    emotion_hold_remaining -= 1
                    emotion_reused_count += 1
                else:
                    emotion = last_emotion

            if not emotion:
                emotion_none_count += 1

            if emotion:
                if emotion == emotion_streak_label:
                    emotion_streak_count += 1
                else:
                    emotion_streak_label = emotion
                    emotion_streak_count = 1
            else:
                emotion_streak_label = None
                emotion_streak_count = 0

            result = _build_frame_result(
                frame_number=frame_number,
                timestamp_sec=timestamp_sec,
                nsfw_label=effective_nsfw_label,
                nsfw_score=round(smoothed_nsfw_score, 4),
                emotion=emotion,
                emotion_persistence=emotion_streak_count,
                nsfw_score_raw=raw_nsfw_score,
                frame_quality=quality,
                nsfw_consecutive_hits=nsfw_consecutive_hits,
            )
            result["nsfw_label_raw"] = raw_nsfw_label
            frame_results.append(result)

    for frame_number, timestamp_sec, raw_frame in extract_frames(
        video_path,
        fps=sample_fps,
        max_frames=FRAME_MAX_SAMPLES if FRAME_MAX_SAMPLES > 0 else None,
    ):
        quality = _frame_quality_metrics(raw_frame)
        quality_blur_sum += float(quality.get("blur_variance") or 0.0)
        quality_brightness_sum += float(quality.get("brightness") or 0.0)

        preprocess_start = time.perf_counter()
        preprocessed_frame = _preprocess_frame(raw_frame)
        preprocess_seconds += time.perf_counter() - preprocess_start

        frame_batch.append((frame_number, timestamp_sec, raw_frame, preprocessed_frame, quality))
        if len(frame_batch) >= frame_batch_size:
            _process_frame_batch(frame_batch)
            frame_batch = []

    if frame_batch:
        _process_frame_batch(frame_batch)

    frame_stage_seconds = time.perf_counter() - frame_stage_start
    frame_flagged_count = sum(1 for row in frame_results if row.get("flagged"))
    logger.info(
        f"Frame analysis done. Total: {len(frame_results)} | "
        f"Flagged: {frame_flagged_count} | "
        f"Effective FPS: {_safe_div(len(frame_results), frame_stage_seconds):.2f}"
    )

    transcript = None
    word_data = None
    transcription_confidence = 0.0
    transcription_engine = "none"

    transcript_stage_start = time.perf_counter()
    if shared_audio is not None:
        transcript = shared_audio.get("transcript")
        word_data = shared_audio.get("word_data")
        transcription_confidence = float(shared_audio.get("transcription_confidence") or 0.0)
        transcription_engine = str(shared_audio.get("transcription_engine") or "shared")
        logger.info("Using shared precomputed audio transcription payload.")
    else:
        logger.info("Attempting audio extraction and robust transcription...")
        transcript, word_data = extract_audio_and_transcribe(video_path)

    if transcript:
        preview = transcript
        if TRANSCRIPT_PRINT_MAX_CHARS > 0 and len(preview) > TRANSCRIPT_PRINT_MAX_CHARS:
            preview = preview[:TRANSCRIPT_PRINT_MAX_CHARS].rstrip() + "..."

        logger.info(f"Transcript extracted ({len(transcript)} chars).")
        print(f"\n{'=' * 70}")
        print(f"  TRANSCRIPT TEXT EXTRACTED ({len(transcript)} chars)")
        print(f"{'=' * 70}")
        print(preview)
        print(f"{'=' * 70}\n")

        logger.info("Running chat analysis on transcript ...")
        sentences, confidences, segment_meta = _split_transcript_with_word_confidence(
            transcript,
            word_data or [],
        )
        try:
            transcript_results = analyze_messages(sentences, whisper_confidences=confidences)
            for index, result in enumerate(transcript_results):
                meta = segment_meta[index] if index < len(segment_meta) else {}
                result["segment_start_time"] = meta.get("start_time")
                result["segment_end_time"] = meta.get("end_time")
                result["segment_confidence"] = float(
                    meta.get(
                        "confidence",
                        confidences[index] if index < len(confidences) else transcription_confidence,
                    )
                )
        except Exception as exc:
            logger.error(f"Transcript analysis failed: {exc}")
            transcript_results = []
    else:
        logger.warning("Transcript extraction failed. Skipping transcript analysis.")

    transcript_stage_seconds = time.perf_counter() - transcript_stage_start
    total_seconds = time.perf_counter() - analysis_start

    metrics = {
        "video_path": video_path,
        "status": "ok",
        "sample_fps": round(sample_fps, 3),
        "frame_batch_size": frame_batch_size,
        "frames_processed": len(frame_results),
        "frames_flagged": frame_flagged_count,
        "frame_flag_rate": round(_safe_div(frame_flagged_count, len(frame_results)), 4),
        "frame_stage_seconds": round(frame_stage_seconds, 3),
        "avg_frame_processing_ms": round(1000.0 * _safe_div(frame_stage_seconds, len(frame_results)), 3),
        "effective_processing_fps": round(_safe_div(len(frame_results), frame_stage_seconds), 3),
        "preprocess_seconds": round(preprocess_seconds, 3),
        "nsfw_inference_seconds": round(nsfw_inference_seconds, 3),
        "emotion_inference_seconds": round(emotion_inference_seconds, 3),
        "emotion_calls": emotion_calls,
        "emotion_detected_count": emotion_detected_count,
        "emotion_reused_count": emotion_reused_count,
        "emotion_none_count": emotion_none_count,
        "emotion_coverage_ratio": round(1.0 - _safe_div(emotion_none_count, len(frame_results)), 4),
        "emotion_skipped_low_quality": emotion_skipped_low_quality,
        "mean_blur_variance": round(_safe_div(quality_blur_sum, len(frame_results)), 3),
        "mean_brightness": round(_safe_div(quality_brightness_sum, len(frame_results)), 3),
        "transcript_engine": transcription_engine,
        "transcription_confidence": round(transcription_confidence, 4),
        "transcript_segments": len(transcript_results),
        "transcript_flagged": sum(1 for row in transcript_results if row.get("flagged")),
        "transcript_stage_seconds": round(transcript_stage_seconds, 3),
        "total_seconds": round(total_seconds, 3),
    }
    _set_last_video_analysis_metrics(metrics)

    return frame_results, transcript_results


def load_video_ground_truth_flags(
    ground_truth_path: str,
    expected_length: Optional[int] = None,
) -> list[bool]:
    """
    Load expected flagged labels for video frames from JSON.

    Supported JSON formats:
      1. [true, false, ...]
      2. [{"frame_number": 0, "flagged": false}, ...]
      3. {"0": false, "1": true, ...}
    """
    resolved = Path(ground_truth_path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"ground truth file not found: {resolved}")

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"unable to parse ground truth json: {exc}") from exc

    flags: list[bool] = []

    if isinstance(payload, list):
        if all(isinstance(item, bool) for item in payload):
            flags = [bool(item) for item in payload]
        elif all(isinstance(item, dict) for item in payload):
            indexed: dict[int, bool] = {}
            for item in payload:
                if "frame_number" not in item or "flagged" not in item:
                    continue
                try:
                    frame_number = int(item["frame_number"])
                except (TypeError, ValueError):
                    continue
                indexed[frame_number] = bool(item.get("flagged"))

            if indexed:
                last_index = max(indexed.keys())
                flags = [bool(indexed.get(index, False)) for index in range(last_index + 1)]
    elif isinstance(payload, dict):
        indexed = {}
        for key, value in payload.items():
            try:
                frame_number = int(key)
            except (TypeError, ValueError):
                continue
            indexed[frame_number] = bool(value)

        if indexed:
            last_index = max(indexed.keys())
            flags = [bool(indexed.get(index, False)) for index in range(last_index + 1)]

    if not flags:
        raise ValueError("ground truth file does not contain recognizable frame labels")

    if expected_length is not None and expected_length > 0:
        if len(flags) < expected_length:
            flags.extend([False] * (expected_length - len(flags)))
        elif len(flags) > expected_length:
            flags = flags[:expected_length]

    return flags


def evaluate_video_results(results: list[dict], ground_truth_flags: list[bool]) -> dict:
    """Evaluate video frame predictions against ground-truth labels."""
    if not results:
        raise ValueError("results must not be empty for evaluation")
    if len(results) != len(ground_truth_flags):
        raise ValueError(
            "results and ground_truth_flags must have same length "
            f"(got {len(results)} vs {len(ground_truth_flags)})"
        )

    predicted = [bool(item.get("flagged", False)) for item in results]
    expected = [bool(item) for item in ground_truth_flags]

    tp = sum(1 for y_true, y_pred in zip(expected, predicted) if y_true and y_pred)
    tn = sum(1 for y_true, y_pred in zip(expected, predicted) if (not y_true) and (not y_pred))
    fp = sum(1 for y_true, y_pred in zip(expected, predicted) if (not y_true) and y_pred)
    fn = sum(1 for y_true, y_pred in zip(expected, predicted) if y_true and (not y_pred))

    total = len(expected)
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1_score = _safe_div(2 * precision * recall, precision + recall)

    confusion_matrix = {
        "labels": ["safe", "flagged"],
        "matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return {
        "total_samples": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "support": {
            "safe": sum(1 for item in expected if not item),
            "flagged": sum(1 for item in expected if item),
        },
        "confusion_matrix": confusion_matrix,
    }


def format_video_evaluation_report(evaluation: dict) -> str:
    """Format a human-readable frame-level evaluation report."""
    matrix = evaluation.get("confusion_matrix", {})
    tn = int(matrix.get("tn", 0))
    fp = int(matrix.get("fp", 0))
    fn = int(matrix.get("fn", 0))
    tp = int(matrix.get("tp", 0))

    return "\n".join(
        [
            "",
            "VIDEO ANALYZER EVALUATION (FRAME LEVEL)",
            "-" * 70,
            f"Samples   : {evaluation.get('total_samples', 0)}",
            f"Accuracy  : {evaluation.get('accuracy', 0.0):.4f}",
            f"Precision : {evaluation.get('precision', 0.0):.4f}",
            f"Recall    : {evaluation.get('recall', 0.0):.4f}",
            f"F1-score  : {evaluation.get('f1_score', 0.0):.4f}",
            "",
            "Confusion Matrix (rows=actual, cols=predicted)",
            "                Pred Safe   Pred Flagged",
            f"Actual Safe     {tn:<10}  {fp:<13}",
            f"Actual Flagged  {fn:<10}  {tp:<13}",
            "-" * 70,
        ]
    )


def _resolve_transcript_segment_max_len(explicit_max_len: Optional[int]) -> int:
    """Resolve transcript segment size cap (0 means no size cap)."""
    if explicit_max_len is not None:
        return max(0, int(explicit_max_len))

    raw_value = os.getenv("MWG_TRANSCRIPT_SEGMENT_MAX_CHARS", "0").strip()
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 0


def _split_long_text_by_sentence(text: str, max_len: int) -> list[str]:
    """Split one long block by sentence boundaries with optional hard word wrapping."""
    import re

    normalized = " ".join(text.split())
    if not normalized:
        return []
    if max_len <= 0 or len(normalized) <= max_len:
        return [normalized]

    sentences = [seg.strip() for seg in re.split(r"(?<=[.?!])\s+", normalized) if seg.strip()]
    if not sentences:
        sentences = [normalized]

    chunks: list[str] = []
    current = ""

    def _append_with_word_wrap(chunk_text: str) -> None:
        if not chunk_text:
            return
        if len(chunk_text) <= max_len:
            chunks.append(chunk_text)
            return

        # Last resort hard wrap when punctuation-based splitting cannot reduce size.
        words = chunk_text.split()
        running = ""
        for word in words:
            candidate = f"{running} {word}".strip() if running else word
            if running and len(candidate) > max_len:
                chunks.append(running)
                running = word
            else:
                running = candidate
        if running:
            chunks.append(running)

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_len:
            _append_with_word_wrap(current)
            current = sentence
        else:
            current = candidate

    if current:
        _append_with_word_wrap(current)

    return chunks


def _split_transcript(transcript: str, max_len: Optional[int] = None) -> list[str]:
    """
    Split transcript into sentence-level segments.

    Args:
        transcript: The full transcription text.
        max_len: Optional approximate max characters per segment.
                 If omitted or <= 0, each sentence is kept whole.

    Returns:
        A list of sentence strings.
    """
    import re

    clean_text = str(transcript or "").strip()
    if not clean_text:
        return []

    normalized = " ".join(clean_text.split())
    sentence_candidates = [
        segment.strip()
        for segment in re.split(r"(?<=[.?!])\s+|(?:\r?\n)+", normalized)
        if segment.strip()
    ]
    if not sentence_candidates:
        sentence_candidates = [normalized]

    resolved_max_len = _resolve_transcript_segment_max_len(max_len)
    sentences: list[str] = []
    for sentence in sentence_candidates:
        if resolved_max_len > 0 and len(sentence) > resolved_max_len:
            sentences.extend(_split_long_text_by_sentence(sentence, resolved_max_len))
        else:
            sentences.append(sentence)

    return sentences if sentences else [normalized]


def _split_transcript_with_word_confidence(
    transcript: str, word_data: list[dict], max_len: Optional[int] = None
) -> tuple[list[str], list[float], list[dict]]:
    """
    Split transcript into sentence-level segments and map word confidences.

    Args:
        transcript: Full transcribed text.
        word_data: List of word dicts from Google Speech API.
                   Each dict has: word, confidence, start_time, end_time.
        max_len: Optional max chunk length. If omitted, respects
                 MWG_TRANSCRIPT_SEGMENT_MAX_CHARS environment variable.

    Returns:
        A tuple of (sentences, confidences, segment_meta).
        segment_meta entries include start_time, end_time, confidence.
    """
    chunks = _split_transcript(transcript, max_len=max_len)
    if not chunks:
        return [transcript], [0.8], [{"start_time": None, "end_time": None, "confidence": 0.8}]

    confidences_list = []
    segment_meta: list[dict] = []

    def _consume_chunk_metadata(chunk_text: str) -> tuple[float, Optional[float], Optional[float]]:
        nonlocal current_word_idx
        sent_words = chunk_text.split()
        sent_confidences = []
        seg_start: Optional[float] = None
        seg_end: Optional[float] = None

        for _ in sent_words:
            if current_word_idx >= len(word_data):
                break

            token = word_data[current_word_idx]
            current_word_idx += 1

            token_conf = float(token.get("confidence") or 0.0)
            sent_confidences.append(token_conf)

            token_start = token.get("start_time")
            token_end = token.get("end_time")
            if isinstance(token_start, (int, float)) and seg_start is None:
                seg_start = float(token_start)
            if isinstance(token_end, (int, float)):
                seg_end = float(token_end)

        avg_conf = sum(sent_confidences) / len(sent_confidences) if sent_confidences else 0.8
        return avg_conf, seg_start, seg_end

    current_word_idx = 0
    for chunk in chunks:
        avg_conf, seg_start, seg_end = _consume_chunk_metadata(chunk)
        confidences_list.append(avg_conf)
        segment_meta.append(
            {
                "start_time": seg_start,
                "end_time": seg_end,
                "confidence": avg_conf,
            }
        )

    return chunks, confidences_list, segment_meta


def _split_transcript_with_confidence(
    transcript: str, segments: list[dict], max_len: int = 300
) -> tuple[list[str], list[float]]:
    """
    DEPRECATED: Use _split_transcript_with_word_confidence instead.
    This function is kept for backward compatibility.
    
    Splits transcript into sentences and maps Whisper confidence scores.

    Args:
        transcript: Full transcribed text.
        segments: List of segment dicts from Whisper (each has 'text', 'no_speech_prob').
        max_len: Max chunk length.

    Returns:
        A tuple of (sentences: list[str], confidences: list[float]).
    """
    import re

    # Get sentence splits
    sentences = re.split(r"(?<=[.?!])\s+", transcript.strip())
    chunks = []
    confidences_list = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < max_len:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
                # Default confidence if segments not available
                confidences_list.append(0.8)
            current = sent

    if current:
        chunks.append(current)
        conf = _get_segment_confidence(current, segments)
        confidences_list.append(conf)

    return chunks if chunks else [transcript], confidences_list if confidences_list else [1.0]


def _get_segment_confidence(text: str, segments: list[dict]) -> float:
    """
    Finds matching segment and extracts confidence (1 - no_speech_prob).

    Args:
        text: The segment text to match.
        segments: List of Whisper segment dicts.

    Returns:
        Confidence score (0-1), or 1.0 if no match found.
    """
    for seg in segments:
        seg_text = seg.get("text", "").strip()
        if seg_text and text.strip().startswith(seg_text[:50]):  # Fuzzy match
            no_speech_prob = seg.get("no_speech_prob", 0.0)
            return round(1 - no_speech_prob, 4)
    return 1.0  # Default to high confidence if not found
