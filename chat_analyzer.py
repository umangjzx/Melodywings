"""
chat_analyzer.py — Part 1: Chat Message Safety Analysis Pipeline
MelodyWings Guard | Real-Time Content Safety System

Analyzes chat messages for:
  1. Profanity (better-profanity)
  2. PII (regex: phone, email, address)
  3. Toxicity (HuggingFace: unitary/toxic-bert)
  4. Sentiment (distilbert-base-uncased-finetuned-sst-2-english)
  5. Named Entity Recognition (spaCy: en_core_web_sm)
"""

import re
import logging
import os
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Configure module-level logger
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LAZY MODEL LOADERS (singleton)
# ─────────────────────────────────────────────
_toxicity_pipeline = None
_sentiment_pipeline = None
_nlp_model = None
_profanity_engine = None
_model_load_lock = threading.Lock()

_stage_health_status: dict[str, str] = {
    "nlp": "available",
    "toxicity": "unknown",
    "sentiment": "unknown",
    "ner": "unknown",
    "profanity": "unknown",
}


def _read_env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read integer env var safely with lower-bound clamping."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


_MODEL_INPUT_MAX_CHARS = 512
_MODEL_WINDOW_OVERLAP_CHARS = _read_env_int("MWG_MODEL_WINDOW_OVERLAP_CHARS", 96, minimum=0)
_ALERT_PREVIEW_CHARS = _read_env_int("MWG_ALERT_PREVIEW_CHARS", 0, minimum=0)


def _strict_mode_enabled() -> bool:
    """Return True when strict mode is enabled via environment variable."""
    return os.getenv("STRICT", "false").strip().lower() in {"1", "true", "yes"}


def _mark_stage(stage: str, status: str) -> None:
    """Update stage health status and roll up the overall NLP state."""
    _stage_health_status[stage] = status

    stage_values = [_stage_health_status.get(k, "unknown") for k in ("toxicity", "sentiment", "ner", "profanity")]
    if any(v == "unavailable" for v in stage_values):
        _stage_health_status["nlp"] = "unavailable"
    elif any(v == "degraded" for v in stage_values):
        _stage_health_status["nlp"] = "degraded"
    else:
        _stage_health_status["nlp"] = "available"


def get_stage_health_status() -> dict[str, str]:
    """Return a copy of current stage health status map."""
    return dict(_stage_health_status)


def _get_toxicity_pipeline():
    """
    Lazily loads and caches the HuggingFace toxicity pipeline.
    Returns the pipeline or None if loading fails.
    """
    global _toxicity_pipeline
    if _toxicity_pipeline is not None:
        return _toxicity_pipeline

    with _model_load_lock:
        if _toxicity_pipeline is not None:
            return _toxicity_pipeline
        try:
            from transformers import pipeline
            logger.info("Loading toxicity model: unitary/toxic-bert ...")
            _toxicity_pipeline = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=1,
            )
            logger.info("Toxicity model loaded successfully.")
            _mark_stage("toxicity", "available")
        except Exception as exc:
            logger.error(f"Failed to load toxicity model: {exc}")
            _mark_stage("toxicity", "unavailable")
            _toxicity_pipeline = None
            if _strict_mode_enabled():
                raise RuntimeError("STRICT mode: toxicity model failed to load") from exc
    return _toxicity_pipeline


def _get_sentiment_pipeline():
    """
    Lazily loads and caches the HuggingFace sentiment pipeline.
    Returns the pipeline or None if loading fails.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    with _model_load_lock:
        if _sentiment_pipeline is not None:
            return _sentiment_pipeline
        try:
            from transformers import pipeline
            logger.info("Loading sentiment model: distilbert-base-uncased-finetuned-sst-2-english ...")
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
            logger.info("Sentiment model loaded successfully.")
            _mark_stage("sentiment", "available")
        except Exception as exc:
            logger.error(f"Failed to load sentiment model: {exc}")
            _mark_stage("sentiment", "unavailable")
            _sentiment_pipeline = None
            if _strict_mode_enabled():
                raise RuntimeError("STRICT mode: sentiment model failed to load") from exc
    return _sentiment_pipeline


def _get_nlp_model():
    """
    Lazily loads and caches the spaCy NLP model for NER.
    Returns the model or None if loading fails.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    with _model_load_lock:
        if _nlp_model is not None:
            return _nlp_model
        try:
            import spacy
            logger.info("Loading spaCy NER model: en_core_web_sm ...")
            _nlp_model = spacy.load("en_core_web_sm")
            logger.info("spaCy NER model loaded successfully.")
            _mark_stage("ner", "available")
        except Exception as exc:
            logger.error(f"Failed to load spaCy model: {exc}")
            _mark_stage("ner", "unavailable")
            _nlp_model = None
            if _strict_mode_enabled():
                raise RuntimeError("STRICT mode: spaCy NER model failed to load") from exc
    return _nlp_model


def _get_profanity_engine():
    """Lazily loads and caches the profanity engine word list."""
    global _profanity_engine
    if _profanity_engine is not None:
        return _profanity_engine

    with _model_load_lock:
        if _profanity_engine is not None:
            return _profanity_engine
        try:
            from better_profanity import profanity
            profanity.load_censor_words()
            _profanity_engine = profanity
            _mark_stage("profanity", "available")
        except Exception as exc:
            logger.error(f"Failed to load profanity engine: {exc}")
            _mark_stage("profanity", "unavailable")
            _profanity_engine = None
            if _strict_mode_enabled():
                raise RuntimeError("STRICT mode: profanity engine failed to load") from exc
    return _profanity_engine


# ─────────────────────────────────────────────
# PROFANITY CHECK
# ─────────────────────────────────────────────

def check_profanity(text: str) -> bool:
    """
    Detects profanity in `text` using the better-profanity library.

    Args:
        text: The message string to check.

    Returns:
        True if profanity detected, False otherwise.
    """
    engine = _get_profanity_engine()
    if engine is None:
        return False
    try:
        return engine.contains_profanity(text)
    except Exception as exc:
        logger.error(f"Profanity check failed: {exc}")
        return False


# ─────────────────────────────────────────────
# PII DETECTION
# ─────────────────────────────────────────────

# Regex patterns for PII detection
_PHONE_PATTERN = re.compile(
    r"""
    (?:
      \+?1[\s\-.]?                         # optional country code
    )?
    (?:\(?\d{3}\)?[\s\-.]?)                # area code
    \d{3}[\s\-.]?\d{4}                     # local number
    """,
    re.VERBOSE,
)

_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

_ADDRESS_PATTERN = re.compile(
    r"""
    \b\d{1,5}\s+                           # street number
    (?:[A-Z][a-z]+\s+){1,3}               # street name words
    (?:Street|St|Avenue|Ave|Road|Rd|
       Boulevard|Blvd|Lane|Ln|Drive|Dr|
       Way|Court|Ct|Place|Pl)\b            # street suffix
    """,
    re.VERBOSE,
)


def check_pii(text: str) -> list[str]:
    """
    Detects personally identifiable information (PII) in `text`.

    Checks for:
      - Phone numbers
      - Email addresses
      - Physical street addresses

    Args:
        text: The message string to check.

    Returns:
        A list of PII type strings found (e.g. ["phone", "email"]).
    """
    found = []
    try:
        if _PHONE_PATTERN.search(text):
            found.append("phone_number")
        if _EMAIL_PATTERN.search(text):
            found.append("email_address")
        if _ADDRESS_PATTERN.search(text):
            found.append("physical_address")
    except Exception as exc:
        logger.error(f"PII check failed: {exc}")
    return found


def _iter_model_windows(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Yield overlapping text windows so long paragraphs are fully scanned."""
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    overlap = min(max(0, overlap_chars), max_chars - 1)
    windows: list[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        tentative_end = min(start + max_chars, text_len)
        end = tentative_end

        # Prefer ending on whitespace to avoid slicing words in half.
        if tentative_end < text_len:
            split_pos = normalized.rfind(" ", start + 1, tentative_end)
            if split_pos > start:
                end = split_pos

        chunk = normalized[start:end].strip()
        if chunk:
            windows.append(chunk)

        if end >= text_len:
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = start + max_chars
        start = next_start

    return windows if windows else [normalized[:max_chars]]


def _parse_pipeline_prediction(item: object) -> tuple[Optional[str], Optional[float]]:
    """Normalize HF pipeline output into (label, score)."""
    top = item[0] if isinstance(item, list) and item else item
    if not isinstance(top, dict):
        return None, None

    label = str(top.get("label", "")).lower() or None
    score_raw = top.get("score")
    score = round(float(score_raw), 4) if score_raw is not None else None
    return label, score


def _merge_window_predictions(
    predictions: list[tuple[Optional[str], Optional[float]]],
    preferred_label: str,
) -> tuple[Optional[str], Optional[float]]:
    """Pick a stable aggregate prediction across windows."""
    preferred_scores = [
        float(score)
        for label, score in predictions
        if label == preferred_label and score is not None
    ]
    if preferred_scores:
        return preferred_label, round(max(preferred_scores), 4)

    scored = [
        (label, float(score))
        for label, score in predictions
        if label is not None and score is not None
    ]
    if scored:
        label, score = max(scored, key=lambda item: item[1])
        return label, round(score, 4)

    first_label = next((label for label, _ in predictions if label), None)
    return first_label, None


def _format_alert_message(message: str) -> str:
    """Render full message by default; optional preview cap via env var."""
    normalized = " ".join(str(message or "").split())
    if _ALERT_PREVIEW_CHARS > 0 and len(normalized) > _ALERT_PREVIEW_CHARS:
        return normalized[:_ALERT_PREVIEW_CHARS].rstrip() + "..."
    return normalized


# ─────────────────────────────────────────────
# TOXICITY DETECTION
# ─────────────────────────────────────────────

def check_toxicity(text: str) -> tuple[Optional[str], Optional[float]]:
    """
    Runs HuggingFace toxic-bert model on `text`.

    Args:
        text: The message string to classify.

    Returns:
        A (label, score) tuple, or (None, None) if the model is unavailable.
        label is "toxic" or "non_toxic"; score is a float in [0, 1].
    """
    pipe = _get_toxicity_pipeline()
    if pipe is None:
        return None, None
    try:
        windows = _iter_model_windows(
            text,
            max_chars=_MODEL_INPUT_MAX_CHARS,
            overlap_chars=_MODEL_WINDOW_OVERLAP_CHARS,
        )
        if not windows:
            return None, None
        results_raw = pipe(windows if len(windows) > 1 else windows[0])
        results_list = results_raw if isinstance(results_raw, list) else [results_raw]
        predictions = [_parse_pipeline_prediction(item) for item in results_list]
        return _merge_window_predictions(predictions, preferred_label="toxic")
    except Exception as exc:
        logger.error(f"Toxicity inference failed: {exc}")
        return None, None


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

def check_sentiment(text: str) -> tuple[Optional[str], Optional[float]]:
    """
    Analyzes sentiment of text using distilbert.

    Args:
        text: The message string to analyze.

    Returns:
        A (sentiment, score) tuple.
        sentiment: "POSITIVE", "NEGATIVE"
        score: confidence in [0, 1], or (None, None) on failure.
    """
    pipe = _get_sentiment_pipeline()
    if pipe is None:
        return None, None
    try:
        windows = _iter_model_windows(
            text,
            max_chars=_MODEL_INPUT_MAX_CHARS,
            overlap_chars=_MODEL_WINDOW_OVERLAP_CHARS,
        )
        if not windows:
            return None, None
        results_raw = pipe(windows if len(windows) > 1 else windows[0])
        results_list = results_raw if isinstance(results_raw, list) else [results_raw]
        predictions = [_parse_pipeline_prediction(item) for item in results_list]
        return _merge_window_predictions(predictions, preferred_label="negative")
    except Exception as exc:
        logger.debug(f"Sentiment analysis failed: {exc}")
        return None, None


# ─────────────────────────────────────────────
# NAMED ENTITY RECOGNITION
# ─────────────────────────────────────────────

def extract_entities(text: str) -> list[dict]:
    """
    Extracts named entities from text using spaCy.

    Args:
        text: The message string to analyze.

    Returns:
        List of entity dicts with keys: text, label.
        E.g., [{"text": "John", "label": "PERSON"}, ...]
    """
    nlp = _get_nlp_model()
    if nlp is None:
        return []
    try:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        return entities
    except Exception as exc:
        logger.debug(f"NER extraction failed: {exc}")
        return []


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

TOXICITY_THRESHOLD = 0.75  # confidence cutoff for flagging
GROOMING_PHRASES = {
    "special friendship",
    "don't tell anyone",
    "dont tell anyone",
    "our little secret",
    "keep this secret",
    "just between us",
    "no one has to know",
}


def _contains_grooming_phrase(text: str) -> bool:
    """Detect simple grooming/manipulative lexicon patterns."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in GROOMING_PHRASES)


def analyze_message(message: str, whisper_confidence: Optional[float] = None) -> dict:
    """
    Runs all safety checks on a single chat message.

    Args:
        message: The raw chat message string.
        whisper_confidence: Optional Whisper transcription confidence (0-1).

    Returns:
        A dict with keys:
          - message (str)
          - flagged (bool)
          - reasons (list[str])
          - confidence (float | None)  — toxicity score if above threshold
          - sentiment (str | None)     — "positive" or "negative"
          - sentiment_score (float | None)
          - entities (list[dict])      — Named entities found
          - whisper_confidence (float | None) — Transcription confidence
          - timestamp (str)            — ISO 8601 UTC
    """
    reasons: list[str] = []
    confidence: Optional[float] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    entities: list[dict] = []

    # 1. Profanity
    if check_profanity(message):
        reasons.append("profanity")

    # 2. PII
    pii_types = check_pii(message)
    for pii in pii_types:
        reasons.append(f"pii:{pii}")

    # 3. Toxicity
    tox_label, tox_score = check_toxicity(message)
    if tox_label == "toxic" and tox_score is not None and tox_score >= TOXICITY_THRESHOLD:
        reasons.append(f"toxicity:{tox_label}")
        confidence = tox_score
    elif tox_score is not None:
        confidence = tox_score  # still record score even if not flagged

    # 4. Sentiment
    sent_label, sent_score = check_sentiment(message)
    grooming_match = _contains_grooming_phrase(message)
    if grooming_match:
        reasons.append("grooming_pattern")

    if sent_label:
        sentiment = sent_label.lower()
        sentiment_score = sent_score
        # Only escalate strong negative sentiment if corroborated by toxicity or grooming lexicon.
        corroborated_negative = (
            (tox_score is not None and tox_score > 0.4)
            or grooming_match
        )
        if sentiment == "negative" and sent_score and sent_score > 0.98 and corroborated_negative:
            reasons.append(f"strong_negative_sentiment")

    # 5. Named Entity Recognition
    entities = extract_entities(message)
    if entities:
        ent_labels = [ent["label"] for ent in entities]
        logger.debug(f"Entities found: {ent_labels}")

    # Check for low transcription confidence
    if whisper_confidence is not None and whisper_confidence < 0.5:
        reasons.append("low_transcription_confidence")

    flagged = len(reasons) > 0
    timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "message": message,
        "flagged": flagged,
        "reasons": reasons,
        "confidence": confidence,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "entities": entities,
        "health_status": get_stage_health_status(),
        "whisper_confidence": whisper_confidence,
        "timestamp": timestamp,
    }

    if flagged:
        _print_alert(result)

    return result


def _print_alert(result: dict) -> None:
    """
    Prints a formatted alert line to stdout for flagged messages.

    Args:
        result: The analysis result dict from analyze_message().
    """
    reasons_str = ", ".join(result["reasons"])
    score_str = f"{result['confidence']:.4f}" if result["confidence"] is not None else "N/A"
    sentiment_str = f"({result.get('sentiment', 'N/A')})" if result.get('sentiment') else ""
    msg_preview = _format_alert_message(str(result.get("message", "")))
    print(
        f"[ALERT] | Type: {reasons_str:<35} | Score: {score_str} | Sentiment: {sentiment_str:<10} | Msg: {msg_preview}"
    )


def _batch_check_toxicity(messages: list[str]) -> list[tuple[Optional[str], Optional[float]]]:
    """Run toxicity checks with full long-text coverage per message."""
    if not messages:
        return []
    return [check_toxicity(message) for message in messages]


def _batch_check_sentiment(messages: list[str]) -> list[tuple[Optional[str], Optional[float]]]:
    """Run sentiment checks with full long-text coverage per message."""
    if not messages:
        return []
    return [check_sentiment(message) for message in messages]


def _batch_extract_entities(messages: list[str]) -> list[list[dict]]:
    """Run spaCy NER over a message batch using nlp.pipe for throughput."""
    if not messages:
        return []
    nlp = _get_nlp_model()
    if nlp is None:
        return [[] for _ in messages]

    try:
        entities_by_message: list[list[dict]] = []
        for doc in nlp.pipe(messages, batch_size=32):
            entities_by_message.append(
                [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                    for ent in doc.ents
                ]
            )
        if len(entities_by_message) < len(messages):
            entities_by_message.extend([[] for _ in range(len(messages) - len(entities_by_message))])
        return entities_by_message[: len(messages)]
    except Exception as exc:
        logger.debug(f"Batch NER extraction failed: {exc}")
        return [[] for _ in messages]


def analyze_messages(
    messages: list[str],
    whisper_confidences: Optional[list[Optional[float]]] = None,
) -> list[dict]:
    """
    Runs the full safety pipeline on a list of chat messages.

    Args:
        messages: List of raw chat message strings.

    Returns:
        List of analysis result dicts (one per message).
    """
    logger.info(f"Analyzing {len(messages)} chat message(s)...")

    if not messages:
        return []

    normalized_confidences: list[Optional[float]]
    if whisper_confidences is None:
        normalized_confidences = [None] * len(messages)
    else:
        normalized_confidences = list(whisper_confidences[: len(messages)])
        if len(normalized_confidences) < len(messages):
            normalized_confidences.extend([None] * (len(messages) - len(normalized_confidences)))

    toxicity_results = _batch_check_toxicity(messages)
    sentiment_results = _batch_check_sentiment(messages)
    entities_results = _batch_extract_entities(messages)

    results = []
    for idx, msg in enumerate(messages):
        reasons: list[str] = []
        confidence: Optional[float] = None
        sentiment: Optional[str] = None
        sentiment_score: Optional[float] = None

        if check_profanity(msg):
            reasons.append("profanity")

        pii_types = check_pii(msg)
        for pii in pii_types:
            reasons.append(f"pii:{pii}")

        tox_label, tox_score = toxicity_results[idx]
        if tox_label == "toxic" and tox_score is not None and tox_score >= TOXICITY_THRESHOLD:
            reasons.append(f"toxicity:{tox_label}")
            confidence = tox_score
        elif tox_score is not None:
            confidence = tox_score

        sent_label, sent_score = sentiment_results[idx]
        grooming_match = _contains_grooming_phrase(msg)
        if grooming_match:
            reasons.append("grooming_pattern")

        if sent_label:
            sentiment = sent_label.lower()
            sentiment_score = sent_score
            corroborated_negative = (
                (tox_score is not None and tox_score > 0.4)
                or grooming_match
            )
            if sentiment == "negative" and sent_score and sent_score > 0.98 and corroborated_negative:
                reasons.append("strong_negative_sentiment")

        entities = entities_results[idx]
        if entities:
            ent_labels = [ent["label"] for ent in entities]
            logger.debug(f"Entities found: {ent_labels}")

        whisper_confidence = normalized_confidences[idx]
        if whisper_confidence is not None and whisper_confidence < 0.5:
            reasons.append("low_transcription_confidence")

        result = {
            "message": msg,
            "flagged": len(reasons) > 0,
            "reasons": reasons,
            "confidence": confidence,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "entities": entities,
            "health_status": get_stage_health_status(),
            "whisper_confidence": whisper_confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if result["flagged"]:
            _print_alert(result)

        results.append(result)

    flagged_count = sum(1 for r in results if r["flagged"])
    logger.info(f"Chat analysis complete. Flagged: {flagged_count}/{len(messages)}")
    return results


# ─────────────────────────────────────────────
# EVALUATION METRICS + CONFUSION MATRIX
# ─────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator while avoiding division-by-zero errors."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_chat_results(
    results: list[dict],
    ground_truth_flags: list[bool],
) -> dict:
    """
    Evaluate chat analyzer predictions against ground truth labels.

    Args:
        results: Chat analysis results (output of analyze_messages).
        ground_truth_flags: Expected flagged labels aligned by index.

    Returns:
        Dict containing accuracy, precision, recall, f1_score, confusion matrix,
        and support counts.

    Raises:
        ValueError: If inputs are empty or lengths do not match.
    """
    if not results:
        raise ValueError("results must not be empty for evaluation")

    if len(results) != len(ground_truth_flags):
        raise ValueError(
            "results and ground_truth_flags must have the same length "
            f"(got {len(results)} vs {len(ground_truth_flags)})"
        )

    predicted = [bool(item.get("flagged", False)) for item in results]
    expected = [bool(label) for label in ground_truth_flags]

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


def format_chat_evaluation_report(evaluation: dict) -> str:
    """Format a human-readable evaluation report with confusion matrix."""
    matrix = evaluation.get("confusion_matrix", {})
    tn = int(matrix.get("tn", 0))
    fp = int(matrix.get("fp", 0))
    fn = int(matrix.get("fn", 0))
    tp = int(matrix.get("tp", 0))

    return "\n".join(
        [
            "",
            "CHAT ANALYZER EVALUATION",
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


def export_confusion_matrix_html(
    evaluation: dict,
    output_path: str,
    title: str = "Chat Analyzer Confusion Matrix",
) -> Optional[str]:
    """
    Export confusion matrix as an interactive HTML heatmap.

    Args:
        evaluation: Output from evaluate_chat_results.
        output_path: Destination HTML path.
        title: Chart title.

    Returns:
        Resolved file path if written successfully; otherwise None.
    """
    matrix = evaluation.get("confusion_matrix", {}).get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 2:
        logger.error("Invalid confusion matrix payload for HTML export.")
        return None

    try:
        import plotly.graph_objects as go
    except Exception as exc:
        logger.error(f"Plotly is unavailable, cannot export confusion matrix HTML: {exc}")
        return None

    try:
        resolved = Path(output_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted Safe", "Predicted Flagged"],
                y=["Actual Safe", "Actual Flagged"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                reversescale=False,
                showscale=True,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            template="plotly_white",
        )
        fig.write_html(str(resolved), include_plotlyjs="cdn", full_html=True)
        return str(resolved)
    except Exception as exc:
        logger.error(f"Failed to export confusion matrix HTML: {exc}")
        return None
