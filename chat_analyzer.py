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


def _get_toxicity_pipeline():
    """
    Lazily loads and caches the HuggingFace toxicity pipeline.
    Returns the pipeline or None if loading fails.
    """
    global _toxicity_pipeline
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
    except Exception as exc:
        logger.error(f"Failed to load toxicity model: {exc}")
        _toxicity_pipeline = None
    return _toxicity_pipeline


def _get_sentiment_pipeline():
    """
    Lazily loads and caches the HuggingFace sentiment pipeline.
    Returns the pipeline or None if loading fails.
    """
    global _sentiment_pipeline
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
    except Exception as exc:
        logger.error(f"Failed to load sentiment model: {exc}")
        _sentiment_pipeline = None
    return _sentiment_pipeline


def _get_nlp_model():
    """
    Lazily loads and caches the spaCy NLP model for NER.
    Returns the model or None if loading fails.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
    try:
        import spacy
        logger.info("Loading spaCy NER model: en_core_web_sm ...")
        _nlp_model = spacy.load("en_core_web_sm")
        logger.info("spaCy NER model loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load spaCy model: {exc}")
        _nlp_model = None
    return _nlp_model


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
    try:
        from better_profanity import profanity
        profanity.load_censor_words()
        return profanity.contains_profanity(text)
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
        results = pipe(text[:512])  # model max input length guard
        # pipeline returns list of lists when top_k=1
        top = results[0][0] if isinstance(results[0], list) else results[0]
        label = top["label"].lower()
        score = round(float(top["score"]), 4)
        return label, score
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
        results = pipe(text[:512])  # model max input length guard
        if results:
            result = results[0]
            label = result["label"].lower()
            score = round(float(result["score"]), 4)
            return label, score
        return None, None
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
    if sent_label:
        sentiment = sent_label.lower()
        sentiment_score = sent_score
        # Flag only EXTREME negative sentiment (>0.98 confidence to avoid false positives)
        # DistilBERT sometimes misclassifies neutral content as negative with high confidence
        if sentiment == "negative" and sent_score and sent_score > 0.98:
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
    msg_preview = result["message"][:80].replace("\n", " ")
    print(
        f"[ALERT] | Type: {reasons_str:<35} | Score: {score_str} | Sentiment: {sentiment_str:<10} | Msg: {msg_preview}"
    )


def analyze_messages(messages: list[str]) -> list[dict]:
    """
    Runs the full safety pipeline on a list of chat messages.

    Args:
        messages: List of raw chat message strings.

    Returns:
        List of analysis result dicts (one per message).
    """
    logger.info(f"Analyzing {len(messages)} chat message(s)...")
    results = []
    for msg in messages:
        result = analyze_message(msg)
        results.append(result)
    flagged_count = sum(1 for r in results if r["flagged"])
    logger.info(f"Chat analysis complete. Flagged: {flagged_count}/{len(messages)}")
    return results
