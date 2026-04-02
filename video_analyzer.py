"""
video_analyzer.py — Part 2: Video Frame + Transcript Safety Analysis
MelodyWings Guard | Real-Time Content Safety System

Analyzes video content for:
  1. NSFW frames  (HuggingFace: Falconsai/nsfw_image_detection)
  2. Emotion detection per frame  (DeepFace)
    3. Transcript toxicity via Whisper (long-form) + fallback SpeechRecognition + chat_analyzer pipeline
"""

import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LAZY MODEL LOADERS
# ─────────────────────────────────────────────
_nsfw_pipeline = None
_whisper_asr_pipeline = None


def _get_nsfw_pipeline():
    """
    Lazily loads and caches the NSFW image classifier pipeline.
    Returns the pipeline or None if loading fails.
    """
    global _nsfw_pipeline
    if _nsfw_pipeline is not None:
        return _nsfw_pipeline
    try:
        from transformers import pipeline
        logger.info("Loading NSFW model: Falconsai/nsfw_image_detection ...")
        _nsfw_pipeline = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
        )
        logger.info("NSFW model loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load NSFW model: {exc}")
        _nsfw_pipeline = None
    return _nsfw_pipeline


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

def extract_frames(video_path: str, fps: int = 1):
    """
    Extracts video frames at the given rate using OpenCV.

    Args:
        video_path: Absolute or relative path to the video file.
        fps:        Target frames per second to sample (default: 1).

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
    interval = max(1, int(video_fps // fps))  # sample every N native frames

    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            timestamp_sec = round(frame_idx / video_fps, 3)
            yield sampled_idx, timestamp_sec, frame
            sampled_idx += 1
        frame_idx += 1

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
        # results is a list of {"label": ..., "score": ...}
        # pick the top label
        top = max(results, key=lambda x: x["score"])
        label = top["label"].lower()
        score = round(float(top["score"]), 4)
        return label, score
    except Exception as exc:
        logger.error(f"NSFW classification failed: {exc}")
        return "unknown", 0.0


# ─────────────────────────────────────────────
# EMOTION DETECTION
# ─────────────────────────────────────────────

FLAGGED_EMOTIONS = {"fear", "disgust", "sad", "angry"}


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
        import cv2

        # DeepFace expects BGR numpy array
        analysis = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        # Returns a list; take the first face
        if isinstance(analysis, list):
            analysis = analysis[0]
        emotion = analysis.get("dominant_emotion", None)
        return emotion.lower() if emotion else None
    except Exception as exc:
        logger.debug(f"Emotion detection skipped or failed: {exc}")
        return None


# ─────────────────────────────────────────────
# PER-FRAME ANALYSIS
# ─────────────────────────────────────────────

NSFW_THRESHOLD = 0.70  # confidence cutoff for NSFW flagging


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
    reasons: list[str] = []

    # NSFW check
    nsfw_label, nsfw_score = classify_nsfw(frame)
    if nsfw_label == "nsfw" and nsfw_score >= NSFW_THRESHOLD:
        reasons.append(f"nsfw:{nsfw_label}(score={nsfw_score})")

    # Emotion check
    emotion = detect_emotion(frame)
    if emotion and emotion in FLAGGED_EMOTIONS:
        reasons.append(f"emotion:{emotion}")

    flagged = len(reasons) > 0

    result = {
        "frame_number": frame_number,
        "timestamp_sec": timestamp_sec,
        "nsfw_label": nsfw_label,
        "nsfw_score": nsfw_score,
        "emotion": emotion or "none",
        "flagged": flagged,
        "reasons": reasons,
    }

    if flagged:
        _print_frame_alert(result)

    return result


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
            words = []
            for chunk in chunks:
                chunk_text = str(chunk.get("text", "")).strip()
                if chunk_text:
                    words.extend(chunk_text.split())
            if words:
                word_data = [{"word": w, "confidence": confidence_estimate} for w in words]
            else:
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

def analyze_video(video_path: str) -> tuple[list[dict], list[dict]]:
    """
    Runs the complete video safety analysis pipeline.

    Performs:
      1. Frame-by-frame NSFW + emotion analysis
            2. Audio extraction + robust full-length transcription + chat safety analysis

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        A tuple of (frame_results, transcript_results):
          - frame_results:      list of per-frame analysis dicts
          - transcript_results: list of chat_analyzer result dicts for transcript
    """
    from chat_analyzer import analyze_message

    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}. Skipping video analysis.")
        return [], []

    logger.info(f"Starting video analysis: {video_path}")

    # ── Frame analysis ──────────────────────────────
    frame_results = []
    for frame_number, timestamp_sec, frame in extract_frames(video_path, fps=1):
        result = analyze_frame(frame_number, timestamp_sec, frame)
        frame_results.append(result)

    logger.info(
        f"Frame analysis done. Total: {len(frame_results)} | "
        f"Flagged: {sum(1 for r in frame_results if r['flagged'])}"
    )

    # ── Transcript analysis ──────
    transcript_results = []
    logger.info("Attempting audio extraction and robust transcription...")
    
    transcript, word_data = extract_audio_and_transcribe(video_path)
    if transcript:
        logger.info(f"TRANSCRIPT EXTRACTED: {transcript}")
        print(f"\n{'='*70}")
        print(f"  TRANSCRIPT TEXT EXTRACTED ({len(transcript)} chars)")
        print(f"{'='*70}")
        print(f"{transcript}")
        print(f"{'='*70}\n")
        
        logger.info("Running chat analysis on transcript ...")
        # Split transcript into sentences and use word confidence
        sentences, confidences = _split_transcript_with_word_confidence(transcript, word_data)
        
        # Analyze each sentence with its confidence
        for sentence, confidence in zip(sentences, confidences):
            result = analyze_message(sentence, whisper_confidence=confidence)
            transcript_results.append(result)
    else:
        logger.warning("Transcript extraction failed. Skipping transcript analysis.")

    return frame_results, transcript_results


def _split_transcript(transcript: str, max_len: int = 300) -> list[str]:
    """
    Splits a long transcript string into manageable sentence-level chunks.

    Args:
        transcript: The full transcription text.
        max_len:    Approximate maximum character length per chunk.

    Returns:
        A list of text chunk strings.
    """
    # Simple sentence splitting: split on ". ", "? ", "! "
    import re
    sentences = re.split(r"(?<=[.?!])\s+", transcript.strip())
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_len:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [transcript]


def _split_transcript_with_word_confidence(
    transcript: str, word_data: list[dict], max_len: int = 300
) -> tuple[list[str], list[float]]:
    """
    Splits transcript into sentences and maps Google Cloud Speech word confidence scores.

    Args:
        transcript: Full transcribed text.
        word_data: List of word dicts from Google Speech API.
                   Each dict has: word, confidence, start_time, end_time.
        max_len: Max chunk length.

    Returns:
        A tuple of (sentences: list[str], confidences: list[float]).
        Confidence = average of word confidence scores in sentence.
    """
    import re

    # Get sentence splits
    sentences = re.split(r"(?<=[.?!])\s+", transcript.strip())
    chunks = []
    confidences_list = []
    current = ""

    # Build position map: word position in transcript
    word_positions = {}
    word_idx = 0
    for word_dict in word_data:
        word = word_dict['word']
        word_positions[word_idx] = word_dict['confidence']
        word_idx += 1

    current_word_idx = 0
    for sent in sentences:
        if len(current) + len(sent) < max_len:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
                # Calculate average confidence from words in current chunk
                sent_words = current.lower().split()
                sent_confidences = []
                for _ in sent_words:
                    if current_word_idx < len(word_data):
                        sent_confidences.append(word_data[current_word_idx]['confidence'])
                        current_word_idx += 1
                
                avg_conf = sum(sent_confidences) / len(sent_confidences) if sent_confidences else 0.5
                confidences_list.append(avg_conf)
            current = sent

    if current:
        chunks.append(current)
        sent_words = current.lower().split()
        sent_confidences = []
        for _ in sent_words:
            if current_word_idx < len(word_data):
                sent_confidences.append(word_data[current_word_idx]['confidence'])
                current_word_idx += 1
        
        avg_conf = sum(sent_confidences) / len(sent_confidences) if sent_confidences else 0.5
        confidences_list.append(avg_conf)

    return chunks if chunks else [transcript], confidences_list if confidences_list else [0.8]


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
