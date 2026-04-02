"""
audio_analyzer.py — Audio-Level Analysis Module
MelodyWings Guard | Real-Time Content Safety System

Analyzes audio for:
  1. Volume/decibel trends (detect shouting, raised voices)
  2. Speech rate analysis (rapid speech = agitation)
  3. Silence detection (long pauses)
  4. Speaker change detection (multi-speaker conversations)
  5. Background noise classification
"""

import logging
import numpy as np
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Configuration thresholds
SILENCE_THRESHOLD_DB = -40  # Below this = silence
LOUD_THRESHOLD_DB = -10    # Above this = shouting/raised voice
NORMAL_SPEECH_RATE = 150   # Average words per minute
AGITATION_SPEECH_RATE = 180  # Elevated speech rate


def analyze_audio_features(audio_path: str) -> Optional[dict]:
    """
    Analyzes audio file for volume trends, speech rate, silence, and noise.

    Args:
        audio_path: Path to WAV audio file.

    Returns:
        Dict with keys: volume_stats, silence_periods, speech_rate_estimate,
        background_noise_level, flagged, reasons. Or None on failure.
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed. Skipping audio feature analysis.")
        return None

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # === Volume Analysis ===
        volume_stats = _analyze_volume(y, sr)
        
        # === Silence Detection ===
        silence_periods = _detect_silence(y, sr)
        
        # === Speech Rate Estimation ===
        speech_rate = _estimate_speech_rate(y, sr)
        
        # === Background Noise ===
        noise_level = _estimate_background_noise(y, sr)
        
        # === Speaker Change Detection ===
        speaker_changes = _detect_speaker_changes(y, sr)
        
        # Determine if flagged
        reasons = []
        if volume_stats["max_db"] >= LOUD_THRESHOLD_DB:
            reasons.append(f"shouting(peak={volume_stats['max_db']:.1f}dB)")
        if speech_rate and speech_rate > AGITATION_SPEECH_RATE:
            reasons.append(f"rapid_speech({speech_rate:.0f}wpm)")
        if len(silence_periods) > 3:
            reasons.append(f"multiple_long_pauses({len(silence_periods)})")
        if speaker_changes > 2:
            reasons.append(f"multiple_speakers({speaker_changes})")
        
        flagged = len(reasons) > 0
        
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "volume_stats": volume_stats,
            "silence_periods": silence_periods,
            "speech_rate_wpm": speech_rate,
            "background_noise_db": noise_level,
            "estimated_speakers": speaker_changes,
            "flagged": flagged,
            "reasons": reasons,
        }
        
        if flagged:
            logger.info(f"Audio flagged: {', '.join(reasons)}")
        
        return result
    except Exception as exc:
        logger.error(f"Audio feature analysis failed: {exc}")
        return None


def _analyze_volume(y: np.ndarray, sr: int) -> dict:
    """
    Analyzes volume profile of audio signal.

    Returns:
        Dict with min_db, max_db, mean_db, std_db.
    """
    try:
        import librosa
        
        # Compute RMS energy
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Flatten and compute statistics
        flat = S_db.flatten()
        rms = librosa.feature.rms(y=y)[0]
        rms_db = 20 * np.log10(np.maximum(1e-5, rms))
        
        return {
            "min_db": float(np.min(rms_db)),
            "max_db": float(np.max(rms_db)),
            "mean_db": float(np.mean(rms_db)),
            "std_db": float(np.std(rms_db)),
        }
    except Exception as exc:
        logger.debug(f"Volume analysis failed: {exc}")
        return {
            "min_db": 0.0,
            "max_db": 0.0,
            "mean_db": 0.0,
            "std_db": 0.0,
        }


def _detect_silence(y: np.ndarray, sr: int) -> list[dict]:
    """
    Detects silent periods (>1 second of near-silence).

    Returns:
        List of dicts with start_sec, end_sec, duration_sec.
    """
    try:
        import librosa
        
        # Compute RMS energy per frame
        rms = librosa.feature.rms(y=y)[0]
        rms_db = 20 * np.log10(np.maximum(1e-5, rms))
        
        # Frames below threshold = silence
        silent_frames = rms_db < SILENCE_THRESHOLD_DB
        
        # Group consecutive silent frames
        frame_length = len(y) / len(rms)
        hop_length = frame_length  # approximate
        
        silence_periods = []
        in_silence = False
        start_frame = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_silence:
                in_silence = True
                start_frame = i
            elif not is_silent and in_silence:
                in_silence = False
                start_sec = start_frame * hop_length / sr
                end_sec = i * hop_length / sr
                duration = end_sec - start_sec
                if duration > 1.0:  # Only flag silences > 1 second
                    silence_periods.append({
                        "start_sec": round(start_sec, 2),
                        "end_sec": round(end_sec, 2),
                        "duration_sec": round(duration, 2),
                    })
        
        return silence_periods
    except Exception as exc:
        logger.debug(f"Silence detection failed: {exc}")
        return []


def _estimate_speech_rate(y: np.ndarray, sr: int) -> Optional[float]:
    """
    Estimates speech rate in words per minute.
    Uses onset detection as proxy for syllables.

    Returns:
        Estimated WPM, or None on failure.
    """
    try:
        import librosa
        
        # Detect onsets (syllable-like events)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        num_onsets = len(onset_frames)
        
        # Convert to seconds
        dur_sec = len(y) / sr
        
        # Rough estimate: ~3 onsets per word
        estimated_words = num_onsets / 3.0
        estimated_wpm = (estimated_words / dur_sec) * 60
        
        return max(0, estimated_wpm)
    except Exception as exc:
        logger.debug(f"Speech rate estimation failed: {exc}")
        return None


def _estimate_background_noise(y: np.ndarray, sr: int) -> float:
    """
    Estimates background noise level as the quietest part of the audio.

    Returns:
        Background noise in dB.
    """
    try:
        import librosa
        
        rms = librosa.feature.rms(y=y)[0]
        rms_db = 20 * np.log10(np.maximum(1e-5, rms))
        
        # Background noise = 10th percentile (quietest parts)
        bg_noise = float(np.percentile(rms_db, 10))
        return round(bg_noise, 2)
    except Exception as exc:
        logger.debug(f"Background noise estimation failed: {exc}")
        return 0.0


def _detect_speaker_changes(y: np.ndarray, sr: int) -> int:
    """
    Detects likely number of different speakers using spectral changes.
    (Simple heuristic: count large changes in spectral centroid)

    Returns:
        Estimated number of speakers (minimum 1).
    """
    try:
        import librosa
        
        # Compute spectral centroid (pitch/timbre indicator)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Find significant jumps in spectral centroid
        diffs = np.abs(np.diff(spec_cent))
        threshold = np.mean(diffs) + 2 * np.std(diffs)
        
        # Count large jumps
        changes = np.sum(diffs > threshold)
        
        # Rough estimate: 1 speaker base + 1 per significant change (every 100 changes = 1 speaker)
        estimated_speakers = max(1, int(changes / 100) + 1)
        return min(estimated_speakers, 5)  # Cap at 5
    except Exception as exc:
        logger.debug(f"Speaker detection failed: {exc}")
        return 1
