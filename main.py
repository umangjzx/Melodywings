"""
main.py — Entry Point: MelodyWings Guard Full Pipeline
MelodyWings Guard | Real-Time Content Safety System

Runs:
  1. Chat analysis on a set of sample messages
  2. Video analysis (if test_video.mp4 exists)
  3. Logs all results through the alert engine
  4. Prints a final summary report
"""

import logging
import os
import sys
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# LOGGING SETUP  (must be first)
# ─────────────────────────────────────────────
# Force UTF-8 encoding for Windows compatibility
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("melodywings_guard.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
from database import init_db, get_db
from chat_analyzer import analyze_messages
from video_analyzer import analyze_video
from audio_analyzer import analyze_audio_features
from alert_engine import log_alerts, log_audio_alert, log_transcript_alert, get_alert_stats, clear_alerts

# ─────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────

SAMPLE_MESSAGES: list[str] = [
    # ── Safe messages ───────────────────────────────────────────────────
    "Hey! Can you help me with today's music theory homework?",
    "I really loved the butterfly song we learned in class today 🦋",
    "What time does the online session start tomorrow?",
    "Thank you for helping me practice the piano scales!",
    # ── PII: phone number ───────────────────────────────────────────────
    "My mom said you can call her at 555-867-5309 if you need to reschedule.",
    # ── PII: email address ──────────────────────────────────────────────
    "Please send the sheet music to littlestar2015@gmail.com",
    # ── Profanity ───────────────────────────────────────────────────────
    "This damn exercise is so stupid, I hate it!",
    # ── Grooming-style / manipulative language ──────────────────────────
    "Don't tell anyone about our special friendship — it's just between us, okay?",
    # ── Combined: mild toxicity + partial PII ───────────────────────────
    "You are the worst teacher ever. Call me on 020 7946 0958 if you dare.",
    # ── Safe ────────────────────────────────────────────────────────────
    "I finished practicing for 30 minutes! Can we do the bird song next week?",
]

VIDEO_PATH = "D:\\319cf974-f244-4575-9fa7-5040020427d5.mp4"

# ─────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────

def run_chat_pipeline(messages: list[str]) -> list[dict]:
    """
    Runs the chat analysis pipeline and logs results to the alert engine.

    Args:
        messages: List of raw chat message strings to analyse.

    Returns:
        List of analysis result dicts for all messages.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  PART 1 — CHAT MESSAGE ANALYSIS" + " " * 36 + "║")
    print("╚" + "═" * 68 + "╝\n")

    results = analyze_messages(messages)

    # Log ALL results (including safe ones) so dashboard can show totals
    log_alerts(results, source="chat", print_summary=True)
    return results


def run_video_pipeline(video_path: str) -> tuple[list[dict], list[dict], dict]:
    """
    Runs the video analysis pipeline if the video file exists.
    Includes: frame analysis, transcript analysis, and audio analysis.

    Args:
        video_path: Path to the video file.

    Returns:
        A (frame_results, transcript_results, audio_result) tuple.
        Frame and transcript are empty lists if file not found.
        audio_result is a dict or empty dict.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  PART 2 — VIDEO ANALYSIS (FRAMES + TRANSCRIPT + AUDIO)" + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝\n")

    if not os.path.exists(video_path):
        print(f"  ⚠  Video file not found at '{video_path}'. Skipping video analysis.")
        print("     To enable video analysis, place a file named 'test_video.mp4'")
        print("     in the same directory as main.py and re-run.\n")
        logger.warning(f"Video file missing: {video_path}")
        return [], [], {}

    frame_results, transcript_results = analyze_video(video_path)

    if frame_results:
        log_alerts(frame_results, source="video_frame", print_summary=True)
    if transcript_results:
        log_alerts(transcript_results, source="transcript", print_summary=True)

    # Audio analysis
    print("\n" + "─" * 70)
    print("  AUDIO-LEVEL ANALYSIS")
    print("─" * 70 + "\n")
    
    import tempfile
    audio_result = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        from video_analyzer import extract_audio
        audio_path = os.path.join(tmp_dir, "extracted_audio.wav")
        if extract_audio(video_path, audio_path):
            audio_result = analyze_audio_features(audio_path) or {}
            if audio_result:
                log_audio_alert(audio_result, print_summary=True)
        else:
            logger.warning("Audio extraction failed for audio analysis.")

    return frame_results, transcript_results, audio_result


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────

def print_final_summary(
    chat_results: list[dict],
    frame_results: list[dict],
    transcript_results: list[dict],
    audio_result: dict | None = None,
) -> None:
    """
    Prints a comprehensive end-of-run summary to the console.

    Args:
        chat_results:       Analysis results from the chat pipeline.
        frame_results:      Frame-level analysis results from video pipeline.
        transcript_results: Transcript analysis results from video pipeline.
        audio_result:       Audio analysis result dict from video pipeline.
    """
    if audio_result is None:
        audio_result = {}

    stats = get_alert_stats()

    chat_flagged = sum(1 for r in chat_results if r.get("flagged"))
    frame_flagged = sum(1 for r in frame_results if r.get("flagged"))
    transcript_flagged = sum(1 for r in transcript_results if r.get("flagged"))
    audio_flagged = 1 if audio_result.get("flagged") else 0

    total_scanned = len(chat_results) + len(frame_results) + len(transcript_results) + (1 if audio_result else 0)
    total_flagged = chat_flagged + frame_flagged + transcript_flagged + audio_flagged

    timestamp = datetime.now(timezone.utc).isoformat()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  MELODYWINGS GUARD — FINAL SAFETY REPORT" + " " * 27 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Run Timestamp : {timestamp:<51}║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Total Scanned : {total_scanned:<51}║")
    print(f"║  Total Flagged : {total_flagged:<51}║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Chat Messages    — Scanned: {len(chat_results):<5} Flagged: {chat_flagged:<20}║")
    print(f"║  Video Frames     — Scanned: {len(frame_results):<5} Flagged: {frame_flagged:<20}║")
    print(f"║  Transcript Segs  — Scanned: {len(transcript_results):<5} Flagged: {transcript_flagged:<20}║")
    print(f"║  Audio Features   — Scanned: {1 if audio_result else 0:<5} Flagged: {audio_flagged:<20}║")

    if stats.get("by_severity"):
        print("╠" + "═" * 68 + "╣")
        print("║  BREAKDOWN BY SEVERITY" + " " * 45 + "║")
        for severity, count in sorted(
            stats.get("by_severity", {}).items(), key=lambda kv: ["low", "medium", "high", "critical"].index(kv[0])
        ):
            line = f"  {severity.upper():<35} → {count} alert(s)"
            print(f"║{line:<69}║")

    if stats.get("by_source"):
        print("╠" + "═" * 68 + "╣")
        print("║  BREAKDOWN BY SOURCE" + " " * 47 + "║")
        for source, count in sorted(
            stats.get("by_source", {}).items(), key=lambda kv: kv[1], reverse=True
        ):
            line = f"  {source:<35} → {count} alert(s)"
            print(f"║{line:<69}║")

    print("╠" + "═" * 68 + "╣")
    print(f"║  Database       : melodywings_guard.db" + " " * 29 + "║")
    print(f"║  Python log     : melodywings_guard.log" + " " * 27 + "║")
    print(f"║  Dashboard      : streamlit run dashboard.py" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    """
    Main entry point for the MelodyWings Guard content safety pipeline.
    Orchestrates all analysis stages and prints a final summary report.
    """
    # Initialize database
    init_db()
    logger.info("Database initialized.")

    logger.info("═" * 60)
    logger.info("MelodyWings Guard — Starting Content Safety Pipeline")
    logger.info("═" * 60)

    # Fresh run: optionally clear old alerts
    # (comment out to accumulate alerts across runs)
    clear_alerts()

    # ── Part 1: Chat ─────────────
    chat_results = run_chat_pipeline(SAMPLE_MESSAGES)

    # ── Part 2: Video ────────────
    frame_results, transcript_results, audio_result = run_video_pipeline(VIDEO_PATH)

    # ── Part 4: Summary ──────────
    print_final_summary(chat_results, frame_results, transcript_results, audio_result)

    logger.info("MelodyWings Guard pipeline complete.")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Run the dashboard: streamlit run dashboard.py")
    logger.info("  - View detailed charts and analytics")
    logger.info("  - For setup and usage: read README.md")


if __name__ == "__main__":
    main()
