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
import uuid
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
from chat_analyzer import (
    analyze_messages,
    evaluate_chat_results,
    format_chat_evaluation_report,
    export_confusion_matrix_html,
)
from video_analyzer import (
    analyze_video,
    extract_audio,
    transcribe_audio_with_fallback,
    evaluate_video_results,
    format_video_evaluation_report,
    get_last_video_analysis_metrics,
    load_video_ground_truth_flags,
)
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
    "I finished practicing for 30 minutes! Can we do the bird song next week?",
    "The new composition assignment looks really fun and creative!",
    "I would like to schedule a lesson for next Tuesday if you have time.",
    "These sheet music arrangements are absolutely beautiful and inspiring.",
    "Could you please explain the music notation rules one more time?",
    "I'm excited to perform at the recital next month!",
    # ── PII: phone number ───────────────────────────────────────────────
    "My mom said you can call her at 555-867-5309 if you need to reschedule.",
    "You can reach me at 1-800-555-0123 for any updates about the concert.",
    "Please contact my guardian at 202-555-0147 before our next session.",
    # ── PII: email address ──────────────────────────────────────────────
    "Please send the sheet music to littlestar2015@gmail.com",
    "My alternate email is musiclover.jane@yahoo.com if you need it.",
    "Forward the assignment details to myname456@hotmail.com please.",
    # ── PII: address ───────────────────────────────────────────────────
    "My address is 123 Maple Street, Springfield, IL 62701 for mail.",
    "We live at 456 Oak Avenue, Brooklyn, New York 11201 if needed.",
    # ── Profanity ───────────────────────────────────────────────────────
    "This damn exercise is so stupid, I hate it!",
    "What the hell is this nonsense? I'm so frustrated with this crap!",
    "This freakin' piece is impossible! I can't get it right, screw this!",
    # ── Grooming-style / manipulative language ──────────────────────────
    "Don't tell anyone about our special friendship — it's just between us, okay?",
    "I think you're very mature for your age. Let's keep our chats private, okay?",
    "Nobody else understands you like I do. You can trust me completely.",
    "Let's meet somewhere private where no one will see us, alright?",
    # ── Toxicity / Mild aggression ──────────────────────────────────────
    "You are the worst teacher ever. Call me on 020 7946 0958 if you dare.",
    "Your teaching methods are terrible and I hate every second of your class!",
    "You're so incompetent. Everyone thinks you're a fraud.",
    # ── Combined: toxicity + sentiment ──────────────────────────────────
    "This is absolutely pathetic. You're wasting everyone's time here.",
    "I despise everything about this lesson. You're the worst!",
    # ── Negative sentiment but safe ─────────────────────────────────────
    "I'm feeling really sad today and don't feel like practicing.",
    "I'm struggling a lot with this new song and feeling discouraged.",
    # ── Safe: questions and learning ────────────────────────────────────
    "How do I improve my sight-reading skills faster?",
    "What are some good practice techniques for beginners?",
    "Can you recommend some famous composers I should study?",
    "What's the best way to memorize complex pieces?",
]

# Ground-truth labels for SAMPLE_MESSAGES (False=safe, True=flagged)
SAMPLE_CHAT_EXPECTED_FLAGS: list[bool] = [
    False,  # Safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
    True,   # PII: phone
    True,   # PII: phone
    True,   # PII: phone
    True,   # PII: email
    True,   # PII: email
    True,   # PII: email
    True,   # PII: address
    True,   # PII: address
    True,   # Profanity
    True,   # Profanity
    True,   # Profanity
    True,   # Grooming
    True,   # Grooming
    True,   # Grooming
    True,   # Grooming
    True,   # Toxicity
    True,   # Toxicity
    True,   # Toxicity
    True,   # Combined
    True,   # Combined
    False,  # Negative but safe
    False,  # Negative but safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
    False,  # Safe
]

VIDEO_PATH = "D:\\319cf974-f244-4575-9fa7-5040020427d5.mp4"

# ─────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────

def run_chat_pipeline(
    messages: list[str],
    run_id: str,
    expected_flags: list[bool] | None = None,
) -> tuple[list[dict], dict | None]:
    """
    Runs the chat analysis pipeline and logs results to the alert engine.

    Args:
        messages: List of raw chat message strings to analyse.

    Returns:
        A tuple of:
          - analysis result dicts for all messages
          - optional evaluation metrics dict
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  PART 1 — CHAT MESSAGE ANALYSIS" + " " * 36 + "║")
    print("╚" + "═" * 68 + "╝\n")

    results = analyze_messages(messages)
    evaluation: dict | None = None

    if expected_flags is not None:
        try:
            evaluation = evaluate_chat_results(results, expected_flags)
            print(format_chat_evaluation_report(evaluation))

            matrix_html = export_confusion_matrix_html(
                evaluation,
                output_path="chat_confusion_matrix.html",
            )
            if matrix_html:
                evaluation["confusion_matrix_html_path"] = matrix_html
                print(f"Confusion matrix HTML: {matrix_html}")
        except ValueError as exc:
            logger.warning(f"Chat evaluation skipped: {exc}")

    # Log ALL results (including safe ones) so dashboard can show totals
    log_alerts(results, source="chat", print_summary=True, run_id=run_id)
    return results, evaluation


def run_video_pipeline(video_path: str, run_id: str) -> tuple[list[dict], list[dict], dict]:
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

    import tempfile

    audio_result = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, "extracted_audio.wav")

        shared_audio: dict = {
            "audio_path": None,
            "transcript": None,
            "word_data": None,
            "transcription_confidence": 0.0,
            "transcription_engine": "none",
        }

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
            logger.warning("Audio extraction failed for shared video/audio analysis.")

        frame_results, transcript_results = analyze_video(video_path, shared_audio=shared_audio)

        if frame_results:
            log_alerts(frame_results, source="video_frame", print_summary=True, run_id=run_id)
        if transcript_results:
            log_alerts(transcript_results, source="transcript", print_summary=True, run_id=run_id)

        print("\n" + "─" * 70)
        print("  AUDIO-LEVEL ANALYSIS")
        print("─" * 70 + "\n")

        if shared_audio.get("audio_path") and os.path.exists(shared_audio["audio_path"]):
            audio_result = analyze_audio_features(shared_audio["audio_path"]) or {}
            if audio_result:
                log_audio_alert(audio_result, print_summary=True, run_id=run_id)
        else:
            logger.warning("Shared audio path unavailable; skipping audio feature analysis.")

    return frame_results, transcript_results, audio_result


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────

def print_final_summary(
    chat_results: list[dict],
    frame_results: list[dict],
    transcript_results: list[dict],
    audio_result: dict | None = None,
    chat_evaluation: dict | None = None,
    video_metrics: dict | None = None,
    video_evaluation: dict | None = None,
    run_id: str | None = None,
) -> None:
    """
    Prints a comprehensive end-of-run summary to the console.

    Args:
        chat_results:       Analysis results from the chat pipeline.
        frame_results:      Frame-level analysis results from video pipeline.
        transcript_results: Transcript analysis results from video pipeline.
        audio_result:       Audio analysis result dict from video pipeline.
        chat_evaluation:    Optional metrics from chat prediction evaluation.
        video_metrics:      Optional runtime/performance metrics from video analyzer.
        video_evaluation:   Optional frame-level quality metrics against ground truth.
    """
    if audio_result is None:
        audio_result = {}
    if video_metrics is None:
        video_metrics = {}

    stats = get_alert_stats(run_id=run_id)

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

    if chat_evaluation:
        print("╠" + "═" * 68 + "╣")
        print("║  CHAT EVALUATION METRICS" + " " * 43 + "║")
        metrics_line = (
            f"  Accuracy: {chat_evaluation.get('accuracy', 0.0):.4f} | "
            f"Precision: {chat_evaluation.get('precision', 0.0):.4f}"
        )
        print(f"║{metrics_line:<69}║")
        metrics_line = (
            f"  Recall: {chat_evaluation.get('recall', 0.0):.4f} | "
            f"F1-score: {chat_evaluation.get('f1_score', 0.0):.4f}"
        )
        print(f"║{metrics_line:<69}║")

    if video_metrics:
        print("╠" + "═" * 68 + "╣")
        print("║  VIDEO PIPELINE PERFORMANCE" + " " * 41 + "║")
        metrics_line = (
            f"  Frame stage: {video_metrics.get('frame_stage_seconds', 0.0):.2f}s | "
            f"Effective FPS: {video_metrics.get('effective_processing_fps', 0.0):.2f}"
        )
        print(f"║{metrics_line:<69}║")
        metrics_line = (
            f"  Avg frame latency: {video_metrics.get('avg_frame_processing_ms', 0.0):.2f}ms | "
            f"Total runtime: {video_metrics.get('total_seconds', 0.0):.2f}s"
        )
        print(f"║{metrics_line:<69}║")
        metrics_line = (
            f"  NSFW infer: {video_metrics.get('nsfw_inference_seconds', 0.0):.2f}s | "
            f"Emotion infer: {video_metrics.get('emotion_inference_seconds', 0.0):.2f}s"
        )
        print(f"║{metrics_line:<69}║")

    if video_evaluation:
        print("╠" + "═" * 68 + "╣")
        print("║  VIDEO EVALUATION METRICS" + " " * 42 + "║")
        metrics_line = (
            f"  Accuracy: {video_evaluation.get('accuracy', 0.0):.4f} | "
            f"Precision: {video_evaluation.get('precision', 0.0):.4f}"
        )
        print(f"║{metrics_line:<69}║")
        metrics_line = (
            f"  Recall: {video_evaluation.get('recall', 0.0):.4f} | "
            f"F1-score: {video_evaluation.get('f1_score', 0.0):.4f}"
        )
        print(f"║{metrics_line:<69}║")

    print("╠" + "═" * 68 + "╣")
    print(f"║  Database       : melodywings_guard.db" + " " * 29 + "║")
    print(f"║  Python log     : melodywings_guard.log" + " " * 27 + "║")
    print(f"║  Dashboard      : python html_dashboard.py" + " " * 19 + "║")
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

    run_id = str(uuid.uuid4())
    logger.info(f"Run ID: {run_id}")

    # Optional hard clear for development-only workflows.
    clear_on_run = os.getenv("MWG_CLEAR_ON_RUN", "false").strip().lower() in {"1", "true", "yes"}
    if clear_on_run:
        clear_alerts()
    else:
        logger.info("Preserving previous runs. Set MWG_CLEAR_ON_RUN=true to clear before execution.")

    # ── Part 1: Chat ─────────────
    chat_results, chat_evaluation = run_chat_pipeline(
        SAMPLE_MESSAGES,
        run_id=run_id,
        expected_flags=SAMPLE_CHAT_EXPECTED_FLAGS,
    )

    # ── Part 2: Video ────────────
    frame_results, transcript_results, audio_result = run_video_pipeline(VIDEO_PATH, run_id=run_id)
    video_metrics = get_last_video_analysis_metrics()
    video_evaluation: dict | None = None

    video_ground_truth_path = os.getenv("MWG_VIDEO_FRAME_GROUND_TRUTH", "").strip()
    if video_ground_truth_path and frame_results:
        try:
            expected_flags = load_video_ground_truth_flags(
                video_ground_truth_path,
                expected_length=len(frame_results),
            )
            video_evaluation = evaluate_video_results(frame_results, expected_flags)
            print(format_video_evaluation_report(video_evaluation))
        except ValueError as exc:
            logger.warning(f"Video evaluation skipped: {exc}")

    # ── Part 4: Summary ──────────
    print_final_summary(
        chat_results,
        frame_results,
        transcript_results,
        audio_result,
        chat_evaluation,
        video_metrics,
        video_evaluation,
        run_id=run_id,
    )

    logger.info("MelodyWings Guard pipeline complete.")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Run HTML dashboard: python html_dashboard.py")
    logger.info("  - View detailed charts and analytics")
    logger.info("  - For setup and usage: read README.md")


if __name__ == "__main__":
    main()
