"""
Interactive Test Script for Video Analyzer
Run: python interactive_video_test.py
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import video_analyzer
import chat_analyzer


def display_frame_results(frame_results: list[dict]) -> None:
    """Display frame analysis results in a formatted way."""
    if not frame_results:
        print("  ℹ️  No frame results")
        return

    flagged_frames = [f for f in frame_results if f.get("flagged")]
    print(f"\n  📊 Total frames analyzed: {len(frame_results)}")
    print(f"  🚨 Flagged frames: {len(flagged_frames)}")

    if flagged_frames:
        print("\n  🔍 Flagged frames details:")
        for i, frame in enumerate(flagged_frames[:10]):  # Show first 10 flagged
            print(f"    [{i+1}] Frame {frame.get('frame_number')} @ {frame.get('timestamp'):.2f}s")
            print(f"        Reasons: {', '.join(frame.get('reasons', []))}")
            if "nsfw_score" in frame:
                print(f"        NSFW Score: {frame.get('nsfw_score'):.4f}")
            if "emotion" in frame:
                print(f"        Emotion: {frame.get('emotion')}")

        if len(flagged_frames) > 10:
            print(f"    ... and {len(flagged_frames) - 10} more flagged frames")


def display_transcript_results(transcript_results: list[dict]) -> None:
    """Display transcript analysis results in a formatted way."""
    if not transcript_results:
        print("  ℹ️  No transcript results")
        return

    flagged_segments = [t for t in transcript_results if t.get("flagged")]
    print(f"\n  📝 Total transcript segments: {len(transcript_results)}")
    print(f"  🚨 Flagged segments: {len(flagged_segments)}")

    if flagged_segments:
        print("\n  🔍 Flagged transcript segments:")
        for i, segment in enumerate(flagged_segments[:10]):  # Show first 10 flagged
            print(f"    [{i+1}] {segment.get('text', '')[:60]}...")
            print(f"        Reasons: {', '.join(segment.get('reasons', []))}")

        if len(flagged_segments) > 10:
            print(f"    ... and {len(flagged_segments) - 10} more flagged segments")


def display_full_results(frame_results: list[dict], transcript_results: list[dict]) -> None:
    """Display complete analysis results."""
    print("\n" + "=" * 70)
    print("VIDEO ANALYSIS COMPLETE")
    print("=" * 70)

    print("\n🎬 FRAME ANALYSIS:")
    display_frame_results(frame_results)

    print("\n🎤 TRANSCRIPT ANALYSIS:")
    display_transcript_results(transcript_results)

    # Get metrics
    metrics = video_analyzer.get_last_video_analysis_metrics()
    if metrics:
        print("\n📈 METRICS:")
        print(f"  Processing time: {metrics.get('total_seconds', 0):.2f}s")
        print(f"  Frames processed: {metrics.get('frames_processed', 0)}")
        print(f"  Frames flagged: {metrics.get('frames_flagged', 0)}")

    print("=" * 70 + "\n")


def find_video_files(directory: str = ".") -> list[str]:
    """Find video files in a directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    video_files = []

    try:
        for file in Path(directory).iterdir():
            if file.is_file() and file.suffix.lower() in video_extensions:
                video_files.append(str(file))
    except Exception as e:
        print(f"❌ Error reading directory: {e}")

    return sorted(video_files)


def interactive_video_test() -> None:
    """Interactive testing loop for video analyzer."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║   MelodyWings Guard - Video Analyzer Interactive Tester      ║
║                                                              ║
║  Test videos for: NSFW frames, emotion, transcript safety   ║
╚══════════════════════════════════════════════════════════════╝
    """)

    while True:
        print("\nOptions:")
        print("  1. Test specific video file")
        print("  2. Browse local video files")
        print("  3. Test all videos in directory")
        print("  4. Show last analysis")
        print("  5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == "1":
            video_path = input("Enter video file path: ").strip()

            if not video_path:
                print("❌ Empty path provided")
                continue

            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                continue

            print(f"\n⏳ Analyzing: {video_path}")
            print("   (This may take a few minutes depending on video length)...\n")

            try:
                start_time = time.time()
                frame_results, transcript_results = video_analyzer.analyze_video(video_path)
                elapsed = time.time() - start_time

                display_full_results(frame_results, transcript_results)
                print(f"✅ Analysis completed in {elapsed:.2f}s\n")

            except Exception as e:
                print(f"❌ Error analyzing video: {e}")
                import traceback
                traceback.print_exc()

        elif choice == "2":
            current_dir = input(
                "Enter directory to search (or press Enter for current): "
            ).strip() or "."

            video_files = find_video_files(current_dir)

            if not video_files:
                print(f"❌ No video files found in {current_dir}")
                continue

            print(f"\n📹 Found {len(video_files)} video file(s):\n")
            for idx, file in enumerate(video_files, 1):
                file_size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
                print(f"  {idx}. {file} ({file_size:.2f} MB)")

            try:
                selection = int(input("\nSelect file number (or 0 to cancel): ").strip())
                if 1 <= selection <= len(video_files):
                    selected_video = video_files[selection - 1]

                    confirm = (
                        input(f"\nAnalyze '{selected_video}'? (yes/no): ").strip().lower()
                    )
                    if confirm in {"yes", "y"}:
                        print(f"\n⏳ Analyzing: {selected_video}")
                        print(
                            "   (This may take a few minutes depending on video length)...\n"
                        )

                        try:
                            start_time = time.time()
                            frame_results, transcript_results = (
                                video_analyzer.analyze_video(selected_video)
                            )
                            elapsed = time.time() - start_time

                            display_full_results(frame_results, transcript_results)
                            print(f"✅ Analysis completed in {elapsed:.2f}s\n")

                        except Exception as e:
                            print(f"❌ Error analyzing video: {e}")
                            import traceback

                            traceback.print_exc()

            except ValueError:
                print("❌ Invalid selection")

        elif choice == "3":
            directory = input("Enter directory to scan (or press Enter for current): ").strip() or "."

            video_files = find_video_files(directory)

            if not video_files:
                print(f"❌ No video files found in {directory}")
                continue

            print(f"\n📹 Found {len(video_files)} video file(s) to analyze")
            confirm = input("Analyze all? (yes/no): ").strip().lower()

            if confirm in {"yes", "y"}:
                results_summary = []

                for idx, video_path in enumerate(video_files, 1):
                    print(
                        f"\n[{idx}/{len(video_files)}] ⏳ Analyzing: {video_path}"
                    )

                    try:
                        start_time = time.time()
                        frame_results, transcript_results = (
                            video_analyzer.analyze_video(video_path)
                        )
                        elapsed = time.time() - start_time

                        flagged_frames = sum(1 for f in frame_results if f.get("flagged"))
                        flagged_transcripts = sum(
                            1 for t in transcript_results if t.get("flagged")
                        )

                        summary = {
                            "video": video_path,
                            "time_seconds": elapsed,
                            "frames_total": len(frame_results),
                            "frames_flagged": flagged_frames,
                            "transcript_segments": len(transcript_results),
                            "transcript_flagged": flagged_transcripts,
                            "status": "completed",
                        }

                        results_summary.append(summary)
                        print(f"   ✅ Done ({elapsed:.2f}s)")
                        print(
                            f"      Flagged frames: {flagged_frames}/{len(frame_results)}"
                        )
                        print(
                            f"      Flagged transcripts: {flagged_transcripts}/{len(transcript_results)}"
                        )

                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                        results_summary.append(
                            {"video": video_path, "status": "failed", "error": str(e)}
                        )

                # Display summary
                print("\n" + "=" * 70)
                print("BATCH ANALYSIS SUMMARY")
                print("=" * 70)
                for summary in results_summary:
                    print(f"\n📹 {summary['video']}")
                    if summary["status"] == "completed":
                        print(f"   ⏱️  Time: {summary['time_seconds']:.2f}s")
                        print(
                            f"   🎬 Frames: {summary['frames_flagged']}/{summary['frames_total']} flagged"
                        )
                        print(
                            f"   🎤 Transcript: {summary['transcript_flagged']}/{summary['transcript_segments']} flagged"
                        )
                    else:
                        print(f"   ❌ Status: {summary['status']}")
                        if "error" in summary:
                            print(f"      Error: {summary['error']}")
                print("=" * 70 + "\n")

        elif choice == "4":
            metrics = video_analyzer.get_last_video_analysis_metrics()
            if metrics:
                print("\n" + "=" * 70)
                print("LAST ANALYSIS METRICS")
                print("=" * 70)
                print(json.dumps(metrics, indent=2))
                print("=" * 70 + "\n")
            else:
                print("❌ No previous analysis metrics available")

        elif choice == "5":
            print("\n👋 Thanks for testing! Goodbye.\n")
            break

        else:
            print("❌ Invalid option. Please select 1-5.")


if __name__ == "__main__":
    print("\nStarting interactive video test...")
    print("Initializing models (this may take a moment)...\n")

    try:
        interactive_video_test()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
