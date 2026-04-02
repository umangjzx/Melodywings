"""
Unit tests for video_analyzer.

Run:
    python test_video_analyzer.py
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import video_analyzer


class VideoAnalyzerTests(unittest.TestCase):
    def test_merge_chunk_text_removes_overlap(self) -> None:
        first = "hello there this is a shared ending"
        second = "this is a shared ending and new words"

        merged = video_analyzer._merge_chunk_text(first, second)
        self.assertEqual(merged, "hello there this is a shared ending and new words")

    def test_split_transcript_respects_max_len(self) -> None:
        text = "One short sentence. Another short sentence. Final short sentence."
        chunks = video_analyzer._split_transcript(text, max_len=25)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 25)

    def test_evaluate_video_results_metrics(self) -> None:
        results = [
            {"flagged": False},
            {"flagged": True},
            {"flagged": True},
            {"flagged": False},
        ]
        ground_truth = [False, True, False, True]

        evaluation = video_analyzer.evaluate_video_results(results, ground_truth)
        self.assertEqual(evaluation["total_samples"], 4)
        self.assertEqual(evaluation["confusion_matrix"]["tp"], 1)
        self.assertEqual(evaluation["confusion_matrix"]["tn"], 1)
        self.assertEqual(evaluation["confusion_matrix"]["fp"], 1)
        self.assertEqual(evaluation["confusion_matrix"]["fn"], 1)

    def test_load_video_ground_truth_flags_from_dict(self) -> None:
        payload = {"0": False, "1": True, "3": True}

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "gt.json"
            file_path.write_text(json.dumps(payload), encoding="utf-8")

            flags = video_analyzer.load_video_ground_truth_flags(str(file_path), expected_length=5)

        self.assertEqual(flags, [False, True, False, True, False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
