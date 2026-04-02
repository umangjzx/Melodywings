"""
Unit tests for chat_analyzer.

Run:
    python test_chat_analyzer.py
"""

import unittest
from unittest.mock import patch

import chat_analyzer


class ChatAnalyzerTests(unittest.TestCase):
    def test_analyze_message_flags_expected_reasons(self) -> None:
        with patch.object(chat_analyzer, "check_profanity", return_value=True), patch.object(
            chat_analyzer, "check_pii", return_value=["email_address"]
        ), patch.object(chat_analyzer, "check_toxicity", return_value=("toxic", 0.91)), patch.object(
            chat_analyzer, "check_sentiment", return_value=("negative", 0.99)
        ), patch.object(chat_analyzer, "extract_entities", return_value=[]), patch.object(
            chat_analyzer, "_contains_grooming_phrase", return_value=False
        ):
            result = chat_analyzer.analyze_message("test sentence", whisper_confidence=0.95)

        self.assertTrue(result["flagged"])
        self.assertIn("profanity", result["reasons"])
        self.assertIn("pii:email_address", result["reasons"])
        self.assertIn("toxicity:toxic", result["reasons"])
        self.assertIn("strong_negative_sentiment", result["reasons"])

    def test_analyze_message_marks_low_transcription_confidence(self) -> None:
        with patch.object(chat_analyzer, "check_profanity", return_value=False), patch.object(
            chat_analyzer, "check_pii", return_value=[]
        ), patch.object(chat_analyzer, "check_toxicity", return_value=("non_toxic", 0.02)), patch.object(
            chat_analyzer, "check_sentiment", return_value=("positive", 0.99)
        ), patch.object(chat_analyzer, "extract_entities", return_value=[]), patch.object(
            chat_analyzer, "_contains_grooming_phrase", return_value=False
        ):
            result = chat_analyzer.analyze_message("clean sentence", whisper_confidence=0.2)

        self.assertTrue(result["flagged"])
        self.assertIn("low_transcription_confidence", result["reasons"])

    def test_analyze_messages_processes_each_message(self) -> None:
        messages = [
            "This is a damn toxic line.",
            "This is completely safe.",
        ]

        def fake_profanity(text: str) -> bool:
            return "damn" in text.lower()

        with patch.object(chat_analyzer, "check_profanity", side_effect=fake_profanity), patch.object(
            chat_analyzer, "check_pii", return_value=[]
        ), patch.object(
            chat_analyzer,
            "_batch_check_toxicity",
            return_value=[("toxic", 0.9), ("non_toxic", 0.01)],
        ), patch.object(
            chat_analyzer,
            "_batch_check_sentiment",
            return_value=[("negative", 0.99), ("positive", 0.99)],
        ), patch.object(chat_analyzer, "_batch_extract_entities", return_value=[[], []]), patch.object(
            chat_analyzer, "_contains_grooming_phrase", return_value=False
        ):
            results = chat_analyzer.analyze_messages(messages, whisper_confidences=[0.9, 0.9])

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["flagged"])
        self.assertIn("toxicity:toxic", results[0]["reasons"])
        self.assertFalse(results[1]["flagged"])

    def test_iter_model_windows_covers_full_text(self) -> None:
        long_text = " ".join(f"token{i}" for i in range(220))
        windows = chat_analyzer._iter_model_windows(long_text, max_chars=60, overlap_chars=10)

        self.assertGreater(len(windows), 1)
        self.assertIn("token0", windows[0])
        self.assertIn("token219", windows[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
