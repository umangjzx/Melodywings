"""
Interactive Test Script for Chat Analyzer
Run: python interactive_test.py
"""

import chat_analyzer
import json
from typing import Optional


def display_result(result: dict) -> None:
    """Display analysis result in a formatted way."""
    print("\n" + "=" * 60)
    print(f"FLAGGED: {result['flagged']}")
    print(f"REASONS: {', '.join(result['reasons']) if result['reasons'] else 'None'}")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print()


def interactive_test() -> None:
    """Interactive testing loop."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     MelodyWings Guard - Chat Analyzer Interactive Tester     ║
║                                                              ║
║  Test messages for: profanity, PII, toxicity, sentiment     ║
╚══════════════════════════════════════════════════════════════╝
    """)

    while True:
        print("\nOptions:")
        print("  1. Test single message")
        print("  2. Test with confidence score")
        print("  3. Test multiple messages")
        print("  4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            message = input("Enter message to analyze: ").strip()
            if message:
                result = chat_analyzer.analyze_message(message)
                display_result(result)

        elif choice == "2":
            message = input("Enter message to analyze: ").strip()
            confidence_str = input("Enter Whisper confidence (0.0-1.0): ").strip()
            try:
                confidence = float(confidence_str)
                if 0.0 <= confidence <= 1.0:
                    result = chat_analyzer.analyze_message(message, whisper_confidence=confidence)
                    display_result(result)
                else:
                    print("❌ Confidence must be between 0.0 and 1.0")
            except ValueError:
                print("❌ Invalid confidence value")

        elif choice == "3":
            num_messages = input("How many messages to test? ").strip()
            try:
                num = int(num_messages)
                if num > 0:
                    messages = []
                    confidences = []
                    for i in range(num):
                        msg = input(f"Message {i+1}: ").strip()
                        messages.append(msg)
                        conf_str = input(f"Confidence for message {i+1} (0.0-1.0, or press Enter for 0.9): ").strip()
                        try:
                            conf = float(conf_str) if conf_str else 0.9
                            confidences.append(conf)
                        except ValueError:
                            confidences.append(0.9)
                    
                    if messages:
                        results = chat_analyzer.analyze_messages(messages, whisper_confidences=confidences)
                        print("\n" + "=" * 60)
                        print("BATCH ANALYSIS RESULTS")
                        print("=" * 60)
                        for idx, (msg, result) in enumerate(zip(messages, results)):
                            print(f"\n[Message {idx+1}] {msg[:50]}...")
                            print(f"  Flagged: {result['flagged']}")
                            print(f"  Reasons: {', '.join(result['reasons']) if result['reasons'] else 'None'}")
                        print("\n")
                else:
                    print("❌ Please enter a positive number")
            except ValueError:
                print("❌ Invalid number")

        elif choice == "4":
            print("\n👋 Thanks for testing! Goodbye.\n")
            break

        else:
            print("❌ Invalid option. Please select 1-4.")


def quick_test() -> None:
    """Quick test with predefined examples."""
    print("\n" + "=" * 60)
    print("QUICK TEST - Predefined Examples")
    print("=" * 60)

    test_cases = [
        ("This is a clean message", 0.95),
        ("This is damn toxic!", 0.85),
        ("My email is test@example.com", 0.90),
        ("I love this!", 0.95),
        ("very bad and horrible", 0.50),
    ]

    for message, confidence in test_cases:
        print(f"\n📝 Message: {message}")
        print(f"📊 Confidence: {confidence}")
        result = chat_analyzer.analyze_message(message, whisper_confidence=confidence)
        print(f"🚨 Flagged: {result['flagged']}")
        print(f"📋 Reasons: {', '.join(result['reasons']) if result['reasons'] else 'None'}")


if __name__ == "__main__":
    print("\nStarting interactive test...")
    print("Initializing models (this may take a moment)...\n")

    try:
        # Run quick test first
        quick_test()
        
        # Then run interactive loop
        interactive_test()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
