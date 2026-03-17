"""
Test script for the SpeechRecognizer module (Phase 1C).

Run with: python scripts/test_asr.py

Two modes:
  1. Mock mode (no GPU, no models) -- verifies class logic with fake data
  2. Live mode (GPU + models) -- runs real transcription on audio

Mock mode runs by default. For live mode, set LIVESELF_TEST_LIVE=1.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

try:
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy")
    sys.exit(1)

from pipeline.asr import SpeechRecognizer, SAMPLE_RATE, CHUNK_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_asr")


def test_init():
    """Verify SpeechRecognizer initializes without loading models."""
    recognizer = SpeechRecognizer()
    assert not recognizer.is_ready, "Should not be ready before load()"
    logger.info("PASS: recognizer init (no model load)")


def test_errors_before_load():
    """Verify proper errors when calling methods before loading."""
    recognizer = SpeechRecognizer()
    dummy_audio = np.zeros(CHUNK_SIZE, dtype=np.float32)

    try:
        recognizer.transcribe_segment(dummy_audio)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: transcribe_segment raises RuntimeError before load()")

    try:
        recognizer.feed_audio(dummy_audio)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: feed_audio raises RuntimeError before load()")


def test_utterance_queue():
    """Verify the utterance queue works correctly."""
    recognizer = SpeechRecognizer()

    assert not recognizer.has_complete_utterance()
    assert recognizer.get_utterance() is None

    # Manually add an utterance to test queue
    recognizer._utterance_queue.append("hello world")
    assert recognizer.has_complete_utterance()
    text = recognizer.get_utterance()
    assert text == "hello world"
    assert not recognizer.has_complete_utterance()
    logger.info("PASS: utterance queue")


def test_reset():
    """Verify reset clears all state."""
    recognizer = SpeechRecognizer()
    recognizer._utterance_queue.append("test")
    recognizer._speech_buffer = np.ones(100, dtype=np.float32)
    recognizer._in_speech = True

    recognizer.reset()

    assert not recognizer.has_complete_utterance()
    assert len(recognizer._speech_buffer) == 0
    assert not recognizer._in_speech
    logger.info("PASS: reset clears state")


def test_unload():
    """Verify unload clears model and state."""
    recognizer = SpeechRecognizer()
    recognizer._utterance_queue.append("test")
    recognizer.unload()

    assert not recognizer.is_ready
    assert recognizer._model is None
    assert not recognizer.has_complete_utterance()
    logger.info("PASS: unload clears everything")


def test_speech_detection():
    """Verify the energy-based VAD pre-filter."""
    recognizer = SpeechRecognizer()

    # Silence (very low energy)
    silence = np.zeros(CHUNK_SIZE, dtype=np.float32)
    assert not recognizer._detect_speech(silence), "Silence should not be speech"

    # Loud signal (high energy)
    loud = np.ones(CHUNK_SIZE, dtype=np.float32) * 0.5
    assert recognizer._detect_speech(loud), "Loud signal should be speech"

    # Quiet noise (below threshold)
    quiet = np.random.randn(CHUNK_SIZE).astype(np.float32) * 0.001
    assert not recognizer._detect_speech(quiet), "Quiet noise should not be speech"

    logger.info("PASS: speech detection (energy VAD)")


def test_live_transcription():
    """
    Live test: loads real model and transcribes audio.
    Only runs when LIVESELF_TEST_LIVE=1 is set.
    """
    logger.info("Running live ASR test...")

    recognizer = SpeechRecognizer(model_size="tiny")  # Use tiny for speed
    start = time.perf_counter()
    recognizer.load()
    logger.info(f"Model load time: {(time.perf_counter() - start) * 1000:.0f}ms")

    assert recognizer.is_ready

    # Generate a simple test signal (sine wave, not real speech)
    # This tests that the pipeline doesn't crash, not that transcription is accurate
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    test_audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz tone

    start = time.perf_counter()
    text = recognizer.transcribe_segment(test_audio)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Transcription time: {elapsed_ms:.0f}ms, result: '{text}'")

    # A sine wave won't produce meaningful text, but it should not crash
    assert isinstance(text, str), "Result should be a string"

    recognizer.unload()
    logger.info("PASS: live transcription test complete")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("SpeechRecognizer Tests -- Phase 1C")
    logger.info("=" * 60)

    test_init()
    test_errors_before_load()
    test_utterance_queue()
    test_reset()
    test_unload()
    test_speech_detection()

    logger.info("-" * 60)
    logger.info("Mock tests: ALL PASSED")
    logger.info("-" * 60)

    if os.environ.get("LIVESELF_TEST_LIVE") == "1":
        success = test_live_transcription()
        if success:
            logger.info("Live tests: ALL PASSED")
        else:
            logger.error("Live tests: FAILED")
            sys.exit(1)
    else:
        logger.info("Skipping live test (set LIVESELF_TEST_LIVE=1 to enable)")

    logger.info("=" * 60)
    logger.info("All tests passed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
