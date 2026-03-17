"""
Test script for the VoiceCloner module (Phase 1B).

Run with: python scripts/test_tts.py

Two modes:
  1. Mock mode (no GPU, no models) -- verifies class logic
  2. Live mode (GPU + CosyVoice) -- runs real voice synthesis

Mock mode runs by default. For live mode, set LIVESELF_TEST_LIVE=1
and ensure CosyVoice is cloned into engine/models/CosyVoice/.
"""

import os
import sys
import logging

# Add engine to path so we can import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

try:
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy")
    sys.exit(1)

from pipeline.tts import VoiceCloner, OUTPUT_SAMPLE_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_tts")


def test_cloner_init():
    """Verify VoiceCloner initializes without loading models."""
    cloner = VoiceCloner()
    assert not cloner.is_ready, "Should not be ready before load()"
    assert cloner.sample_rate == OUTPUT_SAMPLE_RATE
    logger.info("PASS: cloner init (no model load)")


def test_cloner_errors_before_load():
    """Verify proper errors when calling methods before loading."""
    cloner = VoiceCloner()

    try:
        cloner.synthesize("hello")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: synthesize raises RuntimeError before load()")

    try:
        cloner.set_reference_voice("nonexistent.wav")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: set_reference_voice raises RuntimeError before load()")


def test_unload():
    """Verify unload clears all state."""
    cloner = VoiceCloner()
    cloner.unload()
    assert not cloner.is_ready
    assert cloner._model is None
    assert cloner._reference_audio_path is None
    logger.info("PASS: unload clears state")


def test_live_synthesis():
    """
    Live test: loads CosyVoice and synthesizes speech.
    Only runs when LIVESELF_TEST_LIVE=1 is set.
    Requires: GPU, CosyVoice cloned, reference audio file.
    """
    import soundfile as sf

    test_audio = os.environ.get("LIVESELF_TEST_VOICE")
    if not test_audio:
        logger.error("Set LIVESELF_TEST_VOICE to a WAV file path for live test")
        return False

    if not os.path.exists(test_audio):
        logger.error(f"Reference voice audio not found: {test_audio}")
        return False

    logger.info(f"Running live TTS test with voice: {test_audio}")

    # Initialize and load model
    cloner = VoiceCloner()
    cloner.load()

    # Set reference voice
    cloner.set_reference_voice(test_audio)
    assert cloner.is_ready, "Cloner should be ready after setting reference"

    # Synthesize test text
    test_text = "Hello, I am your AI twin. This is a test of voice cloning."
    audio = cloner.synthesize(test_text)

    assert audio is not None, "Audio should not be None"
    assert len(audio) > 0, "Audio should not be empty"
    assert audio.dtype == np.float32, f"Expected float32, got {audio.dtype}"

    duration = len(audio) / OUTPUT_SAMPLE_RATE
    logger.info(f"Synthesized {duration:.1f}s of audio ({len(audio)} samples)")

    # Save result for listening
    output_path = os.path.join(os.path.dirname(__file__), "test_tts_output.wav")
    sf.write(output_path, audio, OUTPUT_SAMPLE_RATE)
    logger.info(f"Audio saved to: {output_path}")

    # Test streaming mode
    stream_chunks = list(cloner.synthesize_stream("Streaming test."))
    assert len(stream_chunks) > 0, "Should produce at least one chunk"
    total_samples = sum(len(c) for c in stream_chunks)
    logger.info(f"Streaming produced {len(stream_chunks)} chunks, {total_samples} total samples")

    # Clean up
    cloner.unload()
    logger.info("PASS: live TTS test complete")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("VoiceCloner Tests -- Phase 1B")
    logger.info("=" * 60)

    # Mock tests (always run, no GPU needed)
    test_cloner_init()
    test_cloner_errors_before_load()
    test_unload()

    logger.info("-" * 60)
    logger.info("Mock tests: ALL PASSED")
    logger.info("-" * 60)

    # Live test (only if explicitly enabled)
    if os.environ.get("LIVESELF_TEST_LIVE") == "1":
        success = test_live_synthesis()
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
