"""
Test script for the LipSyncer module (Phase 1B).

Run with: python scripts/test_lipsync.py

Two modes:
  1. Mock mode (no GPU, no models) -- verifies class logic
  2. Live mode (GPU + MuseTalk) -- runs real lip sync

Mock mode runs by default. For live mode, set LIVESELF_TEST_LIVE=1
and ensure MuseTalk is cloned into engine/models/MuseTalk/.
"""

import os
import sys
import logging

# Add engine to path so we can import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

try:
    import numpy as np
    import cv2
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy opencv-python-headless")
    sys.exit(1)

from pipeline.lipsync import LipSyncer, TARGET_FPS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_lipsync")


def test_syncer_init():
    """Verify LipSyncer initializes without loading models."""
    syncer = LipSyncer()
    assert not syncer.is_ready, "Should not be ready before load()"
    assert syncer.target_fps == TARGET_FPS
    logger.info("PASS: syncer init (no model load)")


def test_syncer_errors_before_load():
    """Verify proper errors when calling methods before loading."""
    syncer = LipSyncer()

    dummy_audio = np.zeros(16000, dtype=np.float32)
    try:
        syncer.sync(dummy_audio)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: sync raises RuntimeError before load()")

    dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    try:
        syncer.set_reference_frame(dummy_frame)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: set_reference_frame raises RuntimeError before load()")


def test_unload():
    """Verify unload clears all state."""
    syncer = LipSyncer()
    syncer.unload()
    assert not syncer.is_ready
    assert syncer._model is None
    assert syncer._reference_frame is None
    logger.info("PASS: unload clears state")


def test_live_lipsync():
    """
    Live test: loads MuseTalk and generates lip-synced frames.
    Only runs when LIVESELF_TEST_LIVE=1 is set.
    Requires: GPU, MuseTalk cloned, reference face image, test audio.
    """
    test_photo = os.environ.get("LIVESELF_TEST_PHOTO")
    test_audio_path = os.environ.get("LIVESELF_TEST_AUDIO")

    if not test_photo:
        logger.error("Set LIVESELF_TEST_PHOTO to a face photo path for live test")
        return False
    if not test_audio_path:
        logger.error("Set LIVESELF_TEST_AUDIO to a WAV file path for live test")
        return False

    import soundfile as sf

    logger.info(f"Running live lip sync test with photo: {test_photo}")

    # Load the reference face frame
    face_frame = cv2.imread(test_photo)
    if face_frame is None:
        logger.error(f"Could not read photo: {test_photo}")
        return False

    # Load test audio
    audio, sr = sf.read(test_audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    logger.info(f"Test audio: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.1f}s)")

    # Initialize and load models
    syncer = LipSyncer()
    syncer.load()

    # Set reference frame
    syncer.set_reference_frame(face_frame)
    assert syncer.is_ready, "Syncer should be ready after setting reference"

    # Generate lip-synced frames
    frames = syncer.sync(audio, sample_rate=sr)
    assert len(frames) > 0, "Should produce at least one frame"

    logger.info(f"Generated {len(frames)} lip-synced frames")

    # Verify frame properties
    for i, frame in enumerate(frames[:3]):
        assert frame is not None, f"Frame {i} is None"
        assert frame.dtype == np.uint8, f"Frame {i} dtype: {frame.dtype}"
        assert len(frame.shape) == 3, f"Frame {i} not 3-channel"

    # Save as video for visual inspection
    output_path = os.path.join(os.path.dirname(__file__), "test_lipsync_output.mp4")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    logger.info(f"Lip-synced video saved to: {output_path}")

    # Clean up
    syncer.unload()
    logger.info("PASS: live lip sync test complete")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("LipSyncer Tests -- Phase 1B")
    logger.info("=" * 60)

    # Mock tests (always run, no GPU needed)
    test_syncer_init()
    test_syncer_errors_before_load()
    test_unload()

    logger.info("-" * 60)
    logger.info("Mock tests: ALL PASSED")
    logger.info("-" * 60)

    # Live test (only if explicitly enabled)
    if os.environ.get("LIVESELF_TEST_LIVE") == "1":
        success = test_live_lipsync()
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
