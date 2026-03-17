"""
Test script for the FaceSwapper module (Phase 1A).

Run with: python scripts/test_faceswap.py

Two modes:
  1. Mock mode (no GPU, no models) -- verifies class logic with fake data
  2. Live mode (GPU + models) -- runs real face swap on a test image

Mock mode runs by default. For live mode, set LIVESELF_TEST_LIVE=1
and ensure inswapper_128.onnx is in engine/models/.
"""

import os
import sys
import time
import logging

# Add engine to path so we can import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

# Check for required dependencies before importing
try:
    import numpy as np
    import cv2
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy opencv-python-headless")
    print("(These are GPU-machine dependencies -- see engine/requirements.txt)")
    sys.exit(1)

from pipeline.faceswap import FaceSwapper, _get_face_area, DEFAULT_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_faceswap")


def test_face_area_calculation():
    """Verify the face area helper computes bbox area correctly."""

    class FakeFace:
        def __init__(self, bbox):
            self.bbox = bbox

    face = FakeFace(bbox=[100, 100, 300, 400])
    area = _get_face_area(face)
    expected = 200 * 300  # width=200, height=300
    assert area == expected, f"Expected {expected}, got {area}"
    logger.info("PASS: face area calculation")


def test_largest_face_selection():
    """Verify that the largest face is selected from a list."""

    class FakeFace:
        def __init__(self, bbox):
            self.bbox = bbox

    faces = [
        FakeFace(bbox=[0, 0, 50, 50]),     # area = 2500
        FakeFace(bbox=[0, 0, 200, 200]),   # area = 40000 (largest)
        FakeFace(bbox=[0, 0, 100, 100]),   # area = 10000
    ]

    largest = max(faces, key=_get_face_area)
    assert _get_face_area(largest) == 40000, "Should pick the 200x200 face"
    logger.info("PASS: largest face selection")


def test_swapper_init():
    """Verify FaceSwapper initializes without loading models."""
    swapper = FaceSwapper()
    assert not swapper.is_ready, "Should not be ready before load()"
    assert not swapper._is_loaded, "Models should not be loaded yet"
    logger.info("PASS: swapper init (no model load)")


def test_swapper_errors_before_load():
    """Verify proper errors when calling methods before loading."""
    swapper = FaceSwapper()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        swapper.swap_face(dummy_frame)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: swap_face raises RuntimeError before load()")

    try:
        swapper.load_reference("nonexistent.jpg")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: load_reference raises RuntimeError before load()")


def test_unload():
    """Verify unload clears all state."""
    swapper = FaceSwapper()
    swapper.unload()
    assert not swapper.is_ready
    assert swapper._face_analyser is None
    assert swapper._swapper_model is None
    assert swapper._reference_face is None
    logger.info("PASS: unload clears state")


def test_live_face_swap():
    """
    Live test: loads real models and swaps a face on a test image.
    Only runs when LIVESELF_TEST_LIVE=1 is set.
    Requires: GPU, insightface, inswapper_128.onnx, a test photo.
    """
    import cv2

    test_photo = os.environ.get("LIVESELF_TEST_PHOTO")
    if not test_photo:
        logger.error("Set LIVESELF_TEST_PHOTO to a face photo path for live test")
        return False

    if not os.path.exists(test_photo):
        logger.error(f"Test photo not found: {test_photo}")
        return False

    logger.info(f"Running live face swap test with photo: {test_photo}")

    # Initialize and load models
    swapper = FaceSwapper()
    start = time.perf_counter()
    swapper.load()
    logger.info(f"Model load time: {(time.perf_counter() - start) * 1000:.0f}ms")

    # Load reference face
    swapper.load_reference(test_photo)
    assert swapper.is_ready, "Swapper should be ready after loading reference"

    # Create a test frame (or use webcam frame)
    test_frame_path = os.environ.get("LIVESELF_TEST_FRAME")
    if test_frame_path and os.path.exists(test_frame_path):
        frame = cv2.imread(test_frame_path)
    else:
        # Use the reference photo itself as the test frame
        frame = cv2.imread(test_photo)

    assert frame is not None, "Could not read test frame"

    # Run face swap
    start = time.perf_counter()
    result = swapper.swap_face(frame)
    swap_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Face swap time: {swap_ms:.1f}ms")

    # Verify output
    assert result is not None, "Result should not be None"
    assert result.shape == frame.shape, f"Shape mismatch: {result.shape} vs {frame.shape}"
    assert result.dtype == frame.dtype, f"Dtype mismatch: {result.dtype} vs {frame.dtype}"

    # Save result for visual inspection
    output_path = os.path.join(os.path.dirname(__file__), "test_faceswap_output.jpg")
    cv2.imwrite(output_path, result)
    logger.info(f"Swapped face saved to: {output_path}")

    # Test with blank frame (no face) -- should return original
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result_blank = swapper.swap_face(blank)
    assert np.array_equal(result_blank, blank), "Blank frame should be returned unchanged"
    logger.info("PASS: blank frame returns original")

    # Clean up
    swapper.unload()
    logger.info("PASS: live face swap test complete")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("FaceSwapper Tests -- Phase 1A")
    logger.info("=" * 60)

    # Mock tests (always run, no GPU needed)
    test_face_area_calculation()
    test_largest_face_selection()
    test_swapper_init()
    test_swapper_errors_before_load()
    test_unload()

    logger.info("-" * 60)
    logger.info("Mock tests: ALL PASSED")
    logger.info("-" * 60)

    # Live test (only if explicitly enabled)
    if os.environ.get("LIVESELF_TEST_LIVE") == "1":
        if not DEFAULT_MODEL_PATH.exists():
            logger.warning(
                f"Skipping live test: model not found at {DEFAULT_MODEL_PATH}. "
                "Download inswapper_128.onnx first."
            )
        else:
            success = test_live_face_swap()
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
