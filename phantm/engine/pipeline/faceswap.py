"""
Face Swap Module -- Phase 1A

Wraps InsightFace (the library Deep-Live-Cam uses under the hood) to swap
a reference photo's face onto live video frames in real-time.

Input: reference photo path + live video frame (numpy array)
Output: face-swapped video frame (numpy array)

Uses two InsightFace components:
  1. FaceAnalysis -- detects and analyzes faces in frames
  2. inswapper_128 -- ONNX model that performs the actual face swap

The inswapper_128.onnx model must be downloaded separately and placed
in engine/models/inswapper_128.onnx (see scripts/setup_engine.sh).
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("liveself.engine.faceswap")

# Path where the inswapper ONNX model is expected
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "inswapper_128.onnx"


def _get_face_area(face) -> float:
    """Return the bounding box area of a detected face for size comparison."""
    bbox = face.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height


class FaceSwapper:
    """
    Real-time face swapper using InsightFace.

    Loads a reference photo once, then swaps that face onto every incoming
    video frame. Thread-safe for single-producer usage (one session at a time).

    Usage:
        swapper = FaceSwapper(model_path="engine/models/inswapper_128.onnx")
        swapper.load_reference("photo.jpg")
        swapped_frame = swapper.swap_face(webcam_frame)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        det_size: tuple[int, int] = (640, 640),
        execution_provider: str = "auto",
    ):
        """
        Initialize the face swapper.

        Args:
            model_path: Path to inswapper_128.onnx. Defaults to engine/models/.
            det_size: Face detection input size. Smaller = faster, less accurate.
            execution_provider: "auto" picks CUDAExecutionProvider if available,
                                falls back to CPUExecutionProvider.
        """
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._det_size = det_size
        self._execution_provider = execution_provider

        # These get set during load()
        self._face_analyser = None
        self._swapper_model = None
        self._reference_face = None
        self._is_loaded = False

        logger.info("FaceSwapper created (model not loaded yet)")

    def load(self) -> None:
        """
        Load InsightFace models into memory. Call this once before swapping.

        Loads both the face analyser (detection + recognition) and the
        inswapper ONNX model. This allocates GPU memory.

        Raises:
            FileNotFoundError: If inswapper_128.onnx is not found at model_path.
            ImportError: If insightface is not installed.
        """
        start = time.perf_counter()

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required but not installed. "
                "Install with: pip install insightface onnxruntime-gpu"
            )

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"inswapper_128.onnx not found at {self._model_path}. "
                "Download it and place it in engine/models/. "
                "See: https://github.com/hacksider/Deep-Live-Cam#models"
            )

        # Determine execution providers for ONNX Runtime
        providers = self._resolve_providers()

        # Initialize face analyser for detection and embedding extraction
        self._face_analyser = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self._face_analyser.prepare(ctx_id=0, det_size=self._det_size)

        # Load the inswapper model
        self._swapper_model = insightface.model_zoo.get_model(
            str(self._model_path),
            providers=providers,
        )

        self._is_loaded = True
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"FaceSwapper models loaded in {elapsed_ms:.0f}ms (providers: {providers})")

    def _resolve_providers(self) -> list[str]:
        """
        Determine which ONNX Runtime execution providers to use.

        Returns a list of provider names. Tries CUDA first, falls back to CPU.
        """
        if self._execution_provider != "auto":
            return [self._execution_provider]

        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                logger.info("CUDA execution provider available -- using GPU")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except ImportError:
            pass

        logger.info("Using CPU execution provider (no GPU detected)")
        return ["CPUExecutionProvider"]

    def load_reference(self, photo_path: str) -> None:
        """
        Load a reference face photo. This is the face that will appear on the video.

        The photo should be a clear, front-facing image with exactly one face.
        If multiple faces are found, the largest one is used.

        Args:
            photo_path: Path to the reference photo (jpg, png, etc.)

        Raises:
            RuntimeError: If models are not loaded yet (call load() first).
            FileNotFoundError: If the photo file does not exist.
            ValueError: If no face is detected in the reference photo.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        start = time.perf_counter()
        photo_path = Path(photo_path)

        if not photo_path.exists():
            raise FileNotFoundError(f"Reference photo not found: {photo_path}")

        # Read the reference image
        image = cv2.imread(str(photo_path))
        if image is None:
            raise ValueError(f"Could not read image file: {photo_path}")

        # Detect faces in the reference photo
        faces = self._face_analyser.get(image)

        if not faces:
            raise ValueError(
                f"No face detected in reference photo: {photo_path}. "
                "Use a clear, front-facing photo with good lighting."
            )

        # Pick the largest face if multiple are found
        if len(faces) > 1:
            logger.warning(
                f"Multiple faces ({len(faces)}) found in reference photo. "
                "Using the largest face."
            )
            self._reference_face = max(faces, key=_get_face_area)
        else:
            self._reference_face = faces[0]

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Reference face loaded from {photo_path.name} in {elapsed_ms:.0f}ms")

    def swap_face(self, frame: np.ndarray) -> np.ndarray:
        """
        Swap the reference face onto the source video frame.

        This is the hot-path function called on every frame (~30 FPS).
        It must be fast. All heavy setup should be done in load() and
        load_reference() beforehand.

        Args:
            frame: BGR video frame as numpy array (H, W, 3), uint8.

        Returns:
            Face-swapped frame as numpy array, same shape and dtype as input.
            If no face is detected in the frame, returns the original frame unchanged.

        Raises:
            RuntimeError: If models or reference face are not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        if self._reference_face is None:
            raise RuntimeError("No reference face loaded. Call load_reference() first.")

        start = time.perf_counter()

        # Detect faces in the source frame
        source_faces = self._face_analyser.get(frame)

        if not source_faces:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"No face in frame, returning original ({elapsed_ms:.1f}ms)")
            return frame

        # Pick the largest face to swap (ignore smaller background faces)
        target_face = max(source_faces, key=_get_face_area)

        # Perform the face swap
        try:
            result = self._swapper_model.get(
                frame,
                target_face,
                self._reference_face,
                paste_back=True,
            )
        except Exception as e:
            logger.error(f"Face swap failed on frame: {e}")
            return frame

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Face swapped in {elapsed_ms:.1f}ms ({len(source_faces)} faces detected)")

        return result

    def swap_face_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Swap faces on a batch of frames. Convenience method for processing
        pre-recorded video or testing.

        Args:
            frames: List of BGR video frames.

        Returns:
            List of face-swapped frames (same length as input).
        """
        return [self.swap_face(frame) for frame in frames]

    @property
    def is_ready(self) -> bool:
        """Check if the swapper is fully initialized and ready to process frames."""
        return self._is_loaded and self._reference_face is not None

    def unload(self) -> None:
        """
        Release models and free GPU memory.
        Call this when the session ends.
        """
        self._face_analyser = None
        self._swapper_model = None
        self._reference_face = None
        self._is_loaded = False
        logger.info("FaceSwapper models unloaded")
