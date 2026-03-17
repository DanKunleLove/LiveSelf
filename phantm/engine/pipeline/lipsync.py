"""
Lip Sync -- Phase 1B

Uses MuseTalk 1.5 (TMElyralab/MuseTalk) to drive realistic lip movements
on the avatar face based on audio input.

Input: audio waveform (numpy array) + face reference frame (numpy array)
Output: video frames with lips moving to match the audio

MuseTalk is installed via git clone into engine/models/MuseTalk/.
It is NOT a pip package. We add its path to sys.path at runtime.

Key features:
  - Real-time lip sync at 30fps+
  - GAN loss for cleaner mouth region
  - Works with any face (not speaker-specific training needed)
  - MIT license, Tencent-backed
"""

import logging
import sys
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger("liveself.engine.lipsync")

# MuseTalk repo is cloned here by setup_engine.sh
MUSETALK_REPO_PATH = Path(__file__).parent.parent / "models" / "MuseTalk"

# Target output fps
TARGET_FPS = 30


def _ensure_musetalk_importable() -> None:
    """
    Add the MuseTalk repo to sys.path so we can import its modules.
    MuseTalk is not a pip package -- it is used via git clone.

    Raises:
        FileNotFoundError: If the MuseTalk repo has not been cloned yet.
    """
    repo_path = str(MUSETALK_REPO_PATH)
    if repo_path not in sys.path:
        if not MUSETALK_REPO_PATH.exists():
            raise FileNotFoundError(
                f"MuseTalk repo not found at {MUSETALK_REPO_PATH}. "
                "Clone it with: git clone https://github.com/TMElyralab/MuseTalk.git "
                "into engine/models/"
            )
        sys.path.insert(0, repo_path)
        logger.info(f"Added MuseTalk to sys.path: {repo_path}")


class LipSyncer:
    """
    Real-time lip sync using MuseTalk 1.5.

    Takes audio waveform and a face reference frame, produces video frames
    with lips accurately moving to match the audio. Designed for the
    real-time avatar pipeline.

    Usage:
        syncer = LipSyncer()
        syncer.load()
        syncer.set_reference_frame(face_frame)
        frames = syncer.sync(audio_array, sample_rate=22050)

    For streaming (lower latency):
        for frame in syncer.sync_stream(audio_chunk, sample_rate=22050):
            send_to_virtual_cam(frame)
    """

    def __init__(
        self,
        musetalk_repo_path: Optional[str] = None,
        output_size: tuple[int, int] = (256, 256),
    ):
        """
        Args:
            musetalk_repo_path: Override path to the cloned MuseTalk repo.
            output_size: Size of the lip-synced face crop (width, height).
                         The orchestrator composites this back onto the full frame.
        """
        self._repo_path = Path(musetalk_repo_path) if musetalk_repo_path else MUSETALK_REPO_PATH
        self._output_size = output_size

        # Set during load()
        self._model = None
        self._audio_processor = None
        self._is_loaded = False

        # Set during set_reference_frame()
        self._reference_frame = None
        self._reference_landmarks = None

        logger.info("LipSyncer created (model not loaded yet)")

    def load(self) -> None:
        """
        Load MuseTalk models into memory. Call once before syncing.

        Loads:
          - The MuseTalk inference model (audio-to-lip weights)
          - Audio feature extractor (whisper-based audio encoder)
          - Face parsing model (for mouth region masking)

        Raises:
            FileNotFoundError: If MuseTalk repo is not cloned.
            ImportError: If MuseTalk dependencies are missing.
        """
        start = time.perf_counter()

        _ensure_musetalk_importable()

        try:
            from musetalk.utils.utils import load_all_model
        except ImportError as e:
            raise ImportError(
                f"Failed to import MuseTalk: {e}. "
                "Make sure MuseTalk is cloned and its requirements are installed. "
                "See: https://github.com/TMElyralab/MuseTalk#installation"
            )

        # load_all_model returns the inference pipeline components
        # MuseTalk 1.5 bundles: audio encoder, lip generation model, face parser
        model_components = load_all_model()

        self._audio_processor = model_components.get("audio_processor")
        self._model = model_components.get("model")

        # Store all components for inference
        self._model_components = model_components
        self._is_loaded = True

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"MuseTalk models loaded in {elapsed_ms:.0f}ms")

    def set_reference_frame(self, frame: np.ndarray) -> None:
        """
        Set the reference face frame that lip movements will be applied to.

        This should be a clear, front-facing face image. Typically this is
        a frame from the faceswap output -- the swapped face with neutral
        expression.

        MuseTalk extracts facial landmarks and a face crop from this frame,
        which it uses as the base for all subsequent lip-sync frames.

        Args:
            frame: BGR face image as numpy array (H, W, 3), uint8.

        Raises:
            RuntimeError: If models are not loaded yet.
            ValueError: If no face is detected in the frame.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        start = time.perf_counter()

        try:
            from musetalk.utils.preprocessing import get_landmark_and_bbox

            # Extract face landmarks and bounding box
            landmarks, bbox = get_landmark_and_bbox(frame)

            if landmarks is None or bbox is None:
                raise ValueError(
                    "No face detected in reference frame. "
                    "Use a clear, front-facing face image."
                )

            self._reference_frame = frame.copy()
            self._reference_landmarks = landmarks
            self._reference_bbox = bbox

        except ImportError:
            # Fallback: store the frame directly and let inference handle detection
            logger.warning("MuseTalk preprocessing not available, storing frame directly")
            self._reference_frame = frame.copy()
            self._reference_landmarks = None
            self._reference_bbox = None

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Reference frame set ({frame.shape[1]}x{frame.shape[0]}) in {elapsed_ms:.0f}ms")

    def sync(self, audio: np.ndarray, sample_rate: int = 22050) -> list[np.ndarray]:
        """
        Generate lip-synced video frames for the given audio.

        Takes the full audio waveform and produces all frames at once.
        For real-time use, prefer sync_stream().

        The number of output frames = audio_duration * TARGET_FPS.

        Args:
            audio: Audio waveform as float32 numpy array.
            sample_rate: Sample rate of the input audio in Hz.

        Returns:
            List of BGR video frames (numpy arrays) with lip movements.

        Raises:
            RuntimeError: If model or reference frame is not set.
        """
        frames = list(self.sync_stream(audio, sample_rate))
        return frames

    def sync_stream(
        self, audio: np.ndarray, sample_rate: int = 22050
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream lip-synced video frames as they are generated.

        This is the preferred method for real-time use. Each yielded frame
        can be immediately composited onto the full video frame and sent
        to the virtual camera.

        Args:
            audio: Audio waveform as float32 numpy array.
            sample_rate: Sample rate of the input audio in Hz.

        Yields:
            BGR video frames (numpy arrays) with lip movements applied.

        Raises:
            RuntimeError: If model or reference frame is not set.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        if self._reference_frame is None:
            raise RuntimeError("No reference frame set. Call set_reference_frame() first.")

        start = time.perf_counter()
        frame_count = 0

        try:
            # Calculate how many frames we need for this audio
            audio_duration = len(audio) / sample_rate
            num_frames = max(1, int(audio_duration * TARGET_FPS))

            # Extract audio features using MuseTalk's audio encoder
            from musetalk.utils.preprocessing import get_audio_features

            audio_features = get_audio_features(
                audio,
                sample_rate=sample_rate,
                num_frames=num_frames,
            )

            # Generate lip-synced frames one at a time
            from musetalk.utils.inference import inference_single_frame

            for i in range(num_frames):
                # Get the audio feature window for this frame
                audio_feat = audio_features[i] if i < len(audio_features) else audio_features[-1]

                # Generate the lip-synced face
                result_frame = inference_single_frame(
                    self._model_components,
                    self._reference_frame,
                    audio_feat,
                    self._reference_bbox,
                )

                if result_frame is not None:
                    frame_count += 1
                    yield result_frame
                else:
                    # If inference fails for a frame, yield the reference frame
                    yield self._reference_frame.copy()
                    frame_count += 1

        except ImportError as e:
            logger.error(
                f"MuseTalk inference modules not available: {e}. "
                "Falling back to static reference frame."
            )
            # Fallback: yield static frames (no lip movement)
            audio_duration = len(audio) / sample_rate
            num_frames = max(1, int(audio_duration * TARGET_FPS))
            for _ in range(num_frames):
                yield self._reference_frame.copy()
                frame_count += 1

        except Exception as e:
            logger.error(f"Lip sync failed: {e}")
            return

        elapsed_ms = (time.perf_counter() - start) * 1000
        actual_fps = frame_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        logger.info(
            f"Lip sync complete: {frame_count} frames in {elapsed_ms:.0f}ms "
            f"({actual_fps:.1f} fps)"
        )

    @property
    def is_ready(self) -> bool:
        """Check if the syncer is fully loaded with a reference frame."""
        return self._is_loaded and self._reference_frame is not None

    @property
    def target_fps(self) -> int:
        """Target output frame rate."""
        return TARGET_FPS

    def unload(self) -> None:
        """Release models and free GPU memory."""
        self._model = None
        self._audio_processor = None
        self._model_components = None
        self._reference_frame = None
        self._reference_landmarks = None
        self._is_loaded = False
        logger.info("LipSyncer models unloaded")
