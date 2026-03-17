"""
Virtual Camera Output -- Phase 1A

Takes processed video frames and outputs them to a virtual camera device.
This is what Zoom/Meet/WhatsApp sees as your webcam.

Uses pyvirtualcam which creates a virtual camera device:
  - Windows: requires OBS Virtual Camera (install OBS Studio)
  - Linux: requires v4l2loopback kernel module
  - macOS: requires OBS Virtual Camera

Fallback: if pyvirtualcam is not available, saves frames to a video file
so you can still verify the pipeline works.

Input: video frames as numpy arrays (BGR, uint8)
Output: virtual camera device visible to Zoom/Meet/WhatsApp
"""

import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("liveself.engine.virtual_cam")


class VirtualCameraOutput:
    """
    Pushes video frames to a virtual camera device that appears in
    Zoom, Google Meet, WhatsApp Desktop, etc.

    Usage:
        cam = VirtualCameraOutput(width=1280, height=720, fps=30)
        cam.start()
        cam.send_frame(frame)  # call this ~30 times per second
        cam.stop()

    If pyvirtualcam is not installed or no virtual camera backend is
    available, falls back to saving frames as a video file.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        fallback_path: str = "output_fallback.mp4",
    ):
        """
        Args:
            width: Output width in pixels.
            height: Output height in pixels.
            fps: Target frames per second.
            fallback_path: Path to save video if virtual camera is unavailable.
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._fallback_path = fallback_path

        self._vcam = None
        self._video_writer = None
        self._is_running = False
        self._frame_count = 0
        self._start_time = None
        self._using_fallback = False

    def start(self) -> None:
        """
        Start the virtual camera output.

        Tries pyvirtualcam first. If that fails (not installed, no OBS,
        no v4l2loopback), falls back to writing a video file.
        """
        # Try pyvirtualcam first
        try:
            import pyvirtualcam
            self._vcam = pyvirtualcam.Camera(
                width=self._width,
                height=self._height,
                fps=self._fps,
                print_fps=False,
            )
            self._using_fallback = False
            logger.info(
                f"Virtual camera started: {self._vcam.device} "
                f"({self._width}x{self._height} @ {self._fps}fps)"
            )
        except ImportError:
            logger.warning(
                "pyvirtualcam not installed. Falling back to video file output. "
                "Install with: pip install pyvirtualcam"
            )
            self._start_fallback_writer()
        except Exception as e:
            logger.warning(
                f"Virtual camera backend not available: {e}. "
                "On Windows, install OBS Studio for virtual camera support. "
                "Falling back to video file output."
            )
            self._start_fallback_writer()

        self._is_running = True
        self._frame_count = 0
        self._start_time = time.perf_counter()

    def _start_fallback_writer(self) -> None:
        """Start a video file writer as fallback when virtual camera is unavailable."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            self._fallback_path, fourcc, self._fps, (self._width, self._height)
        )
        self._using_fallback = True
        logger.info(f"Fallback video writer started: {self._fallback_path}")

    def send_frame(self, frame: np.ndarray) -> None:
        """
        Send a single frame to the virtual camera (or fallback writer).

        The frame is resized to match the output resolution if needed.
        pyvirtualcam expects RGB format, so BGR frames are converted.

        Args:
            frame: Video frame as numpy array (BGR, uint8, any resolution).
        """
        if not self._is_running:
            return

        # Resize if frame dimensions don't match output
        h, w = frame.shape[:2]
        if w != self._width or h != self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        if self._vcam is not None:
            # pyvirtualcam expects RGB, OpenCV uses BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._vcam.send(rgb_frame)
            self._vcam.sleep_until_next_frame()
        elif self._video_writer is not None:
            # VideoWriter expects BGR (same as OpenCV default)
            self._video_writer.write(frame)

        self._frame_count += 1

    def stop(self) -> None:
        """
        Stop the virtual camera and release resources.
        Logs performance stats (total frames, average FPS).
        """
        if not self._is_running:
            return

        self._is_running = False

        # Calculate stats
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0

        if self._vcam is not None:
            self._vcam.close()
            self._vcam = None
            logger.info(
                f"Virtual camera stopped. "
                f"{self._frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} avg fps)"
            )

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            logger.info(
                f"Fallback video saved to {self._fallback_path}. "
                f"{self._frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} avg fps)"
            )

    @property
    def is_running(self) -> bool:
        """Check if the virtual camera is active."""
        return self._is_running

    @property
    def is_fallback(self) -> bool:
        """Check if using video file fallback instead of virtual camera."""
        return self._using_fallback

    @property
    def frame_count(self) -> int:
        """Total frames sent since start()."""
        return self._frame_count


async def virtual_cam_worker(
    frame_queue: asyncio.Queue,
    cam: VirtualCameraOutput,
    stop_event: asyncio.Event,
) -> None:
    """
    Async worker that consumes frames from a queue and sends them
    to the virtual camera. Used by the pipeline orchestrator.

    Args:
        frame_queue: Queue of numpy frames to output.
        cam: VirtualCameraOutput instance (already started).
        stop_event: Set this to stop the worker gracefully.
    """
    logger.info("Virtual camera worker started")
    while not stop_event.is_set():
        try:
            frame = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
            cam.send_frame(frame)
            frame_queue.task_done()
        except asyncio.TimeoutError:
            # No frame available, just loop and check stop_event
            continue
        except Exception as e:
            logger.error(f"Virtual camera worker error: {e}")
            continue

    logger.info(f"Virtual camera worker stopped ({cam.frame_count} frames sent)")
