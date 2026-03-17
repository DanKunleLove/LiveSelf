"""
Automatic Speech Recognition (ASR) -- Phase 1C

Uses faster-whisper (SYSTRAN/faster-whisper) to transcribe what the caller
says in real-time. Includes Voice Activity Detection (VAD) via Silero to
detect when someone has finished speaking so the avatar can respond.

Input: raw audio chunks from microphone (16-bit PCM or float32)
Output: transcribed text segments with end-of-speech signals

Pipeline position: Microphone -> [ASR] -> RAG -> LLM -> TTS -> ...

faster-whisper:
  - 4x faster than OpenAI Whisper (CTranslate2 backend)
  - Built-in Silero VAD for speech detection
  - ~150ms transcription latency on GPU
  - MIT license
"""

import asyncio
import logging
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("liveself.engine.asr")

# Audio format constants
SAMPLE_RATE = 16000  # faster-whisper expects 16kHz
CHUNK_DURATION_S = 0.5  # How many seconds of audio per chunk from mic
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_S)

# VAD settings
VAD_SILENCE_THRESHOLD_S = 1.0  # Seconds of silence to consider end-of-speech
VAD_SPEECH_PAD_MS = 300  # Padding around detected speech segments


class SpeechRecognizer:
    """
    Real-time speech-to-text using faster-whisper with Silero VAD.

    Accumulates audio chunks, runs VAD to detect speech segments,
    and transcribes completed segments. Designed for the async pipeline.

    Usage:
        recognizer = SpeechRecognizer()
        recognizer.load()
        text = recognizer.transcribe_segment(audio_array)

    For real-time pipeline use:
        recognizer = SpeechRecognizer()
        recognizer.load()
        recognizer.feed_audio(chunk)  # call repeatedly with mic data
        if recognizer.has_complete_utterance():
            text = recognizer.get_utterance()
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
    ):
        """
        Args:
            model_size: Whisper model size. Options: tiny, base, small, medium, large-v3.
                        "medium" balances speed and accuracy for real-time use.
            device: "auto" picks CUDA if available, else CPU.
            compute_type: "auto" picks float16 on GPU, int8 on CPU.
            language: Language code for transcription. "en" for English.
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language

        self._model = None
        self._is_loaded = False

        # Audio buffer for accumulating chunks
        self._audio_buffer = np.array([], dtype=np.float32)
        # Completed utterances waiting to be consumed
        self._utterance_queue: list[str] = []
        # Track whether we are currently in a speech segment
        self._in_speech = False
        self._silence_start: Optional[float] = None
        # Speech segment being accumulated
        self._speech_buffer = np.array([], dtype=np.float32)

        logger.info(f"SpeechRecognizer created (model: {model_size}, not loaded yet)")

    def load(self) -> None:
        """
        Load the faster-whisper model into memory.

        Downloads model weights on first run. Allocates GPU memory.

        Raises:
            ImportError: If faster-whisper is not installed.
        """
        start = time.perf_counter()

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required but not installed. "
                "Install with: pip install faster-whisper"
            )

        # Resolve device and compute type
        device = self._device
        compute_type = self._compute_type

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )

        self._is_loaded = True
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"faster-whisper loaded in {elapsed_ms:.0f}ms "
            f"(model: {self._model_size}, device: {device}, compute: {compute_type})"
        )

    def transcribe_segment(self, audio: np.ndarray) -> str:
        """
        Transcribe a complete audio segment.

        This is the simple batch API. For real-time use, use feed_audio()
        and get_utterance() instead.

        Args:
            audio: Audio waveform as float32 numpy array, 16kHz mono.

        Returns:
            Transcribed text string.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.perf_counter()

        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=int(VAD_SILENCE_THRESHOLD_S * 1000),
                speech_pad_ms=VAD_SPEECH_PAD_MS,
            ),
        )

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()

        elapsed_ms = (time.perf_counter() - start) * 1000
        audio_duration_s = len(audio) / SAMPLE_RATE
        logger.info(
            f"Transcribed {audio_duration_s:.1f}s audio in {elapsed_ms:.0f}ms: "
            f"'{text[:80]}'"
        )

        return text

    def feed_audio(self, chunk: np.ndarray) -> None:
        """
        Feed a chunk of audio from the microphone.

        Uses Silero VAD to detect speech boundaries. When an utterance is
        complete (speech followed by silence), it gets transcribed and
        queued for retrieval via get_utterance().

        Args:
            chunk: Audio chunk as float32 numpy array, 16kHz mono.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Run VAD on this chunk
        is_speech = self._detect_speech(chunk)

        if is_speech:
            # Accumulate speech audio
            self._speech_buffer = np.concatenate([self._speech_buffer, chunk])
            self._in_speech = True
            self._silence_start = None
        else:
            if self._in_speech:
                # We were in speech, now silence detected
                if self._silence_start is None:
                    self._silence_start = time.perf_counter()

                # Add silence to buffer (keep a bit of trailing silence)
                self._speech_buffer = np.concatenate([self._speech_buffer, chunk])

                # Check if silence has lasted long enough to end the utterance
                silence_duration = time.perf_counter() - self._silence_start
                if silence_duration >= VAD_SILENCE_THRESHOLD_S:
                    self._finalize_utterance()

    def _detect_speech(self, chunk: np.ndarray) -> bool:
        """
        Detect if a chunk contains speech using energy-based VAD.

        For production, this uses faster-whisper's built-in Silero VAD
        during transcription. For the feed_audio path, we use a simple
        energy threshold as a pre-filter to avoid running full VAD on
        every chunk.

        Args:
            chunk: Audio chunk as float32 numpy array.

        Returns:
            True if speech is detected in the chunk.
        """
        # Simple energy-based pre-filter
        # RMS energy threshold -- calibrated for typical mic input
        rms = np.sqrt(np.mean(chunk ** 2))
        threshold = 0.01  # Adjust based on mic gain
        return rms > threshold

    def _finalize_utterance(self) -> None:
        """Transcribe the accumulated speech buffer and queue the result."""
        if len(self._speech_buffer) < SAMPLE_RATE * 0.3:
            # Too short to be meaningful speech (less than 300ms)
            logger.debug("Discarding short speech segment")
            self._speech_buffer = np.array([], dtype=np.float32)
            self._in_speech = False
            self._silence_start = None
            return

        # Transcribe the complete utterance
        text = self.transcribe_segment(self._speech_buffer)

        if text:
            self._utterance_queue.append(text)
            logger.info(f"Utterance complete: '{text[:80]}'")

        # Reset state for next utterance
        self._speech_buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._silence_start = None

    def has_complete_utterance(self) -> bool:
        """Check if there is a transcribed utterance ready to consume."""
        return len(self._utterance_queue) > 0

    def get_utterance(self) -> Optional[str]:
        """
        Get the next transcribed utterance.

        Returns:
            Transcribed text, or None if no utterance is ready.
        """
        if self._utterance_queue:
            return self._utterance_queue.pop(0)
        return None

    def reset(self) -> None:
        """Clear all buffered audio and pending utterances."""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._utterance_queue.clear()
        self._in_speech = False
        self._silence_start = None
        logger.info("SpeechRecognizer buffers cleared")

    @property
    def is_ready(self) -> bool:
        """Check if the recognizer is loaded and ready."""
        return self._is_loaded

    def unload(self) -> None:
        """Release model and free GPU memory."""
        self._model = None
        self._is_loaded = False
        self.reset()
        logger.info("SpeechRecognizer model unloaded")


async def asr_worker(
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    recognizer: SpeechRecognizer,
    stop_event: asyncio.Event,
) -> None:
    """
    Async worker that consumes audio chunks and produces transcribed text.

    Reads audio chunks from audio_queue, feeds them through VAD and
    transcription, and pushes completed utterances to text_queue.

    Args:
        audio_queue: Queue of audio chunks (float32 numpy arrays, 16kHz).
        text_queue: Queue to push transcribed text strings into.
        recognizer: Loaded SpeechRecognizer instance.
        stop_event: Set this to stop the worker gracefully.
    """
    logger.info("ASR worker started")

    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
            recognizer.feed_audio(chunk)
            audio_queue.task_done()

            # Check if a complete utterance is ready
            while recognizer.has_complete_utterance():
                text = recognizer.get_utterance()
                if text:
                    await text_queue.put(text)
                    logger.info(f"ASR -> text queue: '{text[:60]}'")

        except asyncio.TimeoutError:
            # Check for any pending utterance even without new audio
            # (handles the case where silence threshold was met)
            while recognizer.has_complete_utterance():
                text = recognizer.get_utterance()
                if text:
                    await text_queue.put(text)
            continue
        except Exception as e:
            logger.error(f"ASR worker error: {e}")
            continue

    logger.info("ASR worker stopped")
