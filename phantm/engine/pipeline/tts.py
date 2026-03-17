"""
Text-to-Speech / Voice Clone -- Phase 1B

Uses CosyVoice 2 (FunAudioLLM/CosyVoice) to synthesize speech in the
user's cloned voice from a 10-30 second reference audio sample.

Input: text string + reference voice audio path
Output: synthesized audio as numpy array (or streamed chunks)

CosyVoice 2 is installed via git clone into engine/models/CosyVoice/.
It is NOT a pip package. We add its path to sys.path at runtime.

Key features used:
  - Zero-shot voice cloning from short reference sample
  - Streaming synthesis (~150ms first-chunk latency)
  - CosyVoice2-0.5B model for speed (smaller, faster)

The model weights are downloaded on first use via modelscope/huggingface.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np

logger = logging.getLogger("liveself.engine.tts")

# CosyVoice repo is cloned here by setup_engine.sh
COSYVOICE_REPO_PATH = Path(__file__).parent.parent / "models" / "CosyVoice"

# Default model ID -- CosyVoice2-0.5B is the fast model, good for real-time
DEFAULT_MODEL_ID = "iic/CosyVoice2-0.5B"

# Audio sample rate that CosyVoice 2 outputs
OUTPUT_SAMPLE_RATE = 22050


def _ensure_cosyvoice_importable() -> None:
    """
    Add the CosyVoice repo to sys.path so we can import its modules.
    CosyVoice is not a pip package -- it is used via git clone.

    Raises:
        FileNotFoundError: If the CosyVoice repo has not been cloned yet.
    """
    repo_path = str(COSYVOICE_REPO_PATH)
    if repo_path not in sys.path:
        if not COSYVOICE_REPO_PATH.exists():
            raise FileNotFoundError(
                f"CosyVoice repo not found at {COSYVOICE_REPO_PATH}. "
                "Clone it with: git clone https://github.com/FunAudioLLM/CosyVoice.git "
                "into engine/models/"
            )
        # CosyVoice expects its root and third_party/Matcha-TTS on the path
        sys.path.insert(0, repo_path)
        matcha_path = str(COSYVOICE_REPO_PATH / "third_party" / "Matcha-TTS")
        if Path(matcha_path).exists() and matcha_path not in sys.path:
            sys.path.insert(0, matcha_path)
        logger.info(f"Added CosyVoice to sys.path: {repo_path}")


class VoiceCloner:
    """
    Voice cloning TTS using CosyVoice 2.

    Loads a pre-trained model, registers a reference voice sample,
    then synthesizes any text in that voice. Supports both batch and
    streaming modes.

    Usage:
        cloner = VoiceCloner()
        cloner.load()
        cloner.set_reference_voice("my_voice_sample.wav")
        audio = cloner.synthesize("Hello, I am your AI twin")

    For streaming (lower latency):
        for chunk in cloner.synthesize_stream("Hello, I am your AI twin"):
            process(chunk)  # each chunk is a numpy array
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cosyvoice_repo_path: Optional[str] = None,
    ):
        """
        Args:
            model_id: HuggingFace/ModelScope model ID for CosyVoice2.
                      Default is CosyVoice2-0.5B (fast, good for real-time).
            cosyvoice_repo_path: Override path to the cloned CosyVoice repo.
        """
        self._model_id = model_id
        self._repo_path = Path(cosyvoice_repo_path) if cosyvoice_repo_path else COSYVOICE_REPO_PATH
        self._model = None
        self._reference_audio_path = None
        self._is_loaded = False

        logger.info(f"VoiceCloner created (model: {model_id}, not loaded yet)")

    def load(self) -> None:
        """
        Load the CosyVoice 2 model into memory. Call once before synthesis.

        Downloads model weights on first run (~1-2 GB for 0.5B model).
        This allocates GPU memory.

        Raises:
            FileNotFoundError: If CosyVoice repo is not cloned.
            ImportError: If CosyVoice dependencies are missing.
        """
        start = time.perf_counter()

        _ensure_cosyvoice_importable()

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
        except ImportError as e:
            raise ImportError(
                f"Failed to import CosyVoice2: {e}. "
                "Make sure CosyVoice is cloned and its requirements are installed. "
                "See: https://github.com/FunAudioLLM/CosyVoice#install"
            )

        # Load the model -- this downloads weights on first run
        self._model = CosyVoice2(
            self._model_id,
            load_jit=True,
            load_trt=False,
        )

        self._is_loaded = True
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"CosyVoice2 model loaded in {elapsed_ms:.0f}ms (model: {self._model_id})")

    def set_reference_voice(self, audio_path: str) -> None:
        """
        Register a reference voice sample for zero-shot cloning.

        The audio should be 10-30 seconds of clear speech from the target
        speaker. Longer samples generally produce better cloning quality.

        Args:
            audio_path: Path to a WAV file (16kHz+ mono recommended).

        Raises:
            RuntimeError: If model is not loaded yet.
            FileNotFoundError: If the audio file does not exist.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference voice audio not found: {audio_path}")

        self._reference_audio_path = str(audio_path)
        logger.info(f"Reference voice set: {audio_path.name}")

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text in the cloned voice.

        Collects all streaming chunks into a single numpy array.
        For lower latency, use synthesize_stream() instead.

        Args:
            text: The text to speak. Keep under ~200 characters for best quality.

        Returns:
            Audio waveform as float32 numpy array, sample rate OUTPUT_SAMPLE_RATE.

        Raises:
            RuntimeError: If model or reference voice is not loaded.
        """
        chunks = list(self.synthesize_stream(text))
        if not chunks:
            logger.warning(f"No audio generated for text: {text[:50]}")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(chunks)
        return audio

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Stream synthesized speech chunks as they are generated.

        This is the preferred method for real-time use. Each yielded chunk
        is a numpy array that can be immediately sent to the lipsync module
        or played back.

        CosyVoice 2 has ~150ms latency to the first chunk, then streams
        continuously.

        Args:
            text: The text to speak.

        Yields:
            Audio chunks as float32 numpy arrays, sample rate OUTPUT_SAMPLE_RATE.

        Raises:
            RuntimeError: If model or reference voice is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._reference_audio_path is None:
            raise RuntimeError("No reference voice set. Call set_reference_voice() first.")

        start = time.perf_counter()
        chunk_count = 0
        total_samples = 0

        try:
            # CosyVoice2 inference_zero_shot yields dicts with 'tts_speech' tensor
            for result in self._model.inference_zero_shot(
                tts_text=text,
                prompt_text="",
                prompt_speech_16k=self._load_prompt_audio(),
                stream=True,
            ):
                chunk = result["tts_speech"].numpy().flatten().astype(np.float32)
                chunk_count += 1
                total_samples += len(chunk)

                if chunk_count == 1:
                    first_chunk_ms = (time.perf_counter() - start) * 1000
                    logger.debug(f"TTS first chunk in {first_chunk_ms:.0f}ms")

                yield chunk

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return

        elapsed_ms = (time.perf_counter() - start) * 1000
        duration_s = total_samples / OUTPUT_SAMPLE_RATE if total_samples > 0 else 0
        logger.info(
            f"TTS complete: {chunk_count} chunks, {duration_s:.1f}s audio, "
            f"{elapsed_ms:.0f}ms wall time, text='{text[:50]}'"
        )

    def _load_prompt_audio(self) -> "torch.Tensor":
        """
        Load the reference voice audio as a torch tensor for CosyVoice.

        CosyVoice expects 16kHz mono audio as a torch tensor.
        We use torchaudio for loading and resampling.

        Returns:
            Torch tensor of shape (1, num_samples) at 16kHz.
        """
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(self._reference_audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed (CosyVoice expects 16kHz prompt audio)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        return waveform

    @property
    def is_ready(self) -> bool:
        """Check if the cloner is fully loaded with a reference voice."""
        return self._is_loaded and self._reference_audio_path is not None

    @property
    def sample_rate(self) -> int:
        """Output audio sample rate in Hz."""
        return OUTPUT_SAMPLE_RATE

    def unload(self) -> None:
        """Release the model and free GPU memory."""
        self._model = None
        self._reference_audio_path = None
        self._is_loaded = False
        logger.info("VoiceCloner model unloaded")
