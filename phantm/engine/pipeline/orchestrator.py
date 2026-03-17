"""
Pipeline Orchestrator -- Phase 1C (the main loop)

Central coordinator that creates asyncio queues between all pipeline stages
and runs them concurrently. This is what makes the avatar real-time.

The pipeline flow:
  Mic -> ASR -> RAG -> LLM -> TTS -> Lip Sync -> Virtual Camera

All stages run as concurrent asyncio tasks. Each stage reads from its
input queue and writes to its output queue. The streaming overlap trick
(starting TTS before the full LLM response) happens naturally because
the LLM yields sentences one at a time.

Latency budget (target):
  ASR:      ~150ms (faster-whisper)
  RAG:      ~50ms  (ChromaDB local)
  LLM:      ~300ms to first sentence (Ollama streaming)
  TTS:      ~150ms to first chunk (CosyVoice streaming)
  Lip sync: ~30ms per frame (MuseTalk)
  Total:    ~500ms perceived (with streaming overlap)
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("liveself.engine.orchestrator")


class PipelineOrchestrator:
    """
    Runs the full avatar pipeline as concurrent async workers.

    Each worker reads from an input queue, processes data, and writes to
    an output queue. The orchestrator manages lifecycle (start/stop) and
    provides stats.

    Usage:
        orchestrator = PipelineOrchestrator(config)
        await orchestrator.initialize()
        await orchestrator.run()  # blocks until stop() is called
        await orchestrator.shutdown()
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Pipeline configuration dict with keys:
                - persona_id: str (required)
                - persona_name: str (default: "the user")
                - reference_photo: str (path to face photo)
                - reference_voice: str (path to voice WAV)
                - llm_provider: "ollama" or "claude" (default: "ollama")
                - llm_model: str (optional model override)
                - llm_api_key: str (for Claude provider)
                - whisper_model: str (default: "medium")
                - cam_width: int (default: 1280)
                - cam_height: int (default: 720)
                - cam_fps: int (default: 30)
        """
        self._config = config or {}
        self._stop_event = asyncio.Event()

        # Pipeline queues
        self._text_queue = asyncio.Queue(maxsize=5)  # ASR -> RAG+LLM
        self._sentence_queue = asyncio.Queue(maxsize=10)  # LLM -> TTS
        self._audio_queue = asyncio.Queue(maxsize=10)  # TTS -> Lip Sync
        self._frame_queue = asyncio.Queue(maxsize=60)  # Lip Sync -> Virtual Cam
        self._mic_queue = asyncio.Queue(maxsize=20)  # Mic -> ASR

        # Pipeline modules (initialized in initialize())
        self._asr = None
        self._retriever = None
        self._llm = None
        self._tts = None
        self._lipsync = None
        self._faceswap = None
        self._vcam = None

        # Stats
        self._session_start = None
        self._utterance_count = 0

        logger.info("PipelineOrchestrator created")

    async def initialize(self) -> None:
        """
        Load all pipeline modules. Call once before run().

        Loads models in parallel where possible to minimize startup time.
        """
        start = time.perf_counter()
        logger.info("Initializing pipeline modules...")

        # Import pipeline modules
        from pipeline.asr import SpeechRecognizer
        from pipeline.llm import LLMBrain
        from pipeline.tts import VoiceCloner
        from pipeline.lipsync import LipSyncer
        from pipeline.faceswap import FaceSwapper
        from pipeline.virtual_cam import VirtualCameraOutput
        from knowledge.retriever import KnowledgeRetriever

        # Create instances
        self._asr = SpeechRecognizer(
            model_size=self._config.get("whisper_model", "medium"),
        )
        self._retriever = KnowledgeRetriever()
        self._llm = LLMBrain(
            provider=self._config.get("llm_provider", "ollama"),
            model=self._config.get("llm_model"),
            api_key=self._config.get("llm_api_key"),
            persona_name=self._config.get("persona_name", "the user"),
        )
        self._tts = VoiceCloner()
        self._lipsync = LipSyncer()
        self._faceswap = FaceSwapper()
        self._vcam = VirtualCameraOutput(
            width=self._config.get("cam_width", 1280),
            height=self._config.get("cam_height", 720),
            fps=self._config.get("cam_fps", 30),
        )

        # Load models (these are CPU-bound, run in executor to not block event loop)
        loop = asyncio.get_event_loop()

        load_tasks = [
            loop.run_in_executor(None, self._asr.load),
            loop.run_in_executor(None, self._retriever.load),
            loop.run_in_executor(None, self._llm.load),
            loop.run_in_executor(None, self._tts.load),
            loop.run_in_executor(None, self._lipsync.load),
            loop.run_in_executor(None, self._faceswap.load),
        ]

        # Wait for all models to load
        await asyncio.gather(*load_tasks)

        # Set persona-specific data
        persona_id = self._config.get("persona_id", "default")
        self._retriever.set_persona(persona_id)

        reference_voice = self._config.get("reference_voice")
        if reference_voice:
            self._tts.set_reference_voice(reference_voice)

        reference_photo = self._config.get("reference_photo")
        if reference_photo:
            self._faceswap.load_reference(reference_photo)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"All pipeline modules initialized in {elapsed_ms:.0f}ms")

    async def run(self) -> None:
        """
        Run the full pipeline. Blocks until stop() is called.

        Launches all worker tasks concurrently via asyncio.gather().
        """
        self._session_start = time.perf_counter()
        self._vcam.start()
        logger.info("Pipeline running -- avatar is live")

        try:
            await asyncio.gather(
                self._asr_worker(),
                self._rag_llm_worker(),
                self._tts_worker(),
                self._lipsync_worker(),
                self._cam_worker(),
            )
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            self._vcam.stop()

    async def stop(self) -> None:
        """Signal all workers to stop gracefully."""
        logger.info("Stopping pipeline...")
        self._stop_event.set()

    async def shutdown(self) -> None:
        """Unload all models and free resources."""
        elapsed = time.perf_counter() - self._session_start if self._session_start else 0

        if self._asr:
            self._asr.unload()
        if self._retriever:
            self._retriever.unload()
        if self._llm:
            self._llm.unload()
        if self._tts:
            self._tts.unload()
        if self._lipsync:
            self._lipsync.unload()
        if self._faceswap:
            self._faceswap.unload()

        logger.info(
            f"Pipeline shutdown complete. Session: {elapsed:.0f}s, "
            f"{self._utterance_count} utterances processed"
        )

    # --- Worker tasks ---

    async def _asr_worker(self) -> None:
        """Consume mic audio, produce transcribed text."""
        logger.info("ASR worker started")
        while not self._stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(self._mic_queue.get(), timeout=0.5)
                self._asr.feed_audio(chunk)
                self._mic_queue.task_done()

                while self._asr.has_complete_utterance():
                    text = self._asr.get_utterance()
                    if text:
                        await self._text_queue.put(text)
                        logger.info(f"ASR -> '{text[:60]}'")

            except asyncio.TimeoutError:
                # Check for pending utterances even without new audio
                while self._asr.has_complete_utterance():
                    text = self._asr.get_utterance()
                    if text:
                        await self._text_queue.put(text)
                continue
            except Exception as e:
                logger.error(f"ASR worker error: {e}")
                continue

        logger.info("ASR worker stopped")

    async def _rag_llm_worker(self) -> None:
        """
        Consume transcribed text, retrieve knowledge, generate LLM response.

        RAG and LLM are combined in one worker because they are sequential:
        we need the RAG results before calling the LLM.

        The streaming overlap trick happens here: each sentence from the LLM
        is pushed to the TTS queue immediately, so TTS starts speaking while
        the LLM is still generating.
        """
        logger.info("RAG+LLM worker started")
        while not self._stop_event.is_set():
            try:
                question = await asyncio.wait_for(self._text_queue.get(), timeout=1.0)
                self._text_queue.task_done()
                self._utterance_count += 1

                # Step 1: Retrieve relevant knowledge
                start = time.perf_counter()
                chunks = self._retriever.query(question) if self._retriever.is_ready else []
                rag_ms = (time.perf_counter() - start) * 1000
                logger.debug(f"RAG: {len(chunks)} chunks in {rag_ms:.0f}ms")

                # Step 2: Stream LLM response sentence by sentence
                async for sentence in self._llm.generate_stream(question, chunks):
                    await self._sentence_queue.put(sentence)
                    logger.debug(f"LLM -> TTS: '{sentence[:50]}'")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"RAG+LLM worker error: {e}")
                continue

        logger.info("RAG+LLM worker stopped")

    async def _tts_worker(self) -> None:
        """Consume sentences, produce audio chunks in the cloned voice."""
        logger.info("TTS worker started")
        while not self._stop_event.is_set():
            try:
                sentence = await asyncio.wait_for(self._sentence_queue.get(), timeout=1.0)
                self._sentence_queue.task_done()

                if not self._tts.is_ready:
                    logger.warning("TTS not ready, skipping sentence")
                    continue

                # Stream audio chunks for this sentence
                for audio_chunk in self._tts.synthesize_stream(sentence):
                    await self._audio_queue.put(audio_chunk)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")
                continue

        logger.info("TTS worker stopped")

    async def _lipsync_worker(self) -> None:
        """Consume audio chunks, produce lip-synced video frames."""
        logger.info("Lip sync worker started")
        while not self._stop_event.is_set():
            try:
                audio_chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                self._audio_queue.task_done()

                if not self._lipsync.is_ready:
                    logger.warning("LipSyncer not ready, skipping audio chunk")
                    continue

                # Generate lip-synced frames for this audio chunk
                sample_rate = self._tts.sample_rate if self._tts else 22050
                for frame in self._lipsync.sync_stream(audio_chunk, sample_rate):
                    await self._frame_queue.put(frame)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Lip sync worker error: {e}")
                continue

        logger.info("Lip sync worker stopped")

    async def _cam_worker(self) -> None:
        """Consume video frames and push to virtual camera."""
        logger.info("Virtual camera worker started")
        while not self._stop_event.is_set():
            try:
                frame = await asyncio.wait_for(self._frame_queue.get(), timeout=1.0)
                self._vcam.send_frame(frame)
                self._frame_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Virtual camera worker error: {e}")
                continue

        logger.info("Virtual camera worker stopped")

    # --- External input ---

    async def push_audio(self, chunk: np.ndarray) -> None:
        """
        Push a microphone audio chunk into the pipeline.

        Called by the engine's WebSocket handler or audio capture module.

        Args:
            chunk: Audio chunk as float32 numpy array, 16kHz mono.
        """
        await self._mic_queue.put(chunk)

    # --- Status ---

    @property
    def is_running(self) -> bool:
        """Check if the pipeline is running."""
        return self._session_start is not None and not self._stop_event.is_set()

    @property
    def stats(self) -> dict:
        """Return current pipeline stats."""
        elapsed = time.perf_counter() - self._session_start if self._session_start else 0
        return {
            "running": self.is_running,
            "session_duration_s": round(elapsed, 1),
            "utterances_processed": self._utterance_count,
            "queue_sizes": {
                "mic": self._mic_queue.qsize(),
                "text": self._text_queue.qsize(),
                "sentence": self._sentence_queue.qsize(),
                "audio": self._audio_queue.qsize(),
                "frame": self._frame_queue.qsize(),
            },
            "vcam_frames": self._vcam.frame_count if self._vcam else 0,
        }
