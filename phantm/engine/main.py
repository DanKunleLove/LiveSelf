"""
LiveSelf Engine -- AI Pipeline Server

This is the FastAPI server that runs on the GPU machine (Colab/RunPod).
It hosts the AI pipeline: face swap, lip sync, voice clone, ASR, LLM, RAG.
The main backend (Railway) calls this service to start/stop avatar sessions.

This file is the entrypoint. Run with: uvicorn main:app --host 0.0.0.0 --port 8001
"""

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liveself.engine")

# Active pipeline orchestrator (one session at a time for now)
_active_orchestrator: Optional[PipelineOrchestrator] = None
_pipeline_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown. Load/unload models here."""
    logger.info("LiveSelf Engine starting up...")
    yield
    logger.info("LiveSelf Engine shutting down...")
    # Clean up any active session
    global _active_orchestrator, _pipeline_task
    if _active_orchestrator:
        await _active_orchestrator.stop()
        await _active_orchestrator.shutdown()
        _active_orchestrator = None
    if _pipeline_task:
        _pipeline_task.cancel()
        _pipeline_task = None


app = FastAPI(
    title="LiveSelf Engine",
    description="AI pipeline for real-time avatar sessions",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/engine/health")
async def health_check():
    """
    Health check endpoint.
    Returns GPU status, loaded models, and memory usage.
    The main backend pings this to know if the engine is ready.
    """
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    session_active = _active_orchestrator is not None and _active_orchestrator.is_running
    stats = _active_orchestrator.stats if session_active else None

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "session_active": session_active,
        "session_stats": stats,
        "version": "0.2.0",
    }


class SessionStartRequest(BaseModel):
    """Request body for starting a session."""
    persona_id: str
    persona_name: str = "the user"
    reference_photo: Optional[str] = None
    reference_voice: Optional[str] = None
    llm_provider: str = "ollama"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    whisper_model: str = "medium"
    cam_width: int = 1280
    cam_height: int = 720
    cam_fps: int = 30


@app.post("/engine/session/start")
async def start_session(request: SessionStartRequest):
    """
    Start an avatar session.
    Loads the face photo, voice model, and knowledge base for this persona.
    Called by the main backend when user clicks "Go Live".
    """
    global _active_orchestrator, _pipeline_task

    if _active_orchestrator and _active_orchestrator.is_running:
        return {"status": "error", "message": "A session is already running. Stop it first."}

    logger.info(f"Starting session for persona: {request.persona_id}")

    config = request.model_dump()
    _active_orchestrator = PipelineOrchestrator(config=config)

    try:
        await _active_orchestrator.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        _active_orchestrator = None
        return {"status": "error", "message": str(e)}

    # Run the pipeline in the background
    _pipeline_task = asyncio.create_task(_active_orchestrator.run())

    return {
        "status": "running",
        "persona_id": request.persona_id,
        "message": "Avatar pipeline started",
    }


@app.post("/engine/session/stop")
async def stop_session():
    """
    Stop the active session.
    Frees GPU memory, closes virtual camera, saves session stats.
    """
    global _active_orchestrator, _pipeline_task

    if not _active_orchestrator:
        return {"status": "error", "message": "No active session"}

    stats = _active_orchestrator.stats
    await _active_orchestrator.stop()

    # Wait for the pipeline task to finish
    if _pipeline_task:
        try:
            await asyncio.wait_for(_pipeline_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _pipeline_task.cancel()

    await _active_orchestrator.shutdown()
    _active_orchestrator = None
    _pipeline_task = None

    logger.info("Session stopped")
    return {"status": "stopped", "stats": stats}


@app.get("/engine/session/stats")
async def session_stats():
    """Get stats for the active session."""
    if not _active_orchestrator:
        return {"status": "no_session"}
    return _active_orchestrator.stats


@app.websocket("/engine/ws/audio")
async def audio_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming microphone audio into the pipeline.

    The frontend captures audio from the caller's microphone and sends
    it here as base64-encoded float32 PCM chunks at 16kHz.

    The pipeline processes the audio through:
    ASR -> RAG -> LLM -> TTS -> Lip Sync -> Virtual Camera
    """
    await websocket.accept()
    logger.info("Audio WebSocket connected")

    if not _active_orchestrator or not _active_orchestrator.is_running:
        await websocket.send_json({"error": "No active session"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_bytes()
            # Decode raw float32 audio bytes
            chunk = np.frombuffer(data, dtype=np.float32)
            await _active_orchestrator.push_audio(chunk)

    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Audio WebSocket error: {e}")
