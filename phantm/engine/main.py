"""
LiveSelf Engine — AI Pipeline Server

This is the FastAPI server that runs on the GPU machine (Colab/RunPod).
It hosts the AI pipeline: face swap, lip sync, voice clone, ASR, LLM, RAG.
The main backend (Railway) calls this service to start/stop avatar sessions.

This file is the entrypoint. Run with: uvicorn main:app --host 0.0.0.0 --port 8001
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import logging

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liveself.engine")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown. Load/unload models here."""
    logger.info("LiveSelf Engine starting up...")
    # TODO: Pre-load any models that should be ready immediately
    yield
    logger.info("LiveSelf Engine shutting down...")
    # TODO: Clean up GPU memory, close connections


app = FastAPI(
    title="LiveSelf Engine",
    description="AI pipeline for real-time avatar sessions",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/engine/health")
async def health_check():
    """
    Health check endpoint.
    Returns GPU status, loaded models, and memory usage.
    The main backend pings this to know if the engine is ready.
    """
    # TODO: Add real GPU stats via psutil + torch.cuda
    return {
        "status": "ok",
        "gpu_available": False,  # Will be True when running on GPU
        "models_loaded": [],
        "version": "0.1.0",
    }


@app.post("/engine/session/start")
async def start_session(persona_id: str):
    """
    Start an avatar session.
    Loads the face photo, voice model, and knowledge base for this persona.
    Called by the main backend when user clicks "Go Live".
    """
    # TODO: Implement in Phase 1C
    logger.info(f"Starting session for persona: {persona_id}")
    return {"status": "not_implemented_yet", "message": "Phase 1 — build pipeline modules first"}


@app.post("/engine/session/stop")
async def stop_session(session_id: str):
    """
    Stop an active session.
    Frees GPU memory, closes virtual camera, saves session stats.
    """
    # TODO: Implement in Phase 1C
    logger.info(f"Stopping session: {session_id}")
    return {"status": "stopped"}


# WebSocket endpoint will be added in Phase 1C
# For now, the individual pipeline modules are what we build and test
