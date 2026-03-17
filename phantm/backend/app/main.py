"""
LiveSelf Backend — API Server

This is the main FastAPI server that handles:
- User authentication (Supabase Auth)
- Persona management (CRUD)
- Knowledge base management
- Session management
- Communication with the GPU engine service

Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liveself.backend")

app = FastAPI(
    title="LiveSelf API",
    description="Backend API for LiveSelf -- AI Live Digital Twin Platform",
    version="0.1.0",
)

# Allow frontend to talk to backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    """Basic health check. Returns OK if the server is running."""
    return {"status": "ok", "version": "0.1.0"}


# -- Routers --
from app.routers import auth, personas, sessions

app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(personas.router, prefix="/api/personas", tags=["Personas"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])

# Knowledge router will be added when knowledge base CRUD is built
# from app.routers import knowledge
# app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
