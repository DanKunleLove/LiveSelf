# LiveSelf

**Your AI twin on Zoom, WhatsApp, and Google Meet. Open source. Free.**

LiveSelf lets you create a live AI version of yourself -- your face, your voice, your knowledge -- that attends real-time video calls for you. Upload a photo, record 10 seconds of your voice, add your knowledge, and your AI twin handles the rest.

---

## What's Working Right Now

| Component | Status | Tested On |
|-----------|--------|-----------|
| Face swap (InsightFace) | Working | Colab T4 GPU |
| Voice clone (CosyVoice 2) | Code complete | Colab test pending |
| Lip sync (MuseTalk 1.5) | Code complete | Colab test pending |
| Speech-to-text (faster-whisper) | Working | Colab T4 GPU |
| AI brain (Ollama + Llama 3) | Working | Colab T4 GPU |
| Knowledge retrieval (ChromaDB) | Working | Colab T4 GPU |
| Pipeline orchestrator | Code complete | Integration test pending |
| Backend API (FastAPI) | 12 endpoints built | Local |
| Frontend (Next.js 14) | Scaffold created | Local |

**Latest demo**: Face swap running on Google Colab with a T4 GPU. The AI brain (ASR + RAG + LLM chain) streams responses using your knowledge base.

---

## How It Works

```
Microphone
    |
    v
faster-whisper (speech-to-text, ~150ms)
    |
    v
ChromaDB (retrieve your knowledge, ~50ms)
    |
    v
Llama 3 via Ollama (generate response, ~300ms to first sentence)
    |
    v
CosyVoice 2 (speak in your voice)
    |
    v
MuseTalk 1.5 (sync lips to audio)
    |
    v
InsightFace (swap face onto frame)
    |
    v
Virtual Camera --> Zoom / Google Meet / WhatsApp
```

Total latency target: **~500ms to first spoken word** (with streaming overlap).

---

## Tech Stack

Every component is open source with permissive licenses.

| Layer | Tool | What It Does | License |
|-------|------|-------------|---------|
| Face Swap | [InsightFace](https://github.com/deepinsight/insightface) | Swaps your face onto a video feed | MIT* |
| Lip Sync | [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) | Makes lips move to match speech | MIT |
| Voice Clone | [CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice) | Clones your voice from 10s sample | Apache 2.0 |
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Transcribes incoming speech | MIT |
| AI Brain | [Ollama](https://github.com/ollama/ollama) + Llama 3 | Generates conversational responses | MIT |
| Knowledge Base | [ChromaDB](https://github.com/chroma-core/chroma) | Stores and retrieves your knowledge | Apache 2.0 |
| Virtual Camera | [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) | Outputs video feed to Zoom/Meet | MIT |
| Backend | [FastAPI](https://github.com/tiangolo/fastapi) | API server | MIT |
| Frontend | [Next.js 14](https://github.com/vercel/next.js) | Web dashboard | MIT |
| Database | [Supabase](https://github.com/supabase/supabase) | Auth + PostgreSQL | Apache 2.0 |

*InsightFace inswapper model is non-commercial research only. Fine for open source, needs addressing for paid deployment.

---

## Project Structure

```
LiveSelf/
  phantm/
    engine/                    # AI pipeline (the core)
      pipeline/
        faceswap.py            # InsightFace face swap
        tts.py                 # CosyVoice 2 voice cloning
        lipsync.py             # MuseTalk 1.5 lip sync
        asr.py                 # faster-whisper speech-to-text
        llm.py                 # Ollama/Claude LLM brain
        orchestrator.py        # Async pipeline coordinator
        virtual_cam.py         # Virtual camera output
      knowledge/
        retriever.py           # ChromaDB query
        indexer.py             # Knowledge base builder
      models/                  # Downloaded model weights (gitignored)

    backend/                   # FastAPI API server
      app/
        routers/               # auth, personas, sessions
        models/                # Pydantic schemas
        middleware/             # JWT auth
        database/              # Supabase client

    frontend/                  # Next.js 14 dashboard
      src/

    scripts/                   # Setup and test scripts
      colab_phase1a.py         # Colab notebook: face swap test
      colab_phase1b.py         # Colab notebook: voice + lip sync test
      colab_phase1c.py         # Colab notebook: brain test

    docs/
      ROADMAP.md               # Build phases and progress
      agents/                  # Task files for parallel development
```

---

## Build Strategy: Concentric Circles

Each circle is independently demoable. You can stop at any circle and have something that works.

```
Circle 1: Face swap on Zoom                    <-- DONE
Circle 2: Face + voice clone + lip sync        <-- Code complete, testing
Circle 3: Face + voice + AI brain (WOW demo)   <-- Brain working, integration next
Circle 4: Web dashboard for anyone to use       <-- In progress
Circle 5: Cloud deployment + billing            <-- Planned
```

See [ROADMAP.md](phantm/docs/ROADMAP.md) for detailed progress on each phase.

---

## Try It Yourself (Google Colab)

No GPU? No problem. Run the pipeline on Google Colab's free T4 GPU.

**Phase 1A -- Face Swap:**
1. Open Google Colab
2. Copy cells from [colab_phase1a.py](phantm/scripts/colab_phase1a.py)
3. Upload your face photo
4. Download the face-swapped video

**Phase 1C -- AI Brain:**
1. Copy cells from [colab_phase1c.py](phantm/scripts/colab_phase1c.py)
2. Pulls Llama 3 locally on Colab, sets up ChromaDB
3. Ask questions, get streamed responses using your knowledge

Full setup instructions in each notebook.

---

## Contributing

We are actively looking for contributors. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

**Areas where help is needed:**
- Testing the voice clone + lip sync pipeline on different GPUs
- Frontend development (Next.js 14, TypeScript, Tailwind)
- Adding new knowledge ingestion formats (PDF, YouTube, etc.)
- Performance optimization (latency reduction, memory usage)
- Documentation and tutorials

Current development tasks are tracked in [phantm/docs/agents/](phantm/docs/agents/).

---

## Development Setup

**Requirements:**
- Python 3.11+ (engine/backend)
- Node.js 18+ (frontend)
- NVIDIA GPU with 8GB+ VRAM (or use Google Colab)

```bash
# Clone
git clone https://github.com/DanKunleLove/LiveSelf.git
cd LiveSelf

# Backend
cd phantm/backend
pip install -r requirements.txt
cp ../.env.example ../.env  # Fill in your keys
uvicorn app.main:app --reload

# Frontend
cd phantm/frontend
npm install
npm run dev

# Engine (requires GPU)
cd phantm/engine
pip install -r requirements.txt
```

For GPU testing without a local GPU, use the Colab notebooks in `phantm/scripts/`.

---

## Architecture

```
User on Zoom/Meet
       |
       v
  [Virtual Camera] <-- pyvirtualcam
       ^
       |
  [Face Swap] <-- InsightFace inswapper
       ^
       |
  [Lip Sync] <-- MuseTalk 1.5
       ^
       |
  [Voice Clone] <-- CosyVoice 2
       ^
       |
  [LLM Brain] <-- Ollama + Llama 3
       ^
       |
  [Knowledge] <-- ChromaDB RAG
       ^
       |
  [ASR] <-- faster-whisper
       ^
       |
  Incoming audio from call
```

All pipeline stages run concurrently via asyncio. Each stage communicates through async queues. The orchestrator coordinates startup, shutdown, and error recovery.

---

## License

MIT -- free to use, modify, and commercialize.

---

Built by [@DanKunleLove](https://github.com/DanKunleLove) with AI-assisted development (Claude Code).
