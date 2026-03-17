# LiveSelf -- AI Live Digital Twin Platform

## What This Is
Open-source platform that lets anyone create a live AI version of themselves (face, voice, knowledge) for real-time video calls on Zoom, WhatsApp, Google Meet. Eventually streams live on TikTok/YouTube.

## Who I Am
Non-technical builder working with AI tools. Explain every step in plain English BEFORE coding. Build one piece at a time. Be honest about complexity and timelines.

## Multi-Agent Development
This project uses multiple Claude Code agents working in parallel:
- **Mother Agent**: Architecture, coordination, code review, final decisions
- **Sub Agent 1**: Engine/AI Pipeline (Python, GPU, ML models)
- **Sub Agent 2**: Frontend + Backend (Next.js, FastAPI, Supabase)

Each agent has a task file in `phantm/docs/agents/` describing their role and current task.

## Current Phase
**Phase 0 -> Phase 1A (Face on Zoom)**
Building the AI pipeline first. Dashboard comes after the WOW demo works.

## Build Strategy: Concentric Circles
Each circle is independently demoable:
1. Face swap on Zoom (Phase 1A)
2. Face + voice clone (Phase 1B)
3. Face + voice + AI brain (Phase 1C -- THE WOW DEMO)
4. Web dashboard for others to use (Phase 2)
5. Cloud deployment + billing (Phase 3)

## Tech Stack -- DO NOT CHANGE WITHOUT FLAGGING
| Layer | Tool | License |
|-------|------|---------|
| Face swap | Deep-Live-Cam (hacksider/Deep-Live-Cam) | MIT |
| Lip sync | MuseTalk 1.5 (TMElyralab/MuseTalk) | MIT |
| Voice clone | CosyVoice 2 (FunAudioLLM/CosyVoice) | Apache 2.0 |
| ASR (ears) | faster-whisper (SYSTRAN/faster-whisper) | MIT |
| LLM brain | Ollama + Llama 3 (local free) then Claude API (premium) | MIT |
| Knowledge | ChromaDB (chroma-core/chroma) | Apache 2.0 |
| Virtual cam | pyvirtualcam + OBS Virtual Camera | MIT / GPL |
| Frontend | Next.js 14 + TypeScript + Tailwind + Shadcn/ui | MIT |
| Backend | FastAPI (Python 3.11) | MIT |
| Database | Supabase (PostgreSQL) | Apache 2.0 |
| Storage | Cloudflare R2 | -- |
| GPU | Google Colab (dev) then RunPod RTX 4090 (demo/prod) | -- |
| Image Gen | Google AI Studio (Gemini) for project assets | -- |

## Project Structure
```
phantm/
  frontend/          # Next.js 14 dashboard (Phase 2)
  backend/           # FastAPI API server
    app/
      routers/       # API route handlers
      models/        # Pydantic request/response models
      services/      # Business logic (engine client, storage)
      middleware/     # Auth, rate limiting
      database/      # Supabase connection
  engine/            # AI pipeline (Phase 1 focus)
    pipeline/        # orchestrator, asr, rag, llm, tts, lipsync, faceswap, virtual_cam
    knowledge/       # indexer, retriever (ChromaDB)
    models/          # Downloaded weights (gitignored)
  scripts/           # Setup, dev, test scripts
  docs/
    agents/          # Task files for each sub-agent
    README.md
    ROADMAP.md
```

## Code Rules
1. Python 3.11 for engine (on Colab/RunPod). Python 3.13 OK for backend locally.
2. TypeScript everywhere on frontend -- no `any` types
3. All API calls via typed client in `lib/api.ts`
4. Tailwind classes only -- no custom CSS files
5. Comments on every function explaining what it does
6. Every env variable in `.env.example` with description
7. NO emojis, decorative icons, or stock images anywhere in the project

## NEVER Do
- Store API keys in code (use .env)
- Use localStorage for auth tokens (httpOnly cookies only)
- Write raw SQL (use Supabase client)
- Skip error handling
- Add features not needed for the current phase
- Use XTTS-v2/Coqui TTS (company shut down, non-commercial license)
- Add emojis or decorative elements to code, docs, or UI
- Commit .env files or API keys

## Cost Awareness
Flag any paid service immediately with cost. Current budget: ~$0.
- GPU: Google Colab free first then RunPod ($0.44/hr) for demos
- Everything else must be free tier

## Testing
- Engine: pytest + test scripts that verify each pipeline stage independently
- Backend: pytest + httpx test client
- Frontend: Vitest + React Testing Library (Phase 2)

## Reference Docs
- Full blueprint: phantm_complete_blueprint.html (vision, tools, costs, viral strategy)
- Engineering bible: phantm_engineering_bible.html (schemas, APIs, components, wireframes)
- These are the NORTH STAR, not today's implementation target
- Note: These docs use the old name "Phantm" -- the project is now called "LiveSelf"
