# LiveSelf -- Build Roadmap

Last updated: 2026-04-02

Strategy: Concentric circles. Each phase produces something independently demoable. Multiple developers working in parallel via task files in `docs/agents/`.

---

## Phase 0: Foundation -- COMPLETE

Goal: Project skeleton exists. Dev environment works. One tool proven on GPU.

- [x] Create GitHub repo + project structure
- [x] Write CLAUDE.md, .env.example, .gitignore
- [x] Set up multi-agent task coordination (docs/agents/)
- [x] First commit pushed to GitHub

---

## Phase 1A: Face on Video -- COMPLETE

Goal: Your photo's face appears as a live video output.

- [x] Build engine/pipeline/faceswap.py (InsightFace wrapper)
- [x] Build engine/pipeline/virtual_cam.py (pyvirtualcam + file fallback)
- [x] Test on Google Colab T4 GPU
- [x] Download face-swapped output video

Proven: Face swap runs at real-time speed on Colab T4. Output video saved.

---

## Phase 1B: Voice Clone + Lip Sync -- CODE COMPLETE, TESTING

Goal: Avatar speaks in your cloned voice. Lips move to match.

- [x] Build engine/pipeline/tts.py (CosyVoice 2 voice cloning)
- [x] Build engine/pipeline/lipsync.py (MuseTalk 1.5 lip sync)
- [x] Write Colab notebook with dependency fixes (colab_phase1b.py)
- [ ] Run Phase 1B Colab test end-to-end
- [ ] Confirm voice quality and lip sync accuracy
- [ ] Measure latency (target: real-time on T4)

Key dependency fix: openai-whisper must install from GitHub source, not pypi.

---

## Phase 1C: The Brain -- THE WOW DEMO -- BRAIN WORKING, INTEGRATION NEXT

Goal: Full conversational loop. Someone asks a question, avatar answers in your voice with moving lips.

- [x] Build engine/pipeline/asr.py (faster-whisper + VAD)
- [x] Build engine/knowledge/retriever.py (ChromaDB RAG)
- [x] Build engine/knowledge/indexer.py (knowledge base builder)
- [x] Build engine/pipeline/llm.py (Ollama streaming + Claude fallback)
- [x] Build engine/pipeline/orchestrator.py (async pipeline coordinator)
- [x] Test brain chain on Colab: ASR -> RAG -> LLM streams responses
- [ ] Wire brain output to TTS + lip sync + face swap
- [ ] Full end-to-end: mic to avatar speaking in your voice
- [ ] Record 90-second WOW demo video

Proven: Llama 3 streams multi-sentence responses using ChromaDB knowledge on Colab T4.

---

## Phase 2: Web Dashboard -- IN PROGRESS

Goal: Other people can use LiveSelf through a web interface.

- [x] Build FastAPI backend: 12 MVP endpoints (auth, personas, sessions)
- [x] Pydantic models, Supabase client, auth middleware
- [x] Initialize Next.js 14 frontend scaffold
- [ ] API client + auth (typed fetch wrapper, httpOnly cookies)
- [ ] Landing page with "How it works" section
- [ ] Auth pages (login, register)
- [ ] Dashboard with persona cards and "Go Live" button
- [ ] Setup wizard (upload photo, record voice, add knowledge)
- [ ] Live session page with status HUD
- [ ] Connect frontend to backend to engine via WebSocket

---

## Phase 3: Cloud + Revenue -- PLANNED

Goal: Deployed product with paying users.

- [ ] Deploy engine to RunPod serverless (RTX 4090, $0.44/hr)
- [ ] Deploy backend to Railway
- [ ] Frontend on Vercel
- [ ] Stripe billing (Free: 60 min/mo, Pro: $20/mo unlimited)
- [ ] TikTok Live RTMP streaming
- [ ] Product Hunt + HackerNews launch
- [ ] Target: first 10 paying users

---

## How to Contribute

Each phase has specific tasks you can pick up. See:
- [CONTRIBUTING.md](../../CONTRIBUTING.md) for setup instructions
- [docs/agents/](agents/) for detailed task breakdowns by area

Good areas for new contributors:
- Test Colab notebooks on different GPUs and report results
- Frontend pages (Next.js 14 + TypeScript + Tailwind)
- Knowledge ingestion (PDF, YouTube transcripts)
- Documentation and code comments

---

## Risk Mitigations

| Risk | Solution |
|------|----------|
| GPU OOM on full pipeline | Load models sequentially, INT8 quantization, unload between stages |
| Virtual camera not recognized | Fall back to OBS Virtual Camera or file output |
| Voice clone sounds robotic | Record 30s sample (not 10s), quiet room, high quality mic |
| Colab session disconnects | Save checkpoints frequently, RunPod as backup |
| Python 3.13 breaks AI libs | Engine uses Python 3.10/3.11 on Colab/RunPod |
| openai-whisper won't install | Install from GitHub source, not pypi |
| MuseTalk needs newer torch | Force upgrade to torch 2.4+ with CUDA 12.1 |
| Timeline slips | Each phase is independent -- delay does not block other work |
