# LiveSelf -- Realistic Build Roadmap

Last updated: 2026-03-17

Strategy: 3 agents working in parallel. Each phase produces something demoable. Target: 1 week to WOW demo.

---

## Phase 0: Foundation (Day 1) -- CURRENT

Goal: Project exists. Dev environment works. Proven one tool runs.

- [x] Create GitHub repo + project structure
- [x] Write CLAUDE.md, .env.example, .gitignore
- [ ] Test Deep-Live-Cam on Google Colab with your own photo
- [ ] Set up Supabase project (create account, get keys)
- [ ] First commit pushed to GitHub

Demoable output: Screenshot of your face-swapped in a test video on Colab.

---

## Phase 1A: Face on Zoom (Day 2-3)

Goal: Your photo's face appears as a live webcam on Zoom.

- [ ] Build engine/pipeline/faceswap.py (Deep-Live-Cam wrapper)
- [ ] Build engine/pipeline/virtual_cam.py (pyvirtualcam output)
- [ ] Get virtual camera recognized by Zoom
- [ ] Record 30-second demo clip

Demoable output: 30-second screen recording of your face on Zoom.

---

## Phase 1B: Voice Clone (Day 3-4)

Goal: Avatar speaks in your voice. Lips move correctly.

- [ ] Build engine/pipeline/tts.py (CosyVoice 2 integration)
- [ ] Record 10-30 second voice sample
- [ ] Build engine/pipeline/lipsync.py (MuseTalk integration)
- [ ] Route cloned voice to virtual audio device
- [ ] Face + voice + lip sync working together

Demoable output: Avatar saying a sentence in your voice, lips moving correctly.

---

## Phase 1C: The Brain -- THE WOW DEMO (Day 4-5)

Goal: Full conversational loop. Someone asks a question, avatar answers in your voice.

- [ ] Build engine/pipeline/asr.py (faster-whisper + VAD)
- [ ] Build engine/knowledge/retriever.py (ChromaDB RAG)
- [ ] Build engine/pipeline/llm.py (Ollama streaming)
- [ ] Build engine/pipeline/orchestrator.py (async pipeline coordinator)
- [ ] Create test knowledge base (10 Q&A pairs about yourself)
- [ ] Full end-to-end test: mic to whisper to KB to LLM to voice to lip sync to camera
- [ ] Record 90-second WOW demo video

Demoable output: 90-second video of someone asking your avatar questions and getting real answers in your voice.

Milestone: Show this to Dubai co-founder. Ask for $500.

---

## Phase 2: Dashboard (Day 5-7)

Goal: Other people can use LiveSelf through a web interface.

- [ ] Build Next.js frontend: landing page, auth, setup wizard, dashboard, live session screen
- [ ] Build FastAPI backend: 12 MVP endpoints (auth, personas, sessions, knowledge)
- [ ] Connect frontend to backend to engine
- [ ] GitHub repo goes public with README + demo GIF
- [ ] HackerNews "Show HN" + Reddit launch

Demoable output: Working web app. Upload photo, record voice, click "Go Live."

---

## Phase 3: Cloud + Revenue (Week 2-3)

Goal: Deployed product with paying users.

- [ ] Deploy engine to RunPod serverless
- [ ] Deploy backend to Railway
- [ ] Frontend on Vercel
- [ ] Stripe billing (Free: 60 min/mo, Pro: $20/mo unlimited)
- [ ] TikTok Live RTMP streaming
- [ ] Product Hunt launch
- [ ] Target: first 10 paying users

---

## Risk Mitigations

| Risk | Solution |
|------|----------|
| GPU OOM on full pipeline | Load models sequentially, use INT8 quantized models |
| Virtual camera not recognized | Fall back to OBS Virtual Camera |
| Voice clone sounds robotic | Record 30s sample (not 10s), quiet room |
| Colab session disconnects | Save checkpoints, have RunPod as backup |
| Python 3.13 breaks AI libs | Use Colab/RunPod (Python 3.10/3.11) for engine |
| Timeline slips | Each phase is independent, delay does not kill the project |
