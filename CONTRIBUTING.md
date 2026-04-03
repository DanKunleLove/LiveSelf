# Contributing to LiveSelf

Thanks for wanting to contribute. This project is actively being built and we welcome help at every level -- from fixing typos to building entire features.

---

## What LiveSelf Is

An open-source platform that creates live AI digital twins (your face, voice, and knowledge) for real-time video calls. The goal is making this accessible to everyone.

Read the [README](README.md) for the full picture and [ROADMAP](phantm/docs/ROADMAP.md) for current progress.

---

## Where Help Is Needed Right Now

### High Priority
- **GPU testing**: Run the Colab notebooks on different GPUs and report results
- **Voice clone pipeline**: Test CosyVoice 2 integration, fix edge cases
- **Lip sync quality**: Improve MuseTalk output quality and frame rate
- **Frontend pages**: Build dashboard, setup wizard, live session views (Next.js 14)

### Medium Priority
- **Knowledge ingestion**: Add PDF, YouTube transcript, and plain text importers
- **Performance**: Reduce pipeline latency, optimize memory usage
- **Backend**: Wire Cloudflare R2 for file uploads, add WebSocket endpoint for live sessions

### Always Welcome
- Bug reports with reproduction steps
- Documentation improvements
- Test coverage
- Code review

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/LiveSelf.git
cd LiveSelf
```

### 2. Pick your area

| Area | Directory | Language | Setup |
|------|-----------|----------|-------|
| AI Engine | `phantm/engine/` | Python 3.11 | `pip install -r phantm/engine/requirements.txt` |
| Backend API | `phantm/backend/` | Python 3.11+ | `pip install -r phantm/backend/requirements.txt` |
| Frontend | `phantm/frontend/` | TypeScript | `cd phantm/frontend && npm install` |
| Colab Tests | `phantm/scripts/` | Python | Open in Google Colab |

### 3. Read the task files

Each development area has a task file describing current work and priorities:

- [MOTHER_AGENT.md](phantm/docs/agents/MOTHER_AGENT.md) -- Architecture and coordination
- [SUB_AGENT_1.md](phantm/docs/agents/SUB_AGENT_1.md) -- AI engine pipeline
- [SUB_AGENT_2.md](phantm/docs/agents/SUB_AGENT_2.md) -- Frontend and backend

These explain what's built, what's in progress, and what's next.

### 4. Run it

**Backend (no GPU needed):**
```bash
cd phantm/backend
cp ../.env.example ../.env  # Fill in values
uvicorn app.main:app --reload --port 8000
```

**Frontend (no GPU needed):**
```bash
cd phantm/frontend
npm run dev
```

**Engine (GPU required -- use Colab if you don't have one):**
```bash
# Option A: Local GPU
cd phantm/engine
pip install -r requirements.txt

# Option B: Google Colab (free T4 GPU)
# Copy cells from phantm/scripts/colab_phase1a.py into Colab
```

---

## Code Standards

### Python (engine + backend)
- Python 3.11 for engine code (must run on Colab/RunPod)
- Type hints on function signatures
- Comments explaining what each function does
- `pytest` for tests
- No `print()` for logging -- use Python `logging` module

### TypeScript (frontend)
- Strict TypeScript -- no `any` types
- Tailwind CSS only -- no custom CSS files
- Shadcn/ui as component base
- All API calls through `src/lib/api.ts`

### General
- No emojis in code, UI, or documentation
- No API keys in code (use `.env`)
- Auth tokens in httpOnly cookies, never localStorage
- Every env variable documented in `.env.example`

---

## Pull Request Process

1. Create a branch from `main` with a descriptive name
   ```bash
   git checkout -b feat/pdf-knowledge-ingestion
   ```

2. Make your changes. Write tests if applicable.

3. Test your changes:
   ```bash
   # Backend
   cd phantm/backend && python -m pytest

   # Engine
   cd phantm/engine && python -m pytest

   # Frontend
   cd phantm/frontend && npm run build
   ```

4. Open a PR against `main`. In your PR description:
   - What you changed and why
   - How to test it
   - Screenshots if it's a UI change

5. One maintainer review required before merge.

---

## Reporting Issues

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (OS, Python version, GPU if relevant)
- Error logs or screenshots

---

## Project Decisions

Some decisions are already made and should not be changed without discussion:

- **Tech stack is locked** (see README). Propose changes in an issue first.
- **No XTTS-v2 or Coqui TTS** -- company shut down, non-commercial license.
- **Concentric circle strategy** -- each phase must work independently.
- **Free tier first** -- everything must work on Google Colab free before paid GPUs.

---

## Questions?

Open an issue or check the [docs/](phantm/docs/) folder for architecture details.
