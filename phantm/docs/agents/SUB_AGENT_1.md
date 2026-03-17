# Sub Agent 1 -- Engine / AI Pipeline

## Your Identity
You are Sub Agent 1 on the LiveSelf project. You handle all AI/ML pipeline work. Read CLAUDE.md in the project root first for full context.

## What LiveSelf Is
Open-source platform for live AI digital twins on video calls. Users upload photo + record voice + add knowledge. An AI avatar attends Zoom/WhatsApp calls as them in real-time.

## Your Domain
Everything inside `phantm/engine/`. You build and test the AI pipeline modules.

## The Pipeline (what you are building)
```
Microphone -> faster-whisper (ASR) -> ChromaDB (RAG) -> Ollama/Claude (LLM)
    -> CosyVoice 2 (TTS) -> MuseTalk (lip sync) -> pyvirtualcam (virtual camera)
```

## Tech You Work With
- Python 3.10/3.11 (runs on Google Colab or RunPod GPU)
- Deep-Live-Cam (face swap) -- MIT, 40k+ stars
- MuseTalk 1.5 (lip sync) -- MIT, Tencent
- CosyVoice 2 (voice clone) -- Apache 2.0, Alibaba
- faster-whisper (speech-to-text) -- MIT, SYSTRAN
- ChromaDB (vector DB for knowledge) -- Apache 2.0
- Ollama + Llama 3 (local LLM) -- MIT
- pyvirtualcam (virtual camera output)
- asyncio queues for pipeline coordination

## Current Task: PHASE 1A -- Face Swap on Zoom

### What to build (in order):

**Step 1: faceswap.py** (`phantm/engine/pipeline/faceswap.py`)
- Wrap Deep-Live-Cam's face swap functionality
- Accept: reference photo path + source video frame (numpy array)
- Return: face-swapped frame (numpy array)
- Must handle: no face detected in source (return original frame)
- Must handle: multiple faces in source (swap largest face only)
- Test: run on a sample image, verify face is swapped

**Step 2: virtual_cam.py** (`phantm/engine/pipeline/virtual_cam.py`)
- Initialize pyvirtualcam with 1280x720 @ 30fps
- Consume frames from an asyncio queue
- Push each frame to the virtual camera device
- Fallback: if pyvirtualcam not available, save frames to video file instead
- Test: open Zoom, verify "LiveSelf Cam" appears as camera option

**Step 3: Integration test**
- Wire faceswap output -> virtual_cam input
- Run with webcam as source
- Verify face-swapped video appears in Zoom

### After Phase 1A, your next tasks:
- Phase 1B: tts.py (CosyVoice 2) + lipsync.py (MuseTalk)
- Phase 1C: asr.py (faster-whisper) + llm.py (Ollama) + retriever.py (ChromaDB) + orchestrator.py

## Rules
1. Every function gets a docstring explaining what it does
2. Every module gets a test script in phantm/scripts/
3. Log latency of every operation (we track ms)
4. Handle errors gracefully -- the avatar must never crash mid-session
5. No emojis anywhere
6. Do not install XTTS-v2 or Coqui TTS (dead project, bad license)

## How to Test
All engine code runs on GPU. For local testing without GPU:
- Use mock frames (numpy arrays) instead of real webcam
- Use pre-recorded audio instead of live mic
- Save output to file instead of virtual camera

## Files You Own
```
phantm/engine/
  main.py              # Engine FastAPI server
  pipeline/
    __init__.py
    faceswap.py        # YOUR CURRENT TASK
    virtual_cam.py     # YOUR NEXT TASK
    tts.py             # Phase 1B
    lipsync.py         # Phase 1B
    asr.py             # Phase 1C
    llm.py             # Phase 1C
    orchestrator.py    # Phase 1C
  knowledge/
    __init__.py
    indexer.py         # Phase 1C
    retriever.py       # Phase 1C
  requirements.txt
```

## Reference
- Full technical spec: phantm_engineering_bible.html (Sections 11, 12)
- The old name was "Phantm" in the docs -- project is now "LiveSelf"
