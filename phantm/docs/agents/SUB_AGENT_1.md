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

## COMPLETED: Phase 1A -- Face Swap

faceswap.py and virtual_cam.py are built. Phase 0 Colab test produced output.mp4 (4MB, working face swap). Models verified working:
- inswapper_128_fp16.onnx from https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx
- GFPGANv1.4.onnx from https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx
- InsightFace buffalo_l model (auto-downloaded by insightface library)

LICENSE NOTE: InsightFace inswapper model is non-commercial research only. This is fine for open source / self-hosted. For paid cloud version (Phase 3), we need to address this.

## Current Task: PHASE 1B -- Voice Clone + Lip Sync

### What to build (in order):

**Step 1: tts.py** (`phantm/engine/pipeline/tts.py`) -- CosyVoice 2
- Clone CosyVoice repo: git clone https://github.com/FunAudioLLM/CosyVoice.git
- Install: cd CosyVoice && pip install -r requirements.txt
- Load pre-trained model (CosyVoice2-0.5B recommended for speed)
- Accept: text string + reference voice audio path (10-30 seconds WAV)
- Return: synthesized audio as numpy array in the user's cloned voice
- Must support streaming: yield audio chunks as they are generated
- Test: input "Hello, I am your AI twin" -> output WAV file in cloned voice

**Step 2: lipsync.py** (`phantm/engine/pipeline/lipsync.py`) -- MuseTalk 1.5
- Clone MuseTalk repo: git clone https://github.com/TMElyralab/MuseTalk.git
- Install: follow their setup instructions
- Accept: audio waveform + face reference frame
- Return: video frames with lips moving to match the audio
- Target: 30fps output
- Test: input audio + face photo -> output video with moving lips

**Step 3: Integration test**
- Wire: tts output audio -> lipsync input
- Wire: lipsync output frames -> virtual_cam
- Test: type text -> hear it in your voice -> see lips move on avatar

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
