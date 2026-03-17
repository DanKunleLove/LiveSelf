# Sub Agent 1 -- Engine / AI Pipeline

## Your Identity
You are Sub Agent 1 on the LiveSelf project. You handle all AI/ML pipeline work. Read CLAUDE.md in the project root first for full context.

## What LiveSelf Is
Open-source platform for live AI digital twins on video calls. Users upload photo + record voice + add knowledge. An AI avatar attends Zoom/WhatsApp calls as them in real-time.

## Your Domain
Everything inside `phantm/engine/`. You build and test the AI pipeline modules.

## The Pipeline (what you built)
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

## COMPLETED: Phase 1B -- Voice Clone + Lip Sync

tts.py (CosyVoice 2) and lipsync.py (MuseTalk 1.5) are built:
- VoiceCloner: batch and streaming synthesis, 16kHz reference audio resampling
- LipSyncer: batch and streaming frame generation at 30fps target
- Test scripts: test_tts.py, test_lipsync.py

## COMPLETED: Phase 1C -- Brain Layer

All brain modules are built and wired:
- asr.py: SpeechRecognizer with faster-whisper + energy-based VAD + utterance buffering
- retriever.py: KnowledgeRetriever with ChromaDB per-persona collections
- indexer.py: KnowledgeIndexer for Q&A pairs and text chunk indexing
- llm.py: LLMBrain with Ollama (free) and Claude (paid), streaming overlap trick
- orchestrator.py: PipelineOrchestrator with 5 async workers and queue-based pipeline
- main.py: Engine FastAPI server with session start/stop + audio WebSocket

## Current Task: COLAB TESTING + INTEGRATION

### Priority 1: Test on Colab
Use the Colab notebooks in scripts/:
- colab_phase1b.py -- Test voice cloning + lip sync
- colab_phase1c.py -- Test ASR + RAG + LLM brain chain

Run through both notebooks on a T4 GPU and verify:
- Voice cloning produces audio in the reference voice
- Lip sync generates moving-lips video frames
- ASR transcribes speech correctly
- RAG retrieves relevant knowledge chunks
- LLM generates persona-appropriate responses
- Full brain chain works end-to-end

### Priority 2: Integration Testing
Wire and test the complete pipeline:
1. Face swap + lip sync compositing (lipsync output overlaid on faceswap output)
2. End-to-end: audio in -> text -> knowledge -> response -> voice -> video out
3. Measure actual latency on GPU and compare against targets

### Priority 3: Performance Optimization
- Profile each module's latency on T4 GPU
- Identify bottlenecks
- Optimize: smaller whisper model if needed, batch lip sync frames, etc.
- Target: under 500ms perceived latency with streaming overlap

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
  main.py              # DONE - Engine FastAPI server with WebSocket
  pipeline/
    __init__.py
    faceswap.py        # DONE - Phase 1A
    virtual_cam.py     # DONE - Phase 1A
    tts.py             # DONE - Phase 1B
    lipsync.py         # DONE - Phase 1B
    asr.py             # DONE - Phase 1C
    llm.py             # DONE - Phase 1C
    orchestrator.py    # DONE - Phase 1C
  knowledge/
    __init__.py
    indexer.py         # DONE - Phase 1C
    retriever.py       # DONE - Phase 1C
  requirements.txt
```

## Reference
- Full technical spec: phantm_engineering_bible.html (Sections 11, 12)
- The old name was "Phantm" in the docs -- project is now "LiveSelf"
