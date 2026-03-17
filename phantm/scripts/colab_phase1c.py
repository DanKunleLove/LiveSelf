"""
LiveSelf -- Phase 1C Colab Notebook: Brain Layer (ASR + RAG + LLM)

Copy each section below into a separate Google Colab cell.
Runtime -> Change runtime type -> T4 GPU

This notebook:
  1. Installs faster-whisper, ChromaDB, and Ollama
  2. Tests speech-to-text on a recorded audio sample
  3. Indexes knowledge into ChromaDB
  4. Tests RAG retrieval
  5. Tests LLM response generation with knowledge context
  6. Runs the full brain chain: ASR -> RAG -> LLM -> TTS

After this works, you have a complete avatar brain.
"""

# ==============================================
# CELL 1: Check GPU and install dependencies
# ==============================================
"""
import torch
import sys
print(f"Python: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Install brain layer dependencies
!pip install faster-whisper chromadb ollama soundfile numpy

# Install Ollama (local LLM server)
!curl -fsSL https://ollama.com/install.sh | sh
print("\\nDependencies installed.")
"""

# ==============================================
# CELL 2: Start Ollama and pull Llama 3
# ==============================================
"""
import subprocess
import time

# Start Ollama in the background
proc = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(3)  # Give it time to start

# Pull Llama 3 model (first run downloads ~4 GB)
print("Pulling Llama 3 model (this takes a few minutes on first run)...")
!ollama pull llama3
print("Llama 3 ready.")

# Verify
!ollama list
"""

# ==============================================
# CELL 3: Clone LiveSelf repo
# ==============================================
"""
import os

if not os.path.exists("LiveSelf"):
    !git clone https://github.com/DanKunleLove/LiveSelf.git

%cd LiveSelf/phantm
print("LiveSelf repo ready.")
!ls engine/pipeline/
"""

# ==============================================
# CELL 4: Test ASR (Speech-to-Text)
# ==============================================
"""
import sys
import time
import numpy as np
sys.path.insert(0, "engine")

from pipeline.asr import SpeechRecognizer, SAMPLE_RATE

# Load faster-whisper (medium model for good accuracy)
print("Loading faster-whisper model...")
recognizer = SpeechRecognizer(model_size="medium")
start = time.perf_counter()
recognizer.load()
print(f"Model loaded in {time.perf_counter() - start:.1f}s")

# Option A: Upload a voice recording to transcribe
# from google.colab import files
# print("Upload a WAV file with someone speaking:")
# uploaded = files.upload()
# audio_file = list(uploaded.keys())[0]
# import soundfile as sf
# audio, sr = sf.read(audio_file)
# if sr != 16000:
#     import torchaudio
#     audio_tensor = torch.from_numpy(audio).float()
#     if audio_tensor.dim() == 1:
#         audio_tensor = audio_tensor.unsqueeze(0)
#     resampler = torchaudio.transforms.Resample(sr, 16000)
#     audio = resampler(audio_tensor).squeeze().numpy()

# Option B: Generate test audio (sine wave -- won't produce real text)
# For a real test, uncomment Option A above and upload a recording
print("\\nUsing test audio (uncomment Option A above for real test)...")
duration = 3.0
t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
test_audio = np.sin(2 * np.pi * 440 * t) * 0.3

# Transcribe
print("Transcribing...")
start = time.perf_counter()
text = recognizer.transcribe_segment(test_audio)
elapsed_ms = (time.perf_counter() - start) * 1000
print(f"Transcription ({elapsed_ms:.0f}ms): '{text}'")
print("(Sine wave won't produce text -- upload a real recording for real results)")

# Test feed_audio with simulated speech
print("\\nTesting utterance detection (feed_audio)...")
# Simulate a speech segment followed by silence
speech = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.1  # 2s of noise
silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2s silence

# Feed in chunks
chunk_size = int(SAMPLE_RATE * 0.5)
for i in range(0, len(speech), chunk_size):
    recognizer.feed_audio(speech[i:i+chunk_size])
for i in range(0, len(silence), chunk_size):
    recognizer.feed_audio(silence[i:i+chunk_size])

if recognizer.has_complete_utterance():
    result = recognizer.get_utterance()
    print(f"Detected utterance: '{result}'")
else:
    print("No utterance detected (expected with noise input)")

print("\\nASR module: WORKING")
"""

# ==============================================
# CELL 5: Test RAG (Knowledge Retrieval)
# ==============================================
"""
import sys
import time
sys.path.insert(0, "engine")

from knowledge.retriever import KnowledgeRetriever
from knowledge.indexer import KnowledgeIndexer

# Initialize ChromaDB
print("Loading ChromaDB...")
retriever = KnowledgeRetriever(chroma_path="/tmp/liveself_test_chroma")
retriever.load()
retriever.set_persona("colab_test")
print(f"ChromaDB ready. Collection count: {retriever.get_collection_count()}")

# Add knowledge about you (replace with your own info!)
indexer = KnowledgeIndexer(retriever)

qa_pairs = [
    {"q": "What is your name?", "a": "My name is Daniel. I go by Dan."},
    {"q": "Where are you from?", "a": "I am from Lagos, Nigeria."},
    {"q": "What do you do?", "a": "I am a final year Physics and Computer Science student. I also build AI projects."},
    {"q": "What is LiveSelf?", "a": "LiveSelf is an open-source platform I am building that lets anyone create an AI digital twin for video calls."},
    {"q": "What programming languages do you know?", "a": "I mainly work with Python and JavaScript. I use AI tools like Claude Code to help me build faster."},
    {"q": "What is your goal?", "a": "I want to make AI accessible to everyone in Africa and beyond. I believe AI digital twins will change how we communicate."},
    {"q": "What are your hobbies?", "a": "I enjoy learning about physics, building tech projects, and playing football."},
]

count = indexer.add_qa_pairs(qa_pairs)
print(f"Indexed {count} Q&A pairs")
print(f"Collection count: {retriever.get_collection_count()}")

# Test queries
print("\\n--- Testing RAG Retrieval ---")
test_questions = [
    "What do you do for a living?",
    "Tell me about yourself",
    "Where are you based?",
    "What is your project about?",
]

for question in test_questions:
    start = time.perf_counter()
    chunks = retriever.query(question, top_k=2)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"\\nQ: '{question}' ({elapsed_ms:.0f}ms)")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk[:100]}")

print("\\nRAG module: WORKING")
"""

# ==============================================
# CELL 6: Test LLM (Ollama + Llama 3)
# ==============================================
"""
import asyncio
import sys
import time
sys.path.insert(0, "engine")

from pipeline.llm import LLMBrain

# Initialize LLM with Ollama
print("Connecting to Ollama...")
brain = LLMBrain(
    provider="ollama",
    model="llama3",
    persona_name="Dan",
)
brain.load()
print("LLM ready.")

# Test with knowledge context
knowledge = [
    "Q: What is your name?\\nA: My name is Daniel. I go by Dan.",
    "Q: What do you do?\\nA: I am a final year Physics and Computer Science student. I also build AI projects.",
    "Q: What is LiveSelf?\\nA: LiveSelf is an open-source platform I am building that lets anyone create an AI digital twin for video calls.",
]

# Test streaming response
async def test_streaming():
    print("\\n--- Streaming Response Test ---")
    question = "Hey Dan, what are you working on?"
    print(f"Question: '{question}'")
    print("Response (streamed sentence by sentence):")

    start = time.perf_counter()
    sentence_count = 0
    full_response = ""

    async for sentence in brain.generate_stream(question, knowledge):
        sentence_count += 1
        full_response += sentence + " "
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  [{elapsed:.0f}ms] {sentence}")

    print(f"\\nTotal: {sentence_count} sentences in {(time.perf_counter() - start) * 1000:.0f}ms")
    return full_response

response = asyncio.run(test_streaming())

# Test another question
async def test_question2():
    print("\\n--- Second Question ---")
    question = "Where are you from and what do you study?"
    print(f"Question: '{question}'")
    response = await brain.generate(question, knowledge)
    print(f"Response: {response}")

asyncio.run(test_question2())

print("\\nLLM module: WORKING")
"""

# ==============================================
# CELL 7: Full Brain Chain -- ASR -> RAG -> LLM
# ==============================================
"""
import asyncio
import sys
import time
import numpy as np
sys.path.insert(0, "engine")

from pipeline.asr import SpeechRecognizer, SAMPLE_RATE
from knowledge.retriever import KnowledgeRetriever
from pipeline.llm import LLMBrain

print("=== Full Brain Chain Test ===")
print("Pipeline: Audio -> ASR -> RAG -> LLM -> Text Response\\n")

# Step 1: Transcribe audio
# For real test, upload a recording of someone asking a question
# For now, we simulate with a pre-set question

# OPTION A: Use a real audio recording (uncomment below)
# from google.colab import files
# print("Upload a WAV recording of someone asking 'What do you do?':")
# uploaded = files.upload()
# audio_file = list(uploaded.keys())[0]
# import soundfile as sf
# audio, sr = sf.read(audio_file)
# start = time.perf_counter()
# question = recognizer.transcribe_segment(audio.astype(np.float32))
# asr_ms = (time.perf_counter() - start) * 1000
# print(f"[ASR] Transcribed in {asr_ms:.0f}ms: '{question}'")

# OPTION B: Simulate ASR output (using pre-set question)
question = "What are you working on and where are you from?"
print(f"[ASR] Simulated transcription: '{question}'")

# Step 2: Retrieve knowledge
start = time.perf_counter()
chunks = retriever.query(question, top_k=3)
rag_ms = (time.perf_counter() - start) * 1000
print(f"[RAG] Retrieved {len(chunks)} chunks in {rag_ms:.0f}ms")
for i, chunk in enumerate(chunks):
    print(f"  Context [{i+1}]: {chunk[:80]}...")

# Step 3: Generate LLM response
async def brain_chain():
    start = time.perf_counter()
    sentence_count = 0
    print(f"\\n[LLM] Generating response as Dan...")

    async for sentence in brain.generate_stream(question, chunks):
        sentence_count += 1
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  [{elapsed:.0f}ms] Sentence {sentence_count}: {sentence}")

    total_ms = (time.perf_counter() - start) * 1000
    print(f"\\n[LLM] Done: {sentence_count} sentences in {total_ms:.0f}ms")
    print(f"\\n--- Latency Summary ---")
    print(f"  ASR:  simulated (real: ~150ms)")
    print(f"  RAG:  {rag_ms:.0f}ms")
    print(f"  LLM:  {total_ms:.0f}ms (first sentence much faster)")
    print(f"  TTS:  not tested here (see Phase 1B notebook)")
    print(f"  Lip Sync: not tested here (see Phase 1B notebook)")

asyncio.run(brain_chain())

print("\\n=== Brain Chain: WORKING ===")
print("Next step: Wire this to TTS + Lip Sync for a complete speaking avatar")
"""

# ==============================================
# CELL 8: Full Pipeline Test (Brain + Voice + Lips)
# ==============================================
"""
# This cell requires Phase 1B to be set up (CosyVoice + MuseTalk)
# Only run this if you completed the Phase 1B notebook first

import asyncio
import sys
import time
import numpy as np
import soundfile as sf
import cv2
sys.path.insert(0, "engine")

# Check if TTS is available
try:
    from pipeline.tts import VoiceCloner, OUTPUT_SAMPLE_RATE
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("CosyVoice not available. Run Phase 1B setup first.")

if TTS_AVAILABLE and 'cloner' in dir() and cloner.is_ready:
    print("=== Full Pipeline Test: Brain -> Voice -> Video ===\\n")

    # Question (simulated ASR)
    question = "Tell me about yourself"

    # RAG
    chunks = retriever.query(question, top_k=3)
    print(f"[RAG] {len(chunks)} knowledge chunks")

    # LLM -> collect sentences
    async def get_response():
        sentences = []
        async for s in brain.generate_stream(question, chunks):
            sentences.append(s)
            print(f"[LLM] {s}")
        return sentences

    sentences = asyncio.run(get_response())
    full_text = " ".join(sentences)
    print(f"\\n[LLM] Full response: {full_text}\\n")

    # TTS -> generate audio
    print("[TTS] Synthesizing in your voice...")
    start = time.perf_counter()
    audio = cloner.synthesize(full_text)
    tts_ms = (time.perf_counter() - start) * 1000
    duration_s = len(audio) / OUTPUT_SAMPLE_RATE
    print(f"[TTS] {duration_s:.1f}s audio in {tts_ms:.0f}ms")

    # Save audio
    sf.write("test_data/full_pipeline_audio.wav", audio, OUTPUT_SAMPLE_RATE)

    # Play audio
    from IPython.display import Audio, display
    display(Audio(audio, rate=OUTPUT_SAMPLE_RATE))

    # Lip sync (if available)
    try:
        from pipeline.lipsync import LipSyncer
        if 'syncer' in dir() and syncer.is_ready:
            print("\\n[LipSync] Generating frames...")
            start = time.perf_counter()
            frames = syncer.sync(audio, sample_rate=OUTPUT_SAMPLE_RATE)
            ls_ms = (time.perf_counter() - start) * 1000
            print(f"[LipSync] {len(frames)} frames in {ls_ms:.0f}ms")

            # Write video
            if frames:
                fps = syncer.target_fps
                h, w = frames[0].shape[:2]
                out = cv2.VideoWriter("test_data/full_pipeline_video.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for f in frames:
                    out.write(f)
                out.release()

                # Mux with audio
                !ffmpeg -y -i test_data/full_pipeline_video.mp4 -i test_data/full_pipeline_audio.wav -c:v libx264 -c:a aac -shortest test_data/full_pipeline_final.mp4 2>/dev/null
                print("[Output] test_data/full_pipeline_final.mp4")

                from google.colab import files
                files.download("test_data/full_pipeline_final.mp4")
    except Exception as e:
        print(f"[LipSync] Skipped: {e}")

    print("\\n=== Full Pipeline: COMPLETE ===")
else:
    print("Skipping full pipeline (run Phase 1B setup first)")
    print("The brain chain from Cell 7 is the key test for Phase 1C")
"""

# ==============================================
# NOTES
# ==============================================
"""
WHAT WAS TESTED:
  - ASR: faster-whisper transcription with VAD
  - RAG: ChromaDB knowledge storage and retrieval
  - LLM: Ollama + Llama 3 streaming response generation
  - Full chain: ASR -> RAG -> LLM (the avatar's brain)

LATENCY TARGETS:
  - ASR: ~150ms (faster-whisper on GPU)
  - RAG: ~50ms (ChromaDB local)
  - LLM: ~300ms to first sentence (Ollama streaming)
  - Total brain: ~500ms to first spoken word (with streaming overlap)

NEXT STEPS:
  - If brain chain works -> combine with Phase 1B (voice + lips)
  - Full end-to-end: someone talks -> avatar responds in your voice with moving lips
  - Then deploy on RunPod for the WOW demo video

TROUBLESHOOTING:
  - "Ollama not found": Re-run Cell 1 install and Cell 2
  - "Model not found": Make sure 'ollama pull llama3' completed
  - "ChromaDB error": pip install chromadb --upgrade
  - "CUDA out of memory": Use 'small' whisper model instead of 'medium'
  - Paste any errors into Claude Code for help
"""
