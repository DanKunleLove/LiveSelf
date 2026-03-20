"""
LiveSelf -- Phase 1B Colab Notebook: Voice Clone + Lip Sync

UPDATED with fixes from first Colab run (2026-03-20).

Copy each section below into a separate Google Colab cell.
Runtime -> Change runtime type -> T4 GPU

What this notebook does:
  1. Installs all dependencies with known conflict fixes applied
  2. Clones CosyVoice 2 and MuseTalk
  3. Clones your voice from a short audio sample
  4. Generates lip-synced video of your face speaking in your voice
  5. Downloads the output video

KEY FIXES vs original notebook:
  - openai-whisper installed from GitHub (pypi wheel fails to build)
  - CosyVoice requirements stripped of conflicting deps before install
  - MuseTalk requirements stripped of conflicting deps before install
  - accelerate and transformers force-upgraded for diffusers compatibility
  - torch force-upgraded to 2.4+ (MuseTalk requirement)
  - sys.path set to phantm root (not engine subfolder)
  - Hugging Face token required for MuseTalk model download
"""


# ==============================================
# CELL 1: Check GPU
# ==============================================
"""
import torch
import sys
print(f"Python: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("NO GPU -- Go to Runtime > Change runtime type > T4 GPU")
    print("Stop and fix this before continuing.")
"""


# ==============================================
# CELL 2: Install base system deps + clone repo
# ==============================================
"""
import os
import shutil

# Install system deps needed for audio/video
!apt-get update -y -q && apt-get install -y -q build-essential ffmpeg

# Install base python packages (no pinned numpy -- let deps decide)
!pip install -q opencv-python-headless Pillow soundfile torchaudio

# Clone LiveSelf (fresh every session -- Colab resets between sessions)
if os.path.exists("/content/LiveSelf"):
    shutil.rmtree("/content/LiveSelf")
!git clone https://github.com/DanKunleLove/LiveSelf.git /content/LiveSelf
%cd /content/LiveSelf/phantm
print("Repo ready at:", os.getcwd())
!ls engine/pipeline/
"""


# ==============================================
# CELL 3: Install openai-whisper from source
#          (pypi wheel fails to build -- use GitHub)
# ==============================================
"""
# This is the fix for the broken openai-whisper pypi wheel.
# Installing from GitHub source skips the wheel build entirely.
!pip install -q git+https://github.com/openai/whisper.git

# Verify it works
import whisper
print(f"openai-whisper installed: {whisper.__version__ if hasattr(whisper, '__version__') else 'OK'}")
"""


# ==============================================
# CELL 4: Install CosyVoice 2
# ==============================================
"""
import os

os.makedirs("engine/models", exist_ok=True)

# Clone CosyVoice
if not os.path.exists("engine/models/CosyVoice"):
    !git clone https://github.com/FunAudioLLM/CosyVoice.git engine/models/CosyVoice
    print("CosyVoice cloned.")
else:
    print("CosyVoice already cloned.")

# Strip conflicting entries from requirements before installing.
# We remove: grpcio (version conflict), openai-whisper (already installed),
# numpy (let other deps decide version), opencv-python (using headless).
!sed -i '/^grpcio/d; /^openai-whisper/d; /^numpy/d; /^opencv-python$/d' engine/models/CosyVoice/requirements.txt
print("Stripped conflicting CosyVoice requirements.")

# Install remaining CosyVoice requirements
%cd engine/models/CosyVoice
!pip install -q -r requirements.txt

# Install Matcha-TTS (CosyVoice internal dependency)
if os.path.exists("third_party/Matcha-TTS"):
    %cd third_party/Matcha-TTS
    !pip install -q -e .
    %cd ../..
    print("Matcha-TTS installed.")

%cd /content/LiveSelf/phantm
print("CosyVoice 2: INSTALLED")
"""


# ==============================================
# CELL 5: Install MuseTalk
# ==============================================
"""
import os

if not os.path.exists("engine/models/MuseTalk"):
    !git clone https://github.com/TMElyralab/MuseTalk.git engine/models/MuseTalk
    print("MuseTalk cloned.")
else:
    print("MuseTalk already cloned.")

# Strip conflicting entries: numpy (version conflict), opencv-python (using headless),
# tensorflow (not needed, causes massive install time)
!sed -i '/^numpy/d; /^opencv-python$/d; /^tensorflow/d' engine/models/MuseTalk/requirements.txt
print("Stripped conflicting MuseTalk requirements.")

# Install MuseTalk requirements
%cd engine/models/MuseTalk
!pip install -q -r requirements.txt
%cd /content/LiveSelf/phantm

# Force upgrade accelerate and transformers -- MuseTalk's diffusers needs newer versions
# (diffusers will throw ImportError for clear_device_cache and EncoderDecoderCache otherwise)
!pip install -q --upgrade accelerate transformers

# Force upgrade torch to 2.4+ (MuseTalk minimum requirement)
!pip install -q torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu121

print("MuseTalk: INSTALLED")
"""


# ==============================================
# CELL 6: Set Hugging Face token
#          MuseTalk downloads models from HF -- needs auth
# ==============================================
"""
# BEFORE RUNNING THIS CELL:
# 1. Go to https://huggingface.co/settings/tokens
# 2. Create a read token (free account)
# 3. In Colab: click the key icon (Secrets) in the left sidebar
# 4. Add a secret named HF_TOKEN with your token value
# 5. Enable "Notebook access" for that secret
# 6. Then run this cell

from google.colab import userdata
import os

try:
    hf_token = userdata.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    print("HF_TOKEN loaded from Colab secrets.")
except Exception:
    print("WARNING: HF_TOKEN not found in Colab secrets.")
    print("MuseTalk model download may fail without it.")
    print("Follow the instructions in the cell comment above.")
"""


# ==============================================
# CELL 7: Upload your voice sample + face photo
# ==============================================
"""
from google.colab import files
import os

os.makedirs("test_data", exist_ok=True)

# Upload voice sample: 10-30 seconds of you speaking clearly
# WAV format preferred, MP3 also works
print("Upload your VOICE SAMPLE (WAV, 10-30 seconds of clear speech):")
uploaded_voice = files.upload()
voice_filename = list(uploaded_voice.keys())[0]
voice_path = f"test_data/{voice_filename}"
if voice_filename != voice_path:
    os.rename(voice_filename, voice_path)
print(f"Voice: {voice_path}")

# Upload face photo: clear front-facing photo, no sunglasses
print("\\nUpload your FACE PHOTO (JPG or PNG, front-facing, no hat/glasses):")
uploaded_photo = files.upload()
photo_filename = list(uploaded_photo.keys())[0]
photo_path = f"test_data/{photo_filename}"
if photo_filename != photo_path:
    os.rename(photo_filename, photo_path)
print(f"Photo: {photo_path}")

from IPython.display import display, Image
display(Image(photo_path, width=300))
"""


# ==============================================
# CELL 8: Test Voice Cloning (CosyVoice 2)
# ==============================================
"""
import sys
import time
import numpy as np
import soundfile as sf

# Add phantm root to path so engine.pipeline.tts resolves
sys.path.insert(0, "/content/LiveSelf/phantm")

from engine.pipeline.tts import VoiceCloner, OUTPUT_SAMPLE_RATE

print("Loading CosyVoice 2 (first run downloads ~1-2 GB model weights)...")
cloner = VoiceCloner()
start = time.perf_counter()
cloner.load()
print(f"Model loaded in {time.perf_counter() - start:.1f}s")

# Set your voice as the reference
cloner.set_reference_voice(voice_path)
print(f"Reference voice set from: {voice_path}")

# Sentences to synthesize
test_sentences = [
    "Hello, I am your AI digital twin. Nice to meet you.",
    "I can attend meetings for you and speak in your voice.",
    "This is LiveSelf, making AI avatars accessible to everyone.",
]

all_audio = []
for i, sentence in enumerate(test_sentences):
    print(f"\\nSynthesizing ({i+1}/{len(test_sentences)}): '{sentence}'")
    start = time.perf_counter()
    audio = cloner.synthesize(sentence)
    elapsed_ms = (time.perf_counter() - start) * 1000
    duration_s = len(audio) / OUTPUT_SAMPLE_RATE
    rtf = elapsed_ms / (duration_s * 1000)  # real-time factor
    print(f"  {duration_s:.1f}s audio in {elapsed_ms:.0f}ms (RTF: {rtf:.2f}x)")
    all_audio.append(audio)

# Save combined audio
combined = np.concatenate(all_audio)
output_wav = "test_data/voice_clone_output.wav"
sf.write(output_wav, combined, OUTPUT_SAMPLE_RATE)
print(f"\\nSaved: {output_wav} ({len(combined) / OUTPUT_SAMPLE_RATE:.1f}s total)")

# Play in notebook
from IPython.display import Audio, display
display(Audio(combined, rate=OUTPUT_SAMPLE_RATE))
print("\\nIf you can hear your voice above -> VOICE CLONE WORKING")
"""


# ==============================================
# CELL 9: Test Lip Sync (MuseTalk)
# ==============================================
"""
import cv2
import sys
import time

sys.path.insert(0, "/content/LiveSelf/phantm")
from engine.pipeline.lipsync import LipSyncer

print("Loading MuseTalk (first run downloads model weights)...")
syncer = LipSyncer()
start = time.perf_counter()
syncer.load()
print(f"MuseTalk loaded in {time.perf_counter() - start:.1f}s")

# Load face photo as reference frame
face_frame = cv2.imread(photo_path)
if face_frame is None:
    raise RuntimeError(f"Could not read photo at {photo_path}")
print(f"Face frame: {face_frame.shape[1]}x{face_frame.shape[0]}")
syncer.set_reference_frame(face_frame)
print("Reference frame set.")

# Use the first sentence's audio for lip sync test
print("\\nGenerating lip-synced frames for first sentence...")
start = time.perf_counter()
frames = syncer.sync(all_audio[0], sample_rate=OUTPUT_SAMPLE_RATE)
elapsed_ms = (time.perf_counter() - start) * 1000

if frames:
    fps = syncer.target_fps
    actual_fps = len(frames) / (elapsed_ms / 1000)
    print(f"Generated {len(frames)} frames in {elapsed_ms:.0f}ms")
    print(f"Target: {fps} fps | Actual: {actual_fps:.1f} fps")
    if actual_fps >= fps:
        print("Real-time capable: YES")
    else:
        print(f"Real-time capable: NO ({fps / actual_fps:.1f}x slower than real-time)")
else:
    print("WARNING: No frames generated. Check MuseTalk setup.")
"""


# ==============================================
# CELL 10: Combine into video + download
# ==============================================
"""
import cv2
import soundfile as sf
import subprocess

if not frames:
    print("No frames from previous cell. Stopping here.")
else:
    fps = syncer.target_fps
    h, w = frames[0].shape[:2]

    # Write frames to temp video (no audio yet)
    temp_video = "test_data/lipsync_noaudio.mp4"
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video frames written: {len(frames)} frames, {w}x{h} @ {fps}fps")

    # Save first sentence audio separately for muxing
    first_audio_path = "test_data/first_sentence.wav"
    sf.write(first_audio_path, all_audio[0], OUTPUT_SAMPLE_RATE)

    # Mux video + audio with ffmpeg
    final_output = "test_data/phase1b_output.mp4"
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", first_audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        final_output
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Final video: {final_output}")
    else:
        print("ffmpeg error:")
        print(result.stderr[-500:])  # last 500 chars of error

    # Show preview frame
    from IPython.display import display, Image
    mid = len(frames) // 2
    preview_path = "test_data/preview_frame.jpg"
    cv2.imwrite(preview_path, frames[mid])
    display(Image(preview_path, width=400))

    # Download everything
    from google.colab import files
    if os.path.exists(output_wav):
        files.download(output_wav)
        print("Downloading voice_clone_output.wav...")
    if os.path.exists(final_output):
        files.download(final_output)
        print("Downloading phase1b_output.mp4...")

    print("\\nPhase 1B complete.")
    print("If you see lips moving and hear your voice -> SUCCESS")
    print("Next: Run Phase 1C to add the brain (ASR + RAG + LLM)")
"""


# ==============================================
# TROUBLESHOOTING
# ==============================================
"""
COMMON ERRORS AND FIXES:

openai-whisper fails to build wheel:
  -> Cell 3 installs it from GitHub source, which avoids this. Make sure Cell 3 ran.

ImportError: cannot import name 'clear_device_cache' from 'accelerate':
  -> Run: !pip install --upgrade accelerate
  -> This is fixed in Cell 5 but run again if needed.

ImportError: cannot import name 'EncoderDecoderCache' from 'transformers':
  -> Run: !pip install --upgrade transformers
  -> This is fixed in Cell 5 but run again if needed.

MuseTalk requires PyTorch >= 2.4:
  -> Cell 5 upgrades torch. If still failing: restart runtime and re-run all cells.

HF_TOKEN error when loading MuseTalk models:
  -> Go to https://huggingface.co/settings/tokens
  -> Create a free read token
  -> Add it to Colab Secrets as HF_TOKEN
  -> Re-run Cell 6

No frames generated from MuseTalk:
  -> Most likely MuseTalk could not detect a face in your photo.
  -> Use a clear front-facing photo, good lighting, no occlusion.
  -> MuseTalk expects portrait-style photos, not full body.

CUDA out of memory:
  -> CosyVoice + MuseTalk both on T4 is tight (~15GB needed, T4 has 15GB).
  -> Unload CosyVoice before loading MuseTalk: cloner.unload()
  -> Then run syncer.load()
"""
