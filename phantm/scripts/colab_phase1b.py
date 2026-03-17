"""
LiveSelf -- Phase 1B Colab Notebook: Voice Clone + Lip Sync

Copy each section below into a separate Google Colab cell.
Runtime -> Change runtime type -> T4 GPU

This notebook:
  1. Installs CosyVoice 2 and MuseTalk dependencies
  2. Downloads the models
  3. Clones YOUR voice from a short audio sample
  4. Generates lip-synced video of an avatar speaking in your voice
  5. Outputs a video file for download

After this works, Phase 1C adds the brain (ASR + RAG + LLM).
"""

# ==============================================
# CELL 1: Check GPU and system info
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
"""

# ==============================================
# CELL 2: Clone LiveSelf repo + install base deps
# ==============================================
"""
!git clone https://github.com/DanKunleLove/LiveSelf.git
%cd LiveSelf/phantm

# Install base deps
!pip install numpy opencv-python-headless Pillow soundfile torchaudio
print("Base dependencies installed.")
"""

# ==============================================
# CELL 3: Install CosyVoice 2
# ==============================================
"""
import os
os.makedirs("engine/models", exist_ok=True)

# Clone CosyVoice repo
if not os.path.exists("engine/models/CosyVoice"):
    !git clone https://github.com/FunAudioLLM/CosyVoice.git engine/models/CosyVoice
    print("CosyVoice cloned.")
else:
    print("CosyVoice already cloned.")

# Install CosyVoice requirements
%cd engine/models/CosyVoice
!pip install -r requirements.txt
# Matcha-TTS dependency
if os.path.exists("third_party/Matcha-TTS"):
    %cd third_party/Matcha-TTS
    !pip install -e .
    %cd ../..
%cd /content/LiveSelf/phantm
print("CosyVoice 2 installed.")
"""

# ==============================================
# CELL 4: Install MuseTalk
# ==============================================
"""
import os

if not os.path.exists("engine/models/MuseTalk"):
    !git clone https://github.com/TMElyralab/MuseTalk.git engine/models/MuseTalk
    print("MuseTalk cloned.")
else:
    print("MuseTalk already cloned.")

# Install MuseTalk requirements
%cd engine/models/MuseTalk
!pip install -r requirements.txt
%cd /content/LiveSelf/phantm
print("MuseTalk installed.")
"""

# ==============================================
# CELL 5: Upload your voice sample + face photo
# ==============================================
"""
from google.colab import files
import os

os.makedirs("test_data", exist_ok=True)

# Upload voice sample (10-30 seconds of clear speech, WAV format)
print("Upload your VOICE SAMPLE (WAV file, 10-30 seconds of you speaking):")
uploaded_voice = files.upload()
voice_filename = list(uploaded_voice.keys())[0]
voice_path = f"test_data/{voice_filename}"
os.rename(voice_filename, voice_path)
print(f"Voice sample: {voice_path}")

# Upload face photo
print("\\nUpload your FACE PHOTO (clear, front-facing):")
uploaded_photo = files.upload()
photo_filename = list(uploaded_photo.keys())[0]
photo_path = f"test_data/{photo_filename}"
os.rename(photo_filename, photo_path)
print(f"Face photo: {photo_path}")

from IPython.display import display, Image
display(Image(photo_path, width=300))
"""

# ==============================================
# CELL 6: Test Voice Cloning (CosyVoice 2)
# ==============================================
"""
import sys
import time
import numpy as np
import soundfile as sf

# Add engine to path
sys.path.insert(0, "engine")

from pipeline.tts import VoiceCloner, OUTPUT_SAMPLE_RATE

# Initialize voice cloner
print("Loading CosyVoice 2 model (first run downloads weights ~1-2 GB)...")
cloner = VoiceCloner()
start = time.perf_counter()
cloner.load()
print(f"Model loaded in {time.perf_counter() - start:.1f}s")

# Set your voice as reference
cloner.set_reference_voice(voice_path)
print(f"Reference voice set: {voice_path}")

# Test sentences
test_sentences = [
    "Hello, I am your AI digital twin. Nice to meet you.",
    "I can speak in your voice and attend meetings for you.",
    "This is LiveSelf, making AI avatars accessible to everyone.",
]

all_audio = []
for i, sentence in enumerate(test_sentences):
    print(f"\\nSynthesizing ({i+1}/{len(test_sentences)}): '{sentence}'")
    start = time.perf_counter()
    audio = cloner.synthesize(sentence)
    elapsed_ms = (time.perf_counter() - start) * 1000
    duration_s = len(audio) / OUTPUT_SAMPLE_RATE
    print(f"  Generated {duration_s:.1f}s audio in {elapsed_ms:.0f}ms")
    all_audio.append(audio)

# Concatenate and save
combined = np.concatenate(all_audio)
output_wav = "test_data/voice_clone_output.wav"
sf.write(output_wav, combined, OUTPUT_SAMPLE_RATE)
print(f"\\nSaved combined audio to: {output_wav}")
print(f"Total audio: {len(combined) / OUTPUT_SAMPLE_RATE:.1f}s")

# Play in notebook
from IPython.display import Audio
display(Audio(combined, rate=OUTPUT_SAMPLE_RATE))
"""

# ==============================================
# CELL 7: Test Lip Sync (MuseTalk)
# ==============================================
"""
import cv2
import numpy as np
import time

from pipeline.lipsync import LipSyncer

# Load lip sync model
print("Loading MuseTalk model...")
syncer = LipSyncer()
start = time.perf_counter()
syncer.load()
print(f"MuseTalk loaded in {time.perf_counter() - start:.1f}s")

# Load face photo as reference frame
face_frame = cv2.imread(photo_path)
if face_frame is None:
    print(f"ERROR: Could not read photo at {photo_path}")
else:
    print(f"Face frame: {face_frame.shape[1]}x{face_frame.shape[0]}")
    syncer.set_reference_frame(face_frame)
    print("Reference frame set.")

    # Use the first test sentence audio for lip sync
    print("\\nGenerating lip-synced frames...")
    start = time.perf_counter()
    frames = syncer.sync(all_audio[0], sample_rate=OUTPUT_SAMPLE_RATE)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"Generated {len(frames)} frames in {elapsed_ms:.0f}ms")

    if frames:
        fps = syncer.target_fps
        actual_fps = len(frames) / (elapsed_ms / 1000)
        print(f"Target: {fps} fps, Actual: {actual_fps:.1f} fps")
    else:
        print("WARNING: No frames generated. MuseTalk may need additional setup.")
"""

# ==============================================
# CELL 8: Combine into video with audio
# ==============================================
"""
import cv2
import subprocess

if frames and len(frames) > 0:
    # Write frames to video
    fps = syncer.target_fps
    h, w = frames[0].shape[:2]
    temp_video = "test_data/lipsync_noaudio.mp4"
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video written: {temp_video} ({len(frames)} frames, {w}x{h} @ {fps}fps)")

    # Mux audio + video using ffmpeg
    final_output = "test_data/phase1b_output.mp4"
    audio_path_wav = "test_data/voice_clone_output.wav"

    # Use just the first sentence's worth of audio
    first_audio_path = "test_data/first_sentence.wav"
    sf.write(first_audio_path, all_audio[0], OUTPUT_SAMPLE_RATE)

    !ffmpeg -y -i {temp_video} -i {first_audio_path} -c:v libx264 -c:a aac -shortest {final_output} 2>/dev/null
    print(f"\\nFinal output: {final_output}")

    # Show a preview frame
    from IPython.display import display, Image
    mid = len(frames) // 2
    cv2.imwrite("test_data/preview_frame.jpg", frames[mid])
    display(Image("test_data/preview_frame.jpg", width=400))
    print("Preview frame from middle of video")
else:
    print("No frames to combine. Check MuseTalk setup above.")
"""

# ==============================================
# CELL 9: Download results
# ==============================================
"""
from google.colab import files
import os

# Download the voice clone audio
if os.path.exists("test_data/voice_clone_output.wav"):
    files.download("test_data/voice_clone_output.wav")
    print("Downloading voice clone audio...")

# Download the lip-synced video
if os.path.exists("test_data/phase1b_output.mp4"):
    files.download("test_data/phase1b_output.mp4")
    print("Downloading lip-synced video...")

print("\\nPhase 1B test complete.")
print("If you hear your voice and see lips moving -> SUCCESS")
print("Next: Phase 1C adds the brain (ASR + RAG + LLM)")
"""
