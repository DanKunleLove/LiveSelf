"""
LiveSelf -- Phase 1A Colab Notebook: Face Swap on Live Video

Copy each section below into a separate Google Colab cell.
Runtime -> Change runtime type -> T4 GPU

This notebook:
  1. Installs dependencies
  2. Downloads the face swap model
  3. Loads YOUR face photo
  4. Runs face swap on a test video
  5. Outputs the swapped video for download

After this works, the next step is running it on a live webcam feed
with virtual camera output (done locally or on RunPod, not Colab).
"""

# ==============================================
# CELL 1: Check GPU and install dependencies
# ==============================================
"""
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("NO GPU -- Go to Runtime > Change runtime type > T4 GPU")

# Install InsightFace and ONNX Runtime with GPU support
!pip install insightface==0.7.3 onnxruntime-gpu==1.17.1 opencv-python-headless numpy Pillow
print("Dependencies installed.")
"""

# ==============================================
# CELL 2: Download the face swap model
# ==============================================
"""
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Download inswapper_128_fp16.onnx from HuggingFace
# This is the model that performs the actual face swap
if not os.path.exists("models/inswapper_128_fp16.onnx"):
    !wget -q -O models/inswapper_128_fp16.onnx \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
    print("Downloaded inswapper_128_fp16.onnx")
else:
    print("Model already exists")

# Also download face enhancer (optional, makes output cleaner)
if not os.path.exists("models/GFPGANv1.4.onnx"):
    !wget -q -O models/GFPGANv1.4.onnx \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx"
    print("Downloaded GFPGANv1.4.onnx")
else:
    print("Enhancer already exists")

!ls -lh models/
"""

# ==============================================
# CELL 3: Upload your face photo
# ==============================================
"""
from google.colab import files

print("Upload a clear, front-facing photo of yourself:")
uploaded = files.upload()
photo_filename = list(uploaded.keys())[0]
print(f"Uploaded: {photo_filename}")

# Display the uploaded photo
from IPython.display import display, Image
display(Image(photo_filename, width=300))
"""

# ==============================================
# CELL 4: Initialize the face swapper
# ==============================================
"""
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import insightface

# Initialize face analyser (detects and embeds faces)
print("Loading face analysis model...")
start = time.time()
face_analyser = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_analyser.prepare(ctx_id=0, det_size=(640, 640))
print(f"Face analyser loaded in {time.time()-start:.1f}s")

# Load the inswapper model
print("Loading face swap model...")
start = time.time()
swapper = insightface.model_zoo.get_model(
    "models/inswapper_128_fp16.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(f"Swap model loaded in {time.time()-start:.1f}s")

# Load YOUR reference face
print(f"Analyzing your face in {photo_filename}...")
ref_image = cv2.imread(photo_filename)
ref_faces = face_analyser.get(ref_image)
if not ref_faces:
    print("ERROR: No face detected in your photo. Use a clearer photo.")
else:
    # Use the largest face found
    reference_face = max(ref_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    print(f"Your face loaded. Found {len(ref_faces)} face(s).")
    print("Ready to swap.")
"""

# ==============================================
# CELL 5: Upload a test video OR use a sample
# ==============================================
"""
# Option A: Upload your own test video
# from google.colab import files
# uploaded_vid = files.upload()
# video_path = list(uploaded_vid.keys())[0]

# Option B: Download a short sample video for testing
import urllib.request
video_path = "test_video.mp4"
if not os.path.exists(video_path):
    # Using a public domain sample video
    # Replace this URL with any short video of a person talking
    print("Upload a test video of someone talking (10-30 seconds is ideal)")
    from google.colab import files
    uploaded_vid = files.upload()
    video_path = list(uploaded_vid.keys())[0]

print(f"Test video: {video_path}")
"""

# ==============================================
# CELL 6: Run face swap on the video
# ==============================================
"""
import time

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR: Could not open {video_path}")
else:
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")

    # Output writer
    out = cv2.VideoWriter("output_swapped.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_num = 0
    swap_times = []
    print("Processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()

        # Detect faces in the current frame
        faces = face_analyser.get(frame)

        if faces:
            # Swap the largest face
            target = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            result = swapper.get(frame, target, reference_face, paste_back=True)
        else:
            result = frame

        elapsed_ms = (time.perf_counter() - start) * 1000
        swap_times.append(elapsed_ms)

        out.write(result)
        frame_num += 1

        if frame_num % 30 == 0:
            avg = sum(swap_times[-30:]) / min(30, len(swap_times))
            print(f"  Frame {frame_num}/{total}: {avg:.1f}ms/frame ({1000/avg:.0f} FPS)")

    cap.release()
    out.release()

    # Stats
    avg_ms = sum(swap_times) / len(swap_times)
    print(f"Done. {frame_num} frames processed.")
    print(f"Average: {avg_ms:.1f}ms/frame ({1000/avg_ms:.0f} FPS)")
    print(f"Output: output_swapped.mp4")
"""

# ==============================================
# CELL 7: Download the result
# ==============================================
"""
from google.colab import files
files.download("output_swapped.mp4")
print("Download started. Check your Downloads folder.")
"""

# ==============================================
# CELL 8 (OPTIONAL): Preview a frame in the notebook
# ==============================================
"""
from IPython.display import display, Image
import cv2

cap = cv2.VideoCapture("output_swapped.mp4")
# Jump to the middle of the video
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
cap.release()

if ret:
    # Save frame as jpg and display
    cv2.imwrite("preview_frame.jpg", frame)
    display(Image("preview_frame.jpg", width=600))
    print("Preview of swapped frame (middle of video)")
"""
