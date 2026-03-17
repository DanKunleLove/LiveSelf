"""
LiveSelf -- Phase 0: Test Deep-Live-Cam on Google Colab

INSTRUCTIONS:
1. Go to colab.research.google.com
2. Create a new notebook
3. Runtime → Change runtime type → T4 GPU
4. Copy-paste each cell below into separate Colab cells
5. Run them in order

This tests the FIRST tool in your pipeline: face swap.
If this works, everything else builds on top of it.
"""

# ============================================
# CELL 1: Check GPU is available
# ============================================
"""
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("ERROR: No GPU! Go to Runtime → Change runtime type → T4 GPU")
"""

# ============================================
# CELL 2: Clone Deep-Live-Cam
# ============================================
"""
!git clone https://github.com/hacksider/Deep-Live-Cam.git
%cd Deep-Live-Cam
!pip install -r requirements.txt
"""

# ============================================
# CELL 3: Download required models
# ============================================
"""
# Deep-Live-Cam needs InsightFace models
# Check the README for the latest download links
!mkdir -p models
# The exact model download commands depend on the current version
# Check: https://github.com/hacksider/Deep-Live-Cam#readme
"""

# ============================================
# CELL 4: Upload your photo
# ============================================
"""
from google.colab import files
uploaded = files.upload()  # This opens a file picker — select your face photo
photo_filename = list(uploaded.keys())[0]
print(f"Uploaded: {photo_filename}")
"""

# ============================================
# CELL 5: Run a basic face swap test
# ============================================
"""
# This will vary based on Deep-Live-Cam's current API
# Check their README for the exact command
# The goal: swap your face onto a test video/image
# If you see your face on someone else's body → Phase 0 COMPLETE
"""

# ============================================
# NOTES:
# - Deep-Live-Cam's API may have changed since this was written
# - Always check the latest README at github.com/hacksider/Deep-Live-Cam
# - If you hit errors, paste them into Claude Code and I'll help debug
# - Expected time: ~15 minutes to get working
# - Cost: $0 (Google Colab free tier)
# ============================================
