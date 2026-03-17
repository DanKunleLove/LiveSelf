"""
Face Swap Module — Phase 1A (First thing to build)

Uses Deep-Live-Cam (hacksider/Deep-Live-Cam) to swap a reference photo's
face onto a live webcam feed in real-time.

Input: reference photo path + live video frames
Output: face-swapped video frames pushed to a queue

This is the first "wow" — seeing your photo's face appear on a webcam feed.
"""

# TODO: Phase 1A implementation
# 1. Clone Deep-Live-Cam repo into engine/models/
# 2. Load InsightFace models (face detection + swap)
# 3. Accept reference photo + webcam frame
# 4. Output swapped frame
# 5. Push to virtual_cam queue

# Key files to study in Deep-Live-Cam:
# - modules/core.py (the main swap logic)
# - modules/face_analyser.py (face detection)
# - modules/processors/frame/face_swapper.py (the actual swap)
