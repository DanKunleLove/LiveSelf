"""
Virtual Camera Output — Phase 1A

Takes processed video frames and outputs them to a virtual camera device.
This is what Zoom/Meet/WhatsApp sees as your webcam.

Uses pyvirtualcam (creates a v4l2/OBS virtual camera device).
Fallback: OBS Studio Virtual Camera if pyvirtualcam isn't available.

Input: video frames from the faceswap/lipsync pipeline
Output: virtual camera device visible to other apps
"""

# TODO: Phase 1A implementation
# 1. Initialize pyvirtualcam with correct resolution + fps
# 2. Consume frames from the pipeline queue
# 3. Push each frame to the virtual camera
# 4. Handle graceful shutdown (release camera on stop)
