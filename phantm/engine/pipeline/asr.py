"""
Automatic Speech Recognition (ASR) — Phase 1C

Uses faster-whisper (SYSTRAN/faster-whisper) to transcribe
what the caller says in real-time.

Includes Voice Activity Detection (VAD) to know when someone
has finished speaking so the avatar can start responding.

Input: microphone audio stream
Output: transcribed text + end-of-speech signal

faster-whisper:
- 4x faster than OpenAI Whisper
- Built-in VAD (Silero VAD)
- ~150ms latency
- MIT license
"""

# TODO: Phase 1C implementation
# 1. Initialize faster-whisper with medium model
# 2. Set up audio input stream from microphone
# 3. Run VAD to detect speech segments
# 4. Transcribe speech segment when end-of-speech detected
# 5. Push transcribed text to RAG queue
