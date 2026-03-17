"""
LLM Brain — Phase 1C

Connects to Ollama (local, free) or Claude API (premium, paid)
to generate intelligent responses based on the question + knowledge context.

Uses streaming to start TTS before the full response is generated.

Input: question text + retrieved knowledge chunks
Output: streamed response text (sentence by sentence)
"""

# TODO: Phase 1C implementation
# 1. Build prompt: system instructions + knowledge context + question
# 2. Stream response from Ollama (Llama 3) or Claude
# 3. Detect sentence boundaries in streaming output
# 4. Push each sentence to TTS queue immediately (streaming overlap trick)
# 5. Log full response for session history
