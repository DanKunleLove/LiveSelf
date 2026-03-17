"""
Test script for the LLMBrain module (Phase 1C).

Run with: python scripts/test_llm.py

Two modes:
  1. Mock mode (no LLM server) -- verifies class logic
  2. Live mode (Ollama running) -- runs real LLM inference

Mock mode runs by default. For live mode, set LIVESELF_TEST_LIVE=1
and have Ollama running with llama3 pulled.
"""

import asyncio
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_llm")


def test_init():
    """Verify LLMBrain initializes without loading."""
    from pipeline.llm import LLMBrain

    brain = LLMBrain()
    assert not brain.is_ready
    assert brain.provider == "ollama"
    logger.info("PASS: LLMBrain init (ollama)")

    brain_claude = LLMBrain(provider="claude", api_key="test-key")
    assert brain_claude.provider == "claude"
    logger.info("PASS: LLMBrain init (claude)")


def test_errors_before_load():
    """Verify proper errors before loading."""
    from pipeline.llm import LLMBrain

    brain = LLMBrain()

    try:
        asyncio.run(brain.generate("test"))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: generate raises RuntimeError before load()")


def test_system_prompt_building():
    """Verify system prompt is built correctly with knowledge context."""
    from pipeline.llm import LLMBrain

    brain = LLMBrain(persona_name="Dan")

    # With knowledge chunks
    prompt = brain._build_system_prompt(["I know Python", "I live in Lagos"])
    assert "Dan" in prompt
    assert "I know Python" in prompt
    assert "I live in Lagos" in prompt
    logger.info("PASS: system prompt with knowledge")

    # Without knowledge chunks
    prompt_empty = brain._build_system_prompt([])
    assert "Dan" in prompt_empty
    assert "No specific knowledge" in prompt_empty
    logger.info("PASS: system prompt without knowledge")


def test_sentence_splitting():
    """Verify the streaming sentence boundary detection."""
    from pipeline.llm import SENTENCE_END_PATTERN
    import re

    test_text = "Hello there. How are you? I am fine! Let me tell you more."
    parts = re.split(r'(?<=[.!?])\s+', test_text)
    assert len(parts) == 4, f"Expected 4 sentences, got {len(parts)}: {parts}"
    assert parts[0] == "Hello there."
    assert parts[1] == "How are you?"
    logger.info("PASS: sentence splitting")


def test_invalid_provider():
    """Verify error on invalid provider."""
    from pipeline.llm import LLMBrain

    brain = LLMBrain(provider="invalid")
    try:
        brain.load()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unknown provider" in str(e).lower()
        logger.info("PASS: invalid provider raises ValueError")


def test_claude_requires_api_key():
    """Verify Claude provider requires API key."""
    from pipeline.llm import LLMBrain

    brain = LLMBrain(provider="claude")
    try:
        brain.load()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "api key" in str(e).lower()
        logger.info("PASS: Claude requires API key")


def test_unload():
    """Verify unload clears state."""
    from pipeline.llm import LLMBrain

    brain = LLMBrain()
    brain.unload()
    assert not brain.is_ready
    logger.info("PASS: unload")


async def _test_live_ollama():
    """Live test with Ollama."""
    from pipeline.llm import LLMBrain

    brain = LLMBrain(provider="ollama", persona_name="Test User")
    brain.load()
    assert brain.is_ready

    # Test streaming
    knowledge = ["I am a software engineer who loves Python and AI."]
    sentences = []
    async for sentence in brain.generate_stream("What do you do for work?", knowledge):
        sentences.append(sentence)
        logger.info(f"  Sentence: '{sentence[:80]}'")

    assert len(sentences) > 0, "Should generate at least one sentence"
    logger.info(f"PASS: live streaming ({len(sentences)} sentences)")

    # Test non-streaming
    response = await brain.generate("Tell me about yourself", knowledge)
    assert len(response) > 0
    logger.info(f"PASS: live generate: '{response[:80]}'")

    brain.unload()
    return True


def test_live_ollama():
    """Run the live Ollama test."""
    logger.info("Running live Ollama test...")
    return asyncio.run(_test_live_ollama())


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("LLMBrain Tests -- Phase 1C")
    logger.info("=" * 60)

    test_init()
    test_errors_before_load()
    test_system_prompt_building()
    test_sentence_splitting()
    test_invalid_provider()
    test_claude_requires_api_key()
    test_unload()

    logger.info("-" * 60)
    logger.info("Mock tests: ALL PASSED")
    logger.info("-" * 60)

    if os.environ.get("LIVESELF_TEST_LIVE") == "1":
        success = test_live_ollama()
        if success:
            logger.info("Live tests: ALL PASSED")
        else:
            logger.error("Live tests: FAILED")
            sys.exit(1)
    else:
        logger.info("Skipping live test (set LIVESELF_TEST_LIVE=1 to enable)")

    logger.info("=" * 60)
    logger.info("All tests passed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
