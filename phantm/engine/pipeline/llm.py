"""
LLM Brain -- Phase 1C

Connects to Ollama (local, free) or Claude API (premium, paid) to generate
intelligent responses based on the user's question + retrieved knowledge.

Uses streaming to start TTS on the first sentence while the LLM is still
generating the rest. This is the "streaming overlap trick" that cuts
perceived latency from ~860ms to ~500ms.

Input: question text + knowledge context chunks from RAG
Output: streamed response text, sentence by sentence

Pipeline position: ... -> RAG -> [LLM] -> TTS -> ...
"""

import asyncio
import logging
import re
import time
from typing import AsyncGenerator, Optional

logger = logging.getLogger("liveself.engine.llm")

# Sentence boundary pattern for streaming chunking
SENTENCE_END_PATTERN = re.compile(r'[.!?]\s+')

# System prompt template for the avatar
SYSTEM_PROMPT = """You are {persona_name}'s AI digital twin. You speak exactly as {persona_name} would -- same tone, vocabulary, and personality.

Rules:
- Answer in first person as {persona_name}
- Keep responses concise and conversational (2-4 sentences typical)
- If the knowledge context contains relevant information, use it
- If you don't know something, say so honestly as {persona_name} would
- Never mention that you are an AI, a digital twin, or a language model
- Match the speaking style: casual conversation, not an essay

Knowledge about {persona_name}:
{knowledge_context}"""


class LLMBrain:
    """
    Streaming LLM response generator supporting Ollama and Claude.

    Generates responses sentence-by-sentence so TTS can start speaking
    before the full response is ready.

    Usage (Ollama, free, local):
        brain = LLMBrain(provider="ollama")
        brain.load()
        async for sentence in brain.generate_stream(question, context):
            send_to_tts(sentence)

    Usage (Claude, paid, cloud):
        brain = LLMBrain(provider="claude", api_key="sk-...")
        brain.load()
        async for sentence in brain.generate_stream(question, context):
            send_to_tts(sentence)
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_host: str = "http://localhost:11434",
        persona_name: str = "the user",
    ):
        """
        Args:
            provider: "ollama" (free, local) or "claude" (paid, cloud).
            model: Model name. Defaults to "llama3" for Ollama, "claude-sonnet-4-20250514" for Claude.
            api_key: Anthropic API key (required for Claude provider).
            ollama_host: Ollama server URL.
            persona_name: Name of the persona for the system prompt.
        """
        self._provider = provider
        self._ollama_host = ollama_host
        self._persona_name = persona_name
        self._api_key = api_key

        if model:
            self._model = model
        elif provider == "ollama":
            self._model = "llama3"
        else:
            self._model = "claude-sonnet-4-20250514"

        self._ollama_client = None
        self._anthropic_client = None
        self._is_loaded = False

        logger.info(f"LLMBrain created (provider: {provider}, model: {self._model})")

    def load(self) -> None:
        """
        Initialize the LLM client.

        For Ollama: connects to local server (must be running).
        For Claude: initializes the Anthropic client with API key.

        Raises:
            ImportError: If required client library is not installed.
            ConnectionError: If Ollama server is not reachable.
        """
        start = time.perf_counter()

        if self._provider == "ollama":
            self._load_ollama()
        elif self._provider == "claude":
            self._load_claude()
        else:
            raise ValueError(f"Unknown provider: {self._provider}. Use 'ollama' or 'claude'.")

        self._is_loaded = True
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"LLM client loaded in {elapsed_ms:.0f}ms (provider: {self._provider})")

    def _load_ollama(self) -> None:
        """Initialize Ollama client."""
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama is required but not installed. "
                "Install with: pip install ollama"
            )

        self._ollama_client = ollama.Client(host=self._ollama_host)

        # Verify connection by listing models
        try:
            self._ollama_client.list()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._ollama_host}: {e}. "
                "Start Ollama with: ollama serve"
            )

    def _load_claude(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required but not installed. "
                "Install with: pip install anthropic"
            )

        if not self._api_key:
            raise ValueError("API key required for Claude provider. Set api_key parameter.")

        self._anthropic_client = anthropic.Anthropic(api_key=self._api_key)

    def _build_system_prompt(self, knowledge_chunks: list[str]) -> str:
        """Build the system prompt with persona name and knowledge context."""
        if knowledge_chunks:
            context = "\n---\n".join(knowledge_chunks)
        else:
            context = "(No specific knowledge available. Respond naturally.)"

        return SYSTEM_PROMPT.format(
            persona_name=self._persona_name,
            knowledge_context=context,
        )

    async def generate_stream(
        self,
        question: str,
        knowledge_chunks: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a response sentence-by-sentence via streaming.

        This is the core method for the pipeline. Each yielded string is a
        complete sentence that can be immediately sent to TTS.

        The streaming overlap trick: TTS starts speaking the first sentence
        while the LLM is still generating sentence 2, 3, etc. This makes
        the avatar feel much more responsive.

        Args:
            question: The transcribed question from ASR.
            knowledge_chunks: Relevant knowledge from RAG retriever.

        Yields:
            Complete sentences as strings.

        Raises:
            RuntimeError: If LLM client is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("LLM not loaded. Call load() first.")

        knowledge_chunks = knowledge_chunks or []
        system_prompt = self._build_system_prompt(knowledge_chunks)

        start = time.perf_counter()
        sentence_count = 0
        full_response = ""

        if self._provider == "ollama":
            token_stream = self._stream_ollama(system_prompt, question)
        else:
            token_stream = self._stream_claude(system_prompt, question)

        # Buffer tokens and yield complete sentences
        buffer = ""
        async for token in token_stream:
            buffer += token
            full_response += token

            # Check for sentence boundaries
            while True:
                match = SENTENCE_END_PATTERN.search(buffer)
                if match:
                    # Found a sentence boundary
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]

                    if sentence:
                        sentence_count += 1
                        if sentence_count == 1:
                            first_sentence_ms = (time.perf_counter() - start) * 1000
                            logger.debug(f"First sentence in {first_sentence_ms:.0f}ms")
                        yield sentence
                else:
                    break

        # Yield any remaining text as the final sentence
        if buffer.strip():
            sentence_count += 1
            yield buffer.strip()
            full_response += ""

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"LLM response: {sentence_count} sentences in {elapsed_ms:.0f}ms, "
            f"question='{question[:50]}'"
        )

    async def _stream_ollama(self, system_prompt: str, question: str) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama."""
        try:
            stream = self._ollama_client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                stream=True,
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                    # Yield control to event loop periodically
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"I'm having trouble thinking right now. Could you repeat that?"

    async def _stream_claude(self, system_prompt: str, question: str) -> AsyncGenerator[str, None]:
        """Stream tokens from Claude API."""
        try:
            with self._anthropic_client.messages.stream(
                model=self._model,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
                        await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Claude streaming failed: {e}")
            yield f"I'm having trouble thinking right now. Could you repeat that?"

    async def generate(
        self,
        question: str,
        knowledge_chunks: Optional[list[str]] = None,
    ) -> str:
        """
        Generate a complete response (non-streaming).

        Convenience method for testing. For real-time use, prefer
        generate_stream().

        Args:
            question: The question to answer.
            knowledge_chunks: Relevant knowledge from RAG.

        Returns:
            Full response text.
        """
        sentences = []
        async for sentence in self.generate_stream(question, knowledge_chunks):
            sentences.append(sentence)
        return " ".join(sentences)

    @property
    def is_ready(self) -> bool:
        """Check if the LLM is loaded and ready."""
        return self._is_loaded

    @property
    def provider(self) -> str:
        """Current LLM provider name."""
        return self._provider

    def unload(self) -> None:
        """Release LLM client resources."""
        self._ollama_client = None
        self._anthropic_client = None
        self._is_loaded = False
        logger.info("LLMBrain unloaded")
