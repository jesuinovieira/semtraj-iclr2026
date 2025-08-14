import time
from collections import deque
from typing import Any
from typing import Callable
from typing import Protocol

from google import genai

# Default rate limiter config
_MAX_CALLS = 1500
_PERIOD_SEC = 60.0


class _RateLimiter:
    """Decorator‑style rate‑limiter (max `max_calls` per `period_sec`)."""

    def __init__(self, max_calls: int, period_sec: float):
        self.max_calls = max_calls
        self.period = period_sec
        self.calls: deque[float] = deque()

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args: tuple, **kwargs: dict) -> Any:
            now = time.monotonic()

            # Drop timestamps that are outside the window
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()

            # If we hit the limit, wait for the oldest call to expire
            if len(self.calls) >= self.max_calls:
                wait = self.period - (now - self.calls[0])
                time.sleep(wait)

            result = func(*args, **kwargs)
            self.calls.append(time.monotonic())
            return result

        return wrapper


RATE_LIMITER = _RateLimiter(max_calls=_MAX_CALLS, period_sec=_PERIOD_SEC)


class EmbeddingBackend(Protocol):
    """Interface for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Computes embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            A list of dense vectors (each `list[float]`) aligned with `texts`.
        """


class GeminiBackend:
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @RATE_LIMITER
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Send the batch of texts to the Gemini API for embedding
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=genai.types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )

        # Extract the raw embeddings
        return [emb.values for emb in response.embeddings]


class OpenAIBackend:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    @RATE_LIMITER
    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


def get(api_key: str, backend: str = "gemini") -> EmbeddingBackend:
    """Factory function to get the appropriate embedding backend."""
    if backend == "gemini":
        return GeminiBackend(api_key=api_key)

    if backend == "openai":
        return OpenAIBackend(api_key=api_key)

    raise ValueError(f"Unsupported backend: {backend}")
