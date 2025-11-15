"""Base TTS interface for speech synthesis providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional


class BaseTTS(ABC):
    """Abstract base class for Text-to-Speech providers.

    Implementations should provide a streaming `synthesize` method that yields
    audio chunks (bytes) to allow low-latency playback.
    """

    @abstractmethod
    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio for given text.

        Args:
            text: Text to synthesize
            voice: Optional voice name

        Yields:
            Audio chunks as bytes
        """
        pass

    @abstractmethod
    async def synthesize_to_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize text and return complete audio bytes.

        Useful for tests or when the caller prefers the whole file.
        """
        pass
