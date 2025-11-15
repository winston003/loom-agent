"""Cloud TTS placeholder implementations.

This module provides a placeholder CloudTTS class to be implemented for specific
cloud providers when needed. Kept lightweight as a fallback option.
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from loom.adapters.audio.tts.base import BaseTTS
from loom.core.structured_logger import get_logger

logger = get_logger("audio.tts.cloud")


class CloudTTS(BaseTTS):
    """Placeholder cloud TTS implementation (no-op).

    Implement provider-specific streaming in subclasses (Azure, Google, Volcano).
    """

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        # Simple fallback: yield empty bytes and return
        yield b""

    async def synthesize_to_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        # Return empty bytes as placeholder
        return b""
