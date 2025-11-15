"""Edge TTS wrapper (Edge-TTS Python library)

Provides a simple async streaming interface around `edge_tts.Communicate`.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

import edge_tts

from loom.adapters.audio.tts.base import BaseTTS
from loom.core.structured_logger import get_logger

logger = get_logger("audio.tts.edge")


class EdgeTTS(BaseTTS):
    """Edge TTS implementation using `edge_tts`.

    Notes:
        - `edge_tts` yields chunks via `Communicate.stream()`.
        - We surface an async generator of bytes for streaming playback.
    """

    def __init__(self, voice: Optional[str] = None):
        self.voice = voice or "zh-CN-XiaoxiaoNeural"

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        voice_name = voice or self.voice
        communicate = edge_tts.Communicate(text, voice=voice_name)

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize_to_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        buf = bytearray()
        async for chunk in self.synthesize(text, voice=voice):
            buf.extend(chunk)
        return bytes(buf)
