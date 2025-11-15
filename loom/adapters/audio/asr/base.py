"""Base ASR Interface

Abstract interface for Automatic Speech Recognition providers.

Reference: loom/interfaces/audio_adapter.py - BaseASR
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from loom.adapters.audio.models import TranscriptionResult


class BaseASR(ABC):
    """Abstract base class for Automatic Speech Recognition.

    ASR transcribes audio speech to text using various engines
    (local models, cloud services).

    Example:
        >>> asr = FunASR(model_dir="models/funasr")
        >>> audio_data = b"..." # PCM audio bytes
        >>> result = await asr.transcribe(audio_data)
        >>> print(result.text)
    """

    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str = "auto") -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio_data: Raw PCM audio data (16-bit, 16kHz recommended)
            language: Language code ('auto', 'zh', 'en', etc.)

        Returns:
            TranscriptionResult with text and confidence

        Raises:
            ASRError: If transcription fails
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset ASR internal state.

        Clears caches and temporary resources.
        """
        pass
