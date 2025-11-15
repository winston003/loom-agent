"""Base voiceprint interface for speaker verification."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVoiceprint(ABC):
    """Abstract interface for voiceprint extraction and verification."""

    @abstractmethod
    async def extract_features(self, audio_data: bytes) -> bytes:
        """Extract voice embedding/features from audio sample.

        Args:
            audio_data: Raw audio bytes (2-5 seconds recommended)

        Returns:
            Serialized feature bytes (encrypted or raw as agreed)
        """
        pass

    @abstractmethod
    async def verify(self, audio_data: bytes, reference_features: bytes) -> float:
        """Verify audio against stored features and return similarity score.

        Returns:
            similarity_score: float between 0 and 1
        """
        pass
