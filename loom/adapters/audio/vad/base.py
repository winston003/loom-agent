"""Base VAD Interface

Abstract interface for Voice Activity Detection providers.

Reference: loom/interfaces/audio_adapter.py - BaseVAD
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from loom.adapters.audio.models import AudioFrame, SpeechSegment


class BaseVAD(ABC):
    """Abstract base class for Voice Activity Detection.

    VAD detects speech segments in audio streams using various algorithms
    (neural networks, energy-based, etc.).

    Example:
        >>> vad = SileroVAD(threshold=0.5)
        >>> frame = AudioFrame(data=pcm_bytes, sample_rate=16000)
        >>> segment = vad.detect(frame)
        >>> if segment:
        ...     print(f"Speech detected: {segment.duration_ms}ms")
    """

    @abstractmethod
    def detect(self, frame: AudioFrame) -> Optional[SpeechSegment]:
        """Detect speech activity in audio frame.

        Args:
            frame: Audio frame to analyze

        Returns:
            SpeechSegment if speech detected, None otherwise

        Raises:
            AudioProcessingError: If VAD processing fails
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset VAD internal state.

        Clears frame buffers, sliding windows, and detection state.
        Called at session start or after errors.
        """
        pass
