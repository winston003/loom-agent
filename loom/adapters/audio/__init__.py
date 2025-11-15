"""
Audio adapters for Loom Agent Framework.

Provides audio interaction capabilities for Xiaozhi voice companion:
- VAD (Voice Activity Detection)
- ASR (Automatic Speech Recognition)
- TTS (Text-to-Speech)
- VP (Voiceprint Verification)
- WebSocket communication management

Important references:
- migration/ directory contains original Xiaozhi implementation code
- All implementations follow Loom Agent Framework principles (adapter-first integration)
"""

from loom.adapters.audio.adapter import AudioAdapter, create_audio_adapter
from loom.adapters.audio.models import AudioAdapterConfig

__version__ = "0.0.1"

__all__ = [
    "AudioAdapter",
    "create_audio_adapter",
    "AudioAdapterConfig",
]
