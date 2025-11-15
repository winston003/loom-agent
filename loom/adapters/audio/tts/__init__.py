"""Text-to-Speech (TTS) module for Xiaozhi audio adapter.

Provides EdgeTTS implementation and a common BaseTTS interface.
"""

__version__ = "0.0.1"

from loom.adapters.audio.tts.base import BaseTTS
from loom.adapters.audio.tts.edge import EdgeTTS

__all__ = ["BaseTTS", "EdgeTTS"]
