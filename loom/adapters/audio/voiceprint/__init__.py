"""Voiceprint (speaker verification) module

Provides client and storage helpers for 3DSpeaker voiceprint service.
"""

__version__ = "0.0.1"

from loom.adapters.audio.voiceprint.base import BaseVoiceprint
from loom.adapters.audio.voiceprint.client import ThreeDSpeakerClient
from loom.adapters.audio.voiceprint.storage import VoiceprintStorage

__all__ = ["BaseVoiceprint", "ThreeDSpeakerClient", "VoiceprintStorage"]
