"""Voice Activity Detection (VAD) Module

VAD providers for detecting speech segments in audio streams.

Supported Providers:
- SileroVAD: Local neural VAD with dual-threshold detection

Reference: migration/core/providers/vad/
"""

__version__ = "0.0.1"

from loom.adapters.audio.vad.base import BaseVAD
from loom.adapters.audio.vad.silero import SileroVAD

__all__ = ["BaseVAD", "SileroVAD"]
