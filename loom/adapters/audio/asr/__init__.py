"""Automatic Speech Recognition (ASR) Module

ASR providers for transcribing speech to text.

Supported Providers:
- FunASR: Local offline ASR with FunASR model
- CloudASR: Cloud ASR services (Alibaba, Tencent, Volcano) as fallback

Reference: migration/core/providers/asr/
"""

__version__ = "0.0.1"

from loom.adapters.audio.asr.base import BaseASR
from loom.adapters.audio.asr.funasr import FunASR

__all__ = ["BaseASR", "FunASR"]
