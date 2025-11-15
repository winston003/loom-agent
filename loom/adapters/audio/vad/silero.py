"""Silero VAD Implementation

Neural network-based Voice Activity Detection using Silero VAD model.

Features:
- Dual-threshold detection (high threshold + low threshold hysteresis)
- Sliding window for noise robustness
- Configurable silence duration for end-of-speech detection
- Local model (no cloud dependency)

Reference: migration/core/providers/vad/silero.py
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np
import torch

from loom.adapters.audio.models import AudioFrame, SpeechSegment
from loom.adapters.audio.vad.base import BaseVAD
from loom.interfaces.audio_adapter import AudioProcessingError
from loom.core.structured_logger import get_logger

logger = get_logger("audio.vad")


class SileroVAD(BaseVAD):
    """Silero VAD with dual-threshold and sliding window detection.

    Configuration:
        threshold: High threshold for speech detection (default: 0.5)
        threshold_low: Low threshold for hysteresis (default: 0.2)
        min_silence_duration_ms: Minimum silence to end speech (default: 1000ms)
        frame_window_threshold: Min frames to confirm speech (default: 3)

    Example:
        >>> vad = SileroVAD(
        ...     model_dir="models/silero_vad",
        ...     threshold=0.5,
        ...     threshold_low=0.2,
        ...     min_silence_duration_ms=1000
        ... )
        >>> vad.reset()
        >>> for frame in audio_stream:
        ...     segment = vad.detect(frame)
        ...     if segment:
        ...         print(f"Speech: {segment.duration_ms}ms")
    """

    def __init__(
        self,
        model_dir: str = "models/silero_vad",
        threshold: float = 0.5,
        threshold_low: float = 0.2,
        min_silence_duration_ms: int = 1000,
        frame_window_threshold: int = 3,
        sample_rate: int = 16000,
    ):
        """Initialize Silero VAD.

        Args:
            model_dir: Path to Silero VAD model directory
            threshold: High threshold for speech detection (0-1)
            threshold_low: Low threshold for hysteresis (0-1)
            min_silence_duration_ms: Minimum silence duration to end speech (ms)
            frame_window_threshold: Minimum frames to confirm speech
            sample_rate: Audio sample rate (Hz)
        """
        self.model_dir = model_dir
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.min_silence_duration_ms = min_silence_duration_ms
        self.frame_window_threshold = frame_window_threshold
        self.sample_rate = sample_rate

        # Load Silero VAD model
        import os
        
        try:
            if os.path.exists(os.path.join(model_dir, "hubconf.py")):
                # Load from local directory
                self.model, _ = torch.hub.load(
                    repo_or_dir=model_dir,
                    source="local",
                    model="silero_vad",
                    force_reload=False,
                )
                logger.info("Silero VAD model loaded from local", model_dir=model_dir)
            else:
                # Download from PyTorch Hub
                logger.info("Model directory not found, downloading Silero VAD from PyTorch Hub...")
                self.model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    source="github",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                )
                logger.info("Downloaded and loaded Silero VAD model from PyTorch Hub")
        except Exception as e:
            raise AudioProcessingError(f"Failed to load Silero VAD model: {e}") from e

        # Internal state
        self.audio_buffer = bytearray()  # PCM buffer for incomplete frames
        self.voice_window = deque(maxlen=10)  # Sliding window for voice detection
        self.last_is_voice = False  # Previous frame voice state
        self.last_activity_time = 0.0  # Last speech activity timestamp (ms)
        self.has_speech = False  # Whether speech was detected in session
        self.speech_start_time: Optional[float] = None  # Speech start timestamp

    def detect(self, frame: AudioFrame) -> Optional[SpeechSegment]:
        """Detect speech activity in audio frame.

        Args:
            frame: Audio frame (PCM 16-bit)

        Returns:
            SpeechSegment if speech detected and ended, None otherwise
        """
        try:
            # Add frame to buffer
            self.audio_buffer.extend(frame.data)

            # Process complete 512-sample chunks (1024 bytes @ 16-bit)
            chunk_size = 512 * 2  # 512 samples * 2 bytes
            has_voice_now = False

            while len(self.audio_buffer) >= chunk_size:
                # Extract chunk
                chunk = bytes(self.audio_buffer[:chunk_size])
                self.audio_buffer = self.audio_buffer[chunk_size:]

                # Convert to float32 tensor
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)

                # Run VAD model
                with torch.no_grad():
                    speech_prob = self.model(audio_tensor, self.sample_rate).item()

                # Dual-threshold detection with hysteresis
                if speech_prob >= self.threshold:
                    is_voice = True
                elif speech_prob <= self.threshold_low:
                    is_voice = False
                else:
                    # Intermediate zone: keep previous state
                    is_voice = self.last_is_voice

                self.last_is_voice = is_voice

                # Update sliding window
                self.voice_window.append(is_voice)
                has_voice_now = self.voice_window.count(True) >= self.frame_window_threshold

                # Track speech activity
                if has_voice_now:
                    if not self.has_speech:
                        # Speech started
                        self.speech_start_time = time.time() * 1000
                        logger.debug("Speech started")
                    self.has_speech = True
                    self.last_activity_time = time.time() * 1000

            # Check for end of speech
            if self.has_speech and not has_voice_now:
                silence_duration = time.time() * 1000 - self.last_activity_time
                if silence_duration >= self.min_silence_duration_ms:
                    # Speech ended
                    segment = self._create_speech_segment()
                    self.reset()  # Reset for next speech segment
                    return segment

            return None

        except Exception as e:
            logger.error("VAD detection error", error=str(e), exc_info=e)
            raise AudioProcessingError(f"VAD detection failed: {e}") from e

    def _create_speech_segment(self) -> SpeechSegment:
        """Create SpeechSegment from detected speech.

        Returns:
            SpeechSegment with duration and confidence
        """
        duration_ms = 0.0
        if self.speech_start_time:
            duration_ms = self.last_activity_time - self.speech_start_time

        # Estimate confidence from sliding window
        voice_ratio = self.voice_window.count(True) / len(self.voice_window) if self.voice_window else 0.0

        return SpeechSegment(
            start_time=self.speech_start_time or 0.0,
            end_time=self.last_activity_time,
            duration_ms=duration_ms,
            confidence=voice_ratio,
            audio_data=b"",  # Not storing audio data in VAD
        )

    def reset(self) -> None:
        """Reset VAD internal state.

        Clears buffers, sliding window, and detection state.
        """
        self.audio_buffer.clear()
        self.voice_window.clear()
        self.last_is_voice = False
        self.last_activity_time = 0.0
        self.has_speech = False
        self.speech_start_time = None
        logger.debug("VAD state reset")
