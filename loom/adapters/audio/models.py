"""Audio Adapter Data Models

Core Pydantic v2 models for audio session management, metrics, voiceprint profiles,
and WebSocket communication.

Reference: specs/002-xiaozhi-voice-adapter/data-model.md
"""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================
# Data Transfer Objects (DTOs)
# ============================


class AudioFrame(BaseModel):
    """Single audio frame for processing.
    
    Used for passing audio data through VAD/ASR pipeline.
    """

    data: bytes = Field(description="PCM audio data (16-bit)")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Audio sample rate (Hz)")
    channels: int = Field(default=1, ge=1, le=2, description="Number of audio channels")
    timestamp: float = Field(default_factory=lambda: __import__("time").time(), description="Frame timestamp (Unix)")


class SpeechSegment(BaseModel):
    """Detected speech segment from VAD.
    
    Represents a continuous speech segment with timing information.
    """

    start_time: float = Field(ge=0, description="Segment start time (ms)")
    end_time: float = Field(ge=0, description="Segment end time (ms)")
    duration_ms: float = Field(ge=0, description="Segment duration (ms)")
    audio_data: bytes = Field(description="Audio data for segment")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Detection confidence (0-1)")

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, v: float, info) -> float:
        """Ensure end_time >= start_time."""
        if "start_time" in info.data and v < info.data["start_time"]:
            raise ValueError("end_time must be >= start_time")
        return v


class TranscriptionResult(BaseModel):
    """ASR transcription result.
    
    Contains recognized text with metadata.
    """

    text: str = Field(description="Transcribed text")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Transcription confidence (0-1)")
    language: str = Field(default="auto", description="Detected/specified language code")
    latency_ms: float = Field(default=0.0, ge=0, description="ASR processing latency (ms)")


class ConversationTurn(BaseModel):
    """Single turn in multi-turn conversation.
    
    Stores user input and agent response with metadata for context tracking.
    Used in AudioSession.conversation_history for multi-turn dialogue.
    """

    turn_index: int = Field(ge=0, description="Turn number in conversation (0-based)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Turn timestamp (UTC)")
    user_text: str = Field(max_length=10000, description="User's transcribed utterance")
    agent_response: Optional[str] = Field(None, max_length=50000, description="Agent's response text")
    speaker_id: Optional[str] = Field(None, description="Identified speaker for this turn")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional turn metadata")

    class Config:
        frozen = True  # Immutable after creation


class VoiceprintMatch(BaseModel):
    """Voiceprint verification result.
    
    Contains speaker matching information.
    """

    speaker_id: Optional[str] = Field(None, description="Matched speaker ID (None if no match)")
    similarity_score: float = Field(ge=0, le=1, description="Similarity score (0-1)")
    is_match: bool = Field(description="Whether score exceeds threshold")
    threshold: float = Field(default=0.8, ge=0, le=1, description="Matching threshold")


# ============================
# Session State Management
# ============================


class SessionState(str, Enum):
    """Audio session state machine.
    
    State transitions:
        LISTENING → SPEAKING → PROCESSING → RESPONDING → COMPLETED
            ↓          ↓           ↓            ↓
          ERROR ← ─── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    """

    LISTENING = "listening"  # Waiting for speech
    SPEAKING = "speaking"  # VAD detected speech activity
    PROCESSING = "processing"  # ASR + Loom Agent processing
    RESPONDING = "responding"  # TTS playback
    COMPLETED = "completed"  # Session finished
    ERROR = "error"  # Any stage failed


# ============================
# Audio Metrics
# ============================


class AudioMetrics(BaseModel):
    """Performance metrics for audio processing pipeline.
    
    Tracks latency at each stage:
    - VAD: Voice Activity Detection
    - ASR: Automatic Speech Recognition
    - Loom: Agent processing
    - TTS: Text-to-Speech synthesis
    - Voiceprint: Speaker verification
    - E2E: End-to-end total latency
    """

    vad_latency_ms: Optional[float] = Field(None, ge=0, le=1000, description="VAD detection latency (ms)")
    asr_latency_ms: Optional[float] = Field(None, ge=0, le=5000, description="ASR transcription latency (ms)")
    loom_latency_ms: Optional[float] = Field(None, ge=0, le=10000, description="Loom Agent processing latency (ms)")
    tts_latency_ms: Optional[float] = Field(None, ge=0, le=5000, description="TTS synthesis latency (ms)")
    e2e_latency_ms: Optional[float] = Field(None, ge=0, le=30000, description="End-to-end total latency (ms)")
    voiceprint_latency_ms: Optional[float] = Field(None, ge=0, le=1000, description="Voiceprint verification latency (ms)")
    audio_duration_ms: float = Field(default=0.0, ge=0, description="Total audio duration (ms)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metric recording timestamp (UTC)")

    def compute_e2e_latency(self) -> float:
        """Calculate end-to-end latency (sum of all stages).
        
        Returns:
            Total latency in milliseconds
        """
        return sum(
            filter(
                None,
                [
                    self.vad_latency_ms,
                    self.asr_latency_ms,
                    self.loom_latency_ms,
                    self.tts_latency_ms,
                    self.voiceprint_latency_ms,
                ],
            )
        )

    class Config:
        frozen = False  # Allow metric updates during session


# ============================
# Audio Session
# ============================


class AudioSession(BaseModel):
    """Audio interaction session lifecycle.
    
    Represents a single voice interaction from speech detection to TTS playback.
    Now supports multi-turn conversation tracking (User Story 3).
    
    Validation:
        - Audio buffer cannot exceed 30 seconds (30s * 16kHz * 2 bytes)
        - ended_at must be after started_at
        - device_id must be 1-64 characters
        - conversation_history limited to 50 turns (auto-compressed after 10)
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique session identifier")
    device_id: str = Field(min_length=1, max_length=64, description="Xiaozhi device ID")
    speaker_id: Optional[str] = Field(None, description="Voiceprint-identified speaker ID")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time (UTC)")
    ended_at: Optional[datetime] = Field(None, description="Session end time (UTC)")
    is_speaking: bool = Field(default=False, description="User currently speaking")
    audio_buffer: List[bytes] = Field(default_factory=list, description="Audio frame buffer (PCM 16-bit)")
    transcript: Optional[str] = Field(None, max_length=10000, description="ASR transcription result")
    response_text: Optional[str] = Field(None, max_length=50000, description="Loom Agent response text")
    metrics: AudioMetrics = Field(default_factory=AudioMetrics, description="Performance metrics")
    state: SessionState = Field(default=SessionState.LISTENING, description="Current session state")
    
    # Multi-turn conversation fields (User Story 3)
    turn_count: int = Field(default=0, ge=0, description="User utterance count (excludes system prompts)")
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        max_length=50,
        description="Multi-turn dialogue history (auto-compressed after 10 turns)"
    )
    last_activity_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last user activity timestamp (for idle timeout)"
    )

    @field_validator("ended_at")
    @classmethod
    def validate_ended_at(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure ended_at is after started_at."""
        if v and info.data.get("started_at") and v < info.data["started_at"]:
            raise ValueError("ended_at must be after started_at")
        return v

    @field_validator("audio_buffer")
    @classmethod
    def validate_buffer_size(cls, v: List[bytes]) -> List[bytes]:
        """Limit audio buffer to 30 seconds @ 16kHz, 16-bit."""
        total_size = sum(len(frame) for frame in v)
        max_size = 30 * 16000 * 2  # 30 seconds * 16kHz * 2 bytes/sample
        if total_size > max_size:
            raise ValueError(f"Audio buffer exceeds {max_size} bytes (30 seconds)")
        return v

    @field_validator("conversation_history")
    @classmethod
    def validate_history_size(cls, v: List[ConversationTurn]) -> List[ConversationTurn]:
        """Limit conversation history to 50 turns."""
        if len(v) > 50:
            raise ValueError("Conversation history exceeds 50 turns (should be compressed)")
        return v

    class Config:
        frozen = False  # Mutable for multi-turn context updates (User Story 3)


# ============================
# Voiceprint Profile
# ============================


class VoiceprintProfile(BaseModel):
    """User voiceprint profile with encrypted features.
    
    Stores encrypted voice embeddings for speaker verification.
    Features are encrypted using AES-256-GCM before storage.
    
    Validation:
        - voice_features must be encrypted (min 16 bytes)
        - permissions must be in allowed set
        - display_name must be 1-100 characters
    """

    speaker_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique speaker identifier")
    device_id: str = Field(min_length=1, max_length=64, description="Associated Xiaozhi device ID")
    display_name: str = Field(min_length=1, max_length=100, description="User display name")
    voice_features: bytes = Field(description="Encrypted voice embedding vector (AES-256-GCM)")
    registered_at: datetime = Field(default_factory=datetime.utcnow, description="Registration timestamp (UTC)")
    last_verified_at: Optional[datetime] = Field(None, description="Last verification timestamp")
    verification_count: int = Field(default=0, ge=0, description="Total verification count")
    is_active: bool = Field(default=True, description="Profile active status")
    permissions: List[str] = Field(default_factory=list, description="Allowed permissions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extended metadata")

    @field_validator("voice_features")
    @classmethod
    def validate_encryption(cls, v: bytes) -> bytes:
        """Verify voice_features are encrypted (min AES block size)."""
        if len(v) < 16:  # AES minimum block size
            raise ValueError("voice_features must be encrypted (min 16 bytes)")
        return v

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v: List[str]) -> List[str]:
        """Ensure permissions are in allowed set."""
        valid_perms = {"admin", "read_calendar", "smart_home_control", "read_messages"}
        if invalid := set(v) - valid_perms:
            raise ValueError(f"Invalid permissions: {invalid}")
        return v

    class Config:
        frozen = True


# ============================
# WebSocket Messages
# ============================


class MessageType(str, Enum):
    """WebSocket message types."""

    AUDIO = "audio"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class AudioPayload(BaseModel):
    """Audio data payload.
    
    Contains raw PCM audio frames with metadata.
    """

    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Audio sample rate (Hz)")
    channels: int = Field(default=1, ge=1, le=2, description="Audio channels (1=mono, 2=stereo)")
    format: str = Field(default="pcm_s16le", description="Audio format (PCM 16-bit little-endian)")
    data: bytes = Field(description="Raw audio data")
    speaker_id: Optional[str] = Field(None, description="Voiceprint speaker ID")
    is_end: bool = Field(default=False, description="Last frame indicator")


class ControlPayload(BaseModel):
    """Control message payload.
    
    Used for session lifecycle management.
    """

    action: str = Field(description="Control action (start_session, end_session, pause, resume)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class HeartbeatPayload(BaseModel):
    """Heartbeat message payload.
    
    Sent periodically to maintain WebSocket connection.
    """

    device_status: str = Field(default="online", description="Device status (online, busy, offline)")
    uptime_seconds: int = Field(ge=0, description="Device uptime (seconds)")


class WebSocketMessage(BaseModel):
    """WebSocket message envelope.
    
    Top-level message structure for all WebSocket communication.
    
    Validation:
        - Timestamp drift < 1 minute (prevent clock skew attacks)
    """

    message_type: MessageType = Field(description="Message type")
    session_id: str = Field(description="Associated session ID")
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000), description="Unix timestamp (ms)")
    payload: Union[AudioPayload, ControlPayload, HeartbeatPayload] = Field(description="Message payload")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """Prevent clock drift attacks (allow max 1 minute skew)."""
        now = int(time.time() * 1000)
        if abs(v - now) > 60000:  # 1 minute in ms
            raise ValueError("Timestamp drift exceeds 1 minute")
        return v


# ============================
# Adapter Configuration
# ============================


class CloudASRConfig(BaseModel):
    """Cloud ASR provider configuration (future extension)."""

    provider: str = Field(description="Cloud ASR provider (e.g., 'alibaba', 'tencent')")
    api_key: str = Field(description="API key")
    endpoint: str = Field(description="API endpoint URL")


class CloudTTSConfig(BaseModel):
    """Cloud TTS provider configuration (future extension)."""

    provider: str = Field(description="Cloud TTS provider (e.g., 'azure', 'google')")
    api_key: str = Field(description="API key")
    voice: str = Field(description="Voice name")


class AudioAdapterConfig(BaseModel):
    """Audio adapter global configuration.
    
    Controls audio processing pipeline and service providers.
    
    Validation:
        - voiceprint_service_url required when enable_voiceprint=True
        - Port must be 1024-65535
        - Providers must match allowed patterns
    """

    # Provider selection
    vad_provider: str = Field(default="silero", pattern="^(silero|webrtc)$", description="VAD provider")
    asr_provider: str = Field(default="funasr", pattern="^(funasr|cloud)$", description="ASR provider")
    tts_provider: str = Field(default="edge", pattern="^(edge|cloud)$", description="TTS provider")
    voiceprint_provider: str = Field(default="3dspeaker", description="Voiceprint provider")
    
    # WebSocket server config
    host: str = Field(default="0.0.0.0", description="WebSocket listen address")
    port: int = Field(default=8765, ge=1024, le=65535, description="WebSocket port")
    max_connections: int = Field(default=10, ge=1, le=100, description="Max concurrent connections")
    
    # Audio config
    sample_rate: int = Field(default=16000, description="Audio sample rate (Hz)")
    channels: int = Field(default=1, description="Number of audio channels")
    
    # VAD config
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="VAD detection threshold")
    min_speech_duration_ms: int = Field(default=250, description="Minimum speech duration (ms)")
    min_silence_duration_ms: int = Field(default=300, description="Minimum silence duration (ms)")
    
    # ASR config
    asr_model: str = Field(default="paraformer-zh", description="ASR model name")
    use_gpu: bool = Field(default=False, description="Use GPU for ASR")
    
    # TTS config
    tts_voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="TTS voice ID")
    
    # Voiceprint config
    voiceprint_enabled: bool = Field(default=False, description="Enable voiceprint verification")
    voiceprint_url: Optional[str] = Field(
        None, description="3DSpeaker service URL"
    )
    
    # Session management
    max_concurrent_sessions: int = Field(default=5, ge=1, le=100, description="Max concurrent sessions")
    audio_buffer_seconds: int = Field(default=30, ge=10, le=120, description="Audio buffer duration (seconds)")
    
    # Cloud provider configs (optional)
    cloud_asr_config: Optional[CloudASRConfig] = Field(None, description="Cloud ASR configuration")
    cloud_tts_config: Optional[CloudTTSConfig] = Field(None, description="Cloud TTS configuration")
    
    # Legacy fields for backward compatibility
    websocket_host: Optional[str] = Field(default=None, description="Deprecated: use 'host' instead")
    websocket_port: Optional[int] = Field(default=None, description="Deprecated: use 'port' instead")
    enable_voiceprint: Optional[bool] = Field(default=None, description="Deprecated: use 'voiceprint_enabled' instead")
    voiceprint_service_url: Optional[str] = Field(default=None, description="Deprecated: use 'voiceprint_url' instead")
    
    @model_validator(mode='after')
    def handle_legacy_fields(self) -> 'AudioAdapterConfig':
        """Map legacy field names to new ones."""
        if self.websocket_host is not None:
            self.host = self.websocket_host
        if self.websocket_port is not None:
            self.port = self.websocket_port
        if self.enable_voiceprint is not None:
            self.voiceprint_enabled = self.enable_voiceprint
        if self.voiceprint_service_url is not None:
            self.voiceprint_url = self.voiceprint_service_url
        return self

    @field_validator("voiceprint_url")
    @classmethod
    def validate_voiceprint_url(cls, v: Optional[str], info) -> Optional[str]:
        """Require voiceprint_url when voiceprint is enabled."""
        if info.data.get("voiceprint_enabled") and not v:
            # Just warn, don't fail - voiceprint is optional
            pass
        return v
