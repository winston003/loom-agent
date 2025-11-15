"""Audio-Specific Event Helpers

Helper functions for emitting audio adapter events with structured metadata.

Reference: specs/002-xiaozhi-voice-adapter/
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loom.adapters.audio.models import AudioMetrics, AudioSession
from loom.core.events import AgentEvent, AgentEventType


def emit_audio_session_start(session: AudioSession, **extra_metadata) -> AgentEvent:
    """Emit AUDIO_SESSION_START event.

    Args:
        session: Audio session that started
        **extra_metadata: Additional event metadata

    Returns:
        AgentEvent with session details

    Example:
        >>> session = AudioSession(device_id="xiaozhi_001")
        >>> event = emit_audio_session_start(session, voiceprint_enabled=True)
        >>> print(event.metadata["session_id"])
    """
    metadata = {
        "session_id": session.session_id,
        "device_id": session.device_id,
        "speaker_id": session.speaker_id,
        "started_at": session.started_at.isoformat(),
        "state": session.state.value,
        **extra_metadata,
    }

    return AgentEvent(
        type=AgentEventType.AUDIO_SESSION_START,
        content=f"Audio session started: {session.session_id}",
        metadata=metadata,
    )


def emit_audio_session_end(session: AudioSession, **extra_metadata) -> AgentEvent:
    """Emit AUDIO_SESSION_END event.

    Args:
        session: Audio session that ended
        **extra_metadata: Additional event metadata

    Returns:
        AgentEvent with session completion details

    Example:
        >>> event = emit_audio_session_end(session, final_state="completed")
        >>> print(event.metadata["duration_ms"])
    """
    # Calculate session duration
    duration_ms = None
    if session.ended_at and session.started_at:
        duration_ms = (session.ended_at - session.started_at).total_seconds() * 1000

    metadata = {
        "session_id": session.session_id,
        "device_id": session.device_id,
        "speaker_id": session.speaker_id,
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "duration_ms": duration_ms,
        "state": session.state.value,
        "transcript": session.transcript,
        "response_text": session.response_text,
        **extra_metadata,
    }

    return AgentEvent(
        type=AgentEventType.AUDIO_SESSION_END,
        content=f"Audio session ended: {session.session_id}",
        metadata=metadata,
    )


def emit_audio_metrics_event(session_id: str, metrics: AudioMetrics, **extra_metadata) -> AgentEvent:
    """Emit audio performance metrics event.

    Args:
        session_id: Session identifier
        metrics: Audio performance metrics
        **extra_metadata: Additional event metadata

    Returns:
        AgentEvent with detailed performance data

    Example:
        >>> metrics = AudioMetrics(vad_latency_ms=50, asr_latency_ms=300)
        >>> event = emit_audio_metrics_event("session-123", metrics)
        >>> print(event.metadata["e2e_latency_ms"])
    """
    metadata = {
        "session_id": session_id,
        "vad_latency_ms": metrics.vad_latency_ms,
        "asr_latency_ms": metrics.asr_latency_ms,
        "loom_latency_ms": metrics.loom_latency_ms,
        "tts_latency_ms": metrics.tts_latency_ms,
        "e2e_latency_ms": metrics.e2e_latency_ms or metrics.compute_e2e_latency(),
        "voiceprint_latency_ms": metrics.voiceprint_latency_ms,
        "audio_duration_ms": metrics.audio_duration_ms,
        "timestamp": metrics.timestamp.isoformat(),
        **extra_metadata,
    }

    return AgentEvent(
        type=AgentEventType.PHASE_END,  # Use existing event type for metrics
        content=f"Audio pipeline metrics for session {session_id}",
        metadata=metadata,
    )


def emit_audio_vad_event(
    session_id: str,
    event_name: str,
    speech_detected: bool,
    confidence: Optional[float] = None,
    duration_ms: Optional[float] = None,
    **extra_metadata,
) -> AgentEvent:
    """Emit VAD (Voice Activity Detection) event.

    Args:
        session_id: Session identifier
        event_name: VAD event name (e.g., 'speech_start', 'speech_end')
        speech_detected: Whether speech was detected
        confidence: VAD confidence score (0-1)
        duration_ms: Speech segment duration
        **extra_metadata: Additional event metadata

    Returns:
        AgentEvent with VAD details

    Example:
        >>> event = emit_audio_vad_event(
        ...     "session-123",
        ...     "speech_detected",
        ...     speech_detected=True,
        ...     confidence=0.95
        ... )
    """
    metadata = {
        "session_id": session_id,
        "event_name": event_name,
        "speech_detected": speech_detected,
        "confidence": confidence,
        "duration_ms": duration_ms,
        **extra_metadata,
    }

    return AgentEvent(
        type=AgentEventType.PHASE_START,  # Use existing event type for VAD
        content=f"VAD: {event_name}",
        metadata=metadata,
    )


def emit_audio_asr_event(
    session_id: str,
    transcript: str,
    confidence: Optional[float] = None,
    language: Optional[str] = None,
    **extra_metadata,
) -> AgentEvent:
    """Emit ASR (Automatic Speech Recognition) event.

    Args:
        session_id: Session identifier
        transcript: Recognized text
        confidence: ASR confidence score (0-1)
        language: Detected language code
        **extra_metadata: Additional event metadata

    Returns:
        AgentEvent with transcription details

    Example:
        >>> event = emit_audio_asr_event(
        ...     "session-123",
        ...     "今天天气怎么样",
        ...     confidence=0.92,
        ...     language="zh-CN"
        ... )
    """
    metadata = {
        "session_id": session_id,
        "transcript": transcript,
        "confidence": confidence,
        "language": language,
        "transcript_length": len(transcript),
        **extra_metadata,
    }

    return AgentEvent(
        type=AgentEventType.TOOL_RESULT,  # Use existing event type for ASR
        content=f"ASR transcription: {transcript[:50]}..." if len(transcript) > 50 else f"ASR transcription: {transcript}",
        metadata=metadata,
    )


def emit_audio_tts_event(
    session_id: str,
    text: str,
    audio_duration_ms: float,
    voice: Optional[str] = None,
    **extra_metadata,
) -> AgentEvent:
    """Emit TTS (Text-to-Speech) event.

    Args:
        session_id: Session identifier
        text: Text to synthesize
        audio_duration_ms: Generated audio duration
        voice: TTS voice name
        **extra_metadata: Additional event metadata

    Returns:
        AgentEvent with TTS details

    Example:
        >>> event = emit_audio_tts_event(
        ...     "session-123",
        ...     "今天天气晴朗",
        ...     audio_duration_ms=2500,
        ...     voice="zh-CN-XiaoxiaoNeural"
        ... )
    """
    metadata = {
        "session_id": session_id,
        "text": text,
        "audio_duration_ms": audio_duration_ms,
        "voice": voice,
        "text_length": len(text),
        **extra_metadata,
    }

    return AgentEvent(
        type=AgentEventType.TOOL_RESULT,  # Use existing event type for TTS
        content=f"TTS synthesis: {text[:50]}..." if len(text) > 50 else f"TTS synthesis: {text}",
        metadata=metadata,
    )


def emit_audio_voiceprint_event(
    session_id: str,
    speaker_id: Optional[str],
    verified: bool,
    confidence: Optional[float] = None,
    **extra_metadata,
) -> AgentEvent:
    """Emit voiceprint verification event.

    Args:
        session_id: Session identifier
        speaker_id: Identified speaker ID (None if verification failed)
        verified: Whether voiceprint verification succeeded
        confidence: Voiceprint match confidence score (0-1)
        **extra_metadata: Additional event metadata

    Returns:
        AgentEvent with voiceprint details

    Example:
        >>> event = emit_audio_voiceprint_event(
        ...     "session-123",
        ...     "speaker-456",
        ...     verified=True,
        ...     confidence=0.88
        ... )
    """
    metadata = {
        "session_id": session_id,
        "speaker_id": speaker_id,
        "verified": verified,
        "confidence": confidence,
        **extra_metadata,
    }

    return AgentEvent(
        type=AgentEventType.TOOL_RESULT,  # Use existing event type for voiceprint
        content=f"Voiceprint: {'verified' if verified else 'failed'} (speaker={speaker_id})",
        metadata=metadata,
    )
