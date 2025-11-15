"""AudioSessionManager

Manages audio sessions: creation, buffering, state transitions and orchestration
of the simple VAD -> ASR -> (optional) voiceprint pipeline.

This component is intentionally conservative: it defers to injected service
implementations (`vad`, `asr`, `tts`, `voiceprint_client`, `voiceprint_storage`)
so tests can stub them easily.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from loom.adapters.audio.models import (
    AudioSession,
    AudioMetrics,
    AudioFrame,
    TranscriptionResult,
    SessionState,
    ConversationTurn,
)
from loom.core.structured_logger import get_logger
from loom.adapters.audio.events import (
    emit_audio_session_start,
    emit_audio_session_end,
    emit_audio_asr_event,
    emit_audio_metrics_event,
)

logger = get_logger("audio.session")


class AudioSessionManager:
    """Manage AudioSession lifecycle and processing pipeline.

    Args:
        vad: VAD provider implementing `detect(AudioFrame) -> Optional[SpeechSegment]`
        asr: ASR provider implementing `transcribe(bytes) -> TranscriptionResult`
        tts: Optional TTS provider (for later use)
        voiceprint_client: Optional client to verify voiceprint
        voiceprint_storage: Optional storage for encrypted voiceprints
        max_sessions: Maximum concurrent sessions allowed
    """

    def __init__(
        self,
        vad,
        asr,
        tts=None,
        voiceprint_client=None,
        voiceprint_storage=None,
        max_sessions: int = 5,
        on_emit: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.vad = vad
        self.asr = asr
        self.tts = tts
        self.voiceprint_client = voiceprint_client
        self.voiceprint_storage = voiceprint_storage
        self.max_sessions = max_sessions

        # session_id -> AudioSession
        self.sessions: Dict[str, AudioSession] = {}

        # Keep background processing tasks per session
        self._processing_tasks: Dict[str, asyncio.Task] = {}

        # Optional event emitter callback (e.g., to push to AgentEvent stream)
        self.on_emit = on_emit

        # Simple lock to guard session creation
        self._lock = asyncio.Lock()

    async def create_session(self, device_id: str) -> str:
        """Create a new audio session and return its session_id."""
        async with self._lock:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError("Max sessions exceeded")

            session_id = str(uuid4())
            session = AudioSession(
                session_id=session_id,
                device_id=device_id,
                metrics=AudioMetrics(),
                state=SessionState.LISTENING,
            )
            self.sessions[session_id] = session

            # Emit start event
            ev = emit_audio_session_start(session)
            if self.on_emit:
                self.on_emit(ev)

            logger.info("Created audio session", session_id=session_id, device_id=device_id)
            return session_id

    async def close_session(self, session_id: str) -> None:
        """Close and cleanup a session."""
        session = self.sessions.get(session_id)
        if not session:
            logger.warning("close_session: session not found", session_id=session_id)
            return

        session.ended_at = __import__("datetime").datetime.utcnow()
        session.state = SessionState.COMPLETED

        # Emit end event
        ev = emit_audio_session_end(session)
        if self.on_emit:
            self.on_emit(ev)

        # Cancel any background processing
        task = self._processing_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()

        # Remove session from registry
        self.sessions.pop(session_id, None)

        logger.info("Closed audio session", session_id=session_id)

    async def handle_audio_frame(self, session_id: str, audio_data: bytes, metadata: Dict[str, Any]) -> None:
        """Called when a new audio frame (binary) arrives for a session.

        This will buffer audio, invoke VAD with an `AudioFrame`, and when a speech
        segment ends, dispatch a background task to run ASR and downstream steps.
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning("handle_audio_frame: session not found", session_id=session_id)
            return

        # Append to buffer
        session.audio_buffer.append(audio_data)

        # Update audio duration estimate (bytes -> ms)
        # Assume PCM 16-bit little-endian, channels metadata or default 1
        sample_rate = metadata.get("sample_rate", 16000)
        channels = metadata.get("channels", 1)
        bytes_per_sample = 2
        duration_ms = (len(audio_data) / (sample_rate * channels * bytes_per_sample)) * 1000
        session.metrics.audio_duration_ms += duration_ms

        # Construct AudioFrame and run VAD
        frame = AudioFrame(data=audio_data, sample_rate=sample_rate, channels=channels)

        try:
            seg = self.vad.detect(frame)
        except Exception as e:
            logger.error("VAD processing failed", error=str(e), session_id=session_id, exc_info=e)
            return

        # If VAD indicates a completed speech segment, schedule processing
        if seg is not None:
            # Collect all buffered audio for this segment
            audio_bytes = b"".join(session.audio_buffer)
            # Clear the session buffer for next speech
            session.audio_buffer.clear()

            # Update session state
            session.state = SessionState.PROCESSING

            # Spawn background task to process speech (ASR, voiceprint, events)
            task = asyncio.create_task(self._process_speech_segment(session_id, audio_bytes))
            self._processing_tasks[session_id] = task

    async def _process_speech_segment(self, session_id: str, audio_bytes: bytes) -> None:
        """Process a finished speech segment: ASR -> voiceprint verification -> update session (T062).
        
        Pipeline flow:
        1. ASR transcription (extract text)
        2. Voiceprint verification (identify speaker)
        3. Inject speaker_id into session metadata
        4. Update verification statistics
        5. Emit events for observability
        
        Verification failure handling (T063):
        - Returns None speaker_id → anonymous mode
        - Logs failure for monitoring
        - Continues with transcription (no blocking)
        
        Reference: specs/002-xiaozhi-voice-adapter/tasks.md T062-T063
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning("_process_speech_segment: session missing", session_id=session_id)
            return

        start = time.time()
        try:
            # Step 1: ASR transcription (may be CPU/GPU bound)
            result: TranscriptionResult = await self.asr.transcribe(audio_bytes)
            session.transcript = result.text
            session.metrics.asr_latency_ms = result.latency_ms

            logger.info(
                "ASR transcription completed",
                session_id=session_id,
                text=result.text[:50],  # Log first 50 chars
                confidence=result.confidence,
                latency_ms=result.latency_ms,
            )

            # Emit ASR event
            ev = emit_audio_asr_event(session_id, result.text, confidence=result.confidence, language=result.language)
            if self.on_emit:
                self.on_emit(ev)

            # T077: Add conversation turn after ASR (track user utterance)
            # Note: Agent response will be added later via update_turn_response()
            self.add_conversation_turn(
                session_id=session_id,
                user_text=result.text,
                agent_response=None,  # To be filled later
                speaker_id=None,  # Will be updated after voiceprint verification
                metadata={
                    "asr_confidence": result.confidence,
                    "asr_language": result.language,
                    "asr_latency_ms": result.latency_ms,
                },
            )

            # Step 2: Voiceprint verification (T062 integration)
            # Note: This delegates to AudioAdapter.verify_speaker() which handles:
            # - 3DSpeaker API call
            # - Similarity threshold check
            # - Statistics update (verification_count, last_verified_at)
            # - Session metadata injection
            speaker_id = None
            verification_similarity = None
            
            if self.voiceprint_client and self.voiceprint_storage:
                try:
                    verify_start = time.time()
                    
                    # Call 3DSpeaker verification API
                    verify_resp = await self.voiceprint_client.verify(session.device_id, audio_bytes)
                    
                    # Extract speaker_id and similarity score
                    speaker_id = verify_resp.get("speaker_id")
                    verification_similarity = verify_resp.get("similarity", 0.0)
                    
                    verify_elapsed = (time.time() - verify_start) * 1000
                    
                    if speaker_id:
                        # Verification successful - inject into session
                        session.speaker_id = speaker_id
                        if not session.metadata:
                            session.metadata = {}
                        session.metadata["speaker_id"] = speaker_id
                        session.metadata["verification_similarity"] = verification_similarity
                        
                        # T078: Update speaker_id in latest conversation turn
                        if session.conversation_history:
                            latest_turn = session.conversation_history[-1]
                            updated_turn = ConversationTurn(
                                turn_index=latest_turn.turn_index,
                                timestamp=latest_turn.timestamp,
                                user_text=latest_turn.user_text,
                                agent_response=latest_turn.agent_response,
                                speaker_id=speaker_id,  # Update speaker ID
                                metadata={
                                    **latest_turn.metadata,
                                    "verification_similarity": verification_similarity,
                                    "voiceprint_latency_ms": verify_elapsed,
                                },
                            )
                            session.conversation_history[-1] = updated_turn
                        
                        # Update verification statistics (T061)
                        await self.voiceprint_storage.update_verification_stats(
                            session.device_id,
                            speaker_id,
                        )
                        
                        logger.info(
                            "Voiceprint verification succeeded",
                            session_id=session_id,
                            speaker_id=speaker_id,
                            similarity=verification_similarity,
                            elapsed_ms=verify_elapsed,
                        )
                        
                        # Emit voiceprint success event
                        ev_vp = emit_audio_asr_event(
                            session_id,
                            f"voiceprint_verified:{speaker_id}",
                            confidence=verification_similarity,
                        )
                        if self.on_emit:
                            self.on_emit(ev_vp)
                    else:
                        # Verification failed - log and continue in anonymous mode (T063)
                        logger.warning(
                            "Voiceprint verification failed - no matching speaker",
                            session_id=session_id,
                            similarity=verification_similarity,
                            elapsed_ms=verify_elapsed,
                            fallback="anonymous_mode",
                        )
                        
                        # Emit voiceprint failure event
                        ev_vp_fail = emit_audio_asr_event(
                            session_id,
                            "voiceprint_failed:anonymous",
                            confidence=verification_similarity,
                        )
                        if self.on_emit:
                            self.on_emit(ev_vp_fail)
                
                except Exception as e:
                    # T063: Verification error handling - log and continue
                    logger.error(
                        "Voiceprint verification error",
                        error=str(e),
                        session_id=session_id,
                        fallback="anonymous_mode",
                        exc_info=True,
                    )
                    
                    # Emit error event
                    ev_error = emit_audio_asr_event(session_id, f"voiceprint_error:{type(e).__name__}")
                    if self.on_emit:
                        self.on_emit(ev_error)
            else:
                # Voiceprint not configured - skip verification
                logger.debug(
                    "Voiceprint verification skipped - not configured",
                    session_id=session_id,
                )

            # Mark session processed and ready for responding
            session.state = SessionState.RESPONDING

            # Update Loom metrics (approx)
            session.metrics.loom_latency_ms = (time.time() - start) * 1000

            # Emit metrics event
            ev_metrics = emit_audio_metrics_event(session_id, session.metrics)
            if self.on_emit:
                self.on_emit(ev_metrics)

        except Exception as e:
            logger.error("Error during speech processing", error=str(e), session_id=session_id, exc_info=e)
            # Mark session error state but keep it alive
            session.state = SessionState.ERROR

    def get_session(self, session_id: str) -> Optional[AudioSession]:
        return self.sessions.get(session_id)

    def get_active_session_count(self) -> int:
        return len(self.sessions)

    # ============================
    # T077-T078: Multi-turn conversation management
    # ============================

    def add_conversation_turn(
        self,
        session_id: str,
        user_text: str,
        agent_response: Optional[str] = None,
        speaker_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a conversation turn to session history and increment turn count.
        
        Called after ASR transcription to track multi-turn dialogue context.
        This implements T077 (turn counting) and T078 (history storage).
        
        Args:
            session_id: Target session ID
            user_text: User's transcribed utterance
            agent_response: Agent's response text (optional, can be added later)
            speaker_id: Identified speaker for this turn
            metadata: Additional turn metadata (e.g., permissions, confirmation state)
        
        Raises:
            KeyError: If session_id not found
        """
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")

        # T077: Increment turn count (user utterances only, excludes system prompts)
        session.turn_count += 1
        
        # T078: Store conversation turn in history
        turn = ConversationTurn(
            turn_index=session.turn_count - 1,  # 0-based index
            timestamp=datetime.utcnow(),
            user_text=user_text,
            agent_response=agent_response,
            speaker_id=speaker_id or session.speaker_id,
            metadata=metadata or {},
        )
        session.conversation_history.append(turn)
        
        # Update last activity timestamp for idle timeout tracking
        session.last_activity_at = datetime.utcnow()
        
        logger.info(
            "Added conversation turn",
            session_id=session_id,
            turn_index=turn.turn_index,
            turn_count=session.turn_count,
            speaker_id=turn.speaker_id,
            user_text_preview=user_text[:50],
        )

    def update_turn_response(
        self,
        session_id: str,
        turn_index: int,
        agent_response: str,
    ) -> None:
        """Update agent response for a specific turn.
        
        Used when agent response is generated after ASR transcription.
        
        Args:
            session_id: Target session ID
            turn_index: Turn index to update (0-based)
            agent_response: Agent's response text
        
        Raises:
            KeyError: If session_id not found
            IndexError: If turn_index out of range
        """
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")
        
        if turn_index < 0 or turn_index >= len(session.conversation_history):
            raise IndexError(f"Turn index {turn_index} out of range (0-{len(session.conversation_history) - 1})")
        
        # Note: ConversationTurn is frozen, so we need to replace the entire turn
        old_turn = session.conversation_history[turn_index]
        updated_turn = ConversationTurn(
            turn_index=old_turn.turn_index,
            timestamp=old_turn.timestamp,
            user_text=old_turn.user_text,
            agent_response=agent_response,
            speaker_id=old_turn.speaker_id,
            metadata=old_turn.metadata,
        )
        session.conversation_history[turn_index] = updated_turn
        
        logger.debug(
            "Updated turn response",
            session_id=session_id,
            turn_index=turn_index,
            response_preview=agent_response[:50],
        )

    def get_conversation_context(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
    ) -> List[ConversationTurn]:
        """Retrieve conversation history for context injection.
        
        Args:
            session_id: Target session ID
            max_turns: Maximum number of recent turns to return (None = all)
        
        Returns:
            List of ConversationTurn objects (most recent last)
        
        Raises:
            KeyError: If session_id not found
        """
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")
        
        history = session.conversation_history
        if max_turns is not None and max_turns > 0:
            # Return last N turns
            return history[-max_turns:]
        return history

    def check_idle_timeout(
        self,
        session_id: str,
        timeout_seconds: int = 30,
    ) -> bool:
        """Check if session has exceeded idle timeout.
        
        Used for automatic session cleanup after inactivity.
        Implements User Story 3 requirement: "30 秒无语音自动结束会话"
        
        Args:
            session_id: Target session ID
            timeout_seconds: Idle timeout in seconds (default: 30)
        
        Returns:
            True if session is idle and should be closed
        
        Raises:
            KeyError: If session_id not found
        """
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")
        
        idle_duration = (datetime.utcnow() - session.last_activity_at).total_seconds()
        is_idle = idle_duration >= timeout_seconds
        
        if is_idle:
            logger.info(
                "Session idle timeout detected",
                session_id=session_id,
                idle_duration_s=idle_duration,
                timeout_s=timeout_seconds,
            )
        
        return is_idle

    async def cleanup_idle_sessions(
        self,
        timeout_seconds: int = 30,
    ) -> List[str]:
        """Clean up idle sessions exceeding timeout.
        
        Implements T085 (session idle timeout).
        Should be called periodically by background task.
        
        Args:
            timeout_seconds: Idle timeout in seconds (default: 30)
        
        Returns:
            List of closed session IDs
        """
        closed_sessions = []
        
        for session_id in list(self.sessions.keys()):
            try:
                if self.check_idle_timeout(session_id, timeout_seconds):
                    await self.close_session(session_id)
                    closed_sessions.append(session_id)
                    logger.info(
                        "Closed idle session",
                        session_id=session_id,
                        timeout_s=timeout_seconds,
                    )
            except KeyError:
                # Session was already closed
                pass
        
        return closed_sessions

    def auto_compress_if_needed(self, session_id: str) -> bool:
        """Automatically compress conversation history if threshold exceeded.
        
        Implements T086 (long conversation management).
        This is a simplified version - actual compression is handled by
        ConversationContextManager.assemble_context(compress=True).
        
        This method just checks if compression should be triggered.
        
        Args:
            session_id: Target session ID
        
        Returns:
            True if compression was needed (turn_count >= 10)
        
        Raises:
            KeyError: If session_id not found
        """
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")
        
        should_compress = session.turn_count >= 10
        
        if should_compress:
            logger.info(
                "Auto-compression triggered",
                session_id=session_id,
                turn_count=session.turn_count,
                threshold=10,
            )
        
        return should_compress


# Small convenience factory to wire manager into WebSocketAudioServer callbacks
def make_session_manager_from_config(config: Dict[str, Any]) -> AudioSessionManager:
    """Create a preconfigured AudioSessionManager from a config dict or Pydantic model.

    This factory will lazily import providers based on config to avoid hard
    dependencies at module import time.
    """
    # Lazy imports to avoid heavy dependencies during test-time
    from loom.adapters.audio.vad.silero import SileroVAD
    from loom.adapters.audio.asr.funasr import FunASR
    from loom.adapters.audio.models import AudioAdapterConfig
    
    # Handle both dict and Pydantic model
    if isinstance(config, AudioAdapterConfig):
        # Use Pydantic model fields directly
        vad = SileroVAD(
            model_dir="models/silero_vad",
            threshold=config.vad_threshold,
            threshold_low=0.2,
            min_silence_duration_ms=config.min_silence_duration_ms,
            sample_rate=config.sample_rate,
        )
        
        asr = FunASR(
            model_dir=f"models/{config.asr_model}",
            output_dir="temp/audio",
            delete_temp_files=True,
        )
    else:
        # Handle dict config for backward compatibility
        vad = SileroVAD(
            model_dir=config.get("vad_model_dir", "models/silero_vad"),
            threshold=config.get("vad_threshold", 0.5),
            threshold_low=config.get("vad_threshold_low", 0.2),
            min_silence_duration_ms=config.get("min_silence_duration_ms", 1000),
            frame_window_threshold=config.get("frame_window_threshold", 3),
        )
        
        asr = FunASR(
            model_dir=config.get("asr_model_dir", "models/funasr"),
            output_dir=config.get("asr_output_dir", "temp"),
            delete_temp_files=True,
        )

    # Optionally configure TTS and voiceprint clients later
    tts = None
    voiceprint_client = None
    voiceprint_storage = None

    return AudioSessionManager(vad=vad, asr=asr, tts=tts, voiceprint_client=voiceprint_client, voiceprint_storage=voiceprint_storage)
