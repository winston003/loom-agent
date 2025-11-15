"""
AudioAdapter - Main entry point for Xiaozhi voice companion integration.

This module provides the AudioAdapter class that composes all audio subsystems:
- WebSocket server for real-time bidirectional communication
- AudioSessionManager for session lifecycle and pipeline orchestration
- VAD/ASR/TTS/Voiceprint providers

Architecture:
    Client (WebSocket) → AudioAdapter → {
        WebSocketAudioServer → AudioSessionManager → {
            VAD (Silero) → ASR (FunASR) → Voiceprint → [Loom Agent] → TTS (EdgeTTS)
        }
    }

Usage:
    ```python
    from loom.adapters.audio import AudioAdapter, AudioAdapterConfig
    
    # Create adapter with default config
    adapter = AudioAdapter()
    
    # Or with custom config
    config = AudioAdapterConfig(
        host="0.0.0.0",
        port=8765,
        vad_threshold=0.5,
        sample_rate=16000
    )
    adapter = AudioAdapter(config)
    
    # Start server
    await adapter.start()
    
    # Process audio in your pipeline
    # (WebSocket clients connect and send audio frames automatically)
    
    # Stop server
    await adapter.stop()
    ```

Reference: specs/002-xiaozhi-voice-adapter/contracts/audio_adapter_interface.py
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable
from uuid import uuid4

from loom.interfaces.audio_adapter import (
    BaseAudioAdapter,
    AudioFrame,
    SpeechSegment,
    TranscriptionResult,
    AudioAdapterError,
)
from loom.adapters.audio.models import AudioAdapterConfig, AudioSession
from loom.adapters.audio.websocket.server import WebSocketAudioServer
from loom.adapters.audio.session import AudioSessionManager, make_session_manager_from_config
from loom.adapters.audio.vad import SileroVAD
from loom.adapters.audio.asr import FunASR
from loom.adapters.audio.tts import EdgeTTS
from loom.adapters.audio.voiceprint import ThreeDSpeakerClient, VoiceprintStorage
from loom.adapters.audio.permissions import AudioPermissionManager  # T064
from loom.adapters.audio.confirmation import ConfirmationManager  # T067-T069
from loom.adapters.audio.context import ConversationContextManager  # T079-T081
from loom.adapters.audio.performance import (  # T089-T092
    TTLCache,
    ConcurrencyLimiter,
    ResourceCleanupManager,
)
from loom.adapters.audio.resilience import (  # T097-T100
    RetryPolicy,
    RetryConfig,
    RetryStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    FallbackManager,
    FallbackConfig,
    FallbackStrategy,
    RecoveryManager,
    RecoveryAction,
)
from loom.core.structured_logger import get_logger

logger = get_logger(__name__)


class AudioAdapter(BaseAudioAdapter):
    """
    Main AudioAdapter implementation for Xiaozhi voice companion.
    
    Responsibilities:
    - Orchestrate WebSocket server lifecycle (start/stop)
    - Compose session manager with VAD/ASR/TTS/voiceprint providers
    - Expose high-level API for audio processing and synthesis
    - Integrate with Loom Agent event system for observability
    
    Attributes:
        config: Audio adapter configuration (host, port, thresholds, etc.)
        session_manager: Manages audio sessions and processing pipeline
        websocket_server: WebSocket server for client connections
        vad: Voice Activity Detection provider (default: SileroVAD)
        asr: Automatic Speech Recognition provider (default: FunASR)
        tts: Text-to-Speech provider (default: EdgeTTS)
        voiceprint_client: Voiceprint verification client (optional)
        voiceprint_storage: Encrypted voiceprint storage (optional)
    """

    def __init__(
        self,
        config: Optional[AudioAdapterConfig] = None,
        on_transcription: Optional[Callable[[str, TranscriptionResult], Any]] = None,
        on_session_start: Optional[Callable[[AudioSession], Any]] = None,
        on_session_end: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize AudioAdapter with configuration and optional callbacks.
        
        Args:
            config: Audio adapter configuration. If None, uses defaults.
            on_transcription: Callback invoked when transcription completes.
                             Signature: (session_id: str, result: TranscriptionResult) -> Any
            on_session_start: Callback invoked when session starts.
                             Signature: (session: AudioSession) -> Any
            on_session_end: Callback invoked when session ends.
                           Signature: (session_id: str) -> Any
        """
        self.config = config or AudioAdapterConfig()
        self._on_transcription = on_transcription
        self._on_session_start = on_session_start
        self._on_session_end = on_session_end
        
        # Components (initialized in start())
        self.session_manager: Optional[AudioSessionManager] = None
        self.websocket_server: Optional[WebSocketAudioServer] = None
        self.vad: Optional[SileroVAD] = None
        self.asr: Optional[FunASR] = None
        self.tts: Optional[EdgeTTS] = None
        self.voiceprint_client: Optional[ThreeDSpeakerClient] = None
        self.voiceprint_storage: Optional[VoiceprintStorage] = None
        self.permission_manager: Optional[AudioPermissionManager] = None  # T064
        self.confirmation_manager: Optional[ConfirmationManager] = None  # T067-T069
        self.context_manager: Optional[ConversationContextManager] = None  # T079-T081
        
        # Performance optimization components (T089-T092)
        self.permission_cache: Optional[TTLCache] = None  # Cache permission check results
        self.context_cache: Optional[TTLCache] = None  # Cache assembled contexts
        self.voiceprint_cache: Optional[TTLCache] = None  # Cache voiceprint metadata
        self.concurrency_limiter: Optional[ConcurrencyLimiter] = None  # Limit concurrent sessions
        self.cleanup_manager: Optional[ResourceCleanupManager] = None  # Periodic cleanup
        
        # Resilience components (T097-T100)
        self.retry_policy: Optional[RetryPolicy] = None  # Retry with exponential backoff
        self.asr_circuit_breaker: Optional[CircuitBreaker] = None  # ASR circuit breaker
        self.voiceprint_circuit_breaker: Optional[CircuitBreaker] = None  # Voiceprint circuit breaker
        self.fallback_manager: Optional[FallbackManager] = None  # Fallback strategies
        self.recovery_manager: Optional[RecoveryManager] = None  # Recovery workflows
        
        self._running = False
        logger.info("AudioAdapter initialized", config=self.config.model_dump())

    async def start(self) -> None:
        """
        Start the audio adapter and WebSocket server.
        
        Steps:
        1. Initialize VAD/ASR/TTS providers
        2. Create session manager with providers
        3. Start WebSocket server with session manager
        
        Raises:
            AudioAdapterError: If initialization or server start fails
        """
        if self._running:
            logger.warning("AudioAdapter already running")
            return

        try:
            logger.info("Starting AudioAdapter...")
            
            # Initialize audio providers
            await self._initialize_providers()
            
            # Initialize permission manager (T064)
            self.permission_manager = AudioPermissionManager.from_config()
            logger.info("Audio permission manager initialized")
            
            # Initialize confirmation manager (T067-T069)
            self.confirmation_manager = ConfirmationManager(
                default_timeout=10,  # 10 seconds for confirmation
                max_retries=2,       # Maximum 2 retry attempts
            )
            logger.info("Confirmation manager initialized")
            
            # Initialize conversation context manager (T079-T081)
            self.context_manager = ConversationContextManager(
                max_context_tokens=2000,     # Max 2000 tokens for context
                compression_threshold=10,    # Compress after 10 turns
                keep_recent_turns=5,         # Keep last 5 turns uncompressed
            )
            logger.info("Context manager initialized")
            
            # Initialize performance optimization components (T089-T092)
            self.permission_cache = TTLCache(max_size=1000, ttl_seconds=300)  # 5 min TTL
            self.context_cache = TTLCache(max_size=500, ttl_seconds=60)       # 1 min TTL
            self.voiceprint_cache = TTLCache(max_size=100, ttl_seconds=600)   # 10 min TTL
            self.concurrency_limiter = ConcurrencyLimiter(
                global_limit=100,      # Max 100 concurrent sessions globally
                per_key_limit=5,       # Max 5 concurrent sessions per device
            )
            logger.info("Performance optimization components initialized")
            
            # Initialize cleanup manager and register handlers
            self.cleanup_manager = ResourceCleanupManager(cleanup_interval=60)
            self.cleanup_manager.register_handler(
                "permission_cache",
                self.permission_cache.cleanup_expired
            )
            self.cleanup_manager.register_handler(
                "context_cache",
                self.context_cache.cleanup_expired
            )
            self.cleanup_manager.register_handler(
                "voiceprint_cache",
                self.voiceprint_cache.cleanup_expired
            )
            self.cleanup_manager.register_handler(
                "idle_sessions",
                lambda: self.session_manager.cleanup_idle_sessions(timeout_seconds=30)
            )
            await self.cleanup_manager.start()
            logger.info("Cleanup manager started")
            
            # Initialize resilience components (T097-T100)
            self.retry_policy = RetryPolicy(RetryConfig(
                max_attempts=3,
                initial_delay=0.1,
                max_delay=5.0,
                backoff_factor=2.0,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            ))
            
            self.asr_circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=30.0,
                window_size=20,
                error_rate_threshold=0.5,
            ))
            
            self.voiceprint_circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=30.0,
                window_size=10,
                error_rate_threshold=0.4,
            ))
            
            self.fallback_manager = FallbackManager()
            self.fallback_manager.register(
                'asr',
                FallbackConfig(
                    strategy=FallbackStrategy.RETURN_DEFAULT,
                    default_value={'text': '[ASR service unavailable]', 'confidence': 0.0}
                )
            )
            self.fallback_manager.register(
                'voiceprint',
                FallbackConfig(
                    strategy=FallbackStrategy.USE_CACHE,
                    default_value={'verified': False, 'similarity': 0.0},
                    cache_ttl=300.0,
                )
            )
            
            self.recovery_manager = RecoveryManager()
            # Recovery actions will be registered by components as needed
            logger.info("Resilience components initialized (retry, circuit breakers, fallback)")
            
            # Create session manager from config (factory wires default providers)
            self.session_manager = make_session_manager_from_config(self.config)
            
            # Override VAD/ASR if we have custom instances
            if self.vad:
                self.session_manager.vad = self.vad
            if self.asr:
                self.session_manager.asr = self.asr
            
            # Create WebSocket server with session manager integration
            self.websocket_server = WebSocketAudioServer(
                host=self.config.host,
                port=self.config.port,
                max_connections=self.config.max_connections,
                session_manager=self.session_manager,
                on_session_start=self._on_session_start,
                on_session_end=self._on_session_end,
            )
            
            # Start WebSocket server (non-blocking)
            await self.websocket_server.start()
            
            self._running = True
            logger.info(
                "AudioAdapter started",
                host=self.config.host,
                port=self.config.port,
                vad_enabled=self.vad is not None,
                asr_enabled=self.asr is not None,
                tts_enabled=self.tts is not None,
            )
            
        except Exception as e:
            logger.error("Failed to start AudioAdapter", error=str(e), exc_info=e)
            raise AudioAdapterError(f"Failed to start AudioAdapter: {e}") from e

    async def stop(self) -> None:
        """
        Stop the audio adapter and clean up resources.
        
        Steps:
        1. Stop WebSocket server
        2. Close all active sessions
        3. Clean up provider resources
        """
        if not self._running:
            logger.warning("AudioAdapter not running")
            return

        try:
            logger.info("Stopping AudioAdapter...")
            
            # Stop cleanup manager first
            if self.cleanup_manager:
                await self.cleanup_manager.stop()
                logger.info("Cleanup manager stopped")
            
            # Stop WebSocket server
            if self.websocket_server:
                await self.websocket_server.stop()
            
            # Close all active sessions
            if self.session_manager:
                active_sessions = list(self.session_manager.sessions.keys())
                for session_id in active_sessions:
                    try:
                        await self.session_manager.close_session(session_id)
                    except Exception as e:
                        logger.warning(f"Failed to close session {session_id}", error=str(e))
            
            # Clear caches
            if self.permission_cache:
                await self.permission_cache.clear()
            if self.context_cache:
                await self.context_cache.clear()
            if self.voiceprint_cache:
                await self.voiceprint_cache.clear()
            logger.info("Caches cleared")
            
            # Clean up providers
            await self._cleanup_providers()
            
            self._running = False
            logger.info("AudioAdapter stopped")
            
        except Exception as e:
            logger.error("Failed to stop AudioAdapter", error=str(e), exc_info=e)
            raise AudioAdapterError(f"Failed to stop AudioAdapter: {e}") from e

    async def create_session(self, device_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new audio session.
        
        Args:
            device_id: Unique device identifier
            metadata: Optional session metadata
            
        Returns:
            Session ID (UUID string)
            
        Raises:
            AudioAdapterError: If session manager not initialized or creation fails
        """
        if not self.session_manager:
            raise AudioAdapterError("Session manager not initialized. Call start() first.")
        
        try:
            session_id = await self.session_manager.create_session(device_id)
            logger.info("Session created via adapter", session_id=session_id, device_id=device_id)
            return session_id
        except Exception as e:
            logger.error("Failed to create session", error=str(e), device_id=device_id)
            raise AudioAdapterError(f"Failed to create session: {e}") from e

    async def close_session(self, session_id: str) -> None:
        """
        Close an existing audio session.
        
        Args:
            session_id: Session ID to close
            
        Raises:
            AudioAdapterError: If session manager not initialized or close fails
        """
        if not self.session_manager:
            raise AudioAdapterError("Session manager not initialized. Call start() first.")
        
        try:
            await self.session_manager.close_session(session_id)
            logger.info("Session closed via adapter", session_id=session_id)
        except Exception as e:
            logger.error("Failed to close session", error=str(e), session_id=session_id)
            raise AudioAdapterError(f"Failed to close session: {e}") from e

    async def process_audio(self, session_id: str, audio_frame: AudioFrame) -> Optional[SpeechSegment]:
        """
        Process incoming audio frame.
        
        This method delegates to the session manager's handle_audio_frame,
        which buffers audio, runs VAD, and spawns ASR processing tasks.
        
        Args:
            session_id: Session ID
            audio_frame: Audio frame to process
            
        Returns:
            Optional[SpeechSegment] if speech detected and transcribed, None otherwise.
            Note: Transcription happens asynchronously, so this typically returns None.
            Subscribe to on_transcription callback for results.
            
        Raises:
            AudioAdapterError: If session manager not initialized or processing fails
        """
        if not self.session_manager:
            raise AudioAdapterError("Session manager not initialized. Call start() first.")
        
        try:
            # Convert AudioFrame to bytes and metadata dict
            audio_bytes = audio_frame.data
            metadata = {
                "sample_rate": audio_frame.sample_rate,
                "channels": audio_frame.channels,
                "timestamp": audio_frame.timestamp,
            }
            
            # Delegate to session manager
            await self.session_manager.handle_audio_frame(session_id, audio_bytes, metadata)
            
            # Note: Transcription happens asynchronously in background tasks.
            # Callers should subscribe to on_transcription callback for results.
            return None
            
        except Exception as e:
            logger.error("Failed to process audio", error=str(e), session_id=session_id)
            raise AudioAdapterError(f"Failed to process audio: {e}") from e

    async def synthesize_speech(self, session_id: str, text: str):
        """
        Synthesize speech from text (streaming generator).
        
        Args:
            session_id: Session ID (for context)
            text: Text to synthesize
            
        Yields:
            Audio bytes (PCM format, sample_rate from config)
            
        Raises:
            AudioAdapterError: If TTS provider not initialized or synthesis fails
        """
        if not self.tts:
            raise AudioAdapterError("TTS provider not initialized. Call start() first.")
        
        try:
            voice = self.config.tts_voice
            logger.info("Starting speech synthesis", session_id=session_id, text_length=len(text))
            
            # Stream audio chunks from TTS
            async for chunk in self.tts.synthesize(text, voice=voice):
                yield chunk
            
            logger.info("Speech synthesis completed", session_id=session_id)
            
        except Exception as e:
            logger.error("Failed to synthesize speech", error=str(e), session_id=session_id, text=text[:50])
            raise AudioAdapterError(f"Failed to synthesize speech: {e}") from e

    async def verify_speaker(self, session_id: str, audio_data: bytes) -> Optional[str]:
        """Verify speaker identity using voiceprint (T061).
        
        This method implements voiceprint-based speaker verification:
        - Extracts voiceprint features from audio
        - Compares with registered voiceprints for the device
        - Returns speaker_id if similarity >= threshold
        - Updates verification statistics (count, last_verified_at)
        
        Verification Flow:
            1. Get session and device_id
            2. Call 3DSpeaker service for verification
            3. Check similarity score against threshold
            4. Return speaker_id if verified, None otherwise
            5. Update verification statistics in storage
        
        Args:
            session_id: Session ID for context
            audio_data: Audio data for voiceprint extraction (PCM 16kHz mono, 2-3s)
            
        Returns:
            speaker_id if verified (similarity >= threshold), None otherwise
            
        Raises:
            AudioAdapterError: If verification process fails
            
        Example:
            ```python
            # During ASR callback
            speaker_id = await adapter.verify_speaker(
                session_id="session-001",
                audio_data=audio_segment
            )
            
            if speaker_id:
                print(f"Verified as: {speaker_id}")
                # Inject speaker_id into TurnState.metadata
            else:
                print("Speaker not verified - anonymous mode")
            ```
            
        Performance Target (from tasks.md T061):
            - Average response time: < 300ms
            - Accuracy: > 95% (positive verification)
            - Rejection rate: < 5% (false positives)
        """
        if not self.voiceprint_client:
            logger.debug("Voiceprint verification not enabled", session_id=session_id)
            return None
        
        try:
            # Get session and device_id
            session = self.session_manager.get_session(session_id) if self.session_manager else None
            if not session:
                raise AudioAdapterError(f"Session {session_id} not found")
            
            device_id = session.device_id
            
            logger.debug(
                "Starting speaker verification",
                session_id=session_id,
                device_id=device_id,
                audio_size=len(audio_data),
            )
            
            # Call 3DSpeaker service for verification
            start_time = time.time()
            result = await self.voiceprint_client.verify(
                device_id=device_id,
                audio_data=audio_data,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Check verification result
            verified = result.get("verified", False)
            speaker_id = result.get("speaker_id")
            similarity = result.get("similarity", 0.0)
            threshold = result.get("threshold", self.voiceprint_client.similarity_threshold)
            
            if verified and speaker_id:
                # Speaker verified successfully
                logger.info(
                    "Speaker verified",
                    session_id=session_id,
                    speaker_id=speaker_id,
                    similarity=f"{similarity:.3f}",
                    threshold=threshold,
                    elapsed_ms=f"{elapsed_ms:.1f}",
                )
                
                # Update verification statistics in storage
                if self.voiceprint_storage:
                    try:
                        await self.voiceprint_storage.update_verification_stats(
                            device_id=device_id,
                            speaker_id=speaker_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to update verification stats",
                            error=str(e),
                            speaker_id=speaker_id,
                        )
                
                # Update session metadata
                if session:
                    session.metadata["speaker_id"] = speaker_id
                    session.metadata["speaker_verified_at"] = time.time()
                    session.metadata["verification_similarity"] = similarity
                
                return speaker_id
            else:
                # Verification failed
                logger.info(
                    "Speaker verification failed",
                    session_id=session_id,
                    similarity=f"{similarity:.3f}",
                    threshold=threshold,
                    elapsed_ms=f"{elapsed_ms:.1f}",
                )
                return None
                
        except Exception as e:
            logger.error(
                "Failed to verify speaker",
                error=str(e),
                session_id=session_id,
                exc_info=e,
            )
            # Don't raise - allow fallback to anonymous mode
            return None

    async def register_voiceprint(
        self,
        device_id: str,
        display_name: str,
        audio_samples: list[bytes],
    ) -> str:
        """Register a new voiceprint with guided registration flow.
        
        This method implements T058-T059 from tasks.md:
        - Accepts 3-5 audio samples (2-3 seconds each)
        - Calls 3DSpeaker service for registration
        - Stores encrypted voiceprint data locally
        - Returns speaker_id for use in verification
        
        Registration Flow:
            1. Validate audio samples (3-5 samples, correct format)
            2. Check voiceprint limit (max 5 per device)
            3. Register with 3DSpeaker service
            4. Store encrypted voiceprint metadata locally
            5. Return speaker_id
        
        Args:
            device_id: Device ID (e.g., "xiaozhi-001")
            display_name: User display name (e.g., "Alice", "Bob")
            audio_samples: 3-5 audio samples (PCM 16kHz mono, 2-3s each)
            
        Returns:
            speaker_id: Unique speaker identifier for verification
            
        Raises:
            AudioAdapterError: If registration fails or service not enabled
            
        Example:
            ```python
            # Collect 3 audio samples
            samples = [sample1, sample2, sample3]
            
            # Register voiceprint
            speaker_id = await adapter.register_voiceprint(
                device_id="xiaozhi-001",
                display_name="Alice",
                audio_samples=samples
            )
            
            print(f"Registered with speaker_id: {speaker_id}")
            ```
        """
        if not self.voiceprint_client:
            raise AudioAdapterError(
                "Voiceprint service not enabled. "
                "Set voiceprint_enabled=True and voiceprint_url in config."
            )
        
        if not audio_samples or len(audio_samples) < 3:
            raise AudioAdapterError(
                "At least 3 audio samples required for registration. "
                "Please provide 3-5 samples of 2-3 seconds each."
            )
        
        if len(audio_samples) > 5:
            raise AudioAdapterError(
                "Maximum 5 audio samples allowed. "
                "Please provide 3-5 samples only."
            )
        
        # Check voiceprint limit (T072)
        if self.voiceprint_storage:
            can_add, current_count = self.check_voiceprint_limit(device_id, max_voiceprints=5)
            if not can_add:
                raise AudioAdapterError(
                    f"Voiceprint limit reached: {current_count}/5 users already registered. "
                    "Please delete an existing voiceprint before adding a new one."
                )
            
            logger.info(
                "Voiceprint limit check passed",
                device_id=device_id,
                current_count=current_count,
                max_allowed=5,
            )
        
        try:
            logger.info(
                "Starting voiceprint registration",
                device_id=device_id,
                display_name=display_name,
                sample_count=len(audio_samples),
            )
            
            # Register with voiceprint service
            result = await self.voiceprint_client.register(
                device_id=device_id,
                display_name=display_name,
                audio_samples=audio_samples,
            )
            
            speaker_id = result.get("speaker_id")
            if not speaker_id:
                raise AudioAdapterError("Registration failed: no speaker_id returned")
            
            # Store encrypted voiceprint metadata locally (T060)
            if self.voiceprint_storage:
                await self.voiceprint_storage.save_voiceprint(
                    device_id=device_id,
                    speaker_id=speaker_id,
                    display_name=display_name,
                    metadata={
                        "created_at": result.get("created_at"),
                        "sample_count": len(audio_samples),
                    }
                )
                logger.info(
                    "Voiceprint metadata saved locally",
                    speaker_id=speaker_id,
                    device_id=device_id,
                )
            
            logger.info(
                "Voiceprint registered successfully",
                device_id=device_id,
                speaker_id=speaker_id,
                display_name=display_name,
            )
            
            return speaker_id
            
        except Exception as e:
            logger.error(
                "Failed to register voiceprint",
                error=str(e),
                device_id=device_id,
                display_name=display_name,
            )
            raise AudioAdapterError(f"Failed to register voiceprint: {e}") from e

    async def stream_synthesis(
        self,
        session_id: str,
        text: str,
        voice: Optional[str] = None,
        chunk_size: int = 4096,
    ) -> None:
        """Stream synthesized speech to a session via WebSocket.
        
        This method streams TTS output in real-time to minimize first-packet latency.
        Audio chunks are sent as binary WebSocket messages as soon as they're generated.
        
        Args:
            session_id: Target session ID
            text: Text to synthesize and stream
            voice: Optional voice ID (defaults to config.tts_voice)
            chunk_size: Audio chunk size in bytes (default: 4096 for ~256ms @ 16kHz)
            
        Raises:
            AudioAdapterError: If TTS not initialized or session not found
            
        Example:
            >>> await adapter.stream_synthesis(
            ...     session_id="abc123",
            ...     text="Hello, this is a streaming response",
            ...     chunk_size=2048  # Smaller chunks = lower latency
            ... )
        """
        if not self.tts:
            raise AudioAdapterError("TTS provider not initialized. Call start() first.")
        
        if not self.websocket_server:
            raise AudioAdapterError("WebSocket server not initialized. Call start() first.")
        
        # Find connection handler for session
        handler = None
        for conn in self.websocket_server.active_connections:
            if conn.session_id == session_id:
                handler = conn
                break
        
        if not handler:
            raise AudioAdapterError(f"No active connection found for session {session_id}")
        
        try:
            voice = voice or self.config.tts_voice
            chunk_count = 0
            total_bytes = 0
            start_time = time.time()
            
            logger.info(
                "Starting streaming TTS",
                session_id=session_id,
                text_length=len(text),
                chunk_size=chunk_size,
            )
            
            # Stream audio chunks as they're generated
            buffer = bytearray()
            async for audio_chunk in self.tts.synthesize(text, voice=voice):
                buffer.extend(audio_chunk)
                
                # Send buffered data when we have enough for a chunk
                while len(buffer) >= chunk_size:
                    chunk_to_send = bytes(buffer[:chunk_size])
                    buffer = buffer[chunk_size:]
                    
                    # Serialize and send via WebSocket
                    await self._send_audio_chunk(
                        handler=handler,
                        audio_data=chunk_to_send,
                        session_id=session_id,
                        is_end=False,
                    )
                    
                    chunk_count += 1
                    total_bytes += len(chunk_to_send)
                    
                    # Log first packet latency
                    if chunk_count == 1:
                        first_packet_ms = (time.time() - start_time) * 1000
                        logger.info(
                            "First TTS packet sent",
                            latency_ms=f"{first_packet_ms:.1f}",
                            session_id=session_id,
                        )
            
            # Send remaining buffer (final chunk)
            if buffer:
                await self._send_audio_chunk(
                    handler=handler,
                    audio_data=bytes(buffer),
                    session_id=session_id,
                    is_end=True,
                )
                chunk_count += 1
                total_bytes += len(buffer)
            
            total_time_ms = (time.time() - start_time) * 1000
            logger.info(
                "Streaming TTS completed",
                session_id=session_id,
                chunks=chunk_count,
                total_bytes=total_bytes,
                duration_ms=f"{total_time_ms:.1f}",
            )
            
        except Exception as e:
            logger.error(
                "Failed to stream synthesis",
                error=str(e),
                session_id=session_id,
                text=text[:50],
            )
            raise AudioAdapterError(f"Failed to stream synthesis: {e}") from e

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    async def _send_audio_chunk(
        self,
        handler: Any,  # ConnectionHandler
        audio_data: bytes,
        session_id: str,
        is_end: bool = False,
    ) -> None:
        """Send audio chunk via WebSocket using binary protocol.
        
        Args:
            handler: ConnectionHandler instance
            audio_data: Audio bytes to send
            session_id: Session ID for metadata
            is_end: Whether this is the final chunk
        """
        from loom.adapters.audio.websocket.protocol import WebSocketProtocol
        
        binary_message = WebSocketProtocol.serialize_audio_message(
            session_id=session_id,
            audio_data=audio_data,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            is_end=is_end,
        )
        
        await handler.websocket.send(binary_message)

    async def _initialize_providers(self) -> None:
        """Initialize audio providers (VAD, ASR, TTS, voiceprint)."""
        logger.info("Initializing audio providers...")
        
        # VAD (Silero)
        try:
            self.vad = SileroVAD(
                threshold=self.config.vad_threshold,
                threshold_low=0.2,  # Fixed low threshold for hysteresis
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                sample_rate=self.config.sample_rate,
            )
            logger.info("VAD initialized", provider="SileroVAD")
        except Exception as e:
            logger.warning("Failed to initialize VAD", error=str(e))
            self.vad = None
        
        # ASR (FunASR)
        try:
            # Note: FunASR requires model_dir, not model_name
            # Default to a standard path - users should download models separately
            model_dir = f"models/{self.config.asr_model}"
            self.asr = FunASR(
                model_dir=model_dir,
                output_dir="temp/audio",
                delete_temp_files=True,
            )
            logger.info("ASR initialized", provider="FunASR", model_dir=model_dir)
        except Exception as e:
            logger.warning("Failed to initialize ASR", error=str(e))
            self.asr = None
        
        # TTS (EdgeTTS)
        try:
            self.tts = EdgeTTS()
            logger.info("TTS initialized", provider="EdgeTTS")
        except Exception as e:
            logger.warning("Failed to initialize TTS", error=str(e))
            self.tts = None
        
        # Voiceprint (optional)
        if self.config.voiceprint_enabled:
            try:
                self.voiceprint_client = ThreeDSpeakerClient(
                    base_url=self.config.voiceprint_url or "http://localhost:8000"
                )
                # VoiceprintStorage expects encryption_key - generate one or use from config
                import secrets
                encryption_key = secrets.token_bytes(32)
                self.voiceprint_storage = VoiceprintStorage(encryption_key=encryption_key)
                logger.info("Voiceprint services initialized")
            except Exception as e:
                logger.warning("Failed to initialize voiceprint services", error=str(e))
                self.voiceprint_client = None
                self.voiceprint_storage = None

    async def _cleanup_providers(self) -> None:
        """Clean up provider resources."""
        logger.info("Cleaning up audio providers...")
        
        # Close voiceprint client
        if self.voiceprint_client:
            try:
                await self.voiceprint_client.close()
            except Exception as e:
                logger.warning("Failed to close voiceprint client", error=str(e))
        
        # No explicit cleanup needed for VAD/ASR/TTS (models stay in memory)
        self.vad = None
        self.asr = None
        self.tts = None
        self.voiceprint_client = None
        self.voiceprint_storage = None

    def get_session(self, session_id: str) -> Optional[AudioSession]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            AudioSession if found, None otherwise
        """
        if not self.session_manager:
            return None
        return self.session_manager.get_session(session_id)

    def get_active_session_count(self) -> int:
        """
        Get count of active sessions.
        
        Returns:
            Number of active sessions
        """
        if not self.session_manager:
            return 0
        return self.session_manager.get_active_session_count()
    
    def check_tool_permission(
        self,
        tool_name: str,
        speaker_id: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Check permission for tool execution with speaker context (T064).
        
        This method integrates with AudioPermissionManager to enforce
        voiceprint-based access control:
        - Critical operations: Only owner + requires two-factor confirmation
        - Medium operations: Requires voiceprint verification (owner or guest)
        - Low operations: Public access (no verification)
        
        Args:
            tool_name: Tool or operation name
            speaker_id: Speaker ID from session.metadata["speaker_id"]
            arguments: Tool arguments (optional context)
            
        Returns:
            Permission action string: "allow", "deny", or "ask"
            
        Example:
            # In agent tool execution pipeline
            session = adapter.get_session(session_id)
            speaker_id = session.metadata.get("speaker_id") if session else None
            
            action = adapter.check_tool_permission(
                tool_name="unlock_door",
                speaker_id=speaker_id,
                arguments={"location": "front_door"}
            )
            
            if action == "deny":
                return "拒绝访问：此操作需要主人身份验证"
            elif action == "ask":
                # Trigger two-factor confirmation (T067-T069)
                return "敏感操作，请确认：是的/确认"
        """
        if not self.permission_manager:
            logger.warning("Permission manager not initialized - defaulting to deny")
            return "deny"
        
        action = self.permission_manager.check_permission(
            tool_name=tool_name,
            speaker_id=speaker_id,
            arguments=arguments,
        )
        
        return action.value  # "allow", "deny", or "ask"
    
    def request_confirmation(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        original_command: str,
        speaker_id: Optional[str] = None,
    ) -> str:
        """Request two-factor confirmation for critical operation (T068).
        
        This method should be called when check_tool_permission() returns "ask".
        It creates a pending confirmation and returns a prompt for the user.
        
        Args:
            session_id: Session ID
            tool_name: Tool/operation name requiring confirmation
            arguments: Tool arguments
            original_command: Original user command text
            speaker_id: Speaker who initiated the operation
            
        Returns:
            Confirmation prompt message for TTS
            
        Example:
            action = adapter.check_tool_permission("unlock_door", speaker_id)
            
            if action == "ask":
                prompt = adapter.request_confirmation(
                    session_id=session_id,
                    tool_name="unlock_door",
                    arguments={"location": "front_door"},
                    original_command="打开门锁",
                    speaker_id=speaker_id
                )
                # Returns: "敏感操作需要确认：打开门锁。请说'确认'或重复指令'打开门锁'"
        """
        if not self.confirmation_manager:
            logger.warning("Confirmation manager not initialized")
            return "确认系统未初始化，操作已取消"
        
        return self.confirmation_manager.request_confirmation(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            original_command=original_command,
            speaker_id=speaker_id,
        )
    
    def check_confirmation(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """Check if user input confirms pending operation (T069).
        
        This method should be called for each new user input when there's
        a pending confirmation. It recognizes confirmation keywords or
        command repetition.
        
        Args:
            session_id: Session ID
            user_input: User's ASR transcription text
            
        Returns:
            Dict with keys:
            - state: "confirmed" | "rejected" | "timeout" | "cancelled" | "pending"
            - message: User-facing message
            - tool_name: Tool name (if confirmed)
            - arguments: Tool arguments (if confirmed)
            
        Example:
            result = adapter.check_confirmation(session_id, "确认")
            
            if result["state"] == "confirmed":
                # Execute the tool
                tool_name = result["tool_name"]
                arguments = result["arguments"]
                execute_tool(tool_name, arguments)
                return result["message"]  # "确认成功，正在执行操作"
            elif result["state"] == "timeout":
                return result["message"]  # "确认超时，已取消操作"
        """
        if not self.confirmation_manager:
            logger.warning("Confirmation manager not initialized")
            return {
                "state": "rejected",
                "message": "确认系统未初始化",
                "tool_name": None,
                "arguments": None,
            }
        
        result = self.confirmation_manager.check_confirmation(
            session_id=session_id,
            user_input=user_input,
        )
        
        return {
            "state": result.state.value,
            "message": result.message,
            "tool_name": result.pending.tool_name if result.pending else None,
            "arguments": result.pending.arguments if result.pending else None,
        }
    
    def has_pending_confirmation(self, session_id: str) -> bool:
        """Check if session has pending confirmation.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if pending confirmation exists and not timed out
        """
        if not self.confirmation_manager:
            return False
        return self.confirmation_manager.has_pending_confirmation(session_id)
    
    def cancel_confirmation(self, session_id: str) -> bool:
        """Cancel pending confirmation for session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if confirmation was cancelled, False if no pending confirmation
        """
        if not self.confirmation_manager:
            return False
        return self.confirmation_manager.cancel_confirmation(session_id)
    
    # ========== Multi-User Management (T070-T072) ==========
    
    def list_voiceprints(self, device_id: str) -> list[Dict[str, Any]]:
        """List all registered voiceprints for a device (T070).
        
        Returns metadata for all voiceprints including:
        - speaker_id: Unique speaker identifier
        - display_name: User display name
        - created_at: Registration timestamp
        - verification_count: Number of successful verifications
        - last_verified_at: Last verification timestamp
        
        Args:
            device_id: Device ID
            
        Returns:
            List of voiceprint metadata dicts, sorted by verification_count (descending)
            
        Raises:
            AudioAdapterError: If voiceprint storage not initialized
            
        Example:
            ```python
            voiceprints = adapter.list_voiceprints("xiaozhi-001")
            
            for vp in voiceprints:
                print(f"{vp['display_name']}: {vp['verification_count']} verifications")
            
            # Output:
            # Alice: 42 verifications
            # Bob: 15 verifications
            ```
        """
        if not self.voiceprint_storage:
            raise AudioAdapterError("Voiceprint storage not initialized. Call start() first.")
        
        voiceprints = self.voiceprint_storage.list_voiceprints(device_id)
        
        # Sort by verification count (most active users first)
        voiceprints_sorted = sorted(
            voiceprints,
            key=lambda vp: vp.get("verification_count", 0),
            reverse=True,
        )
        
        logger.info(
            "Listed voiceprints",
            device_id=device_id,
            count=len(voiceprints_sorted),
        )
        
        return voiceprints_sorted
    
    async def delete_voiceprint(
        self,
        device_id: str,
        speaker_id: str,
        remove_from_permissions: bool = True,
    ) -> bool:
        """Delete a voiceprint and optionally remove from permission config (T071).
        
        This method:
        1. Deletes encrypted voiceprint files (.enc and .json)
        2. Optionally removes speaker from permission manager (owner/guest lists)
        3. Logs the deletion for audit trail
        
        Args:
            device_id: Device ID
            speaker_id: Speaker ID to delete
            remove_from_permissions: Also remove from owner/guest lists (default: True)
            
        Returns:
            True if voiceprint was deleted, False if not found
            
        Raises:
            AudioAdapterError: If voiceprint storage not initialized
            
        Example:
            ```python
            # Delete voiceprint and remove from permissions
            deleted = await adapter.delete_voiceprint(
                device_id="xiaozhi-001",
                speaker_id="speaker_002"
            )
            
            if deleted:
                print("Voiceprint deleted successfully")
            ```
        """
        if not self.voiceprint_storage:
            raise AudioAdapterError("Voiceprint storage not initialized. Call start() first.")
        
        # Get metadata before deletion for logging
        metadata = self.voiceprint_storage.load_metadata(device_id, speaker_id)
        display_name = metadata.get("display_name", "Unknown") if metadata else "Unknown"
        
        # Delete voiceprint files
        deleted = self.voiceprint_storage.delete(device_id, speaker_id)
        
        if not deleted:
            logger.warning(
                "Voiceprint not found for deletion",
                device_id=device_id,
                speaker_id=speaker_id,
            )
            return False
        
        logger.info(
            "Voiceprint deleted",
            device_id=device_id,
            speaker_id=speaker_id,
            display_name=display_name,
        )
        
        # Remove from permission manager if requested
        if remove_from_permissions and self.permission_manager:
            removed = self.permission_manager.remove_speaker(speaker_id)
            if removed:
                logger.info(
                    "Speaker removed from permissions",
                    speaker_id=speaker_id,
                )
        
        return True
    
    def get_voiceprint_count(self, device_id: str) -> int:
        """Get number of registered voiceprints for a device (T072).
        
        Args:
            device_id: Device ID
            
        Returns:
            Number of registered voiceprints
            
        Raises:
            AudioAdapterError: If voiceprint storage not initialized
        """
        if not self.voiceprint_storage:
            raise AudioAdapterError("Voiceprint storage not initialized. Call start() first.")
        
        voiceprints = self.voiceprint_storage.list_voiceprints(device_id)
        return len(voiceprints)
    
    def check_voiceprint_limit(
        self,
        device_id: str,
        max_voiceprints: int = 5,
    ) -> tuple[bool, int]:
        """Check if voiceprint limit has been reached (T072).
        
        Default limit: 5 voiceprints per device
        
        Args:
            device_id: Device ID
            max_voiceprints: Maximum allowed voiceprints (default: 5)
            
        Returns:
            Tuple of (can_add, current_count)
            - can_add: True if can add more voiceprints
            - current_count: Current number of voiceprints
            
        Raises:
            AudioAdapterError: If voiceprint storage not initialized
            
        Example:
            ```python
            can_add, count = adapter.check_voiceprint_limit("xiaozhi-001")
            
            if not can_add:
                print(f"Limit reached: {count}/5 voiceprints registered")
            else:
                print(f"Can add more: {count}/5 voiceprints")
            ```
        """
        if not self.voiceprint_storage:
            raise AudioAdapterError("Voiceprint storage not initialized. Call start() first.")
        
        current_count = self.get_voiceprint_count(device_id)
        can_add = current_count < max_voiceprints
        
        logger.debug(
            "Voiceprint limit check",
            device_id=device_id,
            current_count=current_count,
            max_voiceprints=max_voiceprints,
            can_add=can_add,
        )
        
        return can_add, current_count
    
    def get_voiceprint_stats(self, device_id: str) -> Dict[str, Any]:
        """Get voiceprint statistics for a device (T070).
        
        Returns aggregate statistics including:
        - total_voiceprints: Total number of registered voiceprints
        - total_verifications: Sum of all verification counts
        - most_active_user: Display name of most verified user
        - least_active_user: Display name of least verified user
        
        Args:
            device_id: Device ID
            
        Returns:
            Dict with statistics
            
        Raises:
            AudioAdapterError: If voiceprint storage not initialized
        """
        if not self.voiceprint_storage:
            raise AudioAdapterError("Voiceprint storage not initialized. Call start() first.")
        
        voiceprints = self.list_voiceprints(device_id)
        
        if not voiceprints:
            return {
                "total_voiceprints": 0,
                "total_verifications": 0,
                "most_active_user": None,
                "least_active_user": None,
            }
        
        total_verifications = sum(vp.get("verification_count", 0) for vp in voiceprints)
        most_active = max(voiceprints, key=lambda vp: vp.get("verification_count", 0))
        least_active = min(voiceprints, key=lambda vp: vp.get("verification_count", 0))
        
        stats = {
            "total_voiceprints": len(voiceprints),
            "total_verifications": total_verifications,
            "most_active_user": most_active.get("display_name"),
            "most_active_count": most_active.get("verification_count", 0),
            "least_active_user": least_active.get("display_name"),
            "least_active_count": least_active.get("verification_count", 0),
        }
        
        logger.info(
            "Voiceprint stats",
            device_id=device_id,
            **stats,
        )
        
        return stats

    # ============================
    # T079-T081: Multi-turn conversation context management
    # ============================

    def get_conversation_context(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
        compress: bool = True,
    ) -> str:
        """Get assembled conversation context for Agent injection.
        
        Retrieves conversation history from session and assembles into
        context string with compression if needed (> 10 turns).
        
        Implements T079 (context assembly) and T080 (compression strategy).
        
        Args:
            session_id: Target session ID
            system_prompt: System instructions to prepend (optional)
            compress: Whether to apply compression (default: True)
        
        Returns:
            Assembled context string ready for LLM injection
        
        Raises:
            AudioAdapterError: If session not found or context manager not initialized
        
        Example:
            ```python
            context = adapter.get_conversation_context(
                session_id="abc123",
                system_prompt="You are a helpful voice assistant",
                compress=True
            )
            # Pass to Loom Agent: agent.run(user_input, context=context)
            ```
        """
        if not self.session_manager:
            raise AudioAdapterError("Session manager not initialized")
        
        if not self.context_manager:
            raise AudioAdapterError("Context manager not initialized")
        
        # Get session to access conversation history
        session = self.session_manager.get_session(session_id)
        if not session:
            raise AudioAdapterError(f"Session {session_id} not found")
        
        # Assemble context with compression
        context = self.context_manager.assemble_context(
            conversation_history=session.conversation_history,
            speaker_id=session.speaker_id,
            system_prompt=system_prompt,
            compress=compress,
        )
        
        logger.debug(
            "Retrieved conversation context",
            session_id=session_id,
            turn_count=session.turn_count,
            context_length=len(context),
            compressed=compress and self.context_manager.should_compress(session.turn_count),
        )
        
        return context

    # ============================
    # T089-T092: Performance monitoring
    # ============================

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.
        
        Returns statistics for:
        - Cache hit rates and sizes
        - Concurrency limiter usage
        - Cleanup manager status
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "running": self._running,
            "caches": {},
            "concurrency": {},
            "cleanup": {},
        }
        
        if self.permission_cache:
            stats["caches"]["permission"] = self.permission_cache.get_stats()
        if self.context_cache:
            stats["caches"]["context"] = self.context_cache.get_stats()
        if self.voiceprint_cache:
            stats["caches"]["voiceprint"] = self.voiceprint_cache.get_stats()
        
        if self.concurrency_limiter:
            stats["concurrency"] = self.concurrency_limiter.get_stats()
        
        if self.cleanup_manager:
            stats["cleanup"] = self.cleanup_manager.get_stats()
        
        return stats
    
    # ============================
    # T097-T100: Resilience monitoring
    # ============================
    
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get comprehensive resilience statistics.
        
        Returns statistics for:
        - Retry policy (attempts, successes, delays)
        - Circuit breakers (states, transitions, error rates)
        - Fallback manager (fallback counts per service)
        - Recovery manager (recent history)
        
        Returns:
            Dictionary with resilience metrics
        """
        stats = {
            "retry": {},
            "circuit_breakers": {},
            "fallback": {},
            "recovery": {},
        }
        
        if self.retry_policy:
            stats["retry"] = self.retry_policy.get_stats()
        
        if self.asr_circuit_breaker:
            stats["circuit_breakers"]["asr"] = self.asr_circuit_breaker.get_stats()
        if self.voiceprint_circuit_breaker:
            stats["circuit_breakers"]["voiceprint"] = self.voiceprint_circuit_breaker.get_stats()
        
        if self.fallback_manager:
            stats["fallback"] = self.fallback_manager.get_stats()
        
        if self.recovery_manager:
            stats["recovery"] = {
                "recent_attempts": self.recovery_manager.get_history(limit=10)
            }
        
        return stats

    @property
    def is_running(self) -> bool:
        """Check if adapter is running."""
        return self._running


# Factory function for convenience
def create_audio_adapter(
    host: str = "0.0.0.0",
    port: int = 8765,
    vad_threshold: float = 0.5,
    sample_rate: int = 16000,
    **kwargs: Any,
) -> AudioAdapter:
    """
    Factory function to create AudioAdapter with common settings.
    
    Args:
        host: WebSocket server host
        port: WebSocket server port
        vad_threshold: VAD detection threshold (0.0-1.0)
        sample_rate: Audio sample rate (Hz)
        **kwargs: Additional AudioAdapterConfig fields
        
    Returns:
        AudioAdapter instance (not started)
        
    Example:
        ```python
        adapter = create_audio_adapter(port=9000, vad_threshold=0.6)
        await adapter.start()
        ```
    """
    config = AudioAdapterConfig(
        host=host,
        port=port,
        vad_threshold=vad_threshold,
        sample_rate=sample_rate,
        **kwargs,
    )
    return AudioAdapter(config)
