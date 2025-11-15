"""
Hello Voice - Minimal Voice Companion Example

This example demonstrates the complete voice interaction loop:
1. Audio input → VAD detects speech
2. ASR recognizes text
3. Loom Agent (OpenAI GPT-4) processes the request
4. TTS synthesizes response
5. Audio output via WebSocket

Prerequisites:
    pip install loom-agent[audio]
    export OPENAI_API_KEY="sk-..."  # Required for GPT-4

Usage:
    # Terminal 1: Start voice companion
    python examples/voice_companion/hello_voice.py

    # Terminal 2: Connect with WebSocket client
    # See: examples/voice_companion/test_client.py
    # Or use any WebSocket client to send audio to ws://localhost:8765/audio

Expected Behavior:
    - Speak: "今天天气怎么样" (What's the weather today?)
    - Response: GPT-4 generated answer via TTS
    - End-to-end latency: < 1.5 seconds (target)
"""

import asyncio
import os
import sys
from typing import Optional

# Loom Agent imports
from loom import agent
from loom.core.events import AgentEvent, AgentEventType

# Audio Adapter imports
from loom.adapters.audio import AudioAdapter, AudioAdapterConfig
from loom.adapters.audio.models import AudioSession
from loom.interfaces.audio_adapter import TranscriptionResult
from loom.core.structured_logger import get_logger

logger = get_logger(__name__)


class VoiceCompanion:
    """Minimal voice companion integrating AudioAdapter with Loom Agent."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize voice companion.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name (default: gpt-4)
        """
        self.api_key = api_key
        self.model = model
        self.adapter: Optional[AudioAdapter] = None
        self.agent = None
        
        # Session tracking
        self.active_sessions = {}
        
    async def setup(self, host: str = "0.0.0.0", port: int = 8765):
        """Set up audio adapter and Loom agent.
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
        """
        # Create Loom Agent with OpenAI GPT-4
        logger.info("Creating Loom Agent", model=self.model)
        self.agent = agent(
            provider="openai",
            model=self.model,
            api_key=self.api_key,
            temperature=0.7,
            max_iterations=10,
            system_instructions=(
                "你是小智，一个友好的语音助手。请用简洁、自然的语言回答问题。"
                "保持回答在 1-2 句话以内，适合语音播报。"
            )
        )
        
        # Create AudioAdapter with optimized config
        config = AudioAdapterConfig(
            host=host,
            port=port,
            # VAD settings
            vad_threshold=0.5,
            min_speech_duration_ms=300,
            min_silence_duration_ms=700,
            # Audio settings
            sample_rate=16000,
            channels=1,
            # TTS voice
            tts_voice="zh-CN-XiaoxiaoNeural",  # Female Mandarin voice
            # Connection limits
            max_connections=5,
            # Disable voiceprint for simplicity
            voiceprint_enabled=False,
        )
        
        self.adapter = AudioAdapter(
            config=config,
            on_session_start=self._on_session_start,
            on_session_end=self._on_session_end,
        )
        
        # Register transcription callback
        self.adapter._on_transcription = self._on_transcription
        
        logger.info("Voice companion setup complete")
    
    def _on_session_start(self, session: AudioSession) -> None:
        """Called when a new audio session starts."""
        logger.info("Session started", 
                   session_id=session.session_id,
                   device_id=session.device_id,
                   state=session.state.value)
        
        self.active_sessions[session.session_id] = {
            "device_id": session.device_id,
            "start_time": session.created_at,
        }
        
        print(f"\n🎙️  New session: {session.session_id}")
        print(f"   Device: {session.device_id}")
    
    def _on_session_end(self, session_id: str) -> None:
        """Called when an audio session ends."""
        logger.info("Session ended", session_id=session_id)
        
        if session_id in self.active_sessions:
            session_info = self.active_sessions.pop(session_id)
            print(f"\n⏹️  Session closed: {session_id}")
            print(f"   Device: {session_info['device_id']}")
    
    async def _on_transcription(self, session_id: str, result: TranscriptionResult) -> None:
        """Called when speech is transcribed - triggers agent processing.
        
        This is the core integration point between AudioAdapter and Loom Agent.
        
        Args:
            session_id: Audio session ID
            result: ASR transcription result
        """
        logger.info("Transcription received",
                   session_id=session_id,
                   text=result.text,
                   confidence=result.confidence,
                   latency_ms=result.latency_ms)
        
        print(f"\n📝 User [{session_id[:8]}]: {result.text}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   ASR Latency: {result.latency_ms}ms")
        
        if not result.text.strip():
            logger.warning("Empty transcription, skipping agent call")
            return
        
        # Call Loom Agent
        try:
            print(f"🤔 Processing with {self.model}...")
            
            # Run agent (streaming for better observability)
            response_text = ""
            async for event in self.agent.stream(result.text):
                if event.type == AgentEventType.AGENT_FINISH:
                    response_text = event.final_response
                    break
                elif event.type == AgentEventType.ERROR:
                    logger.error("Agent error", error=str(event.error))
                    response_text = "抱歉，处理请求时出现了错误。"
                    break
            
            if not response_text:
                response_text = "抱歉，我没有理解你的问题。"
            
            logger.info("Agent response generated", 
                       response_length=len(response_text))
            
            print(f"🤖 Assistant: {response_text}")
            
            # Stream TTS response
            print(f"🔊 Synthesizing speech...")
            await self.adapter.stream_synthesis(
                session_id=session_id,
                text=response_text,
                chunk_size=4096,  # ~256ms @ 16kHz (balanced)
            )
            
            logger.info("TTS streaming completed", session_id=session_id)
            print(f"✅ Response sent\n")
            
        except Exception as e:
            logger.error("Failed to process transcription",
                        error=str(e),
                        session_id=session_id,
                        exc_info=e)
            
            # Send error message via TTS
            error_message = "抱歉，处理时遇到了问题。"
            try:
                await self.adapter.stream_synthesis(
                    session_id=session_id,
                    text=error_message,
                    chunk_size=4096,
                )
            except Exception as tts_error:
                logger.error("Failed to send error message via TTS",
                           error=str(tts_error))
    
    async def start(self):
        """Start the voice companion server."""
        if not self.adapter:
            raise RuntimeError("Call setup() before start()")
        
        logger.info("Starting voice companion server")
        await self.adapter.start()
        
        print("\n" + "="*60)
        print("✅ Voice Companion Ready")
        print("="*60)
        print(f"🌐 WebSocket: ws://{self.adapter.config.host}:{self.adapter.config.port}/audio")
        print(f"🤖 Agent Model: {self.model}")
        print(f"🔊 TTS Voice: {self.adapter.config.tts_voice}")
        print(f"🎤 VAD Threshold: {self.adapter.config.vad_threshold}")
        print("="*60)
        print("\n📋 Usage:")
        print("   1. Connect WebSocket client to ws://localhost:8765/audio")
        print("   2. Send audio frames (PCM 16kHz mono)")
        print("   3. Receive TTS responses in real-time")
        print("\n💡 Test with: python examples/voice_companion/test_client.py")
        print("\nPress Ctrl+C to stop\n")
        
    async def stop(self):
        """Stop the voice companion server."""
        if self.adapter:
            logger.info("Stopping voice companion server")
            await self.adapter.stop()
            print("\n✅ Voice companion stopped")


async def main():
    """Main entry point."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set your OpenAI API key:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("\nOr pass it directly in the code (not recommended for production)")
        sys.exit(1)
    
    # Optional: Override model via environment
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Create and configure voice companion
    companion = VoiceCompanion(api_key=api_key, model=model)
    await companion.setup(host="0.0.0.0", port=8765)
    
    try:
        # Start server
        await companion.start()
        
        # Keep running until interrupted
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down gracefully...")
    except Exception as e:
        logger.error("Fatal error in voice companion", error=str(e), exc_info=e)
        sys.exit(1)
    finally:
        await companion.stop()


if __name__ == "__main__":
    asyncio.run(main())
