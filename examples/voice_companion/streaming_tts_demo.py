"""
Streaming TTS Demo - Real-time speech synthesis with low latency.

This example demonstrates:
1. Streaming TTS output to minimize first-packet latency
2. Configurable chunk sizes for latency/quality tradeoff
3. Integration with WebSocket for real-time audio delivery

Prerequisites:
    poetry install --extras audio
    # EdgeTTS requires no API key (uses Microsoft Edge's TTS)

Usage:
    # Terminal 1: Start server
    python examples/voice_companion/streaming_tts_demo.py

    # Terminal 2: Connect with WebSocket client (example below)
    # See: examples/voice_companion/fixtures/websocket_client.py

Performance Notes:
    - Default chunk_size=4096 (~256ms @ 16kHz) balances latency and network overhead
    - Smaller chunks (2048 or 1024) reduce latency but increase overhead
    - First packet typically arrives <300ms after request
    - EdgeTTS streaming naturally produces chunks, we just buffer to target size
"""

import asyncio
import sys

from loom.adapters.audio import AudioAdapter, AudioAdapterConfig
from loom.adapters.audio.models import AudioSession
from loom.interfaces.audio_adapter import TranscriptionResult
from loom.core.structured_logger import get_logger

logger = get_logger(__name__)


# Sample responses for demo (in production, these come from Loom Agent)
DEMO_RESPONSES = [
    "你好！我是小智，很高兴为你服务。",
    "今天天气不错，适合出门散步。",
    "请问有什么我可以帮助你的吗？",
    "这是一个流式语音合成的演示，注意首包延迟。",
]

response_index = 0


def on_session_start(session: AudioSession) -> None:
    """Called when a new audio session starts."""
    print(f"🎙️  Session started: {session.session_id}")
    print(f"   Device: {session.device_id}")
    print(f"   State: {session.state.value}")


def on_session_end(session_id: str) -> None:
    """Called when an audio session ends."""
    print(f"⏹️  Session ended: {session_id}")


async def on_transcription(session_id: str, result: TranscriptionResult, adapter: AudioAdapter) -> None:
    """Called when transcription completes - triggers streaming TTS response."""
    global response_index
    
    print(f"\n📝 Transcription [{session_id}]:")
    print(f"   Text: {result.text}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Latency: {result.latency_ms}ms")
    
    # Generate response (in production, this would call Loom Agent)
    response_text = DEMO_RESPONSES[response_index % len(DEMO_RESPONSES)]
    response_index += 1
    
    print(f"\n🔊 Streaming TTS response: {response_text}")
    
    # Stream TTS with configurable chunk size
    # Smaller chunks = lower latency, but more network overhead
    try:
        await adapter.stream_synthesis(
            session_id=session_id,
            text=response_text,
            chunk_size=2048,  # ~128ms @ 16kHz for lower latency demo
        )
        print(f"✅ TTS streaming completed for session {session_id}\n")
    except Exception as e:
        logger.error("Failed to stream TTS", error=str(e), session_id=session_id)


async def main():
    print("🎤 Starting Streaming TTS Demo")
    print("=" * 60)
    print("This demo shows real-time speech synthesis streaming")
    print("Connect a WebSocket client to ws://localhost:8765/audio")
    print("=" * 60)
    print()

    # Create adapter with optimized config for low latency
    config = AudioAdapterConfig(
        host="0.0.0.0",
        port=8765,
        vad_threshold=0.5,
        sample_rate=16000,
        channels=1,
        max_connections=10,
        # TTS voice (EdgeTTS voices)
        tts_voice="zh-CN-XiaoxiaoNeural",  # Female voice
        # tts_voice="zh-CN-YunxiNeural",   # Male voice
    )
    
    adapter = AudioAdapter(
        config=config,
        on_session_start=on_session_start,
        on_session_end=on_session_end,
    )
    
    # Wrap on_transcription to pass adapter
    async def transcription_wrapper(session_id: str, result: TranscriptionResult):
        await on_transcription(session_id, result, adapter)
    
    adapter._on_transcription = transcription_wrapper
    
    try:
        # Start the adapter
        await adapter.start()
        
        print("✅ Streaming TTS server started")
        print(f"🌐 WebSocket: ws://{config.host}:{config.port}/audio")
        print(f"🔊 TTS Voice: {config.tts_voice}")
        print(f"⚡ Chunk size: 2048 bytes (~128ms latency)")
        print()
        print("📊 Performance Metrics:")
        print("   - First packet latency: target <300ms")
        print("   - Streaming overhead: ~10-20ms per chunk")
        print("   - End-to-end latency: VAD + ASR + TTS + network")
        print()
        print("Waiting for WebSocket connections...\n")
        
        # Keep server running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        logger.error("Fatal error in streaming TTS demo", error=str(e), exc_info=e)
        sys.exit(1)
    finally:
        await adapter.stop()
        print("✅ Stopped")


if __name__ == "__main__":
    asyncio.run(main())
