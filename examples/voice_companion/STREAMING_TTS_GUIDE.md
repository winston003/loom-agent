# Streaming TTS Quick Reference

## Overview

The Xiaozhi voice adapter provides streaming Text-to-Speech (TTS) for real-time audio response delivery with optimized latency.

## Key Features

- **Low First-Packet Latency**: <300ms from request to first audio chunk
- **Configurable Chunking**: Balance latency vs network overhead
- **Real-time Streaming**: Audio chunks sent as they're synthesized
- **WebSocket Integration**: Binary protocol with metadata
- **EdgeTTS Backend**: No API key required, free Microsoft Edge voices

## API

### Streaming Synthesis

```python
await adapter.stream_synthesis(
    session_id="abc-123",
    text="Hello, this is a streaming response",
    voice="zh-CN-XiaoxiaoNeural",  # Optional, uses config default
    chunk_size=2048,  # Optional, default 4096
)
```

**Parameters**:
- `session_id` (str): Target session ID (must be active)
- `text` (str): Text to synthesize
- `voice` (str, optional): EdgeTTS voice ID (default from `AudioAdapterConfig.tts_voice`)
- `chunk_size` (int, optional): Audio chunk size in bytes (default 4096)

**Raises**:
- `AudioAdapterError`: If TTS not initialized or session not found

### Batch Synthesis (Non-Streaming)

```python
audio_bytes = await adapter.synthesize_speech(
    text="Complete audio generated before returning",
    voice="zh-CN-XiaoxiaoNeural",
)
# Returns: bytes (full audio PCM data)
```

## Latency Optimization

### Chunk Size Tradeoff

| Chunk Size | Latency @ 16kHz | Network Overhead | Use Case |
|------------|-----------------|------------------|----------|
| 1024 bytes | ~64ms          | High             | Ultra-low latency (interactive games) |
| 2048 bytes | ~128ms         | Medium           | **Recommended** (voice assistants) |
| 4096 bytes | ~256ms         | Low              | High-quality playback (podcasts) |

### End-to-End Latency Breakdown

```
User speaks → VAD detects → ASR transcribes → Agent responds → TTS streams → User hears
   ~50ms        ~200ms         ~500ms           ~100ms          <300ms         total: ~1150ms
```

**Optimization Tips**:
1. **Reduce chunk size** to 2048 or 1024 for interactive scenarios
2. **Use edge deployment** for ASR/TTS to minimize network RTT
3. **Enable VAD aggressive mode** to reduce detection latency
4. **Pre-warm models** (VAD/ASR/TTS) during adapter startup

## Voice Selection

### Chinese Voices (EdgeTTS)

```python
config = AudioAdapterConfig(
    tts_voice="zh-CN-XiaoxiaoNeural",  # Female, warm (default)
    # tts_voice="zh-CN-YunxiNeural",   # Male, clear
    # tts_voice="zh-CN-YunyangNeural", # Male, energetic
)
```

### English Voices

```python
config = AudioAdapterConfig(
    tts_voice="en-US-JennyNeural",  # Female, friendly
    # tts_voice="en-US-GuyNeural",  # Male, professional
)
```

**List all available voices**:
```bash
edge-tts --list-voices | grep zh-CN
```

## Performance Monitoring

### Built-in Metrics

The adapter logs key metrics automatically:

```
[INFO] Starting streaming TTS (session_id=abc-123, text_length=42, chunk_size=2048)
[INFO] First TTS packet sent (latency_ms=287.3, session_id=abc-123)
[INFO] Streaming TTS completed (session_id=abc-123, chunks=8, total_bytes=15872, duration_ms=1234.5)
```

### Custom Monitoring

```python
import time

start = time.time()
await adapter.stream_synthesis(session_id, text)
total_latency = (time.time() - start) * 1000

print(f"Total streaming time: {total_latency:.1f}ms")
```

## Integration Example

### With Loom Agent Response

```python
from loom import agent_from_env
from loom.adapters.audio import AudioAdapter

# Create agent
agent = agent_from_env(
    system_instructions="You are Xiaozhi, a voice assistant. Keep responses concise."
)

# Create audio adapter
adapter = AudioAdapter(
    on_transcription=handle_transcription,
)

async def handle_transcription(session_id: str, result: TranscriptionResult):
    # Get agent response
    response = await agent.run(result.text)
    
    # Stream TTS response
    await adapter.stream_synthesis(
        session_id=session_id,
        text=response.content,
        chunk_size=2048,  # Low latency for interactive feel
    )
```

### Error Handling

```python
try:
    await adapter.stream_synthesis(session_id, text)
except AudioAdapterError as e:
    logger.error(f"Streaming failed: {e}")
    # Fallback: use batch synthesis
    audio = await adapter.synthesize_speech(text)
    # Send audio via alternative method
```

## WebSocket Protocol

Audio chunks are sent using the binary WebSocket protocol:

```
Binary Message Structure:
┌────────────────┬────────────────┬────────────────┬──────────────────┐
│  Header (4B)   │ Meta Length(2B)│  Metadata JSON │  Audio PCM Data  │
└────────────────┴────────────────┴────────────────┴──────────────────┘
```

**Metadata** (JSON):
```json
{
  "session_id": "abc-123",
  "timestamp": 1699876543210,
  "sample_rate": 16000,
  "channels": 1,
  "format": "pcm_s16le",
  "is_end": false
}
```

**Final Chunk**: Set `is_end=true` in metadata.

## Testing

### Manual Test

```bash
# Terminal 1: Start server
python examples/voice_companion/streaming_tts_demo.py

# Terminal 2: Test client
python examples/voice_companion/fixtures/websocket_client.py
```

### Unit Test

```python
import pytest
from loom.adapters.audio import AudioAdapter, AudioAdapterConfig

@pytest.mark.asyncio
async def test_stream_synthesis():
    config = AudioAdapterConfig(port=8766)
    adapter = AudioAdapter(config)
    
    await adapter.start()
    
    # Create mock session
    session_id = await adapter.create_session("test-device")
    
    # Stream synthesis
    await adapter.stream_synthesis(
        session_id=session_id,
        text="Test streaming",
        chunk_size=1024,
    )
    
    await adapter.stop()
```

## Troubleshooting

### High Latency

**Symptom**: First packet >500ms

**Diagnosis**:
```python
# Check TTS provider latency
import time
start = time.time()
async for chunk in adapter.tts.synthesize("test"):
    print(f"First chunk: {(time.time() - start) * 1000:.1f}ms")
    break
```

**Solutions**:
1. Check network connectivity to EdgeTTS servers
2. Reduce chunk size to 1024
3. Pre-warm TTS by synthesizing dummy text on startup

### Session Not Found

**Symptom**: `AudioAdapterError: No active connection found for session`

**Cause**: Session closed or invalid session_id

**Solution**:
```python
# Verify session exists
session = adapter.get_session(session_id)
if not session:
    logger.error(f"Session {session_id} not found")
    return

await adapter.stream_synthesis(session_id, text)
```

### Audio Glitches

**Symptom**: Choppy or corrupted audio playback

**Cause**: Network packet loss or buffer underrun

**Solutions**:
1. Increase chunk size to 4096 (more buffering)
2. Use TCP congestion control optimization
3. Implement client-side jitter buffer

## References

- [EdgeTTS Documentation](https://github.com/rany2/edge-tts)
- [WebSocket Protocol Spec](../../specs/002-xiaozhi-voice-adapter/contracts/websocket_protocol.md)
- [Audio Adapter Implementation](../../loom/adapters/audio/adapter.py)
- [Streaming Example](streaming_tts_demo.py)
