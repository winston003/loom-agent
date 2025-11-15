# End-to-End Testing Guide for Voice Companion

This guide covers testing the complete voice interaction pipeline from audio input to TTS output.

## Prerequisites

1. **Install Dependencies**
   ```bash
   pip install loom-agent[audio]
   pip install websockets  # For test client
   ```

2. **Set Environment Variables**
   ```bash
   export OPENAI_API_KEY="sk-..."  # Required for GPT-4
   export OPENAI_MODEL="gpt-4"     # Optional, defaults to gpt-4
   ```

3. **Generate Test Audio**
   ```bash
   python examples/voice_companion/fixtures/generate_test_audio.py
   ```

## Test Scenarios

### Scenario 1: Basic Voice Interaction (Synthetic Audio)

**Goal**: Verify complete pipeline with synthetic test audio

```bash
# Terminal 1: Start voice companion
python examples/voice_companion/hello_voice.py

# Terminal 2: Send test audio (wait for server to start)
python examples/voice_companion/test_client.py \
    --audio examples/voice_companion/fixtures/test_speech_pattern.wav \
    --output received_tts.wav
```

**Expected Output (Terminal 1)**:
```
✅ Voice Companion Ready
🌐 WebSocket: ws://0.0.0.0:8765/audio
🤖 Agent Model: gpt-4
🔊 TTS Voice: zh-CN-XiaoxiaoNeural

🎙️  New session: <session_id>
   Device: test-client-001

📝 User [<session_id>]: <transcribed_text>
   Confidence: X.XX%
   ASR Latency: XXms

🤔 Processing with gpt-4...
🤖 Assistant: <response_text>
🔊 Synthesizing speech...
✅ Response sent
```

**Expected Output (Terminal 2)**:
```
🔌 Connecting to ws://localhost:8765/audio...
✅ Connected
→ Sent auth: test-client-001
→ Sent control: start_session
← Session created: <session_id>

📁 Sending audio: test_speech_pattern.wav
   Sample rate: 16000 Hz
   Channels: 1
   Duration: 3.00s
   Realtime: True
✅ Sent X chunks in X.XXs

← Audio chunk: X,XXX bytes (session: <session_id>, end: False)
← Audio chunk: X,XXX bytes (session: <session_id>, end: False)
...
✅ Received final audio chunk

💾 Saved X chunks (X.XXs) to: received_tts.wav
🔌 Disconnected
✅ Test completed
```

**Verification**:
- Check `received_tts.wav` exists and contains audio
- Play `received_tts.wav` to verify TTS quality
- Confirm no errors in either terminal

---

### Scenario 2: Real Audio Recording (If Available)

**Goal**: Test with actual voice input

```bash
# Record 2-3 seconds of audio (macOS)
sox -d -r 16000 -c 1 -b 16 examples/voice_companion/fixtures/my_voice.wav trim 0 3

# Test with recorded audio
python examples/voice_companion/test_client.py \
    --audio examples/voice_companion/fixtures/my_voice.wav \
    --output my_response.wav
```

**Test Phrases** (for Chinese TTS):
- "今天天气怎么样" (What's the weather today?)
- "现在几点了" (What time is it?)
- "你好小智" (Hello Xiaozhi)

---

### Scenario 3: Continuous Session Testing

**Goal**: Test multiple interactions in one session

```bash
# Terminal 1: Start server (same as before)
python examples/voice_companion/hello_voice.py

# Terminal 2: Continuous mode
python examples/voice_companion/test_client.py --continuous
```

Then manually send audio via WebSocket protocol or use multiple client instances.

---

### Scenario 4: Performance Testing

**Goal**: Measure end-to-end latency

```bash
# Run hello_voice.py with debug logging
LOG_LEVEL=DEBUG python examples/voice_companion/hello_voice.py
```

**Key Metrics** (from logs):
- **VAD Latency**: Speech detection time
- **ASR Latency**: Transcription time (should be in `result.latency_ms`)
- **LLM Latency**: Agent processing time
- **TTS First Chunk**: Time to first audio packet
- **Total E2E Latency**: < 1.5s (target from spec.md)

Check logs for:
```
Transcription received ... latency_ms=XXX
Agent response generated ... (measure time between transcription and this)
TTS streaming completed ... (measure time from response to completion)
```

---

### Scenario 5: VAD Testing

**Goal**: Verify speech detection and silence rejection

```bash
# Terminal 1: Start server
python examples/voice_companion/hello_voice.py

# Terminal 2: Test with silence (should NOT trigger ASR)
python examples/voice_companion/test_client.py \
    --audio examples/voice_companion/fixtures/test_silence.wav

# Expected: No transcription event (silence filtered by VAD)

# Test with speech pattern (SHOULD trigger ASR)
python examples/voice_companion/test_client.py \
    --audio examples/voice_companion/fixtures/test_speech_pattern.wav

# Expected: Transcription event triggered
```

---

### Scenario 6: Error Handling

**Goal**: Test error recovery

```bash
# Test without OPENAI_API_KEY
unset OPENAI_API_KEY
python examples/voice_companion/hello_voice.py

# Expected: Clear error message about missing API key

# Test with invalid API key
export OPENAI_API_KEY="sk-invalid"
python examples/voice_companion/hello_voice.py
# Send audio - should get error message via TTS

# Test ASR failure (FunASR without models)
# - ASR will initialize but fail during transcription
# - Check for graceful error handling
```

---

## Automated Test Script

Create `examples/voice_companion/run_e2e_tests.sh`:

```bash
#!/bin/bash
set -e

echo "🧪 Running End-to-End Tests for Voice Companion"
echo "================================================"

# Check prerequisites
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    exit 1
fi

if ! python3 -c "import websockets" 2>/dev/null; then
    echo "❌ Error: websockets package not installed"
    echo "   Install with: pip install websockets"
    exit 1
fi

# Generate test audio if not exists
if [ ! -f "examples/voice_companion/fixtures/test_speech_pattern.wav" ]; then
    echo "📁 Generating test audio..."
    python3 examples/voice_companion/fixtures/generate_test_audio.py
fi

# Start server in background
echo "🚀 Starting voice companion server..."
python3 examples/voice_companion/hello_voice.py &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Run test client
echo "🧪 Running test client..."
python3 examples/voice_companion/test_client.py \
    --audio examples/voice_companion/fixtures/test_speech_pattern.wav \
    --output test_output.wav

# Cleanup
echo "🧹 Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Verify output
if [ -f "test_output.wav" ]; then
    echo "✅ Test completed successfully"
    echo "   Output saved to: test_output.wav"
    ls -lh test_output.wav
else
    echo "❌ Test failed: No output file generated"
    exit 1
fi
```

Make it executable:
```bash
chmod +x examples/voice_companion/run_e2e_tests.sh
./examples/voice_companion/run_e2e_tests.sh
```

---

## Debugging Tips

### 1. No Audio Received

**Check**:
- Server logs for session creation
- WebSocket connection established
- Audio file format (must be 16kHz, mono, WAV)

**Debug**:
```python
# Add debug logging in hello_voice.py
logger.setLevel("DEBUG")
```

### 2. ASR Not Working

**Cause**: FunASR requires model files

**Solution**:
```bash
# Download FunASR models (see specs/002-xiaozhi-voice-adapter/README.md)
mkdir -p models/paraformer-zh
# Follow FunASR installation guide
```

**Temporary workaround**: Use EdgeTTS-only mode (TTS will work, ASR will fail gracefully)

### 3. VAD Not Detecting Speech

**Check**:
- `vad_threshold` in config (default: 0.5)
- Audio amplitude (try `test_tone.wav` for strong signal)

**Adjust**:
```python
config = AudioAdapterConfig(
    vad_threshold=0.3,  # Lower = more sensitive
    min_speech_duration_ms=200,  # Shorter minimum speech
)
```

### 4. TTS Latency Too High

**Optimize**:
```python
# In hello_voice.py or streaming_tts_demo.py
await self.adapter.stream_synthesis(
    session_id=session_id,
    text=response_text,
    chunk_size=2048,  # Smaller chunks = lower latency (default: 4096)
)
```

---

## Success Criteria (From tasks.md)

### T038 - Audio Pipeline
- ✅ Pipeline processes test audio files without errors
- ✅ End-to-end latency < 1.5s (measure from audio input to TTS start)
- ✅ Each stage delay recorded in logs
- ✅ No error logs during normal operation

### T044 - AudioAdapter
- ✅ Adapter starts successfully
- ✅ All services (VAD/ASR/TTS) healthy
- ✅ WebSocket server listening on correct port
- ✅ No startup errors

### T055 - hello_voice.py Example
- ✅ Single file runnable
- ✅ OpenAI GPT-4 integration working
- ✅ Complete voice interaction demonstrated

### T056 - Test Audio Fixtures
- ✅ Test files generated (test_tone.wav, test_speech_pattern.wav, etc.)
- ✅ Files in correct format (16kHz, mono, PCM)

### T057 - WebSocket Test Client
- ✅ Client connects to server
- ✅ Audio sending works
- ✅ TTS receiving and saving works
- ✅ Clear command-line interface

---

## Next Steps

After successful E2E testing:

1. **Phase 4**: Voiceprint verification (User Story 2)
   - Implement speaker identification
   - Permission-based access control

2. **Phase 5**: Multi-turn context (User Story 3)
   - Conversation history management
   - Context compression

3. **Phase 6**: Production readiness
   - Health checks
   - Monitoring and metrics
   - Docker deployment

See `specs/002-xiaozhi-voice-adapter/tasks.md` for detailed task breakdown.
