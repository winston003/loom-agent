# Voice Companion Examples

This directory contains examples demonstrating the Xiaozhi Voice Adapter integration with Loom Agent.

## Overview

The Xiaozhi Voice Adapter enables voice-based interaction with Loom agents through:
- **VAD (Voice Activity Detection)**: Detect speech segments (SileroVAD)
- **ASR (Automatic Speech Recognition)**: Convert speech to text (FunASR)
- **TTS (Text-to-Speech)**: Convert agent responses to audio (EdgeTTS)
- **Voiceprint**: Speaker identification and verification (optional, 3DSpeaker)

## Quick Start

### Prerequisites

```bash
# Install Loom Agent with audio support
pip install loom-agent[audio]

# Install test client dependencies
pip install websockets

# Set OpenAI API key (required for GPT-4)
export OPENAI_API_KEY="sk-..."
```

### Run Your First Voice Interaction

```bash
# Generate test audio files
python fixtures/generate_test_audio.py

# Terminal 1: Start voice companion
python hello_voice.py

# Terminal 2: Send test audio
python test_client.py --audio fixtures/test_speech_pattern.wav
```

### Automated End-to-End Test

```bash
./run_e2e_test.sh
```

This will:
- ✅ Check all prerequisites
- ✅ Generate test audio if needed
- ✅ Start the voice companion server
- ✅ Send test audio via WebSocket
- ✅ Receive and save TTS response
- ✅ Verify complete pipeline

## Examples

### `hello_voice.py` - Basic Voice Interaction ⭐

**Complete voice interaction with OpenAI GPT-4 integration**

**Features**:
- ✅ WebSocket server for audio streaming
- ✅ VAD-based speech detection (SileroVAD)
- ✅ ASR transcription (FunASR)
- ✅ Loom Agent with GPT-4 processing
- ✅ Streaming TTS synthesis (EdgeTTS)
- ✅ Real-time audio delivery

**Usage**:
```bash
python hello_voice.py
```

**Configuration**:
```bash
# Environment variables
export OPENAI_API_KEY="sk-..."     # Required
export OPENAI_MODEL="gpt-4"        # Optional (default: gpt-4)
```

**Performance**: Target < 1.5s end-to-end latency (from speech end to TTS start)

### `streaming_tts_demo.py` - Low-Latency TTS

**Optimized streaming TTS with configurable chunking**
**Optimized streaming TTS with configurable chunking**

**Features**:
- ✅ Chunk-based audio streaming
- ✅ Configurable chunk size (latency vs. overhead tradeoff)
- ✅ Performance metrics logging
- ✅ EdgeTTS integration with voice selection
- ✅ Demo responses for testing

**Usage**:
```bash
python streaming_tts_demo.py
```

**Performance Tuning**:
```python
await adapter.stream_synthesis(
    session_id=session_id,
    text=response_text,
    chunk_size=2048,  # Options: 1024, 2048, 4096, 8192
)
```

**Chunk Size Guide**:
- **1024 bytes** (~64ms): Lowest latency, highest overhead
- **2048 bytes** (~128ms): Low latency, balanced
- **4096 bytes** (~256ms): Default, good balance
- **8192 bytes** (~512ms): Lower overhead, higher latency

### `test_client.py` - WebSocket Test Client 🧪

**Comprehensive test client for voice companion testing**

**Features**:
- ✅ Command-line interface with arguments
- ✅ Audio file sending (WAV format)
- ✅ TTS response capture and saving
- ✅ Real-time streaming simulation
- ✅ Connection monitoring

**Usage**:
```bash
# Basic test
python test_client.py --audio fixtures/test_speech_pattern.wav

# Save TTS output
python test_client.py \
    --audio fixtures/test_speech_pattern.wav \
    --output my_response.wav

# Continuous mode
python test_client.py --continuous
```

## Testing

### Automated Testing (Recommended)

```bash
./run_e2e_test.sh
```

### Manual Testing

**Synthetic Test Audio**:
```bash
python fixtures/generate_test_audio.py
python test_client.py --audio fixtures/test_speech_pattern.wav
```

**Real Voice Recording**:
```bash
# Record audio (macOS)
sox -d -r 16000 -c 1 -b 16 fixtures/my_voice.wav trim 0 3

# Test
python test_client.py --audio fixtures/my_voice.wav
```

## Directory Structure

```
voice_companion/
├── README.md                    # This file
├── hello_voice.py               # ⭐ Main voice companion
├── streaming_tts_demo.py        # Low-latency TTS demo
├── test_client.py               # 🧪 WebSocket test client
├── run_e2e_test.sh             # 🚀 Automated E2E test
├── E2E_TESTING.md              # Detailed testing guide
├── STREAMING_TTS_GUIDE.md      # TTS optimization guide
└── fixtures/
    ├── README.md                # Fixtures documentation
    ├── generate_test_audio.py   # Test audio generator
    ├── websocket_client.py      # Basic WebSocket client
    └── test_*.wav              # Generated test files
```

## Architecture

```
Xiaozhi Device (WebSocket)
    ↓
AudioAdapter (ws://localhost:8765)
    ├── VAD (SileroVAD) - Detect speech segments
    ├── ASR (FunASR) - Transcribe to text
    ├── Loom Agent (GPT-4) - Generate response
    └── TTS (EdgeTTS) - Synthesize speech
    ↓
Audio Response (PCM 16kHz)
```

## Troubleshooting

**No audio output**:
- Check OPENAI_API_KEY is set
- Verify WebSocket connection established
- Check server logs for errors

**ASR not working**:
- FunASR requires model files (models/paraformer-zh)
- See `specs/002-xiaozhi-voice-adapter/` for setup

**High latency**:
- Check logs: `LOG_LEVEL=DEBUG python hello_voice.py`
- Reduce TTS chunk_size for lower latency
- Verify network latency to OpenAI

**Import errors**:
- Reinstall: `pip install loom-agent[audio]`
- Check Python version: 3.11+ required

## Next Steps

- Review complete testing guide: `E2E_TESTING.md`
- Explore TTS optimization: `STREAMING_TTS_GUIDE.md`
- See full specification: `specs/002-xiaozhi-voice-adapter/`
- Implement voiceprint verification (Phase 4)
- Add multi-turn context (Phase 5)
