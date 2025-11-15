# Test Audio Fixtures

This directory contains test audio files and scripts for the Voice Companion.

## Test Audio Files

Due to file size limitations, audio test files are not included in the repository. You can create your own test files using the provided scripts.

### Creating Test Audio Files

#### Method 1: Record from Microphone (macOS)

```bash
# Record 3 seconds of audio (16kHz, mono, 16-bit PCM)
sox -d -r 16000 -c 1 -b 16 fixtures/hello.wav trim 0 3

# Or use macOS built-in recorder
rec -r 16000 -c 1 -b 16 fixtures/hello.wav trim 0 3
```

#### Method 2: Convert Existing Audio

```bash
# Convert any audio file to required format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 fixtures/hello.wav
```

#### Method 3: Generate Synthetic Audio (Python)

```python
import numpy as np
import wave

# Generate 2 seconds of test tone (440 Hz)
sample_rate = 16000
duration = 2
frequency = 440

t = np.linspace(0, duration, int(sample_rate * duration))
audio_data = (np.sin(2 * np.pi * frequency * t) * 32767 * 0.3).astype(np.int16)

# Save as WAV
with wave.open('fixtures/test_tone.wav', 'wb') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())
```

### Required Audio Format

All test audio must use the following format:
- **Sample Rate**: 16000 Hz
- **Channels**: 1 (Mono)
- **Bit Depth**: 16-bit signed integer (PCM)
- **Format**: WAV (uncompressed)

### Recommended Test Phrases

For Chinese voice testing (small vocabulary):

1. **hello.wav**: "今天天气怎么样" (What's the weather today?)
2. **time.wav**: "现在几点了" (What time is it?)
3. **name.wav**: "你叫什么名字" (What's your name?)
4. **help.wav**: "你能帮我做什么" (What can you help me with?)

For English testing:

1. **hello_en.wav**: "What's the weather like today?"
2. **time_en.wav**: "What time is it now?"
3. **name_en.wav**: "What's your name?"

### Validating Audio Files

```bash
# Check audio file properties
ffprobe -i fixtures/hello.wav

# Should show:
# - Stream: Audio: pcm_s16le
# - Sample rate: 16000 Hz
# - Channels: 1 (mono)
```

## Test Scripts

### websocket_client.py

WebSocket test client for manual testing. See the file for usage instructions.

```bash
python examples/voice_companion/fixtures/websocket_client.py --audio fixtures/hello.wav
```

## Directory Structure

```
fixtures/
├── README.md                 # This file
├── websocket_client.py       # WebSocket test client
├── .gitkeep                  # Keep directory in git
│
# Add your test audio files here:
├── hello.wav                 # (You create this)
├── time.wav                  # (You create this)
├── name.wav                  # (You create this)
└── test_tone.wav            # (You create this)
```

## Notes

- Test audio files are excluded from Git via `.gitignore` (*.wav pattern)
- Each test file should be 2-3 seconds long to test VAD speech detection
- Ensure background noise is minimal for better ASR accuracy
- Use headphones when recording to avoid feedback
