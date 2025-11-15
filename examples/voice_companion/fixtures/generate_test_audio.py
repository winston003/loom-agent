"""Generate Test Audio Files for Voice Companion Testing

Creates synthetic test audio files in the required format (16kHz, mono, 16-bit PCM).

Prerequisites:
    pip install numpy scipy

Usage:
    python examples/voice_companion/fixtures/generate_test_audio.py

Output:
    - test_tone.wav: 440 Hz sine wave (2 seconds)
    - test_silence.wav: Pure silence (2 seconds)
    - test_noise.wav: White noise (2 seconds)
    - test_speech_pattern.wav: Speech-like pattern (3 seconds)

These files can be used to test VAD, ASR integration, and audio pipeline.
"""

import wave
import numpy as np
from pathlib import Path


SAMPLE_RATE = 16000
OUTPUT_DIR = Path(__file__).parent


def save_wav(filename: str, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Save audio data as WAV file.
    
    Args:
        filename: Output filename
        audio_data: Audio samples (float32 or int16)
        sample_rate: Sample rate in Hz
    """
    # Convert to int16 if needed
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    filepath = OUTPUT_DIR / filename
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✅ Created: {filepath} ({len(audio_data) / sample_rate:.2f}s)")


def generate_tone(frequency: float = 440.0, duration: float = 2.0, 
                 amplitude: float = 0.3) -> np.ndarray:
    """Generate sine wave tone.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        amplitude: Amplitude (0.0 to 1.0)
    
    Returns:
        Audio samples (int16)
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    audio = np.sin(2 * np.pi * frequency * t) * amplitude
    return (audio * 32767).astype(np.int16)


def generate_silence(duration: float = 2.0) -> np.ndarray:
    """Generate silence.
    
    Args:
        duration: Duration in seconds
    
    Returns:
        Audio samples (int16, all zeros)
    """
    samples = int(SAMPLE_RATE * duration)
    return np.zeros(samples, dtype=np.int16)


def generate_white_noise(duration: float = 2.0, amplitude: float = 0.1) -> np.ndarray:
    """Generate white noise.
    
    Args:
        duration: Duration in seconds
        amplitude: Amplitude (0.0 to 1.0)
    
    Returns:
        Audio samples (int16)
    """
    samples = int(SAMPLE_RATE * duration)
    noise = np.random.uniform(-amplitude, amplitude, samples)
    return (noise * 32767).astype(np.int16)


def generate_speech_pattern(duration: float = 3.0) -> np.ndarray:
    """Generate speech-like pattern with pauses.
    
    Creates alternating segments of tones and silence to simulate speech.
    Useful for testing VAD speech/silence detection.
    
    Args:
        duration: Duration in seconds
    
    Returns:
        Audio samples (int16)
    """
    samples = int(SAMPLE_RATE * duration)
    audio = np.zeros(samples, dtype=np.float32)
    
    # Create speech-like segments
    # Segment 1: 0.5s of low-frequency tone (simulates speech)
    t1 = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5))
    speech1 = np.sin(2 * np.pi * 200 * t1) * 0.3
    audio[:len(speech1)] = speech1
    
    # Pause: 0.3s silence
    
    # Segment 2: 0.7s of modulated tone
    t2_start = int(SAMPLE_RATE * 0.8)
    t2 = np.linspace(0, 0.7, int(SAMPLE_RATE * 0.7))
    speech2 = np.sin(2 * np.pi * 250 * t2) * 0.35 * (1 + 0.3 * np.sin(2 * np.pi * 5 * t2))
    audio[t2_start:t2_start + len(speech2)] = speech2
    
    # Pause: 0.2s silence
    
    # Segment 3: 0.5s of higher-frequency tone
    t3_start = int(SAMPLE_RATE * 1.7)
    t3 = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5))
    speech3 = np.sin(2 * np.pi * 300 * t3) * 0.28
    audio[t3_start:t3_start + len(speech3)] = speech3
    
    # Add subtle noise to make it more realistic
    noise = np.random.normal(0, 0.02, samples).astype(np.float32)
    audio += noise
    
    return (audio * 32767).astype(np.int16)


def main():
    """Generate all test audio files."""
    print("🎵 Generating test audio files...")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"⚙️  Sample rate: {SAMPLE_RATE} Hz\n")
    
    # Generate test files
    save_wav("test_tone.wav", generate_tone(frequency=440, duration=2.0))
    save_wav("test_silence.wav", generate_silence(duration=2.0))
    save_wav("test_noise.wav", generate_white_noise(duration=2.0, amplitude=0.1))
    save_wav("test_speech_pattern.wav", generate_speech_pattern(duration=3.0))
    
    print("\n✅ All test audio files generated successfully!")
    print("\n📋 Usage:")
    print("   - test_tone.wav: Pure sine wave for audio pipeline testing")
    print("   - test_silence.wav: Silence for VAD testing (should not trigger)")
    print("   - test_noise.wav: White noise for noise rejection testing")
    print("   - test_speech_pattern.wav: Speech-like pattern for VAD testing")
    print("\n💡 Test with:")
    print("   python examples/voice_companion/fixtures/websocket_client.py --audio fixtures/test_speech_pattern.wav")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("❌ Error: Missing required package")
        print(f"   {e}")
        print("\nPlease install required packages:")
        print("   pip install numpy scipy")
    except Exception as e:
        print(f"❌ Error generating test audio: {e}")
        import traceback
        traceback.print_exc()
