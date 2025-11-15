"""
WebSocket Test Client for Voice Companion

A comprehensive test client for testing the AudioAdapter WebSocket server.
Supports sending audio files, receiving TTS responses, and monitoring sessions.

Prerequisites:
    pip install websockets

Usage:
    # Basic test with generated audio
    python examples/voice_companion/test_client.py

    # Test with specific audio file
    python examples/voice_companion/test_client.py --audio fixtures/test_speech_pattern.wav

    # Save TTS output to file
    python examples/voice_companion/test_client.py --output received_tts.wav

    # Continuous mode (keep connection alive)
    python examples/voice_companion/test_client.py --continuous

Features:
    - Automatic authentication and session management
    - Real-time audio streaming
    - TTS response capture
    - Connection monitoring
    - Error handling and recovery
"""

import argparse
import asyncio
import json
import struct
import wave
from pathlib import Path
from typing import Optional
import sys

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    print("❌ Error: websockets package not installed")
    print("   Install with: pip install websockets")
    sys.exit(1)


# Protocol constants (must match server)
MAGIC_NUMBER = 0x585A  # "XZ"
HEADER_SIZE = 4
META_LENGTH_SIZE = 2


class AudioClient:
    """WebSocket client for voice companion testing."""
    
    def __init__(
        self,
        uri: str = "ws://localhost:8765/audio",
        device_id: str = "test-client-001",
        output_file: Optional[Path] = None,
    ):
        self.uri = uri
        self.device_id = device_id
        self.output_file = output_file
        self.ws: Optional[WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self.audio_chunks = []
        self.connected = False
        
    def serialize_audio_message(
        self,
        session_id: str,
        audio_data: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> bytes:
        """Serialize audio data to binary WebSocket message."""
        import time
        
        metadata = {
            "session_id": session_id,
            "timestamp": int(time.time() * 1000),
            "sample_rate": sample_rate,
            "channels": channels,
            "format": "pcm_s16le",
        }
        
        metadata_json = json.dumps(metadata).encode("utf-8")
        meta_length = len(metadata_json)
        
        # Header: magic(2) + type(1) + flags(1)
        msg_type = 0x01  # Audio
        flags = 0x00
        header = struct.pack(">HBB", MAGIC_NUMBER, msg_type, flags)
        
        # Assemble message
        message = header + struct.pack(">H", meta_length) + metadata_json + audio_data
        return message
    
    def parse_audio_message(self, data: bytes) -> tuple[dict, bytes]:
        """Parse binary audio response from server."""
        if len(data) < HEADER_SIZE + META_LENGTH_SIZE:
            raise ValueError("Message too short")
        
        # Parse header
        magic, msg_type, flags = struct.unpack(">HBB", data[:HEADER_SIZE])
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Invalid magic: 0x{magic:04X}")
        
        # Parse metadata
        meta_length = struct.unpack(">H", data[HEADER_SIZE:HEADER_SIZE + META_LENGTH_SIZE])[0]
        meta_start = HEADER_SIZE + META_LENGTH_SIZE
        meta_end = meta_start + meta_length
        
        metadata = json.loads(data[meta_start:meta_end].decode("utf-8"))
        audio_data = data[meta_end:]
        
        return metadata, audio_data
    
    async def send_json(self, message: dict):
        """Send JSON message."""
        if not self.ws:
            raise RuntimeError("Not connected")
        await self.ws.send(json.dumps(message))
    
    async def send_auth(self):
        """Send authentication message."""
        auth_msg = {
            "type": "auth",
            "device_id": self.device_id,
            "token": "test-token-12345",
        }
        await self.send_json(auth_msg)
        print(f"→ Sent auth: {self.device_id}")
    
    async def send_control(self, action: str, params: Optional[dict] = None):
        """Send control message."""
        control_msg = {
            "type": "control",
            "action": action,
            "params": params or {},
        }
        await self.send_json(control_msg)
        print(f"→ Sent control: {action}")
    
    async def send_audio_file(
        self,
        audio_file: Path,
        chunk_size: int = 4096,
        realtime: bool = True,
    ):
        """Send audio file as binary frames.
        
        Args:
            audio_file: Path to WAV file
            chunk_size: Bytes per chunk
            realtime: Simulate real-time streaming with delays
        """
        if not audio_file.exists():
            print(f"❌ Audio file not found: {audio_file}")
            return
        
        if not self.session_id:
            print("❌ No active session")
            return
        
        # Read WAV file
        with wave.open(str(audio_file), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            total_frames = wf.getnframes()
            duration = total_frames / sample_rate
            
            print(f"\n📁 Sending audio: {audio_file.name}")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Channels: {channels}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Realtime: {realtime}")
            
            # Send chunks
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            
            while True:
                audio_data = wf.readframes(chunk_size // (2 * channels))
                if not audio_data:
                    break
                
                message = self.serialize_audio_message(
                    session_id=self.session_id,
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    channels=channels,
                )
                await self.ws.send(message)
                chunk_count += 1
                
                # Simulate real-time streaming
                if realtime:
                    chunk_duration = len(audio_data) / (2 * channels * sample_rate)
                    await asyncio.sleep(chunk_duration)
            
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"✅ Sent {chunk_count} chunks in {elapsed:.2f}s")
    
    async def receive_loop(self):
        """Receive and process server messages."""
        try:
            async for message in self.ws:
                if isinstance(message, str):
                    # Text message (control/heartbeat)
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "session_created":
                        self.session_id = data.get("session_id")
                        print(f"← Session created: {self.session_id}")
                    
                    elif msg_type == "heartbeat":
                        # Respond to heartbeat
                        await self.send_json({"type": "heartbeat_ack"})
                    
                    elif msg_type == "error":
                        print(f"← Error: {data.get('message', 'Unknown error')}")
                    
                    else:
                        print(f"← {msg_type}: {data}")
                
                else:
                    # Binary message (audio)
                    try:
                        metadata, audio_data = self.parse_audio_message(message)
                        session_id = metadata.get("session_id", "N/A")[:8]
                        is_end = metadata.get("is_end", False)
                        
                        print(f"← Audio chunk: {len(audio_data):,} bytes "
                              f"(session: {session_id}, end: {is_end})")
                        
                        if self.output_file:
                            self.audio_chunks.append(audio_data)
                        
                        if is_end:
                            print("✅ Received final audio chunk")
                    
                    except Exception as e:
                        print(f"❌ Failed to parse audio: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            print("← Connection closed by server")
            self.connected = False
    
    async def connect(self):
        """Connect to WebSocket server."""
        print(f"🔌 Connecting to {self.uri}...")
        self.ws = await websockets.connect(self.uri)
        self.connected = True
        print("✅ Connected")
    
    async def disconnect(self):
        """Disconnect from server."""
        if self.ws:
            await self.ws.close()
            self.connected = False
            print("🔌 Disconnected")
    
    async def save_audio(self):
        """Save received audio chunks to file."""
        if not self.output_file or not self.audio_chunks:
            return
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with wave.open(str(self.output_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(self.audio_chunks))
        
        total_duration = len(b"".join(self.audio_chunks)) / (2 * 16000)
        print(f"💾 Saved {len(self.audio_chunks)} chunks "
              f"({total_duration:.2f}s) to: {self.output_file}")


async def run_test(
    audio_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    continuous: bool = False,
):
    """Run test client."""
    client = AudioClient(output_file=output_file)
    
    try:
        await client.connect()
        
        # Start receive loop in background
        receive_task = asyncio.create_task(client.receive_loop())
        
        # Authenticate
        await client.send_auth()
        await asyncio.sleep(0.5)
        
        # Start session
        await client.send_control("start_session")
        await asyncio.sleep(0.5)
        
        # Send audio if provided
        if audio_file:
            await client.send_audio_file(audio_file, realtime=True)
        else:
            print("\nℹ️  No audio file specified")
            print("   Use --audio <file> to send test audio")
        
        # Wait for responses
        if continuous:
            print("\n⏳ Continuous mode - press Ctrl+C to stop")
            await asyncio.Event().wait()
        else:
            wait_time = 10
            print(f"\n⏳ Waiting for responses ({wait_time}s)...")
            await asyncio.sleep(wait_time)
        
        # End session
        await client.send_control("end_session")
        await asyncio.sleep(0.5)
        
        # Cleanup
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        
        await client.disconnect()
        await client.save_audio()
        
        print("\n✅ Test completed")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if client.connected:
            await client.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WebSocket Test Client for Voice Companion"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        help="Audio file to send (WAV, 16kHz, mono)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output_tts.wav"),
        help="Output file for received TTS (default: output_tts.wav)",
    )
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Keep connection alive (for continuous testing)",
    )
    parser.add_argument(
        "--uri",
        "-u",
        default="ws://localhost:8765/audio",
        help="WebSocket URI (default: ws://localhost:8765/audio)",
    )
    
    args = parser.parse_args()
    
    print("🎤 Voice Companion Test Client")
    print("="*60)
    
    if args.audio and not args.audio.exists():
        print(f"❌ Error: Audio file not found: {args.audio}")
        print("\n💡 Generate test audio with:")
        print("   python examples/voice_companion/fixtures/generate_test_audio.py")
        sys.exit(1)
    
    try:
        asyncio.run(run_test(
            audio_file=args.audio,
            output_file=args.output,
            continuous=args.continuous,
        ))
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
