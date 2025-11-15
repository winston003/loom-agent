"""
Simple WebSocket Client for Testing Xiaozhi Audio Adapter

This is a minimal test client for manually testing the audio adapter WebSocket server.
For production use, integrate with actual Xiaozhi device client.

Prerequisites:
    pip install websockets

Usage:
    python examples/voice_companion/fixtures/websocket_client.py

Protocol:
    1. Connect to ws://localhost:8765/audio
    2. Send auth message: {"type": "auth", "device_id": "test-device", "token": "optional"}
    3. Send start_session: {"type": "control", "action": "start_session"}
    4. Send binary audio frames (see serialize_audio_message in protocol.py)
    5. Receive binary audio responses (TTS output)
"""

import asyncio
import json
import struct
import wave
from pathlib import Path
from typing import Optional

import websockets
from websockets.client import WebSocketClientProtocol


# Protocol constants (must match server)
MAGIC_NUMBER = 0x585A  # "XZ"
HEADER_SIZE = 4
META_LENGTH_SIZE = 2


def serialize_audio_message(
    session_id: str,
    audio_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """Serialize audio data to binary WebSocket message (client version)."""
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


def parse_audio_message(data: bytes) -> tuple[dict, bytes]:
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


async def send_auth(ws: WebSocketClientProtocol, device_id: str = "test-client"):
    """Send authentication message."""
    auth_msg = {
        "type": "auth",
        "device_id": device_id,
        "token": "test-token-12345",
    }
    await ws.send(json.dumps(auth_msg))
    print(f"→ Sent auth: {device_id}")


async def send_control(ws: WebSocketClientProtocol, action: str, params: Optional[dict] = None):
    """Send control message."""
    control_msg = {
        "type": "control",
        "action": action,
        "params": params or {},
    }
    await ws.send(json.dumps(control_msg))
    print(f"→ Sent control: {action}")


async def send_audio_file(
    ws: WebSocketClientProtocol,
    session_id: str,
    audio_file: Path,
    chunk_size: int = 4096,
):
    """Send audio file as binary frames."""
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    # Read WAV file
    with wave.open(str(audio_file), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        total_frames = wf.getnframes()
        
        print(f"📁 Sending audio: {audio_file.name}")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Channels: {channels}")
        print(f"   Duration: {total_frames / sample_rate:.2f}s")
        
        # Send chunks
        chunk_count = 0
        while True:
            audio_data = wf.readframes(chunk_size // (2 * channels))  # 2 bytes per sample
            if not audio_data:
                break
            
            message = serialize_audio_message(
                session_id=session_id,
                audio_data=audio_data,
                sample_rate=sample_rate,
                channels=channels,
            )
            await ws.send(message)
            chunk_count += 1
            
            # Simulate real-time streaming (optional)
            await asyncio.sleep(0.1)
        
        print(f"✅ Sent {chunk_count} audio chunks")


async def receive_loop(ws: WebSocketClientProtocol, output_file: Optional[Path] = None):
    """Receive and process server messages."""
    audio_chunks = []
    
    try:
        async for message in ws:
            if isinstance(message, str):
                # Text message (control/heartbeat)
                data = json.loads(message)
                print(f"← Received: {data.get('type', 'unknown')} - {data}")
            else:
                # Binary message (audio)
                try:
                    metadata, audio_data = parse_audio_message(message)
                    print(f"← Audio chunk: {len(audio_data)} bytes (session: {metadata.get('session_id', 'N/A')})")
                    
                    if output_file:
                        audio_chunks.append(audio_data)
                    
                    # Check if final chunk
                    if metadata.get("is_end"):
                        print("✅ Received final audio chunk")
                        
                except Exception as e:
                    print(f"❌ Failed to parse audio: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by server")
    
    # Save received audio to file
    if output_file and audio_chunks:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(audio_chunks))
        print(f"💾 Saved audio to: {output_file}")


async def main():
    uri = "ws://localhost:8765/audio"
    device_id = "test-client-001"
    
    print("🔌 Connecting to audio adapter...")
    print(f"   URI: {uri}")
    
    async with websockets.connect(uri) as ws:
        print("✅ Connected")
        
        # Start receive loop in background
        receive_task = asyncio.create_task(
            receive_loop(ws, output_file=Path("output_tts.wav"))
        )
        
        # Send auth
        await send_auth(ws, device_id)
        await asyncio.sleep(0.5)
        
        # Start session
        await send_control(ws, "start_session")
        await asyncio.sleep(0.5)
        
        # Send test audio (if available)
        test_audio = Path(__file__).parent / "test_audio.wav"
        if test_audio.exists():
            # Assume we got a session_id from server response
            session_id = "test-session-001"
            await send_audio_file(ws, session_id, test_audio)
        else:
            print(f"ℹ️  No test audio file found at {test_audio}")
            print("   Server will wait for audio input...")
        
        # Keep connection alive for a bit to receive responses
        print("\n⏳ Waiting for server responses (10s)...")
        await asyncio.sleep(10)
        
        # End session
        await send_control(ws, "end_session")
        await asyncio.sleep(0.5)
        
        # Cancel receive task
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        
        print("✅ Test completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
