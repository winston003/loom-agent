"""WebSocket Protocol Implementation

Binary and text message serialization/deserialization for Xiaozhi audio protocol.

Binary Message Structure:
    ┌────────────────┬────────────────┬────────────────┬──────────────────┐
    │  Header (4B)   │ Meta Length(2B)│  Metadata JSON │  Audio PCM Data  │
    └────────────────┴────────────────┴────────────────┴──────────────────┘

Header Format (4 bytes):
    - Bytes 0-1: Magic Number (0x585A = "XZ")
    - Byte 2: Message Type (0x01=Audio, 0x02=Control, 0x03=Heartbeat)
    - Byte 3: Flags (bit0=HasSpeaker, bit1=IsEnd, bit2-7=Reserved)

Reference: specs/002-xiaozhi-voice-adapter/contracts/websocket_protocol.md
"""

from __future__ import annotations

import json
import struct
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

from loom.adapters.audio.models import AudioPayload, ControlPayload, HeartbeatPayload, MessageType


# Binary protocol constants
MAGIC_NUMBER = 0x585A  # "XZ" in hex
HEADER_SIZE = 4
META_LENGTH_SIZE = 2


class BinaryMessageType(IntEnum):
    """Binary message types (for header byte 2)."""

    AUDIO = 0x01
    CONTROL = 0x02
    HEARTBEAT = 0x03


class MessageFlags(IntEnum):
    """Binary message flags (for header byte 3)."""

    HAS_SPEAKER = 1 << 0  # bit 0
    IS_END = 1 << 1  # bit 1


class WebSocketProtocol:
    """WebSocket protocol handler for audio messages."""

    @staticmethod
    def parse_binary_message(data: bytes) -> Tuple[BinaryMessageType, Dict[str, Any], bytes]:
        """Parse binary WebSocket message.

        Args:
            data: Raw binary data from WebSocket

        Returns:
            Tuple of (message_type, metadata_dict, audio_data)

        Raises:
            ValueError: If message format is invalid

        Example:
            >>> msg_type, metadata, audio = WebSocketProtocol.parse_binary_message(data)
            >>> print(f"Type: {msg_type}, Session: {metadata['session_id']}")
        """
        if len(data) < HEADER_SIZE + META_LENGTH_SIZE:
            raise ValueError(f"Message too short: {len(data)} bytes")

        # Parse header (4 bytes)
        magic, msg_type_byte, flags = struct.unpack(">HBB", data[:HEADER_SIZE])

        # Verify magic number
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Invalid magic number: 0x{magic:04X} (expected 0x{MAGIC_NUMBER:04X})")

        # Parse message type
        try:
            msg_type = BinaryMessageType(msg_type_byte)
        except ValueError:
            raise ValueError(f"Unknown message type: 0x{msg_type_byte:02X}")

        # Parse metadata length (2 bytes)
        meta_length = struct.unpack(">H", data[HEADER_SIZE : HEADER_SIZE + META_LENGTH_SIZE])[0]

        # Extract metadata JSON
        meta_start = HEADER_SIZE + META_LENGTH_SIZE
        meta_end = meta_start + meta_length
        if len(data) < meta_end:
            raise ValueError(f"Metadata length mismatch: expected {meta_length}, got {len(data) - meta_start}")

        metadata_bytes = data[meta_start:meta_end]
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid metadata JSON: {e}")

        # Extract audio data
        audio_data = data[meta_end:]

        # Add flags to metadata
        metadata["has_speaker"] = bool(flags & MessageFlags.HAS_SPEAKER)
        metadata["is_end"] = bool(flags & MessageFlags.IS_END)

        return msg_type, metadata, audio_data

    @staticmethod
    def serialize_audio_message(
        session_id: str,
        audio_data: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = "pcm_s16le",
        speaker_id: Optional[str] = None,
        is_end: bool = False,
        sequence: int = 0,
    ) -> bytes:
        """Serialize audio data to binary WebSocket message.

        Args:
            session_id: Audio session identifier
            audio_data: Raw PCM audio bytes
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
            format: Audio format string
            speaker_id: Optional voiceprint speaker ID
            is_end: Whether this is the last frame
            sequence: Frame sequence number

        Returns:
            Binary message bytes ready for WebSocket transmission

        Example:
            >>> msg = WebSocketProtocol.serialize_audio_message(
            ...     session_id="session-123",
            ...     audio_data=pcm_bytes,
            ...     speaker_id="speaker-456",
            ...     is_end=False
            ... )
            >>> await websocket.send(msg)
        """
        # Build metadata
        metadata = {
            "session_id": session_id,
            "timestamp": int(__import__("time").time() * 1000),
            "sample_rate": sample_rate,
            "channels": channels,
            "format": format,
            "sequence": sequence,
        }
        if speaker_id:
            metadata["speaker_id"] = speaker_id

        metadata_json = json.dumps(metadata).encode("utf-8")
        meta_length = len(metadata_json)

        # Build flags
        flags = 0
        if speaker_id:
            flags |= MessageFlags.HAS_SPEAKER
        if is_end:
            flags |= MessageFlags.IS_END

        # Build header
        header = struct.pack(">HBB", MAGIC_NUMBER, BinaryMessageType.AUDIO, flags)

        # Build metadata length
        meta_length_bytes = struct.pack(">H", meta_length)

        # Combine all parts
        return header + meta_length_bytes + metadata_json + audio_data

    @staticmethod
    def parse_control_message(data: str) -> Dict[str, Any]:
        """Parse control message (JSON text).

        Args:
            data: JSON string from WebSocket text message

        Returns:
            Parsed control message dictionary

        Raises:
            ValueError: If JSON is invalid

        Example:
            >>> msg = WebSocketProtocol.parse_control_message('{"type":"control","action":"start_session"}')
            >>> print(msg["action"])
        """
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid control message JSON: {e}")

    @staticmethod
    def serialize_control_response(
        message_type: str,
        session_id: str,
        success: bool = True,
        data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Serialize control response message.

        Args:
            message_type: Response message type
            session_id: Session identifier
            success: Whether operation succeeded
            data: Additional response data
            error_message: Error message if failed

        Returns:
            JSON string for WebSocket text message

        Example:
            >>> resp = WebSocketProtocol.serialize_control_response(
            ...     "control_ack",
            ...     "session-123",
            ...     success=True,
            ...     data={"status": "started"}
            ... )
            >>> await websocket.send(resp)
        """
        response = {
            "type": message_type,
            "session_id": session_id,
            "timestamp": int(__import__("time").time() * 1000),
            "success": success,
        }
        if data:
            response.update(data)
        if error_message:
            response["error"] = error_message

        return json.dumps(response)

    @staticmethod
    def serialize_error_message(
        session_id: str, error_code: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Serialize error message.

        Args:
            session_id: Session identifier
            error_code: Error code (e.g., 'ASR_FAILED')
            message: Human-readable error message
            details: Additional error details

        Returns:
            JSON error message string

        Example:
            >>> err = WebSocketProtocol.serialize_error_message(
            ...     "session-123",
            ...     "ASR_FAILED",
            ...     "ASR service unavailable"
            ... )
            >>> await websocket.send(err)
        """
        error_msg = {
            "type": "error",
            "session_id": session_id,
            "timestamp": int(__import__("time").time() * 1000),
            "error_code": error_code,
            "message": message,
        }
        if details:
            error_msg["details"] = details

        return json.dumps(error_msg)


# Convenience functions (backward compatibility)
parse_binary_message = WebSocketProtocol.parse_binary_message
serialize_audio_message = WebSocketProtocol.serialize_audio_message
