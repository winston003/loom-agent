"""WebSocket Communication Module

WebSocket server and protocol implementation for Xiaozhi device audio streaming.

Reference: specs/002-xiaozhi-voice-adapter/contracts/websocket_protocol.md
"""

__version__ = "0.0.1"

from loom.adapters.audio.websocket.protocol import (
    MessageType,
    WebSocketProtocol,
    parse_binary_message,
    serialize_audio_message,
)
from loom.adapters.audio.websocket.server import WebSocketAudioServer

__all__ = [
    "WebSocketAudioServer",
    "WebSocketProtocol",
    "MessageType",
    "parse_binary_message",
    "serialize_audio_message",
]
