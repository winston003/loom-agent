"""WebSocket Audio Server

WebSocket server implementation for Xiaozhi device audio streaming with connection
management, authentication, heartbeat, and audio frame handling.

Reference:
- migration/core/websocket_server.py - Original implementation
- specs/002-xiaozhi-voice-adapter/contracts/websocket_protocol.md
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional, Set
from uuid import uuid4

import websockets
from websockets.server import WebSocketServerProtocol

from loom.adapters.audio.models import AudioSession, SessionState
from loom.adapters.audio.websocket.protocol import BinaryMessageType, WebSocketProtocol
from loom.adapters.audio.session import AudioSessionManager
from loom.core.structured_logger import get_logger

logger = get_logger("audio.websocket")


class ConnectionHandler:
    """Handler for individual WebSocket connection.

    Manages session lifecycle, authentication, heartbeat, and audio frame processing
    for a single Xiaozhi device connection.

    Attributes:
        websocket: WebSocket connection
        device_id: Xiaozhi device identifier
        session_id: Current audio session ID
        authenticated: Whether connection is authenticated
        last_heartbeat: Last heartbeat timestamp
    """

    # Timeout configurations (seconds)
    AUTH_TIMEOUT = 5
    HEARTBEAT_INTERVAL = 30
    HEARTBEAT_TIMEOUT = 60
    IDLE_TIMEOUT = 300

    def __init__(
        self,
        websocket: WebSocketServerProtocol,
        server: WebSocketAudioServer,
        on_audio_frame: Optional[Callable[[str, bytes, Dict[str, Any]], Any]] = None,
        on_session_start: Optional[Callable[[AudioSession], Any]] = None,
        on_session_end: Optional[Callable[[str], Any]] = None,
    ):
        """Initialize connection handler.

        Args:
            websocket: WebSocket connection
            server: Parent server instance
            on_audio_frame: Callback for audio frame received (session_id, audio_data, metadata)
            on_session_start: Callback for session start
            on_session_end: Callback for session end
        """
        self.websocket = websocket
        self.server = server
        self.on_audio_frame = on_audio_frame
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end

        self.device_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.authenticated = False
        self.last_heartbeat = time.time()
        self.last_activity = time.time()

        self._heartbeat_task: Optional[asyncio.Task] = None
        self._timeout_task: Optional[asyncio.Task] = None

    async def handle_connection(self) -> None:
        """Main connection handling loop.

        Manages authentication, message routing, heartbeat monitoring, and cleanup.
        """
        try:
            logger.info("New WebSocket connection", remote_addr=self.websocket.remote_address)

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self._timeout_task = asyncio.create_task(self._timeout_monitor())

            # Authentication phase
            await self._authenticate()

            # Message handling loop
            async for message in self.websocket:
                self.last_activity = time.time()

                if isinstance(message, bytes):
                    await self._handle_binary_message(message)
                elif isinstance(message, str):
                    await self._handle_text_message(message)
                else:
                    logger.warning("Unknown message type", message_type=type(message))

        except websockets.exceptions.ConnectionClosed as e:
            logger.info("Connection closed", code=e.code, reason=e.reason, device_id=self.device_id)
        except Exception as e:
            logger.error("Connection error", error=str(e), device_id=self.device_id, exc_info=e)
        finally:
            await self._cleanup()

    async def _authenticate(self) -> None:
        """Authenticate device connection.

        Waits for authentication message within AUTH_TIMEOUT seconds.

        Raises:
            TimeoutError: If authentication not received in time
            ValueError: If authentication fails
        """
        try:
            # Wait for auth message
            auth_msg = await asyncio.wait_for(self.websocket.recv(), timeout=self.AUTH_TIMEOUT)

            if isinstance(auth_msg, str):
                msg = WebSocketProtocol.parse_control_message(auth_msg)

                if msg.get("type") != "auth":
                    raise ValueError(f"Expected auth message, got: {msg.get('type')}")

                device_id = msg.get("device_id")
                if not device_id:
                    raise ValueError("Missing device_id in auth message")

                # TODO: Verify token if authentication is enabled
                # token = msg.get("token")
                # if not self._verify_token(device_id, token):
                #     raise ValueError("Invalid authentication token")

                self.device_id = device_id
                self.authenticated = True

                # Send auth response
                response = WebSocketProtocol.serialize_control_response(
                    "auth_response",
                    session_id="",  # No session yet
                    success=True,
                    data={"device_id": device_id, "server_time": int(time.time() * 1000)},
                )
                await self.websocket.send(response)

                logger.info("Device authenticated", device_id=device_id)
            else:
                raise ValueError("Expected text auth message, got binary")

        except asyncio.TimeoutError:
            logger.error("Authentication timeout")
            await self.websocket.close(code=4001, reason="Authentication timeout")
            raise

    async def _handle_binary_message(self, data: bytes) -> None:
        """Handle binary message (audio data).

        Args:
            data: Raw binary WebSocket message
        """
        try:
            msg_type, metadata, audio_data = WebSocketProtocol.parse_binary_message(data)

            if msg_type == BinaryMessageType.AUDIO:
                session_id = metadata.get("session_id")
                if not session_id:
                    logger.warning("Audio frame missing session_id")
                    return

                # Invoke callback
                if self.on_audio_frame:
                    if asyncio.iscoroutinefunction(self.on_audio_frame):
                        await self.on_audio_frame(session_id, audio_data, metadata)
                    else:
                        self.on_audio_frame(session_id, audio_data, metadata)

                logger.debug(
                    "Audio frame received",
                    session_id=session_id,
                    size=len(audio_data),
                    sequence=metadata.get("sequence", 0),
                )

            else:
                logger.warning("Unsupported binary message type", msg_type=msg_type)

        except ValueError as e:
            logger.error("Invalid binary message", error=str(e))
            await self._send_error("INVALID_MESSAGE_FORMAT", str(e))

    async def _handle_text_message(self, data: str) -> None:
        """Handle text message (control/heartbeat).

        Args:
            data: JSON text message
        """
        try:
            msg = WebSocketProtocol.parse_control_message(data)
            msg_type = msg.get("type")

            if msg_type == "control":
                await self._handle_control_message(msg)
            elif msg_type == "heartbeat":
                await self._handle_heartbeat(msg)
            else:
                logger.warning("Unknown text message type", msg_type=msg_type)

        except ValueError as e:
            logger.error("Invalid text message", error=str(e))
            await self._send_error("INVALID_MESSAGE_FORMAT", str(e))

    async def _handle_control_message(self, msg: Dict[str, Any]) -> None:
        """Handle control message.

        Args:
            msg: Parsed control message dictionary
        """
        action = msg.get("action")
        params = msg.get("params", {})

        if action == "start_session":
            await self._start_session(params)
        elif action == "end_session":
            await self._end_session()
        elif action == "pause_session":
            await self._pause_session()
        elif action == "resume_session":
            await self._resume_session()
        else:
            logger.warning("Unknown control action", action=action)
            await self._send_error("UNKNOWN_ACTION", f"Unknown action: {action}")

    async def _start_session(self, params: Dict[str, Any]) -> None:
        """Start new audio session.

        Args:
            params: Session parameters
        """
        if self.session_id:
            logger.warning("Session already active", session_id=self.session_id)
            await self._send_error("SESSION_ALREADY_ACTIVE", "Session already exists")
            return

        # If server has a session_manager, use it to create a session so state is centralized
        if hasattr(self.server, "session_manager") and isinstance(self.server.session_manager, AudioSessionManager):
            try:
                self.session_id = await self.server.session_manager.create_session(self.device_id or "unknown")

                # Optionally invoke on_session_start callback with the created session
                if self.on_session_start:
                    session = self.server.session_manager.get_session(self.session_id)
                    if asyncio.iscoroutinefunction(self.on_session_start):
                        await self.on_session_start(session)
                    else:
                        self.on_session_start(session)

                # Send acknowledgment
                response = WebSocketProtocol.serialize_control_response(
                    "control_ack",
                    self.session_id,
                    success=True,
                    data={"action": "start_session", "session_id": self.session_id},
                )
                await self.websocket.send(response)

                logger.info("Session started (via session_manager)", session_id=self.session_id, device_id=self.device_id)
                return
            except Exception as e:
                logger.error("Failed to create session via session_manager", error=str(e), exc_info=e)

        # Fallback: create local session
        self.session_id = str(uuid4())
        session = AudioSession(
            session_id=self.session_id,
            device_id=self.device_id or "unknown",
            state=SessionState.LISTENING,
        )

        # Invoke callback
        if self.on_session_start:
            if asyncio.iscoroutinefunction(self.on_session_start):
                await self.on_session_start(session)
            else:
                self.on_session_start(session)

        # Send acknowledgment
        response = WebSocketProtocol.serialize_control_response(
            "control_ack", self.session_id, success=True, data={"action": "start_session", "session_id": self.session_id}
        )
        await self.websocket.send(response)

        logger.info("Session started", session_id=self.session_id, device_id=self.device_id)

    async def _end_session(self) -> None:
        """End current audio session."""
        if not self.session_id:
            logger.warning("No active session to end")
            return

        session_id = self.session_id
        # If server has a session_manager, delegate closing
        if hasattr(self.server, "session_manager") and isinstance(self.server.session_manager, AudioSessionManager):
            try:
                await self.server.session_manager.close_session(session_id)
            except Exception as e:
                logger.warning("session_manager.close_session failed", error=str(e), session_id=session_id)

        # Invoke callback
        if self.on_session_end:
            if asyncio.iscoroutinefunction(self.on_session_end):
                await self.on_session_end(session_id)
            else:
                self.on_session_end(session_id)

        # Clear session
        self.session_id = None

        # Send acknowledgment
        response = WebSocketProtocol.serialize_control_response(
            "control_ack", session_id, success=True, data={"action": "end_session"}
        )
        await self.websocket.send(response)

        logger.info("Session ended", session_id=session_id, device_id=self.device_id)

    async def _pause_session(self) -> None:
        """Pause current session (placeholder)."""
        logger.info("Session pause requested", session_id=self.session_id)
        # TODO: Implement session pause logic

    async def _resume_session(self) -> None:
        """Resume paused session (placeholder)."""
        logger.info("Session resume requested", session_id=self.session_id)
        # TODO: Implement session resume logic

    async def _handle_heartbeat(self, msg: Dict[str, Any]) -> None:
        """Handle heartbeat message.

        Args:
            msg: Heartbeat message dictionary
        """
        self.last_heartbeat = time.time()

        # Send heartbeat acknowledgment
        response = WebSocketProtocol.serialize_control_response(
            "heartbeat_ack",
            session_id=self.session_id or "",
            success=True,
            data={"server_status": "healthy", "timestamp": int(time.time() * 1000)},
        )
        await self.websocket.send(response)

        logger.debug("Heartbeat received", device_id=self.device_id)

    async def _heartbeat_monitor(self) -> None:
        """Monitor heartbeat and close connection if timeout."""
        while True:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

            if time.time() - self.last_heartbeat > self.HEARTBEAT_TIMEOUT:
                logger.warning("Heartbeat timeout", device_id=self.device_id)
                await self.websocket.close(code=4002, reason="Heartbeat timeout")
                break

    async def _timeout_monitor(self) -> None:
        """Monitor idle timeout."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds

            if time.time() - self.last_activity > self.IDLE_TIMEOUT:
                logger.warning("Idle timeout", device_id=self.device_id)
                await self.websocket.close(code=4003, reason="Idle timeout")
                break

    async def _send_error(self, error_code: str, message: str) -> None:
        """Send error message to client.

        Args:
            error_code: Error code
            message: Error message
        """
        error_msg = WebSocketProtocol.serialize_error_message(self.session_id or "", error_code, message)
        await self.websocket.send(error_msg)

    async def _cleanup(self) -> None:
        """Cleanup resources on connection close."""
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._timeout_task:
            self._timeout_task.cancel()

        # End active session
        if self.session_id and self.on_session_end:
            if asyncio.iscoroutinefunction(self.on_session_end):
                await self.on_session_end(self.session_id)
            else:
                self.on_session_end(self.session_id)

        logger.info("Connection cleanup complete", device_id=self.device_id)


class WebSocketAudioServer:
    """WebSocket server for Xiaozhi audio streaming.

    Manages multiple device connections, session routing, and audio frame distribution.

    Example:
        >>> server = WebSocketAudioServer(
        ...     host="0.0.0.0",
        ...     port=8000,
        ...     on_audio_frame=handle_audio,
        ... )
        >>> await server.start()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_connections: int = 100,
        on_audio_frame: Optional[Callable[[str, bytes, Dict[str, Any]], Any]] = None,
        on_session_start: Optional[Callable[[AudioSession], Any]] = None,
        on_session_end: Optional[Callable[[str], Any]] = None,
        session_manager: Optional[AudioSessionManager] = None,
    ):
        """Initialize WebSocket audio server.

        Args:
            host: Server listen address
            port: Server listen port
            max_connections: Maximum concurrent connections
            on_audio_frame: Callback for audio frame received
            on_session_start: Callback for session start
            on_session_end: Callback for session end
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        # Session manager (optional). If provided, it will be used to create/close sessions
        self.session_manager = session_manager

        # If a session_manager is provided and no explicit on_audio_frame callback
        # was supplied, use the manager's handler as the default audio frame handler.
        if not on_audio_frame and self.session_manager is not None:
            self.on_audio_frame = self.session_manager.handle_audio_frame
        else:
            self.on_audio_frame = on_audio_frame

        self.on_session_start = on_session_start
        self.on_session_end = on_session_end

        self.active_connections: Set[ConnectionHandler] = set()
        self._server: Optional[websockets.WebSocketServer] = None

    async def start(self) -> None:
        """Start WebSocket server.

        Runs indefinitely until interrupted.
        """
        logger.info("Starting WebSocket server", host=self.host, port=self.port)

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            subprotocols=["xiaozhi-audio-v1"],
            process_request=self._http_health_check,
        )

        logger.info("WebSocket server started", host=self.host, port=self.port)

        # Run until interrupted
        await asyncio.Future()

    async def stop(self) -> None:
        """Stop WebSocket server and close all connections."""
        logger.info("Stopping WebSocket server")

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Close all active connections
        for handler in list(self.active_connections):
            await handler.websocket.close()

        logger.info("WebSocket server stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle new WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        # Check max connections
        if len(self.active_connections) >= self.max_connections:
            logger.warning("Max connections exceeded", count=len(self.active_connections))
            await websocket.close(code=4004, reason="Max connections exceeded")
            return

        # Create handler
        handler = ConnectionHandler(
            websocket,
            self,
            on_audio_frame=self.on_audio_frame,
            on_session_start=self.on_session_start,
            on_session_end=self.on_session_end,
        )

        self.active_connections.add(handler)
        try:
            await handler.handle_connection()
        finally:
            self.active_connections.discard(handler)

    async def _http_health_check(self, path: str, request_headers: Any) -> Optional[tuple]:
        """HTTP health check endpoint.

        Args:
            path: Request path
            request_headers: Request headers

        Returns:
            HTTP response tuple or None to continue WebSocket upgrade
        """
        # Check if WebSocket upgrade request
        if request_headers.get("Connection", "").lower() == "upgrade":
            return None  # Allow WebSocket upgrade

        # Return health check response for HTTP requests
        return (200, {}, b"Server is running\n")

    async def broadcast(self, message: bytes | str, exclude: Optional[Set[str]] = None) -> None:
        """Broadcast message to all connected devices.

        Args:
            message: Binary or text message
            exclude: Set of device IDs to exclude from broadcast
        """
        exclude = exclude or set()

        tasks = []
        for handler in self.active_connections:
            if handler.device_id and handler.device_id not in exclude:
                tasks.append(handler.websocket.send(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_session_count(self) -> int:
        """Get number of active sessions.

        Returns:
            Active session count
        """
        return sum(1 for h in self.active_connections if h.session_id)

    def get_connection_count(self) -> int:
        """Get number of active connections.

        Returns:
            Active connection count
        """
        return len(self.active_connections)
