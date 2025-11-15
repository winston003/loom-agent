"""3DSpeaker HTTP client for voiceprint registration and verification.

This module provides the client interface for the 3DSpeaker voiceprint service,
supporting speaker registration, verification, and management operations.

Reference: specs/002-xiaozhi-voice-adapter/contracts/voiceprint_api.md
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx

from loom.core.structured_logger import get_logger

logger = get_logger("audio.voiceprint.client")


class VoiceprintError(Exception):
    """Base exception for voiceprint operations."""
    pass


class VoiceprintRegistrationError(VoiceprintError):
    """Exception raised when voiceprint registration fails."""
    pass


class VoiceprintVerificationError(VoiceprintError):
    """Exception raised when voiceprint verification fails."""
    pass


class ThreeDSpeakerClient:
    """
    Async HTTP client for 3DSpeaker voiceprint service.
    
    Provides speaker registration, verification, and management operations.
    Supports retry logic, error handling, and connection pooling.
    
    Attributes:
        base_url: 3DSpeaker service base URL
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds
        similarity_threshold: Minimum similarity score for verification (0.0-1.0)
        max_voiceprints_per_device: Maximum voiceprints per device
    
    Example:
        ```python
        client = ThreeDSpeakerClient(
            base_url="http://localhost:5000",
            similarity_threshold=0.75
        )
        
        # Register voiceprint
        result = await client.register(
            device_id="device-001",
            display_name="Alice",
            audio_samples=[sample1, sample2, sample3]
        )
        speaker_id = result["speaker_id"]
        
        # Verify speaker
        result = await client.verify(
            device_id="device-001",
            audio_data=test_sample
        )
        if result["verified"]:
            print(f"Verified as: {result['speaker_id']}")
        ```
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 10,
        similarity_threshold: float = 0.75,
        max_voiceprints_per_device: int = 5,
    ):
        """Initialize 3DSpeaker client.
        
        Args:
            base_url: 3DSpeaker service base URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            similarity_threshold: Min similarity score for verification (default: 0.75)
            max_voiceprints_per_device: Max voiceprints per device (default: 5)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.similarity_threshold = similarity_threshold
        self.max_voiceprints_per_device = max_voiceprints_per_device
        self._client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info(
            "3DSpeaker client initialized",
            base_url=self.base_url,
            similarity_threshold=self.similarity_threshold,
            max_voiceprints=self.max_voiceprints_per_device,
        )

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        await self._client.aclose()
        logger.info("3DSpeaker client closed")

    async def register(
        self,
        device_id: str,
        display_name: str,
        audio_samples: List[bytes],
    ) -> Dict[str, Any]:
        """Register a new voiceprint using provided audio samples.
        
        Args:
            device_id: Device ID (used for multi-tenant isolation)
            display_name: User display name (e.g., "Alice", "Bob")
            audio_samples: 3-5 audio samples (2-3 seconds each, PCM 16kHz mono)
            
        Returns:
            Registration result containing:
                - speaker_id: Unique speaker identifier
                - device_id: Device ID
                - display_name: User display name
                - created_at: Registration timestamp
                - sample_count: Number of samples used
                
        Raises:
            VoiceprintRegistrationError: If registration fails
            ValueError: If audio_samples invalid (< 3 or > 5 samples)
        """
        if not audio_samples or len(audio_samples) < 3:
            raise ValueError("At least 3 audio samples required for registration")
        
        if len(audio_samples) > 5:
            raise ValueError("Maximum 5 audio samples allowed for registration")
        
        # Check voiceprint limit
        try:
            existing = await self.list_voiceprints(device_id)
            if len(existing) >= self.max_voiceprints_per_device:
                raise VoiceprintRegistrationError(
                    f"Maximum {self.max_voiceprints_per_device} voiceprints "
                    f"per device reached. Delete existing voiceprints first."
                )
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            # 404 means no existing voiceprints, OK to proceed
        
        try:
            url = f"{self.base_url}/register"
            
            # Prepare multipart form data
            files = {}
            for i, sample in enumerate(audio_samples):
                files[f"sample_{i}"] = (
                    f"sample_{i}.wav",
                    sample,
                    "audio/wav"
                )
            
            data = {
                "device_id": device_id,
                "display_name": display_name,
            }
            
            headers = self._build_headers()
            
            logger.info(
                "Registering voiceprint",
                device_id=device_id,
                display_name=display_name,
                sample_count=len(audio_samples),
            )
            
            start_time = time.time()
            resp = await self._client.post(url, files=files, data=data, headers=headers)
            resp.raise_for_status()
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = resp.json()
            speaker_id = result.get("speaker_id")
            
            logger.info(
                "Voiceprint registered successfully",
                speaker_id=speaker_id,
                device_id=device_id,
                display_name=display_name,
                elapsed_ms=f"{elapsed_ms:.1f}",
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            logger.error(
                "Voiceprint registration failed (HTTP error)",
                status_code=e.response.status_code,
                error=error_detail,
                device_id=device_id,
            )
            raise VoiceprintRegistrationError(
                f"Registration failed: {e.response.status_code} - {error_detail}"
            ) from e
            
        except Exception as e:
            logger.error(
                "Voiceprint registration failed",
                error=str(e),
                device_id=device_id,
            )
            raise VoiceprintRegistrationError(
                f"Registration failed: {e}"
            ) from e

    async def verify(
        self,
        device_id: str,
        audio_data: bytes,
    ) -> Dict[str, Any]:
        """Verify speaker identity using audio sample.
        
        Args:
            device_id: Device ID (for multi-tenant isolation)
            audio_data: Audio sample for verification (PCM 16kHz mono, 2-3s)
            
        Returns:
            Verification result containing:
                - verified: bool (True if similarity >= threshold)
                - speaker_id: str or None (matched speaker ID)
                - display_name: str or None (matched speaker name)
                - similarity: float (0.0-1.0, cosine similarity score)
                - threshold: float (threshold used)
                
        Raises:
            VoiceprintVerificationError: If verification process fails
        """
        try:
            url = f"{self.base_url}/verify"
            
            headers = self._build_headers()
            headers["Content-Type"] = "audio/wav"
            
            params = {
                "device_id": device_id,
                "threshold": self.similarity_threshold,
            }
            
            logger.debug(
                "Verifying speaker",
                device_id=device_id,
                audio_size=len(audio_data),
                threshold=self.similarity_threshold,
            )
            
            start_time = time.time()
            resp = await self._client.post(
                url,
                content=audio_data,
                headers=headers,
                params=params,
            )
            resp.raise_for_status()
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = resp.json()
            verified = result.get("verified", False)
            speaker_id = result.get("speaker_id")
            similarity = result.get("similarity", 0.0)
            
            logger.info(
                "Speaker verification completed",
                verified=verified,
                speaker_id=speaker_id,
                similarity=f"{similarity:.3f}",
                threshold=self.similarity_threshold,
                elapsed_ms=f"{elapsed_ms:.1f}",
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            logger.error(
                "Speaker verification failed (HTTP error)",
                status_code=e.response.status_code,
                error=error_detail,
                device_id=device_id,
            )
            raise VoiceprintVerificationError(
                f"Verification failed: {e.response.status_code} - {error_detail}"
            ) from e
            
        except Exception as e:
            logger.error(
                "Speaker verification failed",
                error=str(e),
                device_id=device_id,
            )
            raise VoiceprintVerificationError(
                f"Verification failed: {e}"
            ) from e

    async def list_voiceprints(self, device_id: str) -> List[Dict[str, Any]]:
        """List all registered voiceprints for a device.
        
        Args:
            device_id: Device ID
            
        Returns:
            List of voiceprint records, each containing:
                - speaker_id: Unique speaker identifier
                - display_name: User display name
                - created_at: Registration timestamp
                - verification_count: Number of successful verifications
                - last_verified_at: Last verification timestamp (or None)
                
        Raises:
            VoiceprintError: If listing fails
        """
        try:
            url = f"{self.base_url}/voiceprints/{device_id}"
            headers = self._build_headers()
            
            resp = await self._client.get(url, headers=headers)
            resp.raise_for_status()
            
            voiceprints = resp.json()
            
            logger.info(
                "Listed voiceprints",
                device_id=device_id,
                count=len(voiceprints),
            )
            
            return voiceprints
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # No voiceprints registered yet
                return []
            
            error_detail = e.response.text
            logger.error(
                "Failed to list voiceprints",
                status_code=e.response.status_code,
                error=error_detail,
                device_id=device_id,
            )
            raise VoiceprintError(
                f"Failed to list voiceprints: {e.response.status_code} - {error_detail}"
            ) from e
            
        except Exception as e:
            logger.error(
                "Failed to list voiceprints",
                error=str(e),
                device_id=device_id,
            )
            raise VoiceprintError(f"Failed to list voiceprints: {e}") from e

    async def delete_voiceprint(
        self,
        device_id: str,
        speaker_id: str,
    ) -> Dict[str, Any]:
        """Delete a registered voiceprint.
        
        Args:
            device_id: Device ID
            speaker_id: Speaker ID to delete
            
        Returns:
            Deletion result containing:
                - success: bool
                - speaker_id: Deleted speaker ID
                - message: Confirmation message
                
        Raises:
            VoiceprintError: If deletion fails
        """
        try:
            url = f"{self.base_url}/voiceprints/{device_id}/{speaker_id}"
            headers = self._build_headers()
            
            logger.info(
                "Deleting voiceprint",
                device_id=device_id,
                speaker_id=speaker_id,
            )
            
            resp = await self._client.delete(url, headers=headers)
            resp.raise_for_status()
            
            result = resp.json()
            
            logger.info(
                "Voiceprint deleted",
                device_id=device_id,
                speaker_id=speaker_id,
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            logger.error(
                "Failed to delete voiceprint",
                status_code=e.response.status_code,
                error=error_detail,
                device_id=device_id,
                speaker_id=speaker_id,
            )
            raise VoiceprintError(
                f"Failed to delete voiceprint: {e.response.status_code} - {error_detail}"
            ) from e
            
        except Exception as e:
            logger.error(
                "Failed to delete voiceprint",
                error=str(e),
                device_id=device_id,
                speaker_id=speaker_id,
            )
            raise VoiceprintError(f"Failed to delete voiceprint: {e}") from e

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers with optional authentication."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
