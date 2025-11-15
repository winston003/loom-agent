"""Encrypted storage for voiceprint metadata and feature vectors.

Uses AES-GCM (cryptography) to encrypt/decrypt feature bytes and stores them
under `~/.loom/voiceprints/{device_id}/`.

File Structure:
    ~/.loom/voiceprints/
    └── {device_id}/
        ├── {speaker_id}.enc       # Encrypted feature vector
        └── {speaker_id}.json      # Metadata (unencrypted)

Metadata includes:
    - display_name: User display name
    - created_at: Registration timestamp
    - verification_count: Number of successful verifications
    - last_verified_at: Last verification timestamp
    - sample_count: Number of samples used for registration

Reference: specs/002-xiaozhi-voice-adapter/tasks.md T060
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from loom.core.structured_logger import get_logger

logger = get_logger("audio.voiceprint.storage")


class VoiceprintStorage:
    """
    AES-GCM based storage for encrypted voiceprint features and metadata.
    
    Provides encrypted storage for voiceprint feature vectors with
    separate metadata storage for querying and management.
    
    Attributes:
        aesgcm: AES-GCM cipher for encryption
        base_dir: Base directory for voiceprint storage
    """

    def __init__(self, encryption_key: bytes, base_dir: Optional[str] = None):
        """Initialize voiceprint storage.
        
        Args:
            encryption_key: AES key (16/24/32 bytes)
            base_dir: Base directory (default: ~/.loom/voiceprints)
            
        Raises:
            ValueError: If encryption key length invalid
        """
        if len(encryption_key) not in (16, 24, 32):
            raise ValueError("encryption_key must be 16/24/32 bytes for AES")
        
        self.aesgcm = AESGCM(encryption_key)
        self.base_dir = Path(base_dir or Path.home() / ".loom" / "voiceprints")
        os.makedirs(self.base_dir, exist_ok=True)
        
        logger.info("Voiceprint storage initialized", base_dir=str(self.base_dir))

    def _path_for(self, device_id: str, speaker_id: str) -> Path:
        """Get path for encrypted feature file."""
        d = self.base_dir / device_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{speaker_id}.enc"
    
    def _metadata_path_for(self, device_id: str, speaker_id: str) -> Path:
        """Get path for metadata file."""
        d = self.base_dir / device_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{speaker_id}.json"

    def store(self, device_id: str, speaker_id: str, data: bytes) -> str:
        """Encrypt and store feature bytes (legacy method).
        
        Args:
            device_id: Device ID
            speaker_id: Speaker ID
            data: Feature vector bytes
            
        Returns:
            File path where data was stored
        """
        nonce = os.urandom(12)  # AESGCM 96-bit nonce
        ct = self.aesgcm.encrypt(nonce, data, associated_data=None)
        path = self._path_for(device_id, speaker_id)
        
        with open(path, "wb") as f:
            # write: nonce (12) + ciphertext
            f.write(nonce + ct)
        
        logger.debug(
            "Stored encrypted voiceprint",
            device_id=device_id,
            speaker_id=speaker_id,
            size=len(data),
        )
        
        return str(path)
    
    async def save_voiceprint(
        self,
        device_id: str,
        speaker_id: str,
        display_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        feature_data: Optional[bytes] = None,
    ) -> None:
        """Save voiceprint with metadata (T060).
        
        Stores voiceprint metadata as JSON file for easy querying
        and management. Optionally stores encrypted feature vector.
        
        Args:
            device_id: Device ID
            speaker_id: Speaker ID
            display_name: User display name
            metadata: Additional metadata (created_at, sample_count, etc.)
            feature_data: Optional encrypted feature vector
            
        Raises:
            IOError: If file write fails
        """
        try:
            # Prepare metadata
            voiceprint_metadata = {
                "speaker_id": speaker_id,
                "device_id": device_id,
                "display_name": display_name,
                "created_at": metadata.get("created_at") if metadata else time.time(),
                "sample_count": metadata.get("sample_count", 0) if metadata else 0,
                "verification_count": 0,
                "last_verified_at": None,
            }
            
            # Merge additional metadata
            if metadata:
                voiceprint_metadata.update({
                    k: v for k, v in metadata.items()
                    if k not in ["speaker_id", "device_id", "display_name"]
                })
            
            # Save metadata as JSON
            metadata_path = self._metadata_path_for(device_id, speaker_id)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(voiceprint_metadata, f, indent=2)
            
            logger.info(
                "Voiceprint metadata saved",
                device_id=device_id,
                speaker_id=speaker_id,
                display_name=display_name,
                path=str(metadata_path),
            )
            
            # Optionally save encrypted feature data
            if feature_data:
                self.store(device_id, speaker_id, feature_data)
                
        except Exception as e:
            logger.error(
                "Failed to save voiceprint metadata",
                error=str(e),
                device_id=device_id,
                speaker_id=speaker_id,
            )
            raise
    
    async def update_verification_stats(
        self,
        device_id: str,
        speaker_id: str,
    ) -> None:
        """Update verification statistics (T061).
        
        Increments verification_count and updates last_verified_at.
        
        Args:
            device_id: Device ID
            speaker_id: Speaker ID
        """
        try:
            metadata_path = self._metadata_path_for(device_id, speaker_id)
            
            if not metadata_path.exists():
                logger.warning(
                    "Metadata not found for verification stats update",
                    device_id=device_id,
                    speaker_id=speaker_id,
                )
                return
            
            # Load existing metadata
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Update stats
            metadata["verification_count"] = metadata.get("verification_count", 0) + 1
            metadata["last_verified_at"] = time.time()
            
            # Save updated metadata
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(
                "Verification stats updated",
                device_id=device_id,
                speaker_id=speaker_id,
                count=metadata["verification_count"],
            )
            
        except Exception as e:
            logger.error(
                "Failed to update verification stats",
                error=str(e),
                device_id=device_id,
                speaker_id=speaker_id,
            )
            raise

    def load(self, device_id: str, speaker_id: str) -> Optional[bytes]:
        """Load and decrypt stored features, or None if missing.
        
        Args:
            device_id: Device ID
            speaker_id: Speaker ID
            
        Returns:
            Decrypted feature vector bytes or None if not found
        """
        path = self._path_for(device_id, speaker_id)
        if not path.exists():
            return None
        
        with open(path, "rb") as f:
            raw = f.read()
        
        nonce = raw[:12]
        ct = raw[12:]
        return self.aesgcm.decrypt(nonce, ct, associated_data=None)
    
    def load_metadata(
        self,
        device_id: str,
        speaker_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load voiceprint metadata.
        
        Args:
            device_id: Device ID
            speaker_id: Speaker ID
            
        Returns:
            Metadata dict or None if not found
        """
        metadata_path = self._metadata_path_for(device_id, speaker_id)
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_voiceprints(self, device_id: str) -> List[Dict[str, Any]]:
        """List all voiceprints for a device (T070).
        
        Args:
            device_id: Device ID
            
        Returns:
            List of voiceprint metadata dicts
        """
        device_dir = self.base_dir / device_id
        if not device_dir.exists():
            return []
        
        voiceprints = []
        for metadata_file in device_dir.glob("*.json"):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    voiceprints.append(metadata)
            except Exception as e:
                logger.warning(
                    "Failed to load voiceprint metadata",
                    error=str(e),
                    file=str(metadata_file),
                )
        
        logger.debug(
            "Listed voiceprints",
            device_id=device_id,
            count=len(voiceprints),
        )
        
        return voiceprints

    def delete(self, device_id: str, speaker_id: str) -> bool:
        """Delete voiceprint and metadata (T071).
        
        Args:
            device_id: Device ID
            speaker_id: Speaker ID
            
        Returns:
            True if deleted, False if not found
        """
        # Delete encrypted feature file
        feature_path = self._path_for(device_id, speaker_id)
        feature_deleted = False
        if feature_path.exists():
            feature_path.unlink()
            feature_deleted = True
        
        # Delete metadata file
        metadata_path = self._metadata_path_for(device_id, speaker_id)
        metadata_deleted = False
        if metadata_path.exists():
            metadata_path.unlink()
            metadata_deleted = True
        
        if feature_deleted or metadata_deleted:
            logger.info(
                "Voiceprint deleted",
                device_id=device_id,
                speaker_id=speaker_id,
            )
            return True
        
        return False
