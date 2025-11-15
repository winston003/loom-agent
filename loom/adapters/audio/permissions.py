"""Audio-specific permission management for voiceprint-based access control.

This module extends Loom's PermissionManager to support speaker-based permissions:
- Read speaker_id from TurnState.metadata (injected by voiceprint verification)
- Define sensitivity levels (critical/medium/low) for operations
- Load configuration from ~/.loom/audio_config.json
- Support two-factor confirmation for critical operations

Reference: specs/002-xiaozhi-voice-adapter/tasks.md T064-T066
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from loom.core.permissions import PermissionManager, PermissionAction
from loom.core.structured_logger import get_logger

logger = get_logger("audio.permissions")


class SensitivityLevel(str, Enum):
    """Operation sensitivity levels (T065)."""
    CRITICAL = "critical"  # Requires two-factor confirmation
    MEDIUM = "medium"      # Requires voiceprint verification
    LOW = "low"            # No verification required


class AudioPermissionManager:
    """Extended permission manager with voiceprint-based access control (T064).
    
    Features:
    - Speaker-based authorization (from TurnState.metadata["speaker_id"])
    - Sensitivity-based permission policies (critical/medium/low)
    - Configuration file support (~/.loom/audio_config.json)
    - Owner vs guest role differentiation
    - Two-factor confirmation tracking (for critical operations)
    
    Configuration file format (T066):
    {
        "owner_speaker_ids": ["speaker_001"],
        "guest_speaker_ids": ["speaker_002", "speaker_003"],
        "sensitive_operations": {
            "unlock_door": "critical",
            "delete_photos": "critical",
            "turn_off_lights": "medium",
            "send_message": "medium",
            "get_weather": "low"
        },
        "default_sensitivity": "medium"
    }
    
    Usage:
        # In agent execution context
        audio_pm = AudioPermissionManager.from_config()
        
        # Check permission with speaker context
        action = audio_pm.check_permission(
            tool_name="unlock_door",
            speaker_id="speaker_001",  # From TurnState.metadata
            arguments={"location": "front_door"}
        )
        
        if action == PermissionAction.DENY:
            # Reject operation
        elif action == PermissionAction.ASK:
            # Request two-factor confirmation
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        base_permission_manager: Optional[PermissionManager] = None,
    ):
        """Initialize audio permission manager.
        
        Args:
            config_path: Path to audio_config.json (default: ~/.loom/audio_config.json)
            base_permission_manager: Base Loom PermissionManager (optional)
        """
        self.config_path = Path(config_path or Path.home() / ".loom" / "audio_config.json")
        self.base_pm = base_permission_manager
        
        # Load configuration
        self.config = self._load_config()
        
        # Extract configuration sections
        self.owner_speaker_ids = set(self.config.get("owner_speaker_ids", []))
        self.guest_speaker_ids = set(self.config.get("guest_speaker_ids", []))
        self.sensitive_operations = self.config.get("sensitive_operations", {})
        self.default_sensitivity = self.config.get("default_sensitivity", "medium")
        
        logger.info(
            "Audio permission manager initialized",
            config_path=str(self.config_path),
            owners_count=len(self.owner_speaker_ids),
            guests_count=len(self.guest_speaker_ids),
            operations_count=len(self.sensitive_operations),
        )
    
    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> AudioPermissionManager:
        """Create AudioPermissionManager from configuration file.
        
        Args:
            config_path: Optional custom config path
            
        Returns:
            Configured AudioPermissionManager instance
        """
        return cls(config_path=config_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load audio configuration from file (T066).
        
        Returns:
            Configuration dict with default values if file not found
        """
        if not self.config_path.exists():
            logger.warning(
                "Audio config file not found - using defaults",
                path=str(self.config_path),
            )
            return self._create_default_config()
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            logger.info(
                "Audio config loaded",
                path=str(self.config_path),
                owners=len(config.get("owner_speaker_ids", [])),
                guests=len(config.get("guest_speaker_ids", [])),
            )
            
            return config
        except Exception as e:
            logger.error(
                "Failed to load audio config - using defaults",
                error=str(e),
                path=str(self.config_path),
            )
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration (T065)."""
        return {
            "owner_speaker_ids": [],
            "guest_speaker_ids": [],
            "sensitive_operations": {
                # Critical operations (require two-factor confirmation)
                "unlock_door": "critical",
                "delete_photos": "critical",
                "delete_all_data": "critical",
                "authorize_third_party": "critical",
                "change_password": "critical",
                "transfer_money": "critical",
                "reset_device": "critical",
                
                # Medium operations (require voiceprint verification)
                "turn_off_lights": "medium",
                "send_message": "medium",
                "read_messages": "medium",
                "view_calendar": "medium",
                "play_private_photos": "medium",
                "adjust_temperature": "medium",
                
                # Low operations (no verification required)
                "get_weather": "low",
                "get_news": "low",
                "get_time": "low",
                "play_music": "low",
                "set_reminder": "low",
                "adjust_volume": "low",
            },
            "default_sensitivity": "medium",
        }
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info("Audio config saved", path=str(self.config_path))
        except Exception as e:
            logger.error("Failed to save audio config", error=str(e), path=str(self.config_path))
            raise
    
    def get_sensitivity_level(self, tool_name: str) -> SensitivityLevel:
        """Get sensitivity level for a tool/operation (T065).
        
        Args:
            tool_name: Tool or operation name
            
        Returns:
            SensitivityLevel (critical/medium/low)
        """
        level_str = self.sensitive_operations.get(tool_name, self.default_sensitivity)
        try:
            return SensitivityLevel(level_str)
        except ValueError:
            logger.warning(
                "Invalid sensitivity level - using default",
                tool=tool_name,
                level=level_str,
                default=self.default_sensitivity,
            )
            return SensitivityLevel(self.default_sensitivity)
    
    def is_owner(self, speaker_id: Optional[str]) -> bool:
        """Check if speaker is an owner.
        
        Args:
            speaker_id: Speaker ID from voiceprint verification
            
        Returns:
            True if speaker is owner, False otherwise
        """
        if not speaker_id:
            return False
        return speaker_id in self.owner_speaker_ids
    
    def is_guest(self, speaker_id: Optional[str]) -> bool:
        """Check if speaker is a registered guest.
        
        Args:
            speaker_id: Speaker ID from voiceprint verification
            
        Returns:
            True if speaker is guest, False otherwise
        """
        if not speaker_id:
            return False
        return speaker_id in self.guest_speaker_ids
    
    def check_permission(
        self,
        tool_name: str,
        speaker_id: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> PermissionAction:
        """Check permission for tool execution with speaker context (T064).
        
        Permission logic:
        1. Get operation sensitivity level
        2. Check speaker role (owner/guest/anonymous)
        3. Apply permission rules:
           - Critical: Only owner, requires two-factor confirmation → ASK
           - Medium: Owner and guest allowed → ALLOW/DENY based on speaker_id
           - Low: Everyone allowed → ALLOW
        
        Args:
            tool_name: Tool or operation name
            speaker_id: Speaker ID from TurnState.metadata["speaker_id"]
            arguments: Tool arguments (for context)
            
        Returns:
            PermissionAction (ALLOW/DENY/ASK)
        """
        sensitivity = self.get_sensitivity_level(tool_name)
        
        # Determine speaker role
        is_owner = self.is_owner(speaker_id)
        is_guest = self.is_guest(speaker_id)
        is_anonymous = not speaker_id
        
        logger.debug(
            "Checking audio permission",
            tool=tool_name,
            sensitivity=sensitivity.value,
            speaker_id=speaker_id,
            is_owner=is_owner,
            is_guest=is_guest,
            is_anonymous=is_anonymous,
        )
        
        # Apply permission rules based on sensitivity
        if sensitivity == SensitivityLevel.CRITICAL:
            # Critical operations: Only owner + two-factor confirmation
            if is_owner:
                logger.info(
                    "Critical operation requires two-factor confirmation",
                    tool=tool_name,
                    speaker_id=speaker_id,
                )
                return PermissionAction.ASK  # Trigger two-factor confirmation
            else:
                logger.warning(
                    "Critical operation denied - not owner",
                    tool=tool_name,
                    speaker_id=speaker_id,
                    is_guest=is_guest,
                    is_anonymous=is_anonymous,
                )
                return PermissionAction.DENY
        
        elif sensitivity == SensitivityLevel.MEDIUM:
            # Medium operations: Require voiceprint verification (owner or guest)
            if is_owner or is_guest:
                logger.info(
                    "Medium operation allowed - verified speaker",
                    tool=tool_name,
                    speaker_id=speaker_id,
                    role="owner" if is_owner else "guest",
                )
                return PermissionAction.ALLOW
            else:
                logger.warning(
                    "Medium operation denied - no speaker verification",
                    tool=tool_name,
                    is_anonymous=is_anonymous,
                )
                return PermissionAction.DENY
        
        else:  # LOW
            # Low operations: Allow everyone
            logger.debug(
                "Low sensitivity operation allowed",
                tool=tool_name,
                speaker_id=speaker_id,
            )
            return PermissionAction.ALLOW
    
    def add_owner(self, speaker_id: str) -> None:
        """Add speaker as owner.
        
        Args:
            speaker_id: Speaker ID to add
        """
        self.owner_speaker_ids.add(speaker_id)
        self.config["owner_speaker_ids"] = list(self.owner_speaker_ids)
        self.save_config()
        
        logger.info("Owner added", speaker_id=speaker_id)
    
    def add_guest(self, speaker_id: str) -> None:
        """Add speaker as guest.
        
        Args:
            speaker_id: Speaker ID to add
        """
        self.guest_speaker_ids.add(speaker_id)
        self.config["guest_speaker_ids"] = list(self.guest_speaker_ids)
        self.save_config()
        
        logger.info("Guest added", speaker_id=speaker_id)
    
    def remove_speaker(self, speaker_id: str) -> bool:
        """Remove speaker from owners or guests.
        
        Args:
            speaker_id: Speaker ID to remove
            
        Returns:
            True if removed, False if not found
        """
        removed = False
        
        if speaker_id in self.owner_speaker_ids:
            self.owner_speaker_ids.remove(speaker_id)
            self.config["owner_speaker_ids"] = list(self.owner_speaker_ids)
            removed = True
        
        if speaker_id in self.guest_speaker_ids:
            self.guest_speaker_ids.remove(speaker_id)
            self.config["guest_speaker_ids"] = list(self.guest_speaker_ids)
            removed = True
        
        if removed:
            self.save_config()
            logger.info("Speaker removed", speaker_id=speaker_id)
        
        return removed
