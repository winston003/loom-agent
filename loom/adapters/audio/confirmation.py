"""Two-factor confirmation mechanism for critical operations.

This module implements the second verification layer for critical (high-sensitivity)
operations after voiceprint verification:
- Track pending confirmations in session metadata
- Recognize confirmation keywords ("是的"/"确认"/"没错"/"对的")
- Support command repetition confirmation (repeat original command)
- 10-second timeout with retry support
- Maximum 2 retry attempts before cancellation

Reference: specs/002-xiaozhi-voice-adapter/tasks.md T067-T069
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set

from loom.core.structured_logger import get_logger

logger = get_logger("audio.confirmation")


class ConfirmationState(str, Enum):
    """Confirmation state for pending operations."""
    PENDING = "pending"        # Waiting for confirmation
    CONFIRMED = "confirmed"    # User confirmed
    REJECTED = "rejected"      # User rejected
    TIMEOUT = "timeout"        # Confirmation timeout
    CANCELLED = "cancelled"    # Too many retry failures


# Confirmation keywords (T069)
CONFIRMATION_KEYWORDS: Set[str] = {
    "是的", "是", "确认", "没错", "对的", "对", "好的", "好",
    "可以", "同意", "执行", "继续", "确定", "OK", "ok"
}

REJECTION_KEYWORDS: Set[str] = {
    "不", "不是", "取消", "算了", "不用", "停止", "不要", "错了"
}


@dataclass
class PendingConfirmation:
    """Represents a pending confirmation for a critical operation (T068).
    
    Attributes:
        tool_name: Name of the tool/operation requiring confirmation
        arguments: Tool arguments (for context)
        original_command: Original user command text (for repetition check)
        created_at: Timestamp when confirmation was requested
        retry_count: Number of retry attempts (max 2)
        timeout_seconds: Confirmation timeout (default 10s)
        speaker_id: Speaker who initiated the operation
    """
    tool_name: str
    arguments: Dict[str, Any]
    original_command: str
    created_at: float = field(default_factory=time.time)
    retry_count: int = 0
    timeout_seconds: int = 10
    speaker_id: Optional[str] = None
    
    def is_timeout(self) -> bool:
        """Check if confirmation has timed out."""
        elapsed = time.time() - self.created_at
        return elapsed > self.timeout_seconds
    
    def can_retry(self) -> bool:
        """Check if retry is still allowed (max 2 retries)."""
        return self.retry_count < 2
    
    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1


class ConfirmationManager:
    """Manage two-factor confirmations for critical operations (T067-T069).
    
    Features:
    - Track pending confirmations per session
    - Recognize confirmation keywords and command repetition
    - Handle timeout and retry logic
    - Inject confirmation prompts into response
    
    Usage:
        # Step 1: Permission check returns "ask"
        if permission_action == "ask":
            confirmation_mgr.request_confirmation(
                session_id=session_id,
                tool_name="unlock_door",
                arguments={"location": "front_door"},
                original_command="打开门锁",
                speaker_id=speaker_id
            )
            return "敏感操作需要确认：请说'确认'或重复指令"
        
        # Step 2: Next user input
        result = confirmation_mgr.check_confirmation(
            session_id=session_id,
            user_input="确认"
        )
        
        if result.state == ConfirmationState.CONFIRMED:
            # Execute tool
            return execute_tool(result.pending.tool_name, result.pending.arguments)
        elif result.state == ConfirmationState.TIMEOUT:
            return "确认超时，已取消操作"
    """
    
    def __init__(self, default_timeout: int = 10, max_retries: int = 2):
        """Initialize confirmation manager.
        
        Args:
            default_timeout: Default confirmation timeout in seconds
            max_retries: Maximum retry attempts before cancellation
        """
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        
        # session_id -> PendingConfirmation
        self._pending: Dict[str, PendingConfirmation] = {}
        
        logger.info(
            "Confirmation manager initialized",
            default_timeout=default_timeout,
            max_retries=max_retries,
        )
    
    def request_confirmation(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        original_command: str,
        speaker_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Request confirmation for a critical operation (T068).
        
        Args:
            session_id: Session ID
            tool_name: Tool/operation name requiring confirmation
            arguments: Tool arguments
            original_command: Original user command text
            speaker_id: Speaker who initiated the operation
            timeout_seconds: Custom timeout (default: self.default_timeout)
            
        Returns:
            Confirmation prompt message for TTS
        """
        # Create pending confirmation
        pending = PendingConfirmation(
            tool_name=tool_name,
            arguments=arguments,
            original_command=original_command,
            speaker_id=speaker_id,
            timeout_seconds=timeout_seconds or self.default_timeout,
        )
        
        # Store in pending confirmations
        self._pending[session_id] = pending
        
        logger.info(
            "Confirmation requested",
            session_id=session_id,
            tool_name=tool_name,
            original_command=original_command,
            timeout=pending.timeout_seconds,
        )
        
        # Generate confirmation prompt
        prompt = self._generate_confirmation_prompt(tool_name, original_command)
        return prompt
    
    def _generate_confirmation_prompt(self, tool_name: str, original_command: str) -> str:
        """Generate confirmation prompt message.
        
        Args:
            tool_name: Tool name
            original_command: Original command text
            
        Returns:
            Confirmation prompt message
        """
        # Map tool names to user-friendly descriptions
        tool_descriptions = {
            "unlock_door": "打开门锁",
            "delete_photos": "删除照片",
            "delete_all_data": "删除所有数据",
            "transfer_money": "转账",
            "reset_device": "重置设备",
            "change_password": "修改密码",
        }
        
        description = tool_descriptions.get(tool_name, f"执行 {tool_name}")
        
        return f"敏感操作需要确认：{description}。请说'确认'或重复指令'{original_command}'"
    
    def check_confirmation(
        self,
        session_id: str,
        user_input: str,
    ) -> ConfirmationResult:
        """Check if user input confirms pending operation (T069).
        
        Recognition logic:
        1. Check for confirmation keywords ("是的"/"确认"/"没错"/"对的")
        2. Check for command repetition (exact match with original_command)
        3. Check for rejection keywords ("不"/"取消"/"算了")
        4. Check for timeout
        
        Args:
            session_id: Session ID
            user_input: User's ASR transcription text
            
        Returns:
            ConfirmationResult with state and pending confirmation
        """
        # Get pending confirmation
        pending = self._pending.get(session_id)
        
        if not pending:
            # No pending confirmation
            return ConfirmationResult(
                state=ConfirmationState.REJECTED,
                pending=None,
                message="没有待确认的操作",
            )
        
        # Check timeout
        if pending.is_timeout():
            logger.warning(
                "Confirmation timeout",
                session_id=session_id,
                tool_name=pending.tool_name,
                elapsed=time.time() - pending.created_at,
            )
            
            # Remove from pending
            del self._pending[session_id]
            
            return ConfirmationResult(
                state=ConfirmationState.TIMEOUT,
                pending=pending,
                message="确认超时，已取消操作",
            )
        
        # Normalize input for comparison
        user_input_normalized = user_input.strip().lower()
        original_command_normalized = pending.original_command.strip().lower()
        
        # Check rejection keywords first
        if any(keyword in user_input_normalized for keyword in REJECTION_KEYWORDS):
            logger.info(
                "Confirmation rejected by user",
                session_id=session_id,
                tool_name=pending.tool_name,
                user_input=user_input,
            )
            
            # Remove from pending
            del self._pending[session_id]
            
            return ConfirmationResult(
                state=ConfirmationState.REJECTED,
                pending=pending,
                message="操作已取消",
            )
        
        # Check confirmation keywords
        is_keyword_match = any(
            keyword in user_input_normalized for keyword in CONFIRMATION_KEYWORDS
        )
        
        # Check command repetition (exact match)
        is_repetition_match = (
            user_input_normalized == original_command_normalized
            or user_input_normalized in original_command_normalized
            or original_command_normalized in user_input_normalized
        )
        
        if is_keyword_match or is_repetition_match:
            # Confirmation successful
            logger.info(
                "Confirmation successful",
                session_id=session_id,
                tool_name=pending.tool_name,
                user_input=user_input,
                match_type="keyword" if is_keyword_match else "repetition",
            )
            
            # Remove from pending
            del self._pending[session_id]
            
            return ConfirmationResult(
                state=ConfirmationState.CONFIRMED,
                pending=pending,
                message="确认成功，正在执行操作",
            )
        
        else:
            # Confirmation failed - increment retry
            pending.increment_retry()
            
            if not pending.can_retry():
                # Max retries exceeded
                logger.warning(
                    "Confirmation cancelled - max retries exceeded",
                    session_id=session_id,
                    tool_name=pending.tool_name,
                    retry_count=pending.retry_count,
                )
                
                # Remove from pending
                del self._pending[session_id]
                
                return ConfirmationResult(
                    state=ConfirmationState.CANCELLED,
                    pending=pending,
                    message="已取消操作，如需重试请重新发起",
                )
            
            else:
                # Allow retry
                logger.info(
                    "Confirmation unclear - retry allowed",
                    session_id=session_id,
                    tool_name=pending.tool_name,
                    retry_count=pending.retry_count,
                    user_input=user_input,
                )
                
                return ConfirmationResult(
                    state=ConfirmationState.PENDING,
                    pending=pending,
                    message="没听清楚，请再说一次'确认'",
                )
    
    def has_pending_confirmation(self, session_id: str) -> bool:
        """Check if session has pending confirmation.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if pending confirmation exists and not timed out
        """
        pending = self._pending.get(session_id)
        if not pending:
            return False
        
        # Check timeout
        if pending.is_timeout():
            # Cleanup timed out confirmation
            del self._pending[session_id]
            return False
        
        return True
    
    def get_pending_confirmation(self, session_id: str) -> Optional[PendingConfirmation]:
        """Get pending confirmation for session.
        
        Args:
            session_id: Session ID
            
        Returns:
            PendingConfirmation if exists and not timed out, None otherwise
        """
        if not self.has_pending_confirmation(session_id):
            return None
        return self._pending.get(session_id)
    
    def cancel_confirmation(self, session_id: str) -> bool:
        """Cancel pending confirmation for session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if confirmation was cancelled, False if no pending confirmation
        """
        if session_id in self._pending:
            pending = self._pending.pop(session_id)
            logger.info(
                "Confirmation cancelled",
                session_id=session_id,
                tool_name=pending.tool_name,
            )
            return True
        return False
    
    def cleanup_timed_out_confirmations(self) -> int:
        """Cleanup all timed out confirmations (periodic maintenance).
        
        Returns:
            Number of confirmations cleaned up
        """
        timed_out_sessions = [
            session_id
            for session_id, pending in self._pending.items()
            if pending.is_timeout()
        ]
        
        for session_id in timed_out_sessions:
            pending = self._pending.pop(session_id)
            logger.info(
                "Cleaned up timed out confirmation",
                session_id=session_id,
                tool_name=pending.tool_name,
            )
        
        return len(timed_out_sessions)


@dataclass
class ConfirmationResult:
    """Result of confirmation check.
    
    Attributes:
        state: Confirmation state (confirmed/rejected/timeout/cancelled/pending)
        pending: PendingConfirmation object (if available)
        message: User-facing message for TTS
    """
    state: ConfirmationState
    pending: Optional[PendingConfirmation]
    message: str
