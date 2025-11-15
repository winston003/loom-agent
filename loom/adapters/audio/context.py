"""Multi-turn conversation context management and compression.

Implements User Story 3 (T079-T081) requirements:
- Context assembly from conversation history
- Automatic compression after 10 turns
- Per-speaker context isolation
- Integration with Loom's ContextAssembler

Reference: specs/002-xiaozhi-voice-adapter/tasks.md Phase 5
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict

from loom.core.structured_logger import get_logger
from loom.core.context_assembly import ContextAssembler, ComponentPriority
from loom.adapters.audio.models import ConversationTurn

logger = get_logger("audio.context")


class ConversationContextManager:
    """Manage multi-turn conversation context for audio sessions.
    
    Features:
    - Assembles context from conversation history
    - Compresses old turns (keeps recent 5 + summarizes older)
    - Per-speaker context isolation
    - Token budget management (max 2000 tokens)
    
    Args:
        max_context_tokens: Maximum tokens for conversation context (default: 2000)
        compression_threshold: Turn count to trigger compression (default: 10)
        keep_recent_turns: Number of recent turns to keep uncompressed (default: 5)
    """

    def __init__(
        self,
        max_context_tokens: int = 2000,
        compression_threshold: int = 10,
        keep_recent_turns: int = 5,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold
        self.keep_recent_turns = keep_recent_turns
        
        # Speaker-specific context assemblers (lazy initialization)
        self._assemblers: Dict[str, ContextAssembler] = {}

    def get_or_create_assembler(self, speaker_id: str) -> ContextAssembler:
        """Get or create ContextAssembler for specific speaker.
        
        Implements per-speaker context isolation (T082).
        
        Args:
            speaker_id: Unique speaker identifier
        
        Returns:
            ContextAssembler instance for this speaker
        """
        if speaker_id not in self._assemblers:
            self._assemblers[speaker_id] = ContextAssembler(
                max_tokens=self.max_context_tokens
            )
            logger.debug(
                "Created context assembler for speaker",
                speaker_id=speaker_id,
                max_tokens=self.max_context_tokens,
            )
        return self._assemblers[speaker_id]

    def assemble_context(
        self,
        conversation_history: List[ConversationTurn],
        speaker_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        compress: bool = True,
    ) -> str:
        """Assemble conversation context from history.
        
        Implements T079 (context assembly) and T080 (compression strategy).
        
        Context structure:
        1. System prompt (if provided)
        2. Compressed old turns (if > compression_threshold)
        3. Recent full turns (last N turns)
        
        Args:
            conversation_history: List of conversation turns
            speaker_id: Speaker ID for per-speaker assembly (optional)
            system_prompt: System instructions to prepend (optional)
            compress: Whether to apply compression (default: True)
        
        Returns:
            Assembled context string ready for LLM injection
        """
        start = time.time()
        
        # Filter history by speaker if specified (T082 isolation)
        if speaker_id:
            filtered_history = [
                turn for turn in conversation_history
                if turn.speaker_id == speaker_id
            ]
            logger.debug(
                "Filtered conversation history by speaker",
                speaker_id=speaker_id,
                total_turns=len(conversation_history),
                filtered_turns=len(filtered_history),
            )
        else:
            filtered_history = conversation_history
        
        turn_count = len(filtered_history)
        
        # Early return if no history
        if turn_count == 0:
            return system_prompt or ""
        
        # Build context components
        components = []
        
        # Component 1: System prompt (critical priority)
        if system_prompt:
            components.append({
                "name": "system_prompt",
                "content": system_prompt,
                "priority": ComponentPriority.CRITICAL,
                "truncatable": False,
            })
        
        # Component 2: Compressed old turns (if needed)
        if compress and turn_count > self.compression_threshold:
            old_turns = filtered_history[:-self.keep_recent_turns]
            compressed_summary = self._compress_old_turns(old_turns)
            
            components.append({
                "name": "compressed_history",
                "content": compressed_summary,
                "priority": ComponentPriority.MEDIUM,
                "truncatable": True,
            })
            
            recent_turns = filtered_history[-self.keep_recent_turns:]
            logger.info(
                "Applied conversation compression",
                total_turns=turn_count,
                compressed_turns=len(old_turns),
                recent_turns=len(recent_turns),
            )
        else:
            recent_turns = filtered_history
        
        # Component 3: Recent full turns (high priority)
        recent_context = self._format_recent_turns(recent_turns)
        components.append({
            "name": "recent_conversation",
            "content": recent_context,
            "priority": ComponentPriority.HIGH,
            "truncatable": False,
        })
        
        # Assemble with token budget management
        assembler = self.get_or_create_assembler(speaker_id or "default")
        assembler.clear_components()  # Reset for fresh assembly
        
        for comp in components:
            assembler.add_component(
                name=comp["name"],
                content=comp["content"],
                priority=comp["priority"],
                truncatable=comp["truncatable"],
            )
        
        result = assembler.assemble()
        context = result["assembled_context"]
        
        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Assembled conversation context",
            speaker_id=speaker_id,
            turn_count=turn_count,
            compressed=compress and turn_count > self.compression_threshold,
            token_count=result.get("total_tokens", 0),
            elapsed_ms=elapsed_ms,
        )
        
        # T081: Verify compression time < 50ms
        if elapsed_ms > 50:
            logger.warning(
                "Context assembly exceeded target latency",
                elapsed_ms=elapsed_ms,
                target_ms=50,
                turn_count=turn_count,
            )
        
        return context

    def _compress_old_turns(self, old_turns: List[ConversationTurn]) -> str:
        """Compress old conversation turns into summary.
        
        Implements T080 compression strategy:
        - Extract key information (topics, decisions, user preferences)
        - Remove redundant exchanges
        - Preserve important context (task state, user requests)
        
        Args:
            old_turns: Turns to compress (typically first N-5 turns)
        
        Returns:
            Compressed summary string
        """
        if not old_turns:
            return ""
        
        # Simple compression: Extract user requests and key responses
        summary_lines = ["[Earlier conversation summary]"]
        
        for turn in old_turns:
            # Include user queries
            if turn.user_text:
                summary_lines.append(f"User: {turn.user_text[:100]}")
            
            # Include key agent responses (if contains action confirmations)
            if turn.agent_response:
                # Heuristic: Keep responses with action keywords
                action_keywords = ["完成", "已", "成功", "确认", "好的", "明白"]
                if any(kw in turn.agent_response for kw in action_keywords):
                    summary_lines.append(f"Agent: {turn.agent_response[:100]}")
        
        summary = "\n".join(summary_lines)
        
        logger.debug(
            "Compressed conversation turns",
            original_turns=len(old_turns),
            summary_length=len(summary),
        )
        
        return summary

    def _format_recent_turns(self, recent_turns: List[ConversationTurn]) -> str:
        """Format recent turns as full dialogue.
        
        Args:
            recent_turns: Recent turns to format (typically last 5)
        
        Returns:
            Formatted dialogue string
        """
        if not recent_turns:
            return ""
        
        lines = ["[Recent conversation]"]
        
        for turn in recent_turns:
            lines.append(f"User: {turn.user_text}")
            if turn.agent_response:
                lines.append(f"Agent: {turn.agent_response}")
        
        return "\n".join(lines)

    def should_compress(self, turn_count: int) -> bool:
        """Check if compression should be triggered.
        
        Implements T079 compression trigger logic.
        
        Args:
            turn_count: Current conversation turn count
        
        Returns:
            True if compression should be applied
        """
        return turn_count >= self.compression_threshold

    def clear_speaker_context(self, speaker_id: str) -> None:
        """Clear context assembler for specific speaker.
        
        Used for session cleanup and testing.
        
        Args:
            speaker_id: Speaker ID to clear context for
        """
        if speaker_id in self._assemblers:
            del self._assemblers[speaker_id]
            logger.debug("Cleared context for speaker", speaker_id=speaker_id)
