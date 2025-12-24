"""
Memory Interface
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from loom.protocol.interfaces import MemoryStrategy


class MemoryEntry(BaseModel):
    """
    A single unit of memory.
    """
    role: str
    content: str
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    metadata: dict[str, Any] = Field(default_factory=dict)
    tier: str = "session"  # ephemeral, working, session, longterm


class MemoryInterface(MemoryStrategy, ABC):
    """
    Abstract Base Class for Agent Memory.
    """

    @abstractmethod
    async def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a memory entry."""
        pass

    @abstractmethod
    async def get_context(self, task: str = "") -> str:
        """
        Get full context formatted for the LLM.
        May involve retrieval relevant to the 'task'.
        """
        pass

    @abstractmethod
    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent memory entries as a list of dicts (role/content).
        Useful for Chat History.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear short-term memory."""
        pass
