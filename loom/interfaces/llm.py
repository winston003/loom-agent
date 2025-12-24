"""
LLM Provider Interface
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from loom.protocol.interfaces import LLMProviderProtocol


class LLMResponse(BaseModel):
    """
    Standardized response from an LLM.
    """
    content: str
    tool_calls: list[dict[str, Any]] = []
    token_usage: dict[str, int] | None = None


class LLMProvider(LLMProviderProtocol, ABC):
    """
    Abstract Interface for LLM Backends (OpenAI, Anthropic, Local).
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None
    ) -> LLMResponse:
        """
        Generate a response for a given chat history.
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        """
        Stream the response content.
        """
        pass
