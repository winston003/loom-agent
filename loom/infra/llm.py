"""
Mock LLM Provider for Testing
"""

from collections.abc import AsyncIterator
from typing import Any

from loom.interfaces.llm import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """
    A Mock Provider that returns canned responses.
    Useful for unit testing and demos without API keys.
    """

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None
    ) -> LLMResponse:
        last_msg = messages[-1]["content"].lower()

        # Simple keywords
        if "search" in last_msg:
            # Simulate Tool Call
            query = last_msg.replace("search", "").strip() or "fractal"
            return LLMResponse(
                content="",
                tool_calls=[{
                    "name": "search",
                    "arguments": {"query": query},
                    "id": "call_mock_123"
                }]
            )

        return LLMResponse(content=f"Mock response to: {last_msg}")

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        yield "Mock "
        yield "stream "
        yield "response."
