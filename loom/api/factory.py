"""
Loom SDK: Factory Helpers
"""

from collections.abc import Callable
from typing import Any

from loom.adapters.converters import FunctionToMCP
from loom.api.main import LoomApp
from loom.interfaces.llm import LLMProvider
from loom.interfaces.memory import MemoryInterface
from loom.node.agent import AgentNode
from loom.node.crew import CrewNode
from loom.node.tool import ToolNode
from loom.protocol.mcp import MCPToolDefinition


def Agent(
    app: LoomApp,
    name: str,
    role: str = "Assistant",
    tools: list[ToolNode] | None = None,
    provider: LLMProvider | None = None,
    memory: MemoryInterface | None = None
) -> AgentNode:
    """Helper to create an AgentNode."""
    return AgentNode(
        node_id=name,
        dispatcher=app.dispatcher,
        role=role,
        tools=tools,
        provider=provider,
        memory=memory
    )


def Tool(
    app: LoomApp,
    name: str,
    func: Callable[..., Any],
    description: str | None = None,
    parameters: dict[str, Any] | None = None
) -> ToolNode:
    """
    Helper to create a ToolNode.
    Auto-generates schema from function signature if parameters not provided.
    """

    # Auto-generate definition/schema if not provided
    auto_def = FunctionToMCP.convert(func, name=name)

    final_desc = description or auto_def.description
    final_input_schema = parameters or auto_def.input_schema

    tool_def = MCPToolDefinition(
        name=name,
        description=final_desc,
        input_schema=final_input_schema
    )

    return ToolNode(
        node_id=name,
        dispatcher=app.dispatcher,
        tool_def=tool_def,
        func=func
    )

def Crew(
    app: LoomApp,
    name: str,
    agents: list[AgentNode]
) -> CrewNode:
    """Helper to create a CrewNode."""
    return CrewNode(
        node_id=name,
        dispatcher=app.dispatcher,
        agents=agents
    )
