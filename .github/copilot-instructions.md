# Loom Agent - AI Coding Agent Instructions

## Project Overview

**Loom Agent** is a production-ready Python agent framework (v0.0.5) emphasizing **reliability** over features. Core innovation: automatic recursion control, intelligent context management, and event-driven observability. Architecture inspired by Claude Code's tail-recursive (tt) design pattern.

**Key Philosophy:**
- Framework-first: Code registration & composition over configuration
- Progressive complexity: Minimal examples work out-of-box; add features incrementally
- Observable & replaceable: All components (LLM, tools, permissions, callbacks) are pluggable

## Architecture Essentials

### Execution Flow (Tail-Recursive Pattern)

The core is `AgentExecutor.tt()` - a **tail-recursive** execution loop (not iteration-based):

```
User Input → AgentExecutor.tt() → {
  Phase 0: Iteration Start (check recursion control)
  Phase 1: Context Assembly (RAG retrieval if enabled)
  Phase 2: LLM Call (streaming or batch)
  Phase 3: Tool Execution (parallel when safe)
  Phase 4: Recurse (if tools called) OR Finish
}
```

**Critical Design Principles:**
- `tt()` is the ONLY execution method - all others (`run()`, `stream()`) wrap it
- State is **immutable** (`TurnState` dataclass) - no shared mutable state
- No iteration loops - pure recursion with terminal conditions
- Tool results ALWAYS propagate to next turn (Phase 3 guarantee)

### Component Boundaries

```
loom/
├── components/agent.py          # Public API (Agent class)
├── core/agent_executor.py       # Execution engine (tt recursion)
├── core/tool_pipeline.py        # 5-stage tool processing
├── core/permissions.py          # Policy enforcement (allow/deny/ask)
├── core/recursion_control.py    # Loop detection (duplicate/pattern/error)
├── core/events.py               # AgentEvent stream
├── llm/config.py & factory.py   # Multi-provider LLM abstraction
└── interfaces/                  # Contracts for LLM/Tool/Memory/Compressor
```

**Integration Points:**
- RAG: Inject `context_retriever` → called in Phase 1 before LLM
- MCP: Load via `MCPToolRegistry`, wrap as Loom tools
- Callbacks: Subscribe to `AgentEvent` stream for observability

## Developer Workflows

### Environment Setup

```bash
# Install with extras (recommended for development)
pip install -e ".[all]"

# Required for most examples
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional framework shortcuts
export LOOM_PROVIDER=anthropic
export LOOM_MODEL=claude-sonnet-4-20250514
```

### Running Tests

```bash
# Full suite (40 tests)
pytest

# Specific test categories
pytest tests/unit/test_recursion_control.py -v
pytest tests/unit/test_message_passing.py -v
pytest tests/integration/ -v

# With coverage
pytest --cov=loom --cov-report=html
```

### Building & Publishing

```bash
# Build package
poetry build

# Install locally for testing
pip install -e .

# Run type checks
mypy loom/
```

## Critical Patterns & Conventions

### 1. LLM Configuration (Multi-Provider)

**Pattern:** Use `LLMConfig` factory methods, NOT direct instantiation.

```python
# ✅ Correct
config = LLMConfig.anthropic(api_key="...", model="claude-sonnet-4-20250514")
llm = LLMFactory.create(config)

# ❌ Wrong - bypasses capability detection
llm = AnthropicLLM(model="claude-sonnet-4-20250514")
```

**Default Models (as of v0.0.5):**
- Anthropic: `claude-sonnet-4-20250514` (updated from 3.5 Sonnet)
- OpenAI: `gpt-4`
- Model capabilities in `loom/llm/registry.py` → `ANTHROPIC_MODELS`

### 2. Tool Creation (@tool Decorator)

**Pattern:** Use `@tool()` decorator - it auto-generates Pydantic schema from type hints.

```python
from loom import tool

@tool(description="Add two numbers")
async def add(a: int, b: int) -> int:
    """Add two integers"""
    return a + b

# Usage
agent = loom.agent(provider="openai", model="gpt-4", tools=[add()])
```

**Key Details:**
- Sync functions auto-wrapped in async executor
- `concurrency_safe=True` by default (parallel execution enabled)
- Schema inference happens in `loom/tooling.py:_infer_args_schema()`

### 3. Agent Creation (3 Methods)

```python
# Method 1: Direct LLM instance
llm = OpenAILLM(model="gpt-4")
agent = loom.agent(llm=llm, tools=[...])

# Method 2: Provider + model strings
agent = loom.agent(provider="anthropic", model="claude-sonnet-4-20250514")

# Method 3: From environment
agent = loom.agent_from_env()  # Reads LOOM_PROVIDER/LOOM_MODEL
```

### 4. Event-Driven Streaming

**Pattern:** Prefer `stream()` over `run()` for visibility. Events are the debugging surface.

```python
async for event in agent.stream(prompt):
    match event.type:
        case AgentEventType.RECURSION_TERMINATED:
            # Loop detected - check event.metadata['reason']
        case AgentEventType.COMPRESSION_APPLIED:
            # Context compressed - check token savings
        case AgentEventType.TOOL_RESULT:
            # Tool executed - check event.tool_result
        case AgentEventType.AGENT_FINISH:
            # Final answer
```

**Event Types (see `loom/core/events.py`):**
- `ITERATION_START/END` - Recursion boundaries
- `LLM_DELTA` - Streaming tokens
- `TOOL_EXECUTION_START/RESULT/ERROR` - Tool lifecycle
- `RECURSION_TERMINATED` - Loop detection triggered
- `COMPRESSION_APPLIED` - Context management

### 5. Permissions & Safety

**Pattern:** Use `permission_policy` dict for declarative control, `safe_mode` for interactive.

```python
agent = loom.agent(
    provider="openai",
    model="gpt-4",
    tools=[read_file(), write_file()],
    permission_policy={
        "read_file": "allow",
        "write_file": "ask",  # Prompt user
        "dangerous_tool": "deny"
    },
    safe_mode=True,  # All tools require approval by default
    ask_handler=lambda tool, args: input(f"Allow {tool}? y/n: ").lower() == "y"
)
```

**Persistence:** First approval auto-saves to `~/.loom/config.json` → `allowed_tools` list.

### 6. Recursion Control (Automatic)

**Default Thresholds (in `RecursionMonitor`):**
- Max iterations: 50
- Duplicate tool threshold: 3 (same tool called 3+ times)
- Error threshold: 0.3 (30% tool failure rate)
- Pattern detection: Sliding window for repeated outputs

**Override:**
```python
from loom.core.recursion_control import RecursionMonitor

monitor = RecursionMonitor(
    max_iterations=30,
    duplicate_threshold=2,
    error_threshold=0.2
)
agent = loom.agent(llm=llm, tools=tools, recursion_monitor=monitor)
```

### 7. Sub-Agents (Agent Packs)

**Pattern:** Register specs programmatically, reference via `agent_ref()`.

```python
from loom import AgentSpec, register_agent, agent_ref
from loom.builtin.tools.task import TaskTool

# 1. Register spec
register_agent(AgentSpec(
    agent_type="code-explorer",
    description="Search and understand code structure",
    tools=["glob", "grep", "read_file"],
    model_name="gpt-4o",
    system_instructions="You are a code explorer..."
))

# 2. Create main agent with TaskTool
task = TaskTool(agent_factory=lambda **kw: loom.agent(...))
main = loom.agent(provider="openai", model="gpt-4", tools=[task])

# 3. Reference spec
await task.run_ref(
    prompt="List Python files",
    agent=agent_ref("code-explorer")
)
```

**Behavior:** Injects system prompt, applies tool whitelist (deny-by-default policy), optional model override.

### 8. RAG Integration (Context Retrieval)

**Pattern:** Use `ContextRetriever` wrapper around your retriever for automatic document injection.

```python
from loom.core.context_retriever import ContextRetriever
from loom.retrieval import EmbeddingRetriever
from loom.builtin.embeddings import OpenAIEmbedding
from loom.builtin.retriever import FAISSVectorStore

# Setup retriever
embedding_retriever = EmbeddingRetriever(
    embedding=OpenAIEmbedding(api_key="..."),
    vector_store=FAISSVectorStore(dimension=1536),
    domain_adapter=my_adapter
)
await embedding_retriever.initialize()

# Wrap for agent integration
context_retriever = ContextRetriever(
    retriever=embedding_retriever,
    top_k=3,
    auto_retrieve=True,  # Automatic retrieval before each LLM call
    inject_as="system"   # or "user" to prepend to user message
)

# Agent automatically retrieves context
agent = loom.agent(
    provider="openai",
    model="gpt-4",
    context_retriever=context_retriever
)
```

**Key Details:**
- RAG retrieval happens in **Phase 1** (Context Assembly) before LLM call
- Documents injected via `ContextAssembler` with priority system
- Two injection modes: `inject_as="system"` (separate message) or `"user"` (prepend)
- See `examples/retrieval/complete_example.py` for full workflow

### 9. MCP (Model Context Protocol) Integration

**Pattern:** Load MCP servers as Loom tools via `MCPToolRegistry`.

```python
from loom.mcp import MCPToolRegistry

# Auto-discover local MCP servers
registry = MCPToolRegistry()
await registry.discover_local_servers()

# Load specific servers
tools = await registry.load_servers(["filesystem", "github"])

# Use in agent
agent = loom.agent(
    provider="openai",
    model="gpt-4",
    tools=tools  # MCP tools auto-wrapped
)

result = await agent.run("Read config.json and create GitHub issue")
await registry.close()
```

**Available MCP Servers:**
- `@modelcontextprotocol/server-filesystem` - File operations
- `@modelcontextprotocol/server-github` - GitHub API
- `@modelcontextprotocol/server-postgres` - Database queries
- Tool wrapping happens in `loom/mcp/tool_adapter.py`

### 10. Custom Task Handlers (Advanced)

**Pattern:** Extend `TaskHandler` for domain-specific recursion guidance.

```python
from loom.core.agent_executor import TaskHandler

class SQLTaskHandler(TaskHandler):
    def can_handle(self, task: str) -> bool:
        return any(kw in task.lower() for kw in ["sql", "query", "database"])
    
    def generate_guidance(self, original_task: str, 
                         result_analysis: dict, 
                         recursion_depth: int) -> str:
        has_data = result_analysis.get("has_data", False)
        if has_data:
            return f"Based on retrieved data, generate SQL for: {original_task}"
        return f"Retrieve table schema first for: {original_task}"

# Use in executor
executor = AgentExecutor(
    llm=llm,
    tools=tools,
    task_handlers=[SQLTaskHandler(), AnalysisTaskHandler()]
)
```

**Key Details:**
- Injected during recursion to guide LLM behavior
- See `examples/custom_task_handlers.py` for patterns
- Default handlers in `loom/core/agent_executor.py`

## Common Pitfalls

1. **Modifying state during recursion** → Use immutable `TurnState`, return new instances
2. **Directly calling LLM without `AgentExecutor`** → Bypasses safety, events, context management
3. **Ignoring `AgentEventType.RECURSION_TERMINATED`** → Agent may have stopped due to loop, not completion
4. **Forgetting `async`/`await`** → All execution is async; sync tools auto-wrapped
5. **Not checking tool result propagation** → Phase 3 guarantees delivery, but verify in events
6. **Hardcoding model names** → Use `LLMConfig` constants or environment variables

## Testing Guidelines

- **Unit tests:** Mock `BaseLLM` and `BaseTool` (see `tests/unit/test_streaming_api.py`)
- **Integration tests:** Use real LLM with small prompts (see `tests/integration/`)
- **Recursion tests:** Verify termination conditions (`test_recursion_control.py`)
- **Event tests:** Check event ordering and metadata (`test_agent_events.py`)

**Mock Example:**
```python
from loom.builtin.llms import MockLLM

llm = MockLLM(responses=[
    {"content": "Using calculator", "tool_calls": [...]},
    {"content": "Result is 42", "tool_calls": []}
])
```

## Advanced Integration Patterns

### Context Assembly Priority System

**Location:** `loom/core/context_assembly.py`

Components are assembled by priority to prevent token overflow:

```python
from loom.core.context_assembly import ContextAssembler, ComponentPriority

assembler = ContextAssembler(max_tokens=8000)

# Priority order (highest to lowest):
assembler.add_component(
    "base_instructions", 
    "System prompt...",
    priority=ComponentPriority.CRITICAL,  # Never truncated
    truncatable=False
)

assembler.add_component(
    "retrieved_docs",
    "RAG context...",
    priority=ComponentPriority.HIGH,      # Truncated if needed
    truncatable=True
)

assembler.add_component(
    "tool_definitions",
    "Tool specs...",
    priority=ComponentPriority.MEDIUM,    # Lower priority
    truncatable=False
)

result = assembler.assemble()  # Respects token budget
```

**Critical Fix (v0.0.5):** RAG context no longer overwritten by system prompts.

### Tool Execution Pipeline (5 Stages)

**Location:** `loom/core/tool_pipeline.py`

All tool calls pass through these stages:

1. **Discover** - Parse tool calls from LLM response
2. **Validate** - Check arguments against Pydantic schema
3. **Authorize** - Apply permission policy (allow/deny/ask)
4. **Execute** - Run tools (parallel when `concurrency_safe=True`)
5. **Format** - Convert results to LLM-compatible messages

**Customization Points:**
- Override `PermissionManager.check_permission()` for custom auth
- Extend `Scheduler` for custom concurrency control
- Hook into events: `TOOL_EXECUTION_START`, `TOOL_RESULT`, `TOOL_ERROR`

### Memory & Compression

**Pattern:** Inject custom memory/compressor for context management.

```python
from loom.interfaces.memory import BaseMemory
from loom.interfaces.compressor import BaseCompressor

class CustomMemory(BaseMemory):
    async def add_message(self, message: Message) -> None:
        # Custom storage logic (DB, Redis, etc.)
        pass
    
    async def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        # Custom retrieval logic
        pass

class CustomCompressor(BaseCompressor):
    async def compress(self, messages: List[Message]) -> List[Message]:
        # Custom compression (summarization, pruning, etc.)
        pass

agent = loom.agent(
    llm=llm,
    memory=CustomMemory(),
    compressor=CustomCompressor(),
    max_context_tokens=8000  # Triggers compression
)
```

**Built-in Options:**
- `InMemoryMemory` - Default (session-only)
- `SimpleCompressor` - Basic token reduction
- See `loom/builtin/memory/` and `loom/builtin/compression/`

## Key Files for Common Tasks

| Task | File |
|------|------|
| Add new LLM provider | `loom/llm/factory.py`, `loom/builtin/llms/` |
| Modify recursion logic | `loom/core/recursion_control.py` |
| Add event type | `loom/core/events.py` (AgentEventType enum) |
| Tool execution pipeline | `loom/core/tool_pipeline.py` (5 stages) |
| Context compression | `loom/core/context_assembly.py` |
| RAG integration | `loom/core/context_retriever.py` |
| MCP integration | `loom/mcp/registry.py`, `loom/mcp/tool_adapter.py` |
| Custom task handlers | `loom/core/agent_executor.py` (TaskHandler base) |
| Vector stores | `loom/builtin/retriever/` (FAISS, ChromaDB) |
| Domain adapters | `loom/retrieval/` (SimpleDomainAdapter) |

## Debugging & Observability

### Event Monitoring for Debugging

**Pattern:** Collect events for post-execution analysis.

```python
from loom.core.events import AgentEventType

# Collect all events
events = []
async for event in agent.stream(prompt):
    events.append(event)

# Analyze execution
iterations = [e for e in events if e.type == AgentEventType.ITERATION_START]
tool_calls = [e for e in events if e.type == AgentEventType.TOOL_RESULT]
errors = [e for e in events if e.type == AgentEventType.ERROR]
terminations = [e for e in events if e.type == AgentEventType.RECURSION_TERMINATED]

print(f"Iterations: {len(iterations)}")
print(f"Tool calls: {len(tool_calls)}")
print(f"Errors: {len(errors)}")
if terminations:
    print(f"Terminated: {terminations[0].metadata['reason']}")
```

### Callbacks for Production Monitoring

**Pattern:** Implement `BaseCallback` for metrics/logging.

```python
from loom.callbacks.base import BaseCallback

class ProductionCallback(BaseCallback):
    async def on_agent_start(self, **kwargs):
        # Log to APM/metrics system
        pass
    
    async def on_tool_start(self, tool_name: str, **kwargs):
        # Track tool usage
        pass
    
    async def on_error(self, error: Exception, **kwargs):
        # Alert on failures
        pass

agent = loom.agent(
    provider="openai",
    model="gpt-4",
    callbacks=[ProductionCallback(), MetricsCollector()]
)
```

**Built-in Callbacks:**
- `MetricsCollector` - Usage statistics (`agent.get_metrics()`)
- See `loom/callbacks/` for more

### Common Debugging Scenarios

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| Agent stuck in loop | Check `RECURSION_TERMINATED` events | Adjust `RecursionMonitor` thresholds |
| Tools not executing | Check `TOOL_ERROR` events | Verify tool schema & permissions |
| Context overflow | Check token counts in events | Enable compression or reduce `max_context_tokens` |
| RAG not working | Check Phase 1 events | Verify `context_retriever` injection mode |
| Slow execution | Track event timestamps | Identify bottleneck (LLM, tools, retrieval) |

## Project-Specific Terminology

- **tt (Tail-Tail)**: Core recursive execution method (not "tool-tool")
- **TurnState**: Immutable state snapshot for each recursion level
- **AgentEvent**: Observable execution events (not logs)
- **RecursionMonitor**: Loop detection subsystem (Phase 2 feature)
- **ToolExecutionPipeline**: 5-stage tool processing (Discover → Validate → Authorize → Execute → Format)
- **ContextRetriever**: RAG document injection point (Phase 1)

## Examples to Reference

- `examples/agent_events_demo.py` - Event handling patterns
- `examples/recursion_control_demo.py` - Loop detection scenarios
- `examples/message_passing_demo.py` - Context management
- `examples/streaming_example.py` - Stream API usage
- `examples/retrieval/complete_example.py` - RAG integration

## Version-Specific Notes (v0.0.5)

- **Claude Sonnet 4** now default for Anthropic (`claude-sonnet-4-20250514`)
- Recursion control overhead: < 1ms/iteration
- Context management overhead: < 5ms (without compression)
- Test suite: 40/40 passing
- Breaking changes: None (100% backward compatible with v0.0.3)

---

**Quick Reference Links:**
- Architecture: `docs/LOOM_FRAMEWORK_GUIDE.md`
- User Guide: `docs/USAGE_GUIDE_V0_0_5.md`
- API Reference: `docs/user/api-reference.md`
- Quickstart: `docs/QUICKSTART.md`

## Active Technologies
- Python 3.11+ + Loom Agent Framework（内置 `@tool` 装饰器，Pydantic v2 schema 推导） (001-roi-tool)
- N/A（纯计算，无持久化） (001-roi-tool)
- Python 3.11+（与 Loom Agent v0.0.5 一致） (002-xiaozhi-voice-adapter)

## Recent Changes
- 001-roi-tool: Added Python 3.11+ + Loom Agent Framework（内置 `@tool` 装饰器，Pydantic v2 schema 推导）
