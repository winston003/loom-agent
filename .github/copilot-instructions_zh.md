# Loom Agent - AI 编码助手指令

## 项目概述

**Loom Agent** 是一个生产就绪的 Python 智能体框架（v0.0.5），强调**可靠性**而非功能堆砌。核心创新：自动递归控制、智能上下文管理和事件驱动的可观测性。架构灵感源自 Claude Code 的尾递归（tt）设计模式。

**核心理念：**
- 框架优先：通过代码注册和组合而非配置驱动
- 渐进式复杂度：最小示例开箱即用，按需增加功能
- 可观测可替换：所有组件（LLM、工具、权限、回调）均可插拔

## 架构要点

### 执行流程（尾递归模式）

核心是 `AgentExecutor.tt()` - 一个**尾递归**执行循环（非基于迭代）：

```
用户输入 → AgentExecutor.tt() → {
  阶段 0: 迭代开始（检查递归控制）
  阶段 1: 上下文组装（RAG 检索，如果启用）
  阶段 2: LLM 调用（流式或批量）
  阶段 3: 工具执行（安全时并行）
  阶段 4: 递归（如果调用工具）或完成
}
```

**关键设计原则：**
- `tt()` 是唯一的执行方法 - 其他方法（`run()`、`stream()`）都是它的包装
- 状态是**不可变的**（`TurnState` 数据类）- 无共享可变状态
- 没有迭代循环 - 纯递归带终止条件
- 工具结果始终传播到下一轮（阶段 3 保证）

### 组件边界

```
loom/
├── components/agent.py          # 公共 API（Agent 类）
├── core/agent_executor.py       # 执行引擎（tt 递归）
├── core/tool_pipeline.py        # 5 阶段工具处理
├── core/permissions.py          # 策略执行（allow/deny/ask）
├── core/recursion_control.py    # 循环检测（重复/模式/错误）
├── core/events.py               # AgentEvent 流
├── llm/config.py & factory.py   # 多提供商 LLM 抽象
└── interfaces/                  # LLM/Tool/Memory/Compressor 契约
```

**集成点：**
- RAG：注入 `context_retriever` → 在阶段 1（LLM 之前）调用
- MCP：通过 `MCPToolRegistry` 加载，包装为 Loom 工具
- 回调：订阅 `AgentEvent` 流以实现可观测性

## 开发者工作流

### 环境设置

```bash
# 安装所有额外功能（推荐开发使用）
pip install -e ".[all]"

# 大多数示例需要的环境变量
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# 可选的框架快捷方式
export LOOM_PROVIDER=anthropic
export LOOM_MODEL=claude-sonnet-4-20250514
```

### 运行测试

```bash
# 完整测试套件（40 个测试）
pytest

# 特定测试类别
pytest tests/unit/test_recursion_control.py -v
pytest tests/unit/test_message_passing.py -v
pytest tests/integration/ -v

# 带覆盖率
pytest --cov=loom --cov-report=html
```

### 构建与发布

```bash
# 构建包
poetry build

# 本地安装测试
pip install -e .

# 运行类型检查
mypy loom/
```

## 关键模式与约定

### 1. LLM 配置（多提供商）

**模式：** 使用 `LLMConfig` 工厂方法，而非直接实例化。

```python
# ✅ 正确
config = LLMConfig.anthropic(api_key="...", model="claude-sonnet-4-20250514")
llm = LLMFactory.create(config)

# ❌ 错误 - 绕过能力检测
llm = AnthropicLLM(model="claude-sonnet-4-20250514")
```

**默认模型（v0.0.5）：**
- Anthropic: `claude-sonnet-4-20250514`（从 3.5 Sonnet 更新）
- OpenAI: `gpt-4`
- 模型能力在 `loom/llm/registry.py` → `ANTHROPIC_MODELS`

### 2. 工具创建（@tool 装饰器）

**模式：** 使用 `@tool()` 装饰器 - 自动从类型提示生成 Pydantic schema。

```python
from loom import tool

@tool(description="两数相加")
async def add(a: int, b: int) -> int:
    """将两个整数相加"""
    return a + b

# 使用
agent = loom.agent(provider="openai", model="gpt-4", tools=[add()])
```

**关键细节：**
- 同步函数自动包装为异步执行器
- 默认 `concurrency_safe=True`（启用并行执行）
- Schema 推断发生在 `loom/tooling.py:_infer_args_schema()`

### 3. Agent 创建（3 种方法）

```python
# 方法 1: 直接 LLM 实例
llm = OpenAILLM(model="gpt-4")
agent = loom.agent(llm=llm, tools=[...])

# 方法 2: 提供商 + 模型字符串
agent = loom.agent(provider="anthropic", model="claude-sonnet-4-20250514")

# 方法 3: 从环境变量
agent = loom.agent_from_env()  # 读取 LOOM_PROVIDER/LOOM_MODEL
```

### 4. 事件驱动流式处理

**模式：** 优先使用 `stream()` 而非 `run()` 以获得可见性。事件是调试界面。

```python
async for event in agent.stream(prompt):
    match event.type:
        case AgentEventType.RECURSION_TERMINATED:
            # 检测到循环 - 检查 event.metadata['reason']
        case AgentEventType.COMPRESSION_APPLIED:
            # 上下文已压缩 - 检查 token 节省
        case AgentEventType.TOOL_RESULT:
            # 工具已执行 - 检查 event.tool_result
        case AgentEventType.AGENT_FINISH:
            # 最终答案
```

**事件类型（见 `loom/core/events.py`）：**
- `ITERATION_START/END` - 递归边界
- `LLM_DELTA` - 流式 token
- `TOOL_EXECUTION_START/RESULT/ERROR` - 工具生命周期
- `RECURSION_TERMINATED` - 触发循环检测
- `COMPRESSION_APPLIED` - 上下文管理

### 5. 权限与安全

**模式：** 使用 `permission_policy` 字典进行声明式控制，`safe_mode` 用于交互式。

```python
agent = loom.agent(
    provider="openai",
    model="gpt-4",
    tools=[read_file(), write_file()],
    permission_policy={
        "read_file": "allow",
        "write_file": "ask",  # 提示用户
        "dangerous_tool": "deny"
    },
    safe_mode=True,  # 所有工具默认需要批准
    ask_handler=lambda tool, args: input(f"允许 {tool}? y/n: ").lower() == "y"
)
```

**持久化：** 首次批准自动保存到 `~/.loom/config.json` → `allowed_tools` 列表。

### 6. 递归控制（自动）

**默认阈值（在 `RecursionMonitor` 中）：**
- 最大迭代次数：50
- 重复工具阈值：3（同一工具调用 3 次以上）
- 错误阈值：0.3（30% 工具失败率）
- 模式检测：滑动窗口检测重复输出

**覆盖：**
```python
from loom.core.recursion_control import RecursionMonitor

monitor = RecursionMonitor(
    max_iterations=30,
    duplicate_threshold=2,
    error_threshold=0.2
)
agent = loom.agent(llm=llm, tools=tools, recursion_monitor=monitor)
```

### 7. 子 Agent（Agent Packs）

**模式：** 以编程方式注册规格，通过 `agent_ref()` 引用。

```python
from loom import AgentSpec, register_agent, agent_ref
from loom.builtin.tools.task import TaskTool

# 1. 注册规格
register_agent(AgentSpec(
    agent_type="code-explorer",
    description="搜索和理解代码结构",
    tools=["glob", "grep", "read_file"],
    model_name="gpt-4o",
    system_instructions="你是一个代码探索 agent..."
))

# 2. 创建带 TaskTool 的主 agent
task = TaskTool(agent_factory=lambda **kw: loom.agent(...))
main = loom.agent(provider="openai", model="gpt-4", tools=[task])

# 3. 引用规格
await task.run_ref(
    prompt="列出 Python 文件",
    agent=agent_ref("code-explorer")
)
```

**行为：** 注入系统提示，应用工具白名单（默认拒绝策略），可选模型覆盖。

### 8. RAG 集成（上下文检索）

**模式：** 使用 `ContextRetriever` 包装检索器实现自动文档注入。

```python
from loom.core.context_retriever import ContextRetriever
from loom.retrieval import EmbeddingRetriever
from loom.builtin.embeddings import OpenAIEmbedding
from loom.builtin.retriever import FAISSVectorStore

# 设置检索器
embedding_retriever = EmbeddingRetriever(
    embedding=OpenAIEmbedding(api_key="..."),
    vector_store=FAISSVectorStore(dimension=1536),
    domain_adapter=my_adapter
)
await embedding_retriever.initialize()

# 为 agent 集成包装
context_retriever = ContextRetriever(
    retriever=embedding_retriever,
    top_k=3,
    auto_retrieve=True,  # 每次 LLM 调用前自动检索
    inject_as="system"   # 或 "user" 以前置到用户消息
)

# Agent 自动检索上下文
agent = loom.agent(
    provider="openai",
    model="gpt-4",
    context_retriever=context_retriever
)
```

**关键细节：**
- RAG 检索发生在**阶段 1**（上下文组装）LLM 调用之前
- 文档通过 `ContextAssembler` 注入，带优先级系统
- 两种注入模式：`inject_as="system"`（单独消息）或 `"user"`（前置）
- 完整工作流见 `examples/retrieval/complete_example.py`

### 9. MCP（模型上下文协议）集成

**模式：** 通过 `MCPToolRegistry` 将 MCP 服务器加载为 Loom 工具。

```python
from loom.mcp import MCPToolRegistry

# 自动发现本地 MCP 服务器
registry = MCPToolRegistry()
await registry.discover_local_servers()

# 加载特定服务器
tools = await registry.load_servers(["filesystem", "github"])

# 在 agent 中使用
agent = loom.agent(
    provider="openai",
    model="gpt-4",
    tools=tools  # MCP 工具自动包装
)

result = await agent.run("读取 config.json 并创建 GitHub issue")
await registry.close()
```

**可用的 MCP 服务器：**
- `@modelcontextprotocol/server-filesystem` - 文件操作
- `@modelcontextprotocol/server-github` - GitHub API
- `@modelcontextprotocol/server-postgres` - 数据库查询
- 工具包装发生在 `loom/mcp/tool_adapter.py`

### 10. 自定义任务处理器（高级）

**模式：** 扩展 `TaskHandler` 实现特定领域的递归引导。

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
            return f"基于检索到的数据，生成 SQL：{original_task}"
        return f"首先检索表结构：{original_task}"

# 在执行器中使用
executor = AgentExecutor(
    llm=llm,
    tools=tools,
    task_handlers=[SQLTaskHandler(), AnalysisTaskHandler()]
)
```

**关键细节：**
- 在递归期间注入以引导 LLM 行为
- 参考模式见 `examples/custom_task_handlers.py`
- 默认处理器在 `loom/core/agent_executor.py`

## 常见陷阱

1. **在递归期间修改状态** → 使用不可变的 `TurnState`，返回新实例
2. **不通过 `AgentExecutor` 直接调用 LLM** → 绕过安全、事件、上下文管理
3. **忽略 `AgentEventType.RECURSION_TERMINATED`** → Agent 可能因循环而停止，而非完成
4. **忘记 `async`/`await`** → 所有执行都是异步的；同步工具自动包装
5. **不检查工具结果传播** → 阶段 3 保证传递，但需在事件中验证
6. **硬编码模型名称** → 使用 `LLMConfig` 常量或环境变量

## 测试指南

- **单元测试：** Mock `BaseLLM` 和 `BaseTool`（见 `tests/unit/test_streaming_api.py`）
- **集成测试：** 使用真实 LLM 配小提示（见 `tests/integration/`）
- **递归测试：** 验证终止条件（`test_recursion_control.py`）
- **事件测试：** 检查事件顺序和元数据（`test_agent_events.py`）

**Mock 示例：**
```python
from loom.builtin.llms import MockLLM

llm = MockLLM(responses=[
    {"content": "使用计算器", "tool_calls": [...]},
    {"content": "结果是 42", "tool_calls": []}
])
```

## 高级集成模式

### 上下文组装优先级系统

**位置：** `loom/core/context_assembly.py`

组件按优先级组装以防止 token 溢出：

```python
from loom.core.context_assembly import ContextAssembler, ComponentPriority

assembler = ContextAssembler(max_tokens=8000)

# 优先级顺序（从高到低）：
assembler.add_component(
    "base_instructions", 
    "系统提示...",
    priority=ComponentPriority.CRITICAL,  # 永不截断
    truncatable=False
)

assembler.add_component(
    "retrieved_docs",
    "RAG 上下文...",
    priority=ComponentPriority.HIGH,      # 需要时截断
    truncatable=True
)

assembler.add_component(
    "tool_definitions",
    "工具规格...",
    priority=ComponentPriority.MEDIUM,    # 较低优先级
    truncatable=False
)

result = assembler.assemble()  # 遵守 token 预算
```

**关键修复（v0.0.5）：** RAG 上下文不再被系统提示覆盖。

### 工具执行流水线（5 阶段）

**位置：** `loom/core/tool_pipeline.py`

所有工具调用经过这些阶段：

1. **发现** - 从 LLM 响应解析工具调用
2. **验证** - 根据 Pydantic schema 检查参数
3. **授权** - 应用权限策略（allow/deny/ask）
4. **执行** - 运行工具（当 `concurrency_safe=True` 时并行）
5. **格式化** - 将结果转换为 LLM 兼容消息

**自定义点：**
- 覆盖 `PermissionManager.check_permission()` 实现自定义授权
- 扩展 `Scheduler` 实现自定义并发控制
- 钩入事件：`TOOL_EXECUTION_START`、`TOOL_RESULT`、`TOOL_ERROR`

### 内存与压缩

**模式：** 注入自定义内存/压缩器以管理上下文。

```python
from loom.interfaces.memory import BaseMemory
from loom.interfaces.compressor import BaseCompressor

class CustomMemory(BaseMemory):
    async def add_message(self, message: Message) -> None:
        # 自定义存储逻辑（DB、Redis 等）
        pass
    
    async def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        # 自定义检索逻辑
        pass

class CustomCompressor(BaseCompressor):
    async def compress(self, messages: List[Message]) -> List[Message]:
        # 自定义压缩（摘要、修剪等）
        pass

agent = loom.agent(
    llm=llm,
    memory=CustomMemory(),
    compressor=CustomCompressor(),
    max_context_tokens=8000  # 触发压缩
)
```

**内置选项：**
- `InMemoryMemory` - 默认（仅会话）
- `SimpleCompressor` - 基本 token 减少
- 见 `loom/builtin/memory/` 和 `loom/builtin/compression/`

## 常见任务的关键文件

| 任务 | 文件 |
|------|------|
| 添加新 LLM 提供商 | `loom/llm/factory.py`, `loom/builtin/llms/` |
| 修改递归逻辑 | `loom/core/recursion_control.py` |
| 添加事件类型 | `loom/core/events.py`（AgentEventType 枚举）|
| 工具执行流水线 | `loom/core/tool_pipeline.py`（5 阶段）|
| 上下文压缩 | `loom/core/context_assembly.py` |
| RAG 集成 | `loom/core/context_retriever.py` |
| MCP 集成 | `loom/mcp/registry.py`, `loom/mcp/tool_adapter.py` |
| 自定义任务处理器 | `loom/core/agent_executor.py`（TaskHandler 基类）|
| 向量存储 | `loom/builtin/retriever/`（FAISS、ChromaDB）|
| 领域适配器 | `loom/retrieval/`（SimpleDomainAdapter）|

## 调试与可观测性

### 事件监控用于调试

**模式：** 收集事件以进行事后分析。

```python
from loom.core.events import AgentEventType

# 收集所有事件
events = []
async for event in agent.stream(prompt):
    events.append(event)

# 分析执行
iterations = [e for e in events if e.type == AgentEventType.ITERATION_START]
tool_calls = [e for e in events if e.type == AgentEventType.TOOL_RESULT]
errors = [e for e in events if e.type == AgentEventType.ERROR]
terminations = [e for e in events if e.type == AgentEventType.RECURSION_TERMINATED]

print(f"迭代次数: {len(iterations)}")
print(f"工具调用: {len(tool_calls)}")
print(f"错误: {len(errors)}")
if terminations:
    print(f"终止原因: {terminations[0].metadata['reason']}")
```

### 生产环境监控回调

**模式：** 实现 `BaseCallback` 以进行指标/日志记录。

```python
from loom.callbacks.base import BaseCallback

class ProductionCallback(BaseCallback):
    async def on_agent_start(self, **kwargs):
        # 记录到 APM/指标系统
        pass
    
    async def on_tool_start(self, tool_name: str, **kwargs):
        # 跟踪工具使用
        pass
    
    async def on_error(self, error: Exception, **kwargs):
        # 失败时告警
        pass

agent = loom.agent(
    provider="openai",
    model="gpt-4",
    callbacks=[ProductionCallback(), MetricsCollector()]
)
```

**内置回调：**
- `MetricsCollector` - 使用统计（`agent.get_metrics()`）
- 更多见 `loom/callbacks/`

### 常见调试场景

| 问题 | 诊断 | 解决方案 |
|-------|-----------|----------|
| Agent 陷入循环 | 检查 `RECURSION_TERMINATED` 事件 | 调整 `RecursionMonitor` 阈值 |
| 工具不执行 | 检查 `TOOL_ERROR` 事件 | 验证工具 schema 和权限 |
| 上下文溢出 | 检查事件中的 token 计数 | 启用压缩或减少 `max_context_tokens` |
| RAG 不工作 | 检查阶段 1 事件 | 验证 `context_retriever` 注入模式 |
| 执行缓慢 | 跟踪事件时间戳 | 识别瓶颈（LLM、工具、检索）|

## 项目特定术语

- **tt（Tail-Tail）**：核心递归执行方法（不是"tool-tool"）
- **TurnState**：每个递归级别的不可变状态快照
- **AgentEvent**：可观测的执行事件（不是日志）
- **RecursionMonitor**：循环检测子系统（Phase 2 功能）
- **ToolExecutionPipeline**：5 阶段工具处理（发现 → 验证 → 授权 → 执行 → 格式化）
- **ContextRetriever**：RAG 文档注入点（阶段 1）

## 参考示例

- `examples/agent_events_demo.py` - 事件处理模式
- `examples/recursion_control_demo.py` - 循环检测场景
- `examples/message_passing_demo.py` - 上下文管理
- `examples/streaming_example.py` - 流式 API 使用
- `examples/retrieval/complete_example.py` - RAG 集成

## 版本特定说明（v0.0.5）

- **Claude Sonnet 4** 现为 Anthropic 默认（`claude-sonnet-4-20250514`）
- 递归控制开销：< 1ms/迭代
- 上下文管理开销：< 5ms（无压缩）
- 测试套件：40/40 通过
- 破坏性变更：无（100% 向后兼容 v0.0.3）

---

**快速参考链接：**
- 架构：`docs/LOOM_FRAMEWORK_GUIDE.md`
- 用户指南：`docs/USAGE_GUIDE_V0_0_5.md`
- API 参考：`docs/user/api-reference.md`
- 快速开始：`docs/QUICKSTART.md`
