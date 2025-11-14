# Implementation Plan: ROI 计算工具

**Branch**: `001-roi-tool` | **Date**: 2025-11-14 | **Spec**: `specs/001-roi-tool/spec.md`
**Input**: 来自 `specs/001-roi-tool/spec.md` 的功能规范与澄清（商业/产品为默认模式，可切换演示/研发）。

## Summary

实现一个可作为 Loom 工具使用的 ROI 计算能力：
- 单笔与批量计算，返回比率与百分比，并提供可读说明。
- 模式参数 `mode`：`product`（默认，结构化输出、严格校验）、`demo`/`r&d`（文本输出、解释更充分）。
- 数值准确性：使用 Decimal 进行计算与四舍五入（比率 4 位、百分比 2 位）。

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: Loom Agent Framework（内置 `@tool` 装饰器，Pydantic v2 schema 推导）  
**Storage**: N/A（纯计算，无持久化）  
**Testing**: pytest（已有测试基架），新增单元与契约测试  
**Target Platform**: Python 包内工具（可被 Agent 调用）  
**Project Type**: Python 库/Agent 工具  
**Performance Goals**: 单笔计算 p95 ≤ 100ms；批量 1,000 条 ≤ 1s  
**Constraints**: Decimal 计算与固定四舍五入规则；严格输入校验在 `product` 模式下启用  
**Scale/Scope**: 典型批量 ≤ 1k 条；无外部依赖；跨语言/货币假设不涉及

## Constitution Check

Gates（来自《宪章》）：
- 框架优先：通过 `@tool()` 注册、以组合替代配置 → 满足
- 渐进复杂：从单笔→批量→模式切换，按优先级递进 → 满足
- 可观测：工具执行事件将通过 Loom 事件流可见 → 满足
- 可靠性优先：使用 Decimal 避免浮点误差；严格校验 → 满足
- 组件可替换：工具为独立模块，可独立测试与替换 → 满足

结论：通过，无需豁免。

## Project Structure

### Documentation (this feature)

```text
specs/001-roi-tool/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    └── roi-tool.schema.json
```

### Source Code (repository root)

```text
loom/
└── builtin/
    └── tools/
        └── roi.py          # 新增：ROI 工具实现（@tool）

tests/
└── unit/
    └── tools/
        └── test_roi.py     # 新增：单元与边界测试
```

**Structure Decision**: 在现有 Python 库内新增一个工具模块与对应测试，避免引入新服务或进程，符合“框架优先/渐进复杂”。

## Complexity Tracking

无违例项。
