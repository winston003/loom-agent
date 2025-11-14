# Research: ROI 计算工具

Date: 2025-11-14  
Branch: 001-roi-tool

## Decisions

- Decision: 使用 Decimal 进行数学计算与四舍五入
  - Rationale: 避免二进制浮点误差；确保与规范精度一致（比率 4 位、百分比 2 位）
  - Alternatives: float（舍弃，易出现 0.1 精度问题）；Fraction（不便于百分比输出）

- Decision: 工具以 Loom `@tool()` 形式实现
  - Rationale: 与框架一致，可自动生成 schema、支持并发安全
  - Alternatives: 独立 CLI 或 REST（超出当前范围，增加部署复杂度）

- Decision: 模式参数 `mode` = `product` | `demo` | `r&d`
  - Rationale: 满足商业/产品默认稳定性与可观测性，同时兼容演示/研发解释性输出
  - Alternatives: 单一模式（降低适配性）

- Decision: 批量聚合统计仅基于有效条目
  - Rationale: 避免错误项污染聚合指标；错误单独返回清单
  - Alternatives: 将错误计为 0 或剔除并沉默（弱错误可见性）

## Open Questions (resolved)

- 输出格式默认：`product` → 结构化；`demo/r&d` → 文本（保持数值一致）
- 非法模式处理：回退 product 并返回非致命警告（不影响数值结果）
- 规模目标：1k 条/批次（可在未来提高）

## References

- Loom 工具化模式与 Pydantic v2 schema 推导
- Python `decimal` 文档（量化与舍入）
