# Data Model: ROI 计算工具

Date: 2025-11-14

## Entities

- ROIInput
  - cost: decimal(>0, required)
  - revenue: decimal(required)
  - mode: enum {product|demo|r&d} (optional, default=product; aliases: dev→r&d)
  - output: enum {structured|text} (optional; 若未指定，则随模式默认)
  - precision_ratio: int (optional; default=4)
  - precision_percentage: int (optional; default=2)

- ROIResult
  - ratio: decimal (四舍五入至 precision_ratio)
  - percentage: string (四舍五入至 precision_percentage，含 %)
  - explanation: string
  - warnings?: string[] (非致命提示，如非法模式回退)

- BatchROIInput
  - items: ROIInput[] (至少 1 条；items 中可省略 mode/output 继承顶层)

- BatchROIResult
  - results: (ROIResult | ROIError)[]
  - summary:
    - avg_ratio: decimal
    - min_ratio: decimal
    - max_ratio: decimal
  - errors_count: int

- ROIError
  - index: int
  - message: string
  - field?: string

## Rules

- cost 必须 > 0；否则为致命输入错误。
- 百分比 = ratio * 100，并按 precision_percentage 四舍五入，格式化末尾添加 "%"。
- 聚合统计仅基于成功项计算；若无成功项，summary 可为空或字段为 null。
- mode 未知值 → 回退到 product 并在 warnings 中提示。
