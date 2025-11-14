# Quickstart: ROI 计算工具

本示例展示如何以 Loom 工具方式调用 ROI 计算。

## 单笔计算（product 模式，结构化输出）

```python
from loom import tool

# 假设已注册 ROI 工具为 roi()
result = await roi().run({
    "cost": 100,
    "revenue": 130,
    # mode 缺省，默认 product
})
# 期望: ratio=0.30, percentage="30.00%", explanation 包含“投资回报率 30.00%”
```

## 批量计算（demo 模式，文本输出）

```python
payload = {
    "items": [
        {"cost": 100, "revenue": 130},
        {"cost": 200, "revenue": 180},
        {"cost": 50,  "revenue": 75}
    ],
    "mode": "demo",
    "output": "text"
}
res = await roi_batch().run(payload)
# 返回每条文本说明与汇总统计（仅基于有效项）
```

备注：
- `product` 模式：严格校验、结构化输出、隐藏调试字段。
- `demo/r&d` 模式：默认文本输出，允许更丰富说明（但数值与四舍五入一致）。
